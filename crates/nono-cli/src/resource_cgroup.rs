//! Linux cgroup v2 resource enforcement (issue #1102).
//!
//! # What this does, in plain terms
//!
//! When a sandboxed run is given a `--memory` limit, this
//! module puts the program in a kernel-enforced "box" so it cannot use more than
//! that. If it tries, the Linux kernel kills it instantly — and *only* it — so a
//! runaway agent cannot drag down the rest of the machine. Think of a room with
//! a strict capacity sign: step over the line and the door slams, on that room
//! alone.
//!
//! The mechanism is **cgroup v2** ("control groups"), the same kernel feature
//! containers use. A cgroup is just a directory under `/sys/fs/cgroup`: you set
//! limits by writing numbers into files inside it (its "knobs"), and you put a
//! process in it by writing the process id into its `cgroup.procs` file. We
//! create one such directory per run (a "leaf"), set the knobs, the child moves
//! *itself* in, and we delete the directory when the run ends.
//!
//! # Who runs this
//!
//! The unsandboxed *supervisor* (nono's parent process) builds the box and arms
//! it before forking. The sandboxed *child* then puts itself in the box — see
//! the race note below for why the child, not the parent, does the attach.
//!
//! # Containing the whole process tree (the race, and why self-attach)
//!
//! A cgroup caps *every* process inside it together, and a forked child inherits
//! its parent's cgroup automatically — so once a process is in the leaf, its
//! whole subtree is capped and dies atomically. The only hard part is getting
//! the first process *into* the leaf without leaving a gap.
//!
//! If the **parent** moved the child in *after* `fork()`, there would be a brief
//! window in which the child is already running but not yet boxed; a child that
//! forks its own children inside that window could slip a few of them out into
//! the parent's (unconstrained) cgroup. To close that window, the **child
//! attaches itself**: the parent opens the leaf's `cgroup.procs` write fd before
//! forking (so the child inherits it), and the child writes its own pid through
//! that fd before it does anything else — before it can fork or exec. It is
//! therefore in the box by construction, with no escape window, regardless of
//! timing (see [`child_self_attach`]).
//!
//! # Fail-closed (AGENTS.md "Fail Secure")
//!
//! If the box cannot be built, armed, or entered, the run is refused rather than
//! allowed to proceed unprotected:
//! - Creating the box and setting its knobs happen *before* the child is forked,
//!   so any failure is a [`NonoError`] returned while nothing is running yet.
//! - If the child cannot self-attach after fork, it kills itself
//!   (`_exit(126)`) before applying the sandbox or exec'ing — it never runs the
//!   thing the limit was meant to constrain.
//!
//! # The knobs we set
//!
//! These three knobs match the manual cgroup v2 experiment that validated the
//! design:
//! - `memory.max`         — the hard memory ceiling; cross it and the kernel OOM-kills.
//! - `memory.swap.max=0`  — forbid spilling to swap, which would let it dodge the ceiling.
//! - `memory.oom.group=1` — on OOM, kill the *whole* box at once, not one random member.
//!
//! We deliberately do **not** set `memory.high` (a softer "ease off" threshold):
//! with swap forbidden, a runaway allocator has nothing to reclaim, so it would
//! stall for many seconds before finally being killed — defeating the point of a
//! fast, clean kill. (Revisit as an opt-in only if real near-limit workloads
//! show it helps.)
//!
//! Choosing *which* backend to enforce with (the WSL2 / non-systemd probe and
//! the `auto`/`cgroup`/`portable` resolution) is a later step; this targets the
//! common case — a normal desktop/server login, i.e. a systemd `Delegate=yes`
//! user session.

use nix::libc;
use nono::{NonoError, ResourceLimits, Result};
use std::fs::{self, File, OpenOptions};
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};

/// One sandboxed run's resource "box": a cgroup v2 directory we create, arm with
/// limits, the child moves into, and we delete when the run ends.
///
/// Created pre-fork by [`CgroupLeaf::create`]; the child attaches itself through
/// the inherited `cgroup.procs` fd ([`CgroupLeaf::procs_raw_fd`] +
/// [`child_self_attach`]). On drop the box is emptied (any survivors killed) and
/// the directory removed, so an early return or panic cannot leak a cgroup.
pub struct CgroupLeaf {
    /// Absolute path of the leaf directory, e.g.
    /// `/sys/fs/cgroup/.../user@1000.service/nono.<pid>`.
    path: PathBuf,
    /// Write handle to the leaf's `cgroup.procs`, opened pre-fork so the child
    /// inherits it across `fork` and can self-attach before it execs. The
    /// parent never writes through it; it exists only to be inherited. Closed
    /// when the leaf is dropped (and in the child at `execve`, via O_CLOEXEC).
    procs: File,
}

impl CgroupLeaf {
    /// Create the leaf, write the requested knobs, and open its `cgroup.procs`
    /// for the child to self-attach through. Pre-fork; fail-closed.
    ///
    /// # Errors
    /// Returns [`NonoError::SandboxInit`] if the session has no delegated
    /// cgroup v2 subtree, the required controllers are not delegated, or any
    /// knob write / `cgroup.procs` open fails. On any failure no partial leaf is
    /// left behind.
    pub fn create(limits: &ResourceLimits) -> Result<Self> {
        let base = delegated_base()?;
        ensure_controllers_delegated(&base, limits)?;

        let path = base.join(format!("nono.{}", std::process::id()));
        // A leftover leaf with our exact pid is unexpected (pid reuse after a
        // crash). Reusing it could inherit stale members, so treat an existing
        // directory as a hard error rather than silently adopting it.
        fs::create_dir(&path).map_err(|e| {
            NonoError::SandboxInit(format!(
                "resource: failed to create cgroup leaf {}: {e}",
                path.display()
            ))
        })?;

        // The leaf directory now exists; from here any failure must remove it.
        // `arm` does the remaining fallible work; on error we tear the directory
        // down before returning (no `Self` is constructed yet, so there is no
        // double teardown).
        match Self::arm(path.clone(), limits) {
            Ok(leaf) => Ok(leaf),
            Err(e) => {
                teardown(&path);
                Err(e)
            }
        }
    }

    /// Write the knobs and open `cgroup.procs`, building the owned [`CgroupLeaf`].
    /// Separated from [`create`](Self::create) so a failure here is cleaned up by
    /// the caller's `teardown` of the already-created directory.
    fn arm(path: PathBuf, limits: &ResourceLimits) -> Result<Self> {
        write_knobs(&path, limits)?;
        let procs_path = path.join("cgroup.procs");
        // Close-on-exec (`O_CLOEXEC`, Rust's default) is intentional: the fd is
        // inherited across `fork` so the child can self-attach, then the kernel
        // closes it automatically at the child's `execve` — so it never leaks
        // into the sandboxed program.
        let procs = OpenOptions::new()
            .write(true)
            .open(&procs_path)
            .map_err(|e| {
                NonoError::SandboxInit(format!(
                    "resource: failed to open {} for self-attach ({e})",
                    procs_path.display()
                ))
            })?;
        Ok(Self { path, procs })
    }

    /// Raw fd of the leaf's `cgroup.procs`, to be inherited by the forked child
    /// and written through by [`child_self_attach`]. Valid for the lifetime of
    /// this `CgroupLeaf`.
    #[must_use]
    pub fn procs_raw_fd(&self) -> RawFd {
        self.procs.as_raw_fd()
    }
}

impl Drop for CgroupLeaf {
    fn drop(&mut self) {
        teardown(&self.path);
    }
}

/// The child puts **itself** into the resource cgroup by writing its own pid to
/// the inherited `cgroup.procs` write fd, in the post-fork window *before* it
/// applies the sandbox or execs. Because it runs before the child can fork or
/// exec, every descendant it later spawns is inside the cgroup by construction —
/// there is no window for a process to escape into the parent's cgroup.
///
/// Async-signal-safe: it only calls `getpid`, formats the pid into a stack
/// buffer, and `write`s — no allocation and no locks — so it is safe to call in
/// the post-fork child path. Returns `true` on a complete write; on `false` the
/// caller must `_exit` the child (fail-closed: never run unconfined).
///
/// # Safety
/// `procs_fd` must be a valid, writable fd for the leaf's `cgroup.procs`,
/// inherited from the parent across `fork`.
#[must_use = "on `false` the child is unconfined and the caller MUST _exit it (fail-closed)"]
pub fn child_self_attach(procs_fd: RawFd) -> bool {
    // SAFETY: `getpid` is async-signal-safe and always succeeds; in the child it
    // returns the child's own pid.
    let pid = unsafe { libc::getpid() };
    let mut buf = [0u8; MAX_PID_DIGITS];
    let encoded = format_pid_decimal(pid, &mut buf);
    // SAFETY: writing `encoded.len()` bytes from a stack buffer to a raw fd is
    // async-signal-safe. A short or failed write is treated as failure so the
    // caller fails closed.
    let written = unsafe {
        libc::write(
            procs_fd,
            encoded.as_ptr().cast::<libc::c_void>(),
            encoded.len(),
        )
    };
    written == encoded.len() as isize
}

/// `i32::MAX` is `2147483647` (10 digits); pids are positive, so 10 digits is the
/// widest decimal a pid can require.
const MAX_PID_DIGITS: usize = 10;

/// Format `pid` as decimal ASCII into `buf` without allocating (async-signal-safe),
/// returning the populated trailing slice. A non-positive `pid` is defensively
/// rendered as `"0"` (which a real pid never is).
fn format_pid_decimal(pid: i32, buf: &mut [u8; MAX_PID_DIGITS]) -> &[u8] {
    let mut n = u32::try_from(pid).unwrap_or(0);
    // Write digits least-significant-first into the tail of buf, so we never
    // need a second pass to reverse them (keeps this branch- and alloc-light).
    let mut i = buf.len();
    if n == 0 {
        i -= 1;
        buf[i] = b'0';
    }
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    &buf[i..]
}

/// Write the requested limit knobs into the leaf at `path`.
fn write_knobs(path: &Path, limits: &ResourceLimits) -> Result<()> {
    if let Some(max) = limits.memory_bytes {
        // Forbid the swap escape hatch first, then set the ceiling itself.
        write_knob(path, "memory.swap.max", "0")?;
        write_knob(path, "memory.max", &max.to_string())?;
        // On OOM, take down the whole box together rather than letting the
        // kernel pick one member to sacrifice.
        write_knob(path, "memory.oom.group", "1")?;
        // No memory.high on purpose — see the module docs: with swap forbidden
        // it stalls a runaway allocator instead of killing it.
    }
    Ok(())
}

fn write_knob(path: &Path, knob: &str, value: &str) -> Result<()> {
    let file = path.join(knob);
    fs::write(&file, value).map_err(|e| {
        NonoError::SandboxInit(format!(
            "resource: failed to write '{value}' to {} ({e}); \
             is the cgroup v2 controller delegated?",
            file.display()
        ))
    })
}

/// How long to wait for the kernel to reap killed members before `rmdir`:
/// [`REAP_POLL_ATTEMPTS`] checks, [`REAP_POLL_INTERVAL`] apart — a ~500ms ceiling.
/// In the common case (the child has already exited) the first check passes and
/// we never sleep.
const REAP_POLL_ATTEMPTS: u32 = 50;
const REAP_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);

/// Kill anything still in the leaf, then remove it. Best-effort: leaking an
/// empty cgroup directory is not worth failing a completed run over.
fn teardown(path: &Path) {
    // cgroup.kill (Linux 5.14+) kills the whole subtree atomically. Ignored if
    // unavailable; for a reaped single process the leaf is already empty.
    let _ = fs::write(path.join("cgroup.kill"), "1");
    // A cgroup with live members cannot be removed; wait briefly for the kernel
    // to reap before rmdir (see the poll-budget consts above).
    let procs = path.join("cgroup.procs");
    for _ in 0..REAP_POLL_ATTEMPTS {
        match fs::read_to_string(&procs) {
            Ok(contents) if contents.trim().is_empty() => break,
            Ok(_) => std::thread::sleep(REAP_POLL_INTERVAL),
            Err(_) => break,
        }
    }
    let _ = fs::remove_dir(path);
}

/// Find the cgroup directory we're allowed to create our box inside.
///
/// On a normal login, systemd hands each user's session a private cgroup subtree
/// it may manage without root — rooted at `.../user@<uid>.service`. That handover
/// is "delegation", and that directory is the only place we can reliably create
/// child cgroups. We locate it by reading our own cgroup path from
/// `/proc/self/cgroup` and walking up to the `user@<uid>.service` ancestor.
fn delegated_base() -> Result<PathBuf> {
    let raw = fs::read_to_string("/proc/self/cgroup").map_err(|e| {
        NonoError::SandboxInit(format!("resource: cannot read /proc/self/cgroup: {e}"))
    })?;
    let uid = nix::unistd::Uid::current().as_raw();
    let base = parse_delegated_base(&raw, uid)?;
    if !base.is_dir() {
        return Err(NonoError::SandboxInit(format!(
            "resource: delegated cgroup path {} does not exist",
            base.display()
        )));
    }
    Ok(base)
}

/// Pure parser for [`delegated_base`]: take the `/proc/self/cgroup` contents and
/// the uid, return the absolute path of the `user@<uid>.service` ancestor.
fn parse_delegated_base(proc_self_cgroup: &str, uid: u32) -> Result<PathBuf> {
    // Unified cgroup v2 exposes a single `0::<path>` line.
    let rel = proc_self_cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| {
            NonoError::SandboxInit(
                "resource: not a unified cgroup v2 hierarchy (no '0::' line in \
                 /proc/self/cgroup); cgroup resource limits are unavailable"
                    .to_string(),
            )
        })?
        .trim();

    let marker = format!("user@{uid}.service");
    // Keep the path up to and including the user@<uid>.service segment — the
    // systemd delegation boundary we can create children under.
    let mut acc = PathBuf::from("/sys/fs/cgroup");
    let mut found = false;
    for segment in rel.split('/').filter(|s| !s.is_empty()) {
        acc.push(segment);
        if segment == marker {
            found = true;
            break;
        }
    }
    if !found {
        return Err(NonoError::SandboxInit(format!(
            "resource: no delegated cgroup v2 subtree for this session \
             (expected a '{marker}' ancestor in '{rel}'); resource limits \
             require a delegated user session (systemd Delegate=yes)"
        )));
    }
    Ok(acc)
}

/// Verify the controllers we need are enabled for child cgroups. A cgroup only
/// exposes a controller's knobs (here `memory.max`) if that controller is listed
/// in the parent's `cgroup.subtree_control` file; without that the leaf's
/// `memory.max` would not exist and a cap would silently fail to apply.
fn ensure_controllers_delegated(base: &Path, limits: &ResourceLimits) -> Result<()> {
    let subtree_path = base.join("cgroup.subtree_control");
    let subtree = fs::read_to_string(&subtree_path).map_err(|e| {
        NonoError::SandboxInit(format!(
            "resource: cannot read {} ({e})",
            subtree_path.display()
        ))
    })?;
    let has = |controller: &str| subtree.split_whitespace().any(|w| w == controller);

    if limits.memory_bytes.is_some() && !has("memory") {
        return Err(NonoError::SandboxInit(format!(
            "resource: the 'memory' controller is not delegated to {} \
             (cgroup.subtree_control = '{}'); cannot enforce --memory",
            base.display(),
            subtree.trim()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_PID_DIGITS, format_pid_decimal, parse_delegated_base};
    use std::path::PathBuf;

    #[test]
    fn parses_user_service_ancestor_from_deep_path() {
        let raw = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/\
                   app-org.gnome.Terminal.slice/vte-spawn-abc.scope\n";
        let base = parse_delegated_base(raw, 1000).expect("should parse");
        assert_eq!(
            base,
            PathBuf::from("/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service")
        );
    }

    #[test]
    fn stops_at_user_service_even_when_it_is_the_leaf() {
        let raw = "0::/user.slice/user-501.slice/user@501.service\n";
        let base = parse_delegated_base(raw, 501).expect("should parse");
        assert_eq!(
            base,
            PathBuf::from("/sys/fs/cgroup/user.slice/user-501.slice/user@501.service")
        );
    }

    #[test]
    fn rejects_non_unified_hierarchy() {
        // A v1/hybrid line has a non-zero hierarchy id and named controllers.
        let raw = "1:name=systemd:/user.slice/session-2.scope\n";
        assert!(parse_delegated_base(raw, 1000).is_err());
    }

    #[test]
    fn rejects_session_without_user_service_delegation() {
        // e.g. launched from a system service: no user@<uid>.service ancestor.
        let raw = "0::/system.slice/cron.service\n";
        assert!(parse_delegated_base(raw, 1000).is_err());
    }

    #[test]
    fn rejects_mismatched_uid() {
        // The path is for uid 1000 but we are uid 1001 — not our delegation.
        let raw = "0::/user.slice/user-1000.slice/user@1000.service/app.slice\n";
        assert!(parse_delegated_base(raw, 1001).is_err());
    }

    /// The self-attach write depends on a correct, allocation-free decimal
    /// rendering of the child's pid; spot-check the encoder across widths.
    #[test]
    fn formats_pid_as_decimal() {
        let mut buf = [0u8; MAX_PID_DIGITS];
        assert_eq!(format_pid_decimal(1, &mut buf), b"1");
        assert_eq!(format_pid_decimal(7, &mut buf), b"7");
        assert_eq!(format_pid_decimal(12345, &mut buf), b"12345");
        assert_eq!(format_pid_decimal(i32::MAX, &mut buf), b"2147483647");
    }

    #[test]
    fn formats_nonpositive_pid_defensively_as_zero() {
        let mut buf = [0u8; MAX_PID_DIGITS];
        assert_eq!(format_pid_decimal(0, &mut buf), b"0");
        assert_eq!(format_pid_decimal(-1, &mut buf), b"0");
    }

    // ---- #1102 additions: adversarial component-wise matching & pid property ----

    /// SECURITY: the delegation boundary is matched by whole path SEGMENT
    /// (`segment == marker`), never by substring. A `starts_with`/`contains`
    /// implementation would false-match these crafted near-miss segments and
    /// hand an attacker a base outside the real `user@<uid>.service` delegation
    /// (AGENTS.md: string `starts_with` on paths is a vulnerability). Every one
    /// of these must be REJECTED for uid 1000 (no exact-segment match).
    #[test]
    fn rejects_adversarial_segment_lookalikes_componentwise() {
        let poisoned = [
            // marker with a trailing suffix on the same segment (prefix match)
            "0::/user.slice/user-1000.slice/user@1000.service.evil/app.slice\n",
            "0::/user.slice/user-1000.slice/user@1000.serviceX\n",
            // marker as a strict SUFFIX of a single segment
            "0::/user.slice/user-1000.slice/xuser@1000.service/app.slice\n",
            "0::/user.slice/user-1000.slice/evil-user@1000.service\n",
            // marker embedded mid-segment (would be caught by `contains`)
            "0::/user.slice/prefixuser@1000.servicesuffix/app.slice\n",
            // dot/underscore confusable that is NOT a path separator
            "0::/user.slice/user@1000_service\n",
            // a stray space welds trailing junk onto the marker segment; split('/')
            // does not break on spaces, so the segment is "user@1000.service junk".
            "0::/user.slice/user@1000.service junk/app.scope\n",
        ];
        for raw in poisoned {
            assert!(
                parse_delegated_base(raw, 1000).is_err(),
                "adversarial lookalike segment must NOT match the delegation boundary: {raw:?}"
            );
        }
    }

    /// SECURITY: the uid is baked into the marker (`user@<uid>.service`) and must
    /// match a whole segment. uid 100's marker `user@100.service` must NOT match
    /// a `user@1000.service` segment (100 is a substring of 1000), and vice versa.
    #[test]
    fn rejects_uid_that_is_substring_of_another_uid() {
        // Path delegates uid 1000; asking as uid 100 (a substring) must fail.
        let raw_1000 = "0::/user.slice/user-1000.slice/user@1000.service/app.slice\n";
        assert!(
            parse_delegated_base(raw_1000, 100).is_err(),
            "uid 100 must not borrow uid 1000's delegation (substring match)"
        );
        // Symmetric: path delegates uid 100; asking as uid 1000 must fail.
        let raw_100 = "0::/user.slice/user-100.slice/user@100.service/app.slice\n";
        assert!(
            parse_delegated_base(raw_100, 1000).is_err(),
            "uid 1000 must not match uid 100's delegation"
        );
        // Exact uid still works (guards against the test over-rejecting).
        let base = parse_delegated_base(raw_1000, 1000).expect("exact uid must match");
        assert_eq!(
            base,
            PathBuf::from("/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service")
        );
    }

    /// The marker can sit at an interior position with descendants below it; the
    /// returned base must be truncated to exactly the marker (the delegation
    /// boundary), discarding everything below — even with a trailing slash and a
    /// non-first `0::` line.
    #[test]
    fn parses_marker_as_nonfinal_segment_and_truncates_exactly() {
        let raw = "1:name=systemd:/legacy/ignored\n\
                   0::/user.slice/user-1000.slice/user@1000.service/app.slice/svc.scope\n";
        let base = parse_delegated_base(raw, 1000).expect("should parse");
        assert_eq!(
            base,
            PathBuf::from("/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service"),
            "base must stop at the marker even with descendants below it"
        );

        // Trailing slash after the marker: split('/').filter(non-empty) drops the
        // empty tail, so the result is identical.
        let raw_slash = "0::/user.slice/user@1000.service/\n";
        let base_slash = parse_delegated_base(raw_slash, 1000).expect("should parse");
        assert_eq!(
            base_slash,
            PathBuf::from("/sys/fs/cgroup/user.slice/user@1000.service")
        );
    }

    /// `format_pid_decimal` is async-signal-safe and load-bearing (the post-fork
    /// child writes its pid through it to self-attach). It must equal the
    /// allocating `i32::to_string()` for every positive pid, including the
    /// boundaries that exercise digit-count transitions and the EXACT buffer fill
    /// at `i32::MAX` (10 digits == MAX_PID_DIGITS). Asserting against `to_string`
    /// uses std as an independent oracle rather than hand-computed literals.
    #[test]
    fn format_pid_decimal_matches_to_string_over_wide_range_and_boundaries() {
        let mut buf = [0u8; MAX_PID_DIGITS];

        // Explicit boundaries: digit-count transitions + exact buffer fill.
        for &pid in &[
            1i32,
            9,
            10,
            99,
            100,
            999,
            1000,
            9999,
            10000,
            1_000_000,
            i32::MAX,
        ] {
            assert_eq!(
                format_pid_decimal(pid, &mut buf),
                pid.to_string().as_bytes(),
                "pid {pid} must encode identically to to_string()"
            );
        }
        // i32::MAX fills the buffer exactly: 10 digits, no leading slack.
        assert_eq!(format_pid_decimal(i32::MAX, &mut buf).len(), MAX_PID_DIGITS);

        // Dense sweep over a contiguous low range (covers the 9->10, 99->100,
        // 999->1000 width transitions).
        for pid in 1..=10_000_i32 {
            assert_eq!(
                format_pid_decimal(pid, &mut buf),
                pid.to_string().as_bytes(),
                "mismatch at pid {pid}"
            );
        }
        // Wide property sweep across the positive i32 range (a prime stride avoids
        // aliasing to round numbers while staying fast).
        let mut pid: i64 = 1;
        while pid <= i32::MAX as i64 {
            let p = pid as i32;
            assert_eq!(format_pid_decimal(p, &mut buf), p.to_string().as_bytes());
            pid += 7919;
        }

        // Non-positive defensively renders as "0" (a real pid never is); i32::MIN
        // is included because a naive `pid.abs()` would overflow/panic.
        for pid in [0_i32, -1, -42, -2_147_483_647, i32::MIN] {
            assert_eq!(
                format_pid_decimal(pid, &mut buf),
                b"0",
                "non-positive pid {pid} must render as 0"
            );
        }
    }

    // ---- #1102 LIVE cgroup v2 enforcement tests ----
    //
    // These are #[ignore]-gated: they create real cgroup leaves under
    // /sys/fs/cgroup, run a bounded memory bomb, and read kernel knobs. They are
    // NOT run by default CI. The host they target must be a systemd
    // `Delegate=yes` user session with the `memory` controller delegated to
    // `user@<uid>.service` (cgroup v2 unified hierarchy). Run all of them with:
    //
    //   cargo test -p nono-cli --bins -- --ignored
    //
    // or one at a time, e.g.:
    //
    //   cargo test -p nono-cli --bins -- --ignored live_child_over_memory_cap

    /// LIVE: end-to-end enforcement. A forked child that self-attaches and then
    /// allocates past the cap is OOM-killed (SIGKILL); the leaf records
    /// oom_kill>=1; swap stayed at 0 (no escape); the parent survives; and
    /// drop/teardown removes the leaf. This is the actual security property of
    /// #1102, proven against a real cgroupfs.
    #[test]
    #[ignore = "requires live cgroup v2 delegation (memory controller); run with --ignored"]
    fn live_child_over_memory_cap_is_oom_killed_and_only_it() {
        use super::CgroupLeaf;
        use super::child_self_attach;
        use nix::libc;
        use nix::sys::signal::Signal;
        use nix::sys::wait::{WaitStatus, waitpid};
        use nix::unistd::{ForkResult, fork};
        use nono::ResourceLimits;

        const CAP: u64 = 64 * 1024 * 1024; // 64 MiB hard ceiling
        const TOUCH: usize = 128 * 1024 * 1024; // mmap + fault in 128 MiB (2x cap)
        const PAGE: usize = 4096;

        let limits = ResourceLimits {
            memory_bytes: Some(CAP),
        };
        // Real pre-fork construction: creates the leaf dir, writes
        // memory.swap.max=0 / memory.max=CAP / memory.oom.group=1, opens
        // cgroup.procs for the child to inherit.
        let leaf = CgroupLeaf::create(&limits).expect("create leaf on delegated host");
        let procs_fd = leaf.procs_raw_fd();
        let leaf_path = leaf.path.clone(); // private field — reachable in-module

        // SAFETY: single-purpose forked child; after fork it uses only
        // async-signal-safe libc calls (no Rust heap alloc, no locks) — exactly
        // the constraint child_self_attach is built for.
        match unsafe { fork() }.expect("fork") {
            ForkResult::Child => {
                // Self-attach FIRST, before allocating, mirroring the supervisor.
                if !child_self_attach(procs_fd) {
                    unsafe { libc::_exit(126) };
                }
                // Allocate anonymous memory and fault one byte per page so the
                // kernel actually charges it to memory.current.
                let addr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        TOUCH,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                if addr == libc::MAP_FAILED {
                    unsafe { libc::_exit(50) };
                }
                let base = addr.cast::<u8>();
                let mut off = 0usize;
                while off < TOUCH {
                    // Touch the page; the kernel OOM-kills us (SIGKILL) the moment
                    // resident memory crosses CAP, so this loop never completes.
                    unsafe { *base.add(off) = 0xA5 };
                    off += PAGE;
                }
                // If we somehow survive the cap, exit 0 so the parent's SIGKILL
                // assertion FAILS loudly (the cap did not enforce).
                unsafe { libc::_exit(0) };
            }
            ForkResult::Parent { child } => {
                let status = waitpid(child, None).expect("waitpid");
                // The child must be KILLED by the kernel, not exit cleanly.
                match status {
                    WaitStatus::Signaled(_, Signal::SIGKILL, _) => {}
                    other => {
                        panic!("expected child SIGKILL (OOM), got {other:?}; cap did not enforce")
                    }
                }

                // Kernel-side evidence: the leaf recorded an OOM kill.
                let events = std::fs::read_to_string(leaf_path.join("memory.events"))
                    .expect("read memory.events");
                let oom_kill = events
                    .lines()
                    .find_map(|l| l.strip_prefix("oom_kill "))
                    .and_then(|n| n.trim().parse::<u64>().ok())
                    .expect("memory.events has an oom_kill line");
                assert!(
                    oom_kill >= 1,
                    "expected oom_kill>=1 in memory.events, got {oom_kill} (full: {events:?})"
                );

                // The swap escape hatch stayed shut: nothing spilled to swap.
                let swap = std::fs::read_to_string(leaf_path.join("memory.swap.current"))
                    .expect("read memory.swap.current");
                assert_eq!(
                    swap.trim(),
                    "0",
                    "memory.swap.current must be 0 (swap.max=0 forbids the escape)"
                );
                // Reaching here at all proves the parent survived the child OOM
                // kill: oom.group scoped the kill to the leaf, not this process.
            }
        }

        // Drop runs teardown: kill survivors, rmdir the leaf.
        drop(leaf);
        // Fail-closed cleanup is observable: the leaf directory is gone.
        assert!(
            !leaf_path.exists(),
            "leaf {} must be removed after teardown",
            leaf_path.display()
        );
    }

    /// LIVE: leak-free lifecycle. create() materializes exactly one leaf directly
    /// under the delegated base with memory.max set to the requested cap, and
    /// Drop/teardown removes it.
    #[test]
    #[ignore = "requires live cgroup v2 delegation (memory controller); run with --ignored"]
    fn live_teardown_removes_leaf_and_create_leaves_no_leak() {
        use super::CgroupLeaf;
        use super::delegated_base;
        use nono::ResourceLimits;

        let limits = ResourceLimits {
            memory_bytes: Some(64 * 1024 * 1024),
        };
        let base = delegated_base().expect("delegated base on delegated host");

        let leaf = CgroupLeaf::create(&limits).expect("create leaf");
        let leaf_path = leaf.path.clone(); // private field — in-module access
        assert!(
            leaf_path.is_dir(),
            "create() must produce a real leaf dir at {}",
            leaf_path.display()
        );
        // The leaf lives directly under the delegated base.
        assert_eq!(
            leaf_path.parent(),
            Some(base.as_path()),
            "leaf must be a child of the delegated base"
        );
        // memory.max knob actually took (controller delegated, knob written). A
        // page-multiple value is echoed back verbatim by the kernel.
        let max = std::fs::read_to_string(leaf_path.join("memory.max")).expect("read memory.max");
        assert_eq!(max.trim(), (64u64 * 1024 * 1024).to_string());

        drop(leaf); // teardown: rmdir
        assert!(
            !leaf_path.exists(),
            "teardown must remove the leaf dir {}",
            leaf_path.display()
        );
    }

    /// LIVE: fail-closed against a colliding leaf. create() does fs::create_dir
    /// BEFORE arm(), and a leftover leaf with our exact pid is a hard error
    /// (no silent adoption of stale members). With the dir pre-planted, create()
    /// returns Err on EEXIST and must NOT tear down a directory it did not create.
    #[test]
    #[ignore = "requires live cgroup v2 delegation (memory controller); run with --ignored"]
    fn live_create_failure_leaves_no_partial_leaf() {
        use super::CgroupLeaf;
        use super::{delegated_base, teardown};
        use nono::ResourceLimits;

        let limits = ResourceLimits {
            memory_bytes: Some(64 * 1024 * 1024),
        };
        let base = delegated_base().expect("delegated base");
        // Plant a directory with our exact future leaf name so create()'s
        // fs::create_dir hits EEXIST and must error.
        let collide = base.join(format!("nono.{}", std::process::id()));
        std::fs::create_dir(&collide).expect("plant colliding leaf");

        let result = CgroupLeaf::create(&limits);
        assert!(
            result.is_err(),
            "create() must refuse an already-existing leaf (no silent adoption)"
        );
        // The planted directory is OURS: create() errored on EEXIST before arm(),
        // so it must not have torn down a directory it did not create.
        assert!(
            collide.is_dir(),
            "create() must not delete a pre-existing collision it did not own"
        );
        // Clean up our planted dir via the module's own teardown.
        teardown(&collide);
        assert!(!collide.exists(), "cleanup of planted leaf failed");
    }

    /// LIVE: the self-attach MECHANISM (not just pid formatting). A forked child
    /// that calls child_self_attach through the inherited cgroup.procs fd actually
    /// appears in the leaf's cgroup.procs. No bomb — the child just parks.
    #[test]
    #[ignore = "requires live cgroup v2 delegation (memory controller); run with --ignored"]
    fn live_child_self_attach_lands_pid_in_leaf_procs() {
        use super::CgroupLeaf;
        use super::child_self_attach;
        use nix::libc;
        use nix::sys::wait::waitpid;
        use nix::unistd::{ForkResult, fork};
        use nono::ResourceLimits;

        let limits = ResourceLimits {
            memory_bytes: Some(64 * 1024 * 1024),
        };
        let leaf = CgroupLeaf::create(&limits).expect("create leaf");
        let procs_fd = leaf.procs_raw_fd();
        let leaf_path = leaf.path.clone();

        // SAFETY: child uses only async-signal-safe libc calls post-fork.
        match unsafe { fork() }.expect("fork") {
            ForkResult::Child => {
                let ok = child_self_attach(procs_fd);
                if !ok {
                    unsafe { libc::_exit(126) };
                }
                // Park ~300ms so the parent can read cgroup.procs while we live.
                unsafe { libc::usleep(300_000) };
                unsafe { libc::_exit(0) };
            }
            ForkResult::Parent { child } => {
                // Give the child a beat to self-attach.
                std::thread::sleep(std::time::Duration::from_millis(50));
                let procs = std::fs::read_to_string(leaf_path.join("cgroup.procs"))
                    .expect("read leaf cgroup.procs");
                let child_pid = child.as_raw();
                let present = procs
                    .lines()
                    .filter_map(|l| l.trim().parse::<i32>().ok())
                    .any(|p| p == child_pid);
                assert!(
                    present,
                    "child pid {child_pid} must appear in leaf cgroup.procs (got {procs:?})"
                );
                let _ = waitpid(child, None);
            }
        }
        drop(leaf);
        assert!(!leaf_path.exists(), "leaf removed after teardown");
    }
}
