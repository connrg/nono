use crate::cli::OpenUrlHelperArgs;
use nono::supervisor::types::{SupervisorMessage, SupervisorResponse};
use nono::supervisor::{SupervisorSocket, UrlOpenRequest};
use nono::{NonoError, Result};
use std::path::Path;

/// Internal helper invoked via BROWSER env var (Linux) or PATH shim (macOS).
///
/// Reads the supervisor socket path from `NONO_SUPERVISOR_PATH`, connects to
/// the supervisor's named socket, sends an `OpenUrl` IPC message, waits for
/// the response, and exits with the appropriate exit code.
pub(crate) fn run_open_url_helper(args: OpenUrlHelperArgs) -> Result<()> {
    let socket_path = std::env::var("NONO_SUPERVISOR_PATH").map_err(|_| {
        NonoError::SandboxInit(
            "NONO_SUPERVISOR_PATH not set. open-url-helper must be invoked inside a nono sandbox."
                .to_string(),
        )
    })?;

    let mut socket = SupervisorSocket::connect(Path::new(&socket_path))?;
    socket.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;

    let request = UrlOpenRequest {
        request_id: format!("url-{}", std::process::id()),
        url: args.url.clone(),
        child_pid: std::process::id(),
        session_id: String::new(),
    };

    socket.send_message(&SupervisorMessage::OpenUrl(request))?;

    let response = socket.recv_response()?;
    match response {
        SupervisorResponse::UrlOpened { success: true, .. } => Ok(()),
        SupervisorResponse::UrlOpened {
            success: false,
            error,
            ..
        } => {
            let msg = error.unwrap_or_else(|| "Unknown error".to_string());
            Err(NonoError::SandboxInit(format!(
                "Supervisor denied URL open: {msg}"
            )))
        }
        other => Err(NonoError::SandboxInit(format!(
            "Unexpected supervisor response: {other:?}"
        ))),
    }
}
