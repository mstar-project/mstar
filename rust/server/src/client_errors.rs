//! Client-visible text for an error raised while serving a request. Twin of
//! `mstar/utils/client_errors.py`, for the errors this frontend raises itself.
//!
//! A runtime-dir refusal and a failed upload write both name the server's
//! upload path, and the refusal names the server's uid and the directory
//! owner's. Nothing of that may go back to a client, so [`server_error`] logs
//! the whole text here and returns only `internal error (error id <id>)`: the
//! operator matches the id in the log, the client learns nothing about the
//! filesystem. The dangerous text never enters the returned string, so a call
//! site that forwards it without [`status_and_message`] answers with the wrong
//! status at worst, never with a leak.

use std::fmt::Display;

use uuid::Uuid;

/// Marks an error string that is already an error id, so the handler answers
/// 500 rather than the 400 a malformed request gets. A client-supplied NUL can
/// reach a message that reflects its input, so [`status_and_message`] also
/// requires the rest to be exactly an error id; a crafted marker then answers
/// the request's own status with the client's own text, never a server detail.
const SERVER_ERROR: &str = "\u{0}mstar-server-error\u{0}";

const ID_PREFIX: &str = "internal error (error id ";

/// 12 hex chars, as `secrets.token_hex(6)` in the Python helper.
fn error_id() -> String {
    Uuid::new_v4().simple().to_string()[..12].to_string()
}

/// Log `err` in full as `<what> [error id <id>]` and return the marked text a
/// client may see.
pub fn server_error(err: &impl Display, what: &str) -> String {
    let id = error_id();
    eprintln!("mstar-server: {what} [error id {id}]: {err}");
    format!("{SERVER_ERROR}{ID_PREFIX}{id})")
}

/// The status and the text to send for an error from [`server_error`] (500) or
/// from anything else (`default_status`, the caller's own refusal code).
pub fn status_and_message(err: &str, default_status: u16) -> (u16, &str) {
    match err.strip_prefix(SERVER_ERROR) {
        Some(text) if is_error_id(text) => (500, text),
        _ => (default_status, err),
    }
}

/// Exactly what [`server_error`] builds: `internal error (error id <12 hex>)`.
fn is_error_id(text: &str) -> bool {
    text.strip_prefix(ID_PREFIX)
        .and_then(|rest| rest.strip_suffix(')'))
        .is_some_and(|id| id.len() == 12 && id.bytes().all(|b| b.is_ascii_hexdigit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_server_error_keeps_the_detail_out_of_the_text_and_asks_for_500() {
        let refusal = "refusing /tmp/mstar_uploads_bob: it is owned by uid 1074. \
                       It must be a directory owned by this user (uid 1001)";
        let text = server_error(&refusal, "the upload dir check failed");
        let (status, message) = status_and_message(&text, 400);
        assert_eq!(status, 500);
        assert!(message.starts_with("internal error (error id "), "{message}");
        assert_eq!(message.len(), "internal error (error id )".len() + 12);
        for secret in ["/tmp", "uid", "refusing", "owned"] {
            assert!(!text.contains(secret), "{secret} in {text}");
        }
    }

    #[test]
    fn a_request_error_keeps_its_text_and_its_status() {
        let (status, message) = status_and_message("Malformed data URL: missing payload", 400);
        assert_eq!((status, message), (400, "Malformed data URL: missing payload"));
    }

    #[test]
    fn a_crafted_marker_is_not_mistaken_for_one_of_ours() {
        // a client-supplied NUL can reach a message that reflects its input
        let crafted = format!("{SERVER_ERROR}refusing /tmp/secret: it is owned by uid 0");
        let (status, message) = status_and_message(&crafted, 400);
        assert_eq!(status, 400);
        assert_eq!(message, crafted);
        let short = format!("{SERVER_ERROR}internal error (error id abc)");
        assert_eq!(status_and_message(&short, 400).0, 400);
    }

    #[test]
    fn each_error_gets_its_own_id() {
        let a = server_error(&"x", "w");
        let b = server_error(&"x", "w");
        assert_ne!(a, b);
    }
}
