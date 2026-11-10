//! The WebSocket routes: openpi's policy socket at the server's root,
//! `/generate/ws`, and `/v1/realtime`.

pub mod generate;
pub mod openpi;

use axum::extract::ws::{rejection::WebSocketUpgradeRejection, CloseFrame, Message, WebSocketUpgrade};
use axum::response::Response;

/// A socket route's upgrade, or the reason a request isn't one.
pub type Upgrade = Result<WebSocketUpgrade, WebSocketUpgradeRejection>;

/// The upgrade, or the 404 an HTTP request to a socket route gets: Starlette
/// never matches a WebSocket route to a plain request, whatever its method.
pub fn upgrade(ws: Upgrade) -> Result<WebSocketUpgrade, Response> {
    ws.map_err(|_| crate::detail_error(404, "Not Found"))
}

/// The largest WebSocket message the server reads, in bytes: the config's
/// `ws_max_message_mb` (`MSTAR_WS_MAX_BYTES`, set by the launcher), else what
/// an HTTP body may be. A larger one closes its socket (1009), as on the
/// Python frontend.
pub fn max_message_bytes() -> usize {
    std::env::var("MSTAR_WS_MAX_BYTES")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(crate::body_limit_mb() << 20)
}

/// Completes the close handshake a client started. tungstenite queues its
/// reply when it reads the client's Close and writes it on the next read,
/// write or flush, so a route that returns on that Close flushes first, or the
/// client sees the connection drop (1006) where uvicorn completed the
/// handshake.
pub async fn answer_close<S: futures::Sink<Message> + Unpin>(sink: &mut S) {
    let _ = futures::SinkExt::flush(sink).await;
}

/// The close a failed read gets: a message past [`max_message_bytes`] is 1009,
/// with the reason uvicorn's `websockets` gives it. tungstenite answers that
/// read with an error and queues no close, and it is not a dependency of this
/// crate, so its capacity error is known by its text (`Message too long:
/// {size} > {max}`). Any other failed read ends the socket without one.
pub fn read_failure_close(err: &axum::Error) -> Option<Message> {
    let text = err.to_string();
    let (size, max) = text.split("Message too long: ").nth(1)?.split_once(" > ")?;
    let reason = format!("frame with {} bytes exceeds limit of {} bytes", size.trim(), max.trim());
    Some(Message::Close(Some(CloseFrame { code: 1009, reason: reason.into() })))
}
