//! The WebSocket routes: openpi's policy socket at the server's root,
//! `/generate/ws`, and `/v1/realtime`.

pub mod generate;
pub mod openpi;

use axum::extract::ws::{rejection::WebSocketUpgradeRejection, WebSocketUpgrade};
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
