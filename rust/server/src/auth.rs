//! Optional API key (`MSTAR_API_KEY`), the same rule as the Python server's
//! `mstar/api_server/auth.py`: when set, every route except the probes in
//! `UNAUTHENTICATED_PATHS` needs `Authorization: Bearer <key>`, a WebSocket at
//! its handshake. A deployment that serves the openpi socket also takes
//! openpi_client's conventions there, and only there: `Authorization: Api-Key
//! <key>` on the `/` handshake, and `/healthz` without a key. Applied once as a
//! layer over the whole router, so a route added later is covered too.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::header::{AUTHORIZATION, WWW_AUTHENTICATE},
    middleware::Next,
    response::{IntoResponse, Response},
};
use subtle::{Choice, ConstantTimeEq};

pub const API_KEY_ENV: &str = "MSTAR_API_KEY";

/// Served without a key: liveness probes from load balancers and orchestrators,
/// and the Prometheus scrape endpoint (request counts, latencies and the model
/// name; never prompts or outputs).
pub const UNAUTHENTICATED_PATHS: &[&str] = &["/health", "/metrics"];

/// The deployment's key; `None` turns the check off. An empty value is no key.
#[derive(Clone)]
pub struct ApiKey(Option<Arc<[u8]>>);

impl ApiKey {
    pub fn new(key: Option<String>) -> Self {
        Self(
            key.filter(|k| !k.is_empty())
                .map(|k| Arc::from(k.into_bytes())),
        )
    }

    pub fn from_env() -> Self {
        Self::new(std::env::var(API_KEY_ENV).ok())
    }

    pub fn enabled(&self) -> bool {
        self.0.is_some()
    }

    fn accepts(&self, authorization: Option<&[u8]>, schemes: &[&[u8]]) -> bool {
        let Some(key) = &self.0 else { return true };
        let Some(token) = authorization.and_then(|v| scheme_token(v, schemes)) else {
            return false;
        };
        ct_eq(key, token)
    }
}

/// The schemes every route takes, and openpi's socket's.
const SCHEMES: &[&[u8]] = &[b"bearer"];
const OPENPI_SCHEMES: &[&[u8]] = &[b"bearer", b"api-key"];

fn scheme_token<'a>(value: &'a [u8], schemes: &[&[u8]]) -> Option<&'a [u8]> {
    let value = value.trim_ascii();
    let split = value.iter().position(|&b| b == b' ')?;
    let (scheme, token) = value.split_at(split);
    schemes
        .iter()
        .any(|s| scheme.eq_ignore_ascii_case(s))
        .then(|| token.trim_ascii())
}

/// A WebSocket handshake.
fn is_websocket(req: &Request) -> bool {
    req.headers()
        .get(axum::http::header::UPGRADE)
        .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"websocket"))
}

/// Equal in time that depends only on the key's length, never on how much of
/// the token matches or how long it is.
fn ct_eq(key: &[u8], token: &[u8]) -> bool {
    let mut same: Choice = (key.len() as u64).ct_eq(&(token.len() as u64));
    for (i, k) in key.iter().enumerate() {
        same &= k.ct_eq(token.get(i).unwrap_or(&0));
    }
    same.into()
}

pub async fn require_api_key(State(key): State<ApiKey>, req: Request, next: Next) -> Response {
    let authorization = req.headers().get(AUTHORIZATION).map(|v| v.as_bytes());
    let path = req.uri().path();
    let socket = is_websocket(&req);
    let openpi = matches!(path, "/" | "/healthz") && crate::ws::openpi::policy().is_some();
    let openpi_health = openpi && path == "/healthz" && !socket;
    let schemes = if openpi && path == "/" && socket { OPENPI_SCHEMES } else { SCHEMES };
    if UNAUTHENTICATED_PATHS.contains(&path) || openpi_health || key.accepts(authorization, schemes) {
        return next.run(req).await;
    }
    if socket {
        // the Python frontend refuses the handshake (closing before accept): a 403
        return axum::http::StatusCode::FORBIDDEN.into_response();
    }
    let mut resp = crate::error(
        401,
        "missing or invalid API key; send 'Authorization: Bearer <key>'",
        "invalid_request_error",
    );
    resp.headers_mut()
        .insert(WWW_AUTHENTICATE, "Bearer".parse().unwrap());
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::StatusCode, routing::get, Router};
    use tower::ServiceExt;

    fn app(key: Option<&str>) -> Router {
        let key = ApiKey::new(key.map(str::to_string));
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/metrics", get(|| async { "mstar_requests_total 0" }))
            .route("/v1/models", get(|| async { "models" }))
            .layer(axum::middleware::from_fn_with_state(key, require_api_key))
    }

    async fn status(app: Router, path: &str, authorization: Option<&str>) -> StatusCode {
        let mut req = axum::http::Request::builder().uri(path);
        if let Some(a) = authorization {
            req = req.header(AUTHORIZATION, a);
        }
        app.oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn no_key_configured_lets_everything_through() {
        assert_eq!(status(app(None), "/v1/models", None).await, StatusCode::OK);
        assert_eq!(
            status(app(Some("")), "/v1/models", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn a_key_is_required_on_every_route_but_the_probe_and_the_scrape() {
        let k = Some("s3cret");
        assert_eq!(
            status(app(k), "/v1/models", None).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(app(k), "/v1/models", Some("Bearer wrong")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(app(k), "/v1/models", Some("Bearer s3cret-and-more")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(app(k), "/v1/models", Some("Basic s3cret")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(app(k), "/v1/models", Some("Bearer s3cret")).await,
            StatusCode::OK
        );
        assert_eq!(
            status(app(k), "/v1/models", Some("bearer  s3cret ")).await,
            StatusCode::OK
        );
        assert_eq!(status(app(k), "/health", None).await, StatusCode::OK);
        assert_eq!(status(app(k), "/metrics", None).await, StatusCode::OK);
        assert_eq!(
            status(app(k), "/metrics", Some("Bearer wrong")).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn an_unknown_path_is_refused_before_routing() {
        for path in ["/health/", "/metrics/", "/metrics/../v1/models", "/metricsx"] {
            assert_eq!(
                status(app(Some("k")), path, None).await,
                StatusCode::UNAUTHORIZED,
                "{path}"
            );
        }
    }

    #[test]
    fn ct_eq_matches_only_the_exact_key() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab"));
        assert!(!ct_eq(b"abc", b"abcd"));
        assert!(!ct_eq(b"abc", b""));
    }
}
