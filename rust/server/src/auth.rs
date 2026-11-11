//! Optional API key (`MSTAR_API_KEY`), the same rule as the Python server's
//! `mstar/api_server/auth.py`: when set, every route except the probes in
//! `UNAUTHENTICATED_PATHS` needs `Authorization: Bearer <key>`, a WebSocket at
//! its handshake. Anthropic's routes take `x-api-key: <key>` too, and refuse in
//! Anthropic's error shape. A deployment that serves the openpi socket also takes
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

/// Anthropic's routes. Its SDKs and Claude Code send an API key as `x-api-key`
/// (`Authorization: Bearer` only for an auth token), so these take either.
pub const ANTHROPIC_PATHS: &[&str] = &["/v1/messages", "/v1/messages/count_tokens"];

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

    /// The key under `Authorization` with one of `schemes`, or as `x_api_key`
    /// when given (Anthropic's routes). Each token given is compared, whichever matches.
    fn accepts(
        &self,
        authorization: Option<&[u8]>,
        schemes: &[&[u8]],
        x_api_key: Option<&[u8]>,
    ) -> bool {
        let Some(key) = &self.0 else { return true };
        let bearer = authorization
            .and_then(|v| scheme_token(v, schemes))
            .is_some_and(|t| ct_eq(key, t));
        let x = x_api_key.is_some_and(|t| ct_eq(key, t.trim_ascii()));
        bearer | x
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
    let anthropic = ANTHROPIC_PATHS.contains(&path);
    let x_api_key = anthropic
        .then(|| req.headers().get("x-api-key").map(|v| v.as_bytes()))
        .flatten();
    if UNAUTHENTICATED_PATHS.contains(&path)
        || openpi_health
        || key.accepts(authorization, schemes, x_api_key)
    {
        return next.run(req).await;
    }
    if socket {
        // the Python frontend refuses the handshake (closing before accept): a 403
        return axum::http::StatusCode::FORBIDDEN.into_response();
    }
    let mut resp = if anthropic {
        crate::anthropic::error(
            401,
            "missing or invalid API key; send 'x-api-key: <key>' or 'Authorization: Bearer <key>'",
        )
    } else {
        crate::error(
            401,
            "missing or invalid API key; send 'Authorization: Bearer <key>'",
            "invalid_request_error",
        )
    };
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

    #[tokio::test]
    async fn anthropic_routes_take_x_api_key_and_refuse_in_their_shape() {
        let app = |key: Option<&str>| {
            Router::new()
                .route("/v1/messages", axum::routing::post(|| async { "message" }))
                .route("/v1/models", get(|| async { "models" }))
                .layer(axum::middleware::from_fn_with_state(ApiKey::new(key.map(str::to_string)), require_api_key))
        };
        let send = |path: &str, headers: &[(&str, &str)]| {
            let mut req = axum::http::Request::builder().method(if path == "/v1/messages" { "POST" } else { "GET" }).uri(path);
            for (k, v) in headers {
                req = req.header(*k, *v);
            }
            app(Some("s3cret")).oneshot(req.body(Body::empty()).unwrap())
        };
        assert_eq!(send("/v1/messages", &[("x-api-key", " s3cret ")]).await.unwrap().status(), StatusCode::OK);
        assert_eq!(send("/v1/messages", &[("authorization", "Bearer s3cret")]).await.unwrap().status(), StatusCode::OK);
        // a wrong bearer beside the right x-api-key: each is compared, whichever matches
        assert_eq!(send("/v1/messages", &[("x-api-key", "s3cret"), ("authorization", "Bearer no")]).await.unwrap().status(), StatusCode::OK);
        assert_eq!(send("/v1/models", &[("x-api-key", "s3cret")]).await.unwrap().status(), StatusCode::UNAUTHORIZED);
        let denied = send("/v1/messages", &[("x-api-key", "wrong")]).await.unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(denied.headers()[WWW_AUTHENTICATE], "Bearer");
        let body = axum::body::to_bytes(denied.into_body(), 1 << 16).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "authentication_error");
        assert_eq!(app(None).oneshot(axum::http::Request::post("/v1/messages").body(Body::empty()).unwrap()).await.unwrap().status(), StatusCode::OK);
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
