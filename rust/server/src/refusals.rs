//! `mstar_http_refusals_total` for this frontend: requests answered with a
//! 4xx (or the admission cap's 503) without being submitted to the API server,
//! by route template and status, the rule `RefusalMiddleware` applies on the
//! Python frontend (`mstar/observability/metrics.py`). The counts stay here and
//! travel, cumulative, with each `/metrics` request over the bridge; the API
//! server exports them from its one registry, so the name and help text live
//! only there.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use axum::{
    extract::{MatchedPath, Request, State},
    middleware::Next,
    response::Response,
};
use serde::Serialize;

tokio::task_local! {
    static SUBMITTED: Cell<bool>;
}

/// The request being served reached the API server: the API server counts it
/// in `mstar_requests_total` when it ends, so it is never a refusal. A no-op
/// outside [`count_refusals`] (unit tests that call a handler directly).
pub fn mark_submitted() {
    let _ = SUBMITTED.try_with(|s| s.set(true));
}

/// On the admission cap's 503, so it counts while other 503s (`/health`,
/// `/metrics` with the bridge down) do not.
#[derive(Clone, Copy)]
pub struct CapRefusal;

/// Cumulative counts per (route template, status). Touched once per refusal
/// and once per scrape, never on a request that is served.
#[derive(Default)]
pub struct Refusals(Mutex<BTreeMap<(String, u16), u64>>);

/// The counts as the bridge sends them: three parallel lists.
#[derive(Serialize, Debug, Default, PartialEq)]
pub struct Snapshot {
    pub route: Vec<String>,
    pub status: Vec<u16>,
    pub count: Vec<u64>,
}

const UNMATCHED: &str = "unmatched";

impl Refusals {
    pub(crate) fn count(&self, route: &str, status: u16) {
        *self.0.lock().expect("refusals lock").entry((route.to_string(), status)).or_default() += 1;
    }

    pub fn snapshot(&self) -> Snapshot {
        let counts = self.0.lock().expect("refusals lock");
        let mut snap = Snapshot::default();
        for ((route, status), n) in counts.iter() {
            snap.route.push(route.clone());
            snap.status.push(*status);
            snap.count.push(*n);
        }
        snap
    }
}

/// Layer over every route and the fallback, outside the API key check (its
/// 401 comes before the handler) and inside CORS (a preflight is not a
/// request). The route is axum's matched path, `unmatched` for none.
pub async fn count_refusals(
    State(refusals): State<Arc<Refusals>>,
    req: Request,
    next: Next,
) -> Response {
    let route = req.extensions().get::<MatchedPath>().cloned();
    let (resp, submitted) = SUBMITTED
        .scope(Cell::new(false), async {
            let resp = next.run(req).await;
            (resp, SUBMITTED.with(Cell::get))
        })
        .await;
    let status = resp.status();
    if !submitted && (status.is_client_error() || resp.extensions().get::<CapRefusal>().is_some()) {
        refusals.count(route.as_ref().map_or(UNMATCHED, MatchedPath::as_str), status.as_u16());
    }
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{header::AUTHORIZATION, StatusCode},
        response::IntoResponse,
        routing::{get, post},
        Router,
    };
    use tower::ServiceExt;

    use crate::auth::{require_api_key, ApiKey};

    async fn refused_before_submit() -> StatusCode {
        StatusCode::BAD_REQUEST
    }

    async fn failed_after_submit() -> StatusCode {
        mark_submitted();
        StatusCode::BAD_REQUEST
    }

    async fn cap_refusal() -> Response {
        let mut resp = StatusCode::SERVICE_UNAVAILABLE.into_response();
        resp.extensions_mut().insert(CapRefusal);
        resp
    }

    async fn bridge_down() -> StatusCode {
        StatusCode::SERVICE_UNAVAILABLE
    }

    /// The layering `main` uses: routes, then the key check, then the counter.
    fn app(refusals: Arc<Refusals>) -> Router {
        Router::new()
            .route("/health", get(bridge_down))
            .route("/generate", post(refused_before_submit))
            .route("/v1/chat/completions", post(failed_after_submit))
            .route("/v1/audio/speech", post(cap_refusal))
            .route("/v1/models", get(|| async { "models" }))
            .layer(axum::middleware::from_fn_with_state(
                ApiKey::new(Some("k".into())),
                require_api_key,
            ))
            .layer(axum::middleware::from_fn_with_state(refusals, count_refusals))
    }

    async fn status(app: Router, method: &str, path: &str, key: bool) -> StatusCode {
        let mut req = axum::http::Request::builder().method(method).uri(path);
        if key {
            req = req.header(AUTHORIZATION, "Bearer k");
        }
        app.oneshot(req.body(Body::empty()).unwrap()).await.unwrap().status()
    }

    fn counts(refusals: &Refusals) -> Vec<(String, u16, u64)> {
        let s = refusals.snapshot();
        (0..s.route.len()).map(|i| (s.route[i].clone(), s.status[i], s.count[i])).collect()
    }

    #[tokio::test]
    async fn refusals_count_by_route_template_and_status_and_nothing_else_does() {
        let refusals = Arc::new(Refusals::default());
        let calls = [
            ("POST", "/generate", false, StatusCode::UNAUTHORIZED),
            ("POST", "/generate", false, StatusCode::UNAUTHORIZED),
            ("POST", "/no/such/route", false, StatusCode::UNAUTHORIZED),
            ("POST", "/generate", true, StatusCode::BAD_REQUEST),
            ("GET", "/generate", true, StatusCode::METHOD_NOT_ALLOWED),
            ("GET", "/no/such/route", true, StatusCode::NOT_FOUND),
            ("POST", "/v1/audio/speech", true, StatusCode::SERVICE_UNAVAILABLE),
            // Not refusals: submitted then failed, a 503 that is not the cap,
            // and a served request.
            ("POST", "/v1/chat/completions", true, StatusCode::BAD_REQUEST),
            ("GET", "/health", false, StatusCode::SERVICE_UNAVAILABLE),
            ("GET", "/v1/models", true, StatusCode::OK),
        ];
        for (method, path, key, expected) in calls {
            assert_eq!(status(app(refusals.clone()), method, path, key).await, expected, "{method} {path}");
        }
        assert_eq!(
            counts(&refusals),
            [
                ("/generate".to_string(), 400, 1),
                ("/generate".to_string(), 401, 2),
                ("/generate".to_string(), 405, 1),
                ("/v1/audio/speech".to_string(), 503, 1),
                ("unmatched".to_string(), 401, 1),
                ("unmatched".to_string(), 404, 1),
            ]
        );
    }

    #[test]
    fn the_snapshot_goes_over_the_bridge_as_three_parallel_lists() {
        let refusals = Refusals::default();
        assert_eq!(
            rmp_serde::from_slice::<serde_json::Value>(&rmp_serde::to_vec_named(&refusals.snapshot()).unwrap())
                .unwrap(),
            serde_json::json!({"route": [], "status": [], "count": []})
        );
        refusals.count("/generate", 401);
        refusals.count(UNMATCHED, 404);
        refusals.count("/generate", 401);
        let wire: serde_json::Value =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&refusals.snapshot()).unwrap()).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({"route": ["/generate", "unmatched"], "status": [401, 404], "count": [2, 1]})
        );
    }
}
