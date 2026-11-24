//! Optional generation-control HTTP routes. Registered only with an API key
//! and trusted checkpoint roots; the Python bridge validates the canonical
//! path again before sending the existing conductor control message.

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{Map, Value};

use crate::serving::AppState;

const BODY_LIMIT: usize = 4096;

pub fn enabled(key_enabled: bool, roots: Option<&str>) -> bool {
    key_enabled
        && roots.is_some_and(|value| {
            std::env::split_paths(value).any(|path| !path.as_os_str().is_empty())
        })
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/pause_generation", post(pause))
        .route("/update_weights_from_disk", post(update))
        .route("/continue_generation", post(resume))
        .route("/is_paused", get(state))
}

fn body(bytes: &Bytes) -> Result<Map<String, Value>, Response> {
    if bytes.len() > BODY_LIMIT {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(serde_json::json!({"detail": "generation control body too large"})),
        )
            .into_response());
    }
    if bytes.is_empty() {
        return Ok(Map::new());
    }
    serde_json::from_slice::<Value>(bytes)
        .ok()
        .and_then(|v| v.as_object().cloned())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"detail": "body must be a JSON object"})),
            )
                .into_response()
        })
}

fn validate_pause(fields: &Map<String, Value>) -> Result<(), Response> {
    match fields.get("mode").and_then(Value::as_str) {
        Some("abort" | "wait") => Ok(()),
        _ => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"detail": "mode must be 'abort' or 'wait'"})),
        )
            .into_response()),
    }
}

async fn call(st: AppState, op: &str, fields: Map<String, Value>) -> Response {
    let Some(bridge) = &st.bridge else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "Server not ready"})),
        )
            .into_response();
    };
    if op == "pause_generation" {
        if let Err(error) = validate_pause(&fields) {
            return error;
        }
    }
    let timeout = st.request_timeout + std::time::Duration::from_secs(30);
    match bridge.generation_admin(op, &fields, timeout).await {
        Ok(result) => {
            let status = result
                .get("status")
                .and_then(Value::as_u64)
                .and_then(|n| u16::try_from(n).ok())
                .and_then(|n| StatusCode::from_u16(n).ok());
            match status {
                Some(status) => (status, Json(result)).into_response(),
                None => crate::error(500, "generation admin returned no status", "server_error"),
            }
        }
        Err((status, message)) => {
            let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            (status, Json(serde_json::json!({"detail": message}))).into_response()
        }
    }
}

async fn pause(State(st): State<AppState>, bytes: Bytes) -> Response {
    match body(&bytes) {
        Ok(fields) => call(st, "pause_generation", fields).await,
        Err(error) => error,
    }
}

async fn update(State(st): State<AppState>, bytes: Bytes) -> Response {
    match body(&bytes) {
        Ok(fields) => call(st, "update_weights_from_disk", fields).await,
        Err(error) => error,
    }
}

async fn resume(State(st): State<AppState>, bytes: Bytes) -> Response {
    match body(&bytes) {
        Ok(fields) => call(st, "continue_generation", fields).await,
        Err(error) => error,
    }
}

async fn state(State(st): State<AppState>) -> Response {
    call(st, "is_paused", Map::new()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_large_and_non_object_bodies() {
        assert!(body(&Bytes::from(vec![b' '; BODY_LIMIT + 1])).is_err());
        assert!(body(&Bytes::from_static(b"[]")).is_err());
        assert!(body(&Bytes::from_static(b"{")).is_err());
        assert_eq!(body(&Bytes::from_static(b"{}")).unwrap(), Map::new());
    }

    #[test]
    fn routes_require_both_key_and_checkpoint_roots() {
        assert!(!enabled(false, Some("/weights")));
        assert!(!enabled(true, None));
        assert!(!enabled(true, Some(":")));
        assert!(enabled(true, Some("/weights")));
    }

    #[tokio::test]
    async fn pause_requires_explicit_valid_mode() {
        use http_body_util::BodyExt;

        for raw in [
            "{}",
            r#"{"mode":null}"#,
            r#"{"mode":123}"#,
            r#"{"mode":"invalid"}"#,
            r#"{"mode":""}"#,
        ] {
            let fields: Map<String, Value> = serde_json::from_str(raw).unwrap();
            let err = validate_pause(&fields).expect_err(raw);
            assert_eq!(err.status(), StatusCode::BAD_REQUEST, "{raw}");
            let bytes = err.into_body().collect().await.unwrap().to_bytes();
            let json: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                json,
                serde_json::json!({"detail": "mode must be 'abort' or 'wait'"})
            );
        }
        for raw in [r#"{"mode":"abort"}"#, r#"{"mode":"wait"}"#] {
            let fields: Map<String, Value> = serde_json::from_str(raw).unwrap();
            assert!(validate_pause(&fields).is_ok(), "{raw}");
        }
    }
}
