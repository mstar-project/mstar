//! Anthropic's `POST /v1/messages` and `/v1/messages/count_tokens`, for Claude
//! Code: a port of `serving_messages.py` (#813). The request becomes a chat for
//! the model's adapter ([`request`]), and the reply comes back as a message or
//! its server-sent events ([`reply`]). What the model cannot serve is a 400,
//! not dropped: images and documents, server tools, redacted thinking, a final
//! assistant turn (prefill) and unknown fields. `metadata` and `cache_control`
//! are hints and drop.
//!
//! Where the APIs differ:
//! * `thinking` absent leaves the model's default, where Anthropic does not
//!   think unless asked. `budget_tokens` is checked as Anthropic checks it but
//!   not enforced. `output_config.effort` picks the template's reasoning level.
//! * `context_management` is served for keeping every thinking block, which the
//!   template is told to do; other edits are a 400.
//! * A reply's `thinking` block carries an empty `signature`.
//! * A call cut off by `max_tokens` has no `input`: the message leaves it out,
//!   and the stream ends inside its `input_json_delta` pieces.
//! * Stop sequences match the answer's text, not the thinking or a call's input.

mod reply;
mod request;

use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::adapters::OpenAIAdapter;
use crate::client_errors::scrub_paths;
use crate::serving::{rid, AppState};
use reply::MessageReply;
use request::Route;

/// Anthropic's error `type` for an HTTP status.
fn error_type(status: u16) -> &'static str {
    match status {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        400..=499 => "invalid_request_error",
        _ => "api_error",
    }
}

fn error_body(status: u16, message: &str) -> Value {
    json!({"type": "error", "error": {"type": error_type(status), "message": message}})
}

/// An error in Anthropic's shape.
pub fn error(status: u16, message: &str) -> Response {
    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (code, Json(error_body(status, message))).into_response()
}

#[allow(clippy::result_large_err)] // the error is the response to send
fn read(
    st: &AppState,
    body: Result<Bytes, BytesRejection>,
) -> Result<(&'static dyn OpenAIAdapter, Value), Response> {
    crate::translate::read::<MessageReply>(body, || {
        crate::resolve_message(st, crate::Surface::Messages).map_err(|message| error(404, &message))
    })
}

/// `POST /v1/messages`.
pub async fn messages(State(st): State<AppState>, body: Result<Bytes, BytesRejection>) -> Response {
    let (adapter, body) = match read(&st, body) {
        Ok(read) => read,
        Err(r) => return r,
    };
    let translated = match request::to_chat_request(&body, Route::Messages, adapter) {
        Ok(translated) => translated,
        Err(e) => return error(400, &scrub_paths(&e)),
    };
    let sub = match crate::chat_submission(&st, adapter, translated.chat).await {
        Ok(sub) => sub,
        Err(e) => return error(400, &scrub_paths(&e)),
    };
    let stream = sub.req.stream.unwrap_or(false);
    let request_id = rid("msg");
    let format = MessageReply::new(request_id.clone(), sub.model);
    crate::reply::serve(
        &st,
        sub.args,
        &request_id,
        stream,
        sub.parser,
        translated.stop_sequences,
        format,
    )
    .await
}

/// The most `count_tokens` calls rendered at once. Each renders its prompt on
/// the API server and generates nothing, so it runs outside the admission cap;
/// a flood waits here instead of queueing there.
static COUNTING: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(8);

/// `POST /v1/messages/count_tokens`: `input_tokens` of the prompt the request
/// renders, counted by the model's own `process_prompt` behind the bridge.
pub async fn count_tokens(
    State(st): State<AppState>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let (adapter, body) = match read(&st, body) {
        Ok(read) => read,
        Err(r) => return r,
    };
    let req = match request::to_chat_request(&body, Route::CountTokens, adapter) {
        Ok(translated) => translated.chat,
        Err(e) => return error(400, &scrub_paths(&e)),
    };
    let (dir, remote) = (st.upload_dir.clone(), st.allow_remote);
    let args = match crate::blocking(move || adapter.chat_to_request(&req, &dir, remote)).await {
        Ok(args) => args,
        Err(e) => return error(400, &scrub_paths(&e)),
    };
    crate::schedule_upload_cleanup(&st.upload_dir, &args);
    // the semaphore is never closed, so this always holds a permit
    let _permit = COUNTING.acquire().await;
    let Some(bridge) = &st.bridge else {
        return error(
            503,
            "count_tokens needs the model, and this server has no backend",
        );
    };
    match bridge.count_tokens(&args, st.request_timeout).await {
        Ok(n) => Json(json!({"input_tokens": n})).into_response(),
        Err((status, message)) => error(status, &message),
    }
}
