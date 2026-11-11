//! OpenAI's `POST /v1/responses`, stateless, for Codex: a port of
//! `serving_responses.py` (#813). The request becomes a chat for the model's
//! adapter ([`request`]), and the reply comes back as a response object or its
//! server-sent events ([`reply`]). What needs state or a server this does not
//! have (`previous_response_id`, `conversation`, `store: true`, built-in and
//! custom tools), constrained decoding (a `strict` schema, structured output),
//! images and files are a 400, as are unknown fields. `metadata`,
//! `client_metadata`, `user`, `prompt_cache_key`, `safety_identifier` and
//! `stream_options` are hints and drop.
//!
//! Where the APIs differ:
//! * Reasoning comes back as a `reasoning` item's `reasoning_text` content,
//!   which Codex reads and sends back. There is no summary, and
//!   `encrypted_content` stays unset: the reasoning is no secret.
//! * `reasoning.effort` asks the template for `low` (`minimal`, `low`), `high`
//!   (`medium`, `high`) or `max` (`xhigh`), or for its lowest level when it has
//!   no such one; `none` turns thinking off.
//! * A reply cut off by `max_output_tokens` is `incomplete`, and so is a call
//!   it cut, its `arguments` as far as they got.

mod reply;
mod request;

use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::State;
use axum::response::Response;
use serde_json::Value;

use crate::adapters::OpenAIAdapter;
use crate::client_errors::scrub_paths;
use crate::reply::WireFormat;
use crate::serving::{rid, AppState};
use reply::ResponseReply;

#[allow(clippy::result_large_err)] // the error is the response to send
fn read(
    st: &AppState,
    body: Result<Bytes, BytesRejection>,
) -> Result<(&'static dyn OpenAIAdapter, Value), Response> {
    crate::translate::read::<ResponseReply>(body, || crate::resolve(st, crate::Surface::Responses))
}

/// `POST /v1/responses`.
pub async fn responses(
    State(st): State<AppState>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let (adapter, body) = match read(&st, body) {
        Ok(read) => read,
        Err(r) => return r,
    };
    let chat = match request::to_chat_request(&body, adapter) {
        Ok(chat) => chat,
        Err(e) => return ResponseReply::error(400, &scrub_paths(&e)),
    };
    let sub = match crate::chat_submission(&st, adapter, chat).await {
        Ok(sub) => sub,
        Err(e) => return ResponseReply::error(400, &scrub_paths(&e)),
    };
    let stream = sub.req.stream.unwrap_or(false);
    let request_id = rid("resp");
    let format = ResponseReply::new(request_id.clone(), sub.model, &body);
    crate::reply::serve(
        &st,
        sub.args,
        &request_id,
        stream,
        sub.parser,
        sub.stops,
        format,
    )
    .await
}
