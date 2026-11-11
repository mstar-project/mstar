//! What `/v1/messages` (`crate::anthropic`) and `/v1/responses`
//! (`crate::responses`) share in translating their requests onto a chat for the
//! model's adapter; each answers in its own [`WireFormat`].

use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::http::StatusCode;
use axum::response::Response;
use serde_json::{Map, Value};

use crate::adapters::{py_value_repr, OpenAIAdapter};
use crate::reply::WireFormat;

/// The adapter and the body as JSON, or the error to answer with: a body over
/// the limit is a 413 before the route is known, as the body limit answers it,
/// then `resolve` refuses a model without the route, then the body must parse.
// the error is the response to send, as main.rs's resolve returns it
#[allow(clippy::result_large_err)]
pub fn read<F: WireFormat>(
    body: Result<Bytes, BytesRejection>,
    resolve: impl FnOnce() -> Result<&'static dyn OpenAIAdapter, Response>,
) -> Result<(&'static dyn OpenAIAdapter, Value), Response> {
    if body
        .as_ref()
        .is_err_and(|rej| rej.status() == StatusCode::PAYLOAD_TOO_LARGE)
    {
        return Err(F::error(413, &crate::body_too_large()));
    }
    let adapter = resolve()?;
    let bytes = body.map_err(|rej| F::error(400, &rej.body_text()))?;
    let body = serde_json::from_slice(&bytes)
        .map_err(|e| F::error(400, &format!("the request body is not JSON: {e}")))?;
    Ok((adapter, body))
}

/// Refuses a field outside `fields`, listed as Python lists `sorted(set(body) - fields)`.
pub fn refuse_unknown(body: &Map<String, Value>, fields: &[&str]) -> Result<(), String> {
    let mut unknown: Vec<&String> = body
        .keys()
        .filter(|k| !fields.contains(&k.as_str()))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort();
    let listed = Value::Array(
        unknown
            .into_iter()
            .map(|k| Value::from(k.as_str()))
            .collect(),
    );
    Err(format!("unsupported field(s) {}", py_value_repr(&listed)))
}

/// M*'s per-request prefix cache opt-out, as on chat completions.
pub fn prefix_cache(
    body: &Map<String, Value>,
    chat: &mut Map<String, Value>,
) -> Result<(), String> {
    if let Some(v) = body.get("prefix_cache") {
        if !v.is_boolean() {
            return Err(format!(
                "prefix_cache must be a boolean; got {}",
                py_value_repr(v)
            ));
        }
        chat.insert("prefix_cache".into(), v.clone());
    }
    Ok(())
}

/// How hard an agent asks the model to think.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Effort {
    Low,
    Medium,
    High,
    /// The client's top level (`xhigh`, `max`).
    Top,
}

/// The template's `reasoning_effort` for `effort`, by GLM's level names, or the
/// template's lowest where it has no such level. High stays high: GLM's `max`
/// nearly doubles an agent's output tokens. `asked` names the field in a refusal.
pub fn template_level(
    effort: Effort,
    adapter: &dyn OpenAIAdapter,
    asked: &str,
) -> Result<Map<String, Value>, String> {
    let levels = adapter.reasoning_efforts();
    let Some(lowest) = levels.first() else {
        return Err(format!("{asked}: this model has no reasoning levels"));
    };
    let wanted = match effort {
        Effort::Low => "low",
        Effort::Medium | Effort::High => "high",
        Effort::Top => "max",
    };
    let level = if levels.iter().any(|l| l == wanted) {
        wanted
    } else {
        lowest.as_str()
    };
    Ok(Map::from_iter([(
        "reasoning_effort".to_string(),
        Value::from(level),
    )]))
}
