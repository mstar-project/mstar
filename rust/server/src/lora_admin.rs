//! `POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter`: LoRA's
//! runtime updates, as `openai/router.py` serves them. The body is read as
//! FastAPI reads `LoadLoraAdapterRequest` / `UnloadLoraAdapterRequest`; the
//! call itself is the API server's `LoraAdmin`, over the bridge, which also
//! pushes the served set it changes (`Bridge::serve_models`).

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::adapters::py_value_repr;
use crate::serving::AppState;

/// One of pydantic's errors, as `str(exc.errors())` writes it.
struct FieldError {
    kind: &'static str,
    loc: Option<&'static str>,
    msg: &'static str,
    input: String,
}

impl FieldError {
    fn repr(&self) -> String {
        let loc = match self.loc {
            Some(field) => format!("('body', {})", crate::media::py_repr(field)),
            None => "('body',)".to_string(),
        };
        format!("{{'type': '{}', 'loc': {loc}, 'msg': '{}', 'input': {}}}", self.kind, self.msg, self.input)
    }
}

/// Python's `repr` of a `bytes` value: FastAPI hands a body that isn't JSON to
/// the model as its raw bytes.
fn bytes_repr(b: &[u8]) -> String {
    let quote = if b.contains(&b'\'') && !b.contains(&b'"') { '"' } else { '\'' };
    let mut out = format!("b{quote}");
    for &c in b {
        match c {
            b'\\' => out.push_str("\\\\"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            c if c as char == quote => {
                out.push('\\');
                out.push(quote);
            }
            0x20..=0x7e => out.push(c as char),
            c => out.push_str(&format!("\\x{c:02x}")),
        }
    }
    out.push(quote);
    out
}

/// FastAPI's JSON body: a JSON content type is parsed, any other body is its bytes.
fn is_json(headers: &HeaderMap) -> bool {
    let Some(ct) = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()) else { return false };
    let mime = ct.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    mime == "application/json" || (mime.starts_with("application/") && mime.ends_with("+json"))
}

/// The body's string fields `fields`, in order, or the 400 FastAPI answers
/// with: pydantic's errors, every field's in field order.
fn read_body(headers: &HeaderMap, body: &[u8], fields: &[&'static str]) -> Result<Vec<Value>, Response> {
    let refused = |errors: Vec<FieldError>| {
        let list = errors.iter().map(FieldError::repr).collect::<Vec<_>>().join(", ");
        crate::error(400, &format!("[{list}]"), "invalid_request_error")
    };
    if body.is_empty() {
        return Err(refused(vec![FieldError { kind: "missing", loc: None, msg: "Field required", input: "None".into() }]));
    }
    let not_object = |input: String| {
        refused(vec![FieldError {
            kind: "model_attributes_type",
            loc: None,
            msg: "Input should be a valid dictionary or object to extract fields from",
            input,
        }])
    };
    if !is_json(headers) {
        return Err(not_object(bytes_repr(body)));
    }
    // Python's text for malformed JSON quotes its own decoder's error; serde's is
    // worded differently, so the error's kind and position match, its `ctx` not.
    let value: Value = serde_json::from_slice(body).map_err(|e| {
        let text = String::from_utf8_lossy(body);
        let line_start: usize = text.split_inclusive('\n').take(e.line().saturating_sub(1)).map(|l| l.chars().count()).sum();
        let pos = line_start + e.column().saturating_sub(1);
        let message = format!(
            "[{{'type': 'json_invalid', 'loc': ('body', {pos}), 'msg': 'JSON decode error', 'input': {{}}, 'ctx': {{'error': {}}}}}]",
            crate::media::py_repr(&e.to_string())
        );
        crate::error(400, &message, "invalid_request_error")
    })?;
    let Value::Object(map) = &value else { return Err(not_object(py_value_repr(&value))) };
    let mut errors = Vec::new();
    let mut out = Vec::new();
    for &field in fields {
        match map.get(field) {
            None => errors.push(FieldError { kind: "missing", loc: Some(field), msg: "Field required", input: py_value_repr(&value) }),
            Some(Value::String(s)) => out.push(Value::String(s.clone())),
            Some(other) => errors.push(FieldError {
                kind: "string_type",
                loc: Some(field),
                msg: "Input should be a valid string",
                input: py_value_repr(other),
            }),
        }
    }
    if errors.is_empty() { Ok(out) } else { Err(refused(errors)) }
}

async fn call(st: &AppState, op: &str, name: &Value, path: Option<&Value>) -> Response {
    let Some(bridge) = &st.bridge else {
        return crate::error(503, "Server not ready", "server_error");
    };
    // the API server waits up to the request timeout for the workers' answer
    let timeout = st.request_timeout + std::time::Duration::from_secs(30);
    match bridge.lora(op, name, path, timeout).await {
        Ok(text) => (StatusCode::OK, [(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response(),
        Err((status, message)) => crate::error(status, &message, crate::error_type(status)),
    }
}

/// `POST /v1/load_lora_adapter`.
pub async fn load_lora_adapter(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    match read_body(&headers, &body, &["lora_name", "lora_path"]) {
        Ok(fields) => call(&st, "load", &fields[0], Some(&fields[1])).await,
        Err(refused) => refused,
    }
}

/// `POST /v1/unload_lora_adapter`.
pub async fn unload_lora_adapter(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    match read_body(&headers, &body, &["lora_name"]) {
        Ok(fields) => call(&st, "unload", &fields[0], None).await,
        Err(refused) => refused,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_read_as_pythons_repr() {
        assert_eq!(bytes_repr(br#"{"a": 'b'}"#), r#"b'{"a": \'b\'}'"#);
        assert_eq!(bytes_repr(b"it's\n\x00"), r#"b"it's\n\x00""#);
    }
}
