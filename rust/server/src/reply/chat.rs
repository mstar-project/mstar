//! A parsed reply as OpenAI's `chat.completion` and its `chat.completion.chunk`
//! stream, as `serving_chat.py` wrote them.

use axum::response::Response;
use serde_json::{json, Map, Value};

use super::{runs, CallHead, Delta, Event, FinishReason, Run, SseEvent, Usage, WireFormat};

fn finish_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    }
}

/// OpenAI's chat `usage`, with its field for a reasoning model's share.
fn usage_json(usage: &Usage) -> Value {
    let mut out = json!({
        "prompt_tokens": usage.prompt,
        "completion_tokens": usage.completion,
        "total_tokens": usage.prompt + usage.completion,
    });
    if usage.reasoning > 0 {
        out["completion_tokens_details"] = json!({"reasoning_tokens": usage.reasoning});
    }
    out
}

/// The delta as a `chat.completion.chunk` carries it.
fn delta_json(delta: &Delta) -> Map<String, Value> {
    let delta = match delta {
        Delta::Reasoning(text) => json!({"reasoning_content": text}),
        Delta::Content(text) => json!({"content": text}),
        Delta::Call {
            index,
            head: Some(CallHead { id, name }),
            arguments,
        } => json!({"tool_calls": [{
            "index": index, "id": id, "type": "function", "function": {"name": name, "arguments": arguments},
        }]}),
        Delta::Call {
            index,
            head: None,
            arguments,
        } => {
            json!({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]})
        }
    };
    match delta {
        Value::Object(fields) => fields,
        _ => Map::new(),
    }
}

/// A chat's reply in OpenAI's shape. The stream sends a chunk per delta, with
/// the role on the first and the delta just before the finish riding on the
/// finish chunk, then the usage chunk when `include_usage`, then `[DONE]`. A
/// failure is an in-band error with no `[DONE]`, as mstar-server's other chat
/// streams end.
pub struct ChatReply {
    request_id: String,
    model: String,
    created: i64,
    include_usage: bool,
    /// The role rides on the first chunk: clients time the first token from
    /// it, so nothing goes out before the model has produced one.
    role: bool,
    failed: bool,
    usage: Option<Usage>,
}

impl ChatReply {
    pub fn new(request_id: String, model: String, include_usage: bool) -> Self {
        let created = crate::serving::now();
        Self {
            request_id,
            model,
            created,
            include_usage,
            role: true,
            failed: false,
            usage: None,
        }
    }

    fn chunk(&mut self, delta: Map<String, Value>, finish: Option<FinishReason>) -> SseEvent {
        let mut with_role = Map::new();
        if std::mem::take(&mut self.role) {
            with_role.insert("role".into(), Value::from("assistant"));
        }
        with_role.extend(delta);
        SseEvent::data(
            json!({
                "id": self.request_id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "delta": with_role, "finish_reason": finish.map(finish_reason)}],
            })
            .to_string(),
        )
    }
}

impl WireFormat for ChatReply {
    fn body(&mut self, events: &[Event]) -> Value {
        let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
        for run in runs(events) {
            match run {
                Run::Call { id, name, arguments } => calls.push(json!({
                    "id": id, "type": "function", "function": {"name": name, "arguments": arguments},
                })),
                Run::Reasoning(text) => reasoning.push_str(&text),
                Run::Content(text) => content.push_str(&text),
            }
        }
        // no answer beside calls is null, as OpenAI sends it
        let content = if content.is_empty() && !calls.is_empty() {
            Value::Null
        } else {
            Value::from(content)
        };
        let mut message = json!({"role": "assistant", "content": content});
        if !reasoning.is_empty() {
            message["reasoning_content"] = Value::from(reasoning);
        }
        if !calls.is_empty() {
            message["tool_calls"] = Value::Array(calls);
        }
        let (reason, usage) = match events.last() {
            Some(Event::Finish(f)) => (f.reason, f.usage),
            _ => (FinishReason::Stop, Usage::default()),
        };
        json!({
            "id": self.request_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(reason)}],
            "usage": usage_json(&usage),
        })
    }

    fn stream(&mut self, batch: &[Event]) -> Vec<SseEvent> {
        let mut out = Vec::new();
        for (i, event) in batch.iter().enumerate() {
            match event {
                Event::Delta(_) if matches!(batch.get(i + 1), Some(Event::Finish(_))) => {}
                Event::Delta(delta) => out.push(self.chunk(delta_json(delta), None)),
                Event::Finish(f) => {
                    let last = match i.checked_sub(1).map(|j| &batch[j]) {
                        Some(Event::Delta(delta)) => delta_json(delta),
                        _ => Map::new(),
                    };
                    out.push(self.chunk(last, Some(f.reason)));
                    self.usage = Some(f.usage);
                }
                Event::Failure { status, message } => {
                    self.failed = true;
                    out.push(SseEvent::data(crate::sse_error_json(*status, message)));
                }
            }
        }
        out
    }

    fn end(&mut self) -> Vec<SseEvent> {
        if self.failed {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let (true, Some(usage)) = (self.include_usage, &self.usage) {
            // stream_options.include_usage: one last chunk, with no choices
            out.push(SseEvent::data(
                json!({
                    "id": self.request_id, "object": "chat.completion.chunk", "created": self.created,
                    "model": self.model, "choices": [], "usage": usage_json(usage),
                })
                .to_string(),
            ));
        }
        out.push(SseEvent::data("[DONE]"));
        out
    }

    fn error(status: u16, message: &str) -> Response {
        crate::error(status, message, crate::error_type(status))
    }
}
