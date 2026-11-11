//! A reply as an Anthropic message and its stream (`serving_messages.py`).

use axum::response::Response;
use serde_json::{json, Value};

use crate::reply::{runs, CallHead, Delta, Event, Finish, FinishReason, Run, SseEvent, WireFormat};

/// The block a stream has open.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Block {
    Thinking,
    Text,
    ToolUse,
}

/// A reply in Anthropic's shape. `message_start` goes out with the first
/// block, not before it, as clients time the first token from the first event.
pub struct MessageReply {
    request_id: String,
    model: String,
    open: Option<Block>,
    /// The open block's index; `-1` before the first.
    index: i64,
}

impl MessageReply {
    pub fn new(request_id: String, model: String) -> Self {
        Self {
            request_id,
            model,
            open: None,
            index: -1,
        }
    }

    fn close(&mut self, out: &mut Vec<SseEvent>) {
        if self.open.take().is_some() {
            out.push(event(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": self.index}),
            ));
        }
    }

    /// Opens a block of `kind` unless one is open; a call always opens its own.
    fn open(&mut self, out: &mut Vec<SseEvent>, kind: Block, block: Value) {
        if self.open == Some(kind) && kind != Block::ToolUse {
            return;
        }
        self.close(out);
        self.open = Some(kind);
        self.index += 1;
        let start =
            json!({"type": "content_block_start", "index": self.index, "content_block": block});
        out.push(event("content_block_start", start));
    }

    fn delta(&self, delta: Value) -> SseEvent {
        event(
            "content_block_delta",
            json!({"type": "content_block_delta", "index": self.index, "delta": delta}),
        )
    }
}

fn event(name: &'static str, data: Value) -> SseEvent {
    SseEvent::named(name, &data)
}

/// `stop_reason` and `stop_sequence`.
fn stop(finish: &Finish) -> (&'static str, Value) {
    let reason = match finish.reason {
        FinishReason::ToolCalls => "tool_use",
        FinishReason::Length => "max_tokens",
        FinishReason::Stop if finish.stop_string.is_some() => "stop_sequence",
        FinishReason::Stop => "end_turn",
    };
    (
        reason,
        finish.stop_string.clone().map_or(Value::Null, Value::from),
    )
}

impl WireFormat for MessageReply {
    fn body(&mut self, events: &[Event]) -> Value {
        let mut content = Vec::new();
        for run in runs(events) {
            match run {
                Run::Reasoning(t) => {
                    content.push(json!({"type": "thinking", "thinking": t, "signature": ""}))
                }
                Run::Content(t) => content.push(json!({"type": "text", "text": t})),
                Run::Call {
                    id,
                    name,
                    arguments,
                } => {
                    // a call cut off by max_tokens has no input to give
                    if let Ok(input) = serde_json::from_str::<Value>(&arguments) {
                        content.push(
                            json!({"type": "tool_use", "id": id, "name": name, "input": input}),
                        );
                    }
                }
            }
        }
        let (reason, sequence, usage) = match events.last() {
            Some(Event::Finish(f)) => {
                let (reason, sequence) = stop(f);
                (reason, sequence, f.usage)
            }
            _ => ("end_turn", Value::Null, Default::default()),
        };
        json!({
            "id": self.request_id, "type": "message", "role": "assistant", "model": self.model, "content": content,
            "stop_reason": reason, "stop_sequence": sequence,
            "usage": {"input_tokens": usage.prompt, "output_tokens": usage.completion},
        })
    }

    fn stream(&mut self, batch: &[Event]) -> Vec<SseEvent> {
        let mut out = Vec::new();
        for e in batch {
            match e {
                Event::Start { prompt_tokens } => out.push(event("message_start", json!({"type": "message_start", "message": {
                    "id": self.request_id, "type": "message", "role": "assistant", "model": self.model, "content": [],
                    "stop_reason": null, "stop_sequence": null,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                }}))),
                Event::Failure { status, message } => out.push(event("error", super::error_body(*status, message))),
                Event::Finish(f) => {
                    self.close(&mut out);
                    let (reason, sequence) = stop(f);
                    out.push(event("message_delta", json!({"type": "message_delta",
                        "delta": {"stop_reason": reason, "stop_sequence": sequence},
                        "usage": {"output_tokens": f.usage.completion}})));
                    out.push(event("message_stop", json!({"type": "message_stop"})));
                }
                Event::Delta(Delta::Reasoning(t)) => {
                    self.open(&mut out, Block::Thinking, json!({"type": "thinking", "thinking": "", "signature": ""}));
                    out.push(self.delta(json!({"type": "thinking_delta", "thinking": t})));
                }
                Event::Delta(Delta::Content(t)) => {
                    self.open(&mut out, Block::Text, json!({"type": "text", "text": ""}));
                    out.push(self.delta(json!({"type": "text_delta", "text": t})));
                }
                Event::Delta(Delta::Call { head, arguments, .. }) => {
                    if let Some(CallHead { id, name }) = head {
                        let block = json!({"type": "tool_use", "id": id, "name": name, "input": {}});
                        self.open(&mut out, Block::ToolUse, block);
                    }
                    if !arguments.is_empty() {
                        out.push(self.delta(json!({"type": "input_json_delta", "partial_json": arguments})));
                    }
                }
            }
        }
        out
    }

    fn error(status: u16, message: &str) -> Response {
        super::error(status, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reply::Usage;

    fn finish(reason: FinishReason) -> Event {
        let usage = Usage {
            prompt: 11,
            completion: 3,
            ..Usage::default()
        };
        Event::Finish(Finish {
            reason,
            usage,
            stop_string: None,
        })
    }

    fn call(arguments: &str) -> Event {
        let head = CallHead {
            id: "call_1".into(),
            name: "f".into(),
        };
        Event::Delta(Delta::Call {
            index: 0,
            head: Some(head),
            arguments: arguments.into(),
        })
    }

    fn names(events: &[SseEvent]) -> Vec<&str> {
        events.iter().map(|e| e.name.unwrap_or_default()).collect()
    }

    fn reply() -> MessageReply {
        MessageReply::new("msg_1".into(), "glm52".into())
    }

    #[test]
    fn a_call_cut_off_streams_its_input_so_far_and_the_message_leaves_it_out() {
        let events = [
            Event::Start { prompt_tokens: 11 },
            Event::Delta(Delta::Reasoning("x".into())),
            call(r#"{"a": "#),
            finish(FinishReason::Length),
        ];
        let stream = reply().stream(&events);
        assert_eq!(
            names(&stream),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert!(stream[5].data.contains(r#""partial_json":"{\"a\": ""#));
        assert!(stream[7].data.contains(r#""stop_reason":"max_tokens""#));
        let body = reply().body(&events);
        assert_eq!(
            body["content"],
            json!([{"type": "thinking", "thinking": "x", "signature": ""}])
        );
        assert_eq!(
            body["usage"],
            json!({"input_tokens": 11, "output_tokens": 3})
        );
    }

    #[test]
    fn a_failure_is_an_error_event_and_nothing_after() {
        let mut format = reply();
        assert_eq!(
            names(&format.stream(&[Event::Failure {
                status: 500,
                message: "boom".into()
            }])),
            ["error"]
        );
        let mut format = reply();
        let mut stream = format.stream(&[
            Event::Start { prompt_tokens: 1 },
            Event::Delta(Delta::Content("a".into())),
        ]);
        stream.extend(format.stream(&[Event::Failure {
            status: 503,
            message: "busy".into(),
        }]));
        stream.extend(format.end());
        let last = stream.last().unwrap();
        assert_eq!(
            (last.name, last.data.contains("overloaded_error")),
            (Some("error"), true)
        );
        assert!(!names(&stream).contains(&"message_stop"));
    }

    #[test]
    fn an_empty_reply_is_a_message_with_no_blocks() {
        let events = [
            Event::Start { prompt_tokens: 11 },
            finish(FinishReason::Stop),
        ];
        assert_eq!(
            names(&reply().stream(&events)),
            ["message_start", "message_delta", "message_stop"]
        );
        let body = reply().body(&events);
        assert_eq!(
            (body["content"].clone(), body["stop_reason"].clone()),
            (json!([]), json!("end_turn"))
        );
    }
}
