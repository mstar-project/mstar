//! A reply as an OpenAI response object and its stream (`serving_responses.py`).

use axum::response::Response;
use serde_json::{json, Map, Value};

use crate::adapters::truthy;
use crate::reply::{CallHead, Delta, Event, Finish, FinishReason, SseEvent, WireFormat};
use crate::serving::{now, rid};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Status {
    InProgress,
    Completed,
    /// Cut off by the token budget or a failure: a call's arguments are not whole.
    Incomplete,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::InProgress => "in_progress",
            Status::Completed => "completed",
            Status::Incomplete => "incomplete",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Message,
    Reasoning,
    FunctionCall { call_id: String, name: String },
}

/// One output item: its text, or a call's arguments, as far as it has come.
struct Item {
    id: String,
    kind: Kind,
    text: String,
    status: Status,
}

impl Item {
    /// The item's content part, for a message or reasoning item.
    fn part(&self, text: &str) -> Option<Value> {
        match self.kind {
            Kind::Message => Some(
                json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []}),
            ),
            Kind::Reasoning => Some(json!({"type": "reasoning_text", "text": text})),
            Kind::FunctionCall { .. } => None,
        }
    }

    /// The item as the response lists it; `added` leaves its content out, which
    /// arrives in `content_part.added`.
    fn json(&self, added: bool) -> Value {
        let content = if added {
            json!([])
        } else {
            Value::Array(self.part(&self.text).into_iter().collect())
        };
        let status = self.status.as_str();
        match &self.kind {
            Kind::Message => {
                json!({"type": "message", "id": self.id, "status": status, "role": "assistant", "content": content})
            }
            Kind::Reasoning => {
                json!({"type": "reasoning", "id": self.id, "summary": [], "content": content, "status": status})
            }
            Kind::FunctionCall { call_id, name } => {
                json!({"type": "function_call", "id": self.id, "call_id": call_id,
                "name": name, "arguments": self.text, "status": status})
            }
        }
    }

    /// The names of its text's delta and done events, for a message or reasoning item.
    fn text_events(&self) -> (&'static str, &'static str) {
        match self.kind {
            Kind::Reasoning => (
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
            ),
            _ => ("response.output_text.delta", "response.output_text.done"),
        }
    }
}

/// A reply in the Responses shape: the response object as its output items add
/// up, and the stream's events for each step. `response.created` goes out
/// before the first, as clients time the first token from the first event.
pub struct ResponseReply {
    id: String,
    model: String,
    /// The request's fields the response echoes.
    request: Map<String, Value>,
    created: i64,
    output: Vec<Item>,
    finish: Option<Finish>,
    failure: Option<String>,
    started: bool,
    sequence: u64,
}

/// One stream event: its type and its fields.
struct Step {
    kind: &'static str,
    fields: Value,
}

fn step(kind: &'static str, fields: Value) -> Step {
    Step { kind, fields }
}

impl ResponseReply {
    pub fn new(id: String, model: String, request: &Value) -> Self {
        let request = request.as_object().cloned().unwrap_or_default();
        let (output, finish, failure) = (Vec::new(), None, None);
        Self {
            id,
            model,
            request,
            created: now(),
            output,
            finish,
            failure,
            started: false,
            sequence: 0,
        }
    }

    /// Adds one event; returns the stream steps it makes.
    fn add(&mut self, event: &Event) -> Vec<Step> {
        let mut out = Vec::new();
        match event {
            Event::Start { .. } => {}
            Event::Failure { message, .. } => {
                self.failure = Some(message.clone());
                self.close(Status::Incomplete, &mut out);
                out.push(step(
                    "response.failed",
                    json!({"response": self.json(None)}),
                ));
            }
            Event::Finish(f) => {
                self.finish = Some(f.clone());
                let length = f.reason == FinishReason::Length;
                self.close(
                    if length {
                        Status::Incomplete
                    } else {
                        Status::Completed
                    },
                    &mut out,
                );
                let kind = if length {
                    "response.incomplete"
                } else {
                    "response.completed"
                };
                out.push(step(kind, json!({"response": self.json(None)})));
            }
            Event::Delta(Delta::Reasoning(t)) => self.text(Kind::Reasoning, t, &mut out),
            Event::Delta(Delta::Content(t)) => self.text(Kind::Message, t, &mut out),
            Event::Delta(Delta::Call {
                head, arguments, ..
            }) => {
                if let Some(CallHead { id, name }) = head {
                    self.close(Status::Completed, &mut out);
                    let kind = Kind::FunctionCall {
                        call_id: id.clone(),
                        name: name.clone(),
                    };
                    self.open(rid("fc"), kind, &mut out);
                }
                let index = self.output.len().saturating_sub(1);
                if let Some(item) = self.output.last_mut().filter(|_| !arguments.is_empty()) {
                    item.text.push_str(arguments);
                    out.push(step(
                        "response.function_call_arguments.delta",
                        json!({"item_id": item.id, "output_index": index, "delta": arguments}),
                    ));
                }
            }
        }
        out
    }

    fn open(&mut self, id: String, kind: Kind, out: &mut Vec<Step>) {
        let index = self.output.len();
        let item = Item {
            id,
            kind,
            text: String::new(),
            status: Status::InProgress,
        };
        out.push(step(
            "response.output_item.added",
            json!({"output_index": index, "item": item.json(true)}),
        ));
        if let Some(part) = item.part("") {
            out.push(step(
                "response.content_part.added",
                json!({"item_id": item.id, "output_index": index, "content_index": 0, "part": part}),
            ));
        }
        self.output.push(item);
    }

    /// More of a message's or the reasoning's text, in the open item of its kind
    /// or a new one.
    fn text(&mut self, kind: Kind, text: &str, out: &mut Vec<Step>) {
        let open = self
            .output
            .last()
            .is_some_and(|i| i.kind == kind && i.status == Status::InProgress);
        if !open {
            self.close(Status::Completed, out);
            let id = rid(if kind == Kind::Message { "msg" } else { "rs" });
            self.open(id, kind, out);
        }
        let index = self.output.len() - 1;
        let item = &mut self.output[index];
        item.text.push_str(text);
        let mut fields =
            json!({"item_id": item.id, "output_index": index, "content_index": 0, "delta": text});
        if item.kind == Kind::Message {
            fields["logprobs"] = json!([]);
        }
        out.push(step(item.text_events().0, fields));
    }

    /// Ends the open item with `status`.
    fn close(&mut self, status: Status, out: &mut Vec<Step>) {
        let index = self.output.len().saturating_sub(1);
        let Some(item) = self
            .output
            .last_mut()
            .filter(|i| i.status == Status::InProgress)
        else {
            return;
        };
        match (&item.kind, item.part(&item.text)) {
            (_, Some(part)) => {
                let mut fields = json!({"item_id": item.id, "output_index": index, "content_index": 0, "text": item.text});
                if item.kind == Kind::Message {
                    fields["logprobs"] = json!([]);
                }
                out.push(step(item.text_events().1, fields));
                out.push(step(
                    "response.content_part.done",
                    json!({"item_id": item.id, "output_index": index, "content_index": 0, "part": part}),
                ));
            }
            (Kind::FunctionCall { name, .. }, None) => out.push(step(
                "response.function_call_arguments.done",
                json!({"item_id": item.id, "output_index": index, "name": name, "arguments": item.text}),
            )),
            _ => {}
        }
        item.status = status;
        out.push(step(
            "response.output_item.done",
            json!({"output_index": index, "item": item.json(false)}),
        ));
    }

    /// The response object; `status` overrides the one its state gives.
    fn json(&self, status: Option<Status>) -> Value {
        let status = match (status, &self.failure, &self.finish) {
            (Some(status), _, _) => status.as_str(),
            (None, Some(_), _) => "failed",
            (None, None, None) => "in_progress",
            (None, None, Some(f)) if f.reason == FinishReason::Length => "incomplete",
            _ => "completed",
        };
        let echo = |key: &str| self.request.get(key).cloned().unwrap_or(Value::Null);
        let given = |key: &str, default: Value| {
            self.request
                .get(key)
                .filter(|v| truthy(v))
                .cloned()
                .unwrap_or(default)
        };
        let usage = self.finish.as_ref().map(|f| {
            let u = &f.usage;
            json!({"input_tokens": u.prompt, "output_tokens": u.completion,
                   "output_tokens_details": {"reasoning_tokens": u.reasoning}, "total_tokens": u.prompt + u.completion})
        });
        json!({
            "id": self.id, "object": "response", "created_at": self.created, "status": status,
            "background": false,
            "error": self.failure.as_ref().map(|m| json!({"code": "server_error", "message": m})),
            "incomplete_details": if status == "incomplete" { json!({"reason": "max_output_tokens"}) } else { Value::Null },
            "instructions": echo("instructions"), "max_output_tokens": echo("max_output_tokens"),
            "metadata": given("metadata", json!({})), "model": self.model,
            "output": self.output.iter().map(|i| i.json(false)).collect::<Vec<_>>(),
            "parallel_tool_calls": self.request.get("parallel_tool_calls").cloned().unwrap_or(Value::Bool(true)),
            "previous_response_id": null,
            "reasoning": given("reasoning", json!({"effort": null, "summary": null})), "store": false,
            "temperature": echo("temperature"), "text": {"format": {"type": "text"}},
            "tool_choice": self.request.get("tool_choice").cloned().unwrap_or(Value::from("auto")),
            "tools": given("tools", json!([])),
            "top_p": echo("top_p"), "truncation": "disabled", "usage": usage,
        })
    }

    fn sse(&mut self, Step { kind, fields }: Step) -> SseEvent {
        let mut data = Map::new();
        data.insert("type".into(), Value::from(kind));
        data.insert("sequence_number".into(), Value::from(self.sequence));
        if let Value::Object(fields) = fields {
            data.extend(fields);
        }
        self.sequence += 1;
        SseEvent::named(kind, &Value::Object(data))
    }
}

impl WireFormat for ResponseReply {
    fn body(&mut self, events: &[Event]) -> Value {
        for e in events {
            self.add(e);
        }
        self.json(None)
    }

    fn stream(&mut self, batch: &[Event]) -> Vec<SseEvent> {
        let mut out = Vec::new();
        for e in batch {
            if !self.started {
                // before a failure too, as the Python route sent it
                self.started = true;
                let created = step(
                    "response.created",
                    json!({"response": self.json(Some(Status::InProgress))}),
                );
                out.push(self.sse(created));
            }
            for step in self.add(e) {
                out.push(self.sse(step));
            }
        }
        out
    }

    fn error(status: u16, message: &str) -> Response {
        crate::error(status, message, crate::error_type(status))
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

    fn reply() -> ResponseReply {
        ResponseReply::new(
            "resp_1".into(),
            "glm52".into(),
            &json!({"max_output_tokens": 3}),
        )
    }

    fn names(events: &[SseEvent]) -> Vec<&str> {
        events.iter().map(|e| e.name.unwrap_or_default()).collect()
    }

    #[test]
    fn a_call_cut_off_is_incomplete_with_its_arguments_so_far() {
        let head = CallHead {
            id: "call_1".into(),
            name: "f".into(),
        };
        let call = Event::Delta(Delta::Call {
            index: 0,
            head: Some(head),
            arguments: r#"{"a": "#.into(),
        });
        let events = [
            Event::Start { prompt_tokens: 11 },
            call,
            finish(FinishReason::Length),
        ];
        let body = reply().body(&events);
        assert_eq!(
            (body["status"].clone(), body["incomplete_details"].clone()),
            (json!("incomplete"), json!({"reason": "max_output_tokens"}))
        );
        let item = &body["output"][0];
        assert_eq!(
            (item["status"].clone(), item["arguments"].clone()),
            (json!("incomplete"), json!(r#"{"a": "#))
        );
        let stream = reply().stream(&events);
        assert_eq!(names(&stream).last(), Some(&"response.incomplete"));
        let numbers: Vec<u64> = stream
            .iter()
            .map(|e| {
                serde_json::from_str::<Value>(&e.data).unwrap()["sequence_number"]
                    .as_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(numbers, (0..stream.len() as u64).collect::<Vec<_>>());
    }

    #[test]
    fn a_failure_before_any_output_is_created_then_failed() {
        let stream = reply().stream(&[Event::Failure {
            status: 500,
            message: "boom".into(),
        }]);
        assert_eq!(names(&stream), ["response.created", "response.failed"]);
        let failed: Value = serde_json::from_str(&stream[1].data).unwrap();
        assert_eq!(failed["response"]["status"], "failed");
        assert_eq!(
            failed["response"]["error"],
            json!({"code": "server_error", "message": "boom"})
        );
    }

    #[test]
    fn an_empty_reply_is_completed_with_no_output() {
        let events = [
            Event::Start { prompt_tokens: 11 },
            finish(FinishReason::Stop),
        ];
        assert_eq!(
            names(&reply().stream(&events)),
            ["response.created", "response.completed"]
        );
        let body = reply().body(&events);
        assert_eq!(
            (body["status"].clone(), body["output"].clone()),
            (json!("completed"), json!([]))
        );
        assert_eq!(body["usage"]["total_tokens"], 14);
    }
}
