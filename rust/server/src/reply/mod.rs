//! A text reply as events, which an API writes in its own shape: OpenAI chat
//! ([`chat`]). [`events`] reads a submission's result stream once:
//! the adapter's [`OutputParser`] splits the text into reasoning, answer and
//! tool calls, and the request's stop strings end the answer. [`serve`] answers
//! with a [`WireFormat`], whole or as server-sent events. A chat whose adapter
//! parses nothing and takes no stop strings keeps the plain chat path.

pub mod chat;
mod stop;

use std::convert::Infallible;

use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::{Stream, StreamExt};
use serde_json::Value;

use crate::adapters::SubmitArgs;
use crate::serving::{AppState, Out, OutStream, Outcome};
use crate::utf8::Utf8Stream;
use stop::StopMatcher;

/// A tool call's id and name, on its first [`Delta::Call`].
#[derive(Debug, Clone, PartialEq)]
pub struct CallHead {
    pub id: String,
    pub name: String,
}

/// One piece of a reply, as the output parser writes it.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    /// A tool call: its head on its first piece, then more of its `arguments`.
    Call {
        index: usize,
        head: Option<CallHead>,
        arguments: String,
    },
}

/// Splits a reply's text as it streams: an adapter's `output_parser`.
pub trait OutputParser: Send {
    fn feed(&mut self, text: &str) -> Vec<Delta>;
    /// Whatever it held back, at the end of the reply.
    fn finish(&mut self) -> Vec<Delta>;
    /// The text read so far leaves the reply inside its reasoning.
    fn reasoning(&self) -> bool;
    /// The reply is whole before the model stops (`parallel_tool_calls: false`
    /// after its first call).
    fn done(&self) -> bool;
    /// The reply made tool calls, every one of them whole.
    fn made_calls(&self) -> bool;
}

/// A reply's token counts. Text chunks add up `completion` and carry `prompt`
/// (`num_tokens` and `prompt_tokens` on the bridge), so a reply ended at a stop
/// string has both; the `usage` chunk that closes a reply which ran to its end
/// restates them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Usage {
    pub prompt: u64,
    pub completion: u64,
    /// Of `completion`, the tokens of the chunks that began inside the reasoning.
    pub reasoning: u64,
    /// The engine ended the reply at its token budget.
    pub length: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FinishReason {
    /// A stop token or stop string.
    Stop,
    /// The token budget.
    Length,
    /// The reply made calls, all whole.
    ToolCalls,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Finish {
    pub reason: FinishReason,
    pub usage: Usage,
    /// The stop string that ended the reply.
    pub stop_string: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Delta(Delta),
    /// The request failed after it was submitted; nothing follows.
    Failure {
        status: u16,
        message: String,
    },
    /// Last: why the reply ended and its usage.
    Finish(Finish),
}

/// Appends to `out` the deltas that may go out; returns the stop string that
/// ended the reply, if one did. Stops match the answer only: reasoning and tool
/// calls go out whole. Held answer text goes out before a delta of another
/// kind, so a stop matches within one stretch of answer.
fn passed(
    deltas: Vec<Delta>,
    matcher: Option<&mut StopMatcher>,
    out: &mut Vec<Event>,
) -> Option<String> {
    let Some(matcher) = matcher else {
        out.extend(deltas.into_iter().map(Event::Delta));
        return None;
    };
    for delta in deltas {
        let Delta::Content(content) = delta else {
            let held = matcher.flush();
            if !held.is_empty() {
                out.push(Event::Delta(Delta::Content(held)));
            }
            out.push(Event::Delta(delta));
            continue;
        };
        let (text, stop) = matcher.feed(&content);
        if !text.is_empty() {
            out.push(Event::Delta(Delta::Content(text)));
        }
        if stop.is_some() {
            return stop;
        }
    }
    None
}

/// The state [`events`] reads a result stream with.
struct Reader {
    stream: Option<OutStream>,
    outcome: Outcome,
    parser: Option<Box<dyn OutputParser>>,
    matcher: Option<StopMatcher>,
    text: Utf8Stream,
    usage: Usage,
    /// A call's head has gone out: a stop string can cut a call the parser read.
    calls_out: bool,
}

impl Reader {
    fn deltas(&mut self, piece: &str) -> Vec<Delta> {
        match &mut self.parser {
            Some(p) => p.feed(piece),
            None if piece.is_empty() => Vec::new(),
            None => vec![Delta::Content(piece.to_string())],
        }
    }

    /// [`passed`], noting a call that went out.
    fn pass(&mut self, deltas: Vec<Delta>, out: &mut Vec<Event>) -> Option<String> {
        let from = out.len();
        let stop = passed(deltas, self.matcher.as_mut(), out);
        let head = |e: &Event| matches!(e, Event::Delta(Delta::Call { head: Some(_), .. }));
        self.calls_out |= out[from..].iter().any(head);
        stop
    }

    fn finish(&self, stop_string: Option<String>) -> Event {
        let reason = if self.calls_out && self.parser.as_ref().is_some_and(|p| p.made_calls()) {
            FinishReason::ToolCalls
        } else if stop_string.is_some() || !self.usage.length {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };
        Event::Finish(Finish {
            reason,
            usage: self.usage,
            stop_string,
        })
    }

    /// The events of the next chunk that makes any; `None` once the reply is told.
    async fn next(&mut self) -> Option<Vec<Event>> {
        loop {
            let chunk = match self.stream.as_mut()?.next().await {
                Some(Out::Chunk(chunk)) => chunk,
                Some(Out::Error {
                    status, message, ..
                }) => {
                    self.stream = None;
                    return Some(vec![Event::Failure { status, message }]);
                }
                None => {
                    self.stream = None;
                    return Some(self.tail());
                }
            };
            match chunk.modality.as_str() {
                "text" => {}
                "usage" => {
                    let count =
                        |key: &str| chunk.metadata.get(key).and_then(Value::as_u64).unwrap_or(0);
                    self.usage.prompt = count("prompt_tokens");
                    self.usage.completion = count("completion_tokens");
                    self.usage.length = chunk.metadata.get("finish_reason").and_then(Value::as_str)
                        == Some("length");
                    continue;
                }
                // a reply this path serves is text alone
                _ => continue,
            }
            self.usage.completion += chunk.num_tokens;
            if let Some(prompt) = chunk.prompt_tokens {
                self.usage.prompt = prompt;
            }
            if self.parser.as_ref().is_some_and(|p| p.reasoning()) {
                self.usage.reasoning += chunk.num_tokens;
            }
            let piece = self.text.feed(&chunk.data);
            if piece.is_empty() {
                continue;
            }
            let deltas = self.deltas(&piece);
            let mut events = Vec::new();
            let stop = self.pass(deltas, &mut events);
            if stop.is_some() || self.parser.as_ref().is_some_and(|p| p.done()) {
                // the client has its whole reply: dropping the stream stops the
                // request in the engine, counted as a success
                self.outcome.served();
                self.stream = None;
                events.push(self.finish(stop));
                return Some(events);
            }
            if !events.is_empty() {
                return Some(events);
            }
        }
    }

    fn tail(&mut self) -> Vec<Event> {
        let rest = self.text.flush();
        let mut deltas = self.deltas(&rest);
        if let Some(p) = &mut self.parser {
            deltas.extend(p.finish());
        }
        let mut events = Vec::new();
        let stop = self.pass(deltas, &mut events);
        if stop.is_none() {
            if let Some(held) = self
                .matcher
                .as_mut()
                .map(StopMatcher::flush)
                .filter(|h| !h.is_empty())
            {
                events.push(Event::Delta(Delta::Content(held)));
            }
        }
        events.push(self.finish(stop));
        events
    }
}

/// A submitted chat's reply as events, one batch per chunk that makes any. A
/// stop string, or a parser that is `done`, ends the reply: [`Outcome`] marks
/// the request served and the stream drops, which stops it in the engine.
/// Usage counts every token read until then: the stop's own, as vLLM counts
/// them, and the rest of the chunk it ended in.
pub fn events(
    stream: OutStream,
    outcome: Outcome,
    parser: Option<Box<dyn OutputParser>>,
    stops: Vec<String>,
) -> impl Stream<Item = Vec<Event>> + Send {
    let reader = Reader {
        stream: Some(stream),
        outcome,
        parser,
        matcher: (!stops.is_empty()).then(|| StopMatcher::new(stops)),
        text: Utf8Stream::default(),
        usage: Usage::default(),
        calls_out: false,
    };
    futures::stream::unfold(reader, |mut r| async move {
        r.next().await.map(|events| (events, r))
    })
}

/// Every event of a reply, for a whole response; a failure is its `(status, message)`.
pub async fn collect(events: impl Stream<Item = Vec<Event>>) -> Result<Vec<Event>, (u16, String)> {
    let all: Vec<Event> = events.flat_map(futures::stream::iter).collect().await;
    match all.last() {
        Some(Event::Failure { status, message }) => Err((*status, message.clone())),
        _ => Ok(all),
    }
}

/// A stretch of one kind in a whole reply.
#[derive(Debug, Clone, PartialEq)]
pub enum Run {
    Reasoning(String),
    Content(String),
    Call {
        id: String,
        name: String,
        arguments: String,
    },
}

/// A whole reply's deltas as its stretches of reasoning, answer and calls, in order.
pub fn runs(events: &[Event]) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    // a call's index -> its run
    let mut calls: Vec<usize> = Vec::new();
    for event in events {
        let Event::Delta(delta) = event else { continue };
        match delta {
            Delta::Call {
                head: Some(CallHead { id, name }),
                arguments,
                ..
            } => {
                calls.push(out.len());
                out.push(Run::Call {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                });
            }
            Delta::Call {
                index,
                head: None,
                arguments,
            } => {
                if let Some(Run::Call { arguments: all, .. }) =
                    calls.get(*index).and_then(|&i| out.get_mut(i))
                {
                    all.push_str(arguments);
                }
            }
            Delta::Reasoning(text) => match out.last_mut() {
                Some(Run::Reasoning(prev)) => prev.push_str(text),
                _ => out.push(Run::Reasoning(text.clone())),
            },
            Delta::Content(text) => match out.last_mut() {
                Some(Run::Content(prev)) => prev.push_str(text),
                _ => out.push(Run::Content(text.clone())),
            },
        }
    }
    out
}

/// One server-sent event: its `event:` name, if the API names them, and its data.
#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub name: Option<&'static str>,
    pub data: String,
}

impl SseEvent {
    pub fn data(data: impl Into<String>) -> Self {
        Self {
            name: None,
            data: data.into(),
        }
    }
}

/// One API's shape of a reply: its whole body, its stream, and its errors.
pub trait WireFormat: Send + 'static {
    /// The whole reply.
    fn body(&mut self, events: &[Event]) -> Value;
    /// The stream's events for one batch of reply events. A failure arrives
    /// after the 200, so it travels in-band.
    fn stream(&mut self, batch: &[Event]) -> Vec<SseEvent>;
    /// What follows the last batch.
    fn end(&mut self) -> Vec<SseEvent> {
        Vec::new()
    }
    /// The error response before the 200, or for a whole reply that failed.
    fn error(status: u16, message: &str) -> Response
    where
        Self: Sized;
}

/// Submits `args` and answers with `format`, streamed or whole. The reply is
/// read as events, so the submission reports its usage.
pub async fn serve<F: WireFormat>(
    st: &AppState,
    mut args: SubmitArgs,
    request_id: &str,
    stream: bool,
    parser: Option<Box<dyn OutputParser>>,
    stops: Vec<String>,
    mut format: F,
) -> Response {
    args.report_usage = true;
    crate::schedule_upload_cleanup(&st.upload_dir, &args);
    let (base, outcome) = crate::serving::ended_result_stream(st, &args, request_id, stream);
    if !stream {
        return match collect(events(base, outcome, parser, stops)).await {
            Ok(all) => Json(format.body(&all)).into_response(),
            Err((status, message)) => F::error(status, &message),
        };
    }
    let base = match crate::open_stream(base).await {
        Ok(base) => base,
        Err((status, message)) => return F::error(status, &message),
    };
    let batches = events(base, outcome, parser, stops);
    let sse = futures::stream::unfold(
        (Box::pin(batches), Some(format)),
        |(mut batches, format)| async move {
            let mut format = format?;
            match batches.next().await {
                Some(batch) => {
                    let out = format.stream(&batch);
                    Some((out, (batches, Some(format))))
                }
                None => Some((format.end(), (batches, None))),
            }
        },
    )
    .flat_map(futures::stream::iter)
    .map(|e| {
        let event = axum::response::sse::Event::default().data(e.data);
        Ok::<_, Infallible>(match e.name {
            Some(name) => event.event(name),
            None => event,
        })
    });
    Sse::new(sse)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::glm_reply::{ReplyParser, Tools};
    use crate::bridge::ResultChunk;

    fn text(data: &str, tokens: u64) -> Out {
        Out::Chunk(ResultChunk {
            modality: "text".into(),
            data: data.as_bytes().to_vec(),
            num_tokens: tokens,
            prompt_tokens: Some(11),
            ..Default::default()
        })
    }

    fn usage(completion: u64, finish: &str) -> Out {
        Out::Chunk(ResultChunk {
            modality: "usage".into(),
            metadata: serde_json::json!({"prompt_tokens": 11, "completion_tokens": completion, "finish_reason": finish}),
            ..Default::default()
        })
    }

    fn read(
        items: Vec<Out>,
        parser: Option<Box<dyn OutputParser>>,
        stops: &[&str],
    ) -> (Vec<Event>, bool) {
        let outcome = Outcome::default();
        let stream: OutStream = Box::pin(futures::stream::iter(items));
        let stops = stops.iter().map(|s| s.to_string()).collect();
        let all = futures::executor::block_on(
            events(stream, outcome.clone(), parser, stops)
                .flat_map(futures::stream::iter)
                .collect::<Vec<_>>(),
        );
        (all, outcome.is_served())
    }

    fn finish(events: &[Event]) -> &Finish {
        match events.last() {
            Some(Event::Finish(f)) => f,
            other => panic!("no finish: {other:?}"),
        }
    }

    #[test]
    fn a_reply_that_runs_to_its_end_reports_the_usage_chunk_and_counts_its_reasoning() {
        let parser = Box::new(ReplyParser::new(true, None, None, false));
        let chunks = vec![
            text("hm", 1),
            text("m</think>", 1),
            text("Hi", 2),
            usage(4, "length"),
        ];
        let (events, served) = read(chunks, Some(parser), &[]);
        // the chunk that closes the reasoning began inside it, so it counts
        let want = Usage {
            prompt: 11,
            completion: 4,
            reasoning: 2,
            length: true,
        };
        assert_eq!(
            (finish(&events).reason, finish(&events).usage),
            (FinishReason::Length, want)
        );
        assert!(!served);
    }

    #[test]
    fn a_stop_string_ends_the_reply_as_served_with_the_tokens_read() {
        let parser = Box::new(ReplyParser::new(false, None, None, false));
        let chunks = vec![
            text("abc", 1),
            text("ENDdef", 3),
            text("more", 1),
            usage(5, "length"),
        ];
        let (events, served) = read(chunks, Some(parser), &["END"]);
        let f = finish(&events);
        assert_eq!(
            (
                f.reason,
                f.stop_string.as_deref(),
                f.usage.completion,
                f.usage.prompt
            ),
            (FinishReason::Stop, Some("END"), 4, 11)
        );
        assert_eq!(runs(&events), vec![Run::Content("abc".into())]);
        assert!(served, "a whole reply's request counts as a success");
    }

    #[test]
    fn a_call_a_stop_string_cut_off_makes_no_tool_calls_finish() {
        let tools = Tools::from([("get_time".to_string(), serde_json::Map::new())]);
        let parser = Box::new(ReplyParser::new(false, Some(tools), None, false));
        let chunks = vec![text("doneEND<tool_call>get_time</tool_call>", 3)];
        let (events, served) = read(chunks, Some(parser), &["END"]);
        let f = finish(&events);
        assert_eq!(
            (f.reason, f.stop_string.as_deref()),
            (FinishReason::Stop, Some("END"))
        );
        assert_eq!(runs(&events), vec![Run::Content("done".into())]);
        assert!(served);
    }

    #[test]
    fn a_failure_ends_the_reply_without_a_finish() {
        let failure = Out::Error {
            status: 503,
            message: "busy".into(),
            refused: false,
        };
        let parser = Box::new(ReplyParser::new(false, None, None, false));
        let (events, served) = read(vec![text("x", 1), failure], Some(parser), &[]);
        assert_eq!(
            events.last(),
            Some(&Event::Failure {
                status: 503,
                message: "busy".into()
            })
        );
        assert!(!events.iter().any(|e| matches!(e, Event::Finish(_))) && !served);
    }

    #[test]
    fn an_empty_reply_still_finishes() {
        let (events, _) = read(vec![text("", 1), usage(1, "stop")], None, &["END"]);
        let want = Usage {
            prompt: 11,
            completion: 1,
            ..Usage::default()
        };
        let finish = Finish {
            reason: FinishReason::Stop,
            usage: want,
            stop_string: None,
        };
        assert_eq!(events, vec![Event::Finish(finish)]);
    }
}
