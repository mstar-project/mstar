//! GLM-5.2 / GLM-5.3 replies split into OpenAI deltas: `reasoning_content`,
//! `content` and `tool_calls`. A port of `glm_chat.GlmReplyParser`.
//!
//! The template ends the prompt inside `<think>` when the model reasons, so the
//! reply opens with reasoning and `</think>` hands over to the answer. A call is
//! `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`:
//! a string value arrives bare and any other as JSON, as the template renders them.

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use serde::Serialize;
use serde_json::{Map, Value};

use super::py_strip;
use crate::reply::{CallHead, Delta, OutputParser};

pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const CALL_START: &str = "<tool_call>";
pub const CALL_END: &str = "</tool_call>";
const ARG_KEY: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";

/// Each offered function's parameter properties, by name.
pub type Tools = BTreeMap<String, Map<String, Value>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Reasoning,
    Content,
    Call,
}

/// Where a call is being read: `Name` until it ends, then `Args` (between
/// arguments), `Key`, `Open` (past `</arg_key>`, before `<arg_value>`) and
/// `Value`; `Text` for a call to a function not offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Name,
    Text,
    Args,
    Key,
    Open,
    Value,
}

impl Phase {
    /// What ends each phase.
    fn tags(self) -> &'static [&'static str] {
        match self {
            Phase::Name | Phase::Args => &[ARG_KEY, CALL_END],
            Phase::Text => &[CALL_END],
            Phase::Key => &[ARG_KEY_END, CALL_END],
            Phase::Value => &[ARG_VALUE_END, CALL_END],
            Phase::Open => &[],
        }
    }
}

/// As vLLM's `glm47` parser: `<think>` in the answer reopens the reasoning, a
/// stray `<think>` in the reasoning and `</think>` in the answer drop, a call may
/// end the reasoning without `</think>`, and a call to a function the request
/// did not offer stays text. Content is stripped at both ends, as the template
/// strips it on the way back. `feed` holds back text that may be the start of a
/// tag; `finish` flushes the rest.
///
/// A call streams as it is written: its id and name once the name ends, then
/// its `arguments` in pieces that add up to their `json.dumps` in any chunking.
/// A value the schema allows only as a string streams as it comes; any other
/// goes whole at `</arg_value>`. A key given twice is in the JSON twice. A reply
/// cut off inside a call keeps the call as far as it got, with no `tool_calls`
/// finish; cut off before the name ends, the call is text.
pub struct ReplyParser {
    thinking: bool,
    /// End the reply at the end of its first call (`parallel_tool_calls: false`).
    single_call: bool,
    done: bool,
    state: State,
    tools: Option<Tools>,
    buf: String,
    phase: Phase,
    /// The call's text so far, kept while it may still turn out text.
    raw: String,
    calls: usize,
    cut: bool,
    props: Map<String, Value>,
    nargs: usize,
    key: String,
    string: bool,
    value: String,
    content_started: bool,
    held_space: String,
}

impl ReplyParser {
    /// `forced_call` says the prompt opened a call, and named it unless empty.
    pub fn new(
        thinking: bool,
        mut tools: Option<Tools>,
        forced_call: Option<String>,
        single_call: bool,
    ) -> Self {
        if let (Some(name), Some(all)) = (
            forced_call.as_deref().filter(|n| !n.is_empty()),
            tools.as_mut(),
        ) {
            // the prompt opened a call to this function: a name the model goes on to
            // write (get_weather -> get_weather_hourly) is not another tool's call
            if let Some(props) = all.remove(name) {
                *all = Tools::from([(name.to_string(), props)]);
            }
        }
        let (state, raw) = match forced_call {
            Some(name) => (State::Call, name),
            None => (
                if thinking {
                    State::Reasoning
                } else {
                    State::Content
                },
                String::new(),
            ),
        };
        Self {
            thinking,
            single_call,
            done: false,
            state,
            tools,
            buf: String::new(),
            phase: Phase::Name,
            raw,
            calls: 0,
            cut: false,
            props: Map::new(),
            nargs: 0,
            key: String::new(),
            string: false,
            value: String::new(),
            content_started: false,
            held_space: String::new(),
        }
    }

    fn offers_tools(&self) -> bool {
        self.tools.as_ref().is_some_and(|t| !t.is_empty())
    }

    fn tags(&self) -> Vec<&'static str> {
        let mut tags = Vec::new();
        if self.state == State::Reasoning || self.thinking {
            tags.extend([THINK_START, THINK_END]);
        }
        if self.offers_tools() {
            tags.push(CALL_START);
        }
        tags
    }

    fn drain(&mut self, last: bool) -> Vec<Delta> {
        let mut out = Vec::new();
        while !self.buf.is_empty() || (last && self.state == State::Call) {
            if self.state == State::Call {
                if !self.drain_call(&mut out, last) {
                    break;
                }
                continue;
            }
            let tags = self.tags();
            let Some((i, tag)) = first_tag(&self.buf, &tags) else {
                let keep = if last || tags.is_empty() {
                    0
                } else {
                    partial_tag(&self.buf, &tags)
                };
                let at = self.buf.len() - keep;
                let text = self.buf[..at].to_string();
                self.emit(&mut out, &text);
                self.buf.drain(..at);
                break;
            };
            let text = self.buf[..i].to_string();
            self.emit(&mut out, &text);
            self.buf.drain(..i + tag.len());
            if tag == THINK_END && self.state == State::Reasoning {
                self.state = State::Content;
            } else if tag == THINK_START {
                self.state = State::Reasoning;
            } else if tag == CALL_START {
                self.state = State::Call;
                self.phase = Phase::Name;
            }
        }
        merged(out)
    }

    /// Reads what it can of the call; false once it needs more text, or at the
    /// end of the reply. What is set aside or sent is never searched again, so
    /// a long call is read once.
    fn drain_call(&mut self, out: &mut Vec<Delta>, last: bool) -> bool {
        let phase = self.phase;
        if phase == Phase::Open {
            let rest = self.buf.trim_start_matches(super::py_space).to_string();
            if let Some(value) = rest.strip_prefix(ARG_VALUE) {
                self.buf = value.to_string();
                self.open_value(out);
                return true;
            }
            let waits = ARG_VALUE.starts_with(rest.as_str()) && !last;
            self.buf = rest;
            if waits {
                return false;
            }
            self.phase = Phase::Args; // no value follows: the key drops
            return true;
        }
        let buf = std::mem::take(&mut self.buf);
        let Some((i, tag)) = first_tag(&buf, phase.tags()) else {
            if last {
                self.buf = buf;
                self.end_cut_off(out);
                return false;
            }
            let at = buf.len() - partial_tag(&buf, phase.tags());
            self.buf = buf[at..].to_string();
            self.take(out, &buf[..at]);
            return false;
        };
        self.buf = buf[i + tag.len()..].to_string();
        if phase == Phase::Name {
            let name = py_strip(&format!("{}{}", self.raw, &buf[..i])).to_string();
            if !self.tools.as_ref().is_some_and(|t| t.contains_key(&name)) {
                self.raw.push_str(&buf[..i]);
                self.buf = buf[i..].to_string(); // the call's text, tags and all
                self.phase = Phase::Text;
                return true;
            }
            self.start_call(out, &name);
        } else {
            self.take(out, &buf[..i]);
        }
        if tag == ARG_KEY {
            self.phase = Phase::Key;
            self.key.clear();
        } else if tag == ARG_KEY_END {
            self.phase = Phase::Open;
        } else if tag == ARG_VALUE_END {
            self.close_value(out);
            self.phase = Phase::Args;
        } else if phase == Phase::Text {
            let text = format!("{CALL_START}{}{CALL_END}", self.raw);
            self.as_text(out, &text);
        } else {
            if phase == Phase::Value {
                self.close_value(out); // the call closed it
            }
            self.args(out, if self.nargs > 0 { "}" } else { "{}" });
            self.state = State::Content;
            if self.single_call {
                self.done = true;
                self.buf.clear();
                return false;
            }
        }
        true
    }

    /// Text read in the current phase of a call.
    fn take(&mut self, out: &mut Vec<Delta>, text: &str) {
        match self.phase {
            Phase::Name | Phase::Text => self.raw.push_str(text),
            Phase::Key => self.key.push_str(text),
            Phase::Value if self.string => {
                let quoted = dumps(&Value::from(text));
                self.args(out, &quoted[1..quoted.len() - 1]);
            }
            Phase::Value => self.value.push_str(text),
            // between arguments, text drops
            Phase::Args | Phase::Open => {}
        }
    }

    fn start_call(&mut self, out: &mut Vec<Delta>, name: &str) {
        self.raw.clear();
        self.props = self
            .tools
            .as_ref()
            .and_then(|t| t.get(name))
            .cloned()
            .unwrap_or_default();
        self.nargs = 0;
        self.held_space.clear();
        let index = self.calls;
        self.calls += 1;
        let id = format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
        let head = CallHead {
            id,
            name: name.to_string(),
        };
        out.push(Delta::Call {
            index,
            head: Some(head),
            arguments: String::new(),
        });
        self.phase = Phase::Args;
    }

    fn open_value(&mut self, out: &mut Vec<Delta>) {
        let key = py_strip(&self.key).to_string();
        let types = json_types(self.props.get(&key));
        self.string = string_only(&types);
        self.value.clear();
        let opening = format!(
            "{}{}: {}",
            if self.nargs == 0 { "{" } else { ", " },
            dumps(&Value::from(key.as_str())),
            if self.string { "\"" } else { "" }
        );
        self.args(out, &opening);
        self.nargs += 1;
        self.phase = Phase::Value;
    }

    fn close_value(&mut self, out: &mut Vec<Delta>) {
        if self.string {
            self.args(out, "\"");
            return;
        }
        let key = py_strip(&self.key).to_string();
        let value = arg_value(&self.value, self.props.get(&key));
        self.args(out, &dumps(&value));
    }

    fn args(&mut self, out: &mut Vec<Delta>, text: &str) {
        if !text.is_empty() {
            out.push(Delta::Call {
                index: self.calls - 1,
                head: None,
                arguments: text.to_string(),
            });
        }
    }

    /// The reply ended inside the call.
    fn end_cut_off(&mut self, out: &mut Vec<Delta>) {
        if matches!(self.phase, Phase::Name | Phase::Text) {
            // its text is all there is
            let text = format!("{CALL_START}{}{}", self.raw, std::mem::take(&mut self.buf));
            self.as_text(out, &text);
            return;
        }
        if self.phase == Phase::Value && self.string {
            let rest = std::mem::take(&mut self.buf);
            self.take(out, &rest);
        }
        self.buf.clear();
        self.cut = true;
        self.state = State::Content;
    }

    fn as_text(&mut self, out: &mut Vec<Delta>, text: &str) {
        self.raw.clear();
        self.state = State::Content;
        self.emit(out, text);
    }

    fn emit(&mut self, out: &mut Vec<Delta>, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.state == State::Reasoning {
            out.push(Delta::Reasoning(text.to_string()));
            return;
        }
        let text = if self.content_started {
            text
        } else {
            text.trim_start_matches(super::py_space)
        };
        let body = text.trim_end_matches(super::py_space);
        if body.is_empty() {
            self.held_space.push_str(text);
            return;
        }
        out.push(Delta::Content(format!(
            "{}{body}",
            std::mem::take(&mut self.held_space)
        )));
        self.held_space = text[body.len()..].to_string();
        self.content_started = true;
    }
}

impl OutputParser for ReplyParser {
    fn feed(&mut self, text: &str) -> Vec<Delta> {
        if self.done {
            return Vec::new();
        }
        self.buf.push_str(text);
        self.drain(false)
    }

    fn finish(&mut self) -> Vec<Delta> {
        let out = self.drain(true);
        self.held_space.clear();
        out
    }

    fn reasoning(&self) -> bool {
        self.state == State::Reasoning
    }

    fn done(&self) -> bool {
        self.done
    }

    fn made_calls(&self) -> bool {
        self.calls > 0 && !self.cut
    }
}

/// The earliest of `tags` in `text`, by its byte offset.
fn first_tag(text: &str, tags: &[&'static str]) -> Option<(usize, &'static str)> {
    tags.iter()
        .filter_map(|t| text.find(t).map(|i| (i, *t)))
        .min()
}

/// Length of the longest end of `text` that may begin one of `tags`. Only its
/// last `<` can: a tag holds no other. The tail after it is ASCII when it
/// matches, so its length in bytes is Python's in characters.
fn partial_tag(text: &str, tags: &[&str]) -> usize {
    let longest = tags.iter().map(|t| t.len()).max().unwrap_or(0);
    let from = text.len().saturating_sub(longest);
    let Some(start) = text.as_bytes()[from..].iter().rposition(|&b| b == b'<') else {
        return 0;
    };
    let tail = &text[from + start..];
    if tags.iter().any(|t| t.starts_with(tail)) {
        tail.len()
    } else {
        0
    }
}

/// The JSON types a property schema admits; empty when it does not say.
pub fn json_types(schema: Option<&Value>) -> BTreeSet<String> {
    let mut types = BTreeSet::new();
    let Some(Value::Object(schema)) = schema else {
        return types;
    };
    match schema.get("type") {
        Some(Value::String(t)) => {
            types.insert(t.clone());
        }
        Some(Value::Array(kinds)) => {
            types.extend(kinds.iter().filter_map(Value::as_str).map(str::to_string))
        }
        _ => {}
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(Value::Array(branches)) = schema.get(key) {
            for branch in branches {
                types.extend(json_types(Some(branch)));
            }
        }
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        for v in values {
            types.insert(
                match v {
                    Value::String(_) => "string",
                    Value::Bool(_) => "boolean",
                    Value::Number(n) if n.is_f64() => "number",
                    Value::Number(_) => "integer",
                    _ => "object",
                }
                .to_string(),
            );
        }
    }
    types
}

fn string_only(types: &BTreeSet<String>) -> bool {
    !types.is_empty() && types.iter().all(|t| t == "string" || t == "null")
}

/// A string value arrives bare and any other as JSON; a value the schema allows
/// only as a string stays one ("94110" for a zip code), and one off its schema
/// stays as the model wrote it.
pub fn arg_value(raw: &str, schema: Option<&Value>) -> Value {
    let types = json_types(schema);
    let text = || Value::from(raw);
    if string_only(&types) {
        return text();
    }
    // NaN, Infinity and 1e400 are not JSON, which the arguments must stay
    let Ok(mut value) = serde_json::from_str::<Value>(raw) else {
        return text();
    };
    let mut kind = json_type(&value);
    if let Some(f) = value
        .as_f64()
        .filter(|f| kind == "number" && types.contains("integer") && f.fract() == 0.0)
    {
        // an integer the model wrote as 3.0 is still one
        kind = "integer";
        if (i64::MIN as f64..i64::MAX as f64).contains(&f) {
            value = Value::from(f as i64);
        }
    }
    if !types.is_empty()
        && !types.contains(kind)
        && !(kind == "integer" && types.contains("number"))
    {
        return text();
    }
    value
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_f64() => "number",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// `json.dumps`'s separators, `", "` and `": "`, as the arguments vLLM and the
/// template write: a value written whole reads like the pieces streamed around it.
struct PySeparators;

impl serde_json::ser::Formatter for PySeparators {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.begin_array_value(writer, first)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

/// `value` as JSON with [`PySeparators`]; non-ASCII stays as it is.
fn dumps(value: &Value) -> String {
    let mut out = Vec::new();
    // a Value always serializes into memory
    let _ = value.serialize(&mut serde_json::Serializer::with_formatter(
        &mut out,
        PySeparators,
    ));
    String::from_utf8(out).unwrap_or_default()
}

/// Adjacent deltas of one kind as one: text joins, and a call's argument pieces
/// join onto the delta before them.
fn merged(deltas: Vec<Delta>) -> Vec<Delta> {
    let mut out: Vec<Delta> = Vec::new();
    for d in deltas {
        match (out.last_mut(), d) {
            (Some(Delta::Reasoning(prev)), Delta::Reasoning(text))
            | (Some(Delta::Content(prev)), Delta::Content(text)) => prev.push_str(&text),
            (
                Some(Delta::Call {
                    index: prev,
                    arguments,
                    ..
                }),
                Delta::Call {
                    index,
                    head: None,
                    arguments: more,
                },
            ) if *prev == index => arguments.push_str(&more),
            (_, d) => out.push(d),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reply::chat::ChatReply;
    use crate::reply::{Event, Finish, FinishReason, Usage, WireFormat};
    use serde_json::json;

    const CALL: &str = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
        <arg_key>days</arg_key><arg_value>3</arg_value><arg_key>zip</arg_key>\
        <arg_value>75001</arg_value></tool_call>";

    fn tools(defs: Value) -> Tools {
        super::super::glm::tool_parameters(&defs).unwrap().params
    }

    fn weather() -> Tools {
        tools(
            json!([{"type": "function", "function": {"name": "get_weather", "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "zip": {"type": "string"}},
            }}}]),
        )
    }

    fn edit() -> Tools {
        tools(
            json!([{"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": ["string", "null"]},
            "line": {"type": "integer"}, "opts": {"type": "object"}, "tags": {"type": "array"}, "any": {}}}}}]),
        )
    }

    fn parser(thinking: bool, tools: Option<Tools>) -> ReplyParser {
        ReplyParser::new(thinking, tools, None, false)
    }

    /// The deltas of `text` fed in `size`-character pieces.
    fn fed(p: &mut ReplyParser, text: &str, size: usize) -> Vec<Delta> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        for piece in chars.chunks(size.max(1)) {
            out.extend(p.feed(&piece.iter().collect::<String>()));
        }
        out.extend(p.finish());
        out
    }

    /// A whole reply's chat `message`, as the chat route builds it.
    fn message(mut p: ReplyParser, text: &str) -> Value {
        let mut events: Vec<Event> = fed(&mut p, text, text.len())
            .into_iter()
            .map(Event::Delta)
            .collect();
        let reason = if p.made_calls() {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        };
        events.push(Event::Finish(Finish {
            reason,
            usage: Usage::default(),
            stop_string: None,
        }));
        let mut chat = ChatReply::new("chatcmpl-0".into(), "glm52".into(), false);
        chat.body(&events)["choices"][0]["message"].clone()
    }

    /// Reasoning, content, and each call's (name, arguments).
    fn summary(deltas: &[Delta]) -> (String, String, Vec<(String, String)>) {
        let (mut reasoning, mut content, mut calls) =
            (String::new(), String::new(), Vec::<(String, String)>::new());
        for d in deltas {
            match d {
                Delta::Reasoning(t) => reasoning.push_str(t),
                Delta::Content(t) => content.push_str(t),
                Delta::Call {
                    head: Some(head),
                    arguments,
                    ..
                } => calls.push((head.name.clone(), arguments.clone())),
                Delta::Call {
                    index, arguments, ..
                } => calls[*index].1.push_str(arguments),
            }
        }
        (reasoning, content, calls)
    }

    #[test]
    fn a_tool_call_types_its_arguments_by_schema() {
        let p = parser(true, Some(weather()));
        let msg = message(p, &format!("I need the weather.</think>\n{CALL}"));
        assert_eq!(msg["content"], Value::Null);
        assert_eq!(msg["reasoning_content"], "I need the weather.");
        let call = &msg["tool_calls"][0];
        assert!(call["id"].as_str().unwrap().starts_with("call_") && call.get("index").is_none());
        assert_eq!(call["type"], "function");
        // the string-typed zip stays a string though it parses as a number
        assert_eq!(
            call["function"],
            json!({"name": "get_weather", "arguments": r#"{"city": "Paris", "days": 3, "zip": "75001"}"#})
        );
    }

    #[test]
    fn streaming_in_any_chunks_matches_the_whole_reply() {
        for text in [
            format!("Thinking it over.</think>\n\nHere: {CALL}\n{CALL}"),
            "Plan</think>The answer is <b>bold</b> and a < b.".to_string(),
            "No close tag, the reply was cut off mid reasoning <thi".to_string(),
            format!("Call straight from reasoning {CALL}"),
            format!("x</think>{}", &CALL[..40]),
        ] {
            let mut whole = parser(true, Some(weather()));
            let want = (
                summary(&fed(&mut whole, &text, text.len())),
                whole.made_calls(),
            );
            for size in [1, 2, 3, 7] {
                let mut p = parser(true, Some(weather()));
                assert_eq!(
                    (summary(&fed(&mut p, &text, size)), p.made_calls()),
                    want,
                    "{text:?} in {size}s"
                );
            }
        }
    }

    #[test]
    fn a_call_streams_as_it_is_written() {
        let mut p = parser(true, Some(edit()));
        let first = p.feed("x</think><tool_call>edit<arg_key>pa");
        assert!(
            matches!(first.last(), Some(Delta::Call { index: 0, head: Some(head), arguments }) if head.name == "edit" && arguments.is_empty())
        );
        let args = |s: &str| {
            vec![Delta::Call {
                index: 0,
                head: None,
                arguments: s.to_string(),
            }]
        };
        assert_eq!(
            p.feed("th</arg_key><arg_value>src/"),
            args(r#"{"path": "src/"#)
        );
        // a string value goes out as it comes, all but what may start an end tag
        assert_eq!(p.feed("a <"), args("a "));
        assert_eq!(p.feed("b> </arg_v"), args("<b> "));
        assert_eq!(p.feed("alue></tool_call>"), args(r#""}"#));
    }

    #[test]
    fn a_single_call_parser_is_done_at_the_end_of_its_call() {
        let mut p = ReplyParser::new(true, Some(weather()), None, true);
        let mut deltas = p.feed(&format!("Plan</think>{}", &CALL[..CALL.len() - 3]));
        assert!(!p.done());
        deltas.extend(p.feed(&format!("{}\nMore text{CALL}", &CALL[CALL.len() - 3..])));
        assert!(p.done() && p.feed("x").is_empty() && p.finish().is_empty());
        let (_, content, calls) = summary(&deltas);
        assert_eq!(
            (content.as_str(), calls.len(), p.made_calls()),
            ("", 1, true)
        );
    }

    #[test]
    fn calls_take_indices_and_a_call_may_end_the_reasoning() {
        let mut p = parser(true, Some(weather()));
        let deltas = fed(&mut p, &format!("Plan</think>\n{CALL}\n{CALL}\n"), 5);
        let indices: Vec<usize> = deltas
            .iter()
            .filter_map(|d| match d {
                Delta::Call {
                    index,
                    head: Some(_),
                    ..
                } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(
            (indices, summary(&deltas).1.as_str(), p.made_calls()),
            (vec![0, 1], "", true)
        );
        let mut p = parser(true, Some(weather()));
        let (reasoning, content, calls) = summary(&fed(&mut p, &format!("Look it up {CALL}"), 4));
        assert_eq!(
            (reasoning.as_str(), content.as_str(), calls.len()),
            ("Look it up ", "", 1)
        );
    }

    #[test]
    fn a_call_stays_text_unless_an_offered_function_is_named_in_full() {
        let unoffered =
            "ok</think><tool_call>rm_rf<arg_key>path</arg_key><arg_value>/</arg_value></tool_call>";
        let mut p = parser(true, Some(weather()));
        let (_, content, calls) = summary(&fed(&mut p, unoffered, 3));
        assert!(calls.is_empty() && content.starts_with("<tool_call>rm_rf") && !p.made_calls());
        let mut p = parser(true, None);
        assert_eq!(
            summary(&fed(&mut p, &format!("ok</think>{CALL}"), 3)).1,
            CALL,
            "without tools a call tag is text"
        );
        let mut p = parser(true, Some(weather()));
        let (_, content, calls) = summary(&fed(&mut p, &format!("ok</think>{}", &CALL[..20]), 6));
        assert!(
            calls.is_empty() && content == CALL[..20] && !p.made_calls(),
            "cut off before its name ends"
        );
    }

    #[test]
    fn a_call_cut_off_after_its_name_stays_a_call_as_far_as_it_got() {
        for (cut, arguments) in [
            ("<tool_call>get_weather<arg_key>".len(), ""),
            ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par".len(), r#"{"city": "Par"#),
            ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3".len(), r#"{"city": "Paris", "days": "#),
        ] {
            for size in [1, 4, cut] {
                let mut p = parser(true, Some(weather()));
                let (_, content, calls) = summary(&fed(&mut p, &format!("ok</think>{}", &CALL[..cut]), size));
                assert_eq!((content.as_str(), calls), ("", vec![("get_weather".to_string(), arguments.to_string())]));
                // not tool_calls: the arguments are not whole
                assert!(!p.made_calls());
            }
        }
    }

    #[test]
    fn a_whole_reply_splits_into_reasoning_and_content() {
        for (thinking, text, want) in [
            (
                true,
                "Two plus two.</think>\n\nIt is 4.",
                json!({"role": "assistant", "content": "It is 4.", "reasoning_content": "Two plus two."}),
            ),
            (
                false,
                "Hello </think> there",
                json!({"role": "assistant", "content": "Hello </think> there"}),
            ),
            // a stray <think> in the reasoning and </think> in the answer drop
            (
                true,
                "<think>hmm</think>A</think>B",
                json!({"role": "assistant", "content": "AB", "reasoning_content": "hmm"}),
            ),
            (
                true,
                "r</think>A<think>more</think>B",
                json!({"role": "assistant", "content": "AB", "reasoning_content": "rmore"}),
            ),
            (
                true,
                "ok</think>Let me check. <tool_call>",
                json!({"role": "assistant", "content": "Let me check. <tool_call>", "reasoning_content": "ok"}),
            ),
        ] {
            assert_eq!(
                message(parser(thinking, Some(weather())), text),
                want,
                "{text:?}"
            );
        }
    }

    #[test]
    fn arguments_take_the_schema_type() {
        for (schema, raw, want) in [
            (
                json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
                "94110",
                json!("94110"),
            ),
            (json!({"type": ["string", "null"]}), "true", json!("true")),
            (json!({"enum": ["1", "2"]}), "1", json!("1")),
            (
                json!({"anyOf": [{"type": "integer"}, {"type": "string"}]}),
                "5",
                json!(5),
            ),
            (json!({"type": "number"}), "3", json!(3)),
            (json!({"type": "integer"}), "3.5", json!("3.5")), // off the schema: left as the model wrote it
            (json!({"type": "integer"}), "3.0", json!(3)),
            (
                json!({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
                "2e1",
                json!(20),
            ),
            (json!({"type": "number"}), "NaN", json!("NaN")), // not JSON: arguments must stay parseable
            (json!({}), "Infinity", json!("Infinity")),
            (json!({"type": "array"}), r#"["a", 1]"#, json!(["a", 1])),
            (json!({"anyOf": {"type": "string"}}), "3", json!(3)), // a malformed schema admits anything
            (json!({"enum": 5}), "3", json!(3)),
            (json!({}), "1e400", json!("1e400")),
            (json!({}), r#"{"x": 1e400}"#, json!(r#"{"x": 1e400}"#)),
        ] {
            let t = tools(
                json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": schema}}}}]),
            );
            let msg = message(parser(true, Some(t)), &format!("x</think><tool_call>f<arg_key>a</arg_key><arg_value>{raw}</arg_value></tool_call>"));
            let args: Value = serde_json::from_str(
                msg["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(args, json!({"a": want}), "{raw}");
        }
    }

    #[test]
    fn a_partial_tag_is_held_until_it_resolves() {
        let mut p = parser(true, Some(weather()));
        assert_eq!(p.feed("Plan</thi"), vec![Delta::Reasoning("Plan".into())]);
        assert_eq!(p.feed("nk>Done <tool"), vec![Delta::Content("Done".into())]);
        assert_eq!(
            p.feed("box> here"),
            vec![Delta::Content(" <toolbox> here".into())]
        );
        assert!(p.finish().is_empty());
    }

    #[test]
    fn a_forced_call_is_to_its_function_alone() {
        // a forced get_weather's call cannot turn into one to get_weather_hourly
        let p = ReplyParser::new(false, Some(weather()), Some("get_weather".into()), false);
        let msg = message(
            p,
            "_hourly<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        );
        assert!(
            msg.get("tool_calls").is_none()
                && msg["content"]
                    .as_str()
                    .unwrap()
                    .starts_with("<tool_call>get_weather_hourly")
        );
        let p = ReplyParser::new(false, Some(weather()), Some(String::new()), false);
        let msg = message(
            p,
            "rm_rf<arg_key>path</arg_key><arg_value>/</arg_value></tool_call>",
        );
        assert!(msg["content"]
            .as_str()
            .unwrap()
            .starts_with("<tool_call>rm_rf<arg_key>"));
        let p = ReplyParser::new(false, Some(weather()), Some("get_weather".into()), false);
        let msg = message(
            p,
            "<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        );
        assert_eq!(
            msg["tool_calls"][0]["function"],
            json!({"name": "get_weather", "arguments": r#"{"city": "Paris"}"#})
        );
        assert_eq!(msg["content"], Value::Null);
    }

    #[test]
    fn a_long_call_is_read_once() {
        // linear in the call's length: a feed searches its own text for the end tag
        let write = tools(
            json!([{"type": "function", "function": {"name": "write", "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}}]),
        );
        let body = "x = 1  # filler line\n".repeat(5000);
        let text = format!("plan</think><tool_call>write<arg_key>path</arg_key><arg_value>a.py</arg_value><arg_key>content</arg_key><arg_value>{body}</arg_value></tool_call>");
        let mut p = parser(true, Some(write));
        let (mut deltas, mut longest) = (Vec::new(), 0);
        for piece in text.as_bytes().chunks(4) {
            deltas.extend(p.feed(std::str::from_utf8(piece).unwrap()));
            longest = longest.max(p.buf.len());
        }
        deltas.extend(p.finish());
        assert!(longest < CALL_END.len() + 4, "held {longest} bytes");
        let args: Value = serde_json::from_str(&summary(&deltas).2[0].1).unwrap();
        assert_eq!(args, json!({"path": "a.py", "content": body}));
    }

    /// A JSON value, or the text that is not one (a call cut off mid-arguments).
    fn parsed(arguments: &str) -> Value {
        serde_json::from_str(arguments).unwrap_or_else(|_| Value::from(arguments))
    }

    /// What `glm_chat.GlmReplyParser` (#813) made of whole replies, recorded by
    /// `gen_glm_reply_reference.py`: structured replies under each setting, and
    /// randomized ones, one per outcome. Every chunking here must come to the
    /// same reasoning, answer, calls and finish; arguments compare as JSON, as
    /// a float's digits may be written differently.
    #[test]
    fn replies_split_as_the_python_parser_split_them_in_every_recorded_case() {
        let reference: Value =
            serde_json::from_str(include_str!("testdata/glm_reply_reference.json")).unwrap();
        let cases = reference["cases"].as_array().unwrap();
        for (n, case) in cases.iter().enumerate() {
            let text = case["text"].as_str().unwrap();
            let want_calls: Vec<(String, Value)> = case["calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    (
                        c[0].as_str().unwrap().to_string(),
                        parsed(c[1].as_str().unwrap()),
                    )
                })
                .collect();
            let want = (
                case["reasoning"].as_str().unwrap().to_string(),
                case["content"].as_str().unwrap().to_string(),
                want_calls,
                case["finish_reason"] == "tool_calls",
                case["done"].as_bool().unwrap(),
            );
            for size in [text.chars().count(), 1, 3, 7] {
                let params = case["tools"]
                    .as_str()
                    .map(|set| tools(reference["tool_sets"][set].clone()));
                let forced = case["forced_call"].as_str().map(str::to_string);
                let single_call = case["single_call"].as_bool().unwrap();
                let mut p = ReplyParser::new(
                    case["thinking"].as_bool().unwrap(),
                    params,
                    forced,
                    single_call,
                );
                let (reasoning, content, calls) = summary(&fed(&mut p, text, size));
                let calls = calls
                    .into_iter()
                    .map(|(name, arguments)| (name, parsed(&arguments)))
                    .collect();
                let got = (reasoning, content, calls, p.made_calls(), p.done());
                assert_eq!(got, want, "case {n} in {size}s: {text:?}");
            }
        }
        assert!(cases.len() > 100);
    }
}
