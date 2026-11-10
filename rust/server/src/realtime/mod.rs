//! `/v1/realtime`: OpenAI's Realtime API over a WebSocket, a port of
//! `openai/serving_realtime.py` (see its docstring for the events). A
//! full-duplex model serves a conversation ([`conversation`]); the
//! transcription intent (`?intent=transcription`), here, streams speech-to-text
//! with partial results. Every `chunk_seconds` of new audio, the audio heard so far is
//! transcribed again as one engine request whose assistant turn is prefilled
//! with the previous hypothesis minus its last `unfixed_tokens` tokens (the
//! Qwen3-ASR SDK's streaming algorithm); the tail can still change, the head
//! is fixed. Turn detection (`server_vad`, `semantic_vad`) is [`vad`]. A
//! model that hears audio as it comes transcribes live instead ([`live`]).

mod audio;
mod conversation;
mod live;
mod resample;
mod vad;

use std::sync::OnceLock;

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use axum::extract::{Query, State};
use axum::response::Response;
use serde_json::{json, Map, Value};

use futures::StreamExt;

use crate::adapters::{OpenAIAdapter, SubmitArgs};
use crate::serving::{collect, rid, try_result_stream, AppState, Out, OutStream};
use crate::transcription::TranscriptionRequest;
use audio::AudioFormat;
use resample::StreamResampler;
use vad::{Op, ServerVad, SemanticVad, TurnDetection, TurnSegmenter, SAMPLE_RATE};

const DEFAULT_CHUNK_SECONDS: f64 = 2.0;
const DEFAULT_UNFIXED_CHUNKS: i64 = 2;
const DEFAULT_UNFIXED_TOKENS: i64 = 5;
const MIN_CHUNK_SECONDS: f64 = 0.5;
const MAX_ITEM_SECONDS: f64 = 300.0;
const MAX_VAD_MS: f64 = 10_000.0;
const EXCERPT_CHARS: usize = 100;
const MAX_LANGUAGE_CHARS: usize = 32;
const MAX_ERROR_CHARS: usize = 300;
const MAX_FAILURES_PER_ITEM: usize = 8;

// ── reprlib, as the Python route quotes a client's value ─────────────────────

/// `reprlib.Repr`'s cut of a long text: its head and tail around `...`.
fn cut(s: String, limit: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= limit {
        return s;
    }
    let i = (limit - 3) / 2;
    let j = limit - 3 - i;
    let head: String = chars[..i].iter().collect();
    let tail: String = chars[chars.len() - j..].iter().collect();
    format!("{head}...{tail}")
}

/// `_EXCERPT_REPR.repr1`: reprlib with at most two levels, four items of a
/// list or a dict (in order), and 100 characters of a string or number.
fn repr1(v: &Value, level: i32) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) if n.is_f64() => cut(crate::adapters::py_float_repr(n.as_f64().unwrap_or(f64::NAN)), EXCERPT_CHARS),
        Value::Number(n) => cut(n.to_string(), EXCERPT_CHARS),
        Value::String(s) => {
            let chars: Vec<char> = s.chars().collect();
            let head: String = chars.iter().take(EXCERPT_CHARS).collect();
            let r = crate::media::py_repr(&head);
            if r.chars().count() <= EXCERPT_CHARS {
                return r;
            }
            let i = (EXCERPT_CHARS - 3) / 2;
            let j = EXCERPT_CHARS - 3 - i;
            let kept: String = chars[..i].iter().chain(&chars[chars.len().saturating_sub(j)..]).collect();
            let r: Vec<char> = crate::media::py_repr(&kept).chars().collect();
            format!("{}...{}", r[..i].iter().collect::<String>(), r[r.len() - j..].iter().collect::<String>())
        }
        Value::Array(items) => {
            if level <= 0 && !items.is_empty() {
                return "[...]".into();
            }
            let mut pieces: Vec<String> = items.iter().take(4).map(|x| repr1(x, level - 1)).collect();
            if items.len() > 4 {
                pieces.push("...".into());
            }
            format!("[{}]", pieces.join(", "))
        }
        Value::Object(m) => {
            if m.is_empty() {
                return "{}".into();
            }
            if level <= 0 {
                return "{...}".into();
            }
            let mut pieces: Vec<String> = m
                .iter()
                .take(4)
                .map(|(k, x)| format!("{}: {}", repr1(&Value::String(k.clone()), level - 1), repr1(x, level - 1)))
                .collect();
            if m.len() > 4 {
                pieces.push("...".into());
            }
            format!("{{{}}}", pieces.join(", "))
        }
    }
}

/// `_excerpt`: a client value's repr, at most 100 characters long.
fn excerpt(v: &Value) -> String {
    let text = repr1(v, 2);
    if text.chars().count() <= EXCERPT_CHARS {
        return text;
    }
    format!("{}...", text.chars().take(EXCERPT_CHARS - 3).collect::<String>())
}

fn clip(message: &str) -> String {
    if message.chars().count() <= MAX_ERROR_CHARS {
        return message.to_string();
    }
    format!("{}...", message.chars().take(MAX_ERROR_CHARS - 3).collect::<String>())
}

/// `{:g}` for the bounds an error names.
/// `serving_realtime._scrub_paths`: `message` with the upload directory as
/// `<upload>`, then each quoted absolute path (`_QUOTED_PATH`, whose
/// directories may hold spaces) and each other one (`_ABS_PATH`) cut to its
/// base name. Not `client_errors.scrub_paths`, whose rules differ. Rust's
/// `regex` has neither the lookbehind nor the backreference, so this scans.
fn scrub_paths(message: &str, upload_dir: &std::path::Path) -> String {
    let message = message.replace(&*upload_dir.to_string_lossy(), "<upload>");
    abs_paths(&quoted_paths(&message))
}

/// `(['"])/(?:[^'"\n/]*/)+([^'"\n/]+)\1` -> `\1\2\1`: the text up to the next
/// quote or newline must close with the same quote, start with `/`, hold a
/// second `/`, and end in a name.
fn quoted_paths(s: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < c.len() {
        let q = c[i];
        if (q == '\'' || q == '"') && c.get(i + 1) == Some(&'/') {
            let end = (i + 1..c.len()).find(|&k| matches!(c[k], '\'' | '"' | '\n'));
            if let Some(end) = end.filter(|&k| c[k] == q) {
                let inner = &c[i + 1..end];
                let last = inner.iter().rposition(|&ch| ch == '/').unwrap();
                if last > 0 && last + 1 < inner.len() {
                    out.push(q);
                    out.extend(&inner[last + 1..]);
                    out.push(q);
                    i = end + 1;
                    continue;
                }
            }
        }
        out.push(q);
        i += 1;
    }
    out
}

/// `(?<![\w./<>])/(?:[^\s/'"]+/)+([^\s/'"]+)` -> `\1`, as Python's backtracking
/// matches it: the most whole directories it can, then a name.
fn abs_paths(s: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let name_char = |ch: char| !(ch.is_whitespace() || matches!(ch, '/' | '\'' | '"'));
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < c.len() {
        let free = i == 0 || !(c[i - 1].is_alphanumeric() || matches!(c[i - 1], '_' | '.' | '/' | '<' | '>'));
        if c[i] == '/' && free {
            // the components after the slash, up to whitespace or a quote
            let mut comps: Vec<(usize, usize)> = Vec::new();
            let mut k = i + 1;
            loop {
                let start = k;
                while k < c.len() && name_char(c[k]) {
                    k += 1;
                }
                comps.push((start, k));
                if k < c.len() && c[k] == '/' {
                    k += 1;
                } else {
                    break;
                }
            }
            // directories: nonempty components each followed by a slash
            let mut dirs = 0;
            while dirs + 1 < comps.len() && comps[dirs].1 > comps[dirs].0 {
                dirs += 1;
            }
            let name = if dirs >= 1 && comps[dirs].1 > comps[dirs].0 {
                Some(comps[dirs])
            } else if dirs >= 2 {
                Some(comps[dirs - 1]) // the last directory's slash starts no name: it is the name
            } else {
                None
            };
            if let Some((start, end)) = name {
                out.extend(&c[start..end]);
                i = end;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

fn g(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e16 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

// ── events ───────────────────────────────────────────────────────────────────

fn event(kind: &str, fields: Value) -> Value {
    let mut out = Map::new();
    out.insert("event_id".into(), Value::from(format!("event_{}", &uuid::Uuid::new_v4().simple().to_string()[..12])));
    out.insert("type".into(), Value::from(kind));
    if let Value::Object(f) = fields {
        out.extend(f);
    }
    Value::Object(out)
}

fn error_event(message: &str, kind: &str, code: Option<&str>) -> Value {
    let mut error = json!({"type": kind, "message": message});
    if let Some(code) = code {
        error["code"] = Value::from(code);
    }
    event("error", json!({"error": error}))
}

fn invalid(message: &str) -> Value {
    error_event(message, "invalid_request_error", None)
}

/// The OpenAI error `type` for a status (`_util.error_type`).
fn error_type(status: u16) -> &'static str {
    if (400..500).contains(&status) { "invalid_request_error" } else { "server_error" }
}

// ── a call: one engine request whose input stays open ────────────────────────

/// `_call_error` for a call's failure on its output: the message cut and
/// without server paths.
fn call_error(st: &AppState, status: u16, message: &str) -> Value {
    error_event(&clip(&scrub_paths(message, &st.upload_dir)), error_type(status), None)
}

/// `_open_call`: open a realtime call's engine request, its input left open:
/// its submission and output, or the `error` event for the server's refusal
/// (at capacity, say), as `_call_error` words it.
async fn open_call(st: &AppState, args: &SubmitArgs, session_id: &str) -> Result<(u64, OutStream), Value> {
    let rid = rid(&format!("rt-{session_id}"));
    let (status, message) = match crate::serving::live_result_stream(st, args, &rid) {
        Ok((mut stream, Some(sub))) => {
            // The submission is the bridge's to accept: it answers `taking`
            // only once it has, and a refusal has reached the stream by then.
            let bridge = st.bridge.clone().expect("a submission went over the bridge");
            let taking = bridge.live(sub, "taking", Map::new(), crate::BRIDGE_PROBE_TIMEOUT * 5).await;
            if taking.as_ref().and_then(|m| m.get("taking")).and_then(Value::as_bool) == Some(true) {
                return Ok((sub, stream));
            }
            match tokio::time::timeout(crate::BRIDGE_PROBE_TIMEOUT, stream.next()).await {
                Ok(Some(Out::Error { status, message, .. })) => (status, message),
                _ => (500, "the server could not open the call".to_string()),
            }
        }
        Ok((_, None)) => (500, "this server has no backend to open the call on".to_string()),
        Err(refused) => refused,
    };
    // a ValueError's text is cut and loses its paths; an HTTPException's detail is sent as is
    let message = if status == 400 { clip(&scrub_paths(&message, &st.upload_dir)) } else { message };
    Err(error_event(&message, error_type(status), None))
}

/// `stream_input(samples=...)`: append `audio`, at the stream's rate, to the
/// call `sub`'s `ClientAudio` stream `stream`; whether it was taken. Not when
/// the call has ended (its output says why) or the append passed the call's
/// input budget, which ends the call with the refusal (`reject_input`; its
/// output says so too). An append that failed inside the server is the
/// `error` event returned.
async fn stream_audio(st: &AppState, sub: u64, stream: &str, audio: &[f32], session_id: &str) -> Result<bool, Value> {
    let Some(bridge) = st.bridge.clone() else { return Ok(false) };
    let timeout = crate::BRIDGE_PROBE_TIMEOUT * 5;
    let reply = bridge.live_audio(sub, stream, audio, timeout).await;
    match reply.as_ref().and_then(Value::as_object) {
        Some(meta) if meta.get("error").is_none() => Ok(meta.get("accepted").and_then(Value::as_bool).unwrap_or(false)),
        Some(meta) if meta.get("status").and_then(Value::as_u64) == Some(400) => {
            // past the call's input budget: the call ends with it
            let error = meta.get("error").and_then(Value::as_str).unwrap_or_default();
            let mut op = Map::new();
            op.insert("error".into(), Value::from(format!("append rejected: {error}")));
            op.insert("status".into(), Value::from(400));
            bridge.live(sub, "reject", op, timeout).await;
            Ok(false)
        }
        meta => Err(failed_in_server("append", meta, session_id)),
    }
}

/// `_failed_in_server`: the `error` event for an `append` or `commit` whose
/// bridge reply `meta` was an error (logged on the API server under its
/// error id) or never came.
fn failed_in_server(what: &str, meta: Option<&Map<String, Value>>, session_id: &str) -> Value {
    let error_id = meta.and_then(|m| m.get("error_id")).and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| {
        let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
        eprintln!("realtime {what} failed (error id {id}, session {session_id}): the API server did not answer");
        id
    });
    error_event(&format!("the {what} failed (error id {error_id})"), "server_error", None)
}

// ── settings ─────────────────────────────────────────────────────────────────

/// `_append_error`: the `error` for an append of more than `MAX_ITEM_SECONDS`
/// of `rate` audio. One append is capped at this on both intents, since it is
/// resampled before any other cap applies.
fn append_error(samples: usize, rate: usize) -> Option<Value> {
    (samples as f64 > MAX_ITEM_SECONDS * rate as f64).then(|| {
        invalid(&format!("an append holds at most {} s of audio; send it in shorter appends", g(MAX_ITEM_SECONDS)))
    })
}

fn object<'a>(value: Option<&'a Value>, name: &str) -> Result<Option<&'a Map<String, Value>>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(m)) => Ok(Some(m)),
        Some(_) => Err(format!("{name} must be an object")),
    }
}

/// `_input_format`: the transcription intent's `input_audio_format` (`current`
/// when absent): a beta name, or a GA format object.
fn input_format(fmt: Option<&Value>, current: AudioFormat) -> Result<AudioFormat, String> {
    match fmt {
        None | Some(Value::Null) => Ok(current),
        Some(Value::String(name)) if AudioFormat::from_beta_name(name).is_some() => Ok(AudioFormat::from_beta_name(name).unwrap()),
        Some(Value::Object(m)) => audio::audio_format(Some(m), "input_audio_format"),
        Some(other) => Err(format!("input_audio_format must be pcm16, g711_ulaw, g711_alaw or an object; got {}", excerpt(other))),
    }
}

/// `_number`: `knobs[name]` (or `default`) a finite number in `[lo, hi]`, integral
/// when `integer`; a numeric string is read as its number, and with `clamp_low`
/// a value below `lo` is raised to it.
#[allow(clippy::too_many_arguments)]
fn number(knobs: Option<&Map<String, Value>>, name: &str, default: f64, lo: f64, hi: f64, integer: bool, clamp_low: bool, scope: &str) -> Result<f64, String> {
    let raw = knobs.and_then(|k| k.get(name)).cloned().unwrap_or_else(|| {
        if integer { Value::from(default as i64) } else { crate::adapters::json_num(default) }
    });
    let value = match &raw {
        Value::String(s) => {
            let t = crate::adapters::py_strip(s);
            static INT: OnceLock<regex::Regex> = OnceLock::new();
            let int = INT.get_or_init(|| regex::Regex::new(r"^[+-]?[0-9](?:_?[0-9])*$").unwrap());
            if int.is_match(t) {
                t.replace('_', "").parse::<f64>().ok().map(Value::from).unwrap_or(raw.clone())
            } else {
                crate::generation_kwargs::py_float(s).map(crate::adapters::json_num).unwrap_or(raw.clone())
            }
        }
        _ => raw.clone(),
    };
    let kind = if integer { "an integer" } else { "a number" };
    let mut x = value.as_f64().filter(|_| value.is_number());
    if let Some(v) = x {
        if !v.is_finite() || (integer && v.fract() != 0.0) {
            x = None;
        }
    }
    if let (Some(v), true) = (x, clamp_low) {
        if v < lo {
            x = Some(lo);
        }
    }
    match x {
        Some(v) if lo <= v && v <= hi => Ok(v),
        _ => {
            let bound = if hi == f64::INFINITY { format!(">= {}", g(lo)) } else { format!("between {} and {}", g(lo), g(hi)) };
            Err(format!("{scope}.{name} must be {kind} {bound}; got {}", excerpt(&value)))
        }
    }
}

/// `_turn_detection`: null (the client commits), `server_vad` or `semantic_vad`.
fn turn_detection(value: &Value) -> Result<Option<TurnDetection>, String> {
    let Some(td) = object(Some(value), "turn_detection")? else { return Ok(None) };
    let kind = td.get("type");
    let keys: &[&str] = match kind.and_then(Value::as_str) {
        Some("server_vad") => &["type", "threshold", "prefix_padding_ms", "silence_duration_ms", "create_response", "interrupt_response", "idle_timeout_ms"],
        Some("semantic_vad") => &["type", "eagerness", "create_response", "interrupt_response"],
        _ => {
            let what = match kind {
                Some(Value::String(_)) => format!("type {}", excerpt(kind.unwrap())),
                _ => excerpt(value),
            };
            return Err(format!(
                "turn_detection {what} is not supported; use server_vad, semantic_vad, or null to commit with input_audio_buffer.commit"
            ));
        }
    };
    if let Some(unknown) = td.keys().find(|k| !keys.contains(&k.as_str())) {
        return Err(format!("turn_detection.{} is not supported", excerpt(&Value::from(unknown.as_str()))));
    }
    if td.get("idle_timeout_ms").is_some_and(|v| !v.is_null()) {
        return Err("turn_detection.idle_timeout_ms is not supported".into());
    }
    let mut flags = [true, true];
    for (i, name) in ["create_response", "interrupt_response"].iter().enumerate() {
        match td.get(*name) {
            None => {}
            Some(Value::Bool(b)) => flags[i] = *b,
            Some(other) => return Err(format!("turn_detection.{name} must be a boolean; got {}", excerpt(other))),
        }
    }
    if kind.and_then(Value::as_str) == Some("semantic_vad") {
        let eagerness = td.get("eagerness").cloned().unwrap_or(Value::from("auto"));
        return match eagerness.as_str().filter(|e| vad::eagerness(e).is_some()) {
            Some(e) => Ok(Some(TurnDetection::Semantic(SemanticVad { eagerness: e.to_string(), create_response: flags[0], interrupt_response: flags[1] }))),
            None => Err(format!("turn_detection.eagerness must be one of low, medium, high, auto; got {}", excerpt(&eagerness))),
        };
    }
    let d = ServerVad::default();
    Ok(Some(TurnDetection::Server(ServerVad {
        threshold: number(Some(td), "threshold", d.threshold, 0.0, 1.0, false, false, "turn_detection")?,
        prefix_padding_ms: number(Some(td), "prefix_padding_ms", d.prefix_padding_ms as f64, 0.0, MAX_VAD_MS, true, false, "turn_detection")? as i64,
        silence_duration_ms: number(Some(td), "silence_duration_ms", d.silence_duration_ms as f64, 0.0, MAX_VAD_MS, true, false, "turn_detection")? as i64,
        create_response: flags[0],
        interrupt_response: flags[1],
    })))
}

// ── the tokenizer, for the stable prefix ─────────────────────────────────────

fn tokenizer() -> Option<&'static tokenizers::Tokenizer> {
    static TOK: OnceLock<Option<tokenizers::Tokenizer>> = OnceLock::new();
    TOK.get_or_init(|| {
        let facts: Value = std::env::var("MSTAR_REALTIME").ok().and_then(|s| serde_json::from_str(&s).ok())?;
        let path = facts.get("tokenizer")?.as_str()?;
        match tokenizers::Tokenizer::from_file(path) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("WARNING: realtime: cannot load the model's tokenizer from {path}: {e}; partials hold nothing back");
                None
            }
        }
    })
    .as_ref()
}

/// `stable_prefix`: `raw` minus its last `unfixed` tokens, backed off further
/// when the cut lands inside a multi-byte character.
fn stable_prefix(raw: &str, tok: Option<&tokenizers::Tokenizer>, unfixed: i64) -> String {
    let Some(tok) = tok.filter(|_| !raw.is_empty()) else { return String::new() };
    let Ok(enc) = tok.encode(raw, false) else { return String::new() };
    let ids = enc.get_ids();
    let mut k = unfixed.max(0) as usize;
    loop {
        let end = ids.len().saturating_sub(k);
        let prefix = if end > 0 { tok.decode(&ids[..end], false).unwrap_or_default() } else { String::new() };
        if !prefix.contains('\u{fffd}') || end == 0 {
            return prefix;
        }
        k += 1;
    }
}

// ── the session ──────────────────────────────────────────────────────────────

/// The error ids handed out for one item's failures, keyed by type and message.
#[derive(Default)]
struct ItemFailures {
    item_id: String,
    ids: Vec<((String, String), String)>,
}

/// One WebSocket's transcription state (the SDK's `ASRStreamingState`).
struct Session {
    id: String,
    model_name: String,
    request: TranscriptionRequest,
    chunk_seconds: f64,
    unfixed_chunks: i64,
    unfixed_tokens: i64,
    max_item_seconds: f64,
    pending: Vec<f32>,
    heard: Vec<f32>,
    chunk_id: i64,
    raw_hypothesis: String,
    emitted: String,
    item_id: String,
    failures: ItemFailures,
    odd_byte: Option<u8>,
    input_format: AudioFormat,
    resampler: Option<StreamResampler>,
    short: usize,
    turn_detection: Option<TurnDetection>,
    segmenter: Option<TurnSegmenter>,
    stream_samples: i64,
    clock_base: i64,
    clock_in: i64,
    /// samples already on a live model's stream ([`live`]): an item's length
    /// without holding its audio
    streamed_samples: usize,
}

fn item_id() -> String {
    format!("item_{}", &uuid::Uuid::new_v4().simple().to_string()[..12])
}

impl Session {
    fn new(st: &AppState, adapter: &dyn OpenAIAdapter) -> Session {
        Session {
            id: uuid::Uuid::new_v4().simple().to_string()[..16].to_string(),
            model_name: st.model_name.clone(),
            request: TranscriptionRequest { model: Some(st.model_name.clone()), ..TranscriptionRequest::default() },
            chunk_seconds: DEFAULT_CHUNK_SECONDS,
            unfixed_chunks: DEFAULT_UNFIXED_CHUNKS,
            unfixed_tokens: DEFAULT_UNFIXED_TOKENS,
            max_item_seconds: adapter.transcription_policy().max_audio_seconds.unwrap_or(f64::INFINITY).min(MAX_ITEM_SECONDS),
            pending: Vec::new(),
            heard: Vec::new(),
            chunk_id: 0,
            raw_hypothesis: String::new(),
            emitted: String::new(),
            item_id: item_id(),
            failures: ItemFailures::default(),
            odd_byte: None,
            input_format: AudioFormat::pcm(SAMPLE_RATE),
            resampler: None,
            short: 0,
            turn_detection: None,
            segmenter: None,
            stream_samples: 0,
            clock_base: 0,
            clock_in: 0,
            streamed_samples: 0,
        }
    }

    fn chunk_samples(&self) -> usize {
        (self.chunk_seconds * SAMPLE_RATE as f64).round_ties_even() as usize
    }

    fn set_input_format(&mut self, fmt: AudioFormat) {
        self.clock_base = self.stream_samples;
        self.clock_in = 0;
        self.input_format = fmt;
        self.resampler = (fmt.rate != SAMPLE_RATE).then(|| StreamResampler::new(fmt.rate, SAMPLE_RATE));
    }

    fn resample(&mut self, audio: &[f32]) -> Vec<f32> {
        self.clock_in += audio.len() as i64;
        let out = match &mut self.resampler {
            None => audio.to_vec(),
            Some(r) => r.push(audio),
        };
        self.stream_samples += out.len() as i64;
        out
    }

    fn held(&self, extra: usize) -> usize {
        self.resampler.as_ref().map_or(extra, |r| r.held(extra))
    }

    fn reset_stream(&mut self) {
        let skipped = self.resampler.as_ref().map_or(0, |r| r.held(0));
        self.stream_samples += skipped as i64;
        if let Some(r) = &mut self.resampler {
            r.reset();
        }
        if let Some(seg) = &mut self.segmenter {
            seg.reset(skipped);
        }
        self.realign();
    }

    /// Put the timeline back on the input clock after the resampler starts over.
    fn realign(&mut self) {
        let exact = self.clock_base + self.clock_in * SAMPLE_RATE as i64 / self.input_format.rate as i64;
        if let Some(seg) = &mut self.segmenter {
            seg.shift(exact - self.stream_samples);
        }
        self.stream_samples = exact;
    }

    fn drain_resampler(&mut self) -> Vec<f32> {
        let out = self.resampler.as_mut().map(|r| r.flush()).unwrap_or_default();
        self.stream_samples += out.len() as i64;
        out
    }

    fn add(&mut self, audio: &[f32]) {
        self.pending.extend_from_slice(audio);
    }

    /// Samples in the item once `extra` more input samples arrive.
    fn item_samples(&self, extra: usize) -> usize {
        self.pending.len() + self.held(extra) + self.streamed_samples + self.heard.len()
    }

    /// Every full chunk of pending audio, or `None` until enough arrived.
    fn take_chunk(&mut self) -> Option<Vec<f32>> {
        let held = self.held(0);
        let cs = self.chunk_samples();
        let k = (self.pending.len() + held).saturating_sub(self.short) / cs;
        if k == 0 {
            return None;
        }
        let n = self.pending.len().min(k * cs + self.short);
        self.short = k * cs + self.short - n;
        let rest = self.pending.split_off(n);
        Some(std::mem::replace(&mut self.pending, rest))
    }

    fn flush(&mut self) -> Option<Vec<f32>> {
        (!self.pending.is_empty()).then(|| std::mem::take(&mut self.pending))
    }

    fn clear(&mut self) {
        self.pending.clear();
        self.heard.clear();
        self.short = 0;
        self.streamed_samples = 0;
        self.chunk_id = 0;
        self.raw_hypothesis.clear();
        self.emitted.clear();
        self.item_id = item_id();
    }

    fn view(&self) -> Value {
        let fmt = self.input_format.beta_name().map_or_else(|| self.input_format.view(), Value::from);
        json!({
            "id": self.id,
            "object": "realtime.transcription_session",
            "input_audio_format": fmt,
            "turn_detection": self.turn_detection.as_ref().map(TurnDetection::view),
            "input_audio_transcription": {
                "model": self.model_name,
                "language": self.request.language,
                "prompt": self.request.prompt.clone().unwrap_or_default(),
            },
            "mstar": {
                "chunk_seconds": crate::adapters::json_num(self.chunk_seconds),
                "unfixed_chunks": self.unfixed_chunks,
                "unfixed_tokens": self.unfixed_tokens,
                "max_item_seconds": crate::adapters::json_num(self.max_item_seconds),
                "sample_rate": self.input_format.rate,
            },
        })
    }
}

/// `write_wav`: float audio as 16-bit PCM, rounded half to even.
fn wav_bytes(audio: &[f32]) -> Vec<u8> {
    let pcm: Vec<u8> = audio::to_pcm16(audio).iter().flat_map(|s| s.to_le_bytes()).collect();
    crate::media::pcm16_to_wav_bytes(&pcm, SAMPLE_RATE as u32, 1)
}

/// A failed step: the engine's error and its status, or a refusal raised building it.
struct StepError {
    message: String,
    client: bool,
    type_name: &'static str,
}

/// How a transcription session turns an item's audio into text: re-decoded
/// from the start each chunk ([`Redecode`], `RealtimeTranscription`), or heard
/// as it comes on one call ([`live::Live`], `RealtimeLiveTranscription`). The
/// rest of the session (events, formats, caps, turn detection) is
/// [`Realtime`]'s.
trait ItemDecoder: Sized + Send + 'static {
    /// The pending audio to step on now, if any.
    fn take_chunk(&self, s: &mut Session) -> Option<Vec<f32>>;

    /// Hear `chunk` (the item's last when `final_`), sending what it adds to the text.
    async fn step(rt: &mut Realtime<Self>, chunk: Vec<f32>, final_: bool);

    /// End the item: the rest of its text, `committed`, and how it ended.
    async fn commit(rt: &mut Realtime<Self>);

    /// What a session update refuses before the rest of it applies.
    fn refuse_update(_rt: &Realtime<Self>, _session: &Map<String, Value>) -> Result<(), String> {
        Ok(())
    }

    /// `transcription_session.*`'s view of the session, adjusted.
    fn view(&self, _view: &mut Value) {}

    /// A clear drops what the decoder holds for the item.
    fn clear(&mut self) {}

    /// The item's output arriving on its own, to [`ItemDecoder::relay`].
    fn output(&mut self) -> Option<&mut OutStream> {
        None
    }

    /// One item of [`ItemDecoder::output`]; `None`: it ended.
    async fn relay(_rt: &mut Realtime<Self>, _item: Option<Out>) {}
}

/// Re-decoding: every `chunk_seconds`, the audio heard so far is transcribed
/// again, continuing from the stable prefix of the last hypothesis.
struct Redecode;

impl ItemDecoder for Redecode {
    fn take_chunk(&self, s: &mut Session) -> Option<Vec<f32>> {
        s.take_chunk()
    }

    /// Hear `chunk`, decode everything so far again from the stable prefix, and
    /// send the newly firm text and the whole hypothesis.
    async fn step(rt: &mut Realtime<Self>, chunk: Vec<f32>, final_: bool) {
        let chunk_len = chunk.len();
        rt.s.heard.extend(chunk);
        let tok = tokenizer();
        let prefix = if rt.s.chunk_id < rt.s.unfixed_chunks {
            String::new()
        } else {
            stable_prefix(&rt.s.raw_hypothesis, tok, rt.s.unfixed_tokens)
        };
        let (st, adapter, request, heard, id) = (rt.st.clone(), rt.adapter, rt.s.request.clone(), rt.s.heard.clone(), rt.s.id.clone());
        match run_step(st, adapter, request, heard, id, prefix).await {
            Ok(raw) => rt.s.raw_hypothesis = raw,
            Err(e) => {
                rt.step_failed(e).await;
                return;
            }
        }
        // a coalesced step counts every chunk it heard
        let chunks = (chunk_len as f64 / rt.s.chunk_samples() as f64).round_ties_even() as i64;
        rt.s.chunk_id += chunks.max(1);
        let text = rt.current_text();
        let hold = if final_ || tok.is_none() {
            0
        } else {
            let stable = stable_prefix(&rt.s.raw_hypothesis, tok, rt.s.unfixed_tokens);
            let stable_text = rt.adapter.parse_transcript(&stable, &rt.s.request).text;
            text.chars().count().saturating_sub(stable_text.chars().count())
        };
        rt.emit(&text, hold).await;
        let partial = event(
            "mstar.transcription.partial",
            json!({
                "item_id": rt.s.item_id, "text": text, "chunk_id": rt.s.chunk_id,
                "audio_seconds": rt.s.heard.len() as f64 / SAMPLE_RATE as f64, "created": crate::serving::now(),
            }),
        );
        rt.send(partial).await;
    }

    async fn commit(rt: &mut Realtime<Self>) {
        match rt.s.flush() {
            Some(tail) => Self::step(rt, tail, true).await,
            None => {
                // nothing left to hear: what was held back is final now
                let text = rt.current_text();
                rt.emit(&text, 0).await;
            }
        }
        let committed = event("input_audio_buffer.committed", json!({"item_id": rt.s.item_id}));
        rt.send(committed).await;
        let completed = event(
            "conversation.item.input_audio_transcription.completed",
            json!({"item_id": rt.s.item_id, "transcript": rt.current_text()}),
        );
        rt.send(completed).await;
        rt.s.clear();
    }
}

struct Realtime<D> {
    st: AppState,
    adapter: &'static dyn OpenAIAdapter,
    socket: WebSocket,
    s: Session,
    d: D,
}

impl<D: ItemDecoder> Realtime<D> {
    async fn send(&mut self, event: Value) -> bool {
        self.socket.send(Message::Text(event.to_string())).await.is_ok()
    }

    /// The session as `transcription_session.*` shows it.
    fn view(&self) -> Value {
        let mut view = self.s.view();
        self.d.view(&mut view);
        view
    }

    async fn run(mut self) {
        let created = event("transcription_session.created", json!({"session": self.view()}));
        if !self.send(created).await {
            return;
        }
        loop {
            let received = tokio::select! {
                received = self.socket.recv() => received,
                // the item's output, as it comes
                item = async { self.d.output().unwrap().next().await }, if self.d.output().is_some() => {
                    D::relay(&mut self, item).await;
                    continue;
                }
            };
            let message = match received {
                Some(Ok(m)) => m,
                Some(Err(e)) => {
                    if let Some(close) = crate::ws::read_failure_close(&e) {
                        let _ = self.socket.send(close).await;
                    }
                    return;
                }
                None => return,
            };
            let text = match message {
                Message::Text(t) => t,
                Message::Binary(_) => {
                    if !self.send(invalid("an event must be a JSON text message, not binary")).await {
                        return;
                    }
                    continue;
                }
                Message::Close(_) => return,
                _ => continue,
            };
            let event = match serde_json::from_str::<Value>(&text) {
                Ok(v) => v,
                Err(_) => {
                    self.send(invalid("not JSON")).await;
                    continue;
                }
            };
            let Value::Object(event) = event else {
                self.send(invalid("an event must be a JSON object")).await;
                continue;
            };
            self.handle(event).await;
        }
    }

    async fn handle(&mut self, ev: Map<String, Value>) {
        match ev.get("type").and_then(Value::as_str) {
            Some("transcription_session.update") => {
                let result = object(ev.get("session"), "session").and_then(|s| self.update(s.cloned().unwrap_or_default()));
                match result {
                    Err(e) => {
                        self.send(invalid(&e)).await;
                    }
                    Ok(()) => {
                        let updated = event("transcription_session.updated", json!({"session": self.view()}));
                        self.send(updated).await;
                    }
                }
            }
            Some("input_audio_buffer.append") => {
                let payload = ev.get("audio").cloned().unwrap_or(Value::from(""));
                let Ok((audio, odd)) = audio::decode_append(&payload, self.s.odd_byte, self.s.input_format.codec) else {
                    self.send(error_event("audio is not base64", "invalid_request_error", Some("invalid_audio"))).await;
                    return;
                };
                if let Some(error) = append_error(audio.len(), self.s.input_format.rate) {
                    self.send(error).await;
                    return;
                }
                if self.s.segmenter.is_none() && self.s.item_samples(audio.len()) as f64 > self.s.max_item_seconds * SAMPLE_RATE as f64 {
                    let message = format!(
                        "the item would pass {} s of audio; commit or clear the input audio buffer first",
                        g(self.s.max_item_seconds)
                    );
                    self.send(invalid(&message)).await;
                    return;
                }
                self.s.odd_byte = odd;
                if audio.is_empty() {
                    return;
                }
                // off the async workers: a long append is thousands of VAD frames
                let ops = tokio::task::block_in_place(|| {
                    let audio = self.s.resample(&audio);
                    match &mut self.s.segmenter {
                        None => vec![Op::Audio(audio)],
                        Some(seg) => seg.feed(&audio),
                    }
                });
                self.apply(ops).await;
            }
            Some("input_audio_buffer.commit") => {
                let tail = self.s.drain_resampler();
                if self.s.segmenter.is_none() {
                    self.s.realign();
                    self.apply(vec![Op::Audio(tail)]).await;
                } else {
                    let ops = tokio::task::block_in_place(|| {
                        let seg = self.s.segmenter.as_mut().unwrap();
                        let mut ops = seg.feed(&tail);
                        ops.extend(seg.end_turn());
                        ops
                    });
                    self.s.realign();
                    let ended = ops.last() == Some(&Op::Commit);
                    self.apply(ops).await;
                    if ended {
                        // the drained tail ended a turn: that commit was this one
                        return;
                    }
                }
                D::commit(self).await;
            }
            Some("input_audio_buffer.clear") => {
                self.d.clear();
                self.s.clear();
                self.s.odd_byte = None;
                self.s.reset_stream();
                self.send(event("input_audio_buffer.cleared", json!({}))).await;
            }
            _ => {
                let kind = ev.get("type").cloned().unwrap_or(Value::Null);
                self.send(invalid(&format!("unknown event {}", excerpt(&kind)))).await;
            }
        }
    }

    /// Carry out segmenter operations in order; audio steps once, after the last.
    async fn apply(&mut self, ops: Vec<Op>) {
        let mut rest = Vec::new();
        for op in ops {
            match op {
                Op::EndOfTurnFailed(e) => self.end_of_turn_failed(&e).await,
                other => rest.push(other),
            }
        }
        let n = rest.len();
        for (i, op) in rest.into_iter().enumerate() {
            match op {
                Op::Audio(audio) => {
                    self.s.add(&audio);
                    if i == n - 1 {
                        if let Some(chunk) = self.d.take_chunk(&mut self.s) {
                            D::step(self, chunk, false).await;
                        }
                    }
                }
                Op::Commit => D::commit(self).await,
                Op::SpeechStarted(ms) => {
                    let e = event("input_audio_buffer.speech_started", json!({"audio_start_ms": ms, "item_id": self.s.item_id}));
                    self.send(e).await;
                }
                Op::SpeechStopped(ms) => {
                    let e = event("input_audio_buffer.speech_stopped", json!({"audio_end_ms": ms, "item_id": self.s.item_id}));
                    self.send(e).await;
                }
                Op::EndOfTurnFailed(_) => {}
            }
        }
    }

    async fn end_of_turn_failed(&mut self, why: &str) {
        let id = &uuid::Uuid::new_v4().simple().to_string()[..12];
        eprintln!("realtime end-of-turn model failed (error id {id}, session {}): {why}; semantic_vad turns now end on silence alone", self.s.id);
        let silence = match &self.s.turn_detection {
            Some(TurnDetection::Semantic(s)) => vad::eagerness(&s.eagerness).map_or(4000, |e| e.2),
            _ => 4000,
        };
        let message = format!(
            "semantic_vad's end-of-turn model failed (error id {id}); this session's turns now end after {} s of silence; send transcription_session.update to switch to server_vad",
            g(silence as f64 / 1000.0)
        );
        self.send(error_event(&message, "server_error", Some("end_of_turn_failed"))).await;
    }

    /// Apply a session update, all or nothing.
    fn update(&mut self, session: Map<String, Value>) -> Result<(), String> {
        D::refuse_update(self, &session)?;
        let s = &self.s;
        let fmt = input_format(session.get("input_audio_format"), s.input_format)?;
        if fmt != s.input_format && s.item_samples(0) > 0 {
            return Err("commit or clear the input audio buffer before changing the input audio format".into());
        }
        if fmt != s.input_format && s.odd_byte.is_some() {
            // a commit keeps the half sample, so only clear helps
            return Err("half a sample is pending; clear the input audio buffer before changing the input audio format".into());
        }
        let td = match session.get("turn_detection") {
            Some(v) => turn_detection(v)?,
            None => s.turn_detection.clone(),
        };
        let mut end_of_turn = None;
        if td != s.turn_detection {
            let in_item = match &s.segmenter {
                Some(seg) => !s.pending.is_empty() || !s.heard.is_empty() || seg.speaking,
                None => s.item_samples(0) > 0,
            };
            if in_item {
                return Err("commit or clear the input audio buffer before changing turn_detection".into());
            }
            if td.is_some() && !vad::silero_available() {
                return Err("server_vad is unavailable here (Silero VAD is missing or not the expected model; see the server log); set turn_detection to null and commit with input_audio_buffer.commit".into());
            }
            if matches!(td, Some(TurnDetection::Semantic(_))) {
                let why = vad::smart_turn_problem().or_else(|| self.st.bridge.is_none().then(|| "no Smart Turn model is loaded".to_string()));
                if let Some(why) = why {
                    return Err(format!(
                        "semantic_vad is unavailable here ({why}; see the server log); use server_vad, or set turn_detection to null and commit with input_audio_buffer.commit"
                    ));
                }
                let bridge = self.st.bridge.clone().expect("checked above");
                let timeout = self.st.request_timeout;
                // the segmenter runs under block_in_place, where waiting on the bridge is allowed
                let ask: vad::EndOfTurn = std::sync::Arc::new(move |audio: &[f32]| {
                    tokio::runtime::Handle::current().block_on(bridge.end_of_turn(audio, timeout))
                });
                end_of_turn = Some(ask);
            }
        }
        let knobs = object(session.get("mstar"), "mstar")?;
        let chunk_seconds = number(knobs, "chunk_seconds", s.chunk_seconds, MIN_CHUNK_SECONDS, s.max_item_seconds, false, true, "mstar")?;
        let unfixed_chunks = number(knobs, "unfixed_chunks", s.unfixed_chunks as f64, 0.0, f64::INFINITY, true, false, "mstar")? as i64;
        let unfixed_tokens = number(knobs, "unfixed_tokens", s.unfixed_tokens as f64, 0.0, f64::INFINITY, true, false, "mstar")? as i64;
        let transcription = object(session.get("input_audio_transcription"), "input_audio_transcription")?;
        let mut request = s.request.clone();
        for name in ["language", "prompt"] {
            let Some(value) = transcription.and_then(|t| t.get(name)) else { continue };
            // revalidated from its dump, as `model_validate({**model_dump(), **fields})`
            // does: every field counts as sent, temperature included
            request.temperature_sent = true;
            let text = match value {
                Value::Null => None,
                Value::String(v) => Some(v.clone()),
                other => return Err(format!("input_audio_transcription.{name} must be a string; got {}", excerpt(other))),
            };
            if name == "language" {
                if let Some(l) = text.as_ref().filter(|l| l.chars().count() > MAX_LANGUAGE_CHARS) {
                    return Err(format!(
                        "input_audio_transcription.language must be at most {MAX_LANGUAGE_CHARS} characters; got {}",
                        excerpt(&Value::from(l.as_str()))
                    ));
                }
                request.language = text;
            } else {
                request.prompt = text;
            }
        }
        let s = &mut self.s;
        s.request = request;
        s.chunk_seconds = chunk_seconds;
        s.unfixed_chunks = unfixed_chunks;
        s.unfixed_tokens = unfixed_tokens;
        if fmt != s.input_format {
            s.set_input_format(fmt);
        }
        if td != s.turn_detection {
            let old = s.segmenter.take();
            s.turn_detection = td.clone();
            s.segmenter = td.as_ref().map(|td| {
                TurnSegmenter::new(td, (s.max_item_seconds * SAMPLE_RATE as f64) as usize, s.stream_samples, end_of_turn.clone())
            });
            if let Some(old) = old {
                // audio the old segmenter had not scored yet is not dropped
                match &mut s.segmenter {
                    Some(seg) => seg.take_over(old),
                    None => {
                        let unscored = old.unscored().to_vec();
                        s.add(&unscored);
                    }
                }
            }
        }
        Ok(())
    }

    fn current_text(&self) -> String {
        self.adapter.parse_transcript(&self.s.raw_hypothesis, &self.s.request).text
    }

    /// `_step_failed`: a request error keeps its message (cut, without server
    /// paths); anything else is a generic message with an error id. Each
    /// distinct failure is logged once per item.
    async fn step_failed(&mut self, e: StepError) {
        let s = &mut self.s;
        if s.failures.item_id != s.item_id {
            s.failures = ItemFailures { item_id: s.item_id.clone(), ids: Vec::new() };
        }
        let key = (e.type_name.to_string(), e.message.clone());
        let error_id = match s.failures.ids.iter().find(|(k, _)| *k == key) {
            Some((_, id)) => id.clone(),
            None => {
                let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
                if s.failures.ids.len() < MAX_FAILURES_PER_ITEM {
                    s.failures.ids.push((key, id.clone()));
                }
                eprintln!(
                    "realtime step failed (error id {id}, session {}, {}); repeats in this item are not logged: {}",
                    s.id, s.item_id, e.message
                );
                id
            }
        };
        if e.client {
            let message = clip(&scrub_paths(&e.message, &self.st.upload_dir));
            self.send(invalid(&message)).await;
        } else {
            let message = format!("realtime step failed (error id {error_id})");
            self.send(error_event(&message, "server_error", None)).await;
        }
    }

    /// Send the part of `text` past what was sent, minus its last `hold`
    /// characters; a revision that reaches into sent text is not resent.
    async fn emit(&mut self, text: &str, hold: usize) {
        let sent: Vec<char> = self.s.emitted.chars().collect();
        let now: Vec<char> = text.chars().collect();
        let stable = sent.iter().zip(&now).take_while(|(a, b)| a == b).count();
        if stable < sent.len() {
            self.s.emitted = now[..stable].iter().collect();
        }
        let done = self.s.emitted.chars().count();
        let delta = &now[done..];
        let firm: String = delta[..delta.len().saturating_sub(hold)].iter().collect();
        if !firm.is_empty() {
            self.s.emitted.push_str(&firm);
            let e = event("conversation.item.input_audio_transcription.delta", json!({"item_id": self.s.item_id, "delta": firm}));
            self.send(e).await;
        }
    }
}

/// `_run_step`: the model's raw output for the audio heard so far,
/// continuing from `prefix` (included).
async fn run_step(
    st: AppState,
    adapter: &'static dyn OpenAIAdapter,
    request: TranscriptionRequest,
    heard: Vec<f32>,
    session_id: String,
    prefix: String,
) -> Result<String, StepError> {
    let request_id = rid(&format!("rt-{}", session_id));
    let path = st.upload_dir.join(format!("{request_id}.wav"));
    let bytes = wav_bytes(&heard);
    let p = path.clone();
    crate::blocking(move || std::fs::write(&p, bytes)).await.map_err(|e| StepError { message: e.to_string(), client: false, type_name: "OSError" })?;
    let result = async {
        let mut args = adapter
            .realtime_step_request(&request, &path.to_string_lossy(), &prefix)
            .map_err(|e| StepError { message: e, client: true, type_name: "ValueError" })?;
        st.served
            .select(request.model.as_deref(), &mut args.model_kwargs)
            .map_err(|e| StepError { message: e, client: true, type_name: "ValueError" })?;
        let stream = try_result_stream(&st, &args, &request_id, false)
            .map_err(|(status, message)| StepError { message, client: (400..500).contains(&status), type_name: "StepError" })?;
        let chunks = collect(stream)
            .await
            .map_err(|(status, message)| StepError { message, client: (400..500).contains(&status), type_name: "StepError" })?;
        let data: Vec<u8> = chunks.iter().filter(|c| c.modality == "text").flat_map(|c| c.data.iter().copied()).collect();
        Ok(format!("{prefix}{}", String::from_utf8_lossy(&data)))
    }
    .await;
    let _ = std::fs::remove_file(&path);
    result
}

/// `GET /v1/realtime`: the transcription intent only.
pub async fn realtime(
    State(st): State<AppState>,
    Query(query): Query<std::collections::HashMap<String, String>>,
    ws: crate::ws::Upgrade,
) -> Response {
    let ws = match crate::ws::upgrade(ws) {
        Ok(ws) => ws,
        Err(not_found) => return not_found,
    };
    // As the Python router decides: a model that hears transcription as it
    // comes serves it, with the intent named or omitted; a model with a
    // ClientAudio stream converses when it can and no intent is named;
    // otherwise the transcription intent.
    let intent = query.get("intent").map(String::as_str);
    let audio = conversation::facts();
    let no_conversation = audio.and_then(conversation::Facts::problem);
    let converses = audio.is_some() && no_conversation.is_none();
    let limit = crate::ws::max_message_bytes();
    let ws = ws.max_message_size(limit).max_frame_size(limit);
    if let (Some(facts), None | Some("transcription")) = (audio.filter(|f| f.live_transcription()), intent) {
        if let Some(adapter) = crate::adapters::from_python(&st.python_adapter) {
            let s = Session::new(&st, adapter);
            return ws.on_upgrade(move |socket| live::Live::session(st, adapter, socket, s, facts).run());
        }
    }
    if converses && intent.is_none() {
        let facts = audio.unwrap();
        return ws.on_upgrade(move |socket| conversation::Conversation::new(st, socket, facts).run());
    }
    let refusal = match intent {
        None | Some("transcription") => match crate::resolve_message(&st, crate::Surface::RealtimeTranscription) {
            Ok(_) => None,
            Err(_) if converses => Some("this model serves /v1/realtime conversations; omit intent".to_string()),
            Err(_) if audio.is_some() => Some(format!("{}; use /generate/ws", no_conversation.unwrap())),
            Err(message) => Some(message),
        },
        Some(other) => Some(format!(
            "unsupported intent {}; {}",
            crate::media::py_repr(&other.chars().take(32).collect::<String>()),
            if converses { "omit intent for a conversation" } else { "use intent=transcription" }
        )),
    };
    if let Some(mut reason) = refusal {
        // a close frame's reason is at most 123 bytes (RFC 6455); a model's name is in some
        let mut end = reason.len().min(123);
        while !reason.is_char_boundary(end) {
            end -= 1;
        }
        reason.truncate(end);
        return ws.on_upgrade(move |mut socket| async move {
            let _ = socket.send(Message::Close(Some(CloseFrame { code: 1008, reason: reason.into() }))).await;
        });
    }
    let adapter = crate::resolve_message(&st, crate::Surface::RealtimeTranscription).expect("resolved above");
    let s = Session::new(&st, adapter);
    ws.on_upgrade(move |socket| Realtime { st, adapter, socket, s, d: Redecode }.run())
}

/// Whether the model hears transcription as it comes (`Model.live_transcription_stream`).
pub fn live_transcription() -> bool {
    conversation::facts().is_some_and(conversation::Facts::live_transcription)
}

/// Load the turn detection models once at startup, so a missing one is a
/// startup warning rather than a surprise in the first session.
pub fn report_availability() {
    let _ = vad::silero_available();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_scrubbed_as_serving_realtime_scrubs_them() {
        // _scrub_paths' output for each, with the upload directory /srv/up
        // (checked against it on 20,000 random strings too)
        for (message, want) in [
            ("failed reading /srv/up/rt-abc-1.wav", "failed reading <upload>/rt-abc-1.wav"),
            ("no /models/ here", "no /models/ here"),
            ("cd /tmp", "cd /tmp"),
            ("open '/a dir/b c.json' failed", "open 'b c.json' failed"),
            ("x/a/b and (/opt/x/y.json)", "x/a/b and (y.json)"),
            ("/a/b/ then //c", "b/ then //c"),
        ] {
            assert_eq!(scrub_paths(message, std::path::Path::new("/srv/up")), want, "{message}");
        }
    }

    #[test]
    fn excerpts_read_as_reprlib_cuts_them() {
        assert_eq!(excerpt(&json!("abc")), "'abc'");
        assert_eq!(excerpt(&json!([1, 2, 3, 4, 5])), "[1, 2, 3, 4, ...]");
        assert_eq!(excerpt(&json!({"a": [[1]]})), "{'a': [[...]]}");
        let long = excerpt(&json!("x".repeat(300)));
        assert_eq!(long.chars().count(), 100);
    }
}

