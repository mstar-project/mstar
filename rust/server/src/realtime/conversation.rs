//! `/v1/realtime` conversations (OpenAI's GA events) with a full-duplex model,
//! one whose client stream declares `ClientAudio` and that decides when to
//! speak: a port of `serving_realtime.RealtimeConversation` (see its
//! docstring, and docs/clients.rst for the protocol).
//!
//! The call is one engine request, opened by the first append with the
//! session's instructions as its prompt; formats and instructions are fixed
//! from then on. Input is decoded, resampled to the stream's rate and
//! appended over the bridge's `live` operations as it comes; the API server's
//! data worker cuts it into the stream's frames and stamps each output chunk
//! with its loop step and turn marks. Each reply between the model's turn
//! marks is one response, its text and audio paired by step.
//!
//! One task drives the socket, the call's output and the clock that pads a
//! pause with silence, so their events interleave as the Python route's
//! coroutines do on its one event loop.

use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures::StreamExt;
use serde_json::{json, Map, Value};

use super::audio::{self, AudioFormat, OPENAI_PCM_RATE};
use super::resample::StreamResampler;
use super::{append_error, call_error, error_event, event, excerpt, invalid, object, open_call, stream_audio};
use crate::adapters::SubmitArgs;
use crate::serving::{AppState, Out, OutStream};
use crate::utf8::Utf8Stream;

/// Most silence a conversation pads one pause in the client's audio with:
/// longer than a reply, so a client that mutes its mic hears the agent out.
const MAX_PAD_SECONDS: f64 = 30.0;
/// How late a client's audio may run before the call counts a pause and pads
/// it: silence padded into the middle of speech reads to a duplex model as the
/// user yielding the turn.
const JITTER_SECONDS: f64 = 0.2;

/// Client events refused with an `unsupported` error, the call going on: the
/// model decides when it speaks and has no turn to cancel or edit.
const UNSUPPORTED_EVENTS: [&str; 6] = [
    "response.create",
    "response.cancel",
    "output_audio_buffer.clear",
    "conversation.item.truncate",
    "conversation.item.delete",
    "conversation.item.retrieve",
];

/// `conversation_facts`: what the launcher read off a model with a
/// `ClientAudio` client stream.
pub struct Facts {
    pub streams: usize,
    pub turn_tokens: bool,
    pub channels: i64,
    /// the stream of a model that hears transcription as it comes
    /// (`Model.live_transcription_stream`); `None`: it doesn't
    pub(super) live: Option<LiveFacts>,
    /// the rest only for a model that converses: its stream, and how it replies
    pub(super) stream: String,
    pub(super) rate: usize,
    pub(super) frame_samples: usize,
    output_rate: usize,
    voice: Option<String>,
    idle_steps: Option<i64>,
    silence_rms: f64,
}

/// `MSTAR_REALTIME`'s `conversation`, or `None` for a model with no `ClientAudio` stream.
pub fn facts() -> Option<&'static Facts> {
    static FACTS: OnceLock<Option<Facts>> = OnceLock::new();
    FACTS
        .get_or_init(|| {
            let all: Value = serde_json::from_str(&std::env::var("MSTAR_REALTIME").ok()?).ok()?;
            let f = all.get("conversation")?.as_object()?;
            let int = |k: &str| f.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
            let live = f.get("live").and_then(Value::as_object).map(|l| LiveFacts {
                stream: l.get("stream").and_then(Value::as_str).unwrap_or_default().to_string(),
                rate: l.get("rate").and_then(Value::as_u64).unwrap_or(0) as usize,
                frame_samples: l.get("frame_samples").and_then(Value::as_u64).unwrap_or(0) as usize,
                max_items: l.get("max_items").and_then(Value::as_u64).map(|n| n as usize),
            });
            Some(Facts {
                streams: int("streams"),
                turn_tokens: f.get("turn_tokens").and_then(Value::as_bool).unwrap_or(false),
                channels: f.get("channels").and_then(Value::as_i64).unwrap_or(1),
                live,
                stream: f.get("stream").and_then(Value::as_str).unwrap_or_default().to_string(),
                rate: int("rate"),
                frame_samples: int("frame_samples"),
                output_rate: int("output_rate"),
                voice: f.get("voice").and_then(Value::as_str).map(str::to_string),
                idle_steps: f.get("idle_steps").and_then(Value::as_i64),
                silence_rms: f.get("silence_rms").and_then(Value::as_f64).unwrap_or(0.005),
            })
        })
        .as_ref()
}

/// `conversation_facts`' `live`: the stream a live transcription hears.
pub(super) struct LiveFacts {
    pub(super) stream: String,
    pub(super) rate: usize,
    pub(super) frame_samples: usize,
    /// the most samples the stream takes in a call (`None`: no bound)
    pub(super) max_items: Option<usize>,
}

impl Facts {
    /// Why this model can't hold a conversation, as the Python route words it; `None`: it can.
    pub fn problem(&self) -> Option<String> {
        if !self.turn_tokens {
            Some("this model has no turn tokens, so it can't serve OpenAI Realtime conversations".into())
        } else if self.streams > 1 {
            Some(format!("this model takes audio on {} client streams, and a conversation feeds one", self.streams))
        } else if self.channels != 1 {
            Some(format!("this model speaks in {} channels, and a conversation's audio is mono", self.channels))
        } else {
            None
        }
    }
}

fn hex_id(prefix: &str, n: usize) -> String {
    format!("{prefix}_{}", &uuid::Uuid::new_v4().simple().to_string()[..n])
}

/// A GA `realtime.response`, as `response.created` / `.done` carry it.
fn response_object(id: &str, status: &str, output: Vec<Value>, status_details: Option<Value>) -> Value {
    let mut r = json!({"id": id, "object": "realtime.response", "status": status, "output_modalities": ["audio"],
                       "max_output_tokens": "inf", "output": output});
    if let Some(d) = status_details {
        r["status_details"] = d;
    }
    r
}

/// A GA event about a response's one content part.
fn content_event(kind: &str, response_id: &str, item_id: &str, fields: Value) -> Value {
    let mut all = json!({"response_id": response_id, "item_id": item_id, "output_index": 0, "content_index": 0});
    if let (Value::Object(all), Value::Object(f)) = (&mut all, fields) {
        all.extend(f);
    }
    event(kind, all)
}

/// Python's truthiness, for a JSON value.
pub(super) fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(m)) => !m.is_empty(),
    }
}

/// Python's `value == number`: a bool is the integer it stands for.
fn equals_number(v: &Value, n: f64) -> bool {
    match v {
        Value::Bool(b) => (*b as i64) as f64 == n,
        Value::Number(x) => x.as_f64() == Some(n),
        _ => false,
    }
}

/// numpy's float32 sum: pairwise, eight accumulators per block of at most 128.
fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0f32, |s, &v| s + v);
    }
    if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
}

fn pcm16_samples(pcm: &[u8]) -> Vec<f32> {
    pcm.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b) as f32 / 32768.0).collect()
}

/// One agent reply: the text steps from its start mark to its end mark (for a
/// model with `turn_idle_steps`, from any text outside a reply, and to that
/// many steps with neither text nor audible audio), and the audio of the same steps.
struct AgentTurn {
    start: i64,
    end: Option<i64>,
    response_id: String,
    item_id: String,
    text: Utf8Stream,
    transcript: String,
    last_text: Option<i64>,
    last_heard: Option<i64>,
}

impl AgentTurn {
    fn new(start: i64) -> AgentTurn {
        AgentTurn {
            start,
            end: None,
            response_id: hex_id("resp", 16),
            item_id: hex_id("item", 12),
            text: Utf8Stream::default(),
            transcript: String::new(),
            last_text: None,
            last_heard: None,
        }
    }

    fn item(&self, status: &str) -> Value {
        let content = if status == "in_progress" { json!([]) } else { json!([{"type": "output_audio", "transcript": self.transcript}]) };
        json!({"id": self.item_id, "object": "realtime.item", "type": "message", "role": "assistant",
               "status": status, "content": content})
    }
}

/// What ended a call's output.
struct Ending {
    status: &'static str,
    code: u16,
}

pub struct Conversation {
    st: AppState,
    socket: WebSocket,
    facts: &'static Facts,
    session_id: String,
    instructions: String,
    turn_detection: String,
    odd_byte: Option<u8>,
    input_format: AudioFormat,
    output_format: AudioFormat,
    in_resampler: Option<StreamResampler>,
    out_resampler: Option<StreamResampler>,
    /// the call's submission and its output, once the first append opened it
    sub: Option<u64>,
    output: Option<OutStream>,
    /// when the call's audio so far has played, in seconds since `epoch`
    played_until: f64,
    pad_left: f64,
    epoch: Instant,
    /// replies whose last audio is not sent yet, oldest first
    turns: VecDeque<AgentTurn>,
    /// steps whose text has arrived
    text_steps: i64,
    /// audio steps waiting for their text
    early: VecDeque<(i64, Vec<u8>)>,
    closed: bool,
}

impl Conversation {
    pub fn new(st: AppState, socket: WebSocket, facts: &'static Facts) -> Conversation {
        let mut c = Conversation {
            st,
            socket,
            facts,
            session_id: hex_id("sess", 16),
            instructions: String::new(),
            turn_detection: "server_vad".into(),
            odd_byte: None,
            input_format: AudioFormat::pcm(OPENAI_PCM_RATE),
            output_format: AudioFormat::pcm(OPENAI_PCM_RATE),
            in_resampler: None,
            out_resampler: None,
            sub: None,
            output: None,
            played_until: 0.0,
            pad_left: MAX_PAD_SECONDS,
            epoch: Instant::now(),
            turns: VecDeque::new(),
            text_steps: 0,
            early: VecDeque::new(),
            closed: false,
        };
        c.set_formats(c.input_format, c.output_format);
        c
    }

    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    fn frame_seconds(&self) -> f64 {
        self.facts.frame_samples as f64 / self.facts.rate as f64
    }

    async fn send(&mut self, event: Value) {
        if self.closed {
            // the call ended: events still arriving get no answer
            return;
        }
        if self.socket.send(Message::Text(event.to_string())).await.is_err() {
            self.closed = true;
        }
    }

    async fn close(&mut self, code: u16) {
        if !self.closed {
            self.closed = true;
            let _ = self.socket.send(Message::Close(Some(CloseFrame { code, reason: "".into() }))).await;
        }
    }

    fn session_view(&self) -> Value {
        json!({
            "type": "realtime", "object": "realtime.session", "id": self.session_id, "model": self.st.model_name,
            "instructions": self.instructions, "output_modalities": ["audio"],
            "audio": {
                "input": {"format": self.input_format.view(), "turn_detection": {"type": self.turn_detection},
                          "transcription": null},
                "output": {"format": self.output_format.view(), "voice": self.facts.voice},
            },
        })
    }

    pub async fn run(mut self) {
        let created = event("session.created", json!({"session": self.session_view()}));
        self.send(created).await;
        let mut clock: Option<tokio::time::Interval> = None;
        loop {
            tokio::select! {
                frame = self.socket.recv() => {
                    let text = match frame {
                        Some(Ok(Message::Text(t))) => t,
                        Some(Ok(Message::Binary(_))) => {
                            self.send(invalid("an event must be a JSON text message, not binary")).await;
                            continue;
                        }
                        Some(Err(e)) => {
                            if let Some(close) = crate::ws::read_failure_close(&e) {
                                let _ = self.socket.send(close).await;
                                self.closed = true;
                            }
                            break;
                        }
                        Some(Ok(Message::Close(_))) => {
                            crate::ws::answer_close(&mut self.socket).await;
                            break;
                        }
                        None => break,
                        Some(Ok(_)) => continue,
                    };
                    match serde_json::from_str::<Value>(&text) {
                        Ok(Value::Object(ev)) => self.handle(ev).await,
                        Ok(_) => self.send(invalid("an event must be a JSON object")).await,
                        Err(_) => self.send(invalid("not JSON")).await,
                    }
                    if clock.is_none() && self.output.is_some() {
                        let period = Duration::from_secs_f64(self.frame_seconds());
                        let mut every = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
                        every.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                        clock = Some(every);
                    }
                }
                item = async { self.output.as_mut().unwrap().next().await }, if self.output.is_some() => {
                    let ending = match item {
                        Some(Out::Chunk(c)) => match self.on_chunk(c).await {
                            Ok(()) => continue,
                            Err(e) => {
                                let error_id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
                                eprintln!("realtime conversation failed (error id {error_id}, session {}): {e}", self.session_id);
                                self.send(error_event(&format!("the call failed (error id {error_id})"), "server_error", None)).await;
                                // the call goes on without anyone reading it: end it
                                self.output = None;
                                Ending { status: "failed", code: 1011 }
                            }
                        },
                        Some(Out::Error { status, message, .. }) => {
                            self.send(call_error(&self.st, status, &message)).await;
                            Ending { status: "failed", code: if status >= 500 { 1011 } else { 1000 } }
                        }
                        None => Ending { status: "incomplete", code: 1000 },
                    };
                    self.end_call(ending).await;
                    return;
                }
                _ = async { clock.as_mut().unwrap().tick().await }, if clock.is_some() => {
                    if !self.pad().await {
                        clock = None;
                    }
                }
            }
        }
        // the client left: dropping the call's output aborts a call still running
    }

    /// No more text will come: held audio goes to its reply, a reply still
    /// open ends here, and the socket closes, since a call that ended takes no
    /// more audio.
    async fn end_call(&mut self, ending: Ending) {
        while let Some((step, pcm)) = self.early.pop_front() {
            self.play(step, &pcm).await;
        }
        while !self.turns.is_empty() {
            self.finish_turn(ending.status).await;
        }
        self.close(ending.code).await;
    }

    async fn handle(&mut self, ev: Map<String, Value>) {
        let kind = ev.get("type");
        match kind.and_then(Value::as_str) {
            Some("session.update") => {
                let ignored = match object(ev.get("session"), "session").and_then(|s| self.update(s.cloned().unwrap_or_default())) {
                    Ok(ignored) => ignored,
                    Err(e) => {
                        self.send(invalid(&e)).await;
                        return;
                    }
                };
                for message in ignored {
                    self.send(error_event(&message, "invalid_request_error", Some("unsupported"))).await;
                }
                let updated = event("session.updated", json!({"session": self.session_view()}));
                self.send(updated).await;
            }
            Some("input_audio_buffer.append") => {
                // not `or ""`: a null is an error, not an empty append, on every realtime path
                let payload = ev.get("audio").cloned().unwrap_or(Value::from(""));
                self.append(&[payload]).await;
            }
            Some("input_audio_buffer.commit") => {
                // the model hears the audio as it comes: nothing to flush
                let committed = event("input_audio_buffer.committed", json!({"previous_item_id": null, "item_id": hex_id("item", 12)}));
                self.send(committed).await;
            }
            Some("input_audio_buffer.clear") => {
                // what reached the call is the model's, a partial frame included
                self.odd_byte = None;
                if let Some(r) = &mut self.in_resampler {
                    r.reset();
                }
                self.send(event("input_audio_buffer.cleared", json!({}))).await;
            }
            Some("conversation.item.create") => self.create_item(ev.get("item")).await,
            Some(k) if UNSUPPORTED_EVENTS.contains(&k) => {
                let message = format!("{k} is not supported: this model decides when it speaks; keep streaming input_audio_buffer.append");
                self.send(error_event(&message, "invalid_request_error", Some("unsupported"))).await;
            }
            _ => {
                let message = format!("unknown event {}", excerpt(kind.unwrap_or(&Value::Null)));
                self.send(invalid(&message)).await;
            }
        }
    }

    /// Apply a session update. One that is malformed, or would change what the
    /// call has fixed, is refused whole. An option this model can't honour
    /// keeps its served value and is answered with one of the returned
    /// messages, so a client's defaults don't cost it the rest of the update.
    fn update(&mut self, session: Map<String, Value>) -> Result<Vec<String>, String> {
        match session.get("type") {
            None => {}
            Some(Value::String(t)) if t == "realtime" => {}
            Some(other) => {
                return Err(format!(
                    "session.type must be \"realtime\"; got {} (transcription sessions are /v1/realtime?intent=transcription)",
                    excerpt(other)
                ))
            }
        }
        let instructions = match session.get("instructions") {
            None => self.instructions.clone(),
            Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => return Err(format!("instructions must be a string; got {}", excerpt(other))),
        };
        let empty = Map::new();
        let audio = object(session.get("audio"), "audio")?.unwrap_or(&empty);
        let audio_in = object(audio.get("input"), "audio.input")?.unwrap_or(&empty);
        let audio_out = object(audio.get("output"), "audio.output")?.unwrap_or(&empty);
        let format = |m: &Map<String, Value>, name: &str, current: AudioFormat| -> Result<AudioFormat, String> {
            match m.get("format") {
                None | Some(Value::Null) => Ok(current),
                Some(v) => audio::audio_format(object(Some(v), name)?, name),
            }
        };
        let in_fmt = format(audio_in, "audio.input.format", self.input_format)?;
        let out_fmt = format(audio_out, "audio.output.format", self.output_format)?;
        if self.sub.is_some() && (instructions != self.instructions || in_fmt != self.input_format || out_fmt != self.output_format) {
            return Err("instructions and audio formats are fixed once audio flows".into());
        }
        let mut ignored = Vec::new();
        if let Some(m) = session.get("output_modalities").filter(|m| !m.is_null()) {
            if *m != json!(["audio"]) {
                ignored.push(format!("output_modalities must be [\"audio\"] (this model always speaks); got {}", excerpt(m)));
            }
        }
        if truthy(session.get("tools")) {
            ignored.push("tools are not supported".into());
        }
        if !matches!(session.get("max_output_tokens"), None | Some(Value::Null)) && session.get("max_output_tokens") != Some(&json!("inf")) {
            ignored.push("max_output_tokens is not supported: the model decides how long it speaks".into());
        }
        for name in ["transcription", "noise_reduction"] {
            if !matches!(audio_in.get(name), None | Some(Value::Null)) {
                ignored.push(format!("audio.input.{name} is not supported"));
            }
        }
        let mut turn_detection = self.turn_detection.clone();
        if let Some(td) = audio_in.get("turn_detection") {
            if td.is_null() {
                ignored.push(
                    "audio.input.turn_detection null is not supported: this model decides when it speaks; use server_vad or semantic_vad".into(),
                );
            } else {
                let td = object(Some(td), "audio.input.turn_detection")?.unwrap_or(&empty);
                let kind = td.get("type").and_then(Value::as_str).filter(|t| ["server_vad", "semantic_vad"].contains(t));
                let fixed = ["create_response", "interrupt_response"].into_iter().find(|n| td.get(*n).is_some_and(|v| *v != Value::Bool(true)));
                match (kind, fixed) {
                    (None, _) => ignored.push(format!(
                        "audio.input.turn_detection.type must be server_vad or semantic_vad; got {}",
                        excerpt(td.get("type").unwrap_or(&Value::Null))
                    )),
                    (_, Some(name)) => {
                        ignored.push(format!("audio.input.turn_detection.{name} must be true: this model always answers and always yields"))
                    }
                    (Some(kind), None) => turn_detection = kind.to_string(),
                }
            }
        }
        let voice = self.facts.voice.as_deref().map_or(Value::Null, Value::from);
        if audio_out.get("voice").is_some_and(|v| *v != voice) {
            let served = self.facts.voice.as_deref().map_or("None".to_string(), crate::media::py_repr);
            ignored.push(format!(
                "audio.output.voice must be {served}, the voice this server speaks in; got {}",
                excerpt(&audio_out["voice"])
            ));
        }
        if audio_out.get("speed").is_some_and(|v| !v.is_null() && !equals_number(v, 1.0)) {
            ignored.push("audio.output.speed is not supported".into());
        }
        self.instructions = instructions;
        self.turn_detection = turn_detection;
        if (in_fmt, out_fmt) != (self.input_format, self.output_format) {
            self.set_formats(in_fmt, out_fmt);
        }
        Ok(ignored)
    }

    /// Before audio flows: the resamplers hold nothing yet.
    fn set_formats(&mut self, in_fmt: AudioFormat, out_fmt: AudioFormat) {
        self.input_format = in_fmt;
        self.output_format = out_fmt;
        self.in_resampler = (in_fmt.rate != self.facts.rate).then(|| StreamResampler::new(in_fmt.rate, self.facts.rate));
        self.out_resampler = (out_fmt.rate != self.facts.output_rate).then(|| StreamResampler::new(self.facts.output_rate, out_fmt.rate));
    }

    /// `conversation.item.create` of a user message of `input_audio` parts:
    /// their audio goes on the stream as appends do.
    async fn create_item(&mut self, item: Option<&Value>) {
        let empty = Map::new();
        let item = item.and_then(Value::as_object).unwrap_or(&empty);
        let is_user_message = item.get("type") == Some(&json!("message")) && item.get("role") == Some(&json!("user"));
        let parts = is_user_message.then(|| item.get("content")).flatten();
        let parts = match parts {
            Some(Value::Array(parts))
                if !parts.is_empty() && parts.iter().all(|p| p.as_object().is_some_and(|p| p.get("type") == Some(&json!("input_audio")))) =>
            {
                parts.clone()
            }
            _ => {
                let message = "conversation.item.create takes only a user message of input_audio parts";
                self.send(error_event(message, "invalid_request_error", Some("unsupported"))).await;
                return;
            }
        };
        let payloads: Vec<Value> = parts.iter().map(|p| p.get("audio").cloned().unwrap_or(Value::from(""))).collect();
        if !self.append(&payloads).await {
            return; // the error is sent; no part was taken
        }
        // the audio is not echoed back
        let mut done = item.clone();
        let content = parts.iter().map(|p| {
            let mut p = p.as_object().cloned().unwrap_or_default();
            p.remove("audio");
            Value::Object(p)
        });
        done.insert("content".into(), Value::Array(content.collect()));
        if !truthy(item.get("id")) {
            done.insert("id".into(), Value::from(hex_id("item", 12)));
        }
        done.insert("object".into(), Value::from("realtime.item"));
        done.insert("status".into(), Value::from("completed"));
        let done = Value::Object(done);
        self.send(event("conversation.item.added", json!({"previous_item_id": null, "item": done}))).await;
        self.send(event("conversation.item.done", json!({"previous_item_id": null, "item": done}))).await;
    }

    /// Put the audio of an append, or of an item's parts, on the call, all or
    /// none; false, with an `error` sent (or the call over), if it was not taken.
    async fn append(&mut self, payloads: &[Value]) -> bool {
        let mut odd_byte = self.odd_byte;
        let mut audio = Vec::new();
        for payload in payloads {
            match audio::decode_append(payload, odd_byte, self.input_format.codec) {
                Ok((piece, odd)) => {
                    audio.extend(piece);
                    odd_byte = odd;
                }
                Err(()) => {
                    self.send(error_event("audio is not base64", "invalid_request_error", Some("invalid_audio"))).await;
                    return false;
                }
            }
        }
        if let Some(error) = append_error(audio.len(), self.input_format.rate) {
            self.send(error).await;
            return false;
        }
        if self.sub.is_none() && !self.open().await {
            return false; // not taken, its half sample neither: before the call, formats may still change
        }
        self.odd_byte = odd_byte;
        self.put(audio, false).await
    }

    /// Resample `audio` (at the client's rate) to the stream's and append it to
    /// the call; false if the call has ended (its output reports why).
    async fn put(&mut self, audio: Vec<f32>, padding: bool) -> bool {
        // client audio after a pause plays from now; padding fills the pause
        let start = if padding { self.played_until } else { self.played_until.max(self.now()) };
        self.played_until = start + audio.len() as f64 / self.input_format.rate as f64;
        if !audio.is_empty() && !padding {
            self.pad_left = MAX_PAD_SECONDS;
        }
        let audio = match &mut self.in_resampler {
            // a long append is resampled off the async workers
            Some(r) if !audio.is_empty() => tokio::task::block_in_place(|| r.push(&audio)),
            _ => audio,
        };
        if audio.is_empty() {
            return true;
        }
        let Some(sub) = self.sub else { return false };
        match stream_audio(&self.st, sub, &self.facts.stream, &audio, &self.session_id).await {
            Ok(taken) => taken,
            Err(error) => {
                self.send(error).await;
                false
            }
        }
    }

    /// One tick of the clock that keeps the call's audio up with real time
    /// while the client sends none (a muted mic, a client waiting for the
    /// reply), since the model steps only on audio it hears: once it is a frame
    /// past `JITTER_SECONDS` behind, whole frames of silence go on, at most
    /// `MAX_PAD_SECONDS` a pause. Never while half a sample is pending, which
    /// the silence would split. False once the call has ended.
    async fn pad(&mut self) -> bool {
        let frame = self.frame_seconds();
        let behind = (self.now() - JITTER_SECONDS - self.played_until).min(self.pad_left);
        let frames = (behind / frame) as i64; // int(): toward zero
        if frames < 1 || self.odd_byte.is_some() {
            return true;
        }
        self.pad_left -= frames as f64 * frame;
        let n = (frames as f64 * frame * self.input_format.rate as f64).round_ties_even() as usize;
        self.put(vec![0.0; n], true).await
    }

    /// Open the call's engine request; false, with an `error` sent, if the
    /// server refused it (at capacity, say).
    async fn open(&mut self) -> bool {
        let text = (!self.instructions.is_empty()).then(|| self.instructions.clone());
        let args = SubmitArgs {
            input_modalities: if text.is_some() { vec!["text".into()] } else { vec![] },
            output_modalities: vec!["text".into(), "audio".into()],
            text,
            end_input: Some(Value::Bool(false)),
            ..SubmitArgs::default()
        };
        match open_call(&self.st, &args, &self.session_id).await {
            Ok((sub, output)) => {
                self.sub = Some(sub);
                self.output = Some(output);
                self.played_until = self.now();
                true
            }
            Err(error) => {
                self.send(error).await;
                false
            }
        }
    }

    async fn on_chunk(&mut self, chunk: crate::bridge::ResultChunk) -> Result<(), String> {
        match chunk.modality.as_str() {
            "text" => self.on_text(chunk).await,
            "audio" => {
                let step = chunk.metadata.get("step").and_then(Value::as_i64).ok_or("an audio chunk without its step")?;
                if step >= self.text_steps {
                    self.early.push_back((step, chunk.data)); // its text decides which reply it is in
                } else {
                    self.play(step, &chunk.data).await;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// The open reply, the newest one, while it has no end.
    fn open_turn_index(&self) -> Option<usize> {
        self.turns.back().filter(|t| t.end.is_none()).map(|_| self.turns.len() - 1)
    }

    async fn on_text(&mut self, chunk: crate::bridge::ResultChunk) -> Result<(), String> {
        let step = chunk.metadata.get("step").and_then(Value::as_i64).ok_or("a text chunk without its step")?;
        self.text_steps = step + 1;
        let mut turn = self.open_turn_index();
        let mut data = chunk.data;
        let marks: Vec<String> = chunk
            .metadata
            .get("turn")
            .and_then(Value::as_array)
            .map(|m| m.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default();
        if turn.is_none() && !data.is_empty() && self.facts.idle_steps.is_some() && !marks.iter().any(|m| m == "start") {
            // text after a reply ended on quiet, with no start mark: the model speaks again
            turn = Some(self.start_turn(step).await);
        }
        for mark in &marks {
            if mark == "start" {
                if let Some(open) = turn {
                    // no end mark: the open reply ends before this one
                    eprintln!("WARNING: realtime reply {} got no end mark; ending it at step {}", self.turns[open].response_id, step - 1);
                    self.feed(open, b"", true).await;
                    self.turns[open].end = Some(step - 1);
                }
                turn = Some(self.start_turn(step).await);
            } else if mark == "end" {
                if let Some(open) = turn {
                    self.feed(open, &data, true).await;
                    data.clear();
                    self.turns[open].end = Some(step);
                    turn = None;
                }
            }
        }
        if let Some(open) = turn {
            // between replies the text channel is padding
            self.feed(open, &data, false).await;
            if !data.is_empty() || !marks.is_empty() {
                self.turns[open].last_text = Some(step);
            }
        }
        while self.early.front().is_some_and(|(s, _)| *s < self.text_steps) {
            let (s, pcm) = self.early.pop_front().unwrap();
            self.play(s, &pcm).await;
        }
        Ok(())
    }

    async fn start_turn(&mut self, step: i64) -> usize {
        self.turns.push_back(AgentTurn::new(step));
        if self.turns.len() == 1 {
            self.open_reply().await;
        }
        self.turns.len() - 1
    }

    /// Send the opening events of the oldest reply, and its transcript so far.
    async fn open_reply(&mut self) {
        let t = &self.turns[0];
        let (rid, iid, item, transcript) = (t.response_id.clone(), t.item_id.clone(), t.item("in_progress"), t.transcript.clone());
        self.send(event("response.created", json!({"response": response_object(&rid, "in_progress", vec![], None)}))).await;
        self.send(event("conversation.item.added", json!({"previous_item_id": null, "item": item}))).await;
        self.send(event("response.output_item.added", json!({"response_id": rid, "output_index": 0, "item": item}))).await;
        self.send(content_event("response.content_part.added", &rid, &iid, json!({"part": {"type": "audio", "transcript": ""}}))).await;
        if !transcript.is_empty() {
            self.send(content_event("response.output_audio_transcript.delta", &rid, &iid, json!({"delta": transcript}))).await;
        }
    }

    async fn feed(&mut self, index: usize, data: &[u8], last: bool) {
        let t = &mut self.turns[index];
        let mut delta = t.text.feed(data);
        if last {
            delta.push_str(&t.text.flush());
        }
        if delta.is_empty() {
            return;
        }
        t.transcript.push_str(&delta);
        if index == 0 {
            // a later reply's text goes out when it opens
            let (rid, iid) = (t.response_id.clone(), t.item_id.clone());
            self.send(content_event("response.output_audio_transcript.delta", &rid, &iid, json!({"delta": delta}))).await;
        }
    }

    /// Send step `step`'s audio in the reply it belongs to, if any.
    async fn play(&mut self, step: i64, pcm: &[u8]) {
        while self.turns.front().is_some_and(|t| t.end.is_some_and(|end| end < step)) {
            self.finish_turn("completed").await; // a reply whose last step had no audio
        }
        if !self.turns.front().is_some_and(|t| t.start <= step) {
            return; // silence between replies
        }
        if self.turns[0].end.is_none() && self.facts.idle_steps.is_some() {
            self.end_if_idle(step, pcm);
            if self.turns[0].end.is_some() {
                // its text's held tail goes out, as on every other end
                self.feed(0, b"", true).await;
            }
        }
        let last = self.turns[0].end == Some(step);
        let audio = tokio::task::block_in_place(|| self.encode_out(pcm, last));
        if !audio.is_empty() {
            let t = &self.turns[0];
            let (rid, iid) = (t.response_id.clone(), t.item_id.clone());
            self.send(content_event("response.output_audio.delta", &rid, &iid, json!({"delta": audio}))).await;
        }
        if last {
            self.finish_turn("completed").await;
        }
    }

    /// A model that may send no end mark (`turn_idle_steps`): its reply ends
    /// once that many steps have passed with no text and no audio above
    /// `turn_silence_rms`. Counted from the audio, which runs behind the text,
    /// so the speech still sounding after the last text stays in the reply.
    fn end_if_idle(&mut self, step: i64, pcm: &[u8]) {
        let samples = pcm16_samples(pcm);
        let t = &mut self.turns[0];
        if !samples.is_empty() {
            // np.mean of float32 squares, then np.sqrt, all in float32
            let squares: Vec<f32> = samples.iter().map(|s| s * s).collect();
            let rms = (pairwise_sum_f32(&squares) / samples.len() as f32).sqrt();
            if rms as f64 > self.facts.silence_rms {
                t.last_heard = Some(step);
            }
        }
        let since = [Some(t.start), t.last_text, t.last_heard].into_iter().flatten().max().unwrap_or(t.start);
        if step - since >= self.facts.idle_steps.unwrap_or(i64::MAX) {
            t.end = Some(step);
        }
    }

    fn encode_out(&mut self, pcm: &[u8], last: bool) -> String {
        let mut audio = pcm16_samples(pcm);
        if let Some(r) = &mut self.out_resampler {
            audio = r.push(&audio);
            if last {
                // the reply ends: what the resampler holds goes out with it
                audio.extend(r.flush());
            }
        }
        if audio.is_empty() { String::new() } else { audio::encode_audio(&audio, self.output_format.codec) }
    }

    /// Close the oldest reply as `status`: `completed`, or `incomplete` /
    /// `failed` when the call ended inside it; then open the next.
    async fn finish_turn(&mut self, status: &str) {
        if self.turns[0].end.is_none() {
            self.feed(0, b"", true).await;
        }
        let t = self.turns.pop_front().unwrap();
        if let Some(r) = &mut self.out_resampler {
            r.reset(); // already flushed, unless the reply ended early
        }
        let (rid, iid) = (&t.response_id, &t.item_id);
        self.send(content_event("response.output_audio.done", rid, iid, json!({}))).await;
        self.send(content_event("response.output_audio_transcript.done", rid, iid, json!({"transcript": t.transcript}))).await;
        self.send(content_event("response.content_part.done", rid, iid, json!({"part": {"type": "audio", "transcript": t.transcript}}))).await;
        let item = t.item(if status == "completed" { "completed" } else { "incomplete" });
        self.send(event("response.output_item.done", json!({"response_id": rid, "output_index": 0, "item": item}))).await;
        self.send(event("conversation.item.done", json!({"previous_item_id": null, "item": item}))).await;
        let details = (status != "completed").then(|| json!({"type": status}));
        self.send(event("response.done", json!({"response": response_object(rid, status, vec![item], details)}))).await;
        if !self.turns.is_empty() {
            self.open_reply().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_float32_mean_adds_up_as_numpy_adds_it() {
        // np.add.reduce of these float32 values, where a running sum gives 14.968001365661621
        let x: Vec<f32> = (0..300).map(|i| ((i * 37 % 101) as f32) / 1000.0).collect();
        assert_eq!(pairwise_sum_f32(&x) as f64, 14.968000411987305);
        assert_eq!(x.iter().fold(0.0f32, |s, &v| s + v) as f64, 14.968001365661621);
    }
}
