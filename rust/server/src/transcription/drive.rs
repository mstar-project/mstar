//! The long-form drivers of `serving_transcriptions.py`: the upload's windows
//! on the engine, parallel (`_parallel`, `_tail`) or sequential
//! (`_sequential`, openai-whisper's loop with its temperature fallback), each
//! non-streamed (`create_transcription`) or streamed (`_stream`).

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::response::Response;
use futures::StreamExt;
use serde_json::{json, Value};

use super::format::{self, merge_windows};
use super::{reads, Transcript, TranscriptionPolicy, TranscriptionRequest, TIMED_FORMATS};
use crate::adapters::OpenAIAdapter;
use crate::serving::{collect, rid, try_result_stream, AppState, Out, OutStream};

/// openai-whisper's fallback schedule, used unless the caller pinned a temperature.
const TEMPERATURE_FALLBACK: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
/// A window that needed a temperature above this stops conditioning the next.
const PROMPT_RESET_TEMPERATURE: f64 = 0.5;
/// How much of the transcript so far conditions the next window, in characters.
const CARRY_OVER_CHARS: usize = 1200;
/// A seek shorter than this re-hears too much for too little.
const MIN_SEEK_SECONDS: f64 = 1.0;
/// Parallel mode: a window's unfinished tail shorter than this is not decoded again.
const MIN_REDECODE_SECONDS: f64 = 0.5;
/// A tail shorter than this holds no speech.
const MIN_TAIL_SECONDS: f64 = 0.02;
/// Parallel mode: windows on the engine at once for one upload.
const PARALLEL_INFLIGHT: usize = 32;

/// A failure, with what each mode answers it with: the non-streamed route the
/// status and message (an `HTTPException`'s, or `client_error_text`'s), the
/// stream an `HTTPException`'s detail or `transcription failed (<type>)`.
#[derive(Debug)]
pub struct OpError {
    pub status: u16,
    pub message: String,
    http: bool,
    type_name: String,
}

impl OpError {
    pub fn http(status: u16, message: String) -> OpError {
        OpError { status, message, http: true, type_name: String::new() }
    }

    /// A `ValueError` a submission raised (an adapter's refusal, an unknown LoRA adapter).
    fn value(message: String) -> OpError {
        OpError { status: 400, message, http: false, type_name: "ValueError".to_string() }
    }

    /// `_stream_error_event`'s status and message.
    fn in_band(&self) -> (u16, String) {
        if self.http {
            (self.status, self.message.clone())
        } else {
            (500, format!("transcription failed ({})", self.type_name))
        }
    }
}

/// One engine request's worth of the upload, and where it sits in it.
#[derive(Debug, Clone, Default)]
pub struct Window {
    pub path: String,
    pub offset: f64,
    pub duration: f64,
    pub raw_text: String,
    /// The accepted parse (sequential mode drops an unfinished last segment).
    pub transcript: Option<Transcript>,
    /// The temperature it was accepted at; `None` where no retry ladder ran (the request's).
    pub temperature: Option<f64>,
    pub compression_ratio: f64,
}

/// The planner's window policy for one upload (`planner_for`).
#[derive(Debug, Clone, Copy)]
pub struct PlanShape {
    max_seconds: Option<f64>,
    whole_seconds: Option<f64>,
    balanced: bool,
    search_seconds: f64,
    frame_seconds: f64,
}

impl PlanShape {
    /// `window` is a request's `window_seconds`, which sets the window and the
    /// longest upload heard whole.
    pub fn of(policy: &TranscriptionPolicy, window: Option<f64>) -> PlanShape {
        let (max_seconds, whole_seconds) = match window {
            None => (policy.window_seconds.filter(|w| *w != 0.0).or(policy.max_audio_seconds), policy.whole_seconds),
            Some(w) => (Some(w), Some(w)),
        };
        PlanShape {
            max_seconds,
            whole_seconds,
            balanced: policy.balanced_windows,
            search_seconds: policy.window_search_seconds.unwrap_or(super::CUT_SEARCH_SECONDS),
            frame_seconds: policy.window_frame_seconds.unwrap_or(super::CUT_FRAME_SECONDS),
        }
    }
}

/// The API server's `WindowPlanner` for one upload, over the bridge: it
/// decodes the upload with the model's loader and writes each window as a
/// WAV beside it. Dropping it has the planner delete the upload and its
/// windows (`cleanup`).
pub struct Planner {
    bridge: Option<Arc<crate::bridge::Bridge>>,
    id: String,
    timeout: Duration,
    pub audio_path: String,
    /// The upload's length: the header's, then the decoded one.
    pub duration: Option<f64>,
    /// The upload runs as one window.
    pub single: bool,
}

impl Planner {
    /// Plan an upload: the planner refuses one past the 30 minute cap, and
    /// decodes one that will be windowed (or has no length in its header), so a
    /// file its decoder refuses fails before any window runs.
    pub async fn open(st: &AppState, audio_path: String, shape: PlanShape) -> Result<Planner, OpError> {
        let mut planner = Planner {
            bridge: st.bridge.clone(),
            id: uuid::Uuid::new_v4().simple().to_string(),
            timeout: st.request_timeout,
            audio_path,
            duration: None,
            single: true,
        };
        if planner.bridge.is_none() {
            return Ok(planner); // mock mode: the whole file, of unknown length
        }
        let op = json!({
            "op": "plan", "path": planner.audio_path, "max_seconds": shape.max_seconds,
            "whole_seconds": shape.whole_seconds, "balanced": shape.balanced,
            "search_seconds": shape.search_seconds, "frame_seconds": shape.frame_seconds,
        });
        planner.call(op).await?;
        Ok(planner)
    }

    async fn call(&mut self, mut op: Value) -> Result<Vec<Window>, OpError> {
        let bridge = self.bridge.clone().expect("a planner op needs the bridge");
        op["plan"] = Value::from(self.id.clone());
        let Some(reply) = bridge.windows(&op, self.timeout).await else {
            return Err(OpError::http(503, "backend bridge unreachable".to_string()));
        };
        let meta = reply.metadata;
        if let Some(err) = meta.get("error") {
            return Err(OpError {
                status: err["status"].as_u64().unwrap_or(500) as u16,
                message: err["message"].as_str().unwrap_or("").to_string(),
                http: err["http"].as_bool().unwrap_or(false),
                type_name: err["type"].as_str().unwrap_or("Exception").to_string(),
            });
        }
        self.duration = meta.get("duration").and_then(Value::as_f64);
        self.single = meta.get("single").and_then(Value::as_bool).unwrap_or(true);
        let windows = meta.get("windows").and_then(Value::as_array).cloned().unwrap_or_default();
        Ok(windows
            .iter()
            .map(|w| Window {
                path: w["path"].as_str().unwrap_or("").to_string(),
                offset: w["offset"].as_f64().unwrap_or(0.0),
                duration: w["duration"].as_f64().unwrap_or(0.0),
                ..Window::default()
            })
            .collect())
    }

    pub fn whole(&self) -> Window {
        Window { path: self.audio_path.clone(), duration: self.duration.unwrap_or(0.0), ..Window::default() }
    }

    fn done(&self, offset: f64) -> bool {
        self.single || offset >= self.duration.unwrap_or(0.0) - MIN_TAIL_SECONDS
    }

    /// The whole upload cut up front, each cut moved to a quiet frame.
    async fn fixed(&mut self) -> Result<Vec<Window>, OpError> {
        if self.single {
            return Ok(vec![self.whole()]);
        }
        self.call(json!({"op": "fixed"})).await
    }

    /// One window from `offset` on.
    async fn window_at(&mut self, offset: f64) -> Result<Window, OpError> {
        if self.single {
            return Ok(self.whole());
        }
        Ok(self.call(json!({"op": "at", "offset": offset})).await?.remove(0))
    }

    /// `duration` seconds of the upload from `offset` on.
    async fn cut(&mut self, offset: f64, duration: f64) -> Result<Window, OpError> {
        Ok(self.call(json!({"op": "cut", "offset": offset, "duration": duration})).await?.remove(0))
    }
}

impl Drop for Planner {
    fn drop(&mut self) {
        match &self.bridge {
            // the planner deletes the upload and the windows; one it never
            // made (the plan was refused, or never answered) the upload alone
            Some(bridge) => bridge.close_windows(&self.id, &self.audio_path),
            None => {
                let _ = std::fs::remove_file(&self.audio_path);
            }
        }
    }
}

/// `WindowShare`: the windows one upload cut in parallel keeps on the engine.
/// A window past the deployment's `max_concurrent_requests` waits in the
/// conductor with its audio staged, so the uploads being transcribed split the
/// cap, at least one window each.
struct Share;

static UPLOADS: AtomicUsize = AtomicUsize::new(0);

impl Share {
    fn join() -> Share {
        UPLOADS.fetch_add(1, Ordering::SeqCst);
        Share
    }

    fn per_upload(&self) -> usize {
        match super::facts().max_concurrent_requests {
            None | Some(0) => PARALLEL_INFLIGHT,
            Some(cap) => (cap / UPLOADS.load(Ordering::SeqCst).max(1)).clamp(1, PARALLEL_INFLIGHT),
        }
    }
}

impl Drop for Share {
    fn drop(&mut self) {
        UPLOADS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A task aborted when this drops: a tail read, or the streaming driver, whose
/// windows are released when it is (their result streams abort on drop).
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// A tail's text, read as it lands.
type TailRead = AbortOnDrop<Result<String, OpError>>;

pub struct Ctx {
    pub st: AppState,
    pub adapter: &'static dyn OpenAIAdapter,
    pub req: TranscriptionRequest,
    pub mode: &'static str,
    pub planner: Planner,
}

/// `_submit`: one window on the engine.
fn submit(st: &AppState, adapter: &dyn OpenAIAdapter, req: &TranscriptionRequest, window: &Window, streaming: bool) -> Result<OutStream, OpError> {
    let mut args = adapter.transcription_to_request(req, &window.path).map_err(OpError::value)?;
    args.model_kwargs.remove("long_form");
    args.model_kwargs.remove("window_seconds");
    st.served.select(req.model.as_deref(), &mut args.model_kwargs).map_err(OpError::value)?;
    try_result_stream(st, &args, &rid("transcr"), streaming).map_err(|(s, m)| OpError::http(s, m))
}

/// `_collect` / `_text_of`: a window's raw text, the bytes joined before they
/// are decoded, since a byte-level BPE token can end mid-character.
async fn read(stream: OutStream) -> Result<String, OpError> {
    let chunks = collect(stream).await.map_err(|(s, m)| OpError::http(s, m))?;
    let data: Vec<u8> = chunks.iter().filter(|c| c.modality == "text").flat_map(|c| c.data.iter().copied()).collect();
    Ok(String::from_utf8_lossy(&data).into_owned())
}

/// `_carry_over`: the last `CARRY_OVER_CHARS` of the accepted texts.
fn carry_over(texts: &[String]) -> String {
    let joined = texts.iter().filter(|t| !t.is_empty()).cloned().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = crate::adapters::py_strip(&joined).chars().collect();
    chars[chars.len().saturating_sub(CARRY_OVER_CHARS)..].iter().collect()
}

/// `_closed`: `t` without its unfinished last segment.
fn closed(t: &Transcript) -> Transcript {
    let end = t.segments.last().map_or(0.0, |s| s.end);
    Transcript {
        text: t
            .segments
            .iter()
            .map(|s| crate::adapters::py_strip(&s.text))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        language: t.language.clone(),
        segments: t.segments.clone(),
        words: t.words.iter().filter(|w| w.end <= end + 1e-6).cloned().collect(),
        unfinished: false,
    }
}

/// `_window_request`: the caller's request with the language the first window
/// settled, the transcript so far as the conditioning text and, when the
/// driver seeks by timestamps, segment timestamps switched on.
fn window_request(req: &TranscriptionRequest, language: Option<&str>, carry: Option<&str>, timestamps: bool) -> TranscriptionRequest {
    let mut out = req.clone();
    if let Some(language) = language.filter(|l| !l.is_empty()) {
        if req.language.as_deref().is_none_or(str::is_empty) {
            out.language = Some(language.to_string());
        }
    }
    if let Some(carry) = carry {
        out.prompt = Some(carry.to_string());
    }
    if timestamps
        && req.timestamp_granularities.as_ref().is_none_or(Vec::is_empty)
        && !TIMED_FORMATS.contains(&req.response_format.as_str())
    {
        out.timestamp_granularities = Some(vec!["segment".to_string()]);
    }
    out
}

/// `_attempts`: the `(temperature, conditioning text)` ladder for one window.
fn attempts(req: &TranscriptionRequest, carry: Option<String>) -> Vec<(f64, Option<String>)> {
    let base = req.temperature.filter(|t| *t != 0.0).unwrap_or(0.0);
    let carried = carry.as_deref().is_some_and(|c| !c.is_empty());
    let mut ladder = vec![(base, carry)];
    if carried {
        ladder.push((base, None));
    }
    if req.temperature.is_none_or(|t| t == 0.0) {
        ladder.extend(TEMPERATURE_FALLBACK.iter().filter(|t| **t > base).map(|t| (*t, None)));
    }
    ladder
}

/// `_sequential`, one accepted window per [`Sequential::next`].
struct Sequential {
    language: Option<String>,
    texts: Vec<String>,
    reset_since: usize,
    seeks: bool,
    conditions: bool,
    offset: f64,
    finished: bool,
}

impl Sequential {
    fn new(ctx: &Ctx) -> Sequential {
        Sequential {
            language: ctx.req.language.clone(),
            texts: Vec::new(),
            reset_since: 0,
            // route-added conditioning and timestamps go only to a model that reads them
            seeks: ctx.adapter.transcription_policy().seeks_by_timestamps && !ctx.planner.single && reads("timestamps"),
            conditions: reads("initial_prompt"),
            offset: 0.0,
            finished: false,
        }
    }

    async fn next(&mut self, ctx: &mut Ctx) -> Result<Option<Window>, OpError> {
        if self.finished {
            return Ok(None);
        }
        let mut window = ctx.planner.window_at(self.offset).await?;
        let carry = (!self.texts.is_empty() && self.conditions).then(|| carry_over(&self.texts[self.reset_since..]));
        let samples = reads("temperature");
        let threshold = ctx.adapter.transcription_policy().compression_ratio_threshold;
        let mut parsed = Transcript::default();
        for (temperature, prompt) in attempts(&ctx.req, carry) {
            let mut attempt = window_request(&ctx.req, self.language.as_deref(), prompt.as_deref(), self.seeks);
            // every attempt carries its rung, so the greedy retry without the
            // prompt stays greedy whatever the checkpoint defaults to
            if samples {
                attempt.temperature = Some(temperature);
                attempt.temperature_sent = true;
            }
            let stream = submit(&ctx.st, ctx.adapter, &attempt, &window, false)?;
            window.raw_text = read(stream).await?;
            parsed = ctx.adapter.parse_transcript(&window.raw_text, &attempt);
            window.temperature = Some(temperature);
            window.compression_ratio = super::zlib::compression_ratio(&parsed.text);
            if threshold.is_none_or(|t| window.compression_ratio <= t) {
                break;
            }
        }
        let mut consumed = window.duration;
        if self.seeks && parsed.unfinished && !parsed.segments.is_empty() {
            let end = parsed.segments.last().unwrap().end;
            if (MIN_SEEK_SECONDS..window.duration).contains(&end) {
                consumed = end;
                parsed = closed(&parsed);
            }
        }
        self.texts.push(parsed.text.clone());
        if window.temperature.is_some_and(|t| t > PROMPT_RESET_TEMPERATURE) {
            self.reset_since = self.texts.len();
        }
        if self.language.as_deref().is_none_or(str::is_empty) {
            self.language = parsed.language.clone().or(self.language.take());
        }
        window.transcript = Some(parsed);
        self.offset += consumed;
        self.finished = ctx.planner.done(self.offset);
        Ok(Some(window))
    }
}

/// `_tail`: the unfinished tail of a parallel window decoded with timestamps,
/// as a window of its own. Near its end Whisper writes the next segment's
/// start and stops, leaving that speech to the next window, which here starts
/// at the fixed cut instead; the window keeps its closed segments, and the
/// audio from the last of them to the cut is decoded again.
async fn tail(ctx: &mut Ctx, window: &mut Window) -> Result<Option<Window>, OpError> {
    if !ctx.adapter.transcription_policy().seeks_by_timestamps || ctx.planner.single {
        return Ok(None);
    }
    let parsed = ctx.adapter.parse_transcript(&window.raw_text, &ctx.req);
    let Some(end) = parsed.segments.last().map(|s| s.end).filter(|_| parsed.unfinished) else {
        return Ok(None);
    };
    if !(MIN_SEEK_SECONDS <= end && end <= window.duration - MIN_REDECODE_SECONDS) {
        return Ok(None);
    }
    window.transcript = Some(closed(&parsed));
    ctx.planner.cut(window.offset + end, window.duration - end).await.map(Some)
}

/// The non-streamed route's windows: parallel (or the one window), or sequential.
pub async fn transcribe(mut ctx: Ctx) -> Result<(Transcript, Option<f64>, TranscriptionRequest), OpError> {
    // every upload counts against the share, whichever mode it runs in
    let share = Share::join();
    let windows = if ctx.mode == "parallel" || ctx.planner.single {
        parallel(&mut ctx, &share).await?
    } else {
        let mut sequential = Sequential::new(&ctx);
        let mut windows = Vec::new();
        while let Some(window) = sequential.next(&mut ctx).await? {
            windows.push(window);
        }
        windows
    };
    drop(share);
    let transcript = merge_windows(&windows, ctx.adapter, &ctx.req);
    Ok((transcript, ctx.planner.duration, ctx.req.clone()))
}

/// `_parallel` with each window's tail: windows in order with up to the
/// upload's share on the engine, and each tail on it as soon as its window is
/// in, counted against the share until its text lands.
async fn parallel(ctx: &mut Ctx, share: &Share) -> Result<Vec<Window>, OpError> {
    let mut windows = ctx.planner.fixed().await?;
    let n = windows.len();
    let mut pending: VecDeque<OutStream> = VecDeque::new();
    let mut submitted = 0;
    let live = Arc::new(AtomicUsize::new(0));
    let mut tails: Vec<(usize, Window, TailRead)> = Vec::new();
    for index in 0..n {
        while submitted < n && (submitted <= index || pending.len() + live.load(Ordering::SeqCst) < share.per_upload()) {
            pending.push_back(submit(&ctx.st, ctx.adapter, &ctx.req, &windows[submitted], false)?);
            submitted += 1;
        }
        windows[index].raw_text = read(pending.pop_front().expect("window submitted")).await?;
        if let Some(t) = tail(ctx, &mut windows[index]).await? {
            let stream = submit(&ctx.st, ctx.adapter, &ctx.req, &t, false)?;
            live.fetch_add(1, Ordering::SeqCst);
            let counter = live.clone();
            let task = tokio::spawn(async move {
                let text = read(stream).await;
                counter.fetch_sub(1, Ordering::SeqCst);
                text
            });
            tails.push((index, t, AbortOnDrop(task)));
        }
    }
    let mut merged = Vec::with_capacity(n + tails.len());
    let mut tails = tails.into_iter().peekable();
    for (index, window) in windows.into_iter().enumerate() {
        merged.push(window);
        if tails.peek().is_some_and(|(i, _, _)| *i == index) {
            let (_, mut t, mut task) = tails.next().unwrap();
            t.raw_text = match (&mut task.0).await {
                Ok(text) => text?,
                Err(e) => return Err(OpError { status: 500, message: e.to_string(), http: false, type_name: "RuntimeError".to_string() }),
            };
            merged.push(t);
        }
    }
    Ok(merged)
}

/// `_stream_delta`: the part of the clean text so far not sent yet; a revision
/// of sent text is not resent.
fn stream_delta(text: String, emitted: &mut String) -> String {
    match text.strip_prefix(emitted.as_str()) {
        Some(delta) => {
            let delta = delta.to_string();
            *emitted = text;
            delta
        }
        None => String::new(),
    }
}

/// The streamed route: transcription events as they come, closed by `[DONE]`.
pub fn stream_response(ctx: Ctx) -> Response {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Bytes>(64);
    let task = AbortOnDrop(tokio::spawn(async move {
        let send = tx.clone();
        if let Err(e) = stream_events(ctx, &send).await {
            let (status, message) = e.in_band();
            let _ = send.send(Bytes::from(format::error_event(&message, status))).await;
            let _ = send.send(Bytes::from_static(format::SSE_DONE.as_bytes())).await;
        }
    }));
    // dropping the body (the client is gone) aborts the driver, which releases its windows
    let body = futures::stream::poll_fn(move |cx| {
        let _ = &task;
        rx.poll_recv(cx).map(|b| b.map(Ok::<_, Infallible>))
    });
    Response::builder()
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(body))
        .unwrap()
}

async fn emit(tx: &tokio::sync::mpsc::Sender<Bytes>, event: String) {
    let _ = tx.send(Bytes::from(event)).await;
}

/// `_stream`: deltas as they come for one window or parallel windows (in
/// window order); sequential long form sends each window's text once the
/// window is accepted, since only then is it known to stand.
async fn stream_events(mut ctx: Ctx, tx: &tokio::sync::mpsc::Sender<Bytes>) -> Result<(), OpError> {
    let share = Share::join();
    let mut windows: Vec<Window> = Vec::new();
    if ctx.mode == "parallel" || ctx.planner.single {
        windows = ctx.planner.fixed().await?;
        // the stream carries text only, and a window decoded with timestamps can
        // stop early and leave its tail to the next (see `tail`)
        let mut window_req = ctx.req.clone();
        if !ctx.planner.single {
            window_req.response_format = "json".to_string();
            window_req.timestamp_granularities = None;
        }
        let n = windows.len();
        let mut pending: VecDeque<OutStream> = VecDeque::new();
        let mut submitted = 0;
        // the final text joins windows by a space, so the stream sends one before
        // a window's first text: a silent window adds none
        let mut spoken = false;
        for i in 0..n {
            while submitted < n && (submitted <= i || pending.len() < share.per_upload()) {
                pending.push_back(submit(&ctx.st, ctx.adapter, &window_req, &windows[submitted], true)?);
                submitted += 1;
            }
            let mut stream = pending.pop_front().expect("window submitted");
            let (mut raw, mut emitted, mut gap) = (String::new(), String::new(), spoken);
            // holds back a character split across tokens until its last byte
            let mut decoder = crate::utf8::Utf8Stream::default();
            while let Some(item) = stream.next().await {
                match item {
                    Out::Error { status, message, .. } => {
                        emit(tx, format::error_event(&message, status)).await;
                        emit(tx, format::SSE_DONE.to_string()).await;
                        return Ok(());
                    }
                    Out::Chunk(c) if c.modality == "text" && !c.data.is_empty() => {
                        raw.push_str(&decoder.feed(&c.data));
                        let delta = stream_delta(ctx.adapter.stream_text(&raw), &mut emitted);
                        if !delta.is_empty() {
                            if gap {
                                emit(tx, format::delta_event(" ")).await;
                                gap = false;
                            }
                            emit(tx, format::delta_event(&delta)).await;
                        }
                    }
                    Out::Chunk(_) => {}
                }
            }
            let rest = decoder.flush();
            if !rest.is_empty() {
                raw.push_str(&rest);
                let delta = stream_delta(ctx.adapter.stream_text(&raw), &mut emitted);
                if !delta.is_empty() {
                    if gap {
                        emit(tx, format::delta_event(" ")).await;
                    }
                    emit(tx, format::delta_event(&delta)).await;
                }
            }
            windows[i].raw_text = raw;
            // as merge_windows, which joins only the windows with text
            spoken = spoken || !crate::adapters::py_strip(&emitted).is_empty();
        }
    } else {
        let mut spoken = false;
        let mut sequential = Sequential::new(&ctx);
        while let Some(window) = sequential.next(&mut ctx).await? {
            let text = window.transcript.as_ref().map(|t| t.text.clone()).unwrap_or_default();
            if !text.is_empty() {
                if spoken {
                    emit(tx, format::delta_event(" ")).await;
                }
                emit(tx, format::delta_event(&text)).await;
                spoken = true;
            }
            windows.push(window);
        }
    }
    let transcript = merge_windows(&windows, ctx.adapter, &ctx.req);
    emit(tx, format::done_event(&transcript.text, ctx.planner.duration)).await;
    emit(tx, format::SSE_DONE.to_string()).await;
    drop(share);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ladder_drops_the_prompt_then_warms_up_unless_pinned() {
        let req = TranscriptionRequest::default();
        let ladder = attempts(&req, Some("before".to_string()));
        assert_eq!(ladder[0], (0.0, Some("before".to_string())));
        assert_eq!(ladder[1], (0.0, None));
        assert_eq!(ladder.iter().map(|(t, _)| *t).collect::<Vec<_>>(), vec![0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
        let pinned = TranscriptionRequest { temperature: Some(0.3), ..TranscriptionRequest::default() };
        assert_eq!(attempts(&pinned, None), vec![(0.3, None)]);
    }

    #[test]
    fn carry_over_keeps_the_last_characters() {
        let texts = vec!["a".repeat(1000), String::new(), "é".repeat(500)];
        let carried = carry_over(&texts);
        assert_eq!(carried.chars().count(), CARRY_OVER_CHARS);
        assert!(carried.ends_with('é'));
    }
}
