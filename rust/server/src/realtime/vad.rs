//! Server turn detection for `/v1/realtime`: a port of `openai/realtime_vad.py`.
//!
//! [`TurnSegmenter`] cuts a 16 kHz stream into turns from a per-frame speech
//! probability and returns what to do in stream order: start a turn, add audio
//! to it, stop it, commit it. The probability is Silero VAD's, its ONNX export
//! run by ONNX Runtime on one CPU thread; `semantic_vad` adds Pipecat's Smart
//! Turn v3, which judges from the audio whether a pause ends the turn. Both run
//! through the ONNX Runtime library the Python side loads (`ort`, loaded at run
//! time from `MSTAR_REALTIME`'s `onnxruntime`), so both frontends score a frame
//! with the same runtime.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use ort::session::Session;
use ort::value::Tensor;
use serde_json::{json, Value};

pub const SAMPLE_RATE: usize = 16000;
/// Samples per decision: Silero's window at 16 kHz, 32 ms.
pub const FRAME: usize = 512;
/// Silence starts on a frame below threshold - HYSTERESIS; a frame at or above
/// threshold cancels it; one in between does neither (Silero's VADIterator).
const HYSTERESIS: f64 = 0.15;
const SEMANTIC_THRESHOLD: f64 = 0.5;
const SEMANTIC_PREFIX_MS: usize = 300;
/// Smart Turn hears the last 8 s of the turn.
/// The end-of-turn model hears the last 8 s of the turn.
const SMART_TURN_SECONDS: usize = 8;

pub fn samples_to_ms(n: i64) -> i64 {
    (n * 1000).div_euclid(SAMPLE_RATE as i64)
}

/// OpenAI's `server_vad` settings; `create_response` and `interrupt_response`
/// are accepted and echoed.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerVad {
    pub threshold: f64,
    pub prefix_padding_ms: i64,
    pub silence_duration_ms: i64,
    pub create_response: bool,
    pub interrupt_response: bool,
}

impl Default for ServerVad {
    fn default() -> Self {
        ServerVad { threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 500, create_response: true, interrupt_response: true }
    }
}

/// OpenAI's `semantic_vad`: a pause asks the end-of-turn model whether the
/// speaker is done; `eagerness` sets the pause, the confidence needed and the
/// longest silence that ends a turn regardless.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticVad {
    pub eagerness: String,
    pub create_response: bool,
    pub interrupt_response: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnDetection {
    Server(ServerVad),
    Semantic(SemanticVad),
}

impl TurnDetection {
    pub fn view(&self) -> Value {
        match self {
            TurnDetection::Server(v) => json!({
                "type": "server_vad", "threshold": v.threshold, "prefix_padding_ms": v.prefix_padding_ms,
                "silence_duration_ms": v.silence_duration_ms, "create_response": v.create_response,
                "interrupt_response": v.interrupt_response,
            }),
            TurnDetection::Semantic(v) => json!({
                "type": "semantic_vad", "eagerness": v.eagerness, "create_response": v.create_response,
                "interrupt_response": v.interrupt_response,
            }),
        }
    }
}

/// eagerness -> (pause before asking the model, P(end of turn) that commits,
/// silence that always ends the turn), in ms.
pub fn eagerness(name: &str) -> Option<(usize, f64, usize)> {
    match name {
        "low" => Some((400, 0.7, 8000)),
        "medium" | "auto" => Some((300, 0.5, 4000)),
        "high" => Some((200, 0.3, 2000)),
        _ => None,
    }
}

/// What the Python side knows of the turn detection models: where ONNX Runtime
/// and the weights are (`MSTAR_REALTIME`).
fn facts() -> &'static Value {
    static FACTS: OnceLock<Value> = OnceLock::new();
    FACTS.get_or_init(|| {
        std::env::var("MSTAR_REALTIME").ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null)
    })
}

/// The ONNX Runtime environment, loaded once from the Python side's library.
fn environment() -> Option<&'static ort::environment::Environment> {
    static ENV: OnceLock<Option<ort::environment::Environment>> = OnceLock::new();
    ENV.get_or_init(|| {
        let lib = facts().get("onnxruntime").and_then(Value::as_str)?;
        match ort::init_from(lib).map_err(|e| e.to_string()).and_then(|b| b.build().map_err(|e| e.to_string())) {
            Ok(env) => Some(env),
            Err(e) => {
                eprintln!("WARNING: server_vad is unavailable: cannot load ONNX Runtime from {lib}: {e}");
                None
            }
        }
    })
    .as_ref()
}

fn session(path: &str) -> Result<Session, String> {
    let env = environment().ok_or("ONNX Runtime is not loaded")?;
    // one thread: a 32 ms frame is too small to split
    let builder = Session::builder(env).map_err(|e| e.to_string())?;
    let builder = builder.with_intra_threads(1).map_err(|e| e.to_string())?;
    let mut builder = builder.with_inter_threads(1).map_err(|e| e.to_string())?;
    builder.commit_from_file(path).map_err(|e| e.to_string())
}

/// The input names and element types of a session, for the checks that a file
/// is the model the server runs.
fn inputs_of(session: &Session) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = session
        .inputs()
        .iter()
        .map(|i| {
            let ty = match i.dtype() {
                ort::value::ValueType::Tensor { ty, .. } => format!("tensor({})", format!("{ty:?}").to_lowercase().replace("float32", "float")),
                other => format!("{other:?}"),
            };
            (i.name().to_string(), ty)
        })
        .collect();
    v.sort();
    v
}

const STATE_LEN: usize = 2 * 128;
const CONTEXT: usize = 64;

/// The Silero session, shared by every stream behind a lock (a frame is about
/// 0.15 ms); `None` with one warning when the weights or the runtime are missing.
fn silero() -> Option<&'static Mutex<Session>> {
    static SILERO: OnceLock<Option<Mutex<Session>>> = OnceLock::new();
    SILERO
        .get_or_init(|| {
            let Some(path) = facts().get("silero").and_then(Value::as_str) else {
                eprintln!("WARNING: server_vad is unavailable: no Silero VAD weights (set MSTAR_SILERO_VAD_PATH or install mstar-ai[vad])");
                return None;
            };
            match load_silero(path) {
                Ok(s) => Some(Mutex::new(s)),
                Err(why) => {
                    eprintln!("WARNING: server_vad is unavailable: {why}");
                    None
                }
            }
        })
        .as_ref()
}

/// The Silero session from `path`, or why it is not the export this server runs.
fn load_silero(path: &str) -> Result<Session, String> {
    let mut s = session(path).map_err(|e| format!("cannot load Silero VAD from {path}: {e}"))?;
    let want = vec![
        ("input".to_string(), "tensor(float)".to_string()),
        ("sr".to_string(), "tensor(int64)".to_string()),
        ("state".to_string(), "tensor(float)".to_string()),
    ];
    if inputs_of(&s) != want {
        return Err(format!("{path} is not the Silero VAD export this server runs (silero_vad.onnx)"));
    }
    if score(&mut s, &vec![0.0; CONTEXT + FRAME], &vec![0.0; STATE_LEN]).is_err() {
        return Err(format!("{path} cannot score a silent frame"));
    }
    Ok(s)
}

/// One Silero run: the speech probability and the new recurrent state.
fn score(session: &mut Session, x: &[f32], state: &[f32]) -> Result<(f64, Vec<f32>), String> {
    let mut run = || -> ort::Result<(f64, Vec<f32>)> {
        let outputs = session.run(ort::inputs![
            "input" => Tensor::from_array((vec![1i64, x.len() as i64], x.to_vec()))?,
            "state" => Tensor::from_array((vec![2i64, 1, 128], state.to_vec()))?,
            "sr" => Tensor::from_array((Vec::<i64>::new(), vec![SAMPLE_RATE as i64]))?,
        ])?;
        let (_, p) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, s) = outputs[1].try_extract_tensor::<f32>()?;
        Ok((p[0] as f64, s.to_vec()))
    };
    let (p, s) = run().map_err(|e| e.to_string())?;
    if s.len() != STATE_LEN {
        return Err(format!("its state output has {} values, not {STATE_LEN}", s.len()));
    }
    Ok((p, s))
}

/// Whether `server_vad` can run here.
pub fn silero_available() -> bool {
    silero().is_some()
}

/// Silero VAD for one stream: the model's recurrent state and the last
/// `CONTEXT` samples, which it reads before each frame.
pub struct Silero {
    state: Vec<f32>,
    context: Vec<f32>,
}

impl Silero {
    pub fn new() -> Silero {
        Silero { state: vec![0.0; STATE_LEN], context: vec![0.0; CONTEXT] }
    }
}

/// A stream's speech probability, one `FRAME` at a time: Silero in the server,
/// a script in the tests.
trait SpeechProbability: Send {
    fn prob(&mut self, frame: &[f32]) -> f64;
    fn reset(&mut self);
}

impl SpeechProbability for Silero {
    fn prob(&mut self, frame: &[f32]) -> f64 {
        let mut x = self.context.clone();
        x.extend_from_slice(frame);
        let session = silero().expect("server_vad runs only with Silero loaded");
        let (p, state) = score(&mut session.lock().unwrap(), &x, &self.state).unwrap_or((0.0, self.state.clone()));
        self.state = state;
        self.context = x[x.len() - CONTEXT..].to_vec();
        p
    }

    fn reset(&mut self) {
        self.state = vec![0.0; STATE_LEN];
        self.context = vec![0.0; CONTEXT];
    }
}

// ── semantic_vad's end-of-turn model ────────────────────────────────────────

/// `P(the speaker has finished)` from a turn's last audio: the API server's
/// Smart Turn, asked over the bridge (see `Bridge::end_of_turn`).
pub type EndOfTurn = std::sync::Arc<dyn Fn(&[f32]) -> Result<f64, String> + Send + Sync>;

/// Whether the API server loaded Smart Turn, and if not, why, in words safe to
/// send a client.
pub fn smart_turn_problem() -> Option<String> {
    match facts().get("smart_turn_problem") {
        Some(Value::String(why)) => Some(why.clone()),
        Some(Value::Null) => None,
        _ => Some("no Smart Turn model is loaded".to_string()),
    }
}

// ── the segmenter ───────────────────────────────────────────────────────────

/// What a feed asks the session to do, in stream order.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    SpeechStarted(i64),
    Audio(Vec<f32>),
    SpeechStopped(i64),
    Commit,
    EndOfTurnFailed(String),
}

/// `TurnSegmenter`: splits a 16 kHz stream into turns. A turn starts on a frame
/// at or above `threshold` with up to `prefix_padding_ms` of the audio before
/// it, and ends after the frame that brings `silence_duration_ms` of whole
/// frames, counted from a frame below `threshold - HYSTERESIS` with none at or
/// above `threshold` since. With `semantic_vad`, each pause that reaches the
/// eagerness' pause asks the end-of-turn model once over the turn's last 8 s.
/// A turn holds at most `max_turn` samples. A partial frame waits for the next feed.
pub struct TurnSegmenter {
    vad: Box<dyn SpeechProbability>,
    max_turn: usize,
    threshold: f64,
    neg: f64,
    prefix_cap: usize,
    silence_cap: usize,
    eot: Option<EndOfTurn>,
    eot_threshold: f64,
    pause_cap: usize,
    recent: VecDeque<Vec<f32>>,
    recent_cap: usize,
    rest: Vec<f32>,
    pos: i64,
    prefix: Vec<f32>,
    pub speaking: bool,
    silence: usize,
    turn: usize,
}

impl TurnSegmenter {
    pub fn new(cfg: &TurnDetection, max_turn: usize, pos: i64, end_of_turn: Option<EndOfTurn>) -> TurnSegmenter {
        Self::with_vad(Box::new(Silero::new()), cfg, max_turn, pos, end_of_turn)
    }

    fn with_vad(
        vad: Box<dyn SpeechProbability>,
        cfg: &TurnDetection,
        max_turn: usize,
        pos: i64,
        end_of_turn: Option<EndOfTurn>,
    ) -> TurnSegmenter {
        let (threshold, prefix_ms, silence_ms, eot, eot_threshold, pause_ms) = match cfg {
            TurnDetection::Semantic(s) => {
                let (pause, p, silence) = eagerness(&s.eagerness).unwrap_or((300, 0.5, 4000));
                (SEMANTIC_THRESHOLD, SEMANTIC_PREFIX_MS, silence, end_of_turn, p, pause)
            }
            TurnDetection::Server(v) => (v.threshold, v.prefix_padding_ms as usize, v.silence_duration_ms as usize, None, 0.0, 0),
        };
        TurnSegmenter {
            vad,
            max_turn,
            threshold,
            neg: (threshold - HYSTERESIS).max(0.01),
            prefix_cap: prefix_ms * SAMPLE_RATE / 1000,
            silence_cap: silence_ms * SAMPLE_RATE / 1000,
            eot,
            eot_threshold,
            pause_cap: pause_ms * SAMPLE_RATE / 1000,
            recent: VecDeque::new(),
            recent_cap: SMART_TURN_SECONDS * SAMPLE_RATE / FRAME + 1,
            rest: Vec::new(),
            pos,
            prefix: Vec::new(),
            speaking: false,
            silence: 0,
            turn: 0,
        }
    }

    fn remember(&mut self, piece: Vec<f32>) {
        if self.recent.len() == self.recent_cap {
            self.recent.pop_front();
        }
        self.recent.push_back(piece);
    }

    pub fn feed(&mut self, audio: &[f32]) -> Vec<Op> {
        let mut ops = Vec::new();
        let mut buf = std::mem::take(&mut self.rest);
        buf.extend_from_slice(audio);
        let whole = buf.len() - buf.len() % FRAME;
        for i in (0..whole).step_by(FRAME) {
            self.frame(&buf[i..i + FRAME], self.pos + i as i64, &mut ops);
        }
        self.rest = buf[whole..].to_vec();
        self.pos += whole as i64;
        ops
    }

    fn frame(&mut self, frame: &[f32], start: i64, ops: &mut Vec<Op>) {
        let p = self.vad.prob(frame);
        if self.speaking && self.turn + FRAME > self.max_turn {
            // the item cap: the turn ends before this frame; in mid-speech the
            // next turn starts with it, whatever its probability
            let mid_speech = self.silence == 0;
            self.stop(start, ops);
            if mid_speech {
                self.open(start, ops);
            }
        }
        if !self.speaking {
            if p < self.threshold {
                if self.prefix_cap > 0 {
                    self.prefix.extend_from_slice(frame);
                    let excess = self.prefix.len().saturating_sub(self.prefix_cap);
                    self.prefix.drain(..excess);
                }
                return;
            }
            self.open(start, ops);
        }
        ops.push(Op::Audio(frame.to_vec()));
        self.turn += FRAME;
        let paused = self.silence;
        if p >= self.threshold {
            self.silence = 0;
        } else if p < self.neg || self.silence > 0 {
            self.silence += FRAME;
        }
        // at least one silent frame, so silence_duration_ms 0 does not end a turn mid-speech
        let mut done = self.silence > 0 && self.silence >= self.silence_cap;
        if self.eot.is_some() {
            self.remember(frame.to_vec());
            if !done && paused < self.pause_cap && self.pause_cap <= self.silence {
                done = self.ask_end_of_turn(ops);
            }
        }
        if done {
            self.stop(start + FRAME as i64, ops);
        }
    }

    /// Whether the end-of-turn model says the turn is over; if it fails, the
    /// failure goes out as an operation and this segmenter stops asking.
    fn ask_end_of_turn(&mut self, ops: &mut Vec<Op>) -> bool {
        let audio: Vec<f32> = self.recent.iter().flatten().copied().collect();
        match (self.eot.as_ref().unwrap())(&audio) {
            Ok(p) => p >= self.eot_threshold,
            Err(e) => {
                ops.push(Op::EndOfTurnFailed(e));
                self.eot = None;
                self.recent.clear();
                false
            }
        }
    }

    fn open(&mut self, start: i64, ops: &mut Vec<Op>) {
        let keep = self.prefix.len().min(self.max_turn.saturating_sub(FRAME));
        let prefix = self.prefix[self.prefix.len() - keep..].to_vec();
        self.speaking = true;
        self.turn = prefix.len();
        self.silence = 0;
        self.prefix.clear();
        ops.push(Op::SpeechStarted(samples_to_ms(start - prefix.len() as i64)));
        if !prefix.is_empty() {
            ops.push(Op::Audio(prefix.clone()));
        }
        if self.eot.is_some() {
            self.recent.clear();
            self.remember(prefix);
        }
    }

    fn stop(&mut self, end: i64, ops: &mut Vec<Op>) {
        ops.push(Op::SpeechStopped(samples_to_ms(end)));
        ops.push(Op::Commit);
        self.speaking = false;
        self.silence = 0;
        self.turn = 0;
        if self.eot.is_some() {
            self.recent.clear();
        }
    }

    /// A client commit: an open turn takes the partial frame and stops there;
    /// between turns the partial frame and the prefix padding are dropped.
    pub fn end_turn(&mut self) -> Vec<Op> {
        let mut ops = Vec::new();
        let rest = std::mem::take(&mut self.rest);
        if self.speaking {
            if !rest.is_empty() && self.turn + rest.len() > self.max_turn {
                self.stop(self.pos, &mut ops);
                self.open(self.pos, &mut ops);
            }
            if !rest.is_empty() {
                ops.push(Op::Audio(rest.clone()));
            }
            self.pos += rest.len() as i64;
            self.stop(self.pos, &mut ops);
            ops.pop(); // the client's commit is the commit
        } else {
            self.pos += rest.len() as i64;
            self.prefix.clear();
        }
        ops
    }

    /// A clear: drop buffered audio, keep the timeline (`skipped` more samples
    /// were dropped before they reached the segmenter).
    pub fn reset(&mut self, skipped: usize) {
        self.pos += (self.rest.len() + skipped) as i64;
        self.rest.clear();
        self.prefix.clear();
        self.speaking = false;
        self.silence = 0;
        self.turn = 0;
        if self.eot.is_some() {
            self.recent.clear();
        }
        self.vad.reset();
    }

    pub fn shift(&mut self, n: i64) {
        self.pos += n;
    }

    /// Go on from `old` between turns (a settings change): its position, its
    /// partial frame, not scored yet, and its prefix padding.
    pub fn take_over(&mut self, old: TurnSegmenter) {
        self.pos = old.pos;
        self.rest = old.rest;
        let keep = old.prefix.len().min(self.prefix_cap);
        self.prefix = old.prefix[old.prefix.len() - keep..].to_vec();
    }

    pub fn unscored(&self) -> &[f32] {
        &self.rest
    }
}

#[cfg(test)]
mod tests {
    //! Segmentation edge cases, ported from the Python segmenter's tests with
    //! the same inputs and expected outputs, and a differential check against
    //! that segmenter's recorded outputs. A scripted or energy-based speech
    //! probability stands in for Silero, so the turns are exact.

    use super::*;
    use crate::realtime::resample::StreamResampler;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// The session's item cap the Python tests ran with.
    const CAP: usize = 300 * SAMPLE_RATE;

    /// Probabilities from a list, one per frame, the last repeating. Clones
    /// share the position, so a settings change goes on where the old
    /// segmenter stopped.
    #[derive(Clone)]
    struct Scripted {
        probs: Arc<Vec<f64>>,
        next: Arc<AtomicUsize>,
    }

    impl Scripted {
        fn new(probs: Vec<f64>) -> Scripted {
            Scripted { probs: Arc::new(probs), next: Arc::new(AtomicUsize::new(0)) }
        }
    }

    impl SpeechProbability for Scripted {
        fn prob(&mut self, frame: &[f32]) -> f64 {
            assert_eq!(frame.len(), FRAME);
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            self.probs[i.min(self.probs.len() - 1)]
        }

        fn reset(&mut self) {}
    }

    /// 1.0 for a frame with speech-level energy, else 0.0.
    struct Energy;

    impl SpeechProbability for Energy {
        fn prob(&mut self, frame: &[f32]) -> f64 {
            assert_eq!(frame.len(), FRAME);
            let rms = (frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / FRAME as f64).sqrt();
            if rms > 0.05 { 1.0 } else { 0.0 }
        }

        fn reset(&mut self) {}
    }

    /// `frames` frames' worth of a 220 Hz int16 tone at `rate`, as decoded from PCM.
    fn speech(frames: usize, rate: usize) -> Vec<f32> {
        let n = frames * FRAME * rate / SAMPLE_RATE;
        (0..n)
            .map(|i| ((2.0 * std::f64::consts::PI * 220.0 * i as f64 / rate as f64).sin() * 8000.0) as i16 as f32 / 32768.0)
            .collect()
    }

    fn silence(frames: usize, rate: usize) -> Vec<f32> {
        vec![0.0; frames * FRAME * rate / SAMPLE_RATE]
    }

    fn server(v: ServerVad) -> TurnDetection {
        TurnDetection::Server(v)
    }

    fn no_prefix() -> ServerVad {
        ServerVad { prefix_padding_ms: 0, ..ServerVad::default() }
    }

    fn semantic(eagerness: &str) -> TurnDetection {
        TurnDetection::Semantic(SemanticVad { eagerness: eagerness.into(), create_response: true, interrupt_response: true })
    }

    fn energy(td: &TurnDetection, end_of_turn: Option<EndOfTurn>) -> TurnSegmenter {
        TurnSegmenter::with_vad(Box::new(Energy), td, CAP, 0, end_of_turn)
    }

    /// An end-of-turn model answering `p` (failing when `None`), and the
    /// length of the audio each call heard.
    fn end_of_turn(p: Option<f64>) -> (EndOfTurn, Arc<Mutex<Vec<usize>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let seen = calls.clone();
        let model: EndOfTurn = Arc::new(move |audio: &[f32]| {
            seen.lock().unwrap().push(audio.len());
            p.ok_or_else(|| "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION".to_string())
        });
        (model, calls)
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Turn {
        start: i64,
        end: i64,
        audio: Vec<f32>,
    }

    /// What a session makes of the operations: committed turns, the open one,
    /// every start, and end-of-turn failures. Asserts each transition comes
    /// once, in order, with times that never go back.
    #[derive(Default)]
    struct Timeline {
        turns: Vec<Turn>,
        open: Option<(i64, Option<i64>, Vec<f32>)>,
        starts: Vec<i64>,
        failures: Vec<String>,
    }

    impl Timeline {
        fn apply(&mut self, ops: Vec<Op>) {
            for op in ops {
                match op {
                    Op::SpeechStarted(ms) => {
                        assert!(self.open.is_none(), "a start inside a turn");
                        assert!(self.turns.last().is_none_or(|t| t.end <= ms), "a start at {ms} before the last end");
                        self.starts.push(ms);
                        self.open = Some((ms, None, Vec::new()));
                    }
                    Op::Audio(a) => {
                        let (_, end, audio) = self.open.as_mut().expect("audio outside a turn");
                        assert!(end.is_none(), "audio after a stop");
                        audio.extend(a);
                    }
                    Op::SpeechStopped(ms) => {
                        let (start, end, _) = self.open.as_mut().expect("a stop outside a turn");
                        assert!(end.is_none() && *start < ms, "a stop at {ms} for a turn from {start}");
                        *end = Some(ms);
                    }
                    Op::Commit => self.close(),
                    Op::EndOfTurnFailed(e) => self.failures.push(e),
                }
            }
        }

        fn close(&mut self) {
            let (start, end, audio) = self.open.take().expect("a commit outside a turn");
            self.turns.push(Turn { start, end: end.expect("a commit before the stop"), audio });
        }

        /// A client commit: the segmenter stops an open turn and the client's
        /// commit commits it.
        fn client_commit(&mut self, seg: &mut TurnSegmenter) {
            self.apply(seg.end_turn());
            if matches!(self.open, Some((_, Some(_), _))) {
                self.close();
            }
        }

        fn feed(&mut self, seg: &mut TurnSegmenter, audio: &[f32], step: usize) {
            for chunk in audio.chunks(step) {
                self.apply(seg.feed(chunk));
            }
        }

        fn spans(&self) -> Vec<(i64, i64)> {
            self.turns.iter().map(|t| (t.start, t.end)).collect()
        }

        /// Each item is exactly the 16 kHz stream between its start and end.
        fn assert_items_are_the_stream(&self, stream: &[f32]) {
            for t in &self.turns {
                let want = &stream[t.start as usize * 16..t.end as usize * 16];
                assert!(t.audio == want, "the item {}..{} ms is not that span of the stream", t.start, t.end);
            }
        }

        /// At another input rate, each item holds 16 samples a millisecond.
        fn assert_items_fill_their_spans(&self) {
            for t in &self.turns {
                assert_eq!(t.audio.len() as i64, (t.end - t.start) * 16, "the item {}..{} ms", t.start, t.end);
            }
        }
    }

    /// The non-audio operations, by name.
    fn kinds(ops: &[Op]) -> Vec<&'static str> {
        ops.iter()
            .filter_map(|op| match op {
                Op::SpeechStarted(_) => Some("started"),
                Op::SpeechStopped(_) => Some("stopped"),
                Op::Commit => Some("commit"),
                Op::EndOfTurnFailed(_) => Some("failed"),
                Op::Audio(_) => None,
            })
            .collect()
    }

    fn cat(parts: &[Vec<f32>]) -> Vec<f32> {
        parts.concat()
    }

    /// Input at `rate`, resampled to 16 kHz as the session resamples it.
    struct Input(Option<StreamResampler>);

    impl Input {
        fn at(rate: usize) -> Input {
            Input((rate != SAMPLE_RATE).then(|| StreamResampler::new(rate, SAMPLE_RATE)))
        }

        fn push(&mut self, audio: &[f32]) -> Vec<f32> {
            self.0.as_mut().map_or_else(|| audio.to_vec(), |r| r.push(audio))
        }
    }

    // ── server_vad ──────────────────────────────────────────────────────────

    #[test]
    fn one_append_with_speech_silence_speech() {
        // the turn end falls inside the append; the second utterance stays and becomes the next turn
        let stream = cat(&[speech(1, 16000), silence(16, 16000), speech(1, 16000)]);
        let mut seg = energy(&server(ServerVad::default()), None);
        let mut tl = Timeline::default();
        let ops = seg.feed(&stream);
        assert_eq!(kinds(&ops), ["started", "stopped", "commit", "started"]);
        tl.apply(ops);
        tl.client_commit(&mut seg);
        assert_eq!(tl.spans(), [(0, 544), (544, 576)]);
        tl.assert_items_are_the_stream(&stream);
    }

    #[test]
    fn end_of_turn_inside_an_append() {
        // 100 ms appends: the sixteenth silent frame completes mid-append and
        // the next utterance starts in the same append
        let stream = cat(&[speech(10, 16000), silence(16, 16000), speech(10, 16000), silence(16, 16000)]);
        let mut seg = energy(&server(no_prefix()), None);
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &stream, 1600);
        assert_eq!(tl.spans(), [(0, 832), (832, 1664)]);
        tl.assert_items_are_the_stream(&stream);
    }

    #[test]
    fn silence_window_boundary() {
        for (silence_ms, gap, ends) in [
            // 512 ms of silent frames passes all three
            (499, 16, true),
            (500, 16, true),
            (501, 16, true),
            // 480 ms does not
            (500, 15, false),
            // exactly the window ends the turn; 1 ms short does not
            (512, 16, true),
            (513, 16, false),
        ] {
            let stream = cat(&[speech(4, 16000), silence(gap, 16000), speech(4, 16000)]);
            let mut seg = energy(&server(ServerVad { silence_duration_ms: silence_ms, ..no_prefix() }), None);
            let mut tl = Timeline::default();
            tl.feed(&mut seg, &stream, stream.len());
            let case = format!("silence_duration_ms {silence_ms}, {gap} silent frames");
            if ends {
                assert_eq!(tl.spans(), [(0, (4 + 16) * 32)], "{case}");
                assert_eq!(tl.starts, [0, (4 + gap as i64) * 32], "{case}");
            } else {
                assert!(tl.turns.is_empty(), "{case}");
                assert_eq!(tl.starts, [0], "{case}");
            }
            tl.assert_items_are_the_stream(&stream);
        }
    }

    #[test]
    fn zero_silence_window_ends_a_turn_on_its_first_silent_frame() {
        let stream = cat(&[speech(10, 16000), silence(3, 16000)]);
        let mut seg = energy(&server(ServerVad { silence_duration_ms: 0, ..no_prefix() }), None);
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &stream, stream.len());
        assert_eq!(tl.spans(), [(0, 11 * 32)]);
        tl.assert_items_are_the_stream(&stream);
    }

    #[test]
    fn back_to_back_turns_in_10ms_appends() {
        let stream: Vec<f32> = (0..3).flat_map(|k| cat(&[speech(6 + 3 * k, 16000), silence(20, 16000)])).collect();
        let mut seg = energy(&server(ServerVad::default()), None);
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &stream, 160);
        // the 4 silent frames after each commit are the next turn's prefix padding
        let starts: Vec<i64> = tl.turns.iter().map(|t| t.start).collect();
        assert_eq!(starts, [0, (6 + 16) * 32, (6 + 20 + 9 + 16) * 32]);
        tl.assert_items_are_the_stream(&stream);
    }

    #[test]
    fn a_turn_open_at_the_end_is_never_stopped() {
        let mut seg = energy(&server(ServerVad::default()), None);
        let mut ops = Vec::new();
        for chunk in speech(10, 16000).chunks(1600) {
            ops.extend(seg.feed(chunk));
        }
        assert_eq!(kinds(&ops), ["started"]);
    }

    #[test]
    fn a_client_commit_mid_turn_ends_it_with_the_partial_frame() {
        let stream = cat(&[speech(10, 16000), speech(1, 16000)[..100].to_vec()]);
        let mut seg = energy(&server(ServerVad::default()), None);
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &stream, stream.len());
        tl.client_commit(&mut seg);
        assert_eq!(tl.spans(), [(0, 5220 * 1000 / 16000)]);
        assert_eq!(tl.turns[0].audio, stream);
    }

    #[test]
    fn a_turn_reaching_the_item_cap_is_committed_and_speech_goes_on() {
        // 300 s of unbroken speech: the turn ends at the cap, and the speech after it starts the next turn
        let piece = speech(625, 16000); // 20 s
        let mut seg = energy(&server(ServerVad::default()), None);
        let mut tl = Timeline::default();
        for _ in 0..15 {
            tl.feed(&mut seg, &piece, piece.len());
        }
        tl.feed(&mut seg, &speech(63, 16000), 63 * FRAME); // 2.016 s more
        tl.client_commit(&mut seg);
        assert_eq!(tl.spans(), [(0, 300_000), (300_000, 302_016)]);
        let sizes: Vec<usize> = tl.turns.iter().map(|t| t.audio.len()).collect();
        assert_eq!(sizes, [300 * 16000, 63 * FRAME]);
    }

    #[test]
    fn input_at_24khz_keeps_the_timeline() {
        // resampled before segmentation, with the same times in ms as at 16 kHz;
        // the resampler holds its last 1 ms, so the stream goes on one frame past the second turn
        let r = 24000;
        let stream = cat(&[speech(10, r), silence(16, r), speech(10, r), silence(17, r)]);
        let mut input = Input::at(r);
        let mut seg = energy(&server(no_prefix()), None);
        let mut tl = Timeline::default();
        for chunk in stream.chunks(240) {
            tl.apply(seg.feed(&input.push(chunk)));
        }
        assert_eq!(tl.spans(), [(0, 832), (832, 1664)]);
        tl.assert_items_fill_their_spans();
    }

    #[test]
    fn a_clear_drops_the_open_turn_and_keeps_the_timeline() {
        let mut seg = energy(&server(no_prefix()), None);
        let ops = seg.feed(&cat(&[speech(5, 16000), speech(1, 16000)[..200].to_vec()]));
        assert_eq!(kinds(&ops), ["started"]);
        seg.reset(0);
        assert!(!seg.speaking);
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &cat(&[silence(3, 16000), speech(2, 16000), silence(16, 16000)]), FRAME * 21);
        // cleared audio still counts on the timeline: 5 frames + 200 samples, then 3 silent frames
        let starts: Vec<i64> = tl.turns.iter().map(|t| t.start).collect();
        assert_eq!(starts, [(5 * FRAME as i64 + 200) * 1000 / 16000 + 3 * 32]);
    }

    /// Feeds one frame per probability, of samples that are each distinct.
    fn scripted(probs: Vec<f64>, td: &TurnDetection, max_turn: usize) -> (Timeline, Vec<f32>) {
        let stream: Vec<f32> = (0..probs.len() * FRAME).map(|i| i as f32).collect();
        let mut seg = TurnSegmenter::with_vad(Box::new(Scripted::new(probs)), td, max_turn, 0, None);
        let mut tl = Timeline::default();
        tl.apply(seg.feed(&stream));
        (tl, stream)
    }

    #[test]
    fn the_item_cap_is_never_passed_and_speech_across_it_is_kept() {
        // a 3 s cap is not a whole number of frames, and the prefix is not
        // either: no item passes the cap, and speech going on across a cap
        // split, even at a probability between the two thresholds, starts the
        // next turn right there, so every sample from the first start on is in one item
        for prefix_ms in [0, 300] {
            let probs = [vec![0.0; 20], vec![0.9; 90], vec![0.4; 20], vec![0.9; 200], vec![0.0; 20]].concat();
            let td = server(ServerVad { prefix_padding_ms: prefix_ms, ..ServerVad::default() });
            let (tl, stream) = scripted(probs, &td, 3 * 16000);
            assert!(tl.open.is_none());
            assert!(tl.turns.len() >= 3 && tl.turns.iter().all(|t| t.audio.len() <= 3 * 16000), "prefix {prefix_ms}");
            let first = 20 * FRAME - prefix_ms as usize * 16;
            let joined: Vec<f32> = tl.turns.iter().flat_map(|t| t.audio.clone()).collect();
            assert!(joined == stream[first..first + joined.len()], "prefix {prefix_ms}");
            for pair in tl.turns.windows(2) {
                assert_eq!(pair[1].start, pair[0].end, "prefix {prefix_ms}");
            }
        }
    }

    #[test]
    fn borderline_frames_do_not_hold_a_started_silence_open() {
        // silence starts below threshold - 0.15 and only a frame at or above
        // the threshold cancels it: frames alternating 0.1 and 0.4 end the turn
        // on the frame that completes the window from the first 0.1, and a 0.6
        // frame starts the count over
        let alternating = |n: usize| [0.1, 0.4].repeat(n);
        let (tl, _) = scripted([vec![0.9; 10], alternating(30)].concat(), &server(no_prefix()), CAP);
        assert_eq!(tl.turns.iter().map(|t| t.end).collect::<Vec<_>>(), [(10 + 16) * 32]);
        let probs = [vec![0.9; 10], alternating(4), vec![0.6], alternating(30)].concat();
        let (tl, _) = scripted(probs, &server(no_prefix()), CAP);
        assert_eq!(tl.turns.iter().map(|t| t.end).collect::<Vec<_>>(), [(10 + 8 + 1 + 16) * 32]);
    }

    #[test]
    fn a_settings_change_keeps_audio_not_scored_yet() {
        // a partial frame waiting for its decision survives a turn_detection
        // change: into the new segmenter, or into the item when it goes to null
        let stream = cat(&[speech(3, 16000), silence(16, 16000)]);
        let mut old = energy(&server(no_prefix()), None);
        assert!(old.feed(&stream[..511]).is_empty());
        let mut seg = energy(&server(ServerVad { threshold: 0.6, ..no_prefix() }), None);
        seg.take_over(old);
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &stream[511..], stream.len());
        assert_eq!(tl.spans(), [(0, 19 * 32)]);
        tl.assert_items_are_the_stream(&stream);
        assert!(seg.feed(&stream[..300]).is_empty());
        assert_eq!(seg.unscored(), &stream[..300]);
    }

    #[test]
    fn a_settings_change_between_turns_goes_on_at_both_rates() {
        // what the resampler holds back for its lookahead is audio after the
        // turn, and goes on to the new segmenter
        for rate in [16000, 24000] {
            let utterance = cat(&[speech(3, rate), silence(17, rate)]);
            let mut input = Input::at(rate);
            let mut seg = energy(&server(no_prefix()), None);
            let mut tl = Timeline::default();
            tl.apply(seg.feed(&input.push(&utterance)));
            assert_eq!(tl.turns.len(), 1, "{rate} Hz");
            assert!(!seg.speaking);
            let mut new = energy(&server(ServerVad { threshold: 0.6, ..no_prefix() }), None);
            new.take_over(seg);
            tl.apply(new.feed(&input.push(&utterance)));
            assert_eq!(tl.spans(), [(0, 19 * 32), (20 * 32, 39 * 32)], "{rate} Hz");
            if rate == 16000 {
                tl.assert_items_are_the_stream(&utterance.repeat(2));
            } else {
                tl.assert_items_fill_their_spans();
            }
        }
    }

    #[test]
    fn a_drained_tail_that_completes_a_turn_commits_it_once() {
        // at 24 kHz the resampler holds its last 1 ms, so the tail a client
        // commit drains is what completes the silence window: the turn's commit
        // comes last and end_turn adds nothing, so the session sends no second one
        let mut input = Input::at(24000);
        let mut seg = energy(&server(no_prefix()), None);
        let ops = seg.feed(&input.push(&cat(&[speech(1, 24000), silence(16, 24000)])));
        assert_eq!(kinds(&ops), ["started"]);
        let mut tl = Timeline::default();
        tl.apply(ops);
        let mut ops = seg.feed(&input.0.as_mut().unwrap().flush());
        ops.extend(seg.end_turn());
        assert_eq!(kinds(&ops), ["stopped", "commit"]);
        assert_eq!(ops.last(), Some(&Op::Commit));
        tl.apply(ops);
        assert_eq!(tl.turns.iter().map(|t| t.audio.len()).collect::<Vec<_>>(), [17 * FRAME]);
    }

    #[test]
    fn a_client_commit_between_turns_keeps_earlier_audio_out_of_the_next_turn() {
        // audio not in a turn when the client committed (a partial frame, the
        // prefix padding) does not leak into the next turn
        let stream = cat(&[speech(1, 16000)[..511].to_vec(), speech(3, 16000), silence(16, 16000)]);
        let mut seg = energy(&server(ServerVad::default()), None);
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &stream[..511], 511);
        tl.client_commit(&mut seg);
        tl.feed(&mut seg, &stream[511..], stream.len());
        assert_eq!(tl.starts, [511 * 1000 / 16000]);
        assert_eq!(tl.turns.iter().map(|t| t.audio.len()).collect::<Vec<_>>(), [stream.len() - 511]);

        // whole silent frames held as prefix padding are dropped too
        let mut seg = energy(&server(ServerVad::default()), None);
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &silence(3, 16000), FRAME);
        tl.client_commit(&mut seg);
        tl.feed(&mut seg, &cat(&[speech(3, 16000), silence(16, 16000)]), FRAME);
        assert_eq!(tl.spans(), [(3 * 32, 22 * 32)]);
    }

    // ── semantic_vad ────────────────────────────────────────────────────────

    #[test]
    fn semantic_vad_commits_at_a_pause_the_model_calls_final() {
        // P(end of turn) 0.9 at the 300 ms pause ends the turn there, well before server_vad's 500 ms window
        let (model, calls) = end_of_turn(Some(0.9));
        let stream = cat(&[speech(10, 16000), silence(40, 16000), speech(5, 16000), silence(20, 16000)]);
        let mut seg = energy(&semantic("medium"), Some(model));
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &stream, 1600);
        // the pause reaches 300 ms on the 10th silent frame (320 ms); the next turn has 300 ms of prefix
        assert_eq!(tl.spans(), [(0, 20 * 32), (50 * 32 - 300, 65 * 32)]);
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], 20 * FRAME);
        tl.assert_items_are_the_stream(&stream);
    }

    #[test]
    fn semantic_vad_keeps_listening_when_the_model_says_not_done() {
        // the speech after the pause is the same turn, and medium's 4 s silence limit ends it
        let (model, calls) = end_of_turn(Some(0.1));
        let stream = cat(&[speech(10, 16000), silence(20, 16000), speech(10, 16000), silence(130, 16000)]);
        let mut seg = energy(&semantic("medium"), Some(model));
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &stream, 1600);
        assert_eq!(tl.spans(), [(0, (10 + 20 + 10 + 125) * 32)]);
        assert_eq!(calls.lock().unwrap().len(), 2, "once per pause, not once per frame");
        tl.assert_items_are_the_stream(&stream);
    }

    #[test]
    fn semantic_vad_goes_on_after_an_end_of_turn_model_error() {
        // the failure is one operation, the model is not asked again, and
        // turns end at medium's 4 s silence limit
        let (model, calls) = end_of_turn(None);
        let stream = cat(&[speech(10, 16000), silence(130, 16000), speech(10, 16000), silence(130, 16000)]);
        let mut seg = energy(&semantic("medium"), Some(model));
        let mut tl = Timeline::default();
        tl.feed(&mut seg, &stream, 1600);
        assert_eq!(tl.failures, ["[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION"]);
        // the second turn's prefix is the 5 silent frames left after the first one ended
        assert_eq!(tl.spans(), [(0, (10 + 125) * 32), ((10 + 125) * 32, (150 + 125) * 32)]);
        assert_eq!(calls.lock().unwrap().len(), 1);
        tl.assert_items_are_the_stream(&stream);
    }

    #[test]
    fn semantic_vad_eagerness() {
        for (eagerness, end_frames) in [
            ("high", 7),   // 0.6 >= 0.3: ends at the 200 ms pause (7 frames, 224 ms)
            ("medium", 10), // 0.6 >= 0.5: ends at the 300 ms pause (10 frames)
            ("auto", 10),
            ("low", 250), // 0.6 < 0.7: no; the 8 s silence ends it (250 frames)
        ] {
            let (model, _) = end_of_turn(Some(0.6));
            let stream = cat(&[speech(10, 16000), silence(260, 16000)]);
            let mut seg = energy(&semantic(eagerness), Some(model));
            let mut tl = Timeline::default();
            tl.feed(&mut seg, &stream, 1600);
            assert_eq!(tl.spans(), [(0, (10 + end_frames) * 32)], "{eagerness}");
            tl.assert_items_are_the_stream(&stream);
        }
    }

    #[test]
    fn the_semantic_history_holds_only_its_window_and_is_dropped_at_the_turn_end() {
        let (model, _) = end_of_turn(Some(0.1));
        let mut seg = energy(&semantic("auto"), Some(model));
        let append = cat(&[speech(400, 16000), silence(10, 16000)]);
        seg.feed(&append);
        assert!(seg.speaking);
        assert!(seg.recent.iter().map(Vec::len).sum::<usize>() <= SMART_TURN_SECONDS * SAMPLE_RATE + FRAME);
        seg.feed(&silence(130, 16000));
        assert!(!seg.speaking && seg.recent.is_empty());
        seg.feed(&append);
        seg.reset(0);
        assert!(seg.recent.is_empty());
    }

    // ── Silero, through ONNX Runtime ────────────────────────────────────────

    #[test]
    #[ignore = "needs ONNX Runtime and the silero-vad package's exports, named by MSTAR_REALTIME's onnxruntime and silero"]
    fn only_the_silero_export_this_server_runs_is_loaded() {
        let silero = facts()["silero"].as_str().expect("MSTAR_REALTIME names silero");
        let dir = std::path::Path::new(silero).parent().unwrap();
        for (file, loads) in [
            ("silero_vad.onnx", true),
            // lists sr before state; inputs go by name
            ("silero_vad_op18_ifless.onnx", true),
            ("silero_vad_half.onnx", false),
            ("silero_vad_openvino_16k.onnx", false),
            ("silero_vad_16k_sequence.onnx", false),
        ] {
            let path = dir.join(file);
            match load_silero(path.to_str().unwrap()) {
                Ok(_) => assert!(loads, "{file} loaded"),
                Err(why) => {
                    assert!(!loads, "{file}: {why}");
                    assert!(why.contains("is not the Silero VAD export"), "{file}: {why}");
                }
            }
        }
    }

    #[test]
    #[ignore = "needs ONNX Runtime and Silero VAD, named by MSTAR_REALTIME's onnxruntime and silero"]
    fn silero_streams_share_one_model_but_not_its_state() {
        // streams interleaved through the one session score each frame as
        // they would alone, and a reset starts a stream over
        assert!(silero_available());
        let frames: Vec<Vec<f32>> = (0..40)
            .map(|k| (0..FRAME).map(|i| (((k * FRAME + i) as f32) * 0.07).sin() * (k % 5) as f32 * 0.1).collect())
            .collect();
        let alone: Vec<f64> = {
            let mut v = Silero::new();
            frames.iter().map(|f| v.prob(f)).collect()
        };
        let (mut a, mut b) = (Silero::new(), Silero::new());
        let mut together = Vec::new();
        for (f, g) in frames.iter().zip(frames.iter().rev()) {
            together.push(a.prob(f));
            b.prob(g);
        }
        assert_eq!(together, alone);
        a.reset();
        assert_eq!(frames.iter().map(|f| a.prob(f)).collect::<Vec<_>>(), alone);
        assert!(alone.windows(2).any(|w| w[0] != w[1]), "the probabilities should depend on the audio");
    }

    // ── against the Python segmenter ────────────────────────────────────────

    fn float(v: &Value) -> f64 {
        f64::from_bits(v.as_u64().expect("a float as its bits"))
    }

    fn reference_cfg(v: &Value) -> TurnDetection {
        match v[0].as_str() {
            Some("server") => server(ServerVad {
                threshold: float(&v[1]),
                prefix_padding_ms: v[2].as_i64().unwrap(),
                silence_duration_ms: v[3].as_i64().unwrap(),
                ..ServerVad::default()
            }),
            _ => semantic(v[1].as_str().unwrap()),
        }
    }

    /// Audio as runs `[first, len]` of consecutive sample values.
    fn runs(audio: &[f32], into: &mut Vec<[i64; 2]>) {
        for &x in audio {
            let x = x as i64;
            match into.last_mut() {
                Some(r) if r[0] + r[1] == x => r[1] += 1,
                _ => into.push([x, 1]),
            }
        }
    }

    /// One case's operations and end-of-turn calls, in the reference script's encoding.
    fn replay(case: &Value) -> (Value, Value) {
        let palette: Vec<f64> = case["palette"].as_array().unwrap().iter().map(float).collect();
        let probs: Vec<f64> = case["probs"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| std::iter::repeat_n(palette[r[0].as_u64().unwrap() as usize], r[1].as_u64().unwrap() as usize))
            .collect();
        let vad = Scripted::new(probs);
        let answers: Vec<f64> = case["eot"][0].as_array().unwrap().iter().map(float).collect();
        let fail_at = case["eot"][1].as_i64().unwrap();
        let calls = Arc::new(Mutex::new(Vec::<Value>::new()));
        let seen = calls.clone();
        let model: EndOfTurn = Arc::new(move |audio: &[f32]| {
            let mut seen = seen.lock().unwrap();
            let k = seen.len();
            let mut r = Vec::new();
            runs(audio, &mut r);
            seen.push(json!(r));
            if k as i64 == fail_at {
                return Err("boom".into());
            }
            Ok(answers[k.min(answers.len() - 1)])
        });
        let max_turn = case["max_turn"].as_u64().unwrap() as usize;
        let mut seg = TurnSegmenter::with_vad(
            Box::new(vad.clone()),
            &reference_cfg(&case["cfg"]),
            max_turn,
            case["pos"].as_i64().unwrap(),
            Some(model.clone()),
        );
        let mut next = 0usize;
        let mut events: Vec<Value> = Vec::new();
        for a in case["actions"].as_array().unwrap() {
            let ops = match a[0].as_str().unwrap() {
                "feed" => {
                    let n = a[1].as_u64().unwrap() as usize;
                    let audio: Vec<f32> = (next..next + n).map(|i| i as f32).collect();
                    next += n;
                    seg.feed(&audio)
                }
                "end" => seg.end_turn(),
                "reset" => {
                    seg.reset(a[1].as_u64().unwrap() as usize);
                    vec![]
                }
                "shift" => {
                    seg.shift(a[1].as_i64().unwrap());
                    vec![]
                }
                "take" => {
                    let cfg = reference_cfg(&a[1]);
                    let mut new = TurnSegmenter::with_vad(Box::new(vad.clone()), &cfg, max_turn, 0, Some(model.clone()));
                    new.take_over(seg);
                    seg = new;
                    vec![]
                }
                other => panic!("unknown action {other}"),
            };
            // consecutive audio operations are one event
            let mut audio_runs = Vec::new();
            for op in ops {
                if let Op::Audio(audio) = &op {
                    runs(audio, &mut audio_runs);
                    continue;
                }
                if !audio_runs.is_empty() {
                    events.push(json!(["a", std::mem::take(&mut audio_runs)]));
                }
                events.push(match op {
                    Op::SpeechStarted(ms) => json!(["s", ms]),
                    Op::SpeechStopped(ms) => json!(["p", ms]),
                    Op::Commit => json!(["c"]),
                    Op::EndOfTurnFailed(_) => json!(["f"]),
                    Op::Audio(_) => unreachable!(),
                });
            }
            if !audio_runs.is_empty() {
                events.push(json!(["a", audio_runs]));
            }
            events.push(json!(["|"]));
        }
        let calls = Value::Array(calls.lock().unwrap().clone());
        (Value::Array(events), calls)
    }

    /// Replays every case; panics naming the first few that differ.
    fn check_reference(fixture: &str) -> usize {
        let cases: Vec<Value> = serde_json::from_str(fixture).unwrap();
        let mut differ = Vec::new();
        for (i, case) in cases.iter().enumerate() {
            let (ops, calls) = replay(case);
            if ops != case["ops"] || calls != case["eot_calls"] {
                let ops = ops.as_array().unwrap();
                let want = case["ops"].as_array().unwrap();
                let at = ops.iter().zip(want).position(|(a, b)| a != b).unwrap_or(ops.len().min(want.len()));
                differ.push(format!(
                    "case {i}: event {at}: {:?} here, {:?} in Python (eot calls equal: {})",
                    ops.get(at),
                    want.get(at),
                    calls == case["eot_calls"]
                ));
            }
        }
        assert!(differ.is_empty(), "{} of {} cases differ:\n{}", differ.len(), cases.len(), differ[..differ.len().min(5)].join("\n"));
        cases.len()
    }

    /// Cases recorded from the Python `TurnSegmenter` (`openai/realtime_vad.py`
    /// before the FastAPI frontend was removed): random settings, scripted
    /// probabilities and end-of-turn answers, appends of any length, client
    /// commits, clears, timeline shifts and settings changes between turns.
    /// Floats are their bits, audio is runs of consecutive sample values, and
    /// `|` ends an action's operations.
    #[test]
    fn matches_the_python_segmenter() {
        assert!(check_reference(include_str!("testdata/vad_reference.json")) > 100);
    }

    #[test]
    #[ignore = "reads a larger fixture of the same format from VAD_REFERENCE_FIXTURE"]
    fn matches_the_python_segmenter_on_a_larger_fixture() {
        let path = std::env::var("VAD_REFERENCE_FIXTURE").expect("VAD_REFERENCE_FIXTURE names the fixture");
        let n = check_reference(&std::fs::read_to_string(path).unwrap());
        eprintln!("{n} cases match");
    }
}
