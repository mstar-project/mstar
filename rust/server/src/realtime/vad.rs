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
            let mut s = match session(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("WARNING: server_vad is unavailable: cannot load Silero VAD from {path}: {e}");
                    return None;
                }
            };
            let want = vec![
                ("input".to_string(), "tensor(float)".to_string()),
                ("sr".to_string(), "tensor(int64)".to_string()),
                ("state".to_string(), "tensor(float)".to_string()),
            ];
            if inputs_of(&s) != want {
                eprintln!("WARNING: server_vad is unavailable: {path} is not the Silero VAD export this server runs (silero_vad.onnx)");
                return None;
            }
            if score(&mut s, &vec![0.0; CONTEXT + FRAME], &vec![0.0; STATE_LEN]).is_err() {
                eprintln!("WARNING: server_vad is unavailable: {path} cannot score a silent frame");
                return None;
            }
            Some(Mutex::new(s))
        })
        .as_ref()
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
