//! Per-model translation between OpenAI-shaped requests and mstar's request
//! path. Ported from `api_server/openai/adapters.py`.
//!
//! The OpenAI endpoints are model-agnostic; everything model-specific lives
//! here. An adapter translates an OpenAI request into [`SubmitArgs`] (what the
//! bridge submits to the conductor) and declares which OpenAI surfaces the
//! model supports. Output chunks are translated back to OpenAI shapes by the
//! generic serving handlers.
//!
//! `model_kwargs` is non-standardized across models, so each adapter maps the
//! standard OpenAI fields (`temperature`, `top_p`, `max_tokens`, `seed`,
//! `voice`, `modalities`, …) onto the keys its model actually honors. Non-OpenAI
//! knobs (`top_k`, `repetition_penalty`, model-namespaced keys like
//! `talker_top_p`) are not first-class fields — they pass through verbatim from
//! the request's `extra` (the OpenAI client's `extra_body`).
//!
//! Models with no OpenAI-standard output (robot actions, world-model latents)
//! intentionally have no adapter: they are served only through `/generate` and
//! `/v1/*` returns 404 for them.
//!
//! The model's adapter is the one `ADAPTER_REGISTRY` names for it: the
//! launcher passes the Python adapter class in `MSTAR_OPENAI_ADAPTER`, and
//! [`from_python`] finds its port in [`ADAPTERS`].

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::media;
use crate::protocol::{
    ChatCompletionRequest, Content, ImageGenerationRequest, SpeechRequest, VideoGenerationRequest,
};

mod chat;
mod images;
mod messages;
mod speech;
mod video;

pub use chat::*;
pub use images::*;
pub use messages::flatten_messages;
use messages::*;
pub use speech::*;
pub use video::{Cosmos3, Cosmos3Edge};

/// One element of a chat prompt, in request order: mstar's `PromptPart`,
/// field for field, so the bridge builds one from each as it arrives.
#[derive(Debug, Clone, Serialize)]
pub struct Part {
    pub modality: String,
    pub text: Option<String>,
    /// The position within its own modality's attachments.
    pub index: usize,
    /// The chat role of the message it came from; `None` from `/generate`.
    pub role: Option<String>,
}

/// The arguments the bridge submits to the conductor (mirrors mstar's
/// `SubmitArgs` / `PreprocessInput`).
#[derive(Debug, Clone)]
pub struct SubmitArgs {
    pub text: Option<String>,
    /// Pre-tokenized prompt (the frontend-tokenizes fast path): when set, the
    /// model side skips tokenization and the response text streams back as
    /// raw token ids for the frontend to detokenize off the GIL.
    pub tokens: Option<Vec<u32>>,
    /// modality -> list of persisted file paths.
    pub file_paths: BTreeMap<String, Vec<String>>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub model_kwargs: Map<String, Value>,
    /// A chat's parts, in order; empty on every other path.
    pub parts: Vec<Part>,
    /// End the result stream with a `usage` chunk (set by the chat handler).
    pub report_usage: bool,
}

impl Default for SubmitArgs {
    fn default() -> Self {
        Self {
            text: None,
            tokens: None,
            report_usage: false,
            file_paths: BTreeMap::new(),
            input_modalities: Vec::new(),
            output_modalities: vec!["text".to_string()],
            model_kwargs: Map::new(),
            parts: Vec::new(),
        }
    }
}

/// setdefault: insert only if the key is absent, so an explicit `extra_body`
/// value wins over the standard field (matching mstar's `mk.setdefault`).
fn set_default(mk: &mut Map<String, Value>, key: &str, value: Value) {
    if !mk.contains_key(key) {
        mk.insert(key.to_string(), value);
    }
}

/// The OpenAI-standard sampling fields → a model's `model_kwargs` keys.
struct Sampling<'a> {
    temperature: Option<Value>,
    top_p: Option<Value>,
    seed: Option<i64>,
    max_tokens: Option<i64>,
    temperature_key: &'a str,
    top_p_key: &'a str,
    /// `None` disables `max_tokens` mapping (e.g. speech has no such field).
    max_tokens_key: Option<&'a str>,
}

fn apply_sampling(mk: &mut Map<String, Value>, s: Sampling) {
    if let Some(t) = s.temperature {
        set_default(mk, s.temperature_key, t);
    }
    if let Some(p) = s.top_p {
        set_default(mk, s.top_p_key, p);
    }
    if let Some(seed) = s.seed {
        set_default(mk, "seed", Value::from(seed));
    }
    if let Some(key) = s.max_tokens_key {
        if let Some(mt) = s.max_tokens {
            set_default(mk, key, Value::from(mt));
        }
    }
}

fn json_num(f: f64) -> Value {
    serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)
}

/// A client-sent `speed` into `model_kwargs` for a model with no speed
/// control, so the server logs it as ignored (`_forward_sent_speed`). An
/// unsent one stays out; it would warn on every request.
fn forward_sent_speed(mk: &mut Map<String, Value>, req: &SpeechRequest) {
    if let Some(speed) = req.speed {
        set_default(mk, "speed", speed.map_or(Value::Null, json_num));
    }
}

/// A speech request's submission, and the reference clip it carries for the API
/// server to check before anything is submitted (`_resolve_ref_audio`'s check).
#[derive(Debug)]
pub struct Speech {
    pub args: SubmitArgs,
    pub ref_audio: Option<crate::media::RefAudioCheck>,
}

/// `/v1/audio/speech` sentence chunking: the adapter's `speech_chunk_*`
/// attributes (see `speech_chunking::plan_chunks`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechChunking {
    /// Inputs of at least this many characters are split; `None` splits only
    /// when the client sends `sentence_chunking: true`.
    pub min_chars: Option<usize>,
    pub max_chars: usize,
    /// Pieces kept in flight ahead of the one being streamed.
    pub lookahead: usize,
    /// Most pieces one input may split into; more is a 400.
    pub max_pieces: usize,
    pub splitter: crate::speech_chunking::Splitter,
}

const SPEECH_CHUNKING: SpeechChunking = SpeechChunking {
    min_chars: None,
    max_chars: 400,
    lookahead: 2,
    max_pieces: 32,
    splitter: crate::speech_chunking::Splitter::Sentences,
};

/// One Python `OpenAIAdapter` subclass, ported: which OpenAI surfaces its
/// models serve and how a request on each becomes a submission. A surface a
/// class doesn't serve keeps the default, which refuses it, so a new model's
/// adapter is a type implementing what it serves and a row in [`ADAPTERS`].
pub trait OpenAIAdapter: Send + Sync {
    fn supports_chat(&self) -> bool {
        false
    }

    fn supports_speech(&self) -> bool {
        false
    }

    fn supports_images(&self) -> bool {
        false
    }

    fn supports_videos(&self) -> bool {
        false
    }

    /// The adapter's `speech_chunk_*` attributes.
    fn speech_chunking(&self) -> SpeechChunking {
        SPEECH_CHUNKING
    }

    /// Whether this model's prompt processing is expressible in the frontend
    /// (plain `tokenizer.json`, no custom processor), so the server should
    /// tokenize and detokenize in Rust and submit token ids. It needs the server
    /// started with a tokenizer (`MSTAR_TOKENIZER`), and is false for every
    /// registered model: each needs its own processor or prompt formatting. The
    /// mechanism is exercised by /generate's `tokenize` field.
    fn frontend_tokenizes(&self) -> bool {
        false
    }

    /// `chat_to_request`, after the refusal every chat adapter makes
    /// (`refuse_unserved_chat`), whose off values are gone from `req.extra`.
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        let _ = (req, upload_dir, allow_remote);
        Err("chat is not supported by this model".to_string())
    }

    /// `speech_to_request` short of the `ref_audio` clip, which
    /// [`OpenAIAdapter::speech_to_request`] saves as [`OpenAIAdapter::ref_audio`] says.
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let _ = req;
        Err("audio/speech is not supported by this model".to_string())
    }

    /// How a voice-cloning model takes `ref_audio`; `None` for one that takes none
    /// (the field then passes through, as the Python adapter leaves it).
    fn ref_audio(&self) -> Option<RefAudio> {
        None
    }

    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        let _ = req;
        Err("image generation is not supported by this model".to_string())
    }

    fn video_to_request(&self, req: &VideoGenerationRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        let _ = (req, upload_dir, allow_remote);
        Err("video generation is not supported by this model".to_string())
    }

    fn image_edit_to_request(&self, prompt: &str, image_path: &str, extra_kwargs: Map<String, Value>) -> Result<SubmitArgs, String> {
        let _ = (prompt, image_path, extra_kwargs);
        Err("image editing is not supported by this model".to_string())
    }

    /// A chat's submission. Not for adapters to override: they implement [`OpenAIAdapter::chat_args`].
    fn chat_to_request(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        let mut req = req.clone();
        let mut extra = std::mem::take(&mut req.extra);
        refuse_unserved_chat(&req, &mut extra)?;
        req.extra = extra;
        self.chat_args(&req, upload_dir, allow_remote)
    }

    /// A speech's submission, with a `ref_audio` clip saved as the model's audio
    /// input, as the Python adapter's is. Not for adapters to override: they
    /// implement [`OpenAIAdapter::speech_args`] and [`OpenAIAdapter::ref_audio`].
    fn speech_to_request(&self, req: &SpeechRequest, upload_dir: &Path, allow_remote: bool) -> Result<Speech, (u16, String)> {
        let mut args = self.speech_args(req).map_err(|e| (400, e))?;
        let Some(RefAudio { decoder, clip_first }) = self.ref_audio() else {
            return Ok(Speech { args, ref_audio: None });
        };
        let Some(reference) = args.model_kwargs.remove("ref_audio").filter(truthy) else {
            return Ok(Speech { args, ref_audio: None });
        };
        let check = crate::media::save_ref_audio(&reference, upload_dir, allow_remote, decoder)?;
        args.file_paths.insert("audio".to_string(), vec![check.path.clone()]);
        let layout = if clip_first { ["audio", "text"] } else { ["text", "audio"] };
        args.input_modalities = layout.map(str::to_string).to_vec();
        Ok(Speech { args, ref_audio: Some(check) })
    }
}

/// How a voice-cloning adapter takes its reference clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefAudio {
    /// The library its model's loader opens the clip with (`_resolve_ref_audio`'s `decoder`).
    pub decoder: &'static str,
    /// The clip's place in the layout: before the text (Qwen3-TTS) or after it.
    pub clip_first: bool,
}

/// The ported adapters, by the Python class name the launcher passes in
/// `MSTAR_OPENAI_ADAPTER`.
pub static ADAPTERS: &[(&str, &dyn OpenAIAdapter)] = &[
    ("BagelAdapter", &Bagel),
    ("Qwen3OmniAdapter", &Qwen3Omni),
    ("Qwen3_5Adapter", &Qwen3_5),
    ("OrpheusAdapter", &Orpheus),
    ("Cosmos3Adapter", &Cosmos3),
    ("Cosmos3EdgeAdapter", &Cosmos3Edge),
    ("Qwen3TTSAdapter", &Qwen3TTS),
    ("DiffusionImageAdapter", &DiffusionImage),
    ("TextChatAdapter", &TextChat),
    ("KokoroAdapter", &Kokoro),
    ("ChatterboxAdapter", &Chatterbox),
    ("FishS2ProAdapter", &FishS2Pro),
    ("CosyVoice3Adapter", &CosyVoice3),
    ("OmniVoiceAdapter", &OmniVoice),
    ("VoxCPM2Adapter", &VoxCPM2),
];

/// The adapter for a Python class name; `None` for a class with no port here.
pub fn from_python(class: &str) -> Option<&'static dyn OpenAIAdapter> {
    ADAPTERS.iter().find(|(name, _)| *name == class).map(|(_, adapter)| *adapter)
}

fn set_seed(mk: &mut Map<String, Value>, seed: Option<i64>) {
    if let Some(seed) = seed {
        set_default(mk, "seed", Value::from(seed));
    }
}

/// Python's `bool()` of a JSON value.
pub(crate) fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests;
