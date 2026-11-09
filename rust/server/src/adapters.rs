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
}

impl Default for SubmitArgs {
    fn default() -> Self {
        Self {
            text: None,
            tokens: None,
            file_paths: BTreeMap::new(),
            input_modalities: Vec::new(),
            output_modalities: vec!["text".to_string()],
            model_kwargs: Map::new(),
            parts: Vec::new(),
        }
    }
}

/// Flatten OpenAI chat `messages` into (text, file_paths, input_modalities, parts).
///
/// `parts` is the ordered sequence as written; the other three derive from
/// it: text newline-joined, attachments persisted under `upload_dir` and
/// grouped by modality, `input_modalities` the per-part modality sequence.
/// So an attachment's position and a repeated modality both survive. Each
/// text part carries its message's role, so a reply stays its own turn. The rules
/// are `flatten_messages` in `api_server/openai/adapters.py`, line for line.
pub fn flatten_messages(
    messages: &[crate::protocol::ChatMessage],
    upload_dir: &Path,
    allow_remote: bool,
) -> Result<(Option<String>, BTreeMap<String, Vec<String>>, Vec<String>, Vec<Part>), String> {
    let mut parts: Vec<Part> = Vec::new();
    // A refused chat is never submitted, so a refusal below drops `saved`
    // unkept, which deletes the attachments saved before it.
    let mut saved = crate::SavedUploads::new(upload_dir);

    for msg in messages {
        let role = match msg.role.as_str() {
            "developer" => "system", // OpenAI's newer name for system
            role @ ("system" | "user" | "assistant") => role,
            other => {
                return Err(format!(
                    "a message's role must be system, developer, user or assistant, not '{other}'"
                ))
            }
        };
        let content = match &msg.content {
            None => continue,
            Some(Content::Text(s)) => {
                if !s.is_empty() {
                    add_text(&mut parts, s, role);
                }
                continue;
            }
            Some(Content::Parts(content)) => content,
        };
        for part in content {
            let obj = match part.as_object() {
                Some(o) => o,
                None => continue,
            };
            let ptype = obj.get("type").and_then(Value::as_str).unwrap_or("");
            match ptype {
                "text" => {
                    if let Some(t) = obj.get("text").and_then(Value::as_str) {
                        if !t.is_empty() {
                            add_text(&mut parts, t, role);
                        }
                    }
                }
                "image_url" | "video_url" | "audio_url" => {
                    let url = nested_url(obj, ptype);
                    if !url.is_empty() {
                        // An `unknown` MIME type stays unknown, as in Python, for
                        // the submission to refuse; the part's kind is no evidence.
                        let (m, p) = media::resolve_media_ref(&url, upload_dir, allow_remote)?;
                        add_file(&mut parts, &mut saved.paths, m, p);
                    }
                }
                "input_audio" => {
                    // OpenAI-native audio input: base64 + format.
                    let ia = obj.get("input_audio").and_then(Value::as_object);
                    if let Some(ia) = ia {
                        let data = ia.get("data").and_then(Value::as_str).unwrap_or("");
                        let fmt = ia.get("format").and_then(Value::as_str).unwrap_or("wav");
                        if !data.is_empty() {
                            let (m, p) = media::save_base64(data, fmt, "audio", upload_dir)?;
                            add_file(&mut parts, &mut saved.paths, m, p);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let texts: Vec<&str> = parts.iter().filter_map(|p| p.text.as_deref()).collect();
    let text = if texts.is_empty() { None } else { Some(texts.join("\n")) };
    let input_modalities = parts.iter().map(|p| p.modality.clone()).collect();
    Ok((text, saved.keep(), input_modalities, parts))
}

fn add_file(
    parts: &mut Vec<Part>,
    file_paths: &mut BTreeMap<String, Vec<String>>,
    modality: String,
    path: String,
) {
    // shortcut: always a user turn, so the model can't tell an image it made from one it was
    // sent, and a reply that is only an image leaves no assistant turn; a feature like
    // multi-turn editing would give the turn break between two attachments a text slot
    let paths = file_paths.entry(modality.clone()).or_default();
    parts.push(Part {
        modality,
        text: None,
        index: paths.len(),
        role: Some("user".to_string()),
    });
    paths.push(path);
}

fn add_text(parts: &mut Vec<Part>, text: &str, role: &str) {
    // Adjacent text parts were newline-joined before ordering was kept; merge
    // them here, across messages of one role too (the layout has no slot for
    // the boundary between them), so only text an attachment or a role change
    // separates gets its own part.
    if let Some(last) = parts.last_mut() {
        if last.modality == "text" && last.role.as_deref() == Some(role) {
            let merged = last.text.get_or_insert_with(String::new);
            merged.push('\n');
            merged.push_str(text);
            return;
        }
    }
    parts.push(Part {
        modality: "text".to_string(),
        text: Some(text.to_string()),
        index: 0,
        role: Some(role.to_string()),
    });
}

/// `{ "<key>": { "url": "..." } }` -> the url string (or "").
fn nested_url(obj: &Map<String, Value>, key: &str) -> String {
    obj.get(key)
        .and_then(Value::as_object)
        .and_then(|o| o.get("url"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// setdefault: insert only if the key is absent, so an explicit `extra_body`
/// value wins over the standard field (matching mstar's `mk.setdefault`).
fn set_default(mk: &mut Map<String, Value>, key: &str, value: Value) {
    if !mk.contains_key(key) {
        mk.insert(key.to_string(), value);
    }
}

/// OpenAI `size` ("WxH") onto the model's `width` / `height`, as the Python
/// adapters' `_size_kwargs`.
fn size_kwargs(mk: &mut Map<String, Value>, size: Option<&str>) -> Result<(), String> {
    let Some(size) = size.filter(|s| !s.is_empty()) else {
        return Ok(());
    };
    let parsed = size
        .to_lowercase()
        .split_once('x')
        .and_then(|(w, h)| Some((w.trim().parse::<i64>().ok()?, h.trim().parse::<i64>().ok()?)));
    let Some((width, height)) = parsed else {
        return Err(format!("size must be 'WxH' (e.g. '1024x1024'); got {size:?}"));
    };
    set_default(mk, "width", Value::from(width));
    set_default(mk, "height", Value::from(height));
    Ok(())
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

    /// `chat_to_request`, after the tool-call refusal every chat adapter makes.
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
        refuse_tool_calls(req)?;
        self.chat_args(req, upload_dir, allow_remote)
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

/// A speech's model kwargs: the passthrough fields, `voice` when sent, and the
/// standard sampling fields under the model's names.
fn speech_kwargs(req: &SpeechRequest, temperature_key: &str, top_p_key: &str) -> Map<String, Value> {
    let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
    if let Some(v) = &req.voice {
        mk.insert("voice".to_string(), Value::from(v.clone()));
    }
    sampling(&mut mk, req, temperature_key, top_p_key);
    mk
}

/// `_apply_sampling` for a speech: `temperature`, `top_p` and `seed`, no token cap.
fn sampling(mk: &mut Map<String, Value>, req: &SpeechRequest, temperature_key: &str, top_p_key: &str) {
    apply_sampling(
        mk,
        Sampling {
            temperature: req.temperature.clone(),
            top_p: req.top_p.clone(),
            seed: req.seed,
            max_tokens: None,
            temperature_key,
            top_p_key,
            max_tokens_key: None,
        },
    );
}

/// An audio-out submission of the speech's input.
fn speech_submission(req: &SpeechRequest, model_kwargs: Map<String, Value>) -> SubmitArgs {
    SubmitArgs {
        text: Some(req.input.clone()),
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["audio".to_string()],
        model_kwargs,
        ..SubmitArgs::default()
    }
}

/// `SpeechRequest.speed` as Python reads it: unsent is its default 1.0, `null` None.
fn speech_speed(req: &SpeechRequest) -> Option<f64> {
    match req.speed {
        None => Some(1.0),
        Some(sent) => sent,
    }
}

fn set_seed(mk: &mut Map<String, Value>, seed: Option<i64>) {
    if let Some(seed) = seed {
        set_default(mk, "seed", Value::from(seed));
    }
}

/// OpenAI's `instructions` as the model's `instruct`, when non-empty.
fn instructions_as_instruct(mk: &mut Map<String, Value>) {
    if let Some(instructions) = mk.remove("instructions").filter(truthy) {
        set_default(mk, "instruct", instructions);
    }
}

/// `BagelAdapter.image_to_request` (`DiffusionImageAdapter`'s is the same, line for line).
fn sized_image(req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
    let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
    size_kwargs(&mut mk, req.size.as_deref())?;
    set_seed(&mut mk, req.seed);
    Ok(SubmitArgs {
        text: Some(req.prompt.clone()),
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["image".to_string()],
        model_kwargs: mk,
        ..SubmitArgs::default()
    })
}

/// `BagelAdapter.image_edit_to_request` (and `DiffusionImageAdapter`'s).
fn sized_image_edit(prompt: &str, image_path: &str, mut extra_kwargs: Map<String, Value>) -> Result<SubmitArgs, String> {
    // a form field that parsed as JSON (e.g. "512") is still the size text
    let size = extra_kwargs
        .remove("size")
        .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string));
    size_kwargs(&mut extra_kwargs, size.as_deref())?;
    let mut file_paths = BTreeMap::new();
    file_paths.insert("image".to_string(), vec![image_path.to_string()]);
    Ok(SubmitArgs {
        text: Some(prompt.to_string()),
        file_paths,
        input_modalities: vec!["image".to_string(), "text".to_string()],
        output_modalities: vec!["image".to_string()],
        model_kwargs: extra_kwargs,
        ..SubmitArgs::default()
    })
}

/// BAGEL: chat (it reads temperature / top_p from its model config; only
/// max_output_tokens and seed are honored), images and edits.
pub struct Bagel;

impl OpenAIAdapter for Bagel {
    fn supports_chat(&self) -> bool {
        true
    }
    fn supports_images(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        text_chat(req, upload_dir, allow_remote)
    }
    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        sized_image(req)
    }
    fn image_edit_to_request(&self, prompt: &str, image_path: &str, extra_kwargs: Map<String, Value>) -> Result<SubmitArgs, String> {
        sized_image_edit(prompt, image_path, extra_kwargs)
    }
}

/// Qwen3.5: multimodal chat.
pub struct Qwen3_5;

impl OpenAIAdapter for Qwen3_5 {
    fn supports_chat(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        text_chat(req, upload_dir, allow_remote)
    }
}

/// Qwen3-Omni: chat (text, or text and audio) and speech, sampling the Thinker
/// for a chat and the Talker for a speech.
pub struct Qwen3Omni;

impl OpenAIAdapter for Qwen3Omni {
    fn supports_chat(&self) -> bool {
        true
    }
    fn supports_speech(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        let (text, file_paths, in_mods, parts) = flatten_messages(&req.messages, upload_dir, allow_remote)?;
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        // Speech output also emits text, so request both when audio asked.
        let want_audio = req.modalities.as_ref().is_some_and(|m| m.iter().any(|x| x == "audio"));
        let out_mods = if want_audio { vec!["text", "audio"] } else { vec!["text"] };
        apply_sampling(
            &mut mk,
            Sampling {
                temperature: req.temperature.clone(),
                top_p: req.top_p.clone(),
                seed: req.seed,
                max_tokens: chat_max_tokens(req),
                temperature_key: "thinker_temperature",
                top_p_key: "thinker_top_p",
                max_tokens_key: Some("max_output_tokens"),
            },
        );
        if let Some(voice) = chat_voice(req) {
            mk.insert("voice".to_string(), Value::from(voice));
        }
        Ok(SubmitArgs {
            tokens: None,
            text,
            file_paths,
            input_modalities: in_mods,
            output_modalities: out_mods.into_iter().map(str::to_string).collect(),
            model_kwargs: mk,
            parts,
        })
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk = speech_kwargs(req, "talker_temperature", "talker_top_p");
        forward_sent_speed(&mut mk, req);
        Ok(SubmitArgs { output_modalities: vec!["text".to_string(), "audio".to_string()], ..speech_submission(req, mk) })
    }
}

/// Orpheus: speech.
pub struct Orpheus;

impl OpenAIAdapter for Orpheus {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk = speech_kwargs(req, "temperature", "top_p");
        forward_sent_speed(&mut mk, req);
        Ok(speech_submission(req, mk))
    }
}

/// Qwen3-TTS (CustomVoice, VoiceDesign, Base): speech, with `instructions` as
/// `instruct`, long input chunked, and a Base model's reference clip first.
pub struct Qwen3TTS;

impl OpenAIAdapter for Qwen3TTS {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_chunking(&self) -> SpeechChunking {
        // each Talker request stays near the lengths the model was trained on
        SpeechChunking { min_chars: Some(600), ..SPEECH_CHUNKING }
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(v) = &req.voice {
            mk.insert("voice".to_string(), Value::from(v.clone()));
        }
        // OpenAI's field is `instructions`; the model reads `instruct`.
        instructions_as_instruct(&mut mk);
        sampling(&mut mk, req, "temperature", "top_p");
        forward_sent_speed(&mut mk, req);
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile", clip_first: true })
    }
}

/// Cosmos3: images and videos, `size` passed on as the model's own kwarg.
pub struct Cosmos3;

impl OpenAIAdapter for Cosmos3 {
    fn supports_images(&self) -> bool {
        true
    }
    fn supports_videos(&self) -> bool {
        true
    }
    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(size) = &req.size {
            set_default(&mut mk, "size", Value::from(size.clone()));
        }
        set_seed(&mut mk, req.seed);
        Ok(SubmitArgs {
            text: Some(req.prompt.clone()),
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["image".to_string()],
            model_kwargs: mk,
            ..SubmitArgs::default()
        })
    }
    /// Text-to-video, or image/video-to-video when a conditioning reference is
    /// supplied, as `Cosmos3Adapter.video_to_request`.
    fn video_to_request(&self, req: &VideoGenerationRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(size) = &req.size {
            set_default(&mut mk, "size", Value::from(size.clone()));
        }
        set_seed(&mut mk, req.seed);
        // num_frames / fps are first-class video fields (not extra_body).
        if let Some(nf) = req.num_frames {
            set_default(&mut mk, "num_frames", Value::from(nf));
        }
        if let Some(fps) = req.fps {
            set_default(&mut mk, "fps", json_num(fps));
        }
        let mut args = SubmitArgs {
            text: Some(req.prompt.clone()),
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["video".to_string()],
            model_kwargs: mk,
            ..SubmitArgs::default()
        };
        // The conditioning frame (image-to-video) or clip (video-to-video) is
        // persisted and VAE-encoded by the worker into the clean frame-0 anchor /
        // pinned latent prefix. At most one may be given.
        let reference = match (req.image.as_deref(), req.video.as_deref()) {
            (Some(_), Some(_)) => return Err("Provide either 'image' or 'video' conditioning, not both.".to_string()),
            (Some(image), None) => Some(("image", image)),
            (None, Some(video)) => Some(("video", video)),
            (None, None) => None,
        };
        if let Some((modality, reference)) = reference {
            let (_, path) = media::resolve_media_ref(reference, upload_dir, allow_remote)?;
            args.file_paths.insert(modality.to_string(), vec![path]);
            args.input_modalities = vec![modality.to_string(), "text".to_string()];
        }
        Ok(args)
    }
}

/// Cosmos3-Edge: Cosmos3's images and videos, and chat.
pub struct Cosmos3Edge;

impl OpenAIAdapter for Cosmos3Edge {
    fn supports_chat(&self) -> bool {
        true
    }
    fn supports_images(&self) -> bool {
        true
    }
    fn supports_videos(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        // How the vLLM recipe toggles thinking; a flat enable_thinking wins.
        // Read before the attachments are saved, since a malformed value
        // refuses the request.
        let thinking = template_enable_thinking(req.extra.get("chat_template_kwargs"))?;
        let mut args = text_chat(req, upload_dir, allow_remote)?;
        args.model_kwargs.remove("chat_template_kwargs");
        if let Some(on) = thinking {
            set_default(&mut args.model_kwargs, "enable_thinking", on);
        }
        Ok(args)
    }
    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        Cosmos3.image_to_request(req)
    }
    fn video_to_request(&self, req: &VideoGenerationRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        Cosmos3.video_to_request(req, upload_dir, allow_remote)
    }
}

/// The DiT-scaffold image models (FLUX.2 klein, Z-Image): BAGEL's image translations.
pub struct DiffusionImage;

impl OpenAIAdapter for DiffusionImage {
    fn supports_images(&self) -> bool {
        true
    }
    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        sized_image(req)
    }
    fn image_edit_to_request(&self, prompt: &str, image_path: &str, extra_kwargs: Map<String, Value>) -> Result<SubmitArgs, String> {
        sized_image_edit(prompt, image_path, extra_kwargs)
    }
}

/// Text-only LLMs (GLM-5.x): chat, refusing what a text-only model cannot serve.
pub struct TextChat;

impl OpenAIAdapter for TextChat {
    fn supports_chat(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, _allow_remote: bool) -> Result<SubmitArgs, String> {
        text_only_chat(req, upload_dir)
    }
}

/// Kokoro: speech, forwarding `voice` and `speed`, which are real model knobs.
pub struct Kokoro;

impl OpenAIAdapter for Kokoro {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(voice) = &req.voice {
            set_default(&mut mk, "voice", Value::from(voice.clone()));
        }
        if let Some(speed) = speech_speed(req) {
            set_default(&mut mk, "speed", json_num(speed));
        }
        set_seed(&mut mk, req.seed);
        Ok(speech_submission(req, mk))
    }
}

/// Chatterbox: speech with the standard sampling fields, cloning a voice from
/// a clip its loader opens with soundfile, else FFmpeg.
pub struct Chatterbox;

impl OpenAIAdapter for Chatterbox {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(voice) = &req.voice {
            set_default(&mut mk, "voice", Value::from(voice.clone()));
        }
        sampling(&mut mk, req, "temperature", "top_p");
        forward_sent_speed(&mut mk, req);
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile+torchcodec", clip_first: false })
    }
}

/// Fish Audio S2-Pro: speech with the standard sampling fields, long input
/// chunked; `voice` and `speed` are no controls of it.
pub struct FishS2Pro;

impl OpenAIAdapter for FishS2Pro {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_chunking(&self) -> SpeechChunking {
        SpeechChunking { min_chars: Some(600), ..SPEECH_CHUNKING }
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        sampling(&mut mk, req, "temperature", "top_p");
        // a transcript without its clip
        if !mk.get("ref_audio").is_some_and(truthy) {
            mk.remove("ref_text");
        }
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile", clip_first: false })
    }
}

/// Fun-CosyVoice3: speech that always clones a voice, cut by its own splitter.
pub struct CosyVoice3;

impl OpenAIAdapter for CosyVoice3 {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_chunking(&self) -> SpeechChunking {
        // 80 tokens at ~4 characters each; nothing under 80 characters of Chinese is split
        SpeechChunking {
            min_chars: Some(81),
            max_chars: 320,
            splitter: crate::speech_chunking::Splitter::CosyVoice3,
            ..SPEECH_CHUNKING
        }
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(voice) = req.voice.as_ref().filter(|v| !v.is_empty()) {
            set_default(&mut mk, "voice", Value::from(voice.clone()));
        }
        instructions_as_instruct(&mut mk);
        sampling(&mut mk, req, "temperature", "top_p");
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile", clip_first: false })
    }
}

/// OmniVoice: speech whose mode follows from what is sent; `voice` stands in
/// for `instruct` and `speed` is a real knob.
pub struct OmniVoice;

impl OpenAIAdapter for OmniVoice {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        // is-not-None, so an empty voice reaches the model and is refused there
        if let Some(voice) = &req.voice {
            if mk.get("instruct").is_none_or(Value::is_null) {
                mk.insert("instruct".to_string(), Value::from(voice.clone()));
            }
        }
        mk.remove("voice");
        if let Some(speed) = speech_speed(req) {
            set_default(&mut mk, "speed", json_num(speed));
        }
        set_seed(&mut mk, req.seed);
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile+torchcodec", clip_first: false })
    }
}

/// VoxCPM2: speech designing a voice from `instructions` or cloning one; it has
/// no named voices and no speed control.
pub struct VoxCPM2;

impl OpenAIAdapter for VoxCPM2 {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        instructions_as_instruct(&mut mk);
        set_seed(&mut mk, req.seed);
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile", clip_first: false })
    }
}

/// `max_completion_tokens`, else `max_tokens`, as Python reads them: an
/// explicit 0 is kept, so validation refuses it.
fn chat_max_tokens(req: &ChatCompletionRequest) -> Option<i64> {
    req.max_completion_tokens.or(req.max_tokens)
}

/// Text out of a chat's messages, with the standard sampling fields
/// (`BagelAdapter` / `Qwen3_5Adapter.chat_to_request`).
fn text_chat(
    req: &ChatCompletionRequest,
    upload_dir: &Path,
    allow_remote: bool,
) -> Result<SubmitArgs, String> {
    let (text, file_paths, in_mods, parts) = flatten_messages(&req.messages, upload_dir, allow_remote)?;
    let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
    apply_sampling(
        &mut mk,
        Sampling {
            temperature: req.temperature.clone(),
            top_p: req.top_p.clone(),
            seed: req.seed,
            max_tokens: chat_max_tokens(req),
            temperature_key: "temperature",
            top_p_key: "top_p",
            max_tokens_key: Some("max_output_tokens"),
        },
    );
    Ok(SubmitArgs {
        tokens: None,
        text,
        file_paths,
        input_modalities: in_mods,
        output_modalities: vec!["text".to_string()],
        model_kwargs: mk,
        parts,
    })
}

/// `TextChatAdapter.chat_to_request`: text-only LLMs, which refuse what they
/// cannot serve rather than dropping it. Remote media is never fetched.
fn text_only_chat(req: &ChatCompletionRequest, upload_dir: &Path) -> Result<SubmitArgs, String> {
    let (text, file_paths, in_mods, parts) = flatten_messages(&req.messages, upload_dir, false)?;
    let refuse = |message: String| {
        // a refused chat is never submitted, so nothing else removes its attachments
        for path in file_paths.values().flatten() {
            let _ = std::fs::remove_file(path);
        }
        Err(message)
    };
    let other: std::collections::BTreeSet<&str> =
        in_mods.iter().map(String::as_str).filter(|m| *m != "text").collect();
    if !other.is_empty() {
        let listed: Vec<String> = other.iter().map(|m| crate::media::py_repr(m)).collect();
        return refuse(format!("this model takes text only; the messages carry [{}]", listed.join(", ")));
    }
    if text.as_deref().is_none_or(str::is_empty) {
        return refuse("messages carry no text content".to_string());
    }
    for (name, value) in [("max_tokens", req.max_tokens), ("max_completion_tokens", req.max_completion_tokens)] {
        if let Some(v) = value.filter(|v| *v < 1) {
            return refuse(format!("{name} must be at least 1, got {v}"));
        }
    }
    if let Some(n) = req.n.filter(|n| *n != 1) {
        return refuse(format!("n={n}: this model returns one choice"));
    }
    if req.stop.as_ref().is_some_and(truthy) {
        return refuse(
            "stop sequences are not supported; the reply ends at a stop token or max_tokens".to_string(),
        );
    }
    let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
    for name in ["logprobs", "top_logprobs", "logit_bias", "presence_penalty", "frequency_penalty", "stop_token_ids"] {
        if let Some(v) = mk.get(name).filter(|v| truthy(v)) {
            return refuse(format!("{name} is not supported; got {}", py_value_repr(v)));
        }
        mk.remove(name);
    }
    if mk.remove("best_of").is_some_and(|v| !(v.is_null() || v == 1)) {
        return refuse("best_of is not supported; this model returns one choice".to_string());
    }
    if mk.remove("response_format").is_some_and(|v| !(v.is_null() || v == serde_json::json!({"type": "text"}))) {
        return refuse("response_format needs constrained decoding; only {'type': 'text'} is served".to_string());
    }
    apply_sampling(
        &mut mk,
        Sampling {
            temperature: req.temperature.clone(),
            top_p: req.top_p.clone(),
            seed: req.seed,
            max_tokens: chat_max_tokens(req),
            temperature_key: "temperature",
            top_p_key: "top_p",
            max_tokens_key: Some("max_output_tokens"),
        },
    );
    Ok(SubmitArgs {
        tokens: None,
        text,
        file_paths: BTreeMap::new(),
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["text".to_string()],
        model_kwargs: mk,
        parts,
    })
}

/// Python's `repr` of a JSON value as `json.loads` gives it, for a refusal
/// that quotes it as the Python frontend's does. An object's keys come in
/// serde_json's (sorted) order, not the client's.
pub(crate) fn py_value_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) if n.is_f64() => py_float_repr(n.as_f64().unwrap_or(f64::NAN)),
        Value::Number(n) => n.to_string(),
        Value::String(s) => crate::media::py_repr(s),
        Value::Array(items) => format!("[{}]", items.iter().map(py_value_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter().map(|(k, v)| format!("{}: {}", crate::media::py_repr(k), py_value_repr(v))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Python's `repr(float)`: the shortest round-trip digits, positional for a
/// decimal exponent in -4..16 (with at least one fractional digit) and
/// scientific with a two-digit exponent otherwise.
fn py_float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let sci = format!("{x:e}"); // shortest round-trip digits, e.g. "-1.25e-5"
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = mantissa.strip_prefix('-').map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if !(-4..16).contains(&exp) {
        let (head, tail) = digits.split_at(1);
        let frac = if tail.is_empty() { String::new() } else { format!(".{tail}") };
        let esign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{head}{frac}e{esign}{:02}", exp.abs());
    }
    if exp < 0 {
        return format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize));
    }
    let point = exp as usize + 1;
    if digits.len() <= point {
        format!("{sign}{}{}.0", digits, "0".repeat(point - digits.len()))
    } else {
        format!("{sign}{}.{}", &digits[..point], &digits[point..])
    }
}

/// `chat_template_kwargs.enable_thinking` read as `Cosmos3EdgeAdapter` does:
/// `(kwargs or {})`, then `kwargs["enable_thinking"]` as sent when it has the
/// key, so the model refuses a non-boolean rather than reading `"false"` as
/// true. A falsy value is no kwargs; a value Python cannot look the key up in
/// is the `TypeError` that answers 400 there.
fn template_enable_thinking(kwargs: Option<&Value>) -> Result<Option<Value>, String> {
    const KEY: &str = "enable_thinking";
    match kwargs {
        None => Ok(None),
        Some(v) if !truthy(v) => Ok(None),
        Some(Value::Object(m)) => Ok(m.get(KEY).cloned()),
        Some(Value::Array(items)) if items.iter().any(|i| i.as_str() == Some(KEY)) => {
            Err("list indices must be integers or slices, not str".to_string())
        }
        Some(Value::String(s)) if s.contains(KEY) => {
            Err("string indices must be integers, not 'str'".to_string())
        }
        Some(Value::Array(_) | Value::String(_)) => Ok(None),
        Some(Value::Bool(_)) => Err("argument of type 'bool' is not iterable".to_string()),
        Some(Value::Number(n)) => Err(format!(
            "argument of type '{}' is not iterable",
            if n.is_f64() { "float" } else { "int" }
        )),
        Some(Value::Null) => Ok(None),
    }
}

/// Python's `bool()` of a JSON value.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Refuse a chat that asks the model to call a tool. The rules are
/// `_refuse_tool_calls` in `api_server/openai/adapters.py`.
fn refuse_tool_calls(req: &ChatCompletionRequest) -> Result<(), String> {
    let listed = |field: &str| match req.extra.get(field) {
        None | Some(Value::Null) => false,
        Some(Value::Array(t)) => !t.is_empty(),
        Some(_) => true,
    };
    let says_none = |field: &str| req.extra.get(field).and_then(Value::as_str) == Some("none");
    let choice = match req.extra.get("tool_choice") {
        Some(Value::String(c)) => c == "required",
        Some(Value::Object(_)) => true,
        _ => false,
    };
    let function_call = match req.extra.get("function_call") {
        None | Some(Value::Null) => false,
        Some(Value::String(c)) => c != "none" && c != "auto",
        Some(_) => true,
    };
    if (listed("tools") && !says_none("tool_choice"))
        || (listed("functions") && !says_none("function_call"))
        || choice
        || function_call
    {
        return Err("tool calling is not supported for this model".to_string());
    }
    Ok(())
}

/// Chat `audio.voice` wins over a top-level `voice` (Qwen3-Omni).
fn chat_voice(req: &ChatCompletionRequest) -> Option<String> {
    if let Some(Value::Object(a)) = &req.audio {
        if let Some(v) = a.get("voice").and_then(Value::as_str) {
            return Some(v.to_string());
        }
    }
    req.extra
        .get("voice")
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ChatMessage;

    type Flattened = (Option<String>, BTreeMap<String, Vec<String>>, Vec<String>, Vec<Part>);

    fn flatten(messages: &str, upload_dir: &Path) -> Result<Flattened, String> {
        let messages: Vec<ChatMessage> = serde_json::from_str(messages).unwrap();
        flatten_messages(&messages, upload_dir, false)
    }

    fn described(parts: &[Part]) -> Vec<(&str, &str, Option<&str>, usize)> {
        parts
            .iter()
            .map(|p| (p.modality.as_str(), p.role.as_deref().unwrap_or(""), p.text.as_deref(), p.index))
            .collect()
    }

    #[test]
    fn a_reply_stays_its_own_part() {
        let (text, _, in_mods, parts) = flatten(
            r#"[{"role":"system","content":"Be brief."},
                {"role":"user","content":"Name a color."},
                {"role":"assistant","content":"Blue."},
                {"role":"user","content":"Another."}]"#,
            Path::new("/unused"),
        )
        .unwrap();
        assert_eq!(text.as_deref(), Some("Be brief.\nName a color.\nBlue.\nAnother."));
        assert_eq!(in_mods, ["text", "text", "text", "text"]);
        assert_eq!(
            described(&parts),
            [
                ("text", "system", Some("Be brief."), 0),
                ("text", "user", Some("Name a color."), 0),
                ("text", "assistant", Some("Blue."), 0),
                ("text", "user", Some("Another."), 0),
            ],
            "text merged across a role change, so the reply lands inside a user turn"
        );
    }

    #[test]
    fn messages_of_one_role_stay_one_part() {
        let (_, _, in_mods, parts) = flatten(
            r#"[{"role":"user","content":"First."},{"role":"user","content":"Second."}]"#,
            Path::new("/unused"),
        )
        .unwrap();
        assert_eq!(in_mods, ["text"]);
        assert_eq!(
            described(&parts),
            [("text", "user", Some("First.\nSecond."), 0)],
            "two user messages no longer render as the one turn they do on the default server"
        );
    }

    #[test]
    fn a_developer_message_is_a_system_message() {
        let (_, _, _, parts) = flatten(
            r#"[{"role":"developer","content":"Be brief."},{"role":"user","content":"Hi."}]"#,
            Path::new("/unused"),
        )
        .unwrap();
        assert_eq!(
            described(&parts),
            [("text", "system", Some("Be brief."), 0), ("text", "user", Some("Hi."), 0)],
            "a developer message reached the templates under a role they do not render"
        );
    }

    #[test]
    fn a_role_a_chat_cannot_render_is_refused() {
        let err = flatten(r#"[{"role":"tool","content":"42"}]"#, Path::new("/unused")).unwrap_err();
        assert_eq!(err, "a message's role must be system, developer, user or assistant, not 'tool'");
    }

    /// The surfaces a class serves: chat, speech, images, videos.
    fn surfaces(adapter: &dyn OpenAIAdapter) -> [bool; 4] {
        [adapter.supports_chat(), adapter.supports_speech(), adapter.supports_images(), adapter.supports_videos()]
    }

    #[test]
    fn each_ported_python_adapter_class_is_found_by_its_name_once() {
        let mut names: Vec<&str> = ADAPTERS.iter().map(|(name, _)| *name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ADAPTERS.len(), "a class listed twice");
        for (class, adapter) in ADAPTERS {
            assert_eq!(surfaces(from_python(class).unwrap()), surfaces(*adapter), "{class}");
        }
        for class in ["", "WhisperAdapter", "qwen3_5_0.8b", "bagel"] {
            assert!(from_python(class).is_none(), "{class:?}");
        }
        // the surfaces of a few classes, as their Python attributes declare them
        assert_eq!(surfaces(&Bagel), [true, false, true, false]);
        assert_eq!(surfaces(&Qwen3Omni), [true, true, false, false]);
        assert_eq!(surfaces(&Qwen3_5), [true, false, false, false]);
        assert_eq!(surfaces(&Cosmos3), [false, false, true, true]);
        assert_eq!(surfaces(&Cosmos3Edge), [true, false, true, true]);
        assert_eq!(surfaces(&Qwen3TTS), [false, true, false, false]);
        assert_eq!(surfaces(&Orpheus), [false, true, false, false]);
    }

    fn speech(body: &str) -> SpeechRequest {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn a_qwen3_tts_speech_maps_instructions_sampling_and_a_sent_speed() {
        let args = Qwen3TTS
            .speech_args(&speech(
                r#"{"input":"Hello.","voice":"ryan","language":"English","instructions":"calm",
                    "temperature":0.7,"top_p":0.8,"seed":4,"speed":1.25,"top_k":20,"stream":true}"#,
            ))
            .unwrap();
        assert_eq!(args.text.as_deref(), Some("Hello."));
        assert_eq!((args.input_modalities, args.output_modalities), (vec!["text".to_string()], vec!["audio".to_string()]));
        assert!(args.file_paths.is_empty());
        assert_eq!(Value::Object(args.model_kwargs), serde_json::json!({
            "voice": "ryan", "language": "English", "instruct": "calm", "temperature": 0.7,
            "top_p": 0.8, "seed": 4, "speed": 1.25, "top_k": 20,
        }));
    }

    #[test]
    fn a_qwen3_tts_speech_keeps_an_explicit_instruct_and_drops_empty_instructions() {
        let mk = |body: &str| Value::Object(Qwen3TTS.speech_args(&speech(body)).unwrap().model_kwargs);
        assert_eq!(mk(r#"{"input":"x","instructions":"calm","instruct":"bright"}"#), serde_json::json!({"instruct": "bright"}));
        assert_eq!(mk(r#"{"input":"x","instructions":""}"#), serde_json::json!({}));
        assert_eq!(mk(r#"{"input":"x","speed":null}"#), serde_json::json!({"speed": null}));
        // an empty clip is dropped, as the Python adapter pops it
        let speech = Qwen3TTS.speech_to_request(&speech(r#"{"input":"x","ref_audio":""}"#), Path::new("/unused"), false);
        assert_eq!(Value::Object(speech.unwrap().args.model_kwargs), serde_json::json!({}));
    }

    #[test]
    fn every_speech_adapter_forwards_only_a_sent_speed() {
        for (name, adapter) in [("Qwen3TTS", &Qwen3TTS as &dyn OpenAIAdapter), ("Orpheus", &Orpheus), ("Qwen3Omni", &Qwen3Omni)] {
            let args = adapter.speech_args(&speech(r#"{"input":"x"}"#)).unwrap();
            assert!(!args.model_kwargs.contains_key("speed"), "{name}");
            let args = adapter.speech_args(&speech(r#"{"input":"x","speed":"2"}"#)).unwrap();
            assert_eq!(args.model_kwargs["speed"], serde_json::json!(2.0), "{name}");
        }
    }

    #[test]
    fn a_qwen3_tts_voice_clone_saves_its_clip_for_the_loader_to_check() {
        let dir = std::env::temp_dir().join(format!("mstar_ref_{}", uuid::Uuid::new_v4().simple()));
        let clone = |body: &str| Qwen3TTS.speech_to_request(&speech(body), &dir, false);
        for (body, bare) in [
            (r#"{"input":"x","ref_audio":"data:audio/wav;base64,UklGRg=="}"#, false),
            (r#"{"input":"x","ref_audio":"UklGRg"}"#, true), // bare and unpadded, as Python reads it
        ] {
            let Speech { args, ref_audio } = clone(body).unwrap();
            let check = ref_audio.unwrap();
            assert_eq!((check.decoder, check.bare), ("soundfile", bare), "{body}");
            assert_eq!(std::fs::read(&check.path).unwrap(), b"RIFF");
            assert_eq!(args.file_paths["audio"], vec![check.path.clone()]);
            assert_eq!(args.input_modalities, ["audio", "text"]);
            assert!(!args.model_kwargs.contains_key("ref_audio"));
        }
        for (body, want) in [
            (r#"{"input":"x","ref_audio":"/etc/passwd!"}"#, (400, crate::media::BARE_REF_AUDIO_REFUSAL.to_string())),
            (r#"{"input":"x","ref_audio":7}"#, (400, "ref_audio must be a data URL or base64 string, got int".into())),
            (r#"{"input":"x","ref_audio":"http://h/a.wav"}"#, (400, "Remote media fetch is disabled on this server".into())),
        ] {
            assert_eq!(clone(body).unwrap_err(), want, "{body}");
        }
        // no clip: a plain speech, nothing to check
        assert!(clone(r#"{"input":"x","ref_audio":""}"#).unwrap().ref_audio.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_qwen3_tts_chunks_long_speech_by_default() {
        assert_eq!(Qwen3TTS.speech_chunking().min_chars, Some(600));
        assert_eq!(Orpheus.speech_chunking().min_chars, None);
        for (name, adapter) in [("Qwen3TTS", &Qwen3TTS as &dyn OpenAIAdapter), ("Orpheus", &Orpheus), ("Qwen3Omni", &Qwen3Omni)] {
            let c = adapter.speech_chunking();
            assert_eq!((c.max_chars, c.lookahead, c.max_pieces), (400, 2, 32), "{name}");
        }
    }

    #[test]
    fn a_qwen3_5_chat_maps_the_standard_fields_and_passes_the_rest_through() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"海について"}],
                "temperature":0.7,"top_p":0.9,"max_tokens":32,"seed":5,
                "enable_thinking":false,"repetition_penalty":1.1,"max_output_tokens":8}"#,
        )
        .unwrap();
        let args = Qwen3_5.chat_to_request(&req, Path::new("/unused"), false).unwrap();
        assert_eq!(args.text.as_deref(), Some("Be brief.\n海について"));
        assert_eq!(args.input_modalities, ["text", "text"]);
        assert_eq!(args.output_modalities, ["text"]);
        let mk = Value::Object(args.model_kwargs);
        assert_eq!(mk, serde_json::json!({
            "temperature": 0.7, "top_p": 0.9, "seed": 5, "enable_thinking": false,
            "repetition_penalty": 1.1, "max_output_tokens": 8,
        }), "an explicit extra_body max_output_tokens wins over max_tokens");
    }

    #[test]
    fn a_cosmos3_edge_chat_reads_enable_thinking_from_the_template_kwargs() {
        let chat = |fields: &str| Cosmos3Edge.chat_to_request(&chat_with(fields), Path::new("/unused"), false);
        let thinking = |fields: &str| chat(fields).unwrap().model_kwargs.get("enable_thinking").cloned();
        for (fields, want) in [
            (r#""chat_template_kwargs":{"enable_thinking":false}"#, Some(Value::Bool(false))),
            // as sent: the model refuses what is not a boolean
            (r#""chat_template_kwargs":{"enable_thinking":1}"#, Some(serde_json::json!(1))),
            (r#""chat_template_kwargs":{"enable_thinking":"false"}"#, Some(serde_json::json!("false"))),
            (r#""chat_template_kwargs":{"enable_thinking":false},"enable_thinking":true"#, Some(Value::Bool(true))),
            (r#""chat_template_kwargs":{"other":1}"#, None),
            (r#""chat_template_kwargs":{}"#, None),
            (r#""chat_template_kwargs":null"#, None),
            (r#""chat_template_kwargs":0"#, None),
            (r#""chat_template_kwargs":["other"]"#, None),
            (r#""chat_template_kwargs":"other""#, None),
        ] {
            assert_eq!(thinking(fields), want, "{fields}");
            assert!(!chat(fields).unwrap().model_kwargs.contains_key("chat_template_kwargs"), "{fields}");
        }
        // what Python's TypeError says for a value it cannot look the key up in
        for (fields, err) in [
            (r#""chat_template_kwargs":["enable_thinking"]"#, "list indices must be integers or slices, not str"),
            (r#""chat_template_kwargs":"enable_thinking=1""#, "string indices must be integers, not 'str'"),
            (r#""chat_template_kwargs":true"#, "argument of type 'bool' is not iterable"),
            (r#""chat_template_kwargs":3"#, "argument of type 'int' is not iterable"),
            (r#""chat_template_kwargs":1.5"#, "argument of type 'float' is not iterable"),
        ] {
            assert_eq!(chat(fields).unwrap_err(), err, "{fields}");
        }
    }

    #[test]
    fn floats_are_quoted_as_python_reprs_them() {
        for (x, want) in [
            (0.5, "0.5"), (1.0, "1.0"), (-2.0, "-2.0"), (1e-05, "1e-05"), (0.0001, "0.0001"),
            (123.456, "123.456"), (1e16, "1e+16"), (1.5e16, "1.5e+16"), (9999999999999998.0, "9999999999999998.0"),
            (-3.25e-07, "-3.25e-07"), (0.0, "0.0"),
        ] {
            assert_eq!(super::py_float_repr(x), want, "{x}");
        }
        let v: Value = serde_json::from_str(r#"{"b": [true, null, "it's"], "a": 1}"#).unwrap();
        assert_eq!(super::py_value_repr(&v), r#"{'a': 1, 'b': [True, None, "it's"]}"#);
    }

    fn chat_with(fields: &str) -> ChatCompletionRequest {
        let body = format!(r#"{{"messages":[{{"role":"user","content":"Hi."}}],{fields}}}"#);
        serde_json::from_str(&body).unwrap()
    }

    #[test]
    fn a_chat_that_asks_for_a_tool_call_is_refused() {
        for fields in [
            r#""tools":[{"type":"function","function":{"name":"get_weather"}}]"#,
            r#""tool_choice":"required""#,
            r#""tool_choice":{"type":"function","function":{"name":"get_weather"}}"#,
            r#""functions":[{"name":"get_weather"}]"#,
            r#""function_call":{"name":"get_weather"}"#,
        ] {
            for (name, adapter) in ADAPTERS.iter().filter(|(_, a)| a.supports_chat()) {
                let err = adapter.chat_to_request(&chat_with(fields), Path::new("/unused"), false).unwrap_err();
                assert_eq!(err, "tool calling is not supported for this model", "{fields} reached {name}");
            }
        }
    }

    #[test]
    fn the_tool_fields_clients_send_by_default_pass() {
        for fields in [
            r#""tools":[]"#,
            r#""tools":null"#,
            r#""tool_choice":"none""#,
            r#""tool_choice":"auto""#,
            r#""tool_choice":null"#,
            r#""functions":[]"#,
            r#""functions":null"#,
            r#""function_call":"none""#,
            r#""function_call":"auto""#,
            r#""function_call":null"#,
            r#""tools":[{"type":"function","function":{"name":"get_weather"}}],"tool_choice":"none""#,
            r#""functions":[{"name":"get_weather"}],"function_call":"none""#,
        ] {
            for (name, adapter) in ADAPTERS.iter().filter(|(_, a)| a.supports_chat()) {
                let args = adapter.chat_to_request(&chat_with(fields), Path::new("/unused"), false);
                assert!(args.is_ok(), "{fields} refused a plain chat on {name}: {args:?}");
            }
        }
    }

    /// A fresh upload directory, removed on drop. Request media must be inline
    /// (a data URL), so the server persists it here.
    struct UploadDir(std::path::PathBuf);

    impl UploadDir {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("mstar-adapters-test-{}", uuid::Uuid::new_v4())))
        }
    }

    impl Drop for UploadDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// What each persisted upload holds, in `file_paths` order.
    fn contents(paths: &[String]) -> Vec<Vec<u8>> {
        paths.iter().map(|p| std::fs::read(p).unwrap()).collect()
    }

    // The bytes "a" and "b", so a test can tell which image went where.
    const IMAGE_A: &str = "data:image/png;base64,YQ==";
    const IMAGE_B: &str = "data:image/png;base64,Yg==";

    #[test]
    fn an_attachment_keeps_its_place_between_text() {
        let dir = UploadDir::new();
        let (text, file_paths, in_mods, parts) = flatten(
            &format!(
                r#"[{{"role":"user","content":[
                {{"type":"text","text":"A"}},
                {{"type":"image_url","image_url":{{"url":"{IMAGE_A}"}}}},
                {{"type":"text","text":"B"}}]}}]"#
            ),
            &dir.0,
        )
        .unwrap();
        assert_eq!(text.as_deref(), Some("A\nB"));
        assert_eq!(contents(&file_paths["image"]), [b"a"]);
        assert_eq!(in_mods, ["text", "image", "text"], "the layout lost the image's place");
        assert_eq!(
            described(&parts),
            [
                ("text", "user", Some("A"), 0),
                ("image", "user", None, 0),
                ("text", "user", Some("B"), 0),
            ],
            "text written after the image moved ahead of it"
        );
    }

    #[test]
    fn an_image_in_a_reply_lands_in_the_next_user_turn() {
        let dir = UploadDir::new();
        let (_, file_paths, in_mods, parts) = flatten(
            &format!(
                r#"[{{"role":"user","content":"Draw a red cube."}},
                {{"role":"assistant","content":[
                    {{"type":"text","text":"Here it is."}},
                    {{"type":"image_url","image_url":{{"url":"{IMAGE_A}"}}}}]}},
                {{"role":"user","content":"Make it blue."}}]"#
            ),
            &dir.0,
        )
        .unwrap();
        assert_eq!(contents(&file_paths["image"]), [b"a"]);
        assert_eq!(in_mods, ["text", "text", "image", "text"]);
        assert_eq!(
            described(&parts),
            [
                ("text", "user", Some("Draw a red cube."), 0),
                ("text", "assistant", Some("Here it is."), 0),
                ("image", "user", None, 0),
                ("text", "user", Some("Make it blue."), 0),
            ],
            "the reply's image did not land in the next user turn"
        );
    }

    #[test]
    fn two_images_are_indexed_in_order() {
        let dir = UploadDir::new();
        let (_, file_paths, in_mods, parts) = flatten(
            &format!(
                r#"[{{"role":"user","content":[
                {{"type":"image_url","image_url":{{"url":"{IMAGE_A}"}}}},
                {{"type":"image_url","image_url":{{"url":"{IMAGE_B}"}}}},
                {{"type":"text","text":"Compare."}}]}}]"#
            ),
            &dir.0,
        )
        .unwrap();
        assert_eq!(contents(&file_paths["image"]), [b"a", b"b"]);
        assert_eq!(in_mods, ["image", "image", "text"]);
        assert_eq!(
            parts.iter().map(|p| p.index).collect::<Vec<_>>(),
            [0, 1, 0],
            "the second image does not address the second upload"
        );
    }
}
