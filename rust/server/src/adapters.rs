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
//! [`Adapter::from_python`] maps each class with a port here to its variant.

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
    pub role: String,
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
                        let fallback = ptype.trim_end_matches("_url");
                        let (m, p) = media::resolve_media_ref(&url, upload_dir, allow_remote)?;
                        let m = if m == "unknown" { fallback.to_string() } else { m };
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
        role: "user".to_string(),
    });
    paths.push(path);
}

fn add_text(parts: &mut Vec<Part>, text: &str, role: &str) {
    // Adjacent text parts were newline-joined before ordering was kept; merge
    // them here, across messages of one role too (the layout has no slot for
    // the boundary between them), so only text an attachment or a role change
    // separates gets its own part.
    if let Some(last) = parts.last_mut() {
        if last.modality == "text" && last.role == role {
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
        role: role.to_string(),
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
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<i64>,
    max_tokens: Option<i64>,
    temperature_key: &'a str,
    top_p_key: &'a str,
    /// `None` disables `max_tokens` mapping (e.g. speech has no such field).
    max_tokens_key: Option<&'a str>,
}

fn apply_sampling(mk: &mut Map<String, Value>, s: Sampling) {
    if let Some(t) = s.temperature {
        set_default(mk, s.temperature_key, json_num(t));
    }
    if let Some(p) = s.top_p {
        set_default(mk, s.top_p_key, json_num(p));
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

/// Which OpenAI surfaces a model serves + the request translation. One
/// variant per ported `OpenAIAdapter` subclass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adapter {
    Bagel,
    Qwen3Omni,
    Qwen3_5,
    Orpheus,
    Cosmos3,
    Cosmos3Edge,
}

impl Adapter {
    /// The variant for a Python adapter class name (`MSTAR_OPENAI_ADAPTER`);
    /// `None` for a class with no port here.
    pub fn from_python(class: &str) -> Option<Adapter> {
        match class {
            "BagelAdapter" => Some(Adapter::Bagel),
            "Qwen3OmniAdapter" => Some(Adapter::Qwen3Omni),
            "Qwen3_5Adapter" => Some(Adapter::Qwen3_5),
            "OrpheusAdapter" => Some(Adapter::Orpheus),
            "Cosmos3Adapter" => Some(Adapter::Cosmos3),
            "Cosmos3EdgeAdapter" => Some(Adapter::Cosmos3Edge),
            _ => None,
        }
    }

    pub fn supports_chat(&self) -> bool {
        matches!(self, Adapter::Bagel | Adapter::Qwen3Omni | Adapter::Qwen3_5 | Adapter::Cosmos3Edge)
    }

    pub fn supports_speech(&self) -> bool {
        matches!(self, Adapter::Qwen3Omni | Adapter::Orpheus)
    }

    pub fn supports_images(&self) -> bool {
        matches!(self, Adapter::Bagel | Adapter::Cosmos3 | Adapter::Cosmos3Edge)
    }

    pub fn supports_videos(&self) -> bool {
        matches!(self, Adapter::Cosmos3 | Adapter::Cosmos3Edge)
    }

    /// Whether this model's prompt processing is expressible in the frontend
    /// (plain `tokenizer.json`, no custom processor), so the server should
    /// tokenize + detokenize in Rust and submit token ids — the
    /// Rust-default-with-model-override capability flag. Requires the server
    /// to be started with a tokenizer (`MSTAR_TOKENIZER`). Currently false for
    /// every registered model: qwen3_omni needs its multimodal processor +
    /// chat template, bagel/orpheus their own prompt formatting — they keep
    /// the ship-text path. A model opts in here when its processing is a pure
    /// tokenizer encode (the mechanism is exercised by /generate's
    /// `tokenize` field and the echo verification).
    pub fn frontend_tokenizes(&self) -> bool {
        false
    }

    pub fn chat_to_request(
        &self,
        req: &ChatCompletionRequest,
        upload_dir: &Path,
        allow_remote: bool,
    ) -> Result<SubmitArgs, String> {
        refuse_tool_calls(req)?;
        match self {
            // BAGEL reads temperature / top_p from its model config; only
            // max_output_tokens and seed are honored.
            Adapter::Bagel | Adapter::Qwen3_5 => text_chat(req, upload_dir, allow_remote),
            Adapter::Cosmos3Edge => {
                // How the vLLM recipe toggles thinking; a flat enable_thinking wins.
                // Read before the attachments are saved, since a malformed value
                // refuses the request.
                let thinking = template_enable_thinking(req.extra.get("chat_template_kwargs"))?;
                let mut args = text_chat(req, upload_dir, allow_remote)?;
                args.model_kwargs.remove("chat_template_kwargs");
                if let Some(on) = thinking {
                    set_default(&mut args.model_kwargs, "enable_thinking", Value::Bool(on));
                }
                Ok(args)
            }
            Adapter::Qwen3Omni => {
                let (text, file_paths, in_mods, parts) =
                    flatten_messages(&req.messages, upload_dir, allow_remote)?;
                let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
                // Speech output also emits text, so request both when audio asked.
                let want_audio = req
                    .modalities
                    .as_ref()
                    .map(|m| m.iter().any(|x| x == "audio"))
                    .unwrap_or(false);
                let out_mods = if want_audio {
                    vec!["text".to_string(), "audio".to_string()]
                } else {
                    vec!["text".to_string()]
                };
                apply_sampling(
                    &mut mk,
                    Sampling {
                        temperature: req.temperature,
                        top_p: req.top_p,
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
                    output_modalities: out_mods,
                    model_kwargs: mk,
                    parts,
                })
            }
            Adapter::Orpheus | Adapter::Cosmos3 => {
                Err("chat is not supported by this model".to_string())
            }
        }
    }

    pub fn speech_to_request(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        match self {
            Adapter::Qwen3Omni => {
                let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
                if let Some(v) = &req.voice {
                    mk.insert("voice".to_string(), Value::from(v.clone()));
                }
                apply_sampling(
                    &mut mk,
                    Sampling {
                        temperature: req.temperature,
                        top_p: req.top_p,
                        seed: req.seed,
                        max_tokens: None,
                        temperature_key: "talker_temperature",
                        top_p_key: "talker_top_p",
                        max_tokens_key: None,
                    },
                );
                Ok(SubmitArgs {
                    tokens: None,
                    text: Some(req.input.clone()),
                    file_paths: BTreeMap::new(),
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string(), "audio".to_string()],
                    model_kwargs: mk,
                    parts: Vec::new(),
                })
            }
            Adapter::Orpheus => {
                let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
                if let Some(v) = &req.voice {
                    mk.insert("voice".to_string(), Value::from(v.clone()));
                }
                apply_sampling(
                    &mut mk,
                    Sampling {
                        temperature: req.temperature,
                        top_p: req.top_p,
                        seed: req.seed,
                        max_tokens: None,
                        temperature_key: "temperature",
                        top_p_key: "top_p",
                        max_tokens_key: None,
                    },
                );
                Ok(SubmitArgs {
                    tokens: None,
                    text: Some(req.input.clone()),
                    file_paths: BTreeMap::new(),
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["audio".to_string()],
                    model_kwargs: mk,
                    parts: Vec::new(),
                })
            }
            Adapter::Bagel | Adapter::Qwen3_5 | Adapter::Cosmos3 | Adapter::Cosmos3Edge => {
                Err("audio/speech is not supported by this model".to_string())
            }
        }
    }

    pub fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        match self {
            Adapter::Bagel => {
                let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
                size_kwargs(&mut mk, req.size.as_deref())?;
                if let Some(seed) = req.seed {
                    set_default(&mut mk, "seed", Value::from(seed));
                }
                Ok(SubmitArgs {
                    tokens: None,
                    text: Some(req.prompt.clone()),
                    file_paths: BTreeMap::new(),
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["image".to_string()],
                    model_kwargs: mk,
                    parts: Vec::new(),
                })
            }
            Adapter::Cosmos3 | Adapter::Cosmos3Edge => {
                let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
                if let Some(size) = &req.size {
                    set_default(&mut mk, "size", Value::from(size.clone()));
                }
                if let Some(seed) = req.seed {
                    set_default(&mut mk, "seed", Value::from(seed));
                }
                Ok(SubmitArgs {
                    tokens: None,
                    text: Some(req.prompt.clone()),
                    file_paths: BTreeMap::new(),
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["image".to_string()],
                    model_kwargs: mk,
                    parts: Vec::new(),
                })
            }
            _ => Err("image generation is not supported by this model".to_string()),
        }
    }

    /// `/v1/videos/generations` (Cosmos3). Text-to-video, or image/video-to-video
    /// when a conditioning reference is supplied. Mirrors the Python
    /// `Cosmos3Adapter.video_to_request`.
    pub fn video_to_request(
        &self,
        req: &VideoGenerationRequest,
        upload_dir: &Path,
        allow_remote: bool,
    ) -> Result<SubmitArgs, String> {
        match self {
            Adapter::Cosmos3 | Adapter::Cosmos3Edge => {
                let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
                if let Some(size) = &req.size {
                    set_default(&mut mk, "size", Value::from(size.clone()));
                }
                if let Some(seed) = req.seed {
                    set_default(&mut mk, "seed", Value::from(seed));
                }
                // num_frames / fps are first-class video fields (not extra_body).
                if let Some(nf) = req.num_frames {
                    set_default(&mut mk, "num_frames", Value::from(nf));
                }
                if let Some(fps) = req.fps {
                    set_default(&mut mk, "fps", json_num(fps));
                }
                // The conditioning frame (image-to-video) or clip (video-to-video)
                // is persisted and VAE-encoded by the worker into the clean
                // frame-0 anchor / pinned latent prefix. At most one may be given.
                match (req.image.as_deref(), req.video.as_deref()) {
                    (Some(_), Some(_)) => Err(
                        "Provide either 'image' or 'video' conditioning, not both.".to_string(),
                    ),
                    (Some(image), None) => {
                        let (_, path) = media::resolve_media_ref(image, upload_dir, allow_remote)?;
                        let mut file_paths = BTreeMap::new();
                        file_paths.insert("image".to_string(), vec![path]);
                        Ok(SubmitArgs {
                            tokens: None,
                            text: Some(req.prompt.clone()),
                            file_paths,
                            input_modalities: vec!["image".to_string(), "text".to_string()],
                            output_modalities: vec!["video".to_string()],
                            model_kwargs: mk,
                            parts: Vec::new(),
                        })
                    }
                    (None, Some(video)) => {
                        let (_, path) = media::resolve_media_ref(video, upload_dir, allow_remote)?;
                        let mut file_paths = BTreeMap::new();
                        file_paths.insert("video".to_string(), vec![path]);
                        Ok(SubmitArgs {
                            tokens: None,
                            text: Some(req.prompt.clone()),
                            file_paths,
                            input_modalities: vec!["video".to_string(), "text".to_string()],
                            output_modalities: vec!["video".to_string()],
                            model_kwargs: mk,
                            parts: Vec::new(),
                        })
                    }
                    (None, None) => Ok(SubmitArgs {
                        tokens: None,
                        text: Some(req.prompt.clone()),
                        file_paths: BTreeMap::new(),
                        input_modalities: vec!["text".to_string()],
                        output_modalities: vec!["video".to_string()],
                        model_kwargs: mk,
                        parts: Vec::new(),
                    }),
                }
            }
            _ => Err("video generation is not supported by this model".to_string()),
        }
    }

    pub fn image_edit_to_request(
        &self,
        prompt: &str,
        image_path: &str,
        extra_kwargs: Map<String, Value>,
    ) -> Result<SubmitArgs, String> {
        match self {
            Adapter::Bagel => {
                let mut extra_kwargs = extra_kwargs;
                // a form field that parsed as JSON (e.g. "512") is still the size text
                let size = extra_kwargs
                    .remove("size")
                    .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string));
                size_kwargs(&mut extra_kwargs, size.as_deref())?;
                let mut file_paths = BTreeMap::new();
                file_paths.insert("image".to_string(), vec![image_path.to_string()]);
                Ok(SubmitArgs {
                    tokens: None,
                    text: Some(prompt.to_string()),
                    file_paths,
                    input_modalities: vec!["image".to_string(), "text".to_string()],
                    output_modalities: vec!["image".to_string()],
                    model_kwargs: extra_kwargs,
                    parts: Vec::new(),
                })
            }
            _ => Err("image editing is not supported by this model".to_string()),
        }
    }
}

/// `max_completion_tokens or max_tokens`, as Python reads them: a 0 falls
/// through to `max_tokens`.
fn chat_max_tokens(req: &ChatCompletionRequest) -> Option<i64> {
    req.max_completion_tokens.filter(|&n| n != 0).or(req.max_tokens)
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
            temperature: req.temperature,
            top_p: req.top_p,
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

/// `chat_template_kwargs.enable_thinking` read as `Cosmos3EdgeAdapter` does:
/// `(kwargs or {})`, then `bool(kwargs["enable_thinking"])` when it has the
/// key. A falsy value is no kwargs; a value Python cannot look the key up in
/// is the `TypeError` that answers 400 there.
fn template_enable_thinking(kwargs: Option<&Value>) -> Result<Option<bool>, String> {
    const KEY: &str = "enable_thinking";
    match kwargs {
        None => Ok(None),
        Some(v) if !truthy(v) => Ok(None),
        Some(Value::Object(m)) => Ok(m.get(KEY).map(truthy)),
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
            .map(|p| (p.modality.as_str(), p.role.as_str(), p.text.as_deref(), p.index))
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

    const CHAT_ADAPTERS: [Adapter; 4] =
        [Adapter::Bagel, Adapter::Qwen3Omni, Adapter::Qwen3_5, Adapter::Cosmos3Edge];

    #[test]
    fn each_ported_python_adapter_class_has_its_variant() {
        for (class, adapter) in [
            ("BagelAdapter", Adapter::Bagel),
            ("Qwen3OmniAdapter", Adapter::Qwen3Omni),
            ("Qwen3_5Adapter", Adapter::Qwen3_5),
            ("OrpheusAdapter", Adapter::Orpheus),
            ("Cosmos3Adapter", Adapter::Cosmos3),
            ("Cosmos3EdgeAdapter", Adapter::Cosmos3Edge),
        ] {
            assert_eq!(Adapter::from_python(class), Some(adapter));
        }
        for class in ["", "WhisperAdapter", "qwen3_5_0.8b", "bagel"] {
            assert_eq!(Adapter::from_python(class), None, "{class:?}");
        }
        assert!(CHAT_ADAPTERS.iter().all(Adapter::supports_chat));
        assert!(!Adapter::Orpheus.supports_chat() && !Adapter::Cosmos3.supports_chat());
        assert!(Adapter::Cosmos3Edge.supports_images() && Adapter::Cosmos3Edge.supports_videos());
        let q = Adapter::Qwen3_5;
        assert!(!q.supports_speech() && !q.supports_images() && !q.supports_videos());
    }

    #[test]
    fn a_qwen3_5_chat_maps_the_standard_fields_and_passes_the_rest_through() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"海について"}],
                "temperature":0.7,"top_p":0.9,"max_tokens":32,"seed":5,
                "enable_thinking":false,"repetition_penalty":1.1,"max_output_tokens":8}"#,
        )
        .unwrap();
        let args = Adapter::Qwen3_5.chat_to_request(&req, Path::new("/unused"), false).unwrap();
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
        let chat = |fields: &str| Adapter::Cosmos3Edge.chat_to_request(&chat_with(fields), Path::new("/unused"), false);
        let thinking = |fields: &str| chat(fields).unwrap().model_kwargs.get("enable_thinking").cloned();
        for (fields, want) in [
            (r#""chat_template_kwargs":{"enable_thinking":false}"#, Some(Value::Bool(false))),
            (r#""chat_template_kwargs":{"enable_thinking":1}"#, Some(Value::Bool(true))),
            (r#""chat_template_kwargs":{"enable_thinking":""}"#, Some(Value::Bool(false))),
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
            for adapter in CHAT_ADAPTERS {
                let err = adapter.chat_to_request(&chat_with(fields), Path::new("/unused"), false).unwrap_err();
                assert_eq!(err, "tool calling is not supported for this model", "{fields} reached {adapter:?}");
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
            for adapter in CHAT_ADAPTERS {
                let args = adapter.chat_to_request(&chat_with(fields), Path::new("/unused"), false);
                assert!(args.is_ok(), "{fields} refused a plain chat on {adapter:?}");
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
