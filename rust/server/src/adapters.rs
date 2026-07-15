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

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Map, Value};

use crate::media;
use crate::protocol::{ChatCompletionRequest, Content, ImageGenerationRequest, SpeechRequest};

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
        }
    }
}

/// Flatten OpenAI chat `messages` into (text, file_paths, input_modalities).
///
/// Text parts across all messages are newline-joined. Image/audio/video content
/// parts are persisted under `upload_dir` and grouped by modality. Multi-turn
/// role structure is flattened (a v1 simplification, as in mstar — the models
/// apply their own prompt formatting downstream).
pub fn flatten_messages(
    messages: &[crate::protocol::ChatMessage],
    upload_dir: &Path,
    allow_remote: bool,
) -> Result<(Option<String>, BTreeMap<String, Vec<String>>, Vec<String>), String> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut file_paths: BTreeMap<String, Vec<String>> = BTreeMap::new();

    let add_file = |modality: String, path: String, fp: &mut BTreeMap<String, Vec<String>>| {
        fp.entry(modality).or_default().push(path);
    };

    for msg in messages {
        let content = match &msg.content {
            None => continue,
            Some(Content::Text(s)) => {
                if !s.is_empty() {
                    text_parts.push(s.clone());
                }
                continue;
            }
            Some(Content::Parts(parts)) => parts,
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
                            text_parts.push(t.to_string());
                        }
                    }
                }
                "image_url" => {
                    let url = nested_url(obj, "image_url");
                    if !url.is_empty() {
                        let (m, p) = media::resolve_media_ref(&url, upload_dir, allow_remote)?;
                        let m = if m == "unknown" { "image".to_string() } else { m };
                        add_file(m, p, &mut file_paths);
                    }
                }
                "video_url" => {
                    let url = nested_url(obj, "video_url");
                    if !url.is_empty() {
                        let (m, p) = media::resolve_media_ref(&url, upload_dir, allow_remote)?;
                        let m = if m == "unknown" { "video".to_string() } else { m };
                        add_file(m, p, &mut file_paths);
                    }
                }
                "audio_url" => {
                    let url = nested_url(obj, "audio_url");
                    if !url.is_empty() {
                        let (m, p) = media::resolve_media_ref(&url, upload_dir, allow_remote)?;
                        let m = if m == "unknown" { "audio".to_string() } else { m };
                        add_file(m, p, &mut file_paths);
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
                            add_file(m, p, &mut file_paths);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let mut input_modalities: Vec<String> = file_paths.keys().cloned().collect();
    let text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join("\n"))
    };
    if text.is_some() {
        input_modalities.push("text".to_string());
    }
    Ok((text, file_paths, input_modalities))
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

/// Which OpenAI surfaces a model serves + the request translation. Ported from
/// the `OpenAIAdapter` subclasses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adapter {
    Bagel,
    Qwen3Omni,
    Orpheus,
}

impl Adapter {
    pub fn from_model_name(name: &str) -> Option<Adapter> {
        match name {
            "bagel" => Some(Adapter::Bagel),
            "qwen3_omni" => Some(Adapter::Qwen3Omni),
            "orpheus" => Some(Adapter::Orpheus),
            _ => None,
        }
    }

    pub fn supports_chat(&self) -> bool {
        matches!(self, Adapter::Bagel | Adapter::Qwen3Omni)
    }

    pub fn supports_speech(&self) -> bool {
        matches!(self, Adapter::Qwen3Omni | Adapter::Orpheus)
    }

    pub fn supports_images(&self) -> bool {
        matches!(self, Adapter::Bagel)
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
        match self {
            Adapter::Bagel => {
                let (text, file_paths, in_mods) =
                    flatten_messages(&req.messages, upload_dir, allow_remote)?;
                let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
                // BAGEL reads sampling from model config; only max/seed are honored.
                apply_sampling(
                    &mut mk,
                    Sampling {
                        temperature: req.temperature,
                        top_p: req.top_p,
                        seed: req.seed,
                        max_tokens: req.max_completion_tokens.or(req.max_tokens),
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
                })
            }
            Adapter::Qwen3Omni => {
                let (text, file_paths, in_mods) =
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
                        max_tokens: req.max_completion_tokens.or(req.max_tokens),
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
                })
            }
            Adapter::Orpheus => Err("chat is not supported by this model".to_string()),
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
                })
            }
            Adapter::Bagel => Err("audio/speech is not supported by this model".to_string()),
        }
    }

    pub fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        match self {
            Adapter::Bagel => {
                let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
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
                })
            }
            _ => Err("image generation is not supported by this model".to_string()),
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
                let mut file_paths = BTreeMap::new();
                file_paths.insert("image".to_string(), vec![image_path.to_string()]);
                Ok(SubmitArgs {
                    tokens: None,
                    text: Some(prompt.to_string()),
                    file_paths,
                    input_modalities: vec!["image".to_string(), "text".to_string()],
                    output_modalities: vec!["image".to_string()],
                    model_kwargs: extra_kwargs,
                })
            }
            _ => Err("image editing is not supported by this model".to_string()),
        }
    }
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
