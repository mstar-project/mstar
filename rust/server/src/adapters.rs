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
    let mut file_paths: BTreeMap<String, Vec<String>> = BTreeMap::new();

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
                        add_file(&mut parts, &mut file_paths, m, p);
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
                            add_file(&mut parts, &mut file_paths, m, p);
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
    Ok((text, file_paths, input_modalities, parts))
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
    Cosmos3,
}

impl Adapter {
    pub fn from_model_name(name: &str) -> Option<Adapter> {
        match name {
            "bagel" => Some(Adapter::Bagel),
            "qwen3_omni" => Some(Adapter::Qwen3Omni),
            "orpheus" => Some(Adapter::Orpheus),
            // NVIDIA Cosmos3 registers three checkpoints against one adapter.
            "cosmos3" | "cosmos3_droid" | "cosmos3_super" => Some(Adapter::Cosmos3),
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
        matches!(self, Adapter::Bagel | Adapter::Cosmos3)
    }

    pub fn supports_videos(&self) -> bool {
        matches!(self, Adapter::Cosmos3)
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
                let (text, file_paths, in_mods, parts) =
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
                    parts,
                })
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
            Adapter::Bagel | Adapter::Cosmos3 => {
                Err("audio/speech is not supported by this model".to_string())
            }
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
                    parts: Vec::new(),
                })
            }
            Adapter::Cosmos3 => {
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
            Adapter::Cosmos3 => {
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

    #[test]
    fn an_attachment_keeps_its_place_between_text() {
        let (text, file_paths, in_mods, parts) = flatten(
            r#"[{"role":"user","content":[
                {"type":"text","text":"A"},
                {"type":"image_url","image_url":{"url":"/in/a.png"}},
                {"type":"text","text":"B"}]}]"#,
            Path::new("/unused"),
        )
        .unwrap();
        assert_eq!(text.as_deref(), Some("A\nB"));
        assert_eq!(file_paths["image"], ["/in/a.png"]);
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
        let (_, file_paths, in_mods, parts) = flatten(
            r#"[{"role":"user","content":"Draw a red cube."},
                {"role":"assistant","content":[
                    {"type":"text","text":"Here it is."},
                    {"type":"image_url","image_url":{"url":"/in/a.png"}}]},
                {"role":"user","content":"Make it blue."}]"#,
            Path::new("/unused"),
        )
        .unwrap();
        assert_eq!(file_paths["image"], ["/in/a.png"]);
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
        let (_, file_paths, in_mods, parts) = flatten(
            r#"[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"/in/a.png"}},
                {"type":"image_url","image_url":{"url":"/in/b.png"}},
                {"type":"text","text":"Compare."}]}]"#,
            Path::new("/unused"),
        )
        .unwrap();
        assert_eq!(file_paths["image"], ["/in/a.png", "/in/b.png"]);
        assert_eq!(in_mods, ["image", "image", "text"]);
        assert_eq!(
            parts.iter().map(|p| p.index).collect::<Vec<_>>(),
            [0, 1, 0],
            "the second image does not address the second upload"
        );
    }
}
