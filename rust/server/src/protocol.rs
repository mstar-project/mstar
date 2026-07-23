//! Request/response models for the OpenAI-compatible endpoints.
//!
//! Ported from mstar's `api_server/openai/protocol.py`. Requests validate the
//! standard OpenAI fields and keep every unknown field as passthrough
//! `model_kwargs` (pydantic's `extra="allow"`) — here that's a `#[serde(flatten)]`
//! catch-all map, so the OpenAI client's `extra_body` flows through to the
//! model verbatim. Responses are built as plain JSON in the serving handlers to
//! keep the multimodal shapes (audio in `message.audio`, images as data URLs)
//! flexible, exactly as the Python side does.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// One chat message. `content` is either a plain string or an array of
/// multimodal content parts (`text` / `image_url` / `audio_url` / `input_audio`
/// / `video_url`); both are accepted, matching `str | list[dict] | None`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    #[allow(dead_code)] // role is flattened away (v1 simplification, as in mstar)
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
}

/// `content` may be a bare string or a list of content-part objects.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Value>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[allow(dead_code)] // accepted for OpenAI compatibility; the loaded model is fixed
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub max_completion_tokens: Option<i64>,
    /// Accepted for OpenAI compatibility (declared so it does NOT leak into
    /// `extra` -> model_kwargs); single-choice responses only, as in mstar.
    #[allow(dead_code)]
    #[serde(default)]
    pub n: Option<i64>,
    /// Accepted for OpenAI compatibility; stop sequences are not applied.
    #[allow(dead_code)]
    #[serde(default)]
    pub stop: Option<Value>, // str | [str]
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stream: bool,

    // Multimodal output (vllm-omni / sglang-omni style).
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
    #[serde(default)]
    pub audio: Option<Value>, // {"voice": ..., "format": "wav"}

    /// Unknown fields flow through verbatim as model_kwargs (extra_body).
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// OpenAI `/v1/audio/speech` (text-to-speech).
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechRequest {
    pub input: String,
    #[allow(dead_code)] // accepted for OpenAI compatibility; the loaded model is fixed
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default = "default_wav")]
    pub response_format: String,
    /// Accepted for OpenAI compatibility (kept out of `extra`); playback-rate
    /// adjustment is not applied, as in mstar.
    #[allow(dead_code)]
    #[serde(default)]
    pub speed: Option<f64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub seed: Option<i64>,

    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// OpenAI `/v1/images/generations`.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    #[allow(dead_code)] // accepted for OpenAI compatibility; the loaded model is fixed
    #[serde(default)]
    pub model: Option<String>,
    /// Accepted for OpenAI compatibility (kept out of `extra`), as in mstar.
    #[allow(dead_code)]
    #[serde(default)]
    pub n: Option<i64>,
    #[allow(dead_code)]
    #[serde(default)]
    pub size: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub seed: Option<i64>,

    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

fn default_wav() -> String {
    "wav".to_string()
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

impl ModelCard {
    pub fn new(id: impl Into<String>, created: i64) -> Self {
        Self {
            id: id.into(),
            object: "model".to_string(),
            created,
            owned_by: "mstar".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelCard>,
}

impl ModelList {
    pub fn new(data: Vec<ModelCard>) -> Self {
        Self {
            object: "list".to_string(),
            data,
        }
    }
}
