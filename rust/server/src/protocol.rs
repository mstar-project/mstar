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

/// Pydantic-style lax coercion for the *declared* numeric/bool fields. FastAPI
/// accepts string- and integral-float-encoded scalars (`"16"`, `16.0`,
/// `"true"`) that serde would reject with a hard 422 — anything routed through
/// a form, env var, YAML config, or loosely-typed SDK sends strings. These
/// `deserialize_with` helpers accept those forms on the declared fields (the
/// passthrough `extra` map stays verbatim). An absent field never reaches the
/// helper (`#[serde(default)]` yields `None`); a present `null` maps to `None`.
mod flex {
    use serde::{de::Error, Deserialize, Deserializer};
    use serde_json::Value;

    /// `f` as an i64 when integral and in range; `as` alone would saturate `1e20` to `i64::MAX`.
    fn float_to_i64(f: f64) -> Option<i64> {
        const BOUND: f64 = 9_223_372_036_854_775_808.0; // 2^63
        (f.is_finite() && f.fract() == 0.0 && (-BOUND..BOUND).contains(&f)).then_some(f as i64)
    }

    /// An integer literal with an optional zero fraction (`"5"`, `"-3"`, `"5.0"`);
    /// no exponent, as pydantic.
    fn parse_integral(t: &str) -> Option<i64> {
        let (int, frac) = t.split_once('.').unwrap_or((t, ""));
        if frac.bytes().all(|b| b == b'0') {
            int.parse::<i64>().ok()
        } else {
            None
        }
    }

    pub fn opt_i64<'de, D: Deserializer<'de>>(d: D) -> Result<Option<i64>, D::Error> {
        match Option::<Value>::deserialize(d)? {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) => n
                .as_i64()
                .or_else(|| n.as_f64().and_then(float_to_i64))
                .map(Some)
                .ok_or_else(|| Error::custom(format!("expected an integer, got {n}"))),
            Some(Value::String(s)) => {
                parse_integral(s.trim())
                    .map(Some)
                    .ok_or_else(|| Error::custom(format!("expected an integer, got {s:?}")))
            }
            Some(other) => Err(Error::custom(format!("expected an integer, got {other}"))),
        }
    }

    pub fn opt_bool<'de, D: Deserializer<'de>>(d: D) -> Result<Option<bool>, D::Error> {
        match Option::<Value>::deserialize(d)? {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(b)),
            Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
                "true" | "t" | "yes" | "y" | "on" | "1" => Ok(Some(true)),
                "false" | "f" | "no" | "n" | "off" | "0" => Ok(Some(false)),
                _ => Err(Error::custom(format!("expected a boolean, got {s:?}"))),
            },
            Some(Value::Number(n)) => match n.as_i64() {
                Some(0) => Ok(Some(false)),
                Some(1) => Ok(Some(true)),
                _ => Err(Error::custom(format!("expected a boolean, got {n}"))),
            },
            Some(other) => Err(Error::custom(format!("expected a boolean, got {other}"))),
        }
    }
}

/// One chat message. `content` is either a plain string or an array of
/// multimodal content parts (`text` / `image_url` / `audio_url` / `input_audio`
/// / `video_url`); both are accepted, matching `str | list[dict] | None`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    // Required, matching Python's pydantic `ChatMessage` (a message with no
    // `role` is a 422 there).
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
    /// Passed on as sent; the server's normalize_generation_kwargs holds the rules.
    #[serde(default)]
    pub temperature: Option<Value>,
    #[serde(default)]
    pub top_p: Option<Value>,
    #[serde(default, deserialize_with = "flex::opt_i64")]
    pub max_tokens: Option<i64>,
    #[serde(default, deserialize_with = "flex::opt_i64")]
    pub max_completion_tokens: Option<i64>,
    /// Accepted for OpenAI compatibility (declared so it does NOT leak into
    /// `extra` -> model_kwargs); single-choice responses only, as in mstar.
    #[allow(dead_code)]
    #[serde(default, deserialize_with = "flex::opt_i64")]
    pub n: Option<i64>,
    /// Accepted for OpenAI compatibility; stop sequences are not applied.
    #[allow(dead_code)]
    #[serde(default)]
    pub stop: Option<Value>, // str | [str]
    #[serde(default, deserialize_with = "flex::opt_i64")]
    pub seed: Option<i64>,
    /// `bool | None` in Python (default False); accept an explicit `null`
    /// instead of hard-erroring. Absent / null / false all mean non-streaming.
    #[serde(default, deserialize_with = "flex::opt_bool")]
    pub stream: Option<bool>,

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
    /// Forwarded as sent when present; the model's request_kwargs reads or logs it.
    #[serde(default)]
    pub speed: Option<Value>,
    /// `bool | None` in Python (default False); accept an explicit `null`
    /// instead of hard-erroring. Absent / null / false all mean non-streaming.
    #[serde(default, deserialize_with = "flex::opt_bool")]
    pub stream: Option<bool>,
    /// Passed on as sent; the server's normalize_generation_kwargs holds the rules.
    #[serde(default)]
    pub temperature: Option<Value>,
    #[serde(default)]
    pub top_p: Option<Value>,
    #[serde(default, deserialize_with = "flex::opt_i64")]
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
    /// OpenAI `n`: number of images to generate (submitted as n engine requests).
    #[serde(default, deserialize_with = "flex::opt_i64")]
    pub n: Option<i64>,
    #[serde(default)]
    pub size: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default, deserialize_with = "flex::opt_i64")]
    pub seed: Option<i64>,

    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `/v1/videos/generations` (text/image/video-to-video). Not an OpenAI-standard
/// surface; modeled on the image endpoint, mirroring `VideoGenerationRequest`.
/// `image` conditions image-to-video, `video` conditions video-to-video (URL or
/// data URI). Extra knobs (guidance_scale, num_inference_steps, negative_prompt,
/// condition_frame_indexes_vision, condition_video_keep …) flow through `extra`.
#[derive(Debug, Clone, Deserialize)]
pub struct VideoGenerationRequest {
    pub prompt: String,
    #[allow(dead_code)] // accepted for OpenAI compatibility; the loaded model is fixed
    #[serde(default)]
    pub model: Option<String>,
    #[allow(dead_code)]
    #[serde(default, deserialize_with = "flex::opt_i64")]
    pub n: Option<i64>,
    #[serde(default)]
    pub size: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default, deserialize_with = "flex::opt_i64")]
    pub seed: Option<i64>,
    /// First-class video fields (not passed through `extra`), as in mstar.
    #[serde(default, deserialize_with = "flex::opt_i64")]
    pub num_frames: Option<i64>,
    /// Passed on as sent, like the sampling floats.
    #[serde(default)]
    pub fps: Option<Value>,
    /// URL or data URI conditioning frame (image-to-video).
    #[serde(default)]
    pub image: Option<String>,
    /// URL or data URI conditioning clip (video-to-video).
    #[serde(default)]
    pub video: Option<String>,

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(body: &str) -> ChatCompletionRequest {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn coerces_stringy_scalars_like_pydantic() {
        // String- and integral-float-encoded scalars are accepted (forms / env
        // vars / YAML / loosely-typed SDKs send strings), matching FastAPI.
        let r = chat(
            r#"{"messages":[{"role":"user","content":"hi"}],
                "stream":"true","temperature":"0.5","top_p":"0.9",
                "max_tokens":"16","n":"2","seed":"7"}"#,
        );
        assert_eq!(r.stream, Some(true));
        assert_eq!(r.temperature, Some(json!("0.5")));
        assert_eq!(r.top_p, Some(json!("0.9")));
        assert_eq!(r.max_tokens, Some(16));
        assert_eq!(r.n, Some(2));
        assert_eq!(r.seed, Some(7));
        // An integral float coerces to the int field.
        assert_eq!(chat(r#"{"messages":[],"max_tokens":16.0}"#).max_tokens, Some(16));
    }

    #[test]
    fn still_accepts_native_types_and_null_and_absent() {
        let r = chat(
            r#"{"messages":[],"stream":true,"temperature":0.5,"max_tokens":16,
                "seed":null}"#,
        );
        assert_eq!(r.stream, Some(true));
        assert_eq!(r.temperature, Some(json!(0.5)));
        assert_eq!(r.max_tokens, Some(16));
        assert_eq!(r.seed, None); // explicit null
        let r = chat(r#"{"messages":[]}"#);
        assert_eq!(r.stream, None); // absent
        assert_eq!(r.temperature, None);
    }

    #[test]
    fn rejects_genuinely_invalid_scalars() {
        // A non-integral value for an int field is still an error.
        assert!(serde_json::from_str::<ChatCompletionRequest>(
            r#"{"messages":[],"max_tokens":"1.5"}"#
        )
        .is_err());
        assert!(serde_json::from_str::<ChatCompletionRequest>(
            r#"{"messages":[],"stream":"maybe"}"#
        )
        .is_err());
    }

    #[test]
    fn int_fields_refuse_what_python_refuses() {
        let max_tokens = |v: &str| {
            let body = format!(r#"{{"messages":[],"max_tokens":{v}}}"#);
            serde_json::from_str::<ChatCompletionRequest>(&body).map(|r| r.max_tokens)
        };
        assert_eq!(max_tokens(r#"" -3 ""#).unwrap(), Some(-3));
        assert_eq!(max_tokens(r#""5.000""#).unwrap(), Some(5));
        assert_eq!(max_tokens(r#""9007199254740993.0""#).unwrap(), Some(9007199254740993));
        // an exponent string, as pydantic refuses it
        assert!(max_tokens(r#""5e0""#).is_err());
        assert!(max_tokens(r#""1e20""#).is_err());
        // out of i64: refused, not saturated to i64::MAX
        assert!(max_tokens("1e20").is_err());
        assert!(max_tokens("-1e20").is_err());
        assert!(max_tokens("9223372036854775808.0").is_err());
        assert_eq!(max_tokens("-9223372036854775808.0").unwrap(), Some(i64::MIN));
        let seed = r#"{"messages":[],"seed":1e20}"#;
        assert!(serde_json::from_str::<ChatCompletionRequest>(seed).is_err());
    }

    #[test]
    fn coercion_covers_image_and_video_fields() {
        let img: ImageGenerationRequest =
            serde_json::from_str(r#"{"prompt":"x","n":"2","seed":"3"}"#).unwrap();
        assert_eq!(img.n, Some(2));
        assert_eq!(img.seed, Some(3));
        let vid: VideoGenerationRequest =
            serde_json::from_str(r#"{"prompt":"x","fps":"12","num_frames":"24"}"#).unwrap();
        assert_eq!(vid.fps, Some(json!("12")));
        assert_eq!(vid.num_frames, Some(24));
    }
}
