//! `POST /v1/audio/transcriptions`: a port of `router.audio_transcriptions` and
//! `openai/serving_transcriptions.py`.
//!
//! The route reads the multipart form (the audio upload and the OpenAI fields),
//! validates the fields as `TranscriptionRequest` does, saves the upload, and
//! hands it to the API server's `WindowPlanner` over the bridge
//! ([`Planner`]): the planner decodes it with the model's own loader, which
//! reads formats no Rust decoder does (WebM, Opus, m4a), and cuts the windows a
//! model with a clip bound hears. Everything else runs here, as the Python
//! module runs it: the parallel and sequential long-form drivers, the
//! temperature fallback, the merge, and the response formats (see that
//! module's docstring).

mod drive;
mod format;
mod request;
mod zlib;

use std::path::PathBuf;
use std::sync::OnceLock;

use axum::extract::{FromRequest, Multipart, State};
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

pub use request::{FieldValue, TranscriptionRequest};

use crate::serving::AppState;

/// The adapter attributes that steer the route (`max_audio_seconds`,
/// `window_seconds`, ...), one value per Python class attribute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranscriptionPolicy {
    /// Longest clip the model hears in one request; `None` is unbounded.
    pub max_audio_seconds: Option<f64>,
    /// Window length for a long upload when shorter than the clip bound.
    pub window_seconds: Option<f64>,
    /// The fewest windows of about equal length, instead of full ones and a remainder.
    pub balanced_windows: bool,
    /// Uploads up to this long run whole (`None`: the window length).
    pub whole_seconds: Option<f64>,
    /// How far back a cut may move to the quietest frame, and that frame's
    /// length (`None`: the route's defaults).
    pub window_search_seconds: Option<f64>,
    pub window_frame_seconds: Option<f64>,
    /// `sequential` or `parallel`.
    pub long_form: &'static str,
    /// Sequential windows decoded with timestamps, each starting where the last
    /// closed segment ended.
    pub seeks_by_timestamps: bool,
    /// A window whose text compresses better than this is decoded again.
    pub compression_ratio_threshold: Option<f64>,
}

impl TranscriptionPolicy {
    /// `OpenAIAdapter`'s class attributes.
    pub const DEFAULT: TranscriptionPolicy = TranscriptionPolicy {
        max_audio_seconds: None,
        window_seconds: None,
        balanced_windows: false,
        whole_seconds: None,
        window_search_seconds: None,
        window_frame_seconds: None,
        long_form: "sequential",
        seeks_by_timestamps: false,
        compression_ratio_threshold: Some(2.4),
    };
}

/// A transcript segment, in seconds; `temperature` and `compression_ratio` are
/// those its window was accepted at, once merged.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub id: Option<usize>,
    pub temperature: Option<f64>,
    pub compression_ratio: Option<f64>,
}

impl Segment {
    pub fn new(start: f64, end: f64, text: String) -> Segment {
        Segment { start, end, text, id: None, temperature: None, compression_ratio: None }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// `adapters.Transcript`: the clean text, the language when the model reported
/// one, timestamps when it emitted them, and whether it stopped inside a
/// segment (the text after the last closed one is provisional).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<Segment>,
    pub words: Vec<Word>,
    pub unfinished: bool,
}

impl Transcript {
    pub fn text(text: &str) -> Transcript {
        Transcript { text: text.to_string(), ..Transcript::default() }
    }
}

/// What the route needs from the served model that only the Python side
/// knows, handed over by the launcher as `MSTAR_TRANSCRIPTION`
/// (`rust_frontend.transcription_facts`).
#[derive(Debug, Default, Deserialize)]
pub struct Facts {
    /// The kwargs the model reads (`request_kwargs()`); `None` when it
    /// declares none, which reads as "may read anything".
    #[serde(default)]
    pub request_kwargs: Option<Vec<String>>,
    /// The deployment's `max_concurrent_requests`, which the uploads cut in
    /// parallel share.
    #[serde(default)]
    pub max_concurrent_requests: Option<usize>,
    /// The languages the adapter checks a request's against: Canary's, or
    /// Parakeet's detected ones.
    #[serde(default)]
    pub languages: Vec<String>,
    /// Whether the transducer adapter checks `language` (Parakeet-TDT).
    #[serde(default)]
    pub detects_language: bool,
    /// Qwen3-ASR's language names by ISO code, in its order.
    #[serde(default)]
    pub language_codes: Vec<(String, String)>,
    /// pydantic's error docs prefix for its version, which its error text cites.
    #[serde(default = "default_errors_url")]
    pub errors_url: String,
}

fn default_errors_url() -> String {
    "https://errors.pydantic.dev/2.14/v/".to_string()
}

pub fn facts() -> &'static Facts {
    static FACTS: OnceLock<Facts> = OnceLock::new();
    FACTS.get_or_init(|| match std::env::var("MSTAR_TRANSCRIPTION") {
        Ok(s) if !s.is_empty() => serde_json::from_str(&s).expect("MSTAR_TRANSCRIPTION is the launcher's JSON"),
        _ => Facts { errors_url: default_errors_url(), ..Facts::default() },
    })
}

/// Whether the model reads `key`; one that declares nothing may.
fn reads(key: &str) -> bool {
    facts().request_kwargs.as_ref().is_none_or(|keys| keys.iter().any(|k| k == key))
}

pub const RESPONSE_FORMATS: [&str; 5] = ["json", "text", "verbose_json", "srt", "vtt"];
const TIMED_FORMATS: [&str; 3] = ["verbose_json", "srt", "vtt"];
/// The parallel-mode cut search and frame defaults.
const CUT_SEARCH_SECONDS: f64 = 2.0;
const CUT_FRAME_SECONDS: f64 = 0.02;
/// Shortest window a request may ask for.
const MIN_WINDOW_SECONDS: f64 = 5.0;
/// Longest upload the route takes, in bytes.
pub const MAX_UPLOAD_BYTES: usize = 512 << 20;
/// The route's body limit: the upload and its form fields.
pub fn body_limit_bytes(default: usize) -> usize {
    default.max(MAX_UPLOAD_BYTES + (1 << 20))
}

fn error(status: u16, message: &str) -> Response {
    crate::error(status, message, crate::error_type(status))
}

/// `f"{x:g}"` for the bounds the route names.
fn fmt_g(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e16 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// The form: the last `file` part, and the other fields as the route collects them.
struct Form {
    file: Option<(Vec<u8>, Option<String>)>,
    fields: Vec<(String, FieldValue)>,
}

/// The fields `TranscriptionRequest` declares as text (the router's
/// `_TRANSCRIPTION_TEXT_FIELDS`): kept as sent, so a prompt of "2024" is not a number.
const TEXT_FIELDS: [&str; 4] = ["model", "language", "prompt", "response_format"];

/// One field into the collected fields: `name[]` gathers raw strings, a text
/// field stays text, any other name is `json.loads`-ed, a repeat keeping the
/// first one's place.
fn collect_field(fields: &mut Vec<(String, FieldValue)>, name: &str, text: String) {
    if let Some(list) = name.strip_suffix("[]") {
        let item = serde_json::Value::String(text);
        match fields.iter_mut().find(|(k, _)| k == list) {
            Some((_, FieldValue::Json(serde_json::Value::Array(items)))) => items.push(item),
            Some((_, v)) => *v = FieldValue::Json(serde_json::Value::Array(vec![item])),
            None => fields.push((list.to_string(), FieldValue::Json(serde_json::Value::Array(vec![item])))),
        }
        return;
    }
    let value = if TEXT_FIELDS.contains(&name) {
        FieldValue::Json(serde_json::Value::String(text))
    } else {
        FieldValue::from_form(text)
    };
    match fields.iter_mut().find(|(k, _)| k == name) {
        Some((_, v)) => *v = value,
        None => fields.push((name.to_string(), value)),
    }
}

async fn read_form(mut mp: Multipart, limit_mb: usize) -> Result<Form, Response> {
    let failure = |e: &axum::extract::multipart::MultipartError, fallback: String| {
        if e.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
            error(413, &format!("request body is larger than the {limit_mb} MiB limit (MSTAR_MAX_BODY_MB)"))
        } else {
            error(400, &fallback)
        }
    };
    let mut form = Form { file: None, fields: Vec::new() };
    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => return Ok(form),
            Err(e) => return Err(failure(&e, format!("malformed multipart body: {e}"))),
        };
        let name = field.name().unwrap_or("").to_string();
        let filename = field.file_name().map(str::to_string);
        if filename.is_some() {
            let bytes = field.bytes().await.map_err(|e| failure(&e, e.to_string()))?;
            if name == "file" {
                form.file = Some((bytes.to_vec(), filename));
            }
            continue;
        }
        let text = field.text().await.map_err(|e| failure(&e, e.to_string()))?;
        if name == "file" {
            // the route reads the last `file`, here one with no upload in it
            form.file = None;
        }
        collect_field(&mut form.fields, &name, text);
    }
}

/// `router.audio_transcriptions`.
pub async fn audio_transcriptions(State(st): State<AppState>, req: axum::extract::Request) -> Response {
    let adapter = match crate::resolve(&st, crate::Surface::Transcriptions) {
        Ok(a) => a,
        Err(r) => return r,
    };
    // a body that is not a form has no upload in it
    let multipart = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.to_ascii_lowercase().starts_with("multipart/form-data"));
    if !multipart {
        return error(400, "audio/transcriptions requires a 'file' upload");
    }
    let mp = match Multipart::from_request(req, &st).await {
        Ok(mp) => mp,
        Err(e) => return error(400, &e.body_text()),
    };
    let limit_mb = body_limit_bytes(crate::body_limit_mb() << 20) >> 20;
    let form = match read_form(mp, limit_mb).await {
        Ok(form) => form,
        Err(r) => return r,
    };
    let Some((audio, filename)) = form.file else {
        return error(400, "audio/transcriptions requires a 'file' upload");
    };
    if audio.len() > MAX_UPLOAD_BYTES {
        return error(413, &format!("audio/transcriptions accepts files up to {} MiB", MAX_UPLOAD_BYTES >> 20));
    }
    if audio.is_empty() {
        return error(400, "audio/transcriptions received an empty 'file'");
    }
    let req = match request::validate(form.fields, &facts().errors_url) {
        Ok(req) => req,
        Err(e) => return error(400, &e),
    };
    create_transcription(st, adapter, req, audio, filename).await
}

/// `create_transcription`'s checks, in its order, before the upload is planned.
fn check(adapter: &dyn crate::adapters::OpenAIAdapter, req: &TranscriptionRequest) -> Result<(String, &'static str, Option<f64>), String> {
    let fmt = req.response_format.to_lowercase();
    if !RESPONSE_FORMATS.contains(&fmt.as_str()) {
        return Err(format!(
            "response_format must be one of ['json', 'text', 'verbose_json', 'srt', 'vtt']; got {}",
            crate::media::py_repr(&fmt)
        ));
    }
    let policy = adapter.transcription_policy();
    let mode = match req.extra("long_form") {
        Some(FieldValue::Json(v)) if crate::adapters::truthy(v) => match v.as_str() {
            Some("sequential") => "sequential",
            Some("parallel") => "parallel",
            _ => {
                return Err(format!(
                    "long_form must be one of ['sequential', 'parallel']; got {}",
                    crate::adapters::py_value_repr(v)
                ))
            }
        },
        Some(FieldValue::Float(f)) if *f != 0.0 => {
            return Err(format!("long_form must be one of ['sequential', 'parallel']; got {}", crate::adapters::py_float_repr(*f)))
        }
        _ => policy.long_form,
    };
    // the adapter checks the fields, before a stream commits to a 200
    adapter.transcription_to_request(req, "")?;
    let window = match req.extra("window_seconds") {
        None | Some(FieldValue::Json(serde_json::Value::Null)) => None,
        Some(v) => {
            let value = match v {
                FieldValue::Float(f) => *f,
                FieldValue::Json(serde_json::Value::Bool(b)) => f64::from(u8::from(*b)),
                FieldValue::Json(serde_json::Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
                FieldValue::Json(serde_json::Value::String(s)) => crate::generation_kwargs::py_float(s).unwrap_or(f64::NAN),
                FieldValue::Json(_) => f64::NAN,
            };
            match policy.max_audio_seconds {
                Some(cap) if (MIN_WINDOW_SECONDS..=cap).contains(&value) => Some(value),
                cap => {
                    let bound = match cap {
                        Some(cap) => format!("between {} and {}", fmt_g(MIN_WINDOW_SECONDS), fmt_g(cap)),
                        None => "unset (no clip bound)".to_string(),
                    };
                    return Err(format!("window_seconds must be {bound} for this model"));
                }
            }
        }
    };
    Ok((fmt, mode, window))
}

/// `save_upload`: under the upload directory, as `<uuid>_<basename>`, the
/// extension kept for the decoder to sniff the container by.
fn save_upload(audio: &[u8], filename: Option<&str>, upload_dir: &std::path::Path) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(upload_dir)?;
    let base = filename.and_then(|f| f.rsplit('/').next()).filter(|b| !b.is_empty()).unwrap_or("audio.wav");
    let path = upload_dir.join(format!("{}_{base}", uuid::Uuid::new_v4()));
    std::fs::write(&path, audio)?;
    Ok(path)
}

async fn create_transcription(
    st: AppState,
    adapter: &'static dyn crate::adapters::OpenAIAdapter,
    req: TranscriptionRequest,
    audio: Vec<u8>,
    filename: Option<String>,
) -> Response {
    let (fmt, mode, window) = match check(adapter, &req) {
        Ok(checked) => checked,
        Err(e) => return error(400, &e),
    };
    let dir = st.upload_dir.clone();
    let path = match crate::blocking(move || save_upload(&audio, filename.as_deref(), &dir)).await {
        Ok(path) => path,
        Err(e) => return error(500, &e.to_string()),
    };
    let policy = adapter.transcription_policy();
    let shape = drive::PlanShape::of(&policy, window);
    let planner = match drive::Planner::open(&st, path.to_string_lossy().into_owned(), shape).await {
        Ok(planner) => planner,
        Err(e) => return error(e.status, &e.message),
    };
    let ctx = drive::Ctx { st, adapter, req, mode, planner };
    if ctx.req.stream.unwrap_or(false) {
        return drive::stream_response(ctx);
    }
    match drive::transcribe(ctx).await {
        Ok((transcript, duration, req)) => format::response(&fmt, &transcript, &req, duration),
        Err(e) => error(e.status, &e.message),
    }
}

impl IntoResponse for drive::OpError {
    fn into_response(self) -> Response {
        error(self.status, &self.message)
    }
}
