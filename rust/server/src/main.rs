//! mstar-server: the Rust (axum) HTTP frontend — the "scale path".
//!
//! A faithful port of mstar's `api_server` HTTP surface to axum, moving the
//! GIL-heavy edge work off Python: HTTP, JSON, base64, media file IO, and the
//! SSE / WAV / NDJSON serialization of results. The endpoints are
//! model-agnostic — each looks up the loaded model's adapter, checks the
//! surface is supported, and translates to/from mstar's request path. Media
//! preprocessing and the model itself stay in the Python data plane (the
//! conductor + data-worker), reached over the msgpack bridge protocol.
//!
//! Endpoints (mirroring `api_server/openai/router.py` + `entrypoint.py`):
//!   GET  /v1/models
//!   POST /v1/chat/completions      (SSE stream or JSON; text/audio/image out)
//!   POST /v1/audio/speech          (streaming WAV or full container)
//!   POST /v1/images/generations
//!   POST /v1/images/edits          (multipart)
//!   POST /generate                 (native multipart -> NDJSON)
//!   GET  /health
//!
//! Two result sources, selected at startup by whether a conductor socket dir is
//! passed: **real** (bridge to the Python conductor) or **mock** (self-contained
//! generator that proves the HTTP/adapter/media/streaming layer standalone).

mod adapters;
mod bridge;
mod media;
mod protocol;
mod serving;
mod tokenizer;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    body::{Body, Bytes},
    extract::{Multipart, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use serde_json::{json, Map, Value};
use uuid::Uuid;

use adapters::{Adapter, SubmitArgs};
use bridge::{Bridge, ResultChunk};
use protocol::{ChatCompletionRequest, ImageGenerationRequest, ModelCard, ModelList, SpeechRequest};
use serving::{collect, now, result_stream, rid, AppState, Out};

#[tokio::main]
async fn main() {
    let model_name = std::env::args()
        .nth(1)
        .expect("usage: mstar-server <model_name> [port] [conductor_socket_dir] [upload_dir]");
    let port: u16 = std::env::args()
        .nth(2)
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);
    let socket_dir = std::env::args().nth(3);
    let upload_dir = PathBuf::from(
        std::env::args()
            .nth(4)
            .unwrap_or_else(|| "/tmp/mstar_uploads".to_string()),
    );
    let sample_rate: u32 = std::env::var("MSTAR_SAMPLE_RATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24000);
    let allow_remote = std::env::var("MSTAR_ALLOW_REMOTE").as_deref() == Ok("1");
    // Total per-request budget (mstar's APIServer timeout_seconds default).
    let request_timeout = std::time::Duration::from_secs(
        std::env::var("MSTAR_REQUEST_TIMEOUT_S")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600),
    );

    let bridge = socket_dir.map(|dir| {
        eprintln!("bridging to conductor at {dir}");
        Arc::new(Bridge::new(&dir).expect("bind bridge"))
    });
    if bridge.is_none() {
        eprintln!("no conductor socket dir given — mock generator (HTTP/adapter/media layer only)");
    }

    // Optional Rust-side tokenizer (the frontend-tokenizes fast path): loaded
    // from MSTAR_TOKENIZER for adapters that flag frontend_tokenizes() and for
    // /generate's `tokenize` field. Absent => every model ships text and the
    // model side tokenizes (the multimodal-safe default).
    let tok = std::env::var("MSTAR_TOKENIZER").ok().map(|path| {
        let t = tokenizer::Tok::from_file(&path).expect("load MSTAR_TOKENIZER");
        eprintln!("tokenizer loaded from {path} ({} vocab)", t.vocab_size());
        Arc::new(t)
    });

    let state = AppState {
        model_name: model_name.clone(),
        upload_dir,
        allow_remote,
        sample_rate,
        request_timeout,
        tok,
        bridge,
    };

    // Allow-all CORS, matching mstar's CORSMiddleware configuration.
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/images/generations", post(images_generations))
        .route("/v1/images/edits", post(images_edits))
        .route("/generate", post(generate))
        .layer(cors)
        .with_state(state);

    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    eprintln!("mstar-server (Rust frontend) for model {model_name:?} on http://{addr}");
    axum::serve(listener, app).await.unwrap();
}

// ---- shared helpers -------------------------------------------------------

enum Surface {
    Chat,
    Speech,
    Images,
}

fn error(status: u16, message: &str, type_: &str) -> Response {
    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        code,
        Json(json!({"error": {"message": message, "type": type_, "code": status}})),
    )
        .into_response()
}

/// Resolve the loaded model's adapter and check it serves `surface`; otherwise
/// an OpenAI-shaped error response (404), matching router.py `_resolve`.
fn resolve(st: &AppState, surface: Surface) -> Result<Adapter, Response> {
    let adapter = Adapter::from_model_name(&st.model_name).ok_or_else(|| {
        error(
            404,
            &format!(
                "Model {:?} has no OpenAI-compatible adapter; use POST /generate",
                st.model_name
            ),
            "model_not_found",
        )
    })?;
    let ok = match surface {
        Surface::Chat => adapter.supports_chat(),
        Surface::Speech => adapter.supports_speech(),
        Surface::Images => adapter.supports_images(),
    };
    if !ok {
        return Err(error(
            404,
            &format!("Model {:?} does not support this endpoint", st.model_name),
            "invalid_request_error",
        ));
    }
    Ok(adapter)
}

/// Delete a request's persisted media 60 s after translation, matching
/// mstar's deferred upload cleanup. Only files under `upload_dir` are ours to
/// delete — local paths passed through by reference are left alone.
fn schedule_upload_cleanup(upload_dir: &std::path::Path, args: &SubmitArgs) {
    let files: Vec<PathBuf> = args
        .file_paths
        .values()
        .flatten()
        .map(PathBuf::from)
        .filter(|p| p.starts_with(upload_dir))
        .collect();
    if files.is_empty() {
        return;
    }
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        for f in files {
            let _ = std::fs::remove_file(f);
        }
    });
}

/// An SSE data event carrying an OpenAI error envelope (terminal mid-stream).
fn sse_error_event(msg: &str) -> Event {
    Event::default().data(
        json!({"error": {"message": msg, "type": "server_error", "code": 500}}).to_string(),
    )
}

// ---- GET /health, /v1/models ---------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({"status": "healthy"}))
}

async fn list_models(State(st): State<AppState>) -> Json<ModelList> {
    Json(ModelList::new(vec![ModelCard::new(st.model_name, now())]))
}

// ---- POST /v1/chat/completions -------------------------------------------

async fn chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let adapter = match resolve(&st, Surface::Chat) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let mut args = match adapter.chat_to_request(&req, &st.upload_dir, st.allow_remote) {
        Ok(a) => a,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    // Frontend-tokenizes fast path: flagged adapters with a loaded tokenizer
    // submit token ids and detokenize the reply in Rust, off the GIL.
    if adapter.frontend_tokenizes() {
        if let (Some(tok), Some(text)) = (&st.tok, args.text.take()) {
            args.tokens = Some(tok.encode(&text));
        }
    }
    schedule_upload_cleanup(&st.upload_dir, &args);
    let request_id = rid("chatcmpl");

    if req.stream {
        return chat_sse(st, args, request_id).into_response();
    }
    match collect(result_stream(&st, &args, &request_id, false)).await {
        Ok(chunks) => Json(build_chat_response(
            &st.model_name, &request_id, chunks, st.sample_rate,
        ))
        .into_response(),
        Err(e) => error(500, &e, "server_error"),
    }
}

/// SSE `chat.completion.chunk`s: role delta, per-chunk deltas, finish, `[DONE]`.
fn chat_sse(
    st: AppState,
    args: SubmitArgs,
    request_id: String,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let created = now();
    let model = st.model_name.clone();
    let base = result_stream(&st, &args, &request_id, true);

    let head = {
        let s = chunk_json(&request_id, created, &model, json!({"role": "assistant"}), None);
        futures::stream::once(async move { Ok(Event::default().data(s)) })
    };
    let body = {
        let model = model.clone();
        let id = request_id.clone();
        base.map(move |item| match item {
            Out::Chunk(c) => Ok(Event::default().data(chunk_json(
                &id,
                created,
                &model,
                chat_delta(&c),
                None,
            ))),
            // Terminal mid-stream failure: an OpenAI error event (the stream
            // ends right after — mstar's generator exception drops the
            // connection; this is strictly more informative).
            Out::Error(e) => Ok(sse_error_event(&e)),
        })
    };
    let tail = {
        let finish = chunk_json(&request_id, created, &model, json!({}), Some("stop"));
        futures::stream::iter(vec![
            Ok(Event::default().data(finish)),
            Ok(Event::default().data("[DONE]")),
        ])
    };
    Sse::new(head.chain(body).chain(tail)).keep_alive(KeepAlive::default())
}

/// One streaming chunk -> its OpenAI delta object.
fn chat_delta(c: &ResultChunk) -> Value {
    match c.modality.as_str() {
        "text" => json!({"content": String::from_utf8_lossy(&c.data)}),
        // Streaming audio deltas are base64 16-bit PCM at the model rate.
        "audio" => json!({"audio": {"id": rid("audio"), "data": media::b64(&c.data)}}),
        "image" => json!({"content": media::png_to_data_url(&c.data)}),
        _ => json!({}),
    }
}

fn chunk_json(id: &str, created: i64, model: &str, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
    .to_string()
}

/// Non-streaming chat: text into `content`, audio into `message.audio` (b64
/// WAV), images into `image_url` data-URL content parts (matching serving_chat).
fn build_chat_response(
    model: &str,
    request_id: &str,
    chunks: Vec<ResultChunk>,
    sample_rate: u32,
) -> Value {
    let mut text_parts: Vec<String> = Vec::new();
    let mut audio_pcm: Vec<u8> = Vec::new();
    let mut images: Vec<Vec<u8>> = Vec::new();
    for c in &chunks {
        match c.modality.as_str() {
            "text" => text_parts.push(String::from_utf8_lossy(&c.data).into_owned()),
            "audio" => audio_pcm.extend_from_slice(&c.data),
            "image" => images.push(c.data.clone()),
            _ => {}
        }
    }
    let text = text_parts.concat();
    let mut message = json!({"role": "assistant", "content": text.clone()});

    if !audio_pcm.is_empty() {
        let wav = media::pcm16_to_wav_bytes(&audio_pcm, sample_rate, 1);
        message["audio"] = json!({
            "id": rid("audio"),
            "data": media::b64(&wav),
            "expires_at": now() + 86400,
            "transcript": text.clone(),
        });
    }
    if !images.is_empty() {
        let mut parts: Vec<Value> = Vec::new();
        if !text.is_empty() {
            parts.push(json!({"type": "text", "text": text}));
        }
        for img in &images {
            parts.push(json!({"type": "image_url", "image_url": {"url": media::png_to_data_url(img)}}));
        }
        message["content"] = Value::Array(parts);
    }

    json!({
        "id": request_id,
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })
}

// ---- POST /v1/audio/speech -----------------------------------------------

async fn audio_speech(State(st): State<AppState>, Json(req): Json<SpeechRequest>) -> Response {
    let adapter = match resolve(&st, Surface::Speech) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let args = match adapter.speech_to_request(&req) {
        Ok(a) => a,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    let request_id = rid("speech");
    let fmt = req.response_format.to_ascii_lowercase();

    if req.stream {
        // Open-ended WAV: streaming header, then PCM16 frames as they arrive.
        // A backend error ends the byte stream (the connection closes
        // mid-WAV, matching mstar's generator exception).
        let header = media::wav_stream_header(st.sample_rate, 1);
        let base = result_stream(&st, &args, &request_id, true);
        let audio_only = base.filter_map(|item| async move {
            match item {
                Out::Chunk(c) if c.modality == "audio" && !c.data.is_empty() => {
                    Some(Ok::<Bytes, Infallible>(Bytes::from(c.data)))
                }
                _ => None,
            }
        });
        let body = futures::stream::once(async move { Ok::<Bytes, Infallible>(Bytes::from(header)) })
            .chain(audio_only);
        return Response::builder()
            .header("content-type", "audio/wav")
            .header("cache-control", "no-cache")
            .body(Body::from_stream(body))
            .unwrap();
    }

    let chunks = match collect(result_stream(&st, &args, &request_id, false)).await {
        Ok(chunks) => chunks,
        Err(e) => return error(500, &e, "server_error"),
    };
    let mut pcm: Vec<u8> = Vec::new();
    for c in &chunks {
        if c.modality == "audio" {
            pcm.extend_from_slice(&c.data);
        }
    }
    let (bytes, mime) = media::pcm16_to_container(&pcm, st.sample_rate, &fmt);
    Response::builder()
        .header("content-type", mime)
        .body(Body::from(bytes))
        .unwrap()
}

// ---- POST /v1/images/generations, /v1/images/edits -----------------------

async fn images_generations(
    State(st): State<AppState>,
    Json(req): Json<ImageGenerationRequest>,
) -> Response {
    let adapter = match resolve(&st, Surface::Images) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let args = match adapter.image_to_request(&req) {
        Ok(a) => a,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    let request_id = rid("img");
    match collect(result_stream(&st, &args, &request_id, false)).await {
        Ok(chunks) => Json(images_response(chunks)).into_response(),
        Err(e) => error(500, &e, "server_error"),
    }
}

async fn images_edits(State(st): State<AppState>, mut mp: Multipart) -> Response {
    let adapter = match resolve(&st, Surface::Images) {
        Ok(a) => a,
        Err(r) => return r,
    };
    // Multipart (image file + prompt + passthrough fields), parsed manually so
    // arbitrary model knobs (e.g. cfg_*_scale) flow through as model_kwargs.
    let known = ["image", "prompt", "model", "n", "size", "response_format"];
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut image_filename: Option<String> = None;
    let mut prompt = String::new();
    let mut extra: Map<String, Value> = Map::new();

    while let Ok(Some(field)) = mp.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        let has_file = field.file_name().is_some();
        if name == "image" {
            image_filename = field.file_name().map(str::to_string);
            image_bytes = field.bytes().await.ok().map(|b| b.to_vec());
        } else if name == "prompt" {
            prompt = field.text().await.unwrap_or_default();
        } else if known.contains(&name.as_str()) || has_file {
            let _ = field.bytes().await; // consume + discard
        } else {
            let v = field.text().await.unwrap_or_default();
            let parsed = serde_json::from_str::<Value>(&v).unwrap_or(Value::String(v));
            extra.insert(name, parsed);
        }
    }

    let image_bytes = match image_bytes {
        Some(b) if !b.is_empty() => b,
        _ => return error(400, "images/edits requires an 'image' file upload", "invalid_request_error"),
    };
    // Persist the uploaded image so the data-worker's loader can read it by path.
    if let Err(e) = std::fs::create_dir_all(&st.upload_dir) {
        return error(500, &e.to_string(), "server_error");
    }
    let ext = image_filename
        .as_deref()
        .and_then(|f| PathBuf::from(f).extension().map(|e| format!(".{}", e.to_string_lossy())))
        .unwrap_or_else(|| ".png".to_string());
    let image_path = st.upload_dir.join(format!("{}{}", Uuid::new_v4().simple(), ext));
    if let Err(e) = std::fs::write(&image_path, &image_bytes) {
        return error(500, &e.to_string(), "server_error");
    }

    let args = match adapter.image_edit_to_request(&prompt, &image_path.to_string_lossy(), extra) {
        Ok(a) => a,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    schedule_upload_cleanup(&st.upload_dir, &args);
    let request_id = rid("img");
    match collect(result_stream(&st, &args, &request_id, false)).await {
        Ok(chunks) => Json(images_response(chunks)).into_response(),
        Err(e) => error(500, &e, "server_error"),
    }
}

fn images_response(chunks: Vec<ResultChunk>) -> Value {
    let data: Vec<Value> = chunks
        .iter()
        .filter(|c| c.modality == "image")
        .map(|c| json!({"b64_json": media::b64(&c.data), "url": Value::Null}))
        .collect();
    json!({"created": now(), "data": data})
}

// ---- POST /generate (native multipart -> NDJSON) -------------------------

async fn generate(State(st): State<AppState>, mut mp: Multipart) -> Response {
    let mut text: Option<String> = None;
    let mut in_mods_raw: Option<String> = None;
    let mut out_mods_raw = "text".to_string();
    let mut streaming = true;
    let mut tokenize = false;
    let mut mk_raw: Option<String> = None;
    let mut request_id: Option<String> = None;
    let mut file_paths: BTreeMap<String, Vec<String>> = BTreeMap::new();

    while let Ok(Some(field)) = mp.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "files" {
            let filename = field.file_name().unwrap_or("").to_string();
            let modality = media::modality_from_filename(&filename);
            if modality == "unknown" {
                return error(400, &format!("Cannot determine modality for file: {filename}"), "invalid_request_error");
            }
            let bytes = match field.bytes().await {
                Ok(b) => b,
                Err(e) => return error(400, &e.to_string(), "invalid_request_error"),
            };
            if let Err(e) = std::fs::create_dir_all(&st.upload_dir) {
                return error(500, &e.to_string(), "server_error");
            }
            let save_path = st.upload_dir.join(format!("{}_{}", Uuid::new_v4().simple(), filename));
            if let Err(e) = std::fs::write(&save_path, &bytes) {
                return error(500, &e.to_string(), "server_error");
            }
            file_paths
                .entry(modality.to_string())
                .or_default()
                .push(save_path.to_string_lossy().into_owned());
        } else {
            let val = field.text().await.unwrap_or_default();
            match name.as_str() {
                "text" => text = Some(val),
                "input_modalities" => in_mods_raw = Some(val),
                "output_modalities" => out_mods_raw = val,
                "streaming" => streaming = matches!(val.as_str(), "true" | "True" | "1"),
                "tokenize" => tokenize = matches!(val.as_str(), "true" | "True" | "1"),
                "model_kwargs" => mk_raw = Some(val),
                "request_id" => request_id = Some(val),
                _ => {}
            }
        }
    }

    let output_modalities: Vec<String> = out_mods_raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let input_modalities: Vec<String> = match in_mods_raw {
        Some(raw) => raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        None => {
            let mut m: Vec<String> = file_paths.keys().cloned().collect();
            if text.as_ref().map(|t| !t.is_empty()).unwrap_or(false) {
                m.push("text".to_string());
            }
            m
        }
    };
    let model_kwargs: Map<String, Value> = match mk_raw {
        Some(raw) if !raw.is_empty() => match serde_json::from_str::<Value>(&raw) {
            Ok(Value::Object(m)) => m,
            Ok(_) => return error(400, "model_kwargs must be a JSON object", "invalid_request_error"),
            Err(e) => return error(400, &format!("model_kwargs JSON: {e}"), "invalid_request_error"),
        },
        _ => Map::new(),
    };
    let request_id = request_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    // `tokenize=true`: encode the text in Rust and submit token ids; the
    // reply's text streams back as token ids and detokenizes here too.
    let mut tokens: Option<Vec<u32>> = None;
    if tokenize {
        let Some(tok) = &st.tok else {
            return error(400, "tokenize=true requires the server to be started \
                               with MSTAR_TOKENIZER", "invalid_request_error");
        };
        tokens = Some(tok.encode(text.as_deref().unwrap_or("")));
        text = None; // the model side must use the ids, not re-tokenize
    }

    let args = SubmitArgs {
        text,
        tokens,
        file_paths,
        input_modalities,
        output_modalities,
        model_kwargs,
    };

    schedule_upload_cleanup(&st.upload_dir, &args);
    if streaming {
        let base = result_stream(&st, &args, &request_id, true);
        let body = base.map(|item| {
            Ok::<Bytes, Infallible>(Bytes::from(match item {
                Out::Chunk(c) => ndjson_line(&c),
                Out::Error(e) => {
                    let mut s = json!({"error": e}).to_string();
                    s.push('\n');
                    s.into_bytes()
                }
            }))
        });
        return Response::builder()
            .header("content-type", "application/x-ndjson")
            .header("cache-control", "no-cache")
            .body(Body::from_stream(body))
            .unwrap();
    }

    let chunks = match collect(result_stream(&st, &args, &request_id, false)).await {
        Ok(chunks) => chunks,
        Err(e) => return error(500, &e, "server_error"),
    };
    let mut outputs: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for c in &chunks {
        outputs
            .entry(c.modality.clone())
            .or_default()
            .push(json!({"data": media::b64(&c.data), "metadata": c.metadata}));
    }
    Json(json!({"request_id": request_id, "outputs": outputs})).into_response()
}

fn ndjson_line(c: &ResultChunk) -> Vec<u8> {
    let mut s = json!({
        "modality": c.modality,
        "data": media::b64(&c.data),
        "metadata": c.metadata,
    })
    .to_string();
    s.push('\n');
    s.into_bytes()
}
