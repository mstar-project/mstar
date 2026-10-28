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
//!   GET  /metrics                  (the API server's Prometheus text, via the bridge)
//!
//! With `MSTAR_API_KEY` set, every route but `/health` and `/metrics` needs
//! `Authorization: Bearer <key>` (`auth.rs`).
//!
//! Two result sources, selected at startup by whether a conductor socket dir is
//! passed: **real** (bridge to the Python conductor) or **mock** (self-contained
//! generator that proves the HTTP/adapter/media/streaming layer standalone).

mod adapters;
mod auth;
mod bridge;
mod media;
mod protocol;
mod refusals;
mod serving;
mod tokenizer;
mod utf8;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    body::{Body, Bytes},
    extract::{multipart::MultipartError, rejection::JsonRejection, DefaultBodyLimit, FromRequest, Multipart, State},
    http::{header::CONTENT_TYPE, StatusCode},
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
use protocol::{
    ChatCompletionRequest, ImageGenerationRequest, ModelCard, ModelList, SpeechRequest,
    VideoGenerationRequest,
};
use serving::{
    collect, now, result_stream, rid, try_result_stream, AppState, Out, DUPLICATE_ID_GRACE,
};
use utf8::Utf8Stream;

#[tokio::main]
async fn main() {
    const USAGE: &str =
        "usage: mstar-server <model_name> [port] [conductor_socket_dir] [upload_dir]";
    let model_name = std::env::args().nth(1).expect(USAGE);
    if model_name == "-h" || model_name == "--help" {
        println!("{USAGE}");
        return;
    }
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
    // Same variable and default as serving_images.max_images_per_request.
    let max_images_per_request =
        parse_max_images(std::env::var("MSTAR_MAX_IMAGES_PER_REQUEST").ok().as_deref());

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

    let python_adapter = std::env::var("MSTAR_OPENAI_ADAPTER").unwrap_or_default();
    if !python_adapter.is_empty() && Adapter::from_python(&python_adapter).is_none() {
        eprintln!(
            "WARNING: {model_name}'s OpenAI adapter {python_adapter} has no port in this \
             frontend: /v1/* answers 404 here, /generate is served"
        );
    }

    let refusals = Arc::new(refusals::Refusals::default());
    let state = AppState {
        model_name: model_name.clone(),
        python_adapter,
        upload_dir,
        allow_remote,
        sample_rate,
        request_timeout,
        max_images_per_request,
        tok,
        bridge,
        refusals: refusals.clone(),
    };

    // Allow-all CORS, matching mstar's CORSMiddleware configuration.
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    // Admission control: bound concurrent generation requests; past the cap
    // clients get an immediate 503 instead of queueing into the timeout.
    // /health, /metrics and /v1/models bypass it (a saturated server is
    // still alive, and still scrapeable).
    let max_concurrent: usize = std::env::var("MSTAR_MAX_CONCURRENT_REQUESTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let permits = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
    // Uploads (audio/video multipart) routinely exceed axum's 2 MB default.
    let body_limit_mb = body_limit_mb();

    let api = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/images/generations", post(images_generations))
        .route("/v1/images/edits", post(images_edits))
        .route("/v1/videos/generations", post(videos_generations))
        .route("/generate", post(generate))
        .route_layer(axum::middleware::from_fn_with_state(permits, admission));

    let api_key = auth::ApiKey::from_env();
    let key_enabled = api_key.enabled();

    // The key check sits inside CORS: CORS answers browser preflights (which
    // carry no credentials) and still stamps its headers on a 401. It is
    // outside admission, so an unauthenticated request never takes a permit.
    // The refusal counter sits between the two, so it sees the 401 and not
    // the preflights.
    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .merge(api)
        .layer(DefaultBodyLimit::max(body_limit_mb << 20))
        .layer(axum::middleware::from_fn_with_state(
            api_key,
            auth::require_api_key,
        ))
        .layer(axum::middleware::from_fn_with_state(refusals, refusals::count_refusals))
        .layer(cors)
        .with_state(state);

    // Bind host: MSTAR_SERVER_HOST (the launcher forwards --host here);
    // 127.0.0.1 by default. 0.0.0.0 for the multi-node / container case.
    let host = std::env::var("MSTAR_SERVER_HOST")
        .unwrap_or_else(|_| "127.0.0.1".to_string());
    let addr = format!("{host}:{port}");
    if !key_enabled {
        eprintln!(
            "WARNING: no API key set (--api-key or {}): any client that can reach {addr} \
             can run requests. Set a key, or bind --host 127.0.0.1, unless a gateway in \
             front of this server authenticates clients.",
            auth::API_KEY_ENV
        );
    }
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    eprintln!("mstar-server (Rust frontend) for model {model_name:?} on http://{addr}");
    axum::serve(listener, app)
        // Disable Nagle: SSE/NDJSON streams write many small chunks, and
        // Nagle + delayed ACK turns each flush into a ~40 ms stall.
        .tcp_nodelay(true)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

/// SIGTERM/SIGINT: stop accepting, drain in-flight requests, exit — with a
/// hard 30 s cap so a wedged stream can't hold the process forever.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.ok();
    };
    #[cfg(unix)]
    let term = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
    eprintln!("mstar-server: shutting down (draining in-flight, 30s cap)");
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        eprintln!("mstar-server: drain deadline reached, exiting");
        std::process::exit(0);
    });
}

/// Reject work beyond the concurrency cap with an immediate 503.
async fn admission(
    State(permits): State<Arc<tokio::sync::Semaphore>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let permit = match permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            let mut resp = (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": {
                    "message": "server saturated: too many concurrent requests",
                    "type": "overloaded_error"
                }})),
            )
                .into_response();
            resp.extensions_mut().insert(refusals::CapRefusal);
            return resp;
        }
    };
    // Hold the permit until the RESPONSE BODY is fully sent, not just until
    // the Response is built. A streaming handler returns as soon as
    // Body::from_stream is constructed, so dropping the permit here would
    // free the slot while the stream is still being served — the cap would
    // not bound concurrent streams at all. Moving it into the body drops it
    // at end-of-body, uniformly for streaming and non-streaming handlers.
    let resp = next.run(req).await;
    let (parts, body) = resp.into_parts();
    let guarded = body.into_data_stream().map(move |chunk| {
        let _hold = &permit; // keep the permit alive for the stream's life
        chunk
    });
    Response::from_parts(parts, Body::from_stream(guarded))
}

// ---- shared helpers -------------------------------------------------------

enum Surface {
    Chat,
    Speech,
    Images,
    Videos,
}

/// The OpenAI error `type` for a status: a 4xx is the client's
/// (`invalid_request_error`), anything else is ours (`server_error`).
fn error_type(status: u16) -> &'static str {
    if (400..500).contains(&status) {
        "invalid_request_error"
    } else {
        "server_error"
    }
}

/// `MSTAR_MAX_BODY_MB`, default 128. The Python frontend parses it the same
/// way (`mstar/api_server/body_limit.py`).
fn body_limit_mb() -> usize {
    std::env::var("MSTAR_MAX_BODY_MB")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(128)
}

/// Status and message for a failed multipart read: 413 naming the limit when
/// the body passed `MSTAR_MAX_BODY_MB`, as the Python frontend answers, else
/// 400 with `fallback`.
fn multipart_failure(e: &MultipartError, fallback: String) -> (u16, String) {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        let mb = body_limit_mb();
        (413, format!("request body is larger than the {mb} MiB limit (MSTAR_MAX_BODY_MB)"))
    } else {
        (400, fallback)
    }
}

fn error(status: u16, message: &str, type_: &str) -> Response {
    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        code,
        Json(json!({"error": {"message": message, "type": type_, "code": status}})),
    )
        .into_response()
}

/// FastAPI-`HTTPException`-shaped error (`{"detail": "..."}`). The native
/// `/generate` endpoint (entrypoint.py) raises `HTTPException`, so its errors
/// carry `detail`, not the OpenAI `{"error": {...}}` envelope the /v1 endpoints
/// use — keep the two interfaces byte-consistent with Python.
fn detail_error(status: u16, message: &str) -> Response {
    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (code, Json(json!({"detail": message}))).into_response()
}

/// Turn a `Json` extractor rejection into FastAPI + pydantic's `422` +
/// `{"detail": [{"loc", "msg", "type"}]}` shape, instead of axum's default
/// `400` + `text/plain`, so a malformed/invalid body looks the same as the
/// Python frontend's validation error. Handlers take
/// `Result<Json<T>, JsonRejection>` and route the error through here.
fn json_422(rej: JsonRejection) -> Response {
    if rej.status() == StatusCode::PAYLOAD_TOO_LARGE {
        let mb = body_limit_mb();
        return error(
            413,
            &format!("request body is larger than the {mb} MiB limit (MSTAR_MAX_BODY_MB)"),
            "invalid_request_error",
        );
    }
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({"detail": [{
            "loc": ["body"],
            "msg": rej.body_text(),
            "type": "value_error",
        }]})),
    )
        .into_response()
}

/// Parse a form-field boolean the way FastAPI's `bool = Form(...)` does (pydantic
/// v2 lax coercion): a fixed truthy/falsy set, case-insensitive, and a hard
/// error on anything else. The naive `== "true"` accepted a subset and silently
/// turned `TRUE` / `yes` / invalid values into `false` — e.g. `streaming=TRUE`
/// would quietly return a non-streaming body.
fn parse_form_bool(s: &str) -> Result<bool, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "t" | "yes" | "y" | "on" | "1" => Ok(true),
        "false" | "f" | "no" | "n" | "off" | "0" => Ok(false),
        other => Err(format!("value is not a valid boolean: {other:?}")),
    }
}

/// A request's id rides in every control message about it, so a client-pinned
/// one is kept short; the same check and message as `check_request_id` in
/// `mstar/api_server/entrypoint.py`.
const MAX_REQUEST_ID_LEN: usize = 128;

fn check_request_id(id: &str) -> Result<(), String> {
    let allowed = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'-');
    if id.len() <= MAX_REQUEST_ID_LEN && id.bytes().all(allowed) {
        Ok(())
    } else {
        Err(format!(
            "request_id must be at most {MAX_REQUEST_ID_LEN} characters from [A-Za-z0-9._:-]"
        ))
    }
}

/// Resolve the loaded model's adapter and check it serves `surface`; otherwise
/// an OpenAI-shaped error response (404), matching router.py `_resolve`.
fn resolve(st: &AppState, surface: Surface) -> Result<Adapter, Response> {
    let message = if st.python_adapter.is_empty() {
        format!("Model '{}' has no OpenAI-compatible adapter; use POST /generate", st.model_name)
    } else {
        format!(
            "Model '{}' has an OpenAI adapter ({}) the Rust frontend does not serve; \
             use POST /generate, or serve without --rust-frontend",
            st.model_name, st.python_adapter
        )
    };
    let adapter = Adapter::from_python(&st.python_adapter)
        .ok_or_else(|| error(404, &message, "model_not_found"))?;
    let ok = match surface {
        Surface::Chat => adapter.supports_chat(),
        Surface::Speech => adapter.supports_speech(),
        Surface::Images => adapter.supports_images(),
        Surface::Videos => adapter.supports_videos(),
    };
    if !ok {
        return Err(error(
            404,
            &format!("Model '{}' does not support this endpoint", st.model_name),
            "invalid_request_error",
        ));
    }
    Ok(adapter)
}

/// Delete a request's persisted media 60 s after translation, matching
/// mstar's deferred upload cleanup. Only files under `upload_dir` are deleted.
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

/// An SSE data payload carrying an OpenAI error envelope (terminal mid-stream).
fn sse_error_json(status: u16, msg: &str) -> String {
    json!({"error": {"message": msg, "type": error_type(status), "code": status}}).to_string()
}

// ---- GET /health, /metrics, /v1/models ------------------------------------

/// How long `/health` and `/metrics` wait for the bridge loop to answer.
const BRIDGE_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Deep health: liveness of the whole serving path, not just this process.
/// With a bridge configured, a ping must round-trip to the Python backend —
/// so a dead bridge loop or conductor turns the LB check red instead of
/// letting requests queue into the timeout. Mock mode stays shallow.
async fn health(State(st): State<AppState>) -> Response {
    if let Some(bridge) = &st.bridge {
        if !bridge.ping(BRIDGE_PROBE_TIMEOUT).await {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"status": "unhealthy", "reason": "backend bridge unreachable"})),
            )
                .into_response();
        }
    }
    Json(json!({"status": "healthy"})).into_response()
}

/// Prometheus metrics: the text the API server renders from its one
/// registry, asked over the bridge, so both frontends serve the same names.
/// The request carries this frontend's refusal counts, which the API server
/// exports as `mstar_http_refusals_total`.
/// 404 when the server runs without `--enable-metrics` (and in mock mode,
/// which has no registry), with the Python route's body; 503 when the bridge
/// does not answer.
async fn metrics(State(st): State<AppState>) -> Response {
    let off = || detail_error(404, "metrics are off; start the server with --enable-metrics");
    let Some(bridge) = &st.bridge else {
        return off();
    };
    let Some(reply) = bridge.metrics(&st.refusals.snapshot(), BRIDGE_PROBE_TIMEOUT).await else {
        return detail_error(503, "backend bridge unreachable");
    };
    match reply.metadata.get("content_type").and_then(Value::as_str) {
        Some(content_type) => ([(CONTENT_TYPE, content_type.to_string())], reply.data).into_response(),
        None => off(),
    }
}

async fn list_models(State(st): State<AppState>) -> Json<ModelList> {
    Json(ModelList::new(vec![ModelCard::new(st.model_name, now())]))
}

// ---- POST /v1/chat/completions -------------------------------------------

async fn chat_completions(
    State(st): State<AppState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(j) => j,
        Err(e) => return json_422(e),
    };
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

    if req.stream.unwrap_or(false) {
        return chat_sse(st, args, request_id).into_response();
    }
    match collect(result_stream(&st, &args, &request_id, false)).await {
        Ok(chunks) => Json(build_chat_response(
            &st.model_name, &request_id, chunks, st.sample_rate,
        ))
        .into_response(),
        Err((status, msg)) => error(status, &msg, error_type(status)),
    }
}

/// SSE `chat.completion.chunk`s: per-chunk deltas, finish, `[DONE]`. The role
/// rides on the first delta (the finish one when there is none), as in
/// `serving_chat.py`: clients time the first token from the first chunk.
fn chat_sse(
    st: AppState,
    args: SubmitArgs,
    request_id: String,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let base = result_stream(&st, &args, &request_id, true);
    let payloads = chat_sse_payloads(base, request_id, now(), st.model_name.clone());
    Sse::new(payloads.map(|s| Ok(Event::default().data(s)))).keep_alive(KeepAlive::default())
}

/// The `data:` payloads of [`chat_sse`], in order.
fn chat_sse_payloads(
    base: serving::OutStream,
    request_id: String,
    created: i64,
    model: String,
) -> impl futures::Stream<Item = String> {
    // Set if a terminal error is emitted mid-stream, so the tail below does NOT
    // append a clean `finish_reason:"stop"` + `[DONE]` — a failed request must
    // not look cleanly completed to the client.
    let errored = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let role_sent = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // One decoder for the whole stream; the tail flushes what it held back.
    let text = Arc::new(std::sync::Mutex::new(Utf8Stream::default()));
    let body = {
        let model = model.clone();
        let id = request_id.clone();
        let errored = errored.clone();
        let role_sent = role_sent.clone();
        let text = text.clone();
        base.filter_map(move |item| futures::future::ready(match item {
            Out::Chunk(c) => chat_delta(&c, &mut text.lock().unwrap()).map(|mut delta| {
                if !role_sent.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    delta["role"] = json!("assistant");
                }
                chunk_json(&id, created, &model, delta, None)
            }),
            // Terminal mid-stream failure: an OpenAI error event, and suppress
            // the normal stop/[DONE] tail (mstar's generator exception drops the
            // connection; the explicit error event is strictly more informative).
            Out::Error { status, message } => {
                errored.store(true, std::sync::atomic::Ordering::SeqCst);
                Some(sse_error_json(status, &message))
            }
        }))
    };
    let tail = futures::stream::once(async move {
        let mut out = Vec::new();
        if !errored.load(std::sync::atomic::Ordering::SeqCst) {
            let with_role = |mut delta: Value| {
                if !role_sent.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    delta["role"] = json!("assistant");
                }
                delta
            };
            let rest = text.lock().unwrap().flush();
            if !rest.is_empty() {
                out.push(chunk_json(&request_id, created, &model, with_role(json!({"content": rest})), None));
            }
            out.push(chunk_json(&request_id, created, &model, with_role(json!({})), Some("stop")));
            out.push("[DONE]".to_string());
        }
        futures::stream::iter(out)
    })
    .flatten();
    body.chain(tail)
}

/// One streaming chunk -> its OpenAI delta object. Text goes through the
/// stream's decoder, so a delta never ends mid-character; a token that
/// completes no character (or an end token) sends no delta.
fn chat_delta(c: &ResultChunk, text: &mut Utf8Stream) -> Option<Value> {
    Some(match c.modality.as_str() {
        "text" => {
            let content = text.feed(&c.data);
            if content.is_empty() {
                return None;
            }
            json!({"content": content})
        }
        // Streaming audio deltas are base64 16-bit PCM at the model rate.
        "audio" => json!({"audio": {"id": rid("audio"), "data": media::b64(&c.data)}}),
        "image" => json!({"content": media::png_to_data_url(&c.data)}),
        _ => json!({}),
    })
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
    let mut text_bytes: Vec<u8> = Vec::new();
    let mut audio_pcm: Vec<u8> = Vec::new();
    let mut images: Vec<Vec<u8>> = Vec::new();
    for c in &chunks {
        match c.modality.as_str() {
            "text" => text_bytes.extend_from_slice(&c.data),
            "audio" => audio_pcm.extend_from_slice(&c.data),
            "image" => images.push(c.data.clone()),
            _ => {}
        }
    }
    // Joined before decoding: a token can end mid-character.
    let text = String::from_utf8_lossy(&text_bytes).into_owned();
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

async fn audio_speech(
    State(st): State<AppState>,
    payload: Result<Json<SpeechRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(j) => j,
        Err(e) => return json_422(e),
    };
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
    // Only wav / pcm are produced natively; unlike Python (soundfile encodes
    // the real container), the Rust frontend has no compressed-audio encoder,
    // so reject an unsupported format instead of silently returning WAV bytes
    // under the requested container's name.
    if !matches!(fmt.as_str(), "wav" | "pcm") {
        return error(
            400,
            &format!(
                "response_format {fmt:?} is not supported by the Rust frontend \
                 (only 'wav' and 'pcm'); run the Python frontend for compressed \
                 containers"
            ),
            "invalid_request_error",
        );
    }

    if req.stream.unwrap_or(false) {
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
        Err((status, msg)) => return error(status, &msg, error_type(status)),
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
    payload: Result<Json<ImageGenerationRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(j) => j,
        Err(e) => return json_422(e),
    };
    let adapter = match resolve(&st, Surface::Images) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let base = match adapter.image_to_request(&req) {
        Ok(a) => a,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    // OpenAI `n`: submit n engine requests up front and aggregate their images,
    // matching serving_images.create_images. Seeded requests follow the seed
    // contract — image i uses seed + i (i > 0), so image 0 is bit-identical to
    // the same request with n=1; unseeded requests draw independent seeds.
    let n = match image_count(req.n, st.max_images_per_request) {
        Ok(n) => n,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    let seed = base.model_kwargs.get("seed").and_then(Value::as_i64);
    let futs = (0..n).map(|i| {
        let mut args = base.clone();
        if let (Some(s), true) = (seed, i > 0) {
            args.model_kwargs.insert("seed".to_string(), Value::from(s + i));
        }
        let st = st.clone();
        async move {
            let request_id = rid("img");
            collect(result_stream(&st, &args, &request_id, false)).await
        }
    });
    let mut all: Vec<ResultChunk> = Vec::new();
    for r in futures::future::join_all(futs).await {
        match r {
            Ok(chunks) => all.extend(chunks),
            Err((status, msg)) => return error(status, &msg, error_type(status)),
        }
    }
    Json(images_response(all)).into_response()
}

/// The request's `n`, 1 when omitted; outside `1..=limit` the 400 message,
/// matching serving_images.image_count.
/// `MSTAR_MAX_IMAGES_PER_REQUEST` as Python's `max_images_per_request` reads
/// it: an ASCII integer, surrounding whitespace allowed, at least 1; anything
/// else is the default 10.
fn parse_max_images(raw: Option<&str>) -> i64 {
    raw.map(str::trim)
        .filter(|s| {
            let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
            !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
        })
        .and_then(|s| s.parse::<i64>().ok())
        .map_or(10, |n| n.max(1))
}

fn image_count(n: Option<i64>, limit: i64) -> Result<i64, String> {
    match n {
        None => Ok(1),
        Some(n) if (1..=limit).contains(&n) => Ok(n),
        Some(n) => Err(format!("n must be between 1 and {limit}, got {n}")),
    }
}

// ---- POST /v1/videos/generations (Cosmos3) -------------------------------

async fn videos_generations(
    State(st): State<AppState>,
    payload: Result<Json<VideoGenerationRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(j) => j,
        Err(e) => return json_422(e),
    };
    let adapter = match resolve(&st, Surface::Videos) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let args = match adapter.video_to_request(&req, &st.upload_dir, st.allow_remote) {
        Ok(a) => a,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    schedule_upload_cleanup(&st.upload_dir, &args);
    let request_id = rid("vid");
    match collect(result_stream(&st, &args, &request_id, false)).await {
        Ok(chunks) => Json(videos_response(chunks)).into_response(),
        Err((status, msg)) => error(status, &msg, error_type(status)),
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

    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                let (status, msg) = multipart_failure(&e, format!("malformed multipart body: {e}"));
                return error(status, &msg, "invalid_request_error");
            }
        };
        let name = field.name().unwrap_or("").to_string();
        let has_file = field.file_name().is_some();
        // A field read failure (body-limit overrun, mid-upload disconnect,
        // non-UTF-8 text) must surface as a 400, not be flattened to an empty
        // value — otherwise a truncated upload becomes a misleading error.
        if name == "image" {
            image_filename = field.file_name().map(str::to_string);
            match field.bytes().await {
                Ok(b) => image_bytes = Some(b.to_vec()),
                Err(e) => {
                    let (status, msg) = multipart_failure(&e, e.to_string());
                    return error(status, &msg, "invalid_request_error");
                }
            }
        } else if name == "prompt" {
            prompt = match field.text().await {
                Ok(v) => v,
                Err(e) => {
                    let (status, msg) = multipart_failure(&e, e.to_string());
                    return error(status, &msg, "invalid_request_error");
                }
            };
        } else if known.contains(&name.as_str()) || has_file {
            if let Err(e) = field.bytes().await {
                let (status, msg) = multipart_failure(&e, e.to_string());
                return error(status, &msg, "invalid_request_error");
            }
        } else {
            let v = match field.text().await {
                Ok(v) => v,
                Err(e) => {
                    let (status, msg) = multipart_failure(&e, e.to_string());
                    return error(status, &msg, "invalid_request_error");
                }
            };
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
        Err(e) => {
            let _ = std::fs::remove_file(&image_path);
            return error(400, &e, "invalid_request_error");
        }
    };
    schedule_upload_cleanup(&st.upload_dir, &args);
    let request_id = rid("img");
    match collect(result_stream(&st, &args, &request_id, false)).await {
        Ok(chunks) => Json(images_response(chunks)).into_response(),
        Err((status, msg)) => error(status, &msg, error_type(status)),
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

/// Each Cosmos3 video chunk is an mp4 (H.264), returned base64-encoded in the
/// image endpoint's `b64_json` shape. A sound request (`generate_sound`) also
/// emits a raw PCM audio chunk that mstar muxes into the mp4 as an AAC track;
/// the Rust frontend has no mp4/PCM muxer, so it returns the video-only
/// container — the same shape mstar degrades to when muxing fails. If audio
/// chunks are present they are DROPPED and a warning is logged, since the model
/// spent extra compute generating a track the client won't receive. Muxing the
/// audio track (or a 501 when `generate_sound` is set) is a follow-up.
fn videos_response(chunks: Vec<ResultChunk>) -> Value {
    let dropped_audio = chunks.iter().filter(|c| c.modality == "audio").count();
    if dropped_audio > 0 {
        eprintln!(
            "mstar-server: WARNING dropping {dropped_audio} audio chunk(s) from a Cosmos3 \
             video response — the Rust frontend cannot mux audio into the mp4; the generated \
             sound track is discarded. Do not set 'generate_sound' on the Rust frontend, or \
             use the Python frontend for sound video."
        );
    }
    let data: Vec<Value> = chunks
        .iter()
        .filter(|c| c.modality == "video")
        .map(|c| json!({"b64_json": media::b64(&c.data), "url": Value::Null}))
        .collect();
    json!({"created": now(), "data": data})
}

// ---- POST /generate (native multipart -> NDJSON) -------------------------

async fn generate(State(st): State<AppState>, req: axum::extract::Request) -> Response {
    let mut text: Option<String> = None;
    let mut in_mods_raw: Option<String> = None;
    let mut out_mods_raw = "text".to_string();
    let mut streaming = true;
    let mut tokenize = false;
    let mut mk_raw: Option<String> = None;
    let mut request_id: Option<String> = None;
    let mut uploads = SavedUploads::new(&st.upload_dir);

    // The FastAPI endpoint this replaces uses `Form(...)` fields, which
    // Starlette parses from EITHER `multipart/form-data` (with file uploads)
    // OR `application/x-www-form-urlencoded` (what `requests.post(data=...)`
    // sends when there are no files). axum's `Multipart` extractor only
    // accepts the former, so dispatch on Content-Type to keep parity.
    let content_type = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if content_type.starts_with("application/x-www-form-urlencoded") {
        // No file uploads on this path (files force multipart), matching how
        // these clients already behave.
        let body = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
            Ok(b) => b,
            Err(e) => return detail_error(400, &format!("reading form body: {e}")),
        };
        for (k, v) in form_urlencoded::parse(&body) {
            match k.as_ref() {
                "text" => text = Some(v.into_owned()),
                "input_modalities" => in_mods_raw = Some(v.into_owned()),
                "output_modalities" => out_mods_raw = v.into_owned(),
                "streaming" => match parse_form_bool(&v) {
                    Ok(b) => streaming = b,
                    Err(e) => return detail_error(400, &e),
                },
                "tokenize" => match parse_form_bool(&v) {
                    Ok(b) => tokenize = b,
                    Err(e) => return detail_error(400, &e),
                },
                "model_kwargs" => mk_raw = Some(v.into_owned()),
                "request_id" => request_id = Some(v.into_owned()),
                _ => {}
            }
        }
        return generate_finish(
            st, text, in_mods_raw, out_mods_raw, streaming, tokenize, mk_raw,
            request_id, uploads,
        )
        .await;
    }
    if !content_type.starts_with("multipart/form-data") {
        return detail_error(
            400,
            &format!(
                "unsupported Content-Type for /generate: {content_type:?} \
                 (expected multipart/form-data or application/x-www-form-urlencoded)"
            ),
        );
    }
    let mut mp = match Multipart::from_request(req, &st).await {
        Ok(m) => m,
        Err(e) => return detail_error(400, &e.to_string()),
    };

    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                let (status, msg) = multipart_failure(&e, format!("malformed multipart body: {e}"));
                return detail_error(status, &msg);
            }
        };
        let name = field.name().unwrap_or("").to_string();
        if name == "files" {
            let filename = field.file_name().unwrap_or("").to_string();
            let modality = media::modality_from_filename(&filename);
            if modality == "unknown" {
                return detail_error(400, &format!("Cannot determine modality for file: {filename}"));
            }
            let bytes = match field.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    let (status, msg) = multipart_failure(&e, e.to_string());
                    return detail_error(status, &msg);
                }
            };
            if let Err(e) = std::fs::create_dir_all(&st.upload_dir) {
                return detail_error(500, &e.to_string());
            }
            // Sanitize: take ONLY the final path component of the
            // client-supplied name. `join` treats embedded separators as
            // separators, so `../../x` would escape upload_dir (arbitrary
            // write — and the cleanup filter is component-based, so it would
            // then arbitrary-delete). file_name() strips any directory parts.
            let safe = std::path::Path::new(&filename)
                .file_name()
                .and_then(|s| s.to_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("upload");
            let save_path = st.upload_dir.join(format!("{}_{}", Uuid::new_v4().simple(), safe));
            if let Err(e) = std::fs::write(&save_path, &bytes) {
                return detail_error(500, &e.to_string());
            }
            uploads
                .paths
                .entry(modality.to_string())
                .or_default()
                .push(save_path.to_string_lossy().into_owned());
        } else {
            // Propagate a field read failure (body-limit overrun, mid-upload
            // disconnect, non-UTF-8) as a 400 rather than silently treating it
            // as an empty value.
            let val = match field.text().await {
                Ok(v) => v,
                Err(e) => {
                    let (status, msg) = multipart_failure(&e, format!("reading form field {name:?}: {e}"));
                    return detail_error(status, &msg);
                }
            };
            match name.as_str() {
                "text" => text = Some(val),
                "input_modalities" => in_mods_raw = Some(val),
                "output_modalities" => out_mods_raw = val,
                "streaming" => match parse_form_bool(&val) {
                    Ok(b) => streaming = b,
                    Err(e) => return detail_error(400, &e),
                },
                "tokenize" => match parse_form_bool(&val) {
                    Ok(b) => tokenize = b,
                    Err(e) => return detail_error(400, &e),
                },
                "model_kwargs" => mk_raw = Some(val),
                "request_id" => request_id = Some(val),
                _ => {}
            }
        }
    }

    generate_finish(
        st, text, in_mods_raw, out_mods_raw, streaming, tokenize, mk_raw,
        request_id, uploads,
    )
    .await
}

/// The files a request saved under the upload directory (a `/generate` form,
/// a chat's attachments), until the request takes them with
/// [`SavedUploads::keep`]. A refusal drops them unkept, which deletes them:
/// the deferred cleanup only covers submitted requests.
struct SavedUploads {
    upload_dir: PathBuf,
    paths: BTreeMap<String, Vec<String>>,
}

impl SavedUploads {
    fn new(upload_dir: &std::path::Path) -> Self {
        Self { upload_dir: upload_dir.to_path_buf(), paths: BTreeMap::new() }
    }

    fn keep(mut self) -> BTreeMap<String, Vec<String>> {
        std::mem::take(&mut self.paths)
    }
}

impl Drop for SavedUploads {
    fn drop(&mut self) {
        remove_uploads(&self.upload_dir, &self.paths);
    }
}

/// Shared tail of `/generate` once the form fields are parsed (from either
/// multipart or urlencoded): build the request and stream NDJSON.
#[allow(clippy::too_many_arguments)]
async fn generate_finish(
    st: AppState,
    mut text: Option<String>,
    in_mods_raw: Option<String>,
    out_mods_raw: String,
    streaming: bool,
    tokenize: bool,
    mk_raw: Option<String>,
    request_id: Option<String>,
    uploads: SavedUploads,
) -> Response {
    // An empty id means none, as the Python /generate's form parsing reads it.
    let request_id = request_id.filter(|id| !id.is_empty());
    if let Err(msg) = request_id.as_deref().map_or(Ok(()), check_request_id) {
        // `uploads` drops unkept here, which deletes the saved files.
        return detail_error(400, &msg);
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
            let mut m: Vec<String> = uploads.paths.keys().cloned().collect();
            if text.as_ref().map(|t| !t.is_empty()).unwrap_or(false) {
                m.push("text".to_string());
            }
            m
        }
    };
    if let Some(message) = video_frame_refusal(&input_modalities, &output_modalities, streaming) {
        return detail_error(400, message);
    }
    let model_kwargs: Map<String, Value> = match mk_raw {
        Some(raw) if !raw.is_empty() => match serde_json::from_str::<Value>(&raw) {
            Ok(Value::Object(m)) => m,
            Ok(_) => return detail_error(400, "model_kwargs must be a JSON object"),
            Err(e) => return detail_error(400, &format!("model_kwargs JSON: {e}")),
        },
        _ => Map::new(),
    };
    let request_id = request_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    // `tokenize=true`: encode the text in Rust and submit token ids; the
    // reply's text streams back as token ids and detokenizes here too.
    let mut tokens: Option<Vec<u32>> = None;
    if tokenize {
        let Some(tok) = &st.tok else {
            return detail_error(400, "tokenize=true requires the server to be started \
                                      with MSTAR_TOKENIZER");
        };
        tokens = Some(tok.encode(text.as_deref().unwrap_or("")));
        text = None; // the model side must use the ids, not re-tokenize
    }

    let args = SubmitArgs {
        text,
        tokens,
        file_paths: uploads.keep(),
        input_modalities,
        output_modalities,
        model_kwargs,
        parts: Vec::new(),
    };

    schedule_upload_cleanup(&st.upload_dir, &args);
    if let Some(bridge) = &st.bridge {
        bridge.wait_for_request_id(&request_id, DUPLICATE_ID_GRACE).await;
    }
    if streaming {
        let base = match try_result_stream(&st, &args, &request_id, true) {
            Ok(base) => base,
            Err((status, message)) => return detail_error(status, &message),
        };
        let body = base.map(|item| {
            Ok::<Bytes, Infallible>(Bytes::from(match item {
                Out::Chunk(c) => ndjson_line(&c),
                Out::Error { message, .. } => {
                    let mut s = json!({"error": message}).to_string();
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
        Err((status, msg)) => return detail_error(status, &msg),
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

/// The Python `/generate`'s refusals for `video_frame`, worded the same: raw
/// frames are an output only, and only a stream can carry them.
fn video_frame_refusal(input: &[String], output: &[String], streaming: bool) -> Option<&'static str> {
    if !streaming && output.iter().any(|m| m == "video_frame") {
        return Some(
            "Output modality video_frame requires streaming=true; raw frame \
             chunks cannot be returned as an aggregated response.",
        );
    }
    if input.iter().any(|m| m == "video_frame") {
        return Some("'video_frame' is an output-only modality");
    }
    None
}

/// Delete a refused request's uploads now: it never reaches the deferred
/// cleanup in `schedule_upload_cleanup`.
fn remove_uploads(upload_dir: &std::path::Path, file_paths: &BTreeMap<String, Vec<String>>) {
    for path in file_paths.values().flatten().map(PathBuf::from) {
        if path.starts_with(upload_dir) {
            let _ = std::fs::remove_file(path);
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use mstar_rust::communicator::RawZmqCommunicator;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn state(bridge: Option<Arc<Bridge>>) -> AppState {
        AppState {
            model_name: "m".into(),
            python_adapter: String::new(),
            upload_dir: std::env::temp_dir(),
            allow_remote: false,
            sample_rate: 24000,
            request_timeout: std::time::Duration::from_secs(5),
            max_images_per_request: 10,
            tok: None,
            bridge,
            refusals: Arc::new(refusals::Refusals::default()),
        }
    }

    /// The Python bridge loop's side of the socket: answers every `metrics`
    /// message with `metadata`, as `RustFrontendBridge` does.
    struct FakeBackend {
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
        /// Every `metrics` request it answered.
        asked: Arc<std::sync::Mutex<Vec<Value>>>,
    }

    impl FakeBackend {
        fn start(dir: &std::path::Path, text: &'static [u8], metadata: Value) -> Self {
            #[derive(serde::Serialize)]
            struct Reply<'a> {
                t: &'static str,
                rid: &'a str,
                sub: u64,
                #[serde(with = "serde_bytes")]
                data: &'a [u8],
                metadata: &'a Value,
            }
            let peer = RawZmqCommunicator::bind("bridge", dir).unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let stopping = stop.clone();
            let asked = Arc::new(std::sync::Mutex::new(Vec::new()));
            let asking = asked.clone();
            let thread = std::thread::spawn(move || {
                while !stopping.load(Ordering::Relaxed) {
                    let Some(frame) = peer.recv_timeout(std::time::Duration::from_millis(20)) else {
                        continue;
                    };
                    let msg: Value = rmp_serde::from_slice(&frame).unwrap();
                    if msg["t"] != "metrics" {
                        continue;
                    }
                    asking.lock().unwrap().push(msg.clone());
                    let reply = Reply {
                        t: "metrics",
                        rid: msg["rid"].as_str().unwrap(),
                        sub: msg["sub"].as_u64().unwrap(),
                        data: text,
                        metadata: &metadata,
                    };
                    peer.send("frontend", &rmp_serde::to_vec_named(&reply).unwrap()).unwrap();
                }
            });
            Self { stop, thread: Some(thread), asked }
        }
    }

    impl Drop for FakeBackend {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    fn socket_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mstar-metrics-{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn body(resp: Response) -> (StatusCode, Option<String>, Vec<u8>) {
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(CONTENT_TYPE)
            .map(|v| v.to_str().unwrap().to_string());
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, content_type, bytes.to_vec())
    }

    #[tokio::test]
    async fn metrics_serve_the_backend_s_text_with_its_content_type() {
        let dir = socket_dir();
        let text: &[u8] = b"# TYPE mstar_requests_total counter\nmstar_requests_total{model_name=\"m\"} 3.0\n";
        let content_type = "text/plain; version=1.0.0; charset=utf-8";
        let _backend = FakeBackend::start(&dir, text, json!({"content_type": content_type}));
        let bridge = Arc::new(Bridge::new(dir.to_str().unwrap()).unwrap());

        let (status, ct, data) = body(metrics(State(state(Some(bridge)))).await).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(ct.as_deref(), Some(content_type));
        assert_eq!(data, text);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn metrics_ask_the_backend_with_this_frontend_s_refusal_counts() {
        let dir = socket_dir();
        let backend = FakeBackend::start(&dir, b"", json!({"content_type": "text/plain"}));
        let bridge = Arc::new(Bridge::new(dir.to_str().unwrap()).unwrap());
        let st = state(Some(bridge));
        st.refusals.count("/generate", 409);
        st.refusals.count("unmatched", 404);
        st.refusals.count("/generate", 409);

        assert_eq!(body(metrics(State(st.clone())).await).await.0, StatusCode::OK);
        st.refusals.count("/generate", 409);
        assert_eq!(body(metrics(State(st)).await).await.0, StatusCode::OK);

        let asked: Vec<Value> = backend.asked.lock().unwrap().iter().map(|m| m["refusals"].clone()).collect();
        assert_eq!(asked, [
            json!({"route": ["/generate", "unmatched"], "status": [409, 404], "count": [2, 1]}),
            json!({"route": ["/generate", "unmatched"], "status": [409, 404], "count": [3, 1]}),
        ]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn metrics_are_404_when_the_backend_runs_without_them_and_in_mock_mode() {
        let dir = socket_dir();
        let _backend = FakeBackend::start(&dir, b"", json!({}));
        let bridge = Arc::new(Bridge::new(dir.to_str().unwrap()).unwrap());

        let (status, _, data) = body(metrics(State(state(Some(bridge)))).await).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(String::from_utf8_lossy(&data).contains("--enable-metrics"));

        let (status, _, _) = body(metrics(State(state(None))).await).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn metrics_are_503_when_the_bridge_does_not_answer() {
        let dir = socket_dir();
        let bridge = Arc::new(Bridge::new(dir.to_str().unwrap()).unwrap());

        let (status, _, _) = body(metrics(State(state(Some(bridge)))).await).await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn image_count_refuses_n_outside_one_to_limit() {
        assert_eq!(image_count(None, 10), Ok(1));
        assert_eq!(image_count(Some(1), 10), Ok(1));
        assert_eq!(image_count(Some(10), 10), Ok(10));
        for n in [0, -1, 11, i64::MAX] {
            assert_eq!(
                image_count(Some(n), 10),
                Err(format!("n must be between 1 and 10, got {n}"))
            );
        }
        assert_eq!(image_count(Some(3), 2), Err("n must be between 1 and 2, got 3".to_string()));
    }

    async fn generate_with(
        st: AppState, input: Option<&str>, output: &str, streaming: bool,
        file_paths: BTreeMap<String, Vec<String>>,
    ) -> (StatusCode, Vec<u8>) {
        let mut uploads = SavedUploads::new(&st.upload_dir);
        uploads.paths = file_paths;
        let resp = generate_finish(
            st, Some("hi".into()), input.map(str::to_string), output.into(), streaming,
            false, None, None, uploads,
        )
        .await;
        let (status, _, data) = body(resp).await;
        (status, data)
    }

    #[tokio::test]
    async fn a_streamed_chat_puts_the_role_on_its_first_delta() {
        for (outputs, roles) in [
            (vec!["text", "audio"], vec![true, false, false]),
            (vec![], vec![true]),
        ] {
            let args = SubmitArgs {
                text: Some("hi".into()),
                output_modalities: outputs.iter().map(|m| m.to_string()).collect(),
                ..SubmitArgs::default()
            };
            let (status, _, data) =
                body(chat_sse(state(None), args, "r".into()).into_response()).await;
            assert_eq!(status, StatusCode::OK);
            let text = String::from_utf8(data).unwrap();
            let events: Vec<&str> = text
                .lines()
                .filter_map(|l| l.strip_prefix("data:"))
                .map(str::trim)
                .collect();
            assert_eq!(events.last(), Some(&"[DONE]"), "{outputs:?}");
            let chunks: Vec<Value> = events[..events.len() - 1]
                .iter()
                .map(|e| serde_json::from_str(e).unwrap())
                .collect();
            let got: Vec<bool> = chunks
                .iter()
                .map(|c| c["choices"][0]["delta"].get("role").is_some())
                .collect();
            assert_eq!(got, roles, "which deltas carry the role, for {outputs:?}");
            assert_eq!(
                chunks.last().unwrap()["choices"][0]["finish_reason"],
                "stop"
            );
            if outputs.first() == Some(&"text") {
                assert_eq!(
                    chunks[0]["choices"][0]["delta"],
                    json!({"role": "assistant", "content": "[mock] hi"})
                );
            }
        }
    }

    #[tokio::test]
    async fn generate_refuses_video_frame_as_the_python_frontend_does() {
        let streaming_only = "Output modality video_frame requires streaming=true; raw frame \
                              chunks cannot be returned as an aggregated response.";
        for (input, output, streaming, detail) in [
            (None, "video_frame", false, streaming_only),
            (None, "text, video_frame", false, streaming_only),
            (Some("video_frame"), "text", true, "'video_frame' is an output-only modality"),
            (Some("text,video_frame"), "text", false, "'video_frame' is an output-only modality"),
        ] {
            let (status, data) = generate_with(state(None), input, output, streaming, BTreeMap::new()).await;
            let body: Value = serde_json::from_slice(&data).unwrap();
            assert_eq!((status, body), (StatusCode::BAD_REQUEST, json!({"detail": detail})), "{input:?} {output:?}");
        }
        // a stream carries the frames
        let (status, _) = generate_with(state(None), None, "video_frame", true, BTreeMap::new()).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn a_refused_video_frame_request_leaves_no_upload_behind() {
        let mut st = state(None);
        st.upload_dir = socket_dir();
        let upload = st.upload_dir.join("clip.wav");
        std::fs::write(&upload, b"RIFF").unwrap();
        let files = BTreeMap::from([("audio".to_string(), vec![upload.to_string_lossy().into_owned()])]);

        let (status, _) = generate_with(st.clone(), None, "video_frame", false, files).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(!upload.exists());
        let _ = std::fs::remove_dir_all(&st.upload_dir);
    }

    fn multipart_request(uri: &str, parts: &[(&str, Option<&str>, &[u8])]) -> axum::extract::Request {
        let mut payload = Vec::new();
        for (name, filename, data) in parts {
            payload.extend_from_slice(b"--XBOUNDARY\r\n");
            let disposition = match filename {
                Some(f) => format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\n\r\n"),
                None => format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"),
            };
            payload.extend_from_slice(disposition.as_bytes());
            payload.extend_from_slice(data);
            payload.extend_from_slice(b"\r\n");
        }
        payload.extend_from_slice(b"--XBOUNDARY--\r\n");
        axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header(CONTENT_TYPE, "multipart/form-data; boundary=XBOUNDARY")
            .body(Body::from(payload))
            .unwrap()
    }

    #[tokio::test]
    async fn every_refusal_after_the_form_leaves_no_upload_behind() {
        let mut st = state(None);
        st.upload_dir = socket_dir();
        let long_id = "a".repeat(MAX_REQUEST_ID_LEN + 1);
        for (field, value, detail) in [
            ("model_kwargs", "[1]", "model_kwargs must be a JSON object"),
            ("model_kwargs", "{", "model_kwargs JSON: EOF while parsing an object at line 1 column 1"),
            ("tokenize", "true", "tokenize=true requires the server to be started with MSTAR_TOKENIZER"),
            ("input_modalities", "video_frame", "'video_frame' is an output-only modality"),
            ("request_id", "a/b", "request_id must be at most 128 characters from [A-Za-z0-9._:-]"),
            ("request_id", long_id.as_str(), "request_id must be at most 128 characters from [A-Za-z0-9._:-]"),
        ] {
            let req = multipart_request("/generate", &[
                ("files", Some("clip.wav"), b"RIFF"),
                (field, None, value.as_bytes()),
            ]);

            let (status, _, data) = body(generate(State(st.clone()), req).await).await;

            let body: Value = serde_json::from_slice(&data).unwrap();
            assert_eq!((status, body), (StatusCode::BAD_REQUEST, json!({"detail": detail})), "{field}={value}");
            assert_eq!(std::fs::read_dir(&st.upload_dir).unwrap().count(), 0, "{field}={value}");
        }
        let _ = std::fs::remove_dir_all(&st.upload_dir);
    }

    #[tokio::test]
    async fn a_file_refused_mid_form_takes_the_saved_ones_with_it() {
        let mut st = state(None);
        st.upload_dir = socket_dir();
        let req = multipart_request("/generate", &[
            ("files", Some("clip.wav"), b"RIFF"),
            ("files", Some("notes.xyz"), b"?"),
        ]);

        let (status, _, data) = body(generate(State(st.clone()), req).await).await;

        let body: Value = serde_json::from_slice(&data).unwrap();
        assert_eq!(
            (status, body),
            (StatusCode::BAD_REQUEST, json!({"detail": "Cannot determine modality for file: notes.xyz"}))
        );
        assert_eq!(std::fs::read_dir(&st.upload_dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&st.upload_dir);
    }

    #[tokio::test]
    async fn an_edit_the_model_cannot_do_leaves_no_upload_behind() {
        let mut st = state(None);
        st.upload_dir = socket_dir();
        st.python_adapter = "Cosmos3EdgeAdapter".into();
        let req = multipart_request("/v1/images/edits", &[
            ("image", Some("cat.png"), b"\x89PNG"),
            ("prompt", None, b"a hat"),
        ]);
        let mp = Multipart::from_request(req, &st).await.unwrap();

        let (status, _, data) = body(images_edits(State(st.clone()), mp).await).await;

        let body: Value = serde_json::from_slice(&data).unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["message"], "image editing is not supported by this model");
        assert_eq!(std::fs::read_dir(&st.upload_dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&st.upload_dir);
    }

    #[tokio::test]
    async fn a_chat_refused_after_an_attachment_leaves_no_upload_behind() {
        let mut st = state(None);
        st.upload_dir = socket_dir();
        let image = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}});
        let bad_image = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,***"}});
        let audio = json!({"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}});
        let mut left = Vec::new();
        for (adapter, request, detail) in [
            (
                "Qwen3_5Adapter",
                json!({"messages": [{"role": "user", "content": [image, audio, bad_image]}]}),
                "data URL base64 decode failed: Invalid symbol 42, offset 0.",
            ),
            (
                "Qwen3OmniAdapter",
                json!({"messages": [{"role": "user", "content": [image, audio]}, {"role": "tool", "content": "42"}]}),
                "a message's role must be system, developer, user or assistant, not 'tool'",
            ),
            (
                "Cosmos3EdgeAdapter",
                json!({"messages": [{"role": "user", "content": [image]}], "chat_template_kwargs": true}),
                "argument of type 'bool' is not iterable",
            ),
        ] {
            st.python_adapter = adapter.into();
            let req: ChatCompletionRequest = serde_json::from_value(request).unwrap();

            let (status, _, data) = body(chat_completions(State(st.clone()), Ok(Json(req))).await).await;

            let body: Value = serde_json::from_slice(&data).unwrap();
            assert_eq!((status, body["error"]["message"].as_str()), (StatusCode::BAD_REQUEST, Some(detail)), "{adapter}");
            left.push((adapter, std::fs::read_dir(&st.upload_dir).unwrap().count()));
            let _ = std::fs::remove_dir_all(&st.upload_dir);
            std::fs::create_dir_all(&st.upload_dir).unwrap();
        }
        let _ = std::fs::remove_dir_all(&st.upload_dir);
        assert_eq!(left, [("Qwen3_5Adapter", 0), ("Qwen3OmniAdapter", 0), ("Cosmos3EdgeAdapter", 0)]);
    }

    #[test]
    fn max_images_parses_as_python_does() {
        for (raw, cap) in [
            (None, 10), (Some("2"), 2), (Some(" 2 "), 2), (Some("2\n"), 2), (Some("+3"), 3),
            (Some("0"), 1), (Some("-1"), 1), (Some(""), 10), (Some("abc"), 10), (Some("2_0"), 10),
            (Some("1.5"), 10), (Some("\u{ff12}"), 10),
        ] {
            assert_eq!(parse_max_images(raw), cap, "{raw:?}");
        }
    }

    #[test]
    fn request_id_check_matches_python() {
        let longest = "a".repeat(MAX_REQUEST_ID_LEN);
        let minted = Uuid::new_v4().to_string();
        let prefixed = rid("chatcmpl");
        for ok in ["", "warmup-0", "obs.1:a_b-C", &longest, &minted, &prefixed] {
            assert!(check_request_id(ok).is_ok(), "{ok:?}");
        }
        let too_long = "a".repeat(MAX_REQUEST_ID_LEN + 1);
        for bad in [too_long.as_str(), "a b", "a/b", "a#1", "caf\u{e9}", "a\n"] {
            assert_eq!(
                check_request_id(bad),
                Err("request_id must be at most 128 characters from [A-Za-z0-9._:-]".to_string()),
                "{bad:?}"
            );
        }
    }

    /// "中文 😀" as byte-level tokens would arrive: cut inside 中 and inside
    /// the emoji (the same pieces as `test/modular/test_utf8_split_tokens.py`).
    fn split_text_chunks() -> Vec<ResultChunk> {
        let data = "中文 😀".as_bytes();
        [&data[..1], &data[1..4], &data[4..8], &data[8..]]
            .iter()
            .map(|p| ResultChunk { modality: "text".to_string(), data: p.to_vec(), metadata: Value::Null })
            .collect()
    }

    fn streamed_content(items: Vec<Out>) -> (String, Vec<Value>) {
        let base: serving::OutStream = Box::pin(futures::stream::iter(items));
        let payloads: Vec<String> = futures::executor::block_on(
            chat_sse_payloads(base, "r".to_string(), 0, "m".to_string()).collect::<Vec<_>>(),
        );
        let events: Vec<Value> = payloads
            .iter()
            .filter(|p| *p != "[DONE]")
            .map(|p| serde_json::from_str(p).unwrap())
            .collect();
        let content = events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        (content, events)
    }

    #[test]
    fn chat_response_joins_text_before_decoding() {
        let resp = build_chat_response("m", "r", split_text_chunks(), 24000);
        assert_eq!(resp["choices"][0]["message"]["content"], "中文 😀");
    }

    #[test]
    fn chat_stream_deltas_hold_split_characters() {
        let items = split_text_chunks().into_iter().map(Out::Chunk).collect();
        let (content, events) = streamed_content(items);
        assert_eq!(content, "中文 😀");
        assert!(events.iter().all(|e| !e.to_string().contains('\u{FFFD}')));
        assert_eq!(events.last().unwrap()["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn chat_stream_sends_no_empty_deltas() {
        // the first piece completes no character, and an end token decodes to nothing
        let mut chunks = split_text_chunks();
        chunks.push(ResultChunk { modality: "text".to_string(), data: Vec::new(), metadata: Value::Null });
        let (content, events) = streamed_content(chunks.into_iter().map(Out::Chunk).collect());
        assert_eq!(content, "中文 😀");
        let deltas: Vec<&Value> = events.iter().map(|e| &e["choices"][0]["delta"]).collect();
        assert_eq!(deltas.len(), 4, "3 text deltas (the role on the first), stop: {deltas:?}");
        assert!(deltas.iter().all(|d| d.get("content").map_or(true, |c| c != "")), "{deltas:?}");
    }

    #[test]
    fn chat_stream_flushes_an_unfinished_character_before_stop() {
        let mut chunks = split_text_chunks();
        chunks.last_mut().unwrap().data.pop(); // the stream ends inside the emoji
        let (content, events) = streamed_content(chunks.into_iter().map(Out::Chunk).collect());
        assert_eq!(content, "中文 \u{FFFD}");
        let n = events.len();
        assert_eq!(events[n - 2]["choices"][0]["delta"]["content"], "\u{FFFD}");
        assert_eq!(events[n - 1]["choices"][0]["finish_reason"], "stop");
    }
}
