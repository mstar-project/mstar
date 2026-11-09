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
//!   POST /v1/audio/speech          (streaming WAV/PCM, SSE deltas, or full container)
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
mod generation_kwargs;
mod media;
mod protocol;
mod refusals;
mod served_models;
mod serving;
mod sndfile;
mod speech_chunking;
mod tokenizer;
mod transcription;
mod unicode_marks;
mod utf8;
mod ws;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    body::{Body, Bytes},
    extract::{multipart::MultipartError, rejection::JsonRejection, DefaultBodyLimit, FromRequest, Multipart, State},
    http::{header::CONTENT_TYPE, Request, StatusCode},
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

use adapters::{OpenAIAdapter, Part, SubmitArgs};
use bridge::{Bridge, ResultChunk};
use protocol::{
    ChatCompletionRequest, ImageGenerationRequest, SpeechRequest,
    VideoGenerationRequest,
};
use serving::{
    collect, now, result_stream, rid, try_result_stream, AppState, Out, OutStream, DUPLICATE_ID_GRACE,
};
use utf8::Utf8Stream;

#[tokio::main]
async fn main() {
    const USAGE: &str = concat!(
        "usage: mstar-server <model_name> [port] [conductor_socket_dir] [upload_dir]\n",
        "       mstar-server --source-hash",
    );
    // The content hashes of the rust/ and rust/server/ sources this binary was
    // built from: the rlib's own, then build.rs's. find_server_binary reads
    // them from the no-argument usage, since every mstar-server exits on no
    // arguments while an older one would serve `--source-hash` as a model.
    let sources = format!(
        "{} {}",
        mstar_rust::BUILD_SOURCE_HASH,
        env!("MSTAR_SERVER_SOURCE_HASH")
    );
    let Some(model_name) = std::env::args().nth(1) else {
        eprintln!("{USAGE}\nbuilt from sources {sources}");
        std::process::exit(2);
    };
    if model_name == "-h" || model_name == "--help" {
        println!("{USAGE}\nbuilt from sources {sources}");
        return;
    }
    if model_name == "--source-hash" {
        println!("{sources}");
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
    let request_timeout = request_timeout(std::env::var("MSTAR_REQUEST_TIMEOUT_S").ok().as_deref());
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
    if !python_adapter.is_empty() && adapters::from_python(&python_adapter).is_none() {
        eprintln!(
            "WARNING: {model_name}'s OpenAI adapter {python_adapter} has no port in this \
             frontend: /v1/* answers 404 here, /generate is served"
        );
    }

    sndfile::init();
    let served = Arc::new(served_models::ServedModels::from_env(&model_name));

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
        served,
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
        .unwrap_or(512);
    let permits = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
    // Uploads (audio/video multipart) routinely exceed axum's 2 MB default.
    let body_limit_mb = body_limit_mb();

    let api = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/audio/speech", post(audio_speech))
        // an upload of up to 30 minutes: the route's own, larger body limit
        .route(
            "/v1/audio/transcriptions",
            post(transcription::audio_transcriptions)
                .layer(DefaultBodyLimit::max(transcription::body_limit_bytes(body_limit_mb << 20))),
        )
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
        .route("/v1/audio/voices", get(audio_voices))
        // long-lived sockets: outside the admission cap, as on the Python frontend
        .route("/", get(ws::openpi::openpi_ws))
        .route("/healthz", get(ws::openpi::healthz))
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
    // MSTAR_ROOT_PATH: routed before the router sees the path, as Starlette's
    // root_path is, so the prefixed and the bare path both reach a route.
    let root_path = std::env::var("MSTAR_ROOT_PATH").unwrap_or_default();
    let app = tower::Layer::layer(&tower::util::MapRequestLayer::new(move |req: Request<Body>| {
        strip_root_path(&root_path, req)
    }), app);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    eprintln!("mstar-server (Rust frontend) for model {model_name:?} on http://{addr}");
    axum::serve(listener, axum::ServiceExt::<Request<Body>>::into_make_service(app))
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
    Transcriptions,
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

/// The path a route sees under `MSTAR_ROOT_PATH`, by Starlette's
/// `get_route_path`: a path under `root` loses it, and any other path is
/// routed as sent. `root` is taken as set, so `/proj/job/` leaves
/// `/proj/job/v1/models` whole, as Starlette does.
fn route_path<'a>(root: &str, path: &'a str) -> &'a str {
    if root.is_empty() {
        return path;
    }
    match path.strip_prefix(root) {
        // Starlette's "" matches no route; neither does "/"
        Some("") => "/",
        Some(rest) if rest.starts_with('/') => rest,
        _ => path,
    }
}

fn strip_root_path(root: &str, mut req: Request<Body>) -> Request<Body> {
    let path = req.uri().path();
    let routed = route_path(root, path);
    if routed.len() != path.len() {
        let mut target = routed.to_string();
        if let Some(q) = req.uri().query() {
            target.push('?');
            target.push_str(q);
        }
        if let Ok(pq) = target.parse::<axum::http::uri::PathAndQuery>() {
            let mut parts = req.uri().clone().into_parts();
            parts.path_and_query = Some(pq);
            if let Ok(uri) = axum::http::Uri::from_parts(parts) {
                *req.uri_mut() = uri;
            }
        }
    }
    req
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
        (413, body_too_large())
    } else {
        (400, fallback)
    }
}

/// Blocking work off the async workers: decoding a data URL, writing an
/// upload, a remote fetch of up to 30 s. On a worker it stalls every
/// connection that worker drives; the Python routes run it in a thread too.
async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    match tokio::task::spawn_blocking(work).await {
        Ok(v) => v,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

fn body_too_large() -> String {
    let mb = body_limit_mb();
    format!("request body is larger than the {mb} MiB limit (MSTAR_MAX_BODY_MB)")
}

/// A body read that stopped at its limit (`to_bytes` wraps `LengthLimitError`).
fn is_length_limit(e: &axum::Error) -> bool {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(err) = source {
        if err.is::<http_body_util::LengthLimitError>() {
            return true;
        }
        source = err.source();
    }
    false
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

/// Turn a `Json` extractor rejection into the Python frontend's answer to a
/// schema-invalid body: `400` in the OpenAI envelope, like every other refused
/// input (axum's default is `400` + `text/plain`). Handlers take
/// `Result<Json<T>, JsonRejection>` and route the error through here.
fn json_400(rej: JsonRejection) -> Response {
    if rej.status() == StatusCode::PAYLOAD_TOO_LARGE {
        let mb = body_limit_mb();
        return error(
            413,
            &format!("request body is larger than the {mb} MiB limit (MSTAR_MAX_BODY_MB)"),
            "invalid_request_error",
        );
    }
    error(400, &rej.body_text(), "invalid_request_error")
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
fn resolve(st: &AppState, surface: Surface) -> Result<&'static dyn OpenAIAdapter, Response> {
    let message = if st.python_adapter.is_empty() {
        format!("Model '{}' has no OpenAI-compatible adapter; use POST /generate", st.model_name)
    } else {
        format!(
            "Model '{}' has an OpenAI adapter ({}) the Rust frontend does not serve; \
             use POST /generate, or serve without --rust-frontend",
            st.model_name, st.python_adapter
        )
    };
    let adapter = adapters::from_python(&st.python_adapter)
        .ok_or_else(|| error(404, &message, "model_not_found"))?;
    let ok = match surface {
        Surface::Chat => adapter.supports_chat(),
        Surface::Speech => adapter.supports_speech(),
        Surface::Images => adapter.supports_images(),
        Surface::Videos => adapter.supports_videos(),
        Surface::Transcriptions => adapter.supports_transcriptions(),
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

/// A request's uploads, scheduled for removal ([`schedule_upload_cleanup`])
/// when this drops: when whatever it moved into is done with them.
struct DeferredCleanup {
    upload_dir: PathBuf,
    args: SubmitArgs,
}

impl Drop for DeferredCleanup {
    fn drop(&mut self) {
        if tokio::runtime::Handle::try_current().is_ok() {
            schedule_upload_cleanup(&self.upload_dir, &self.args);
        }
    }
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

/// `GET /v1/models`: the base model, then each served LoRA adapter naming it as its parent.
async fn list_models(State(st): State<AppState>) -> Json<Value> {
    Json(st.served.cards(&st.model_name, now()))
}

/// `APIServer.lora_kwargs`: the LoRA adapter a request runs under, named by its
/// `model` field or its `lora_adapter` kwarg, put in its kwargs; `None` for the
/// base model. An adapter this deployment does not serve is the 400.
fn select_lora(st: &AppState, model: Option<&str>, args: &mut SubmitArgs) -> Result<Option<String>, Response> {
    st.served.select(model, &mut args.model_kwargs).map_err(|e| error(400, &e, "invalid_request_error"))
}

/// `GET /v1/audio/voices`, as router.py's `audio_voices`: the model's list,
/// asked over the bridge (none in mock mode).
async fn audio_voices(State(st): State<AppState>) -> Response {
    if let Err(r) = resolve(&st, Surface::Speech) {
        return r;
    }
    let voices = match &st.bridge {
        Some(bridge) => bridge.voices(BRIDGE_PROBE_TIMEOUT).await,
        None => Ok(None),
    };
    match voices {
        Ok(Some(list)) => Json(list).into_response(),
        Ok(None) => {
            let message = format!("Model '{}' does not publish a voice list", st.model_name);
            error(404, &message, error_type(404))
        }
        Err((status, message)) => error(status, &message, error_type(status)),
    }
}

// ---- POST /v1/chat/completions -------------------------------------------

async fn chat_completions(
    State(st): State<AppState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(j) => j,
        Err(e) => return json_400(e),
    };
    let adapter = match resolve(&st, Surface::Chat) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let audio_format = match chat_audio_format(&req, st.sample_rate, crate::sndfile::available()) {
        Ok(fmt) => fmt,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    let (dir, remote) = (st.upload_dir.clone(), st.allow_remote);
    let (req, translated) = blocking(move || {
        let args = adapter.chat_to_request(&req, &dir, remote);
        (req, args)
    })
    .await;
    let mut args = match translated {
        Ok(a) => a,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    // Resolved before it is submitted: the response names the adapter that served it.
    let model_name = match select_lora(&st, req.model.as_deref(), &mut args) {
        Ok(served) => served.unwrap_or_else(|| st.model_name.clone()),
        Err(r) => {
            remove_uploads(&st.upload_dir, &args.file_paths);
            return r;
        }
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
        let base = match open_stream(result_stream(&st, &args, &request_id, true)).await {
            Ok(base) => base,
            Err((status, msg)) => return error(status, &msg, error_type(status)),
        };
        return chat_sse(base, request_id, model_name).into_response();
    }
    match collect(result_stream(&st, &args, &request_id, false)).await {
        Ok(chunks) => {
            let rate = st.sample_rate;
            match blocking(move || build_chat_response(&model_name, &request_id, chunks, rate, audio_format)).await {
                Ok(body) => Json(body).into_response(),
                Err(e) => error(500, &e, "server_error"),
            }
        }
        Err((status, msg)) => error(status, &msg, error_type(status)),
    }
}

/// `base` once its first item is in, replayed; a refusal is `Err` before any 200 is sent.
/// A client that leaves meanwhile drops this future, and the stream's guard aborts.
async fn open_stream(mut base: OutStream) -> Result<OutStream, (u16, String)> {
    match base.next().await {
        Some(Out::Error { status, message }) => Err((status, message)),
        first => Ok(Box::pin(futures::stream::iter(first).chain(base))),
    }
}

/// SSE `chat.completion.chunk`s: per-chunk deltas, finish, `[DONE]`. The role
/// rides on the first delta (the finish one when there is none), as in
/// `serving_chat.py`: clients time the first token from the first chunk.
fn chat_sse(
    base: OutStream,
    request_id: String,
    model_name: String,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let payloads = chat_sse_payloads(base, request_id, now(), model_name);
    Sse::new(payloads.map(|s| Ok(Event::default().data(s)))).keep_alive(KeepAlive::default())
}

/// The `data:` payloads of [`chat_sse`], in order.
fn chat_sse_payloads(
    base: OutStream,
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

/// The container `message.audio` is sent in, from OpenAI's `audio.format`, as
/// `serving_chat._audio_format` reads it (`pcm16` names raw PCM), or the 400's
/// text. Stream deltas are raw PCM16, so a stream takes only that; a whole reply
/// takes the speech route's containers (`media::audio_formats`, the compressed
/// ones when libsndfile is loaded).
fn chat_audio_format(req: &ChatCompletionRequest, sample_rate: u32, compressed: bool) -> Result<&'static str, String> {
    let stream = req.stream.unwrap_or(false);
    // `str(format or default).lower()`
    let fmt = match req.audio.as_ref().and_then(|a| a.get("format")) {
        Some(v) if !adapters::truthy(v) => String::new(),
        Some(Value::String(s)) => s.to_lowercase(),
        Some(v) => v.to_string().to_lowercase(),
        None => String::new(),
    };
    let fmt = if fmt.is_empty() { if stream { "pcm16" } else { "wav" }.to_string() } else { fmt };
    let allowed: Vec<&'static str> = if stream {
        vec!["pcm16"]
    } else {
        media::audio_formats(false, sample_rate, compressed)
            .into_iter()
            .map(|f| if f == "pcm" { "pcm16" } else { f })
            .collect()
    };
    let Some(&container) = allowed.iter().find(|&&f| f == fmt) else {
        let how = if stream { "streamed" } else { "produced" };
        return Err(format!(
            "audio.format {} cannot be {how} by this server; use one of: {}",
            media::py_repr(&fmt),
            allowed.join(", ")
        ));
    };
    Ok(if container == "pcm16" { "pcm" } else { container })
}

/// Non-streaming chat: text into `content`, audio into `message.audio` (b64,
/// in `audio_format`), images into `image_url` data-URL content parts
/// (matching serving_chat).
fn build_chat_response(
    model: &str,
    request_id: &str,
    chunks: Vec<ResultChunk>,
    sample_rate: u32,
    audio_format: &str,
) -> Result<Value, String> {
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
        let (audio, _) = media::pcm16_to_container(&audio_pcm, sample_rate, audio_format)?;
        message["audio"] = json!({
            "id": rid("audio"),
            "data": media::b64(&audio),
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

    Ok(json!({
        "id": request_id,
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }))
}

// ---- POST /v1/audio/speech -----------------------------------------------

/// OpenAI's `stream_format`, the switch its SDK (and so LiveKit's TTS plugin)
/// sends instead of `stream`: `"audio"` streams the bytes, `"sse"` wraps them
/// in `speech.audio.delta` events closed by `speech.audio.done`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamFormat {
    Audio,
    Sse,
}

/// `create_speech`'s check on `stream_format`: absent and an explicit `null`
/// both mean "not sent" (`model_extra.pop(..., None)`), `"audio"` and `"sse"`
/// are the two it takes, and anything else is a 400 that quotes a string and
/// names any other type, exactly as the Python frontend's `describe_rejected`.
fn speech_stream_format(v: Option<&Value>) -> Result<Option<StreamFormat>, String> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s == "audio" => Ok(Some(StreamFormat::Audio)),
        Some(Value::String(s)) if s == "sse" => Ok(Some(StreamFormat::Sse)),
        // A string is quoted, as the Python frontend's `repr` quotes it; any
        // other type is named rather than read back. See `json_type_name`.
        Some(Value::String(s)) => Err(format!(
            "stream_format must be 'audio' or 'sse', got {}",
            media::py_repr(s)
        )),
        Some(other) => Err(format!(
            "stream_format must be 'audio' or 'sse', got {}",
            media::json_type_name(other)
        )),
    }
}

/// `create_speech`'s streaming rule: `bool(req.stream) or (stream_format is not
/// None and (req.response_format or "wav").lower() in media_io.audio_formats
/// (stream=True))`. A compressed container (the SDK's default mp3) cannot be
/// streamed here, so `stream_format` then falls back to one whole body, which
/// those clients read too.
fn speech_streams(stream: Option<bool>, stream_format: Option<StreamFormat>, response_format: &str) -> bool {
    stream.unwrap_or(false)
        || (stream_format.is_some() && media::is_streamable_audio_format(response_format))
}

/// One step of a streaming speech body: a piece of audio as `response_format`
/// carries it, or the failure that ends the stream. Lets the raw body and the
/// `stream_format="sse"` wrapper share one generator, as `_as_sse` wraps
/// `_stream_pcm` on the Python side.
enum SpeechOut {
    Audio(Vec<u8>),
    Failure { status: u16, message: String },
}

/// `serving_speech._as_sse`: one `speech.audio.delta` per piece of audio
/// (base64, in `response_format`), then `speech.audio.done`. A failure after
/// the stream opened ends it with an `error` event instead of
/// `speech.audio.done`: the 200 is already sent.
fn speech_sse_events(
    body: impl futures::Stream<Item = SpeechOut> + Send + 'static,
) -> impl futures::Stream<Item = String> + Send + 'static {
    let errored = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let events = {
        let errored = errored.clone();
        body.map(move |item| match item {
            SpeechOut::Audio(data) => {
                let audio = media::b64(&data);
                sse_frame(&SpeechDeltaEvent { type_: "speech.audio.delta", audio: &audio })
            }
            SpeechOut::Failure { status, message } => {
                errored.store(true, std::sync::atomic::Ordering::SeqCst);
                sse_frame(&SpeechErrorEvent {
                    type_: "error",
                    error: SpeechErrorBody { message: &message, type_: error_type(status), code: status },
                })
            }
        })
    };
    let tail = futures::stream::once(async move {
        let done = (!errored.load(std::sync::atomic::Ordering::SeqCst))
            .then(|| sse_frame(&SpeechDoneEvent { type_: "speech.audio.done" }));
        futures::stream::iter(done)
    })
    .flatten();
    events.chain(tail)
}

/// `openai._util.sse`: `data: <compact json>\n\n`. The events below are structs,
/// not `json!` maps, so their keys serialize in the order the Python dicts write
/// them — serde_json carries no `preserve_order`, and a map would come out
/// alphabetical.
fn sse_frame<T: serde::Serialize>(event: &T) -> String {
    format!("data: {}\n\n", serde_json::to_string(event).unwrap())
}

#[derive(serde::Serialize)]
struct SpeechDeltaEvent<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    audio: &'a str,
}

#[derive(serde::Serialize)]
struct SpeechDoneEvent {
    #[serde(rename = "type")]
    type_: &'static str,
}

#[derive(serde::Serialize)]
struct SpeechErrorEvent<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    error: SpeechErrorBody<'a>,
}

#[derive(serde::Serialize)]
struct SpeechErrorBody<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    type_: &'static str,
    code: u16,
}

async fn audio_speech(
    State(st): State<AppState>,
    payload: Result<Json<SpeechRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(j) => j,
        Err(e) => return json_400(e),
    };
    let adapter = match resolve(&st, Surface::Speech) {
        Ok(a) => a,
        Err(r) => return r,
    };
    // OpenAI's own streaming switch, checked first and never forwarded, as
    // `create_speech` pops it out of `model_extra` before anything else.
    let stream_format = match speech_stream_format(req.stream_format.as_ref()) {
        Ok(sf) => sf,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    let stream = speech_streams(req.stream, stream_format, &req.response_format);
    // `create_speech`'s `if not stream: stream_format = None` — a whole body
    // gets no SSE wrapper, whichever switch asked for one.
    let as_sse = stream && stream_format == Some(StreamFormat::Sse);
    // Before the request is translated, as in `create_speech`: a format this
    // frontend cannot produce is the client's error.
    let fmt = match media::check_audio_format(&req.response_format, stream, st.sample_rate) {
        Ok(fmt) => fmt,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    // "inf" / "nan" parse but have no JSON number; refused as the server's normalizer does
    if let Some(x) = req.speed.flatten().filter(|x| !x.is_finite()) {
        return error(400, &format!("speed must be a finite number > 0; got {x}"), "invalid_request_error");
    }
    // Blocking: a data URL decoded, a clip saved, a remote fetch of up to 30 s.
    let (dir, remote) = (st.upload_dir.clone(), st.allow_remote);
    let (req, translated) = blocking(move || {
        let speech = adapter.speech_to_request(&req, &dir, remote);
        (req, speech)
    })
    .await;
    let adapters::Speech { mut args, ref_audio } = match translated {
        Ok(s) => s,
        Err((status, e)) => return error(status, &e, error_type(status)),
    };
    // only a sent speed (an explicit null too), as Python's speech_request; an adapter-set one wins
    if let Some(speed) = req.speed {
        let value = speed.and_then(serde_json::Number::from_f64).map_or(Value::Null, Value::Number);
        args.model_kwargs.entry("speed").or_insert(value);
    }
    // Every piece reads the reference clip, and a stream submits the later ones
    // while it plays, so the clip goes when the last piece is done with it.
    let uploads = DeferredCleanup { upload_dir: st.upload_dir.clone(), args: args.clone() };
    // The model's own loader judges the clip, before any piece is submitted.
    if let (Some(check), Some(bridge)) = (&ref_audio, &st.bridge) {
        if let Err((status, e)) = bridge.check_ref_audio(check, st.request_timeout).await {
            return error(status, &e, error_type(status));
        }
    }
    let chunking = adapter.speech_chunking();
    let pieces =
        match speech_chunking::plan_chunks(req.extra.get("sentence_chunking"), &chunking, &req.input) {
            Ok(pieces) => pieces,
            Err(e) => return error(400, &e, "invalid_request_error"),
        };
    if let Err(r) = select_lora(&st, req.model.as_deref(), &mut args) {
        return r;
    }
    let n = pieces.len();
    let submit = piece_submitter(st.clone(), args, pieces, rid("speech"), stream, uploads);
    // Pieces in flight ahead of the one being read; dropping one unread aborts it.
    let mut pending: std::collections::VecDeque<OutStream> =
        (0..chunking.lookahead.clamp(1, n)).map(&*submit).collect();
    let mut submitted = pending.len();

    if stream {
        let first = match open_stream(pending.pop_front().unwrap()).await {
            Ok(first) => first,
            Err((status, message)) => return error(status, &message, error_type(status)),
        };
        // The next piece generates while this one plays.
        if submitted < n {
            pending.push_back(submit(submitted));
            submitted += 1;
        }
        // Open-ended WAV (a streaming header, then PCM16 frames), or bare PCM,
        // the pieces' audio in order, and whatever ended it.
        let playing = futures::stream::unfold(
            Some((first, pending, submitted)),
            move |state| {
                let submit = submit.clone();
                async move {
                    let (mut cur, mut pending, mut submitted) = state?;
                    loop {
                        match cur.next().await {
                            Some(Out::Chunk(c)) if c.modality == "audio" && !c.data.is_empty() => {
                                return Some((SpeechOut::Audio(c.data), Some((cur, pending, submitted))));
                            }
                            Some(Out::Chunk(_)) => {}
                            Some(Out::Error { status, message }) => {
                                return Some((SpeechOut::Failure { status, message }, None));
                            }
                            None => {
                                cur = pending.pop_front()?;
                                if submitted < n {
                                    pending.push_back(submit(submitted));
                                    submitted += 1;
                                }
                            }
                        }
                    }
                }
            },
        );
        // `_stream_pcm` yields the streaming WAV header as a piece of its own,
        // so the SSE wrapper around it carries the header as the first delta.
        let header = (fmt == "wav").then(|| SpeechOut::Audio(media::wav_stream_header(st.sample_rate, 1)));
        let body = futures::stream::iter(header).chain(playing);
        if as_sse {
            let events = speech_sse_events(body).map(|e| Ok::<Bytes, std::io::Error>(Bytes::from(e)));
            return Response::builder()
                // Starlette appends the charset to a `text/*` media type, so
                // `_ClosingStreamingResponse`'s header reads exactly this.
                .header("content-type", "text/event-stream; charset=utf-8")
                .header("cache-control", "no-cache")
                .body(Body::from_stream(events))
                .unwrap();
        }
        // A later error aborts the raw body: the status is already sent, and a
        // cleanly ended stream would pass for a complete answer.
        let raw = body.map(|item| match item {
            SpeechOut::Audio(data) => Ok(Bytes::from(data)),
            SpeechOut::Failure { message, .. } => Err(std::io::Error::other(message)),
        });
        return Response::builder()
            .header("content-type", if fmt == "wav" { "audio/wav" } else { "audio/pcm" })
            .header("cache-control", "no-cache")
            .body(Body::from_stream(raw))
            .unwrap();
    }

    let mut pcm: Vec<u8> = Vec::new();
    for _ in 0..n {
        if submitted < n {
            pending.push_back(submit(submitted));
            submitted += 1;
        }
        let chunks = match collect(pending.pop_front().unwrap()).await {
            Ok(chunks) => chunks,
            Err((status, msg)) => return error(status, &msg, error_type(status)),
        };
        for c in &chunks {
            if c.modality == "audio" {
                pcm.extend_from_slice(&c.data);
            }
        }
    }
    let sample_rate = st.sample_rate;
    let (bytes, mime) = match blocking(move || media::pcm16_to_container(&pcm, sample_rate, &fmt)).await {
        Ok(encoded) => encoded,
        Err(e) => return error(500, &e, "server_error"),
    };
    Response::builder()
        .header("content-type", mime)
        .body(Body::from(bytes))
        .unwrap()
}

/// Submits piece `i` of a speech request and returns its result stream. One
/// piece goes out under the request's id; several under `<id>-<i>`, each with
/// its own text and `chunk_kwargs`.
fn piece_submitter(
    st: AppState,
    args: SubmitArgs,
    pieces: Vec<String>,
    request_id: String,
    stream: bool,
    uploads: DeferredCleanup,
) -> Arc<dyn Fn(usize) -> OutStream + Send + Sync> {
    Arc::new(move |i| {
        let _ = &uploads; // dropped with the last submitter, which schedules the cleanup
        let piece = SubmitArgs {
            text: Some(pieces[i].clone()),
            model_kwargs: speech_chunking::chunk_kwargs(&args.model_kwargs, i),
            ..args.clone()
        };
        let id = if pieces.len() == 1 { request_id.clone() } else { format!("{request_id}-{i}") };
        result_stream(&st, &piece, &id, stream)
    })
}

// ---- POST /v1/images/generations, /v1/images/edits -----------------------

async fn images_generations(
    State(st): State<AppState>,
    payload: Result<Json<ImageGenerationRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(j) => j,
        Err(e) => return json_400(e),
    };
    let adapter = match resolve(&st, Surface::Images) {
        Ok(a) => a,
        Err(r) => return r,
    };
    // n is served below
    if let Err(e) = check_inline_result(req.response_format.as_deref(), None) {
        return error(400, &e, "invalid_request_error");
    }
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
    let mut base = base;
    if let Err(r) = select_lora(&st, req.model.as_deref(), &mut base) {
        return r;
    }
    let seed = base.model_kwargs.get("seed").and_then(Value::as_i64);
    let futs = (0..n).map(|i| {
        let mut args = base.clone();
        if let (Some(s), true) = (seed, i > 0) {
            args.model_kwargs.insert("seed".to_string(), Value::from(s.wrapping_add(i)));
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
/// `MSTAR_REQUEST_TIMEOUT_S`, the total per-request budget: a finite number of
/// seconds at least 0, where 0 times every request out, as a `--timeout` at or
/// below 0 does in Python; anything else is mstar's default 600 s.
fn request_timeout(raw: Option<&str>) -> std::time::Duration {
    std::time::Duration::from_secs_f64(
        raw.and_then(|s| s.trim().parse::<f64>().ok()).filter(|s| s.is_finite() && *s >= 0.0).unwrap_or(600.0),
    )
}

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

/// Refuse what an image or video route would otherwise drop, as
/// `_util.check_inline_result` does: results come back inline as `b64_json`,
/// and a route that runs one request returns one result.
fn check_inline_result(response_format: Option<&str>, n: Option<i64>) -> Result<(), String> {
    if let Some(fmt) = response_format.filter(|f| !f.is_empty() && *f != "b64_json") {
        return Err(format!("response_format {} is not served; results are returned as b64_json", media::py_repr(fmt)));
    }
    match n {
        None | Some(1) => Ok(()),
        Some(n) => Err(format!("n={n}: this route returns one result per request")),
    }
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
        Err(e) => return json_400(e),
    };
    let adapter = match resolve(&st, Surface::Videos) {
        Ok(a) => a,
        Err(r) => return r,
    };
    if let Err(e) = check_inline_result(req.response_format.as_deref(), req.n) {
        return error(400, &e, "invalid_request_error");
    }
    let (dir, remote) = (st.upload_dir.clone(), st.allow_remote);
    let model = req.model.clone();
    let translated = blocking(move || adapter.video_to_request(&req, &dir, remote)).await;
    let mut args = match translated {
        Ok(a) => a,
        Err(e) => return error(400, &e, "invalid_request_error"),
    };
    schedule_upload_cleanup(&st.upload_dir, &args);
    if let Err(r) = select_lora(&st, model.as_deref(), &mut args) {
        return r;
    }
    let request_id = rid("vid");
    // `stream_video`: a windowed request's mp4s as NDJSON lines as they land.
    // A refusal at submission is still its status; a later failure is in-band.
    if args.model_kwargs.get("stream_video").is_some_and(adapters::truthy) {
        return match open_stream(result_stream(&st, &args, &request_id, true)).await {
            Ok(base) => Response::builder()
                .header("content-type", "application/x-ndjson")
                .header("cache-control", "no-cache")
                .body(Body::from_stream(video_ndjson(base)))
                .unwrap(),
            Err((status, msg)) => error(status, &msg, error_type(status)),
        };
    }
    let fps = args.model_kwargs.get("fps").cloned();
    match collect(result_stream(&st, &args, &request_id, false)).await {
        // the mux shells out to ffmpeg
        Ok(chunks) => Json(blocking(move || videos_response(chunks, fps.as_ref())).await).into_response(),
        Err((status, msg)) => error(status, &msg, error_type(status)),
    }
}

/// `serving_videos._ndjson_lines`: one line per chunk in `/generate`'s shape
/// with a running `index`, closed by a `done` line carrying the count; a
/// failure ends it with its `error` line and no `done`.
fn video_ndjson(base: OutStream) -> impl futures::Stream<Item = Result<Bytes, Infallible>> {
    let state = Arc::new(std::sync::Mutex::new((0u64, false)));
    let lines = {
        let state = state.clone();
        base.map(move |item| {
            let mut st = state.lock().unwrap();
            let line = match item {
                Out::Chunk(mut c) => {
                    if !c.metadata.is_object() {
                        c.metadata = json!({});
                    }
                    c.metadata["index"] = json!(st.0);
                    st.0 += 1;
                    ndjson_line(&c)
                }
                Out::Error { status, message } => {
                    st.1 = true;
                    ndjson_line(&ResultChunk {
                        modality: "error".to_string(),
                        data: message.into_bytes(),
                        metadata: json!({"status": status}),
                    })
                }
            };
            Ok(Bytes::from(line))
        })
    };
    let done = futures::stream::once(async move {
        let (count, failed) = *state.lock().unwrap();
        (!failed).then(|| {
            let line = json!({"modality": "done", "data": "", "metadata": {"chunks": count}}).to_string() + "\n";
            Ok(Bytes::from(line))
        })
    })
    .filter_map(futures::future::ready);
    lines.chain(done)
}

async fn images_edits(State(st): State<AppState>, mut mp: Multipart) -> Response {
    let adapter = match resolve(&st, Surface::Images) {
        Ok(a) => a,
        Err(r) => return r,
    };
    // Multipart (image file + prompt + passthrough fields), parsed manually so
    // arbitrary model knobs (e.g. cfg_*_scale) flow through as model_kwargs.
    // `size` is kept: the adapter maps "WxH" onto the model's width / height
    let known = ["model"];
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut image_filename: Option<String> = None;
    let mut images = 0;
    let mut requested_model: Option<String> = None;
    let mut prompt = String::new();
    let mut n: Option<String> = None;
    let mut response_format: Option<String> = None;
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
        // One image, edited whole: a mask or a second image would be dropped.
        // `image[]` is how OpenAI clients name the image list.
        if has_file && name != "image" && name != "image[]" {
            let msg = format!("images/edits does not take a '{name}' upload; the whole image is edited");
            return error(400, &msg, "invalid_request_error");
        }
        if name == "image" || name == "image[]" {
            images += 1;
            image_filename = field.file_name().map(str::to_string);
            match field.bytes().await {
                Ok(b) => image_bytes = Some(b.to_vec()),
                Err(e) => {
                    let (status, msg) = multipart_failure(&e, e.to_string());
                    return error(status, &msg, "invalid_request_error");
                }
            }
        } else if name == "model" {
            requested_model = match field.text().await {
                Ok(v) => Some(v),
                Err(e) => {
                    let (status, msg) = multipart_failure(&e, e.to_string());
                    return error(status, &msg, "invalid_request_error");
                }
            };
        } else if matches!(name.as_str(), "prompt" | "n" | "response_format") {
            let v = match field.text().await {
                Ok(v) => v,
                Err(e) => {
                    let (status, msg) = multipart_failure(&e, e.to_string());
                    return error(status, &msg, "invalid_request_error");
                }
            };
            match name.as_str() {
                "prompt" => prompt = v,
                "n" => n = Some(v),
                _ => response_format = Some(v),
            }
        } else if known.contains(&name.as_str()) {
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

    if images > 1 {
        return error(400, &format!("images/edits takes one image, got {images}"), "invalid_request_error");
    }
    let image_bytes = match image_bytes {
        Some(b) if !b.is_empty() => b,
        _ => return error(400, "images/edits requires an 'image' file upload", "invalid_request_error"),
    };
    let n = match n.filter(|v| !v.is_empty()).map(|v| v.trim().parse::<i64>()) {
        None => None,
        Some(Ok(n)) => Some(n),
        Some(Err(_)) => return error(400, "n must be an integer", "invalid_request_error"),
    };
    if let Err(e) = check_inline_result(response_format.as_deref(), n) {
        return error(400, &e, "invalid_request_error");
    }
    // Persist the uploaded image so the data-worker's loader can read it by path.
    if let Err(e) = std::fs::create_dir_all(&st.upload_dir) {
        return error(500, &e.to_string(), "server_error");
    }
    let ext = image_filename
        .as_deref()
        .and_then(|f| PathBuf::from(f).extension().map(|e| format!(".{}", e.to_string_lossy())))
        .unwrap_or_else(|| ".png".to_string());
    let image_path = st.upload_dir.join(format!("{}{}", Uuid::new_v4().simple(), ext));
    let path = image_path.clone();
    if let Err(e) = blocking(move || std::fs::write(&path, &image_bytes)).await {
        return error(500, &e.to_string(), "server_error");
    }

    let mut args = match adapter.image_edit_to_request(&prompt, &image_path.to_string_lossy(), extra) {
        Ok(a) => a,
        Err(e) => {
            let _ = std::fs::remove_file(&image_path);
            return error(400, &e, "invalid_request_error");
        }
    };
    schedule_upload_cleanup(&st.upload_dir, &args);
    if let Err(r) = select_lora(&st, requested_model.as_deref(), &mut args) {
        return r;
    }
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

/// `serving_videos.create_videos`' body: each mp4, with the generated audio
/// chunk of the same index muxed in as an AAC track (one playable file), its
/// timestamps rescaled to the request's `fps`. A failed mux returns the video
/// alone. Blocking.
fn videos_response(chunks: Vec<ResultChunk>, fps: Option<&Value>) -> Value {
    let videos = chunks.iter().filter(|c| c.modality == "video");
    let audios: Vec<&ResultChunk> = chunks.iter().filter(|c| c.modality == "audio").collect();
    // `float(fps) if fps else None`; a value float() refuses fails the mux, so the video goes alone
    let fps: Result<Option<f64>, String> = match fps {
        Some(v) if adapters::truthy(v) => match v {
            Value::Number(n) => Ok(n.as_f64()),
            Value::String(s) => s.trim().parse().map(Some).map_err(|_| format!("could not convert {s:?} to float")),
            Value::Bool(b) => Ok(Some(f64::from(u8::from(*b)))),
            other => Err(format!("float() argument must be a string or a real number, not {other}")),
        },
        _ => Ok(None),
    };
    let int = |meta: &Value, key: &str, default: i64| {
        meta.get(key).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))).unwrap_or(default)
    };
    let data: Vec<Value> = videos
        .enumerate()
        .map(|(i, video)| {
            let mut mp4 = std::borrow::Cow::Borrowed(&video.data);
            if let Some(audio) = audios.get(i) {
                let muxed = fps.clone().and_then(|fps| {
                    let rate = int(&audio.metadata, "sample_rate", 48000);
                    let channels = int(&audio.metadata, "num_channels", 2);
                    media::mux_mp4_with_pcm16(&video.data, &audio.data, rate, channels, fps)
                });
                match muxed {
                    Ok(with_sound) => mp4 = std::borrow::Cow::Owned(with_sound),
                    Err(e) => eprintln!("Muxing generated audio into the mp4 failed; returning video only: {e}"),
                }
            }
            json!({"b64_json": media::b64(&mp4), "url": Value::Null})
        })
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
    let mut parts: Vec<Part> = Vec::new();

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
    // A substring match, as Python's: the vendor type is in no other range.
    let binary = req
        .headers()
        .get(axum::http::header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains(BINARY_STREAM_MEDIA_TYPE));

    // Every field is a form field, as in Python: another body (JSON, say) would
    // parse as an empty form and be submitted as an empty request.
    let base_type = content_type.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    if !base_type.is_empty()
        && base_type != "multipart/form-data"
        && base_type != "application/x-www-form-urlencoded"
    {
        return detail_error(
            400,
            &format!("/generate takes multipart/form-data or urlencoded fields, not {content_type}"),
        );
    }
    // Starlette reads a body with no content type as an empty form.
    if base_type != "multipart/form-data" {
        // No file uploads on this path (files force multipart), matching how
        // these clients already behave.
        // Bounded here: `DefaultBodyLimit` binds only the extractors, and this
        // path reads the body itself.
        let body = match axum::body::to_bytes(req.into_body(), body_limit_mb() << 20).await {
            Ok(b) => b,
            Err(e) if is_length_limit(&e) => return detail_error(413, &body_too_large()),
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
            request_id, uploads, Vec::new(), binary,
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
            let path = save_path.clone();
            if let Err(e) = blocking(move || std::fs::write(&path, &bytes)).await {
                return detail_error(500, &e.to_string());
            }
            let paths = uploads.paths.entry(modality.to_string()).or_default();
            parts.push(Part { modality: modality.to_string(), text: None, index: paths.len(), role: None });
            paths.push(save_path.to_string_lossy().into_owned());
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
        request_id, uploads, parts, binary,
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
    mut parts: Vec<Part>,
    binary: bool,
) -> Response {
    // An empty field means none, as the Python /generate's form parsing reads
    // its optional fields.
    let request_id = request_id.filter(|id| !id.is_empty());
    text = text.filter(|t| !t.is_empty());
    let in_mods_raw = in_mods_raw.filter(|m| !m.is_empty());
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
    // The prompt in request order: each file, then the text, as Python's
    // /generate lays it out.
    let has_text = text.is_some();
    if has_text {
        parts.push(Part { modality: "text".to_string(), text: text.clone(), index: 0, role: None });
    }
    let input_modalities: Vec<String> = match in_mods_raw {
        // An explicit list is then the layout on its own, so the derived
        // parts go rather than disagree with it.
        Some(raw) => {
            parts.clear();
            let mut m: Vec<String> = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            // A layout without a text slot would drop the prompt, and a text
            // slot with no text renders to nothing.
            if has_text && !m.iter().any(|x| x == "text") {
                m.push("text".to_string());
            }
            if !has_text {
                m.retain(|x| x != "text");
            }
            m
        }
        None => parts.iter().map(|p| p.modality.clone()).collect(),
    };
    if let Some(message) = video_frame_refusal(&input_modalities, &output_modalities, streaming) {
        return detail_error(400, message);
    }
    let model_kwargs: Map<String, Value> = match mk_raw {
        Some(raw) if !raw.is_empty() => match serde_json::from_str::<Value>(&raw) {
            Ok(Value::Object(m)) => m,
            // `null` is no kwargs, as `json.loads` gives None
            Ok(Value::Null) => Map::new(),
            Ok(_) => return detail_error(400, "model_kwargs must be a JSON object"),
            Err(_) => return detail_error(400, "model_kwargs must be valid JSON"),
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
        parts.clear();
    }

    let args = SubmitArgs {
        text,
        tokens,
        file_paths: uploads.keep(),
        input_modalities,
        output_modalities,
        model_kwargs,
        parts,
        ..SubmitArgs::default()
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
        let base = match open_stream(base).await {
            Ok(base) => base,
            Err((status, message)) => return detail_error(status, &message),
        };
        let body = base.flat_map(move |item| {
            let chunk = match item {
                Out::Chunk(c) => c,
                // After the 200 the status travels in-band, as the error chunk
                // Python's `errors_in_band` ends the stream with.
                Out::Error { status, message } => ResultChunk {
                    modality: "error".to_string(),
                    data: message.into_bytes(),
                    metadata: json!({"status": status}),
                },
            };
            let pieces = if binary {
                let (header, payload) = binary_frame(chunk);
                vec![header, payload]
            } else {
                vec![Bytes::from(ndjson_line(&chunk))]
            };
            futures::stream::iter(pieces.into_iter().map(Ok::<Bytes, Infallible>))
        });
        let media_type = if binary { BINARY_STREAM_MEDIA_TYPE } else { "application/x-ndjson" };
        return Response::builder()
            .header("content-type", media_type)
            .header("cache-control", "no-cache")
            .header("vary", "Accept")
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

/// `Accept`ed on `/generate` for length-framed chunks (Python's
/// `BINARY_STREAM_MEDIA_TYPE`).
const BINARY_STREAM_MEDIA_TYPE: &str = "application/vnd.mstar.frames";

/// One chunk as Python's `_chunk_to_binary_frame` writes it: a JSON header
/// line naming the payload's length, then the payload as is, which spares a
/// video frame the base64 and JSON passes of the NDJSON form.
fn binary_frame(c: ResultChunk) -> (Bytes, Bytes) {
    let mut header = json!({
        "modality": c.modality,
        "nbytes": c.data.len(),
        "metadata": c.metadata,
    })
    .to_string()
    .into_bytes();
    header.push(b'\n');
    (Bytes::from(header), Bytes::from(c.data))
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
    #[tokio::test(start_paused = true)]
    async fn a_speech_clip_is_removed_once_the_last_piece_is_done_with_it() {
        let dir = std::env::temp_dir().join(format!("mstar_cleanup_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("ref.wav");
        std::fs::write(&clip, b"RIFF").unwrap();
        let mut args = super::SubmitArgs::default();
        args.file_paths.insert("audio".into(), vec![clip.to_string_lossy().into_owned()]);
        let uploads = super::DeferredCleanup { upload_dir: dir.clone(), args };
        let held = std::sync::Arc::new(uploads); // as the speech's piece submitter holds it
        let second = held.clone();
        drop(held);
        tokio::time::sleep(std::time::Duration::from_secs(120)).await;
        assert!(clip.exists(), "removed while a piece still held it");
        drop(second);
        tokio::time::sleep(std::time::Duration::from_secs(59)).await;
        assert!(clip.exists(), "removed before its 60 s grace");
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        tokio::task::yield_now().await;
        assert!(!clip.exists(), "never removed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_root_path_is_stripped_as_starlette_strips_it() {
        use super::route_path;
        // unset: every path as sent
        assert_eq!(route_path("", "/v1/models"), "/v1/models");
        // a path under the prefix loses it; the bare path still routes
        assert_eq!(route_path("/proj/job", "/proj/job/v1/models"), "/v1/models");
        // the prefix as set: a trailing slash leaves the path whole, as Starlette does
        assert_eq!(route_path("/proj/job/", "/proj/job/v1/models"), "/proj/job/v1/models");
        assert_eq!(route_path("/proj/job", "/v1/models"), "/v1/models");
        // a sibling that only shares the prefix's characters is not under it
        assert_eq!(route_path("/proj/job", "/proj/jobs/v1/models"), "/proj/jobs/v1/models");
        assert_eq!(route_path("/proj/job", "/proj/job"), "/");
    }

    #[test]
    fn the_rewrite_keeps_the_query() {
        let req = axum::http::Request::builder()
            .uri("/p/generate?tokenize=true")
            .body(axum::body::Body::empty())
            .unwrap();
        let req = super::strip_root_path("/p", req);
        assert_eq!(req.uri().path_and_query().unwrap().as_str(), "/generate?tokenize=true");
    }

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
            served: Arc::new(served_models::ServedModels::from_env("m")),
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

    #[tokio::test]
    async fn voices_are_503_when_the_bridge_does_not_answer() {
        let dir = socket_dir();
        let bridge = Arc::new(Bridge::new(dir.to_str().unwrap()).unwrap());
        let st = AppState { python_adapter: "OrpheusAdapter".into(), ..state(Some(bridge)) };

        let (status, _, _) = body(audio_voices(State(st)).await).await;

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
            false, None, None, uploads, Vec::new(), false,
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
            let st = state(None);
            let base = result_stream(&st, &args, "r", true);
            let (status, _, data) = body(chat_sse(base, "r".into(), st.model_name.clone()).into_response()).await;
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
            ("model_kwargs", "{", "model_kwargs must be valid JSON"),
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
    async fn an_edit_refuses_what_it_would_drop() {
        let mut st = state(None);
        st.upload_dir = socket_dir();
        st.python_adapter = "BagelAdapter".into();
        for (fields, detail) in [
            (vec![("image", Some("a.png")), ("mask", Some("m.png"))], "images/edits does not take a 'mask' upload; the whole image is edited"),
            (vec![("image", Some("a.png")), ("image", Some("b.png"))], "images/edits takes one image, got 2"),
            (vec![("image[]", Some("a.png")), ("n", None)], "n=2: this route returns one result per request"),
            (vec![("image", Some("a.png")), ("response_format", None)], "response_format 'url' is not served; results are returned as b64_json"),
        ] {
            let parts: Vec<(&str, Option<&str>, &[u8])> = fields
                .iter()
                .map(|&(name, file)| {
                    let value: &[u8] = match name {
                        "n" => b"2",
                        "response_format" => b"url",
                        _ => b"\x89PNG",
                    };
                    (name, file, value)
                })
                .collect();
            let mp = Multipart::from_request(multipart_request("/v1/images/edits", &parts), &st).await.unwrap();

            let (status, _, data) = body(images_edits(State(st.clone()), mp).await).await;

            let body: Value = serde_json::from_slice(&data).unwrap();
            assert_eq!((status, body["error"]["message"].as_str()), (StatusCode::BAD_REQUEST, Some(detail)));
            assert_eq!(std::fs::read_dir(&st.upload_dir).unwrap().count(), 0, "{detail}");
        }
        assert_eq!(check_inline_result(Some("b64_json"), Some(1)), Ok(()));
        assert_eq!(
            check_inline_result(Some("it's"), None),
            Err("response_format \"it's\" is not served; results are returned as b64_json".into())
        );
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
                "chat_template_kwargs must be an object",
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
    fn chat_audio_is_sent_in_the_format_serving_chat_sends() {
        let check = |fields: Value| {
            let mut req = json!({"messages": [{"role": "user", "content": "hi"}]});
            req.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
            chat_audio_format(&serde_json::from_value(req).unwrap(), 24000, false)
        };
        for (fields, want) in [
            (json!({"audio": {"voice": "Ethan"}}), "wav"),
            (json!({"audio": {"format": ""}}), "wav"),
            (json!({"audio": {"format": "WAV"}}), "wav"),
            // raw PCM whole, as the Python frontend sends it
            (json!({"audio": {"format": "pcm16"}}), "pcm"),
            (json!({"stream": true}), "pcm"),
            (json!({"audio": {"format": "pcm16"}, "stream": true}), "pcm"),
        ] {
            assert_eq!(check(fields.clone()), Ok(want), "{fields}");
        }
        for (fields, err) in [
            (json!({"audio": {"format": "mp3"}}), "audio.format 'mp3' cannot be produced by this server; use one of: wav, pcm16"),
            (json!({"audio": {"format": "pcm"}}), "audio.format 'pcm' cannot be produced by this server; use one of: wav, pcm16"),
            (json!({"audio": {"format": 5}}), "audio.format '5' cannot be produced by this server; use one of: wav, pcm16"),
            (json!({"audio": {"format": "wav"}, "stream": true}), "audio.format 'wav' cannot be streamed by this server; use one of: pcm16"),
            (json!({"audio": {"format": "it's"}}), "audio.format \"it's\" cannot be produced by this server; use one of: wav, pcm16"),
        ] {
            assert_eq!(check(fields.clone()), Err(err.to_string()), "{fields}");
        }
        // with libsndfile loaded, a whole reply takes the speech route's compressed containers
        let compressed = |format: &str| {
            let req = json!({"messages": [{"role": "user", "content": "hi"}], "audio": {"format": format}});
            chat_audio_format(&serde_json::from_value(req).unwrap(), 24000, true)
        };
        assert_eq!((compressed("MP3"), compressed("opus")), (Ok("mp3"), Ok("opus")));
        let pcm = vec![1u8, 0, 2, 0];
        let chunks = vec![ResultChunk { modality: "audio".into(), data: pcm.clone(), metadata: Value::Null }];
        let resp = build_chat_response("m", "r", chunks, 24000, "pcm").unwrap();
        assert_eq!(resp["choices"][0]["message"]["audio"]["data"], media::b64(&pcm));
    }

    #[test]
    fn a_timeout_of_0_times_every_request_out_and_an_unreadable_one_is_the_default() {
        let secs = |raw| request_timeout(raw).as_secs_f64();
        assert_eq!((secs(Some("0")), secs(Some(" 1800.5 ")), secs(Some("1e9"))), (0.0, 1800.5, 1e9));
        for raw in [None, Some("-1"), Some("inf"), Some("nan"), Some("abc")] {
            assert_eq!(secs(raw), 600.0, "{raw:?}");
        }
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
        let resp = build_chat_response("m", "r", split_text_chunks(), 24000, "wav").unwrap();
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

    // ---- stream_format (OpenAI's switch on /v1/audio/speech) --------------

    /// `create_speech`: `stream_format not in (None, "audio", "sse")`. Absent
    /// and an explicit `null` are both "not sent".
    #[test]
    fn stream_format_takes_audio_sse_and_nothing_sent() {
        let sf = |body: &str| {
            speech_stream_format(
                serde_json::from_str::<SpeechRequest>(body).unwrap().stream_format.as_ref(),
            )
        };
        assert_eq!(sf(r#"{"input":"hi"}"#), Ok(None));
        assert_eq!(sf(r#"{"input":"hi","stream_format":null}"#), Ok(None));
        assert_eq!(sf(r#"{"input":"hi","stream_format":"audio"}"#), Ok(Some(StreamFormat::Audio)));
        assert_eq!(sf(r#"{"input":"hi","stream_format":"sse"}"#), Ok(Some(StreamFormat::Sse)));
    }

    /// The refusal quotes a string as Python's `repr` does and names any other
    /// JSON type, which is what the Python frontend's `describe_rejected` says.
    #[test]
    fn stream_format_refuses_anything_else_quoting_a_string_and_naming_a_type() {
        let msg = |body: &str| {
            speech_stream_format(
                serde_json::from_str::<SpeechRequest>(body).unwrap().stream_format.as_ref(),
            )
            .unwrap_err()
        };
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":"SSE"}"#),
            "stream_format must be 'audio' or 'sse', got 'SSE'"
        );
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":""}"#),
            "stream_format must be 'audio' or 'sse', got ''"
        );
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":true}"#),
            "stream_format must be 'audio' or 'sse', got a boolean"
        );
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":false}"#),
            "stream_format must be 'audio' or 'sse', got a boolean"
        );
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":1}"#),
            "stream_format must be 'audio' or 'sse', got a number"
        );
        // -0 and an integer past u64 both arrive as f64; the old code had to
        // reproduce Python's repr of each, and now neither is read back.
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":-0.0}"#),
            "stream_format must be 'audio' or 'sse', got a number"
        );
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":18446744073709551616}"#),
            "stream_format must be 'audio' or 'sse', got a number"
        );
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":1.5}"#),
            "stream_format must be 'audio' or 'sse', got a number"
        );
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":["sse"]}"#),
            "stream_format must be 'audio' or 'sse', got an array"
        );
        assert_eq!(
            msg(r#"{"input":"hi","stream_format":{"a":1}}"#),
            "stream_format must be 'audio' or 'sse', got an object"
        );
    }

    /// `bool(req.stream) or (stream_format is not None and the format is one of
    /// `media_io.audio_formats(stream=True)`)`: the switch forces a stream only
    /// for a format that can actually be streamed.
    #[test]
    fn stream_format_streams_only_a_streamable_format() {
        use StreamFormat::{Audio, Sse};
        // stream_format alone, for the formats a stream carries (wav when unset).
        for sf in [Audio, Sse] {
            for fmt in ["wav", "WAV", "pcm", ""] {
                assert!(speech_streams(None, Some(sf), fmt), "{sf:?} {fmt:?}");
            }
            // a compressed container falls back to one whole body
            for fmt in ["mp3", "flac", "opus", "aac"] {
                assert!(!speech_streams(None, Some(sf), fmt), "{sf:?} {fmt:?}");
                assert!(!speech_streams(Some(false), Some(sf), fmt));
                // `stream` still wins on its own, as it does on the Python side
                assert!(speech_streams(Some(true), Some(sf), fmt));
            }
        }
        // no switch at all: `stream` decides, absent and null meaning false
        assert!(!speech_streams(None, None, "wav"));
        assert!(!speech_streams(Some(false), None, "pcm"));
        assert!(speech_streams(Some(true), None, "wav"));
        assert!(speech_streams(Some(true), None, "mp3"));
    }

    fn sse_events(items: Vec<SpeechOut>) -> Vec<String> {
        futures::executor::block_on(
            speech_sse_events(futures::stream::iter(items)).collect::<Vec<String>>(),
        )
    }

    /// `_as_sse`: a delta per piece of audio, then `speech.audio.done`, framed
    /// and keyed exactly as `openai._util.sse` writes them.
    #[test]
    fn sse_wraps_each_piece_of_audio_and_closes_with_done() {
        let events = sse_events(vec![SpeechOut::Audio(b"\x01\x02".to_vec()), SpeechOut::Audio(b"ab".to_vec())]);
        assert_eq!(
            events,
            vec![
                "data: {\"type\":\"speech.audio.delta\",\"audio\":\"AQI=\"}\n\n".to_string(),
                "data: {\"type\":\"speech.audio.delta\",\"audio\":\"YWI=\"}\n\n".to_string(),
                "data: {\"type\":\"speech.audio.done\"}\n\n".to_string(),
            ]
        );
    }

    /// A failure after the 200: the in-band `error` event, and no
    /// `speech.audio.done` after it — a failed stream must not read as complete.
    #[test]
    fn sse_ends_a_failed_stream_with_an_error_event_and_no_done() {
        let events = sse_events(vec![
            SpeechOut::Audio(b"ab".to_vec()),
            SpeechOut::Failure { status: 400, message: "input refused by the model".into() },
        ]);
        assert_eq!(events.len(), 2, "{events:?}");
        assert_eq!(
            events[1],
            concat!(
                r#"data: {"type":"error","error":{"message":"input refused by the model","#,
                r#""type":"invalid_request_error","code":400}}"#,
                "\n\n",
            )
        );
        // a 5xx is ours, as `_util.error_type` has it
        let events = sse_events(vec![SpeechOut::Failure { status: 500, message: "boom".into() }]);
        assert_eq!(
            events,
            vec![concat!(
                r#"data: {"type":"error","error":{"message":"boom","type":"server_error","code":500}}"#,
                "\n\n",
            )
            .to_string()]
        );
    }

    /// An empty stream still closes: the Python generator falls through to
    /// `speech.audio.done`.
    #[test]
    fn sse_closes_an_empty_stream_with_done() {
        assert_eq!(sse_events(vec![]), vec!["data: {\"type\":\"speech.audio.done\"}\n\n".to_string()]);
    }
}
