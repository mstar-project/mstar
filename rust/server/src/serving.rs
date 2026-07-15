//! Shared serving state + the result-chunk source.
//!
//! `result_stream` is the single source both output surfaces draw from (as in
//! mstar's `iter_result_chunks`): the `/v1/*` handlers translate the chunks
//! into SSE / WAV / JSON, `/generate` into NDJSON. Two sources, selected at
//! startup by whether a conductor socket dir was given:
//!  * **real** — submit the flattened request to the Python conductor over the
//!    bridge, stream `ResultChunk`s back;
//!  * **mock** — a self-contained generator (echoes text, emits a short silent
//!    PCM for audio, a 1x1 PNG for image) that proves the HTTP/adapter/media/
//!    streaming layer standalone with no conductor.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::stream::Stream;
use uuid::Uuid;

use crate::adapters::SubmitArgs;
use crate::bridge::{Bridge, ResultChunk, StreamItem};
use crate::tokenizer::{IncrementalDecoder, Tok};

#[derive(Clone)]
pub struct AppState {
    pub model_name: String,
    pub upload_dir: PathBuf,
    /// Fetching arbitrary http(s) media is SSRF surface; off by default.
    pub allow_remote: bool,
    /// Output audio sample rate (mstar reads it from the model; the frontend
    /// has no model, so it's configured, default 24 kHz).
    pub sample_rate: u32,
    /// Total per-request budget (mstar's `timeout_seconds`, default 600 s):
    /// exceeded => a terminal error + an abort to the conductor.
    pub request_timeout: Duration,
    /// Loaded when `MSTAR_TOKENIZER` points at a tokenizer.json — enables the
    /// frontend-tokenizes fast path (tokenize + incremental detok in Rust, off
    /// the GIL) for flagged adapters and `/generate?tokenize=true`.
    pub tok: Option<Arc<Tok>>,
    /// Present when a conductor socket dir is given (real streaming); absent =
    /// the self-contained mock generator.
    pub bridge: Option<Arc<Bridge>>,
}

/// One item of a request's result stream.
pub enum Out {
    Chunk(ResultChunk),
    /// Terminal backend failure (worker exception, bad input, timeout).
    Error(String),
}

pub type OutStream = Pin<Box<dyn Stream<Item = Out> + Send>>;

/// Aborts the request on drop unless it completed — covers both the client
/// disconnecting mid-stream (axum drops the body stream, dropping this) and
/// the timeout path (mstar's `abort_request` releases engine state).
struct AbortGuard {
    bridge: Arc<Bridge>,
    rid: String,
    done: bool,
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        if !self.done {
            self.bridge.abort(&self.rid);
        }
    }
}

/// The unified result source: bridge to the conductor, or the mock generator.
/// On the frontend-tokenizes path (`args.tokens` set, tokenizer loaded), raw
/// token-id chunks (modality "token", 4-byte LE) coming back are detokenized
/// HERE, incrementally, so every handler downstream still sees text chunks.
pub fn result_stream(
    state: &AppState,
    args: &SubmitArgs,
    request_id: &str,
    streaming: bool,
) -> OutStream {
    let base = raw_result_stream(state, args, request_id, streaming);
    match (&args.tokens, &state.tok) {
        (Some(_), Some(tok)) => {
            use futures::StreamExt;
            let detok = IncrementalDecoder::new(tok.clone());
            Box::pin(
                base.scan(detok, |detok, item| {
                    let mapped = match item {
                        Out::Chunk(c) if c.modality == "token" && c.data.len() == 4 => {
                            let id = u32::from_le_bytes([c.data[0], c.data[1], c.data[2], c.data[3]]);
                            detok.step(id).map(|piece| {
                                Out::Chunk(ResultChunk {
                                    modality: "text".to_string(),
                                    data: piece.into_bytes(),
                                    metadata: c.metadata,
                                })
                            })
                        }
                        other => Some(other),
                    };
                    futures::future::ready(Some(mapped))
                })
                .filter_map(futures::future::ready),
            )
        }
        _ => base,
    }
}

fn raw_result_stream(
    state: &AppState,
    args: &SubmitArgs,
    request_id: &str,
    streaming: bool,
) -> OutStream {
    match &state.bridge {
        Some(b) => {
            let rx = b.submit(args, request_id, streaming);
            let guard = AbortGuard {
                bridge: b.clone(),
                rid: request_id.to_string(),
                done: false,
            };
            let deadline = Instant::now() + state.request_timeout;
            // finished: a terminal item was yielded — end on the next poll.
            let st = (rx, guard, deadline, false);
            Box::pin(futures::stream::unfold(
                st,
                |(mut rx, mut guard, deadline, finished)| async move {
                    if finished {
                        return None;
                    }
                    let left = deadline.saturating_duration_since(Instant::now());
                    match tokio::time::timeout(left, rx.recv()).await {
                        Ok(Some(StreamItem::Chunk(c))) => {
                            Some((Out::Chunk(c), (rx, guard, deadline, false)))
                        }
                        Ok(Some(StreamItem::Error(e))) => {
                            guard.done = true; // conductor already cleaned up
                            Some((Out::Error(e), (rx, guard, deadline, true)))
                        }
                        Ok(Some(StreamItem::Done)) => {
                            guard.done = true;
                            drop(guard); // completed: Drop sees done and skips the abort
                            None
                        }
                        Ok(None) => None, // sender dropped; guard aborts
                        Err(_elapsed) => {
                            // Total budget exceeded — mstar raises 500
                            // "Request timed out" and aborts. The guard (still
                            // not done) sends the abort when the state drops.
                            Some((
                                Out::Error("Request timed out".to_string()),
                                (rx, guard, deadline, true),
                            ))
                        }
                    }
                },
            ))
        }
        None => Box::pin(futures::stream::iter(
            mock_chunks(args, state.sample_rate).into_iter().map(Out::Chunk),
        )),
    }
}

/// Gather a non-streaming request's chunks; the first error wins (mstar's
/// `collect_results` raising into the handler).
pub async fn collect(stream: OutStream) -> Result<Vec<ResultChunk>, String> {
    use futures::StreamExt;
    let mut chunks = Vec::new();
    let mut s = stream;
    while let Some(item) = s.next().await {
        match item {
            Out::Chunk(c) => chunks.push(c),
            Out::Error(e) => return Err(e),
        }
    }
    Ok(chunks)
}

/// Mock output: one chunk per requested output modality.
fn mock_chunks(args: &SubmitArgs, sample_rate: u32) -> Vec<ResultChunk> {
    let mut out = Vec::new();
    let prompt = args.text.clone().unwrap_or_default();
    for modality in &args.output_modalities {
        match modality.as_str() {
            "text" => out.push(ResultChunk {
                modality: "text".to_string(),
                data: format!("[mock] {prompt}").into_bytes(),
                metadata: serde_json::Value::Null,
            }),
            "audio" => out.push(ResultChunk {
                modality: "audio".to_string(),
                // 0.1 s of silence as 16-bit PCM at the model rate.
                data: vec![0u8; (sample_rate as usize / 10) * 2],
                metadata: serde_json::Value::Null,
            }),
            "image" => out.push(ResultChunk {
                modality: "image".to_string(),
                data: ONE_PX_PNG.to_vec(),
                metadata: serde_json::Value::Null,
            }),
            _ => {}
        }
    }
    out
}

/// A 1x1 opaque PNG, for the mock image path.
const ONE_PX_PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
    0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
];

/// Unix timestamp (seconds), for OpenAI `created` fields.
pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// `<prefix>-<uuid4-hex>`, matching mstar's request-id shape.
pub fn rid(prefix: &str) -> String {
    format!("{}-{}", prefix, Uuid::new_v4().simple())
}
