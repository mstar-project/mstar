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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::stream::Stream;
use uuid::Uuid;

use crate::adapters::SubmitArgs;
use crate::bridge::{Bridge, ResultChunk, StreamItem};
use crate::refusals::{mark_submitted, Refusals};
use crate::tokenizer::{IncrementalDecoder, Tok};

#[derive(Clone)]
pub struct AppState {
    pub model_name: String,
    /// The model's adapter class in Python's `ADAPTER_REGISTRY`
    /// (`MSTAR_OPENAI_ADAPTER`, set by the launcher); empty when it has none.
    pub python_adapter: String,
    pub upload_dir: PathBuf,
    /// Fetching arbitrary http(s) media is SSRF surface; off by default.
    pub allow_remote: bool,
    /// Output audio sample rate (mstar reads it from the model; the frontend
    /// has no model, so it's configured, default 24 kHz).
    pub sample_rate: u32,
    /// Total per-request budget (mstar's `timeout_seconds`, default 600 s):
    /// exceeded => a terminal error + an abort to the conductor.
    pub request_timeout: Duration,
    /// Largest `n` one `/v1/images/generations` request may ask for
    /// (`MSTAR_MAX_IMAGES_PER_REQUEST`, default 10).
    pub max_images_per_request: i64,
    /// Loaded when `MSTAR_TOKENIZER` points at a tokenizer.json — enables the
    /// frontend-tokenizes fast path (tokenize + incremental detok in Rust, off
    /// the GIL) for flagged adapters and `/generate?tokenize=true`.
    pub tok: Option<Arc<Tok>>,
    /// Present when a conductor socket dir is given (real streaming); absent =
    /// the self-contained mock generator.
    pub bridge: Option<Arc<Bridge>>,
    /// Requests refused before submission; sent with each `/metrics` request.
    pub refusals: Arc<Refusals>,
}

/// One item of a request's result stream.
pub enum Out {
    Chunk(ResultChunk),
    /// Terminal backend failure (worker exception, bad input, timeout).
    /// `status` is the HTTP status to return (carried in-band from the error
    /// chunk's `metadata.status`; defaults to 500).
    Error { status: u16, message: String },
}

pub type OutStream = Pin<Box<dyn Stream<Item = Out> + Send>>;

/// Aborts the request on drop unless it completed — covers both the client
/// disconnecting mid-stream (axum drops the body stream, dropping this) and
/// the timeout path (mstar's `abort_request` releases engine state).
struct AbortGuard {
    bridge: Arc<Bridge>,
    rid: String,
    sub: u64,
    done: bool,
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        if !self.done {
            self.bridge.abort(&self.rid, self.sub);
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
    try_result_stream(state, args, request_id, streaming).unwrap_or_else(|(status, message)| {
        Box::pin(futures::stream::once(async move { Out::Error { status, message } }))
    })
}

/// How long `/generate` waits for a submission still in flight under a client's
/// `request_id` to end before refusing a new one (mstar's `DUPLICATE_ID_GRACE_S`).
pub const DUPLICATE_ID_GRACE: Duration = Duration::from_millis(500);

/// [`result_stream`], with a submission the bridge refused (its `request_id`
/// still in flight) returned as `Err((409, detail))` before any stream exists,
/// so a streaming handler can still answer with that status.
pub fn try_result_stream(
    state: &AppState,
    args: &SubmitArgs,
    request_id: &str,
    streaming: bool,
) -> Result<OutStream, (u16, String)> {
    let base = raw_result_stream(state, args, request_id, streaming)?;
    Ok(match (&args.tokens, &state.tok) {
        (Some(_), Some(tok)) => detokenize(base, tok.clone()),
        _ => base,
    })
}

/// Token-id chunks -> text chunks through one [`IncrementalDecoder`]; at the
/// end of the stream, whatever it still holds goes out as a last text chunk.
fn detokenize(base: OutStream, tok: Arc<Tok>) -> OutStream {
    use futures::StreamExt;
    // (decoder, failed): the tail flushes what the decoder still holds,
    // unless the stream already ended on an error.
    let detok = Arc::new(Mutex::new((IncrementalDecoder::new(tok), false)));
    let tail_detok = detok.clone();
    let body = base.filter_map(move |item| {
        let mut st = detok.lock().unwrap();
        let mapped = match item {
            Out::Chunk(c) if c.modality == "token" && c.data.len() == 4 => {
                let id = u32::from_le_bytes([c.data[0], c.data[1], c.data[2], c.data[3]]);
                st.0.step(id).map(|piece| {
                    Out::Chunk(ResultChunk {
                        modality: "text".to_string(),
                        data: piece.into_bytes(),
                        metadata: c.metadata,
                    })
                })
            }
            // A "token" chunk that isn't a 4-byte LE id is malformed
            // — drop it rather than forwarding an unknown modality.
            Out::Chunk(c) if c.modality == "token" => {
                eprintln!(
                    "mstar-server: dropping malformed token chunk ({} bytes, expected 4)",
                    c.data.len()
                );
                None
            }
            err @ Out::Error { .. } => {
                st.1 = true;
                Some(err)
            }
            other => Some(other),
        };
        futures::future::ready(mapped)
    });
    let tail = futures::stream::once(async move {
        let mut st = tail_detok.lock().unwrap();
        if st.1 {
            return None;
        }
        st.0.finish().map(|rest| {
            Out::Chunk(ResultChunk {
                modality: "text".to_string(),
                data: rest.into_bytes(),
                metadata: serde_json::Value::Null,
            })
        })
    })
    .filter_map(futures::future::ready);
    Box::pin(body.chain(tail))
}

fn raw_result_stream(
    state: &AppState,
    args: &SubmitArgs,
    request_id: &str,
    streaming: bool,
) -> Result<OutStream, (u16, String)> {
    let stream: OutStream = match &state.bridge {
        Some(b) => {
            let (sub, rx) = b
                .submit(args, request_id, streaming)
                .map_err(|message| (409, message))?;
            let guard = AbortGuard {
                bridge: b.clone(),
                rid: request_id.to_string(),
                sub,
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
                        Ok(Some(StreamItem::Error { status, message })) => {
                            guard.done = true; // conductor already cleaned up
                            Some((
                                Out::Error { status, message },
                                (rx, guard, deadline, true),
                            ))
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
                                Out::Error {
                                    status: 500,
                                    message: "Request timed out".to_string(),
                                },
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
    };
    mark_submitted();
    Ok(stream)
}

/// Gather a non-streaming request's chunks; the first error wins (mstar's
/// `collect_results` raising into the handler). The error carries the HTTP
/// status to return `(status, message)`.
pub async fn collect(stream: OutStream) -> Result<Vec<ResultChunk>, (u16, String)> {
    use futures::StreamExt;
    let mut chunks = Vec::new();
    let mut s = stream;
    while let Some(item) = s.next().await {
        match item {
            Out::Chunk(c) => chunks.push(c),
            Out::Error { status, message } => return Err((status, message)),
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn token(id: u32) -> Out {
        Out::Chunk(ResultChunk {
            modality: "token".to_string(),
            data: id.to_le_bytes().to_vec(),
            metadata: serde_json::Value::Null,
        })
    }

    fn texts(out: Vec<Out>) -> Vec<String> {
        out.into_iter()
            .map(|o| match o {
                Out::Chunk(c) => String::from_utf8(c.data).unwrap(),
                Out::Error { message, .. } => format!("error: {message}"),
            })
            .collect()
    }

    #[test]
    fn detokenize_keeps_split_characters_and_flushes_the_tail() {
        let tok = crate::tokenizer::byte_tok();
        let mut ids = tok.encode("中文 😀");
        ids.pop(); // the stream ends inside the emoji
        let base: OutStream = Box::pin(futures::stream::iter(ids.into_iter().map(token).collect::<Vec<_>>()));
        let out = futures::executor::block_on(detokenize(base, tok).collect::<Vec<_>>());
        assert_eq!(texts(out), vec!["中", "文", " ", "\u{FFFD}"]);
    }

    #[test]
    fn detokenize_adds_nothing_after_an_error() {
        let tok = crate::tokenizer::byte_tok();
        let ids = tok.encode("中");
        let items = vec![
            token(ids[0]),
            Out::Error { status: 500, message: "boom".to_string() },
        ];
        let base: OutStream = Box::pin(futures::stream::iter(items));
        let out = futures::executor::block_on(detokenize(base, tok).collect::<Vec<_>>());
        assert_eq!(texts(out), vec!["error: boom"]);
    }
}
