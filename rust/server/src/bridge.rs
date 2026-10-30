//! Bridge from the axum frontend to the Python bridge loop over the
//! Rust ZMQ transport (the `mstar_rust` library crate).
//!
//! Mirrors mstar's `APIServer.submit_request` / result-collection contract:
//! the frontend translates an OpenAI request into a submit (text + media file
//! paths + in/out modalities + model_kwargs, and a chat's ordered parts) and
//! receives a stream of
//! multimodal `ResultChunk`s ({modality, data, metadata}) back — which the
//! serving handlers translate into SSE / WAV / NDJSON. Media preprocessing and
//! the model run in the Python data plane; no Python touches the request/HTTP
//! hot path.
//!
//! Wire format is msgpack (the conductor speaks msgpack), sent as the
//! `ZmqCommunicator`'s opaque byte payload:
//!   frontend -> conductor: SubmitMsg {t:"submit", rid, sub, text, file_paths,
//!       input_modalities, output_modalities, model_kwargs, prompt_parts,
//!       streaming, accepts_chunks}, `prompt_parts` present for a chat only
//!                        | {t:"abort", rid, sub}
//!                        | {t:"ping", rid, sub}
//!                        | {t:"metrics", rid, sub, refusals:{route, status, count}}
//!   conductor -> frontend: {t:"chunk", rid, sub, modality, data(bin), metadata}
//!                        | {t:"chunks", subs, modalities, payloads(bin), metadatas}
//!                        | {t:"done",  rid, sub}
//!                        | {t:"pong",  rid, sub}
//!                        | {t:"metrics", rid, sub, data(bin), metadata}
//!
//! A `metrics` request carries this frontend's cumulative refusal counts
//! (`refusals.rs`) as three parallel lists, for the API server to export. Its
//! reply carries the API server's rendered Prometheus text in `data` and its
//! `content_type` in `metadata`; `metadata` has no `content_type` when the
//! server runs without `--enable-metrics`.
//!
//! `chunks` is many submissions' non-error chunks as parallel lists, in
//! order: one message per pass of the API server's message thread rather than
//! one per request.
//!
//! `sub` names one submission and is never reused; `rid` is the client's id,
//! which a client may reuse as soon as it drops a stream (that stream's `done`
//! can cross the retry), so routes and aborts go by `sub`. A `rid` with a
//! submission still in flight is refused (mstar's 409); a dropped stream's
//! abort is sent before its `rid` is freed, so it reaches the backend before
//! any retry's submit.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mstar_rust::communicator::RawZmqCommunicator;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::adapters::{Part, SubmitArgs};
use crate::refusals::Snapshot;

/// One chunk of generated output, matching mstar's `ResultChunk`.
#[derive(Debug, Clone)]
pub struct ResultChunk {
    pub modality: String,
    pub data: Vec<u8>,
    pub metadata: Value,
}

/// A result chunk, a backend error, or end-of-stream, delivered to a
/// request's serving task.
#[derive(Debug)]
pub enum StreamItem {
    Chunk(ResultChunk),
    /// The backend failed this request (worker exception, bad input);
    /// terminal — the route is removed. `status` is the HTTP status the
    /// frontend should return (from the error chunk's `metadata.status`).
    Error { status: u16, message: String },
    Done,
}

// ---- wire messages ----

#[derive(Serialize)]
struct SubmitMsg<'a> {
    t: &'static str,
    rid: &'a str,
    sub: u64,
    text: Option<&'a str>,
    /// Pre-tokenized prompt (frontend-tokenizes path); `text` is None then.
    tokens: Option<&'a [u32]>,
    /// The frontend will detokenize: text emissions come back as raw token
    /// ids (modality "token", 4-byte LE) instead of detokenized pieces.
    frontend_detok: bool,
    /// This frontend takes `chunks` messages; a backend sends them only to a
    /// frontend that says so, so either side can be older than the other.
    accepts_chunks: bool,
    file_paths: &'a BTreeMap<String, Vec<String>>,
    input_modalities: &'a [String],
    output_modalities: &'a [String],
    model_kwargs: &'a serde_json::Map<String, Value>,
    /// `submit_request`'s `prompt_parts`, one map per part; left out when
    /// empty, so every other path's message is unchanged.
    #[serde(skip_serializing_if = "no_parts")]
    prompt_parts: &'a [Part],
    streaming: bool,
}

fn no_parts(parts: &&[Part]) -> bool {
    parts.is_empty()
}

/// A `ping` or `metrics` request; only `metrics` carries `refusals`.
#[derive(Serialize)]
struct ProbeMsg<'a> {
    t: &'a str,
    rid: &'a str,
    sub: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    refusals: Option<&'a Snapshot>,
}

/// Inbound message; `t` selects chunk / err / done. `data` is msgpack binary.
#[derive(Deserialize)]
struct Inbound {
    t: String,
    #[serde(default)]
    sub: Option<u64>,
    #[serde(default)]
    modality: String,
    #[serde(default)]
    data: serde_bytes::ByteBuf,
    #[serde(default)]
    metadata: Value,
    #[serde(default)]
    msg: String, // error message for t == "err"
    // t == "chunks": one entry per chunk
    #[serde(default)]
    subs: Vec<u64>,
    #[serde(default)]
    modalities: Vec<String>,
    #[serde(default)]
    payloads: Vec<serde_bytes::ByteBuf>,
    #[serde(default)]
    metadatas: Vec<Value>,
}

struct Route {
    tx: UnboundedSender<StreamItem>,
    /// The client's id; None for a ping.
    rid: Option<String>,
}

/// sub -> the serving task's route (the demux thread routes chunks here),
/// rid -> the sub in flight under it, and the rids a caller is waiting on in
/// [`Bridge::wait_for_request_id`].
#[derive(Default)]
struct Table {
    routes: HashMap<u64, Route>,
    live: HashMap<String, u64>,
    waiting: HashSet<String>,
}

impl Table {
    /// Route submission `sub` under `rid`, unless a submission under `rid` is
    /// still in flight (the 409 detail then).
    fn claim(&mut self, rid: &str, sub: u64, tx: UnboundedSender<StreamItem>) -> Result<(), String> {
        if self.live.contains_key(rid) {
            return Err(request_id_in_use(rid));
        }
        self.live.insert(rid.to_string(), sub);
        self.routes.insert(sub, Route { tx, rid: Some(rid.to_string()) });
        Ok(())
    }

    /// Drop submission `sub`'s route and free its rid.
    fn remove(&mut self, sub: u64) {
        let Some(route) = self.routes.remove(&sub) else {
            return;
        };
        if let Some(rid) = route.rid {
            if self.live.get(&rid) == Some(&sub) {
                self.live.remove(&rid);
            }
        }
    }
}

type Routes = Mutex<Table>;

/// The one caller waiting out a live rid's grace. Dropping it frees the slot,
/// also when the waiting request is cancelled mid-wait.
struct Waiter<'a> {
    routes: &'a Routes,
    rid: String,
}

impl<'a> Waiter<'a> {
    /// None when nothing is in flight under `rid`, or another caller is
    /// already waiting on it.
    fn start(routes: &'a Routes, rid: &str) -> Option<Self> {
        let mut table = routes.lock().expect("routes lock");
        if !table.live.contains_key(rid) || !table.waiting.insert(rid.to_string()) {
            return None;
        }
        Some(Self { routes, rid: rid.to_string() })
    }
}

impl Drop for Waiter<'_> {
    fn drop(&mut self) {
        self.routes.lock().expect("routes lock").waiting.remove(&self.rid);
    }
}

/// The 409 detail for a `request_id` that still has a request in flight
/// (mstar's `request_id_in_use`).
pub fn request_id_in_use(request_id: &str) -> String {
    format!(
        "request_id '{request_id}' already has a request in flight; \
         wait for it to finish or abort it first"
    )
}

pub struct Bridge {
    mbox: Arc<RawZmqCommunicator>,
    routes: Arc<Routes>,
    next_sub: AtomicU64,
}

impl Bridge {
    /// Bind the frontend inbox and start the demux thread that fans inbound
    /// conductor messages out to per-request channels.
    pub fn new(socket_dir: &str) -> Result<Self, String> {
        let mbox = Arc::new(
            RawZmqCommunicator::bind("frontend", socket_dir).map_err(|e| e.to_string())?,
        );
        let routes: Arc<Routes> = Arc::new(Mutex::new(Table::default()));

        let mbox_r = mbox.clone();
        let routes_r = routes.clone();
        std::thread::spawn(move || Self::demux_loop(mbox_r, routes_r));

        Ok(Self { mbox, routes, next_sub: AtomicU64::new(1) })
    }

    fn demux_loop(mbox: Arc<RawZmqCommunicator>, routes: Arc<Routes>) {
        loop {
            let Some(payload) = mbox.recv_timeout(Duration::from_millis(500)) else {
                continue;
            };
            let msg = match rmp_serde::from_slice::<Inbound>(&payload) {
                Ok(m) => m,
                Err(e) => {
                    // A frame we can't decode is a bridge-protocol mismatch or a
                    // corrupt message; log it rather than dropping silently
                    // (otherwise a systematic mismatch looks like a hang).
                    eprintln!(
                        "mstar-server bridge: dropping undecodable message ({e}); {} bytes",
                        payload.len()
                    );
                    continue;
                }
            };
            deliver(&routes, msg);
        }
    }

    /// Whether a submission under `request_id` is still in flight.
    pub fn in_flight(&self, request_id: &str) -> bool {
        self.routes.lock().expect("routes lock").live.contains_key(request_id)
    }

    /// Give a submission still in flight under `request_id` up to `grace` to
    /// end before [`Bridge::submit`] refuses a new one: a client that drops a
    /// stream and resubmits at once can arrive before the drop is noticed
    /// (mstar's `wait_for_request_id`). One caller per rid waits; another
    /// that arrives meanwhile returns at once and is refused while the rid is
    /// in flight, so N duplicates of a live rid cost one poll loop.
    pub async fn wait_for_request_id(&self, request_id: &str, grace: Duration) {
        let Some(_waiter) = Waiter::start(&self.routes, request_id) else {
            return;
        };
        let deadline = tokio::time::Instant::now() + grace;
        while self.in_flight(request_id) && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Submit a flattened request under `request_id`; returns its `sub` (for
    /// [`Bridge::abort`]) and a receiver that yields `ResultChunk`s until
    /// `Done`, or the 409 detail when a submission under `request_id` is
    /// still in flight. The `request_id` is caller-supplied
    /// (the serving handler mints `chatcmpl-…`/`speech-…`), matching mstar,
    /// whose per-request RNG seed derives from `hash(request_id)`.
    pub fn submit(
        &self,
        args: &SubmitArgs,
        request_id: &str,
        streaming: bool,
    ) -> Result<(u64, UnboundedReceiver<StreamItem>), String> {
        let (tx, rx) = unbounded_channel();
        let sub = self.next_sub.fetch_add(1, Ordering::Relaxed);
        self.routes.lock().expect("routes lock").claim(request_id, sub, tx.clone())?;
        let msg = SubmitMsg {
            t: "submit",
            rid: request_id,
            sub,
            text: args.text.as_deref(),
            tokens: args.tokens.as_deref(),
            frontend_detok: args.tokens.is_some(),
            accepts_chunks: true,
            file_paths: &args.file_paths,
            input_modalities: &args.input_modalities,
            output_modalities: &args.output_modalities,
            model_kwargs: &args.model_kwargs,
            prompt_parts: &args.parts,
            streaming,
        };
        let payload = rmp_serde::to_vec_named(&msg).expect("encode submit");
        if self.mbox.send("bridge", &payload).is_err() {
            // Backend bridge unreachable: fail the request now instead of
            // leaving the client hanging until request_timeout with no
            // signal. The receiver is already routed to `tx`.
            let _ = tx.send(StreamItem::Error {
                status: 500,
                message: "backend unreachable (bridge send failed)".to_string(),
            });
            self.routes.lock().expect("routes lock").remove(sub);
        }
        Ok((sub, rx))
    }

    /// Send `{t}` to the bridge loop and wait for its one reply (routed like
    /// a completed request). None on timeout: the bridge loop or the backend
    /// behind it is gone.
    async fn round_trip(
        &self,
        t: &str,
        refusals: Option<&Snapshot>,
        timeout: Duration,
    ) -> Option<StreamItem> {
        let rid = format!("{t}-{}", uuid::Uuid::new_v4());
        let (tx, mut rx) = unbounded_channel();
        let sub = self.next_sub.fetch_add(1, Ordering::Relaxed);
        self.routes.lock().expect("routes lock").routes.insert(sub, Route { tx, rid: None });
        let msg = ProbeMsg { t, rid: &rid, sub, refusals };
        if let Ok(payload) = rmp_serde::to_vec_named(&msg) {
            let _ = self.mbox.send("bridge", &payload);
        }
        let item = tokio::time::timeout(timeout, rx.recv()).await.ok().flatten();
        self.routes.lock().expect("routes lock").remove(sub);
        item
    }

    /// Liveness round-trip for the deep health check: send `{t:"ping"}` and
    /// wait for the backend's pong. False on timeout.
    pub async fn ping(&self, timeout: Duration) -> bool {
        self.round_trip("ping", None, timeout).await.is_some()
    }

    /// The API server's `/metrics` reply, asked with this frontend's refusal
    /// counts: the rendered text in `data` and, when metrics are on, its
    /// `content_type` in `metadata`. None on timeout.
    pub async fn metrics(&self, refusals: &Snapshot, timeout: Duration) -> Option<ResultChunk> {
        match self.round_trip("metrics", Some(refusals), timeout).await? {
            StreamItem::Chunk(chunk) => Some(chunk),
            _ => None,
        }
    }

    /// Abort submission `sub` of `request_id` (client disconnected or timed
    /// out): tell the backend to release its state (a finished submission is
    /// a no-op there) and drop the local route.
    pub fn abort(&self, request_id: &str, sub: u64) {
        let msg = serde_json::json!({"t": "abort", "rid": request_id, "sub": sub});
        // Sent before the rid is freed, so a retry's submit (which needs the
        // rid free) goes out after it and the backend never refuses the
        // retry. Not under the routes lock: a send can block, and the demux
        // thread needs that lock to drain the replies that unblock it.
        if let Ok(payload) = rmp_serde::to_vec_named(&msg) {
            let _ = self.mbox.send("bridge", &payload);
        }
        self.routes.lock().expect("routes lock").remove(sub);
    }
}

/// Route one inbound message to its submission's serving task; a terminal one
/// (error, done, pong) also drops the route.
fn deliver(routes: &Routes, msg: Inbound) {
    if msg.t == "chunks" {
        return deliver_batch(routes, msg);
    }
    let (item, done) = match msg.t.as_str() {
        // A failed request is delivered in-band as a terminal chunk with
        // modality "error" (message in `data`, HTTP status in
        // `metadata.status`) that ends cleanly — the backend does NOT
        // raise. Surface it as a terminal Error so non-streaming handlers
        // return the real status instead of a 200 with the error chunk
        // buried in `outputs` (/generate) or silently filtered out
        // (chat/speech/images/videos).
        "chunk" if msg.modality == "error" => {
            let status = msg
                .metadata
                .get("status")
                .and_then(Value::as_u64)
                .unwrap_or(500) as u16;
            let message = String::from_utf8_lossy(&msg.data.into_vec()).into_owned();
            (Some(StreamItem::Error { status, message }), true)
        }
        "chunk" => (
            Some(StreamItem::Chunk(ResultChunk {
                modality: msg.modality,
                data: msg.data.into_vec(),
                metadata: msg.metadata,
            })),
            false,
        ),
        // Legacy out-of-band error (the bridge relay caught an exception
        // and sent `t="err"`); no status carried, default to 500.
        "err" => (
            Some(StreamItem::Error { status: 500, message: msg.msg }),
            true,
        ),
        "done" => (Some(StreamItem::Done), true),
        // Liveness reply: completes the ping's route like a request.
        "pong" => (Some(StreamItem::Done), true),
        "metrics" => (
            Some(StreamItem::Chunk(ResultChunk {
                modality: "metrics".to_string(),
                data: msg.data.into_vec(),
                metadata: msg.metadata,
            })),
            true,
        ),
        _ => (None, false),
    };
    let (Some(item), Some(sub)) = (item, msg.sub) else {
        return;
    };
    let mut table = routes.lock().expect("routes lock");
    if let Some(route) = table.routes.get(&sub) {
        let _ = route.tx.send(item);
    }
    if done {
        table.remove(sub);
    }
}

/// Route a `chunks` message's chunks, each to its submission, under one lock.
/// The backend sends error chunks only as `chunk`, so none here is terminal.
fn deliver_batch(routes: &Routes, msg: Inbound) {
    let n = msg.subs.len();
    if msg.modalities.len() != n || msg.payloads.len() != n || msg.metadatas.len() != n {
        eprintln!("mstar-server bridge: dropping a chunks message with ragged lists");
        return;
    }
    let table = routes.lock().expect("routes lock");
    let items = msg.subs.into_iter().zip(msg.modalities).zip(msg.payloads).zip(msg.metadatas);
    for (((sub, modality), data), metadata) in items {
        if let Some(route) = table.routes.get(&sub) {
            let chunk = ResultChunk { modality, data: data.into_vec(), metadata };
            let _ = route.tx.send(StreamItem::Chunk(chunk));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn encoded(parts: &[Part]) -> Value {
        let file_paths = BTreeMap::new();
        let model_kwargs = serde_json::Map::new();
        let msg = SubmitMsg {
            t: "submit",
            rid: "r",
            sub: 1,
            text: Some("hi"),
            tokens: None,
            frontend_detok: false,
            accepts_chunks: true,
            file_paths: &file_paths,
            input_modalities: &[],
            output_modalities: &[],
            model_kwargs: &model_kwargs,
            prompt_parts: parts,
            streaming: true,
        };
        rmp_serde::from_slice(&rmp_serde::to_vec_named(&msg).unwrap()).unwrap()
    }

    #[test]
    fn a_submit_with_no_parts_leaves_the_key_out() {
        assert!(
            encoded(&[]).get("prompt_parts").is_none(),
            "every non-chat submit gained a key the bridge did not have before"
        );
    }

    #[test]
    fn a_chat_submit_carries_each_part_by_name() {
        let parts = [
            Part { modality: "image".into(), text: None, index: 0, role: "user".into() },
            Part { modality: "text".into(), text: Some("what is it".into()), index: 0, role: "user".into() },
        ];
        assert_eq!(
            encoded(&parts)["prompt_parts"],
            json!([
                {"modality": "image", "text": null, "index": 0, "role": "user"},
                {"modality": "text", "text": "what is it", "index": 0, "role": "user"},
            ]),
            "a part did not travel as a map of PromptPart's fields"
        );
    }

    fn inbound(t: &str, sub: u64) -> Inbound {
        let msg = json!({"t": t, "rid": "X", "sub": sub, "modality": "text"});
        rmp_serde::from_slice(&rmp_serde::to_vec_named(&msg).unwrap()).unwrap()
    }

    fn error_chunk(sub: u64) -> Inbound {
        let msg = json!({"t": "chunk", "rid": "X", "sub": sub, "modality": "error",
                         "metadata": {"status": 500}});
        rmp_serde::from_slice(&rmp_serde::to_vec_named(&msg).unwrap()).unwrap()
    }

    #[test]
    fn a_rid_in_flight_is_refused_and_the_running_submission_is_untouched() {
        let routes: Routes = Mutex::new(Table::default());
        let (tx, mut rx) = unbounded_channel();
        routes.lock().unwrap().claim("X", 1, tx).unwrap();

        let (dup_tx, _dup_rx) = unbounded_channel();
        let refused = routes.lock().unwrap().claim("X", 2, dup_tx);

        assert_eq!(refused, Err(request_id_in_use("X")));
        assert!(!routes.lock().unwrap().routes.contains_key(&2));
        deliver(&routes, inbound("chunk", 1));
        assert!(matches!(rx.try_recv(), Ok(StreamItem::Chunk(_))), "the running one lost its route");
        assert_eq!(routes.lock().unwrap().live.get("X"), Some(&1));
    }

    #[test]
    fn a_rid_is_free_once_its_submission_ends_fails_or_is_aborted() {
        let routes: Routes = Mutex::new(Table::default());
        let claim = |sub| routes.lock().unwrap().claim("X", sub, unbounded_channel().0);

        claim(1).unwrap();
        deliver(&routes, inbound("done", 1));
        claim(2).expect("refused after the previous one finished");
        deliver(&routes, error_chunk(2));
        claim(3).expect("refused after the previous one failed");
        routes.lock().unwrap().remove(3); // Bridge::abort
        claim(4).expect("refused after the previous one was aborted");
    }

    #[test]
    fn an_older_submission_s_end_does_not_reach_a_retry_under_its_id() {
        // The client dropped submission 1 of X and retried as submission 2;
        // 1's done arrives after 2's submit.
        let routes: Routes = Mutex::new(Table::default());
        let (old_tx, _old_rx) = unbounded_channel();
        let (new_tx, mut new_rx) = unbounded_channel();
        routes.lock().unwrap().claim("X", 1, old_tx).unwrap();
        routes.lock().unwrap().remove(1); // Bridge::abort
        routes.lock().unwrap().claim("X", 2, new_tx).unwrap();

        deliver(&routes, inbound("done", 1));
        deliver(&routes, inbound("chunk", 2));

        let table = routes.lock().unwrap();
        assert!(table.routes.contains_key(&2), "the retry's route was dropped");
        assert_eq!(table.live.get("X"), Some(&2), "the older end freed the retry's rid");
        assert!(matches!(new_rx.try_recv(), Ok(StreamItem::Chunk(_))));
        assert!(new_rx.try_recv().is_err(), "the retry got the older submission's end");
    }

    fn chunks_msg(subs: Vec<u64>, payloads: Vec<&[u8]>) -> Inbound {
        #[derive(Serialize)]
        struct Chunks {
            t: &'static str,
            subs: Vec<u64>,
            modalities: Vec<String>,
            payloads: Vec<serde_bytes::ByteBuf>,
            metadatas: Vec<Value>,
        }
        let n = payloads.len();
        let msg = Chunks {
            t: "chunks",
            subs,
            modalities: vec!["text".into(); n],
            payloads: payloads.into_iter().map(|p| serde_bytes::ByteBuf::from(p.to_vec())).collect(),
            metadatas: vec![json!({}); n],
        };
        rmp_serde::from_slice(&rmp_serde::to_vec_named(&msg).unwrap()).unwrap()
    }

    #[test]
    fn a_chunks_message_reaches_each_submission_in_order_and_ends_none() {
        let routes: Routes = Mutex::new(Table::default());
        let (tx1, mut rx1) = unbounded_channel();
        let (tx2, mut rx2) = unbounded_channel();
        routes.lock().unwrap().claim("A", 1, tx1).unwrap();
        routes.lock().unwrap().claim("B", 2, tx2).unwrap();

        // 9 is a submission already gone: its chunk is dropped
        deliver(&routes, chunks_msg(vec![1, 2, 9, 1], vec![b"a1", b"b1", b"x", b"a2"]));

        let data = |rx: &mut UnboundedReceiver<StreamItem>| -> Vec<Vec<u8>> {
            std::iter::from_fn(|| match rx.try_recv() {
                Ok(StreamItem::Chunk(c)) => Some(c.data),
                Ok(_) => panic!("a chunks message delivered a terminal item"),
                Err(_) => None,
            })
            .collect()
        };
        assert_eq!(data(&mut rx1), [b"a1".to_vec(), b"a2".to_vec()]);
        assert_eq!(data(&mut rx2), [b"b1".to_vec()]);
        let table = routes.lock().unwrap();
        assert!(table.routes.contains_key(&1) && table.routes.contains_key(&2));
    }

    #[test]
    fn a_chunks_message_with_ragged_lists_is_dropped_whole() {
        let routes: Routes = Mutex::new(Table::default());
        let (tx, mut rx) = unbounded_channel();
        routes.lock().unwrap().claim("A", 1, tx).unwrap();
        let mut msg = chunks_msg(vec![1, 1], vec![b"a", b"b"]);
        msg.metadatas.pop();
        deliver(&routes, msg);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn a_metrics_reply_reaches_its_caller_and_ends_the_route() {
        #[derive(Serialize)]
        struct Reply {
            t: &'static str,
            sub: u64,
            #[serde(with = "serde_bytes")]
            data: Vec<u8>,
            metadata: Value,
        }
        let reply = Reply {
            t: "metrics",
            sub: 7,
            data: b"mstar_requests_in_flight{model_name=\"m\"} 2.0\n".to_vec(),
            metadata: json!({"content_type": "text/plain; version=1.0.0; charset=utf-8"}),
        };
        let routes: Routes = Mutex::new(Table::default());
        let (tx, mut rx) = unbounded_channel();
        routes.lock().unwrap().routes.insert(7, Route { tx, rid: None });

        deliver(&routes, rmp_serde::from_slice(&rmp_serde::to_vec_named(&reply).unwrap()).unwrap());

        let Ok(StreamItem::Chunk(chunk)) = rx.try_recv() else {
            panic!("the metrics reply did not reach its caller");
        };
        assert_eq!(chunk.data, reply.data);
        assert_eq!(chunk.metadata, reply.metadata);
        assert!(routes.lock().unwrap().routes.is_empty(), "the reply left its route behind");
    }

    #[test]
    fn only_a_metrics_request_carries_the_refusal_counts() {
        let encode = |msg: &ProbeMsg| -> Value {
            rmp_serde::from_slice(&rmp_serde::to_vec_named(msg).unwrap()).unwrap()
        };
        let snap = Snapshot { route: vec!["/generate".into()], status: vec![401], count: vec![3] };
        assert_eq!(
            encode(&ProbeMsg { t: "metrics", rid: "m", sub: 4, refusals: Some(&snap) }),
            json!({"t": "metrics", "rid": "m", "sub": 4,
                   "refusals": {"route": ["/generate"], "status": [401], "count": [3]}})
        );
        assert_eq!(
            encode(&ProbeMsg { t: "ping", rid: "p", sub: 5, refusals: None }),
            json!({"t": "ping", "rid": "p", "sub": 5})
        );
    }

    #[test]
    fn one_caller_at_a_time_waits_out_a_live_rid_s_grace() {
        let routes: Routes = Mutex::new(Table::default());
        assert!(Waiter::start(&routes, "X").is_none(), "waited on a rid with nothing in flight");
        routes.lock().unwrap().claim("X", 1, unbounded_channel().0).unwrap();

        let first = Waiter::start(&routes, "X").expect("the first caller did not wait");
        assert!(Waiter::start(&routes, "X").is_none(), "a second caller waited too");
        drop(first);
        assert!(Waiter::start(&routes, "X").is_some(), "the slot stayed taken after the wait");
    }
}
