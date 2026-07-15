//! Per-entity mailbox over ZeroMQ PUSH/PULL — the direct analogue of mstar's
//! `ZMQCommunicator`.
//!
//! Layered as **transport + codec**, so the wire format is a seam rather than
//! a baked-in choice (mstar migrates pickle → language-neutral encodings by
//! swapping the codec, not the transport):
//!
//! * [`RawZmqCommunicator`] — the transport. Sends/receives **opaque byte
//!   frames** (one zmq frame = exactly the payload, no added framing), so a
//!   Python pickle blob or a msgpack blob passes through
//!   untouched. Also owns the delivery machinery: ipc *and* tcp endpoints,
//!   lazily-cached PUSH sockets, and **wakeup fds** (poll an eventfd alongside
//!   the PULL socket so an external event — e.g. a completed compute future —
//!   wakes the receive loop immediately instead of on the poll timeout).
//! * [`ZmqCommunicator<M, C>`] — a typed wrapper: `Codec` encodes/decodes `M`
//!   to bytes ([`MsgpackCodec`] by default — the language-neutral wire for
//!   Rust-internal messaging).
//!
//! Each entity binds one **PULL** inbox (default `ipc://<dir>/<my_id>.ipc`;
//! or any zmq endpoint via [`RawZmqCommunicator::bind_endpoint`], e.g.
//! `tcp://0.0.0.0:5701` for the multi-node path) and connects a lazily-cached
//! **PUSH** socket per peer. PUSH/PULL gives fire-and-forget, ordered,
//! load-balanced delivery; libzmq queues to a not-yet-bound peer and
//! transparently reconnects when a peer restarts, so there is no
//! "unreachable" error to handle.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::os::fd::RawFd;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CommError {
    #[error("zmq: {0}")]
    Zmq(#[from] zmq::Error),
    #[error("serialize: {0}")]
    Serialize(rmp_serde::encode::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("no endpoint for peer '{0}' (register_peer it, or bind with an ipc dir)")]
    UnknownPeer(String),
}

/// The default ipc endpoint an entity binds/connects to.
fn ipc_endpoint(dir: &Path, id: &str) -> String {
    format!("ipc://{}/{}.ipc", dir.display(), id)
}

/// The backing socket file (for clearing a stale one before bind).
fn sock_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.ipc"))
}

/// What a wake-aware receive returned.
#[derive(Debug, PartialEq, Eq)]
pub enum RecvEvent {
    /// An inbound message frame.
    Message(Vec<u8>),
    /// A registered wakeup fd is readable (the registrant reads/clears it —
    /// e.g. `read(2)` on the eventfd — the transport only polls it).
    Wake,
    /// The timeout elapsed with neither.
    Timeout,
}

/// Byte-frame transport: PUSH/PULL mailbox with pluggable endpoints and
/// wakeup-fd polling. This is the language-neutral seam — payloads are opaque.
///
/// Field order matters for `Drop`: the sockets must close before the
/// `Context` is dropped (`zmq_ctx_term` blocks until its sockets are gone).
pub struct RawZmqCommunicator {
    my_id: String,
    /// ipc directory for the default peer-endpoint scheme (None when bound via
    /// an explicit endpoint and peers must be `register_peer`ed).
    dir: Option<PathBuf>,
    // PULL inbox. Behind a Mutex because a zmq `Socket` is `!Sync` (and must be
    // used from one thread at a time); the drive loop is the single consumer.
    pull: Mutex<zmq::Socket>,
    // peer id -> connected PUSH socket (created on first send to that peer).
    peers: Mutex<HashMap<String, zmq::Socket>>,
    // peer id -> explicit endpoint (tcp or ipc), overriding the dir scheme.
    peer_endpoints: Mutex<HashMap<String, String>>,
    // External fds polled alongside the PULL socket (mstar's worker registers
    // an eventfd so a completed compute future wakes the message loop
    // immediately — "done wrong, this silently stalls the async worker").
    wakeup_fds: Mutex<Vec<RawFd>>,
    ctx: zmq::Context,
}

impl RawZmqCommunicator {
    /// Bind this entity's PULL inbox at `ipc://<dir>/<my_id>.ipc` (the default
    /// single-node scheme: peers resolve by id within the same dir).
    pub fn bind(my_id: impl Into<String>, dir: impl Into<PathBuf>) -> Result<Self, CommError> {
        let my_id = my_id.into();
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let _ = std::fs::remove_file(sock_path(&dir, &my_id)); // clear a stale socket
        let endpoint = ipc_endpoint(&dir, &my_id);
        Self::bind_inner(my_id, Some(dir), &endpoint)
    }

    /// Bind this entity's PULL inbox at an explicit zmq endpoint — e.g.
    /// `tcp://0.0.0.0:5701` for the multi-node path, or `tcp://127.0.0.1:*`
    /// for an OS-assigned port (query it with [`Self::last_endpoint`]).
    /// Peers have no implicit scheme here: `register_peer` each one.
    pub fn bind_endpoint(
        my_id: impl Into<String>,
        endpoint: &str,
    ) -> Result<Self, CommError> {
        Self::bind_inner(my_id.into(), None, endpoint)
    }

    fn bind_inner(
        my_id: String,
        dir: Option<PathBuf>,
        endpoint: &str,
    ) -> Result<Self, CommError> {
        let ctx = zmq::Context::new();
        let pull = ctx.socket(zmq::PULL)?;
        pull.set_linger(0)?; // don't block on close
        pull.bind(endpoint)?;
        Ok(Self {
            my_id,
            dir,
            pull: Mutex::new(pull),
            peers: Mutex::new(HashMap::new()),
            peer_endpoints: Mutex::new(HashMap::new()),
            wakeup_fds: Mutex::new(Vec::new()),
            ctx,
        })
    }

    pub fn id(&self) -> &str {
        &self.my_id
    }

    /// The bound endpoint as zmq reports it — with `tcp://…:*` this carries
    /// the OS-assigned port, which is what peers must `register_peer`.
    pub fn last_endpoint(&self) -> Result<String, CommError> {
        let pull = self.pull.lock().expect("pull lock");
        Ok(pull
            .get_last_endpoint()?
            .unwrap_or_else(|_| String::new()))
    }

    /// Map `peer_id` to an explicit endpoint (tcp or ipc). Overrides the
    /// ipc-dir scheme; required for peers reached over tcp. Replacing a
    /// mapping drops the cached socket so the next send reconnects.
    pub fn register_peer(&self, peer_id: &str, endpoint: &str) {
        self.peer_endpoints
            .lock()
            .expect("peer endpoints lock")
            .insert(peer_id.to_string(), endpoint.to_string());
        self.peers.lock().expect("peers lock").remove(peer_id);
    }

    /// Poll an external fd alongside the PULL socket: when it becomes readable
    /// the receive loop wakes immediately ([`RecvEvent::Wake`]) instead of on
    /// the poll timeout. The registrant owns clearing it (e.g. reading the
    /// eventfd); until cleared, wake-aware receives keep returning `Wake`
    /// (level-triggered).
    pub fn register_wakeup_fd(&self, fd: RawFd) {
        self.wakeup_fds.lock().expect("wakeup fds lock").push(fd);
    }

    fn resolve(&self, peer_id: &str) -> Result<String, CommError> {
        if let Some(ep) = self
            .peer_endpoints
            .lock()
            .expect("peer endpoints lock")
            .get(peer_id)
        {
            return Ok(ep.clone());
        }
        match &self.dir {
            Some(dir) => Ok(ipc_endpoint(dir, peer_id)),
            None => Err(CommError::UnknownPeer(peer_id.to_string())),
        }
    }

    /// Send one opaque byte frame to `peer_id` (fire-and-forget). Queues if
    /// the peer isn't bound yet and reconnects transparently if it restarted.
    pub fn send(&self, peer_id: &str, payload: &[u8]) -> Result<(), CommError> {
        let mut peers = self.peers.lock().expect("peers lock");
        if !peers.contains_key(peer_id) {
            let endpoint = self.resolve(peer_id)?;
            let push = self.ctx.socket(zmq::PUSH)?;
            push.set_linger(0)?;
            push.connect(&endpoint)?;
            peers.insert(peer_id.to_string(), push);
        }
        peers
            .get(peer_id)
            .expect("just inserted")
            .send(payload, 0)?;
        Ok(())
    }

    /// Non-blocking: next inbound frame, or None.
    pub fn try_recv(&self) -> Option<Vec<u8>> {
        let pull = self.pull.lock().expect("pull lock");
        pull.recv_bytes(zmq::DONTWAIT).ok()
    }

    /// Block until the next inbound frame (ignores wakeup fds).
    pub fn recv(&self) -> Option<Vec<u8>> {
        let pull = self.pull.lock().expect("pull lock");
        pull.recv_bytes(0).ok()
    }

    /// Block up to `timeout` for the next inbound frame. Wakeup fds cut the
    /// wait short (returns None early); use [`Self::recv_or_wake`] to
    /// distinguish a wake from a timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<Vec<u8>> {
        match self.recv_or_wake(timeout) {
            RecvEvent::Message(b) => Some(b),
            _ => None,
        }
    }

    /// Wake-aware receive: a message frame, a wakeup-fd trip, or a timeout —
    /// whichever comes first. Messages win a simultaneous wake (the wake is
    /// level-triggered, so it is not lost: the next call reports it).
    pub fn recv_or_wake(&self, timeout: Duration) -> RecvEvent {
        let pull = self.pull.lock().expect("pull lock");
        let ms = timeout.as_millis().min(i64::MAX as u128) as i64;
        let fds: Vec<RawFd> = self.wakeup_fds.lock().expect("wakeup fds lock").clone();

        let mut items = Vec::with_capacity(1 + fds.len());
        items.push(pull.as_poll_item(zmq::POLLIN));
        for &fd in &fds {
            items.push(zmq::PollItem::from_fd(fd, zmq::POLLIN));
        }
        let n = zmq::poll(&mut items, ms).unwrap_or(0);
        if n <= 0 {
            return RecvEvent::Timeout;
        }
        if items[0].is_readable() {
            if let Ok(b) = pull.recv_bytes(zmq::DONTWAIT) {
                return RecvEvent::Message(b);
            }
        }
        if items[1..].iter().any(|it| it.is_readable()) {
            return RecvEvent::Wake;
        }
        RecvEvent::Timeout
    }

    /// Drain all currently-queued inbound frames.
    pub fn drain(&self) -> Vec<Vec<u8>> {
        let pull = self.pull.lock().expect("pull lock");
        let mut out = Vec::new();
        while let Ok(b) = pull.recv_bytes(zmq::DONTWAIT) {
            out.push(b);
        }
        out
    }
}

impl Drop for RawZmqCommunicator {
    fn drop(&mut self) {
        // Sockets (pull + peers) close first via field-drop order; zmq unlinks
        // the bound ipc file on close, but remove it defensively too.
        if let Some(dir) = &self.dir {
            let _ = std::fs::remove_file(sock_path(dir, &self.my_id));
        }
    }
}

// ---------------------------------------------------------------------------
// Codec seam + typed wrapper
// ---------------------------------------------------------------------------

/// Message encoding — the seam mstar migrates across (pickle for a
/// Python-to-Python mesh, msgpack for language-neutral ones). The
/// transport never looks inside the bytes.
pub trait Codec<M> {
    fn encode(msg: &M) -> Result<Vec<u8>, CommError>;
    fn decode(bytes: &[u8]) -> Option<M>;
}

/// The default codec: MessagePack, the language-neutral encoding the
/// migration standardizes on — Python peers read the same frames with
/// `msgpack` (`to_vec_named`: maps with field names, like Python dicts).
pub struct MsgpackCodec;

impl<M: Serialize + DeserializeOwned> Codec<M> for MsgpackCodec {
    fn encode(msg: &M) -> Result<Vec<u8>, CommError> {
        rmp_serde::to_vec_named(msg).map_err(CommError::Serialize)
    }
    fn decode(bytes: &[u8]) -> Option<M> {
        rmp_serde::from_slice(bytes).ok()
    }
}

/// A typed mailbox: [`RawZmqCommunicator`] + a [`Codec`]. `M` is the entity's
/// message type; the default codec is MessagePack.
pub struct ZmqCommunicator<M, C = MsgpackCodec> {
    raw: RawZmqCommunicator,
    _marker: PhantomData<fn() -> (M, C)>,
}

impl<M, C> ZmqCommunicator<M, C>
where
    M: Send + 'static,
    C: Codec<M>,
{
    /// Bind this entity's PULL inbox at `ipc://<dir>/<my_id>.ipc`.
    pub fn bind(my_id: impl Into<String>, dir: impl Into<PathBuf>) -> Result<Self, CommError> {
        Ok(Self {
            raw: RawZmqCommunicator::bind(my_id, dir)?,
            _marker: PhantomData,
        })
    }

    /// Bind at an explicit zmq endpoint (e.g. `tcp://0.0.0.0:5701`).
    pub fn bind_endpoint(my_id: impl Into<String>, endpoint: &str) -> Result<Self, CommError> {
        Ok(Self {
            raw: RawZmqCommunicator::bind_endpoint(my_id, endpoint)?,
            _marker: PhantomData,
        })
    }

    /// The underlying byte transport (peer registration, wakeup fds, …).
    pub fn raw(&self) -> &RawZmqCommunicator {
        &self.raw
    }

    pub fn id(&self) -> &str {
        self.raw.id()
    }

    /// Send `msg` to peer `peer_id` (fire-and-forget).
    pub fn send(&self, peer_id: &str, msg: &M) -> Result<(), CommError> {
        self.raw.send(peer_id, &C::encode(msg)?)
    }

    /// Non-blocking: next inbound message, or None.
    pub fn try_recv(&self) -> Option<M> {
        self.raw.try_recv().and_then(|b| C::decode(&b))
    }

    /// Block until the next inbound message.
    pub fn recv(&self) -> Option<M> {
        self.raw.recv().and_then(|b| C::decode(&b))
    }

    /// Block up to `timeout` for the next inbound message (a wakeup fd cuts
    /// the wait short — see [`RawZmqCommunicator::recv_or_wake`]).
    pub fn recv_timeout(&self, timeout: Duration) -> Option<M> {
        self.raw.recv_timeout(timeout).and_then(|b| C::decode(&b))
    }

    /// Drain all currently-queued inbound messages.
    pub fn drain(&self) -> Vec<M> {
        self.raw
            .drain()
            .into_iter()
            .filter_map(|b| C::decode(&b))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    enum Msg {
        Hello(String),
        Batch { id: u64, node: String },
    }

    fn tmpdir(tag: &str) -> PathBuf {
        // Unique per test via the tag + thread id (no Date/rand available).
        let t = format!("{:?}", std::thread::current().id());
        let dir = std::env::temp_dir().join(format!("mstar_comm_{tag}_{t}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn wait_for<M, C, F: Fn(&ZmqCommunicator<M, C>) -> Option<Msg>>(
        mb: &ZmqCommunicator<M, C>,
        f: F,
    ) -> Msg
    where
        M: Send + 'static,
        C: Codec<M>,
    {
        for _ in 0..500 {
            if let Some(m) = f(mb) {
                return m;
            }
            std::thread::sleep(Duration::from_millis(4));
        }
        panic!("timed out waiting for message");
    }

    #[test]
    fn two_entities_exchange_messages() {
        let dir = tmpdir("exch");
        let conductor: ZmqCommunicator<Msg> = ZmqCommunicator::bind("conductor", &dir).unwrap();
        let worker: ZmqCommunicator<Msg> = ZmqCommunicator::bind("worker_0", &dir).unwrap();

        conductor
            .send("worker_0", &Msg::Batch {
                id: 1,
                node: "LLM".into(),
            })
            .unwrap();
        let got = wait_for(&worker, |w| w.try_recv());
        assert_eq!(got, Msg::Batch { id: 1, node: "LLM".into() });

        worker.send("conductor", &Msg::Hello("done".into())).unwrap();
        let got = wait_for(&conductor, |c| c.try_recv());
        assert_eq!(got, Msg::Hello("done".into()));
    }

    #[test]
    fn many_messages_arrive_in_order_from_one_sender() {
        let dir = tmpdir("order");
        let a: ZmqCommunicator<Msg> = ZmqCommunicator::bind("a", &dir).unwrap();
        let b: ZmqCommunicator<Msg> = ZmqCommunicator::bind("b", &dir).unwrap();
        for i in 0..100 {
            a.send("b", &Msg::Batch { id: i, node: "n".into() }).unwrap();
        }
        // PUSH->PULL over one connection preserves FIFO order.
        let mut seen = 0u64;
        for _ in 0..1000 {
            for m in b.drain() {
                if let Msg::Batch { id, .. } = m {
                    assert_eq!(id, seen);
                    seen += 1;
                }
            }
            if seen == 100 {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(seen, 100);
    }

    #[test]
    fn send_before_peer_binds_is_queued() {
        // Unlike the old UDS transport (which errored on a missing peer), zmq
        // PUSH queues to a not-yet-bound endpoint and delivers once it binds.
        let dir = tmpdir("queue");
        let a: ZmqCommunicator<Msg> = ZmqCommunicator::bind("a", &dir).unwrap();
        a.send("late", &Msg::Hello("queued".into())).unwrap(); // no peer yet — no error
        let late: ZmqCommunicator<Msg> = ZmqCommunicator::bind("late", &dir).unwrap();
        let got = wait_for(&late, |m| m.try_recv());
        assert_eq!(got, Msg::Hello("queued".into()));
    }

    #[test]
    fn reconnects_after_peer_restart() {
        let dir = tmpdir("restart");
        let a: ZmqCommunicator<Msg> = ZmqCommunicator::bind("a", &dir).unwrap();
        {
            let b: ZmqCommunicator<Msg> = ZmqCommunicator::bind("b", &dir).unwrap();
            a.send("b", &Msg::Hello("1".into())).unwrap();
            wait_for(&b, |b| b.try_recv());
        } // b drops; zmq unlinks its inbox
        std::thread::sleep(Duration::from_millis(20));
        // New b at the same id: a's cached PUSH auto-reconnects to it.
        let b2: ZmqCommunicator<Msg> = ZmqCommunicator::bind("b", &dir).unwrap();
        a.send("b", &Msg::Hello("2".into())).unwrap();
        let got = wait_for(&b2, |b| b.try_recv());
        assert_eq!(got, Msg::Hello("2".into()));
    }

    // ---- raw transport: byte passthrough, tcp, wakeup fds ----------------

    #[test]
    fn raw_frames_pass_through_unmodified() {
        // The wire is the payload — no framing added. A foreign blob (e.g. a
        // Python pickle) must arrive byte-identical.
        let dir = tmpdir("raw");
        let a = RawZmqCommunicator::bind("a", &dir).unwrap();
        let b = RawZmqCommunicator::bind("b", &dir).unwrap();
        let blob: Vec<u8> = (0..=255).collect();
        a.send("b", &blob).unwrap();
        for _ in 0..500 {
            if let Some(got) = b.try_recv() {
                assert_eq!(got, blob);
                return;
            }
            std::thread::sleep(Duration::from_millis(4));
        }
        panic!("frame never arrived");
    }

    #[test]
    fn tcp_endpoint_exchange() {
        // Multi-node shape: bind on an OS-assigned tcp port, peer registers
        // the reported endpoint explicitly (no ipc dir involved).
        let a = RawZmqCommunicator::bind_endpoint("a", "tcp://127.0.0.1:*").unwrap();
        let b = RawZmqCommunicator::bind_endpoint("b", "tcp://127.0.0.1:*").unwrap();
        let a_ep = a.last_endpoint().unwrap();
        let b_ep = b.last_endpoint().unwrap();
        assert!(a_ep.starts_with("tcp://"), "{a_ep}");
        a.register_peer("b", &b_ep);
        b.register_peer("a", &a_ep);

        a.send("b", b"over tcp").unwrap();
        for _ in 0..500 {
            if let Some(got) = b.try_recv() {
                assert_eq!(got, b"over tcp");
                // and the reverse direction
                b.send("a", b"ack").unwrap();
                for _ in 0..500 {
                    if let Some(back) = a.try_recv() {
                        assert_eq!(back, b"ack");
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(4));
                }
                panic!("ack never arrived");
            }
            std::thread::sleep(Duration::from_millis(4));
        }
        panic!("tcp frame never arrived");
    }

    #[test]
    fn unknown_peer_without_dir_errors() {
        let a = RawZmqCommunicator::bind_endpoint("a", "tcp://127.0.0.1:*").unwrap();
        assert!(matches!(
            a.send("nowhere", b"x"),
            Err(CommError::UnknownPeer(_))
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn wakeup_fd_cuts_recv_wait_short() {
        // mstar's worker registers an eventfd alongside the PULL socket so a
        // completed compute future wakes the loop immediately — not on the
        // poll timeout. Fire the eventfd from a thread mid-wait and require
        // the wake to arrive far sooner than the 2 s timeout.
        let dir = tmpdir("wake");
        let a = RawZmqCommunicator::bind("a", &dir).unwrap();
        let efd = unsafe { libc::eventfd(0, 0) };
        assert!(efd >= 0);
        a.register_wakeup_fd(efd);

        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            let one: u64 = 1;
            let n = unsafe {
                libc::write(efd, &one as *const u64 as *const libc::c_void, 8)
            };
            assert_eq!(n, 8);
        });

        let start = std::time::Instant::now();
        let ev = a.recv_or_wake(Duration::from_secs(2));
        let waited = start.elapsed();
        t.join().unwrap();
        assert_eq!(ev, RecvEvent::Wake);
        assert!(
            waited < Duration::from_millis(500),
            "wake took {waited:?} — the fd poll isn't cutting the wait short"
        );

        // Level-triggered until the registrant clears it...
        assert_eq!(a.recv_or_wake(Duration::from_millis(10)), RecvEvent::Wake);
        let mut buf = 0u64;
        unsafe {
            libc::read(efd, &mut buf as *mut u64 as *mut libc::c_void, 8);
        }
        // ...and quiet after the read (times out).
        assert_eq!(
            a.recv_or_wake(Duration::from_millis(10)),
            RecvEvent::Timeout
        );

        // A message still wins while the fd is quiet.
        let b = RawZmqCommunicator::bind("b", &dir).unwrap();
        b.send("a", b"msg").unwrap();
        for _ in 0..500 {
            match a.recv_or_wake(Duration::from_millis(20)) {
                RecvEvent::Message(m) => {
                    assert_eq!(m, b"msg");
                    unsafe { libc::close(efd) };
                    return;
                }
                _ => continue,
            }
        }
        panic!("message never arrived");
    }

    // ---- codec seam --------------------------------------------------------

    /// A toy foreign codec (length-prefixed utf8) proving the transport is
    /// format-agnostic — the pickle/msgpack seam mstar wants.
    struct TextCodec;
    impl Codec<String> for TextCodec {
        fn encode(msg: &String) -> Result<Vec<u8>, CommError> {
            Ok(msg.as_bytes().to_vec())
        }
        fn decode(bytes: &[u8]) -> Option<String> {
            String::from_utf8(bytes.to_vec()).ok()
        }
    }

    #[test]
    fn custom_codec_over_same_transport() {
        let dir = tmpdir("codec");
        let a: ZmqCommunicator<String, TextCodec> = ZmqCommunicator::bind("a", &dir).unwrap();
        // The peer reads RAW bytes: what TextCodec sent is exactly the utf8 —
        // no transport framing in between.
        let b = RawZmqCommunicator::bind("b", &dir).unwrap();
        a.send("b", &"hello seam".to_string()).unwrap();
        for _ in 0..500 {
            if let Some(bytes) = b.try_recv() {
                assert_eq!(bytes, b"hello seam");
                return;
            }
            std::thread::sleep(Duration::from_millis(4));
        }
        panic!("frame never arrived");
    }
}
