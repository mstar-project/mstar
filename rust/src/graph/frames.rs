//! Outgoing wire frames, built here instead of in Python.
//!
//! `wire.py` is type-driven: the decoder knows every field's declared type, so
//! the frame carries values rather than type tags. A message is
//! `[tag, {field: value}]`, an enum is its `.value` string, a field whose
//! declared type is ABSTRACT carries `[tag, payload]`, and a field that is
//! None and has a default is omitted.
//!
//! What has to match is what the frame DECODES to, not its bytes. The receiver
//! runs `unpackb` and then looks fields up by name, so map order and integer
//! width are free -- `test_a_rust_frame_decodes_like_the_python_one` pins the
//! decoded equality rather than a byte comparison.
//!
//! Fields whose types belong to Python arrive already encoded (see
//! `wire.encode_field`) and are spliced in untouched. That is deliberate:
//! `resource_publish_info` is a `PublishedInfo`, which is abstract, so owning
//! it here would mean a Rust change for every new resource.

use rmp::encode as enc;
use rmpv::Value;

use crate::graph::compile::EMIT_TO_CLIENT;
use crate::graph::spec::{StrToId, Sym};
use crate::tensors::TensorPointerInfo;
use rustc_hash::FxHashMap;

// -- writers -----------------------------------------------------------------
//
// Straight into the output buffer. Going through `rmpv::Value` first allocated
// a `String` for every map KEY -- thirteen per tensor descriptor -- and then
// copied the whole tree again on serialise.
//
// The cost of writing directly is that a map or array length has to be known
// before its entries, so anything with optional fields counts them first. The
// `w_*` names keep those call sites short enough to read.
//
// Every one of these is infallible against a `Vec<u8>`: the only error a
// `RmpWrite` can raise is from the sink, and a Vec never fails. Hence the
// `.expect`s -- they mark "cannot happen" rather than handling anything.

fn w_map(out: &mut Vec<u8>, len: u32) {
    enc::write_map_len(out, len).expect("vec sink");
}

/// A map whose length is DISCOVERED as it is written.
///
/// The header slot is reserved, the fields are written, and the count is
/// patched in afterwards -- so the length comes from the same code that writes
/// the entries and cannot drift from them. A hand-maintained
/// `w_map(out, 12 + u32::from(..))` very much can, and it gets no louder as
/// fields are added or made optional: the frame just goes out malformed.
///
/// Always map16, even for two entries, because a fixed-width header is what
/// makes the slot patchable. Two bytes per map wider than the minimal
/// encoding, and it decodes identically -- msgpack lengths are not canonical,
/// the same licence this module already takes with integer widths.
fn w_map_counted(out: &mut Vec<u8>, fields: impl FnOnce(&mut MapWriter)) {
    let at = out.len();
    out.extend_from_slice(&[0xde, 0, 0]); // map16, patched below
    let mut w = MapWriter { out, n: 0 };
    fields(&mut w);
    // `n` out first: that is the writer's last use, so its borrow of `out`
    // ends here and the header slot becomes writable again.
    let n = w.n;
    out[at + 1..at + 3].copy_from_slice(&n.to_be_bytes());
}

/// Counts the fields it is asked for, so the caller never states a total.
struct MapWriter<'a> {
    out: &'a mut Vec<u8>,
    n: u16,
}

impl MapWriter<'_> {
    /// Write a key and hand back the buffer for its value:
    /// `w_i64(m.key("nbytes"), i.nbytes)`.
    fn key(&mut self, k: &str) -> &mut Vec<u8> {
        self.n += 1;
        w_str(self.out, k);
        self.out
    }

    /// The buffer, for a value that takes more than one call to write (an
    /// array's elements, a nested map).
    fn buf(&mut self) -> &mut Vec<u8> {
        self.out
    }
}

fn w_arr(out: &mut Vec<u8>, len: u32) {
    enc::write_array_len(out, len).expect("vec sink");
}

fn w_str(out: &mut Vec<u8>, v: &str) {
    enc::write_str(out, v).expect("vec sink");
}

fn w_bool(out: &mut Vec<u8>, v: bool) {
    enc::write_bool(out, v).expect("vec sink");
}

fn w_i64(out: &mut Vec<u8>, v: i64) {
    enc::write_sint(out, v).expect("vec sink");
}

fn w_u64(out: &mut Vec<u8>, v: u64) {
    enc::write_uint(out, v).expect("vec sink");
}

fn w_nil(out: &mut Vec<u8>) {
    enc::write_nil(out).expect("vec sink");
}

/// A value Python already encoded, appended verbatim.
///
/// msgpack is self-delimiting, so a complete encoded value needs no decode and
/// re-encode to nest -- which is what the old path did, and it was the whole
/// reason a generic `Value` had to exist here at all.
fn w_raw(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(bytes);
}

/// One `rmpv::Value`, for the two spliced fields that genuinely need one: they
/// are read back OUT of a Python blob, so there is no byte range to copy.
/// Both are small and one is profiling-only.
fn w_value(out: &mut Vec<u8>, v: &Value) {
    rmpv::encode::write_value(out, v).expect("vec sink");
}

/// Decode a field Python already encoded for forwarding into an outgoing frame.
fn spliced(bytes: &[u8]) -> Value {
    rmpv::decode::read_value(&mut &bytes[..]).unwrap_or(Value::Nil)
}

pub fn spliced_value(bytes: Option<&[u8]>) -> Option<Value> {
    bytes.map(spliced)
}

/// The profiling payload is one blob holding `[rx_info, tx_info,
/// graph_timings]` (see `wire.encode_fields`). Split so each rides its own
/// field. A blob of the wrong shape yields nothing rather than a wrong
/// frame -- profiling is diagnostic, and must not take the send down.
pub fn split_profiling(bytes: Option<&[u8]>) -> [Option<Value>; 3] {
    let Some(b) = bytes else { return [None, None, None] };
    match spliced(b) {
        Value::Array(v) if v.len() == 3 => {
            let mut it = v.into_iter();
            [it.next(), it.next(), it.next()]
        }
        _ => [None, None, None],
    }
}

/// One `TensorPointerInfo`. `shm_segment` and the two `_source_*` fields have
/// defaults, so Python omits them when None and so must this -- an explicit
/// nil would decode as a segment literally named nil.
fn tensor_info(out: &mut Vec<u8>, bk: &StrToId, i: &TensorPointerInfo) {
    // `bk`, not the runtime's interner: a descriptor's symbols were interned
    // by the BOOKKEEPER, which keeps its own table.
    w_map_counted(out, |m| {
        w_arr(m.key("dims"), i.dims.len() as u32);
        for &d in &i.dims {
            w_i64(m.buf(), d);
        }
        w_str(m.key("dtype"), bk.name(i.dtype));
        w_i64(m.key("nbytes"), i.nbytes);
        w_u64(m.key("address"), i.address);
        w_arr(m.key("stride"), i.stride.len() as u32);
        for &d in &i.stride {
            w_i64(m.buf(), d);
        }
        w_u64(m.key("uuid"), i.uuid);
        w_str(m.key("source_session_id"), bk.name(i.source_session_id));
        w_str(m.key("source_entity"), bk.name(i.source_entity));
        w_i64(m.key("offset"), i.offset);
        w_u64(m.key("source_tp_size"), i.source_tp_size as u64);
        w_u64(m.key("source_tp_rank"), i.source_tp_rank as u64);
        w_i64(m.key("shm_offset"), i.shm_offset);
        // These three have defaults in Python, so an absent one must be
        // OMITTED -- an explicit nil would decode as a segment literally
        // named None. The count follows the writes, so adding one here needs
        // nothing else changed.
        if let Some(seg) = i.shm_segment {
            w_str(m.key("shm_segment"), bk.name(seg));
        }
        if let Some(n) = i.source_node_name {
            w_str(m.key("_source_node_name"), bk.name(n));
        }
        if let Some(w) = i.source_graph_walk {
            w_str(m.key("_source_graph_walk"), bk.name(w));
        }
    });
}

/// A list of descriptors.
fn tensor_infos(out: &mut Vec<u8>, bk: &StrToId, infos: &[TensorPointerInfo]) {
    w_arr(out, infos.len() as u32);
    for i in infos {
        tensor_info(out, bk, i);
    }
}

/// Python's `NestedLoopIndices`.
fn nested_loop_indices(
    out: &mut Vec<u8>, it: &StrToId, order: &[Sym], indices: &[(Sym, u32)],
    fwd: u32,
) {
    w_map_counted(out, |m| {
        w_arr(m.key("loop_name_order"), order.len() as u32);
        for &n in order {
            w_str(m.buf(), it.name(n));
        }
        w_map(m.key("loop_indices"), indices.len() as u32);
        for &(n, i) in indices {
            w_str(m.buf(), it.name(n));
            w_u64(m.buf(), i as u64);
        }
        w_u64(m.key("wg_fwd_pass_idx"), fwd as u64);
    });
}

/// `dict[str, int]`, as the token/consumed counts ride.
fn str_counts(out: &mut Vec<u8>, pairs: &[(String, i64)]) {
    w_map(out, pairs.len() as u32);
    for (k, v) in pairs {
        w_str(out, k);
        w_i64(out, *v);
    }
}

/// Everything one WORKER_GRAPHS_DONE carries. The `*_encoded` fields are
/// Python's and ride through untouched.
pub struct WorkerGraphsDone<'a> {
    pub request_id: &'a str,
    pub worker_graph_ids: &'a [u32],
    pub is_first_tp_rank: bool,
    pub persist_signals: Vec<(Sym, Vec<TensorPointerInfo>)>,
    pub new_token_counts: &'a [(String, i64)],
    pub output_signal_names: Vec<Sym>,
    pub partition_name: &'a str,
    pub partition_done: bool,
    pub graph_walk: &'a str,
    pub stream_tokens_consumed: &'a [(String, i64)],
    /// `dict[str, dict[str, int]]`: stream edge -> destination worker -> items.
    pub stream_items_sent: Vec<(Sym, Vec<(Sym, i64)>)>,
    /// `dict[str, PublishedInfo]` -- abstract, so it stays Python's. Read
    /// encoded by Python and forwarded without interpreting resource types.
    pub resource_publish_info: Option<Value>,
    /// `rx_info`, `tx_info`, `graph_timings`, in that order: only populated
    /// under enable_prof, and already encoded when they reach us.
    pub profiling: [Option<Value>; 3],
}

impl WorkerGraphsDone<'_> {
    fn body(&self, out: &mut Vec<u8>, it: &StrToId, bk: &StrToId) {
        w_map_counted(out, |m| {
            w_str(m.key("request_id"), self.request_id);
            w_arr(m.key("worker_graph_ids"), self.worker_graph_ids.len() as u32);
            for &w in self.worker_graph_ids {
                w_u64(m.buf(), w as u64);
            }
            w_bool(m.key("is_first_tp_rank"), self.is_first_tp_rank);
            w_map(m.key("persist_signals"), self.persist_signals.len() as u32);
            for (sig, infos) in &self.persist_signals {
                w_str(m.buf(), it.name(*sig));
                tensor_infos(m.buf(), bk, infos);
            }
            str_counts(m.key("new_token_counts"), self.new_token_counts);
            w_arr(
                m.key("output_signal_names"),
                self.output_signal_names.len() as u32,
            );
            for &n in &self.output_signal_names {
                w_str(m.buf(), it.name(n));
            }
            w_str(m.key("partition_name"), self.partition_name);
            w_bool(m.key("partition_done"), self.partition_done);
            w_str(m.key("graph_walk"), self.graph_walk);
            str_counts(
                m.key("stream_tokens_consumed"), self.stream_tokens_consumed,
            );
            w_map(m.key("stream_items_sent"), self.stream_items_sent.len() as u32);
            for (edge, per_worker) in &self.stream_items_sent {
                w_str(m.buf(), it.name(*edge));
                w_map(m.buf(), per_worker.len() as u32);
                for (worker, n) in per_worker {
                    w_str(m.buf(), it.name(*worker));
                    w_i64(m.buf(), *n);
                }
            }
            // Omitted when absent, so the decoder's default stands.
            if let Some(v) = &self.resource_publish_info {
                w_value(m.key("resource_publish_info"), v);
            }
            for (key, value) in ["rx_info", "tx_info", "graph_timings"]
                .iter()
                .zip(&self.profiling)
            {
                if let Some(v) = value {
                    w_value(m.key(key), v);
                }
            }
        });
    }

    /// The full frame: a ConductorMessage wrapping this body. `body` is
    /// declared `MessageBody`, which is abstract, hence the `[tag, payload]`.
    pub fn encode(&self, it: &StrToId, bk: &StrToId) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        open_frame(&mut out, "conductor_msg", "worker_graphs_done",
                   Some("wgs_done"));
        self.body(&mut out, it, bk);
        out
    }
}

/// One `GraphEdge`. Every field has a non-None default, so none is omitted --
/// including `output_modality`, which defaults to `""` rather than nil.
fn graph_edge(
    out: &mut Vec<u8>,
    bk: &StrToId,
    name: &str,
    next_node: &str,
    modality: &str,
    is_streaming: bool,
    infos: &[TensorPointerInfo],
    shard_dim: Option<u32>,
    total_fanin: u32,
) {
    w_map_counted(out, |m| {
        w_str(m.key("next_node"), next_node);
        w_str(m.key("name"), name);
        tensor_infos(m.key("tensor_info"), bk, infos);
        // Always false, and that is checked against the receivers: `persist`
        // and `conductor_new_token` are producer-side (read in route_outputs
        // on the SENDER, never off an arriving edge), `_persist_for_loop` is
        // set by the ingesting loop itself, and `_final_stream_chunk` is
        // stamped by the receiver's own StreamBuffer. Same as Python sends.
        w_bool(m.key("persist"), false);
        w_bool(m.key("conductor_new_token"), false);
        w_bool(m.key("is_streaming"), is_streaming);
        w_str(m.key("output_modality"), modality);
        w_bool(m.key("_persist_for_loop"), false);
        w_bool(m.key("_final_stream_chunk"), false);
        w_u64(m.key("_total_fanin"), total_fanin as u64);
        // `_shard_dim` defaults to None, and wire.py omits a
        // None-with-a-default, so a replicated signal must NOT carry the key.
        if let Some(d) = shard_dim {
            w_u64(m.key("_shard_dim"), d as u64);
        }
    });
}

/// One edge on an INPUT_SIGNALS frame.
pub struct OutEdge {
    pub name: String,
    pub next_node: String,
    pub is_streaming: bool,
    pub infos: Vec<TensorPointerInfo>,
    /// Both stamped by the sender's fanout, as Python does it. They tell the
    /// receiver how to put a sharded signal back together.
    pub shard_dim: Option<u32>,
    pub total_fanin: u32,
}

/// INPUT_SIGNALS to a peer worker. One frame per (request, worker): the plan
/// is per edge, and a worker taking several of a request's signals should see
/// one message.
pub struct InputSignals<'a> {
    pub request_id: &'a str,
    pub partition_name: &'a str,
    pub edges: &'a [OutEdge],
    /// `CurrentForwardPassInfo`, encoded by Python. Required with no default,
    /// so when absent it goes as an explicit nil rather than being omitted.
    pub request_info_encoded: Option<&'a [u8]>,
}

impl InputSignals<'_> {
    pub fn encode(&self, bk: &StrToId) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        open_frame(&mut out, "worker_msg", "input_signals",
                   Some("input_signals"));
        w_map_counted(&mut out, |m| {
            w_str(m.key("request_id"), self.request_id);
            w_arr(m.key("inputs"), self.edges.len() as u32);
            for e in self.edges {
                graph_edge(
                    m.buf(), bk, &e.name, &e.next_node, "", e.is_streaming,
                    &e.infos, e.shard_dim, e.total_fanin,
                );
            }
            match self.request_info_encoded {
                // Already encoded, so it is copied in whole -- msgpack is
                // self-delimiting, and nothing here has to understand it.
                Some(b) => w_raw(m.key("request_info"), b),
                // Required with no default, so an absent one is an explicit
                // nil rather than an omission.
                None => w_nil(m.key("request_info")),
            }
            w_str(m.key("partition_name"), self.partition_name);
            // A set, which msgpack cannot express; the codec sends a list.
            // Always empty: only the conductor ever populates it.
            w_arr(m.key("producer_done"), 0);
        });
        out
    }
}

/// STREAM_ITEMS to one peer worker: a step's streamed tensors as columns
/// (Python's `StreamItems`). Rows go in with `try_add`; a row whose
/// descriptors differ from the frame's shared fields is refused, and the
/// caller sends it as INPUT_SIGNALS.
pub struct StreamItems {
    partition_name: String,
    // (session, entity, node, tp_size, tp_rank), fixed by the first row
    shared: Option<(Sym, Sym, Option<Sym>, u32, u32)>,
    names: Vec<String>,
    name_at: FxHashMap<String, i64>,
    dtype_names: Vec<String>,
    dtype_at: FxHashMap<Sym, i64>,
    request_ids: Vec<String>,
    signal_idx: Vec<i64>,
    next_node_idx: Vec<i64>,
    walk_idx: Vec<i64>,
    num_items: Vec<i64>,
    uuids: Vec<u64>,
    addresses: Vec<u64>,
    nbytes: Vec<i64>,
    offsets: Vec<i64>,
    ndims: Vec<i64>,
    dims_flat: Vec<i64>,
    stride_flat: Vec<i64>,
    dtype_idx: Vec<i64>,
    shm_segment_idx: Vec<i64>,
    shm_offsets: Vec<i64>,
}

impl StreamItems {
    pub fn new(partition_name: &str) -> Self {
        StreamItems {
            partition_name: partition_name.to_string(),
            shared: None,
            names: Vec::new(), name_at: FxHashMap::default(),
            dtype_names: Vec::new(), dtype_at: FxHashMap::default(),
            request_ids: Vec::new(), signal_idx: Vec::new(), next_node_idx: Vec::new(),
            walk_idx: Vec::new(), num_items: Vec::new(),
            uuids: Vec::new(), addresses: Vec::new(), nbytes: Vec::new(), offsets: Vec::new(),
            ndims: Vec::new(), dims_flat: Vec::new(), stride_flat: Vec::new(),
            dtype_idx: Vec::new(), shm_segment_idx: Vec::new(), shm_offsets: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.request_ids.is_empty()
    }

    fn name_idx(&mut self, name: &str) -> i64 {
        if let Some(&i) = self.name_at.get(name) {
            return i;
        }
        let i = self.names.len() as i64;
        self.names.push(name.to_string());
        self.name_at.insert(name.to_string(), i);
        i
    }

    /// Add one (request, edge) row of streamed tensors; false, adding
    /// nothing, when its descriptors don't share the frame's fields (or
    /// each other's walk).
    pub fn try_add(
        &mut self, bk: &StrToId, request_id: &str, signal: &str, next_node: &str,
        infos: &[TensorPointerInfo],
    ) -> bool {
        let Some(first) = infos.first() else { return false };
        let shared = (
            first.source_session_id, first.source_entity, first.source_node_name,
            first.source_tp_size, first.source_tp_rank,
        );
        if self.shared.is_some_and(|s| s != shared) {
            return false;
        }
        let walk = first.source_graph_walk;
        if infos.iter().any(|i| {
            (i.source_session_id, i.source_entity, i.source_node_name, i.source_tp_size,
             i.source_tp_rank) != shared || i.source_graph_walk != walk
        }) {
            return false;
        }
        // only once the row is taken, as the Python builder: a refused first
        // row mustn't fix the frame's fields for the rows after it
        self.shared = Some(shared);
        self.request_ids.push(request_id.to_string());
        let sig = self.name_idx(signal);
        self.signal_idx.push(sig);
        let node = self.name_idx(next_node);
        self.next_node_idx.push(node);
        let w = walk.map_or(-1, |w| self.name_idx(bk.name(w)));
        self.walk_idx.push(w);
        self.num_items.push(infos.len() as i64);
        for i in infos {
            self.uuids.push(i.uuid);
            self.addresses.push(i.address);
            self.nbytes.push(i.nbytes);
            self.offsets.push(i.offset);
            self.ndims.push(i.dims.len() as i64);
            self.dims_flat.extend_from_slice(&i.dims);
            self.stride_flat.extend_from_slice(&i.stride);
            let n = self.dtype_names.len() as i64;
            let d = *self.dtype_at.entry(i.dtype).or_insert(n);
            if d == n {
                self.dtype_names.push(bk.name(i.dtype).to_string());
            }
            self.dtype_idx.push(d);
            let seg = i.shm_segment.map_or(-1, |s| self.name_idx(bk.name(s)));
            self.shm_segment_idx.push(seg);
            self.shm_offsets.push(i.shm_offset);
        }
        true
    }

    pub fn encode(&self, bk: &StrToId) -> Vec<u8> {
        let mut out = Vec::with_capacity(256 + 64 * self.uuids.len());
        open_frame(&mut out, "worker_msg", "stream_items", Some("stream_items"));
        let (session, entity, node, tp_size, tp_rank) =
            self.shared.expect("StreamItems::encode with no rows");
        fn ints(out: &mut Vec<u8>, v: &[i64]) {
            w_arr(out, v.len() as u32);
            for &x in v {
                w_i64(out, x);
            }
        }
        fn uints(out: &mut Vec<u8>, v: &[u64]) {
            w_arr(out, v.len() as u32);
            for &x in v {
                w_u64(out, x);
            }
        }
        fn strs(out: &mut Vec<u8>, v: &[String]) {
            w_arr(out, v.len() as u32);
            for x in v {
                w_str(out, x);
            }
        }
        w_map_counted(&mut out, |m| {
            w_str(m.key("partition_name"), &self.partition_name);
            w_str(m.key("source_session_id"), bk.name(session));
            w_str(m.key("source_entity"), bk.name(entity));
            // Omitted when absent, so the decoder's None default stands.
            if let Some(n) = node {
                w_str(m.key("source_node_name"), bk.name(n));
            }
            w_u64(m.key("tp_size"), tp_size as u64);
            w_u64(m.key("tp_rank"), tp_rank as u64);
            strs(m.key("names"), &self.names);
            strs(m.key("request_ids"), &self.request_ids);
            ints(m.key("signal_idx"), &self.signal_idx);
            ints(m.key("next_node_idx"), &self.next_node_idx);
            ints(m.key("walk_idx"), &self.walk_idx);
            ints(m.key("num_items"), &self.num_items);
            uints(m.key("uuids"), &self.uuids);
            uints(m.key("addresses"), &self.addresses);
            ints(m.key("nbytes"), &self.nbytes);
            ints(m.key("offsets"), &self.offsets);
            ints(m.key("ndims"), &self.ndims);
            ints(m.key("dims_flat"), &self.dims_flat);
            ints(m.key("stride_flat"), &self.stride_flat);
            ints(m.key("dtype_idx"), &self.dtype_idx);
            strs(m.key("dtype_names"), &self.dtype_names);
            ints(m.key("shm_segment_idx"), &self.shm_segment_idx);
            ints(m.key("shm_offsets"), &self.shm_offsets);
        });
        out
    }
}

/// One emitted output to the api server.
pub struct ResultTensors<'a> {
    pub request_id: &'a str,
    pub modality: &'a str,
    pub signal: &'a str,
    pub infos: Vec<TensorPointerInfo>,
    pub loop_indices: Option<(Vec<Sym>, Vec<(Sym, u32)>, u32)>,
}

impl ResultTensors<'_> {
    pub fn encode(&self, it: &StrToId, bk: &StrToId) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        // ResultTensors is APIServerMessage.body's DECLARED type, not an
        // abstract one, so the body rides untagged.
        open_frame(&mut out, "api_msg", "result_tensors", None);
        w_map_counted(&mut out, |m| {
            w_str(m.key("request_id"), self.request_id);
            w_str(m.key("modality"), self.modality);
            graph_edge(
                m.key("graph_edge"), bk, self.signal, EMIT_TO_CLIENT,
                self.modality, false, &self.infos, None, 1,
            );
            match &self.loop_indices {
                Some((order, idxs, fwd)) => {
                    nested_loop_indices(m.key("loop_indices"), it, order, idxs,
                                        *fwd)
                }
                None => w_nil(m.key("loop_indices")),
            }
            w_map(m.key("metadata"), 0);
        });
        out
    }
}

/// One step's inline emits to the api server: every request's value in one
/// frame, where ResultTensors was one frame (and one tensor) per request.
pub struct ResultTokens<'a> {
    pub request_ids: &'a [String],
    pub values: &'a [i64],
    pub loop_indices: &'a [Option<(Vec<Sym>, Vec<(Sym, u32)>, u32)>],
    pub signal: &'a str,
    pub modality: &'a str,
}

impl ResultTokens<'_> {
    pub fn encode(&self, it: &StrToId) -> Vec<u8> {
        let mut out = Vec::with_capacity(96 + 48 * self.request_ids.len());
        // Untagged like ResultTensors: a declared body type of APIServerMessage.
        open_frame(&mut out, "api_msg", "result_tokens", None);
        w_map_counted(&mut out, |m| {
            w_arr(m.key("request_ids"), self.request_ids.len() as u32);
            for r in self.request_ids {
                w_str(m.buf(), r);
            }
            w_arr(m.key("values"), self.values.len() as u32);
            for &v in self.values {
                w_i64(m.buf(), v);
            }
            w_arr(m.key("loop_indices"), self.loop_indices.len() as u32);
            for li in self.loop_indices {
                match li {
                    Some((order, idxs, fwd)) => {
                        nested_loop_indices(m.buf(), it, order, idxs, *fwd)
                    }
                    None => w_nil(m.buf()),
                }
            }
            w_str(m.key("signal"), self.signal);
            w_str(m.key("modality"), self.modality);
            w_map(m.key("metadata"), 0);
        });
        out
    }
}

/// Opens `[outer_tag, {message_type, body}]` and leaves the caller to write the
/// body, which is the next value due. `body` is tagged only where its declared
/// type is abstract.
///
/// Split from the body so nothing has to hold a whole frame in a `Value` tree
/// just to nest it one level down.
fn open_frame(
    out: &mut Vec<u8>, outer_tag: &str, message_type: &str,
    body_tag: Option<&str>,
) {
    w_arr(out, 2);
    w_str(out, outer_tag);
    // Fixed at two, and it opens rather than closes, so nothing is counted.
    w_map(out, 2);
    w_str(out, "message_type");
    w_str(out, message_type);
    w_str(out, "body");
    if let Some(t) = body_tag {
        w_arr(out, 2);
        w_str(out, t);
    }
}

/// STOP_LOOPS to a peer worker: which loops ended, and the loop context each
/// was observed to end at.
pub struct StopLoops<'a> {
    pub request_id: &'a str,
    pub partition_name: &'a str,
    pub loop_names: &'a [String],
    pub loop_stop_times: Vec<(Sym, (Vec<Sym>, Vec<(Sym, u32)>, u32))>,
}

impl StopLoops<'_> {
    pub fn encode(&self, it: &StrToId) -> Vec<u8> {
        let mut out = Vec::with_capacity(128);
        open_frame(&mut out, "worker_msg", "stop_loops", Some("stop_loops"));
        w_map_counted(&mut out, |m| {
            w_str(m.key("request_id"), self.request_id);
            // A set, which msgpack cannot express; the codec sends a list and
            // rebuilds it.
            w_arr(m.key("loop_names"), self.loop_names.len() as u32);
            for n in self.loop_names {
                w_str(m.buf(), n);
            }
            w_str(m.key("partition_name"), self.partition_name);
            w_map(m.key("loop_stop_times"), self.loop_stop_times.len() as u32);
            for (name, (order, idxs, fwd)) in &self.loop_stop_times {
                w_str(m.buf(), it.name(*name));
                nested_loop_indices(m.buf(), it, order, idxs, *fwd);
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of `w_map_counted`: the length comes from the writes.
    #[test]
    fn a_counted_map_reports_the_fields_it_was_given() {
        for present in [false, true] {
            let mut out = Vec::new();
            w_map_counted(&mut out, |m| {
                w_str(m.key("a"), "x");
                w_i64(m.key("b"), 7);
                if present {
                    w_bool(m.key("c"), true);
                }
            });
            let v = rmpv::decode::read_value(&mut &out[..]).unwrap();
            let Value::Map(entries) = v else { panic!("not a map") };
            assert_eq!(entries.len(), if present { 3 } else { 2 });
            // ...and the entries themselves survived the patched header.
            assert_eq!(entries[0].0.as_str(), Some("a"));
            assert_eq!(entries[1].1.as_i64(), Some(7));
        }
    }

    /// As Python's `StreamItemsBuilder.try_add`: a refused row leaves the
    /// frame as it was, so it doesn't decide which rows come after it.
    #[test]
    fn a_refused_first_row_does_not_fix_the_frames_fields() {
        let mut bk = StrToId::default();
        let info = |bk: &mut StrToId, entity: &str, walk: &str| TensorPointerInfo {
            dims: vec![1], dtype: bk.intern("int64"), nbytes: 8, address: 0, stride: vec![1],
            uuid: 1, source_session_id: bk.intern("h:1"), source_entity: bk.intern(entity),
            offset: 0, source_tp_size: 1, source_tp_rank: 0, shm_segment: None, shm_offset: 0,
            source_node_name: Some(bk.intern("LLM")), source_graph_walk: Some(bk.intern(walk)),
        };
        // two walks in one row: refused
        let mixed = [info(&mut bk, "w0", "a"), info(&mut bk, "w0", "b")];
        let other = [info(&mut bk, "w1", "a")];
        let first = [info(&mut bk, "w0", "a")];
        let mut frame = StreamItems::new("p");
        assert!(!frame.try_add(&bk, "r1", "s", "n", &mixed));
        assert!(frame.is_empty());
        assert!(frame.try_add(&bk, "r2", "s", "n", &other));
        assert!(!frame.try_add(&bk, "r3", "s", "n", &first));
    }

    /// A nested counted map must not disturb its parent's count.
    #[test]
    fn nesting_counted_maps_keeps_both_counts() {
        let mut out = Vec::new();
        w_map_counted(&mut out, |m| {
            w_str(m.key("outer"), "v");
            w_map_counted(m.key("inner"), |i| {
                w_i64(i.key("one"), 1);
                w_i64(i.key("two"), 2);
            });
        });
        let Value::Map(entries) =
            rmpv::decode::read_value(&mut &out[..]).unwrap()
        else {
            panic!("not a map")
        };
        assert_eq!(entries.len(), 2, "the nested map is ONE parent entry");
        let Value::Map(inner) = &entries[1].1 else { panic!("not nested") };
        assert_eq!(inner.len(), 2);
    }
}
