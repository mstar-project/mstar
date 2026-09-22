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

use rmpv::Value;

use crate::graph::compile::EMIT_TO_CLIENT;
use crate::graph::spec::{StrToId, Sym};
use crate::tensors::TensorPointerInfo;

fn s(v: &str) -> Value {
    Value::String(v.into())
}

/// A field Python already encoded. Decoded to a generic value so it can be
/// nested; re-serialising a generic value is what makes splicing exact.
fn spliced(bytes: &[u8]) -> Value {
    rmpv::decode::read_value(&mut &bytes[..]).unwrap_or(Value::Nil)
}

/// One key out of an already-encoded map.
///
/// `resource_publish_info` is a FIELD of CurrentForwardPassInfo, and Python
/// sends that whole object anyway -- so it is read back out of the same blob
/// rather than encoded a second time. Absent yields None, which omits the
/// field and leaves the decoder's default.
pub fn spliced_field(bytes: Option<&[u8]>, key: &str) -> Option<Value> {
    match spliced(bytes?) {
        Value::Map(m) => m
            .into_iter()
            .find(|(k, _)| k.as_str() == Some(key))
            .map(|(_, v)| v),
        _ => None,
    }
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
fn tensor_info(bk: &StrToId, i: &TensorPointerInfo) -> Value {
    // `bk`, not the runtime's interner: a descriptor's symbols were interned
    // by the BOOKKEEPER, which keeps its own table.
    let name = |sym: Sym| s(bk.name(sym));
    let mut m = vec![
        (s("dims"), Value::Array(i.dims.iter().map(|&d| d.into()).collect())),
        (s("dtype"), name(i.dtype)),
        (s("nbytes"), i.nbytes.into()),
        (s("address"), i.address.into()),
        (s("stride"), Value::Array(i.stride.iter().map(|&d| d.into()).collect())),
        (s("uuid"), i.uuid.into()),
        (s("source_session_id"), name(i.source_session_id)),
        (s("source_entity"), name(i.source_entity)),
        (s("offset"), i.offset.into()),
        (s("source_tp_size"), i.source_tp_size.into()),
        (s("source_tp_rank"), i.source_tp_rank.into()),
        (s("shm_offset"), i.shm_offset.into()),
    ];
    if let Some(seg) = i.shm_segment {
        m.push((s("shm_segment"), name(seg)));
    }
    if let Some(n) = i.source_node_name {
        m.push((s("_source_node_name"), name(n)));
    }
    if let Some(w) = i.source_graph_walk {
        m.push((s("_source_graph_walk"), name(w)));
    }
    Value::Map(m)
}

/// Python's `NestedLoopIndices`.
fn nested_loop_indices(
    it: &StrToId, order: &[Sym], indices: &[(Sym, u32)], fwd: u32,
) -> Value {
    Value::Map(vec![
        (
            s("loop_name_order"),
            Value::Array(order.iter().map(|&n| s(it.name(n))).collect()),
        ),
        (
            s("loop_indices"),
            Value::Map(
                indices.iter().map(|&(n, i)| (s(it.name(n)), i.into())).collect(),
            ),
        ),
        (s("wg_fwd_pass_idx"), fwd.into()),
    ])
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
    pub stream_tokens_consumed: &'a [(String, i64)],
    pub output_loop_indices: Vec<(Sym, (Vec<Sym>, Vec<(Sym, u32)>, u32))>,
    /// `dict[str, PublishedInfo]` -- abstract, so it stays Python's. Read
    /// out of the CurrentForwardPassInfo blob by `spliced_field`.
    pub resource_publish_info: Option<Value>,
    /// `rx_info`, `tx_info`, `graph_timings`, in that order: only populated
    /// under enable_prof, and already encoded when they reach us.
    pub profiling: [Option<Value>; 3],
}

fn str_counts(pairs: &[(String, i64)]) -> Value {
    Value::Map(pairs.iter().map(|(k, v)| (s(k), (*v).into())).collect())
}

impl WorkerGraphsDone<'_> {
    fn body(&self, it: &StrToId, bk: &StrToId) -> Value {
        let mut m = vec![
            (s("request_id"), s(self.request_id)),
            (
                s("worker_graph_ids"),
                Value::Array(self.worker_graph_ids.iter().map(|&w| w.into()).collect()),
            ),
            (s("is_first_tp_rank"), self.is_first_tp_rank.into()),
            (
                s("persist_signals"),
                Value::Map(
                    self.persist_signals
                        .iter()
                        .map(|(sig, infos)| {
                            (
                                s(it.name(*sig)),
                                Value::Array(
                                    infos.iter().map(|i| tensor_info(bk, i)).collect(),
                                ),
                            )
                        })
                        .collect(),
                ),
            ),
            (s("new_token_counts"), str_counts(self.new_token_counts)),
            (
                s("output_signal_names"),
                Value::Array(
                    self.output_signal_names.iter().map(|&n| s(it.name(n))).collect(),
                ),
            ),
            (s("partition_name"), s(self.partition_name)),
            (s("partition_done"), self.partition_done.into()),
            (
                s("stream_tokens_consumed"),
                str_counts(self.stream_tokens_consumed),
            ),
            (
                s("output_loop_indices"),
                Value::Map(
                    self.output_loop_indices
                        .iter()
                        .map(|(sig, (order, idxs, fwd))| {
                            (
                                s(it.name(*sig)),
                                nested_loop_indices(it, order, idxs, *fwd),
                            )
                        })
                        .collect(),
                ),
            ),
        ];
        if let Some(v) = &self.resource_publish_info {
            m.push((s("resource_publish_info"), v.clone()));
        }
        for (key, value) in
            ["rx_info", "tx_info", "graph_timings"].iter().zip(&self.profiling)
        {
            if let Some(v) = value {
                m.push((s(key), v.clone()));
            }
        }
        Value::Map(m)
    }

    /// The full frame: a ConductorMessage wrapping this body. `body` is
    /// declared `MessageBody`, which is abstract, hence the `[tag, payload]`.
    pub fn encode(&self, it: &StrToId, bk: &StrToId) -> Vec<u8> {
        encode_frame(
            "conductor_msg", "worker_graphs_done", Some("wgs_done"),
            self.body(it, bk),
        )
    }
}

/// One `GraphEdge`. Every field has a non-None default, so none is omitted --
/// including `output_modality`, which defaults to `""` rather than nil.
fn graph_edge(
    bk: &StrToId,
    name: &str,
    next_node: &str,
    modality: &str,
    is_streaming: bool,
    infos: &[TensorPointerInfo],
    shard_dim: Option<u32>,
    total_fanin: u32,
) -> Value {
    let mut m = vec![
        (s("next_node"), s(next_node)),
        (s("name"), s(name)),
        (
            s("tensor_info"),
            Value::Array(infos.iter().map(|i| tensor_info(bk, i)).collect()),
        ),
        (s("persist"), false.into()),
        (s("conductor_new_token"), false.into()),
        (s("is_streaming"), is_streaming.into()),
        (s("output_modality"), s(modality)),
        (s("_persist_for_loop"), false.into()),
        (s("_final_stream_chunk"), false.into()),
        (s("_total_fanin"), total_fanin.into()),
    ];
    // `_shard_dim` defaults to None, and wire.py omits a None-with-a-default,
    // so a replicated signal must NOT carry the key at all.
    if let Some(d) = shard_dim {
        m.push((s("_shard_dim"), d.into()));
    }
    Value::Map(m)
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
        let body = Value::Map(vec![
            (s("request_id"), s(self.request_id)),
            (
                s("inputs"),
                Value::Array(
                    self.edges
                        .iter()
                        .map(|e| {
                            graph_edge(
                                bk, &e.name, &e.next_node, "", e.is_streaming,
                                &e.infos, e.shard_dim, e.total_fanin,
                            )
                        })
                        .collect(),
                ),
            ),
            (
                s("request_info"),
                match self.request_info_encoded {
                    Some(b) => spliced(b),
                    None => Value::Nil,
                },
            ),
            (s("partition_name"), s(self.partition_name)),
            // A set, which msgpack cannot express; the codec sends a list.
            (s("producer_done"), Value::Array(vec![])),
        ]);
        encode_frame("worker_msg", "input_signals", Some("input_signals"), body)
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
        let body = Value::Map(vec![
            (s("request_id"), s(self.request_id)),
            (s("modality"), s(self.modality)),
            (
                s("graph_edge"),
                graph_edge(
                    bk, self.signal, EMIT_TO_CLIENT, self.modality, false,
                    &self.infos, None, 1,
                ),
            ),
            (
                s("loop_indices"),
                match &self.loop_indices {
                    Some((order, idxs, fwd)) => {
                        nested_loop_indices(it, order, idxs, *fwd)
                    }
                    None => Value::Nil,
                },
            ),
            (s("metadata"), Value::Map(vec![])),
        ]);
        // ResultTensors is APIServerMessage.body's DECLARED type, not an
        // abstract one, so the body rides untagged.
        encode_frame("api_msg", "result_tensors", None, body)
    }
}

/// `[outer_tag, {message_type, body}]`, with `body` tagged only where its
/// declared type is abstract.
fn encode_frame(
    outer_tag: &str, message_type: &str, body_tag: Option<&str>, body: Value,
) -> Vec<u8> {
    let body = match body_tag {
        Some(t) => Value::Array(vec![s(t), body]),
        None => body,
    };
    let frame = Value::Array(vec![
        s(outer_tag),
        Value::Map(vec![
            (s("message_type"), s(message_type)),
            (s("body"), body),
        ]),
    ]);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &frame).expect("msgpack encode");
    out
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
        let body = Value::Map(vec![
            (s("request_id"), s(self.request_id)),
            // A set, which msgpack cannot express; the codec sends a list and
            // rebuilds it.
            (
                s("loop_names"),
                Value::Array(self.loop_names.iter().map(|n| s(n)).collect()),
            ),
            (s("partition_name"), s(self.partition_name)),
            (
                s("loop_stop_times"),
                Value::Map(
                    self.loop_stop_times
                        .iter()
                        .map(|(name, (order, idxs, fwd))| {
                            (s(it.name(*name)), nested_loop_indices(it, order, idxs, *fwd))
                        })
                        .collect(),
                ),
            ),
        ]);
        encode_frame("worker_msg", "stop_loops", Some("stop_loops"), body)
    }
}
