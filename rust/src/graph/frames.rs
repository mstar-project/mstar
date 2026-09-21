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
    /// `dict[str, PublishedInfo]` -- abstract, so it stays Python's.
    pub resource_publish_info_encoded: Option<&'a [u8]>,
    /// `graph_timings`, `rx_info`, `tx_info`: only populated under
    /// enable_prof, and they reach us as bytes already.
    pub graph_timings_encoded: Option<&'a [u8]>,
    pub rx_info_encoded: Option<&'a [u8]>,
    pub tx_info_encoded: Option<&'a [u8]>,
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
        for (key, bytes) in [
            ("resource_publish_info", self.resource_publish_info_encoded),
            ("graph_timings", self.graph_timings_encoded),
            ("rx_info", self.rx_info_encoded),
            ("tx_info", self.tx_info_encoded),
        ] {
            if let Some(b) = bytes {
                m.push((s(key), spliced(b)));
            }
        }
        Value::Map(m)
    }

    /// The full frame: a ConductorMessage wrapping this body. `body` is
    /// declared `MessageBody`, which is abstract, hence the `[tag, payload]`.
    pub fn encode(&self, it: &StrToId, bk: &StrToId) -> Vec<u8> {
        let frame = Value::Array(vec![
            s("conductor_msg"),
            Value::Map(vec![
                (s("message_type"), s("worker_graphs_done")),
                (
                    s("body"),
                    Value::Array(vec![s("wgs_done"), self.body(it, bk)]),
                ),
            ]),
        ]);
        let mut out = Vec::new();
        // Infallible for a Value tree with no unrepresentable node.
        rmpv::encode::write_value(&mut out, &frame).expect("msgpack encode");
        out
    }
}
