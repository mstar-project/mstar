"""Frames built in Rust have to decode to what Python would have built.

Not byte-equality: the receiver runs ``unpackb`` and looks fields up by name,
so map order and integer width are free. What must hold is that
``wire.decode`` of the Rust frame equals the message Python would have sent --
which is what every test here asserts.

The fields whose types belong to Python (``resource_publish_info``, and the
profiling trio) are handed over already encoded and spliced in untouched, so
Rust never learns ``PublishedInfo``. That splice is the part most likely to
go wrong, hence its own cases.
"""
import sys

sys.path.insert(0, ".")

import pytest
import torch

pytest.importorskip("mstar_rust")

from mstar_rust import GraphRuntime

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.communication.wire import decode, encode, encode_field
from mstar.graph.base import TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    WorkerGraphsDone,
)

WALK = "decode"


def _runtime(book):
    rt = GraphRuntime(
        worker_graphs=[], remote_worker_graphs=[],
        sharding={"groups": [], "shard_dim": [],
                  "tp_enabled_nodes": [], "sp_enabled_nodes": []},
        bookkeeping=book._rust, me="worker_0",
    )
    return rt


def _info(uuid, **over):
    base = dict(
        dims=[2, 3], dtype=torch.bfloat16, nbytes=12, address=0xDEAD,
        stride=(3, 1), uuid=uuid, source_session_id="h:1",
        source_entity="worker_0",
    )
    base.update(over)
    return TensorPointerInfo(**base)


def _emit(rt, rid, **over):
    args = dict(
        worker_graph_ids=[0, 2], is_first_tp_rank=True,
        partition_name="default", partition_done=False,
        persist_uuids=[], new_token_counts=[], output_signal_names=[],
        stream_tokens_consumed=[], output_loop_indices=[],
    )
    args.update(over)
    # send=False: these runtimes have no transport, and the frame is the
    # thing under test.
    return rt.emit_worker_graphs_done(rid=rid, send=False, **args)


def _expected(**over):
    """What Python would have SENT, decoded.

    Decoded, not the object handed in: the codec is lossy in ways that are
    fine but real -- ``stride`` is declared list[int] and callers pass a
    tuple, so a round trip returns a list. Comparing to the pre-encode object
    would fail Python against itself.
    """
    body = dict(
        request_id="r1", worker_graph_ids=[0, 2], is_first_tp_rank=True,
        partition_name="default", partition_done=False,
    )
    body.update(over)
    return decode(encode(ConductorMessage(
        message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
        body=WorkerGraphsDone(**body),
    )))


def test_a_rust_frame_decodes_like_the_python_one():
    book = RustTensorBookkeeping()
    rt = _runtime(book)
    rid = rt.add_request("r1", "default", WALK, [], [], [])
    assert decode(_emit(rt, rid)) == _expected()


def test_the_scalar_fields_survive():
    book = RustTensorBookkeeping()
    rt = _runtime(book)
    rid = rt.add_request("r1", "default", WALK, [], [], [])
    got = decode(_emit(
        rt, rid, worker_graph_ids=[7], is_first_tp_rank=False,
        partition_name="audio", partition_done=True,
        new_token_counts=[("tok", 3), ("other", 5)],
        output_signal_names=["a", "b"],
        stream_tokens_consumed=[("edge", 9)],
    ))
    assert got == _expected(
        worker_graph_ids=[7], is_first_tp_rank=False,
        partition_name="audio", partition_done=True,
        new_token_counts={"tok": 3, "other": 5},
        output_signal_names=["a", "b"],
        stream_tokens_consumed={"edge": 9},
    )


def test_persist_descriptors_come_out_of_the_bookkeeper():
    """Rust resolves uuids to descriptors itself -- it shares the bookkeeper.

    The descriptor's symbols were interned by the BOOKKEEPER, not the runtime,
    so this also pins that the right table is used: a mix-up would silently
    swap one string for another.
    """
    book = RustTensorBookkeeping()
    rt = _runtime(book)
    rid = rt.add_request("r1", "default", WALK, [], [], [])
    full = _info(7, offset=16, source_tp_size=4, source_tp_rank=2,
                 shm_segment="seg-3", shm_offset=64,
                 _source_node_name="prefill", _source_graph_walk="decode")
    bare = _info(8)
    book.put_tensor(7, full)
    book.put_tensor(8, bare)

    got = decode(_emit(rt, rid, persist_uuids=[("kv", [7, 8])]))
    assert got == _expected(persist_signals={"kv": [full, bare]})
    # An absent segment must stay None, not become the string "None": the
    # consumer reads None as "spilled to a file".
    assert got.body.persist_signals["kv"][1].shm_segment is None
    assert got.body.persist_signals["kv"][0].dtype is torch.bfloat16


def test_a_uuid_whose_descriptor_is_gone_is_skipped():
    # A tensor can be collected before the WGD that names it goes out; the
    # Python shim's _infos drops those and so must this.
    book = RustTensorBookkeeping()
    rt = _runtime(book)
    rid = rt.add_request("r1", "default", WALK, [], [], [])
    book.put_tensor(7, _info(7))
    got = decode(_emit(rt, rid, persist_uuids=[("kv", [7, 999])]))
    assert [i.uuid for i in got.body.persist_signals["kv"]] == [7]


def test_nested_loop_indices_survive():
    book = RustTensorBookkeeping()
    rt = _runtime(book)
    rid = rt.add_request("r1", "default", WALK, [], [], [])
    got = decode(_emit(rt, rid, output_loop_indices=[
        ("out", (["outer", "inner"], [("outer", 1), ("inner", 2)], 5)),
    ]))
    assert got == _expected(output_loop_indices={
        "out": NestedLoopIndices(
            loop_name_order=["outer", "inner"],
            loop_indices={"outer": 1, "inner": 2}, wg_fwd_pass_idx=5,
        ),
    })


# --- the splice ---------------------------------------------------------------

def test_a_spliced_polymorphic_field_keeps_its_subclass():
    """``resource_publish_info`` stays Python's: PublishedInfo is abstract, so
    owning it in Rust would mean a Rust change per resource."""
    from mstar.communication.wire import _TYPE_TO_TAG
    from mstar.engine.resources.base import PublishedInfo

    cls = next(c for c in _TYPE_TO_TAG
               if isinstance(c, type) and issubclass(c, PublishedInfo))
    value = {"kv": cls()}

    book = RustTensorBookkeeping()
    rt = _runtime(book)
    rid = rt.add_request("r1", "default", WALK, [], [], [])
    got = decode(_emit(
        rt, rid,
        resource_publish_info=encode_field(value, dict[str, PublishedInfo]),
    ))
    assert got == _expected(resource_publish_info=value)
    assert isinstance(got.body.resource_publish_info["kv"], cls)


def test_the_profiling_trio_splices_too():
    """Today's shim decodes these bytes only for the frame to re-encode them.
    Spliced, they never round-trip through Python objects at all."""
    from mstar.profile.format import GraphTiming, RxInfo, TxInfo

    timings = {("prefill", WALK): GraphTiming(
        node="prefill", graph_walk=WALK, exec_count=2, total_time=1.5,
        forward_time=1.0, preprocess_time=0.25, postprocess_time=0.25,
    )}
    rx = [RxInfo(edge_name="e", source_entity="w1", dest_entity="w0",
                 count=1, num_bytes=64, time=0.5)]
    tx = [TxInfo(edge_name="e", source_entity="w0", count=1,
                 num_bytes=64, time=0.5)]

    book = RustTensorBookkeeping()
    rt = _runtime(book)
    rid = rt.add_request("r1", "default", WALK, [], [], [])
    got = decode(_emit(
        rt, rid,
        graph_timings=encode_field(timings, dict[tuple[str, str], GraphTiming]),
        rx_info=encode_field(rx, list[RxInfo]),
        tx_info=encode_field(tx, list[TxInfo]),
    ))
    # graph_timings is keyed by a TUPLE, which msgpack cannot express as a map
    # key -- the codec turns it into a list of entries and rebuilds it.
    assert got == _expected(graph_timings=timings, rx_info=rx, tx_info=tx)


def test_an_omitted_splice_leaves_the_default():
    book = RustTensorBookkeeping()
    rt = _runtime(book)
    rid = rt.add_request("r1", "default", WALK, [], [], [])
    got = decode(_emit(rt, rid))
    assert got.body.resource_publish_info == {}
    assert got.body.graph_timings == {}
    assert got.body.rx_info == [] and got.body.tx_info == []


def test_an_unknown_rid_is_refused():
    book = RustTensorBookkeeping()
    rt = _runtime(book)
    with pytest.raises(ValueError, match="unknown rid"):
        _emit(rt, 9999)
