"""Frames built in Rust have to decode to what Python would have built.

Not byte-equality: the receiver runs ``unpackb`` and looks fields up by name,
so map order and integer width are free. What must hold is that
``wire.decode`` of the Rust frame equals the message Python would have sent --
which is what every test here asserts.

These go through the real path: a completion is routed, ``send_outputs``
builds every frame and pushes it down the SHARED transport, and the test
reads them off a peer's inbox. Nothing is inspected before it has actually
travelled.

The fields whose types belong to Python (``request_info``,
``resource_publish_info``, the profiling trio) are handed over already encoded
and spliced in untouched, so Rust never learns ``PublishedInfo``. That splice
is the part most likely to go wrong, hence its own cases.
"""
import sys

sys.path.insert(0, ".")

import pytest
import torch

pytest.importorskip("mstar_rust")

from mstar_rust import GraphRuntime, ZmqCommunicator

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.communication.wire import decode, encode, encode_field, encode_fields
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)

WALK = "decode"
WG_ID = 0
ME = "worker_0"
PEER = "worker_1"


def _node(name, inputs, outs, modality=""):
    return {
        "name": name, "async_enabled": True, "inputs": sorted(inputs),
        "streaming_inputs": [],
        "outputs": [
            {"name": sig, "dest": dest, "persist": persist,
             "new_token": False, "streaming": False, "modality": modality}
            for sig, dest, persist in outs
        ],
    }


class _Mesh:
    """A runtime wired to a real transport, plus every peer's inbox."""

    def __init__(self, tmp_path, outs, remote=()):
        self.dir = str(tmp_path)
        self.book = RustTensorBookkeeping()
        self.comm = ZmqCommunicator(ME, self.dir)
        self.inboxes = {
            peer: ZmqCommunicator(peer, self.dir)
            for peer in ("conductor", "api_server", PEER)
        }
        wg = {"wg_id": WG_ID, "graph_walks": [WALK],
              "nodes": [_node("only", ["prompt"], outs)], "loops": []}
        self.rt = GraphRuntime(
            worker_graphs=[wg],
            remote_worker_graphs=list(remote),
            sharding={"groups": [], "shard_dim": [],
                      "tp_enabled_nodes": [], "sp_enabled_nodes": []},
            bookkeeping=self.book._rust, me=ME, communicator=self.comm,
        )

    def admit(self, wgs=(WG_ID,), workers=(ME,)):
        return self.rt.add_request(
            "r1", "default", WALK, list(wgs), list(workers),
            [1] * len(wgs),
        )

    def run(self, rid, uuids, **send_kwargs):
        """Ingest, pop, complete and send. Returns the frames each peer got."""
        self.rt.ingest_inputs_batch([rid], [{
            "signal": "prompt", "next_node": "only", "uuids": [],
            "is_final_streaming_chunk": False,
        }])
        self.rt.pop_rids("only", WALK, [rid])
        signals = self.rt.get_output_signals("only", WALK)
        out = self.rt.complete_and_route_batch({
            "partition": "default", "graph_walk": WALK, "node_name": "only",
            "output_signals": signals, "rids": [rid], "wg_ids": [WG_ID],
            "tensors": list(uuids),
            "num_tensors": [
                len(uuids) if s == signals[0] else 0 for s in signals
            ],
        })
        args = dict(
            request_infos=[(rid, None)], new_token_counts=[],
            nested=[], stream_tokens_consumed=[], profiling=[],
        )
        args.update(send_kwargs)
        self.rt.send_outputs(completion_id=out.completion_id, **args)
        return {peer: _collect(box) for peer, box in self.inboxes.items()}


def _collect(box, first_wait_ms=1000):
    """Everything queued on an inbox.

    A blocking first read, then drain: zmq delivery is asynchronous, so a bare
    drain() right after the send races it and returns nothing.
    """
    out = []
    first = box.recv_timeout(first_wait_ms)
    if first is None:
        return out
    out.append(decode(first))
    while (more := box.recv_timeout(50)) is not None:
        out.append(decode(more))
    return out


def _info(uuid, **over):
    base = dict(
        dims=[2, 3], dtype=torch.bfloat16, nbytes=12, address=0xDEAD,
        # a LIST: stride is declared list[int], so a round trip returns one
        # even though every caller passes a tuple.
        stride=[3, 1], uuid=uuid, source_session_id="h:1",
        source_entity=ME,
    )
    base.update(over)
    return TensorPointerInfo(**base)


def _put(mesh, uuid, **over):
    info = _info(uuid, **over)
    mesh.book.put_tensor(uuid, info)
    mesh.book.increment_ref(uuid, 1)  # the safety hold
    return info


def _sent(msg):
    """A message as it arrives, for comparing against one Python built."""
    return decode(encode(msg))


def test_a_completion_sends_the_conductor_a_worker_graphs_done(tmp_path):
    mesh = _Mesh(tmp_path, outs=[("out", "", False)])
    rid = mesh.admit()
    got = mesh.run(rid, uuids=[])

    assert got["conductor"] == [_sent(ConductorMessage(
        message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
        body=WorkerGraphsDone(
            request_id="r1", worker_graph_ids=[WG_ID], is_first_tp_rank=True,
            partition_name="default",
        ),
    ))]
    assert got["api_server"] == [] and got[PEER] == []


def test_an_emitted_signal_reaches_the_api_server(tmp_path):
    """``emit_to_client``: the frame carries the descriptors and the loop
    context, and the signal name is remembered for the WGD that follows."""
    from mstar.api_server.request_types import APIServerMessage, ResultTensors

    mesh = _Mesh(tmp_path, outs=[("out", "emit_to_client", False)])
    rid = mesh.admit()
    info = _put(mesh, 1)
    idx = NestedLoopIndices(
        loop_name_order=["ar"], loop_indices={"ar": 2}, wg_fwd_pass_idx=1,
    )
    got = mesh.run(rid, uuids=[1], nested=[(rid, (
        list(idx.loop_name_order), list(idx.loop_indices.items()),
        idx.wg_fwd_pass_idx,
    ))])

    assert got["api_server"] == [_sent(APIServerMessage(
        message_type="result_tensors",
        body=ResultTensors(
            request_id="r1", modality="", graph_edge=GraphEdge(
                name="out", next_node="emit_to_client", output_modality="",
                tensor_info=[info],
            ),
            loop_indices=idx, metadata={},
        ),
    ))]
    # The WGD reports what was emitted, and at which loop index.
    wgd = got["conductor"][0].body
    assert wgd.output_signal_names == ["out"]
    assert wgd.output_loop_indices == {"out": idx}


def test_a_remote_destination_gets_input_signals(tmp_path):
    """One frame per (request, worker), even for several signals."""
    mesh = _Mesh(
        tmp_path, outs=[("out", "remote", False)],
        remote=[{"wg_id": 1, "graph_walks": [WALK],
                 "nodes": ["remote"], "dyn_loops": []}],
    )
    rid = mesh.admit(wgs=(WG_ID, 1), workers=(ME, PEER))
    info = _put(mesh, 1)
    fwd = CurrentForwardPassInfo(
        request_id="r1", graph_walk=WALK, fwd_index=2, random_seed=0,
        max_tokens=8, partition_name="default",
    )
    got = mesh.run(rid, uuids=[1], request_infos=[
        (rid, encode_field(fwd, CurrentForwardPassInfo)),
    ])

    assert got[PEER] == [_sent(WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(
            request_id="r1",
            inputs=[GraphEdge(name="out", next_node="remote",
                              tensor_info=[info])],
            request_info=fwd, partition_name="default",
        ),
    ))]
    # The spliced CurrentForwardPassInfo survived without Rust owning it.
    assert got[PEER][0].body.request_info.fwd_index == 2


def test_a_persist_signal_rides_the_worker_graphs_done(tmp_path):
    """Buffered, not sent immediately: a persist signal must not race the
    message that announces it."""
    mesh = _Mesh(tmp_path, outs=[("out", "", True)])
    rid = mesh.admit()
    info = _put(mesh, 1)
    got = mesh.run(rid, uuids=[1])

    assert got["conductor"][0].body.persist_signals == {"out": [info]}
    # Descriptor symbols are interned in the BOOKKEEPER's table, not the
    # runtime's; the wrong one would silently swap a string.
    assert got["conductor"][0].body.persist_signals["out"][0].dtype is (
        torch.bfloat16
    )


def test_new_token_counts_accumulate_across_sends(tmp_path):
    """They are Python's (numel needs the tensors) but Rust buffers them, and
    a repeated signal adds up rather than overwriting."""
    mesh = _Mesh(tmp_path, outs=[("out", "emit_to_client", False)])
    rid = mesh.admit()
    _put(mesh, 1)
    mesh.run(rid, uuids=[1], new_token_counts=[(rid, [("tok", 2)])])
    _put(mesh, 2)
    got = mesh.run(rid, uuids=[2], new_token_counts=[(rid, [("tok", 3)])])

    # First send's WGD already carried 2 and drained; this one carries 3.
    assert got["conductor"][0].body.new_token_counts == {"tok": 3}
    # Emitted names accumulate the same way and drain the same way.
    assert got["conductor"][0].body.output_signal_names == ["out"]


def test_the_profiling_trio_splices_into_three_fields(tmp_path):
    """One blob holding [rx, tx, timings]; Rust splits it.

    encode_fields, not encode: a bare list has no wire tag, so encode would
    PICKLE the payload -- which Rust cannot splice and a Rust peer cannot
    decode.
    """
    from mstar.profile.format import GraphTiming, RxInfo, TxInfo

    timings = {("only", WALK): GraphTiming(
        node="only", graph_walk=WALK, exec_count=2, total_time=1.5,
        forward_time=1.0, preprocess_time=0.25, postprocess_time=0.25,
    )}
    rx = [RxInfo(edge_name="e", source_entity=PEER, dest_entity=ME,
                 count=1, num_bytes=64, time=0.5)]
    tx = [TxInfo(edge_name="e", source_entity=ME, count=1,
                 num_bytes=64, time=0.5)]
    blob = encode_fields(
        [rx, tx, timings],
        (list[RxInfo], list[TxInfo], dict[tuple[str, str], GraphTiming]),
    )

    mesh = _Mesh(tmp_path, outs=[("out", "", False)])
    rid = mesh.admit()
    body = mesh.run(rid, uuids=[], profiling=[(rid, blob)])["conductor"][0].body
    assert body.rx_info == rx and body.tx_info == tx
    # graph_timings is keyed by a TUPLE, which msgpack cannot express as a map
    # key -- the codec sends a list of entries and rebuilds it.
    assert body.graph_timings == timings


def test_a_malformed_profiling_blob_does_not_take_the_send_down(tmp_path):
    # Profiling is diagnostic. A bad blob must not cost the frame.
    mesh = _Mesh(tmp_path, outs=[("out", "", False)])
    rid = mesh.admit()
    body = mesh.run(rid, uuids=[], profiling=[(rid, b"\x90")])["conductor"][0].body
    assert body.rx_info == [] and body.tx_info == [] and body.graph_timings == {}


def test_resource_publish_info_keeps_its_subclass(tmp_path):
    """PublishedInfo is abstract, so owning it in Rust would mean a Rust
    change per resource. It splices instead."""
    from mstar.communication.wire import _TYPE_TO_TAG
    from mstar.engine.resources.base import PublishedInfo

    cls = next(c for c in _TYPE_TO_TAG
               if isinstance(c, type) and issubclass(c, PublishedInfo))
    fwd = CurrentForwardPassInfo(
        request_id="r1", graph_walk=WALK, fwd_index=0, random_seed=0,
        max_tokens=1, partition_name="default",
        resource_publish_info={"kv": cls()},
    )
    mesh = _Mesh(tmp_path, outs=[("out", "", False)])
    rid = mesh.admit()
    got = mesh.run(rid, uuids=[], request_infos=[
        (rid, encode_field(fwd, CurrentForwardPassInfo)),
    ])
    published = got["conductor"][0].body.resource_publish_info
    assert isinstance(published["kv"], cls)


def test_a_uuid_whose_descriptor_is_gone_is_skipped(tmp_path):
    # A tensor can be collected before the frame naming it goes out.
    mesh = _Mesh(tmp_path, outs=[("out", "", True)])
    rid = mesh.admit()
    info = _put(mesh, 1)
    mesh.book.forget_tensor(1)
    mesh.book.put_tensor(2, info)
    got = mesh.run(rid, uuids=[1])
    assert got["conductor"][0].body.persist_signals == {"out": []}


# --- stop loops ---------------------------------------------------------------

def _loop_mesh(tmp_path):
    """A dynamic loop owned by this rank and a peer."""
    book = RustTensorBookkeeping()
    comm = ZmqCommunicator(ME, str(tmp_path))
    inboxes = {p: ZmqCommunicator(p, str(tmp_path))
               for p in ("conductor", "api_server", PEER)}
    wg = {
        "wg_id": WG_ID, "graph_walks": [WALK],
        "nodes": [_node("ar_decode", ["token"], [("token", "ar_decode", False)])],
        "loops": [{
            "name": "ar_loop", "max_iters": 4, "parent": None,
            "member_nodes": ["ar_decode"],
            "outputs": [{"name": "token", "dest": "post", "persist": False,
                         "new_token": False, "streaming": False,
                         "modality": ""}],
            "accumulated": [], "loop_back": [("token", "ar_decode")],
            "external_inputs": [("token", "ar_decode")],
        }],
    }
    rt = GraphRuntime(
        worker_graphs=[wg],
        remote_worker_graphs=[{"wg_id": 1, "graph_walks": [WALK],
                               "nodes": ["ar_decode"],
                               "dyn_loops": ["ar_loop"]}],
        sharding={"groups": [], "shard_dim": [],
                  "tp_enabled_nodes": [], "sp_enabled_nodes": []},
        bookkeeping=book._rust, me=ME, communicator=comm,
    )
    rid = rt.add_request("r1", "default", WALK, [WG_ID, 1], [ME, PEER], [1, 1])
    return rt, rid, inboxes


def test_stopping_a_loop_tells_the_peers_that_run_it(tmp_path):
    rt, rid, inboxes = _loop_mesh(tmp_path)
    rt.stop_loops_batched("default", WALK, "ar_decode", [rid], [["ar_loop"]])

    got = _collect(inboxes[PEER])
    assert len(got) == 1
    msg = got[0]
    assert msg.message_type is WorkerMessageType.STOP_LOOPS
    assert msg.body.request_id == "r1"
    assert msg.body.loop_names == {"ar_loop"}
    assert msg.body.partition_name == "default"
    # The observation the peer needs to tell a newer stop from a duplicate.
    assert msg.body.loop_stop_times["ar_loop"].wg_fwd_pass_idx == 0

    # Never to ourselves: this rank already applied the stop, and a self-send
    # would land in apply_peer_loop_stops and stop it a second time.
    assert _collect(inboxes["conductor"], first_wait_ms=100) == []


def test_a_loop_nobody_else_runs_sends_nothing(tmp_path):
    rt, rid, inboxes = _loop_mesh(tmp_path)
    rt.stop_loops_batched("default", WALK, "ar_decode", [rid], [["nope"]])
    assert _collect(inboxes[PEER], first_wait_ms=100) == []


# --- a replicated result leaves from rank 0 only ------------------------------

def _emit_mesh(tmp_path, tp_rank, shard_dim=None):
    """One node emitting to the client, inside a TP/SP group of two."""
    book = RustTensorBookkeeping()
    comm = ZmqCommunicator(ME, str(tmp_path))
    inbox = ZmqCommunicator("api_server", str(tmp_path))
    wg = {"wg_id": WG_ID, "graph_walks": [WALK],
          "nodes": [_node("only", ["prompt"], [("out", "emit_to_client", False)])],
          "loops": []}
    rt = GraphRuntime(
        worker_graphs=[wg], remote_worker_graphs=[],
        sharding={
            "groups": [{"nodes": ["only"], "tp_size": 2,
                        "graph_walks": None, "tp_rank": tp_rank}],
            "shard_dim": [] if shard_dim is None else [("out", shard_dim)],
            "tp_enabled_nodes": [], "sp_enabled_nodes": [],
        },
        bookkeeping=book._rust, me=ME, communicator=comm,
    )
    rid = rt.add_request("r1", "default", WALK, [WG_ID], [ME, PEER], [2])
    info = TensorPointerInfo(
        dims=[2, 3], dtype=torch.bfloat16, nbytes=12, address=1,
        stride=[3, 1], uuid=1, source_session_id="h:1", source_entity=ME,
    )
    book.put_tensor(1, info)
    book.increment_ref(1, 1)
    rt.ingest_inputs_batch([rid], [{
        "signal": "prompt", "next_node": "only", "uuids": [],
        "is_final_streaming_chunk": False,
    }])
    rt.pop_rids("only", WALK, [rid])
    out = rt.complete_and_route_batch({
        "partition": "default", "graph_walk": WALK, "node_name": "only",
        "output_signals": ["out"], "rids": [rid], "wg_ids": [WG_ID],
        "tensors": [1], "num_tensors": [1],
    })
    rt.send_outputs(
        completion_id=out.completion_id, request_infos=[(rid, None)],
        new_token_counts=[], nested=[], stream_tokens_consumed=[], profiling=[],
    )
    return _collect(inbox, first_wait_ms=300)


def test_a_replicated_result_is_emitted_by_rank_0_only(tmp_path):
    """Every rank emitting means the api server gets the same result twice,
    and the duplicate lands after the request is gone -- "Message for unknown
    request". The client has no sharding group, so Python's fanout yields a
    destination only when source_tp_rank == 0.
    """
    assert len(_emit_mesh(tmp_path / "r0", tp_rank=0)) == 1
    assert _emit_mesh(tmp_path / "r1", tp_rank=1) == []


def test_a_sharded_result_is_emitted_by_every_rank(tmp_path):
    """The opposite case: each rank holds a different slice, so each sends
    its own -- gating on rank 0 there would drop half the output."""
    assert len(_emit_mesh(tmp_path / "s0", tp_rank=0, shard_dim=0)) == 1
    assert len(_emit_mesh(tmp_path / "s1", tp_rank=1, shard_dim=0)) == 1


def test_a_speculative_completion_does_not_report_the_partition_done(tmp_path):
    """A speculatively-scheduled node has not really finished the partition.

    Reported done, the conductor believes the stream ended a pass early --
    which is why Python carries `and not speculative`.
    """
    mesh = _Mesh(tmp_path, outs=[("out", "", False)])
    rid = mesh.admit()
    mesh.rt.mark_stream_partition_done(rid, "default")

    # Not speculative: the flag rides through as the stream reported it.
    assert mesh.run(rid, uuids=[])["conductor"][0].body.partition_done

    mesh.rt.set_speculatively_scheduled("only", WG_ID, [rid], True)
    body = mesh.run(rid, uuids=[])["conductor"][0].body
    assert not body.partition_done, "a speculative pass reported done"
