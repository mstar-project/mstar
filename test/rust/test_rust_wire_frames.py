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
            stream_tokens_consumed=[], profiling=[],
        )
        args.update(send_kwargs)
        _send(self.rt, out.completion_id, **args)
        return {peer: _collect(box) for peer, box in self.inboxes.items()}



def _send(rt, completion_id, **aos):
    """The runtime takes struct-of-arrays; these cases read better as pairs.

    Splits the (rid, value) lists the tests write into the parallel rid/value
    lists ``send_outputs`` wants, and hands the count payloads over as dicts.
    """
    def split(key):
        pairs = aos.get(key) or ()
        return [r for r, _ in pairs], [v for _, v in pairs]

    info_rids, infos = split("request_infos")
    ntc_rids, ntc = split("new_token_counts")
    con_rids, con = split("stream_tokens_consumed")
    prof_rids, prof = split("profiling")
    rt.send_outputs(
        completion_id=completion_id,
        info_rids=info_rids, request_infos=infos,
        ntc_rids=ntc_rids, new_token_counts=[dict(c) for c in ntc],
        consumed_rids=con_rids,
        stream_tokens_consumed=[dict(c) for c in con],
        prof_rids=prof_rids, profiling=prof,
    )


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
    context, and the signal name is remembered for the WGD that follows.

    The loop context is the runtime's own route-time snapshot -- nothing
    passes it in. This node is in no loop, so it is the pass index alone."""
    from mstar.api_server.request_types import APIServerMessage, ResultTensors

    mesh = _Mesh(tmp_path, outs=[("out", "emit_to_client", False)])
    rid = mesh.admit()
    info = _put(mesh, 1)
    idx = NestedLoopIndices(
        loop_name_order=[], loop_indices={}, wg_fwd_pass_idx=0,
    )
    got = mesh.run(rid, uuids=[1])

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
    _send(rt, out.completion_id, request_infos=[(rid, None)])
    return _collect(inbox, first_wait_ms=300)


def test_a_replicated_result_is_emitted_by_rank_0_only(tmp_path):
    """Every rank emitting means the api server gets the same result twice,
    and the duplicate lands after the request is gone -- "Message for unknown
    request". The client has no sharding group, so Python's fanout yields a
    destination only when source_tp_rank == 0.
    """
    assert len(_emit_mesh(tmp_path / "r0", tp_rank=0)) == 1
    assert _emit_mesh(tmp_path / "r1", tp_rank=1) == []


def _persist_mesh(tmp_path, tp_rank):
    """One node persisting to EMPTY_DESTINATION, inside a TP group of two."""
    book = RustTensorBookkeeping()
    comm = ZmqCommunicator(ME, str(tmp_path))
    inbox = ZmqCommunicator("conductor", str(tmp_path))
    wg = {"wg_id": WG_ID, "graph_walks": [WALK],
          "nodes": [_node("only", ["prompt"], [("out", "", True)])],
          "loops": []}
    rt = GraphRuntime(
        worker_graphs=[wg], remote_worker_graphs=[],
        sharding={
            "groups": [{"nodes": ["only"], "tp_size": 2,
                        "graph_walks": None, "tp_rank": tp_rank}],
            "shard_dim": [], "tp_enabled_nodes": [], "sp_enabled_nodes": [],
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
    _send(rt, out.completion_id, request_infos=[(rid, None)])
    return book, info, _collect(inbox, first_wait_ms=300)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_every_rank_reports_its_own_persist_signal(tp_rank, tmp_path):
    """Persist is NOT the fanout's business -- Python takes `to_conductor`
    straight off the node's outputs, before any of it.

    EMPTY_DESTINATION has no sharding group, so the replicated fanout gives it
    a pseudo-worker and emits it from rank 0 alone. Reading persist off the
    routed edges therefore loses every other rank's copy -- and the conductor
    fans the next walk's inputs back out PER SOURCE RANK, so a rank that never
    reported has nothing sent to it, never becomes ready, and sits on a
    ScheduleTPNode it can never pop (Orpheus tp2, prefill -> decode)."""
    _book, info, got = _persist_mesh(tmp_path, tp_rank=tp_rank)
    assert [m.body.persist_signals for m in got] == [{"out": [info]}]


def test_a_persist_signal_is_held_alive_on_every_rank(tmp_path):
    """The marker goes with the report. Taken off the routed edges it is not
    set on a rank whose persist edge the fanout dropped, and the settle from
    the safety hold then frees a tensor the conductor is about to ask for."""
    book, _info, _got = _persist_mesh(tmp_path, tp_rank=1)
    assert not book._rust.can_gc(1), "rank 1's persisted tensor was collected"


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


# --- the fanout reaches each destination worker once, with its own slice -----

P2 = "worker_2"


def _fanout_mesh(tmp_path, shard_dim=None):
    """A tp1 source feeding a tp2 destination group on two peers.

    The case the same-group edges never exercise: source and destination in
    DIFFERENT groups, so the fanout really does name more than one worker.
    """
    book = RustTensorBookkeeping()
    comm = ZmqCommunicator(ME, str(tmp_path))
    boxes = {p: ZmqCommunicator(p, str(tmp_path)) for p in (PEER, P2)}
    wg = {"wg_id": WG_ID, "graph_walks": [WALK], "loops": [],
          "nodes": [_node("src", ["prompt"], [("out", "sink", False)])]}
    rt = GraphRuntime(
        worker_graphs=[wg],
        remote_worker_graphs=[{"wg_id": 1, "graph_walks": [WALK],
                               "nodes": ["sink"], "dyn_loops": []}],
        sharding={
            "groups": [
                {"nodes": ["src"], "tp_size": 1,
                 "graph_walks": None, "tp_rank": 0},
                {"nodes": ["sink"], "tp_size": 2,
                 "graph_walks": None, "tp_rank": 0},
            ],
            "shard_dim": [] if shard_dim is None else [("out", shard_dim)],
            "tp_enabled_nodes": [], "sp_enabled_nodes": [],
        },
        bookkeeping=book._rust, me=ME, communicator=comm,
    )
    rid = rt.add_request("r1", "default", WALK, [WG_ID, 1], [ME, PEER, P2],
                         [1, 2])
    # 4 rows of 6 bytes.
    book.put_tensor(1, _info(1, dims=[4, 3], nbytes=24))
    book.increment_ref(1, 1)

    rt.ingest_inputs_batch([rid], [{
        "signal": "prompt", "next_node": "src", "uuids": [],
        "is_final_streaming_chunk": False,
    }])
    rt.pop_rids("src", WALK, [rid])
    out = rt.complete_and_route_batch({
        "partition": "default", "graph_walk": WALK, "node_name": "src",
        "output_signals": ["out"], "rids": [rid], "wg_ids": [WG_ID],
        "tensors": [1], "num_tensors": [1],
    })
    _send(rt, out.completion_id, request_infos=[(rid, None)])
    return book, {p: _collect(box, first_wait_ms=600) for p, box in boxes.items()}


def test_a_replicated_edge_reaches_each_worker_exactly_once(tmp_path):
    """The fanout names two workers, and take_send_plan expands per worker
    too. Doing both turns one edge into four sends -- which arrive as the
    signal listed twice in one frame, since frames group by (rid, worker)."""
    _book, got = _fanout_mesh(tmp_path)
    assert [e.name for e in got[PEER][0].body.inputs] == ["out"]
    assert [e.name for e in got[P2][0].body.inputs] == ["out"]


def test_each_destination_rank_gets_its_own_slice(tmp_path):
    """The slice the fanout computed has to reach the WIRE. The bookkeeper
    still holds the whole tensor, so a frame built from the uuid alone puts
    the unsliced dims back on."""
    _book, got = _fanout_mesh(tmp_path, shard_dim=0)
    lo = got[PEER][0].body.inputs[0].tensor_info[0]
    hi = got[P2][0].body.inputs[0].tensor_info[0]
    # 4 rows split two ways: 2 rows each, the second starting 12 bytes in.
    assert (lo.dims[0], lo.nbytes, lo.offset) == (2, 12, 0)
    assert (hi.dims[0], hi.nbytes, hi.offset) == (2, 12, 12)


def test_the_hold_settles_to_the_number_of_readers(tmp_path):
    """Two workers read it, so it takes two releases to free -- counting the
    post-fanout edges AND the per-worker expansion settles it to four, and
    the tensor is never collected."""
    book, _got = _fanout_mesh(tmp_path)
    releases = 0
    while not book._rust.can_gc(1) and releases < 8:
        book._rust.dereference(1, 1)
        releases += 1
    assert releases == 2


# --- what the receiver needs to reassemble a sharded arrival ----------------

def _gather_mesh(tmp_path, src_tp, dest_tp, my_rank=0, shard_dim=0,
                 src_first=True):
    """A src_tp -> dest_tp edge, viewed from source rank `my_rank`.

    ``src_first=False`` puts another node ahead of ``src`` in the worker
    graph, so its node id and its interned name are different numbers.
    """
    book = RustTensorBookkeeping()
    comm = ZmqCommunicator(ME, str(tmp_path))
    boxes = {p: ZmqCommunicator(p, str(tmp_path)) for p in (PEER, P2)}
    nodes = [_node("src", ["prompt"], [("out", "sink", False)])]
    if not src_first:
        nodes.insert(0, _node("pre", ["text"], [("prompt", "src", False)]))
    wg = {"wg_id": WG_ID, "graph_walks": [WALK], "loops": [], "nodes": nodes}
    rt = GraphRuntime(
        worker_graphs=[wg],
        remote_worker_graphs=[{"wg_id": 1, "graph_walks": [WALK],
                               "nodes": ["sink"], "dyn_loops": []}],
        sharding={
            "groups": [
                {"nodes": [n["name"] for n in nodes], "tp_size": src_tp,
                 "graph_walks": None, "tp_rank": my_rank},
                {"nodes": ["sink"], "tp_size": dest_tp,
                 "graph_walks": None, "tp_rank": 0},
            ],
            "shard_dim": [] if shard_dim is None else [("out", shard_dim)],
            "tp_enabled_nodes": [], "sp_enabled_nodes": [],
        },
        bookkeeping=book._rust, me=ME, communicator=comm,
    )
    srcs = [ME, PEER][:src_tp]
    dests = [PEER, P2][:dest_tp]
    rid = rt.add_request("r1", "default", WALK, [WG_ID, 1],
                         srcs + dests, [src_tp, dest_tp])
    book.put_tensor(1, _info(1, dims=[4, 3], nbytes=24))
    book.increment_ref(1, 1)
    rt.ingest_inputs_batch([rid], [{
        "signal": "prompt", "next_node": "src", "uuids": [],
        "is_final_streaming_chunk": False,
    }])
    rt.pop_rids("src", WALK, [rid])
    out = rt.complete_and_route_batch({
        "partition": "default", "graph_walk": WALK, "node_name": "src",
        "output_signals": ["out"], "rids": [rid], "wg_ids": [WG_ID],
        "tensors": [1], "num_tensors": [1],
    })
    _send(rt, out.completion_id, request_infos=[(rid, None)])
    return {p: _collect(box, first_wait_ms=600) for p, box in boxes.items()}


def test_a_gather_tells_the_receiver_to_wait_for_both_halves(tmp_path):
    """src tp2 -> dest tp1: the one destination is fed by BOTH source ranks.

    `tensors.py` only buffers shards when _total_fanin > 1, so left at 1 the
    destination takes whichever half lands first and drops the other --
    silently, with a full-looking tensor.
    """
    got = _gather_mesh(tmp_path, src_tp=2, dest_tp=1)
    edge = got[PEER][0].body.inputs[0]
    assert edge._total_fanin == 2
    assert edge._shard_dim == 0


@pytest.mark.parametrize("src_first", [True, False])
def test_only_source_rank_0_broadcasts_a_replicated_signal(tmp_path, src_first):
    """Rank 1 of a tp2 source sends a replicated signal only to its own
    worker; rank 0 is the one that reaches the rest of the destination group.

    The fanout finds the source's group by its interned NAME. It was handed
    the node id -- the same integer type -- which coincides with the name only
    for the first node a graph interns. For any other node the lookup missed,
    the source read as an unsharded rank 0, and every rank broadcast: the
    TP2 symptom was rank 1 of the destination getting the wrong copies.
    """
    got = _gather_mesh(tmp_path, src_tp=2, dest_tp=2, my_rank=1,
                       shard_dim=None, src_first=src_first)
    # Rank 1's own worker (PEER) is in the destination group, so it gets it.
    assert [e.name for e in got[PEER][0].body.inputs] == ["out"]
    assert got[P2] == [], "rank 1 broadcast as if it were rank 0"


def test_an_aligned_edge_has_no_fanin(tmp_path):
    """tp2 -> tp2: each destination rank is fed by exactly one source rank."""
    got = _gather_mesh(tmp_path, src_tp=2, dest_tp=2)
    assert got[PEER][0].body.inputs[0]._total_fanin == 1


def test_a_replicated_signal_carries_no_shard_dim(tmp_path):
    """`_shard_dim` defaults to None and wire.py omits a None-with-a-default,
    so the key must be absent rather than 0 -- 0 is a real dim."""
    got = _gather_mesh(tmp_path, src_tp=1, dest_tp=1, shard_dim=None)
    edge = got[PEER][0].body.inputs[0]
    assert edge._shard_dim is None
    assert edge._total_fanin == 1


# --- loop indices come from the runtime's own snapshot ------------------------

def _looping_emit_mesh(tmp_path):
    """A looping node that emits, with a real transport.

    The combination the other fixtures miss: `_Mesh` has no loop, `_loop_mesh`
    never completes a worker graph. Without both at once nothing exercises a
    loop-index-carrying frame built from the runtime's OWN snapshot, which is
    where a timing bug in that snapshot hides.
    """
    book = RustTensorBookkeeping()
    comm = ZmqCommunicator(ME, str(tmp_path))
    inboxes = {p: ZmqCommunicator(p, str(tmp_path))
               for p in ("conductor", "api_server", PEER)}
    wg = {
        "wg_id": WG_ID, "graph_walks": [WALK],
        "nodes": [_node("ar_decode", ["token"],
                        [("out", "emit_to_client", False),
                         ("token", "ar_decode", False)])],
        "loops": [{
            "name": "ar_loop", "max_iters": 8, "parent": None,
            "member_nodes": ["ar_decode"],
            # EMPTY_DESTINATION: routed nowhere. A destination no worker
            # runs is an error -- see test_an_output_nobody_runs_is_an_error.
            "outputs": [{"name": "token", "dest": "", "persist": False,
                         "new_token": False, "streaming": False,
                         "modality": ""}],
            "accumulated": [], "loop_back": [("token", "ar_decode")],
            "external_inputs": [("token", "ar_decode")],
        }],
    }
    rt = GraphRuntime(
        worker_graphs=[wg], remote_worker_graphs=[],
        sharding={"groups": [], "shard_dim": [],
                  "tp_enabled_nodes": [], "sp_enabled_nodes": []},
        bookkeeping=book._rust, me=ME, communicator=comm,
    )
    rid = rt.add_request("r1", "default", WALK, [WG_ID], [ME], [1])
    return rt, book, rid, inboxes


def _iterate(rt, book, rid, uuid, stop_first=False):
    """One pass of the loop, driven exactly as the worker drives it."""
    rt.ingest_inputs_batch([rid], [{
        "signal": "token", "next_node": "ar_decode", "uuids": [],
        "is_final_streaming_chunk": False,
    }])
    rt.pop_rids("ar_decode", WALK, [rid])
    book.put_tensor(uuid, _info(uuid))
    book.increment_ref(uuid, 1)
    if stop_first:
        rt.stop_loops_batched("default", WALK, "ar_decode", [rid], [["ar_loop"]])
    out = rt.complete_and_route_batch({
        "partition": "default", "graph_walk": WALK, "node_name": "ar_decode",
        "output_signals": ["out", "token"], "rids": [rid], "wg_ids": [WG_ID],
        "tensors": [uuid], "num_tensors": [1, 0],
    })
    _send(rt, out.completion_id, request_infos=[(rid, None)])


def test_the_emitted_frame_carries_the_runtimes_own_loop_snapshot(tmp_path):
    """No `nested` argument: the indices must come from Rust's snapshot."""
    rt, book, rid, inboxes = _looping_emit_mesh(tmp_path)
    _iterate(rt, book, rid, 1)
    got = _collect(inboxes["api_server"])
    assert len(got) == 1
    idx = got[0].body.loop_indices
    assert idx is not None, "the snapshot never reached the frame"
    assert idx.loop_indices == {"ar_loop": 0}, idx.loop_indices
    assert idx.loop_name_order == ["ar_loop"]


def test_the_loop_index_advances_with_the_iteration(tmp_path):
    """A stale snapshot would report the same index twice."""
    rt, book, rid, inboxes = _looping_emit_mesh(tmp_path)
    _iterate(rt, book, rid, 1)
    _iterate(rt, book, rid, 2)
    got = _collect(inboxes["api_server"])
    assert len(got) == 2
    seen = [g.body.loop_indices.loop_indices["ar_loop"] for g in got]
    assert seen == [0, 1], seen


def test_a_stop_in_the_same_pass_does_not_move_the_index(tmp_path):
    """A stop lands between the worker's check and the routing call.

    The frame must still report the iteration the pass actually ran at.
    ``register_loop_finish`` only raises ``finish_signal`` -- ``curr_iter``
    moves in ``advance_loop``, during the completion -- so the routing call is
    a valid place to snapshot. This pins that: if a stop ever starts advancing
    the counter, the snapshot has to move earlier and this goes red.
    """
    rt, book, rid, inboxes = _looping_emit_mesh(tmp_path)
    _iterate(rt, book, rid, 1)
    _collect(inboxes["api_server"])

    # Second pass, with a stop landing between the snapshot and the send.
    _iterate(rt, book, rid, 2, stop_first=True)
    got = _collect(inboxes["api_server"])
    assert len(got) == 1
    assert got[0].body.loop_indices.loop_indices == {"ar_loop": 1}, (
        "a stop moved the reported index; the snapshot needs to move earlier"
    )


def test_an_output_nobody_runs_is_an_error(tmp_path):
    """A destination no worker runs must raise, not panic.

    The fanout marks such a destination with a sentinel worker id. Carried
    through to the frame it indexes past the interner, and an index-out-of-
    bounds panic crosses the FFI boundary as a bare PanicException with no
    hint about the graph. Python raises ValueError from route_node_outputs
    naming the node; this should say the same thing.
    """
    mesh = _Mesh(tmp_path, outs=[("out", "nowhere", False)])
    rid = mesh.admit()
    _put(mesh, 1)
    mesh.rt.ingest_inputs_batch([rid], [{
        "signal": "prompt", "next_node": "only", "uuids": [],
        "is_final_streaming_chunk": False,
    }])
    mesh.rt.pop_rids("only", WALK, [rid])
    out = mesh.rt.complete_and_route_batch({
        "partition": "default", "graph_walk": WALK, "node_name": "only",
        "output_signals": ["out"], "rids": [rid], "wg_ids": [WG_ID],
        "tensors": [1], "num_tensors": [1],
    })
    with pytest.raises(ValueError, match="unknown node/graph walk.*nowhere"):
        _send(mesh.rt, out.completion_id, request_infos=[(rid, None)])
