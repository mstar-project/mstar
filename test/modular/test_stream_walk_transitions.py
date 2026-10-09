"""Producer-triggered graph-walk transitions.

A connection with ``consumer_walk`` lets the producer decide its consumer
partition's walk: the producer worker assigns each emitting pass's streamed
items a consumer walk, a chunk never spans two assigned walks, and the consumer
switches walk before popping a chunk, but only when the partition is idle.
Conductor inputs for a walk the stream has not reached yet wait on the worker.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
    PartitionDefinition,
    PartitionState,
)
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.graph_io import WorkerGraphIO
from mstar.model.qwen3_omni.qwen3_omni_model import _talker_walk
from mstar.streaming.chunk_policy import FixedChunkPolicy, LeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.streaming.topology import Connection, PartitionTopology, ProducerWalkCtx
from mstar.utils.ipc_format import InputSignals, WorkerGraphsDone
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.worker import Worker


def _buffer(chunk_size=1, tagged=True):
    return StreamBuffer(
        request_id=0, edge_name="thinker_states", from_partition="Thinker",
        policy=FixedChunkPolicy(chunk_size, continue_after_done=True),
        walk_tagged=tagged,
    )


def _feed(sbuf, walks):
    for i, walk in enumerate(walks):
        uuid = f"u{sbuf._num_tensors_registered}-{i}"
        sbuf.pre_read_register(uuid, walk)
        sbuf.put(uuid, torch.tensor([i]))


def _drain(sbuf):
    """(walk, items) per chunk, as the consumer would see them."""
    out = []
    while sbuf.has_chunk_ready() and sbuf._buffer:
        walk = sbuf.peek_walk()
        chunk = sbuf.pop_chunk()
        out.append((walk, chunk.num_items))
    return out


# --- StreamBuffer ------------------------------------------------------------

def test_a_chunk_never_spans_two_assigned_walks():
    sbuf = _buffer(chunk_size=3)
    _feed(sbuf, ["talker_prefill", "talker_prefill", "talker_last_prefill",
                 "talker_decode", "talker_decode", "talker_decode"])
    assert _drain(sbuf) == [
        ("talker_prefill", 2), ("talker_last_prefill", 1), ("talker_decode", 3),
    ]


def test_an_incomplete_run_waits_for_more_items():
    sbuf = _buffer(chunk_size=3)
    _feed(sbuf, ["talker_decode", "talker_decode"])
    assert not sbuf.has_chunk_ready()
    assert sbuf.peek_walk() == "talker_decode"


def test_empty_chunks_after_the_producer_ends_come_only_in_the_listed_walks():
    sbuf = StreamBuffer(
        request_id=0, edge_name="thinker_states", from_partition="Thinker",
        policy=FixedChunkPolicy(1, continue_after_done=frozenset({"talker_decode"})),
        walk_tagged=True,
    )
    sbuf.signal_done()
    # A prefill pass on an empty chunk would have nothing to run on
    assert not sbuf.has_chunk_ready("talker_last_prefill")
    assert sbuf.has_chunk_ready("talker_decode")
    assert not sbuf.pop_chunk().is_final


def test_an_untagged_buffer_is_unchanged():
    sbuf = _buffer(chunk_size=3, tagged=False)
    _feed(sbuf, ["a", "a", "b", "b"])
    assert sbuf.peek_walk() is None
    assert _drain(sbuf) == [(None, 3)]


# --- topology checks ---------------------------------------------------------

def _conn(edge, policy=None, walk=_talker_walk, producer="Thinker"):
    return Connection(
        from_partition=producer, to_partition="Talker", edge_name=edge,
        chunk_policy_factory=policy or (lambda: FixedChunkPolicy(1)),
        consumer_walk=walk,
    )


@pytest.mark.parametrize("other", [
    _conn("thinker_mask", walk=None),
    _conn("thinker_mask", walk=lambda ctx: "talker_prefill"),
    _conn("thinker_mask", producer="Encoder"),
])
def test_one_producer_and_one_function_decide_a_partitions_walk(other):
    topology = PartitionTopology(
        partitions=["Thinker", "Encoder", "Talker"],
        connections=[_conn("thinker_states"), other],
    )
    assert topology.producer_triggered_partitions() == {"Talker"}
    with pytest.raises(ValueError, match="thinker_mask"):
        topology.check_walk_driving_connections()


def test_a_walk_driving_edge_needs_non_overlapping_chunks():
    topology = PartitionTopology(
        partitions=["Thinker", "Talker"],
        connections=[_conn(
            "thinker_states", policy=lambda: LeftContextChunkPolicy(4, 2),
        )],
    )
    with pytest.raises(ValueError, match="LeftContextChunkPolicy"):
        topology.check_walk_driving_connections()


# --- idle --------------------------------------------------------------------

def test_a_worker_graph_is_idle_again_once_its_pass_resets_it():
    wgio = WorkerGraphIO(GraphNode(
        name="Talker", input_names={"thinker_states"}, outputs=[],
    ))
    assert wgio.is_idle()
    wgio.ingest_input(GraphEdge(next_node="Talker", name="thinker_states"))
    assert not wgio.is_idle()
    wgio.mark_node_complete("Talker")
    wgio.clear()
    assert wgio.is_idle()


# --- worker ------------------------------------------------------------------

def _fwd(walk, partition="Talker"):
    return CurrentForwardPassInfo(
        request_id="r", graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=0, partition_name=partition,
    )


def _worker(idle=True):
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w._draining_rids = set()
    w._producer_triggered_partitions = {"Talker"}
    w._walk_driven_edges = {"thinker_states": "Talker"}
    w.request_state = RequestStateManager(node_to_partition={"Talker": "Talker"})
    w.request_state.add_request(0, _fwd("talker_prefill"))
    w.walks = []
    w._graph_runtime = SimpleNamespace(
        get_rid_handle=lambda rid: 0,
        is_partition_idle=lambda rid, partition: idle,
        set_walk=lambda rid, partition, walk: w.walks.append(walk),
    )
    return w


def _conductor_input(walk):
    return InputSignals(
        request_id="r",
        inputs=[GraphEdge(next_node="Talker", name="talker_input_embeds")],
        request_info=_fwd(walk), partition_name="Talker",
    )


def test_a_conductor_input_for_a_later_walk_waits_for_the_stream():
    w = _worker()
    body = _conductor_input("talker_decode")
    w._process_new_inputs(body)
    assert w.request_state.per_request_info[0].parked_inputs == [body]

    replayed = []
    w._process_new_inputs = replayed.append
    w._switch_partition_walk(0, "Talker", "talker_last_prefill")
    assert replayed == []  # still not its walk
    w._switch_partition_walk(0, "Talker", "talker_decode")
    assert replayed == [body]
    assert w.request_state.per_request_info[0].parked_inputs == []
    assert w.walks == ["talker_last_prefill", "talker_decode"]


@pytest.mark.parametrize("queued, idle, released", [
    (False, True, True),
    (True, True, False),    # the stream decides first
    (False, False, False),  # mid-pass
])
def test_a_parked_input_applies_once_the_stream_has_nothing_queued(queued, idle, released):
    """A one-token reply: the Thinker's only decode pass ran talker_last_prefill
    and nothing will assign talker_decode, so the conductor's input moves it."""
    w = _worker(idle=idle)
    w._switch_partition_walk(0, "Talker", "talker_last_prefill")
    req_info = w.request_state.per_request_info[0]
    sbuf = _buffer()
    req_info.stream_buffers["thinker_states"] = sbuf
    if queued:
        _feed(sbuf, ["talker_decode"])
    body = _conductor_input("talker_decode")
    w._process_new_inputs(body)
    assert req_info.parked_inputs == [body]

    replayed = []
    w._process_new_inputs = replayed.append
    w._release_parked_inputs(0, req_info)
    assert (replayed == [body]) is released
    walk = w.request_state.get_fwd_info(0, "Talker").graph_walk
    assert walk == ("talker_decode" if released else "talker_last_prefill")


@pytest.mark.parametrize("idle, allow, switched", [
    (True, True, True),
    (False, True, False),   # mid-pass: the chunk waits
    (True, False, False),   # speculation: the step was built for this walk
])
def test_a_chunk_for_another_walk_switches_only_an_idle_partition(idle, allow, switched):
    w = _worker(idle=idle)
    sbuf = _buffer()
    _feed(sbuf, ["talker_last_prefill"])
    assert w._enter_chunk_walk(sbuf, "thinker_states", 0, allow) is switched
    walk = w.request_state.get_fwd_info(0, "Talker").graph_walk
    assert walk == ("talker_last_prefill" if switched else "talker_prefill")


def test_a_chunk_for_the_current_walk_needs_no_switch():
    w = _worker(idle=False)
    sbuf = _buffer()
    _feed(sbuf, ["talker_prefill"])
    assert w._enter_chunk_walk(sbuf, "thinker_states", 0, allow_walk_change=False)
    assert w.walks == []


# --- producer ----------------------------------------------------------------

_EMITS = {
    ("Thinker", "prefill_text"), ("Thinker", "prefill_audio"),
    ("Thinker", "thinker_decode"),
}


def _producer():
    """A Thinker worker with Qwen3-Omni's two walk-driving edges."""
    topology = PartitionTopology(
        partitions=["Thinker", "Talker"],
        connections=[_conn("thinker_states"), _conn("thinker_mask")],
    )
    w = Worker.__new__(Worker)
    w.partition_topology = topology
    w._walk_drivers = topology.walk_drivers()
    w._signal_consumer = {c.edge_name: c.to_partition for c in topology.connections}
    w._emitted_walk_drivers = {}
    w._graph_runtime = SimpleNamespace(
        get_output_signals=lambda node, walk: (
            ["thinker_mask", "thinker_states"] if (node, walk) in _EMITS else ["audio_embeds"]
        ),
    )
    w.request_state = RequestStateManager()
    w.request_state.add_request(0, _fwd("prefill_text", partition="Thinker"))
    return w


def _pass(w, node, walk):
    batch = SimpleNamespace(node_name=node, graph_walk=walk, partition="Thinker")
    return w._assign_consumer_walks(batch, [0]).get(0)


def test_the_thinker_walks_the_talker_through_its_prefill():
    """The Talker needs no count of the Thinker's prefill walks: the Thinker,
    which knows when its prefill is over, assigns each pass's states a walk."""
    w = _producer()
    passes = [
        ("Thinker", "prefill_text"),
        ("audio_encoder", "prefill_audio"),  # emits nothing to the Talker
        ("Thinker", "prefill_audio"), ("Thinker", "prefill_text"),
        ("Thinker", "thinker_decode"), ("Thinker", "thinker_decode"),
        ("Thinker", "thinker_decode"),
    ]
    got = [_pass(w, node, walk) for node, walk in passes]
    assert [g and g["Talker"] for g in got] == [
        "talker_prefill", None, "talker_prefill", "talker_prefill",
        "talker_last_prefill", "talker_decode", "talker_decode",
    ]
    # Stamped on the fwd_info that rides with the remote sends
    fwd = w.request_state.get_fwd_info(0, "Thinker")
    assert fwd.stream_consumer_walks == {"Talker": "talker_decode"}


def test_the_hook_sees_the_producers_state():
    seen = []

    def record(ctx: ProducerWalkCtx) -> str:
        seen.append(ctx)
        return "talker_prefill"

    w = _producer()
    for conn in w.partition_topology.connections:
        conn.consumer_walk = record
    w._walk_drivers = w.partition_topology.walk_drivers()
    fwd = w.request_state.get_fwd_info(0, "Thinker")
    fwd.step_metadata = {"is_last_prefill": True}
    _pass(w, "Thinker", "prefill_text")
    _pass(w, "Thinker", "prefill_text")
    _pass(w, "Thinker", "thinker_decode")
    assert seen == [
        ProducerWalkCtx("prefill_text", 0, None, fwd),
        ProducerWalkCtx("prefill_text", 1, "talker_prefill", fwd),
        ProducerWalkCtx("thinker_decode", 0, "talker_prefill", fwd),
    ]
    assert seen[0].fwd_info.step_metadata == {"is_last_prefill": True}


# --- conductor ---------------------------------------------------------------

def test_the_conductor_follows_the_walk_the_worker_ran():
    c = object.__new__(Conductor)
    c.enable_prof = False
    c.producer_triggered_partitions = {"Talker"}
    c._set_partition_worker_graph_ids = (
        lambda rid, partition, walk: setattr(pstate, "current_worker_graph_ids", {7})
    )
    pstate = PartitionState(
        partition_name="Talker",
        metadata=CurrentForwardConductorMetadata(
            graph_walk="talker_prefill", is_prefill=True,
        ),
    )
    c.requests = {"r": RequestData(
        persist_signals={}, persist_signal_ref_cnt={},
        worker_graph_to_workers={7: ["w0"]}, all_worker_graph_ids={7},
        max_output_tokens=100, random_seed=0, resource_configs={},
        partition_states={"Talker": pstate},
        partition_definitions={"Talker": PartitionDefinition(
            name="Talker", graph_walks={"talker_prefill", "talker_last_prefill"},
        )},
    )}
    done = c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", worker_graph_ids=[7], is_first_tp_rank=True,
        partition_name="Talker", graph_walk="talker_last_prefill",
    ))
    assert pstate.metadata.graph_walk == "talker_last_prefill"
    assert done == ["Talker"]
