"""Producer-triggered graph-walk transitions.

A connection with ``consumer_walk`` drives its consumer partition's walk from
the stream: every streamed item carries the walk its producer emitted it
under, a chunk never spans two producer walks, and the consumer switches walk
before popping a chunk, but only when the partition is idle. Conductor inputs
for a walk the stream has not reached yet wait on the worker.
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
from mstar.streaming.topology import Connection, PartitionTopology, WalkTransitionCtx
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
    """(walk ctx, items) per chunk, as the consumer would see them."""
    out = []
    while sbuf.has_chunk_ready() and sbuf._buffer:
        ctx = sbuf.peek_walk()
        chunk = sbuf.pop_chunk()
        out.append((ctx, chunk.num_items))
    return out


# --- StreamBuffer ------------------------------------------------------------

def test_a_chunk_never_spans_two_producer_walks():
    sbuf = _buffer(chunk_size=3)
    _feed(sbuf, ["prefill_text", "prefill_text", "thinker_decode",
                 "thinker_decode", "thinker_decode"])
    assert _drain(sbuf) == [
        (WalkTransitionCtx("prefill_text", True), 2),
        (WalkTransitionCtx("thinker_decode", True), 3),
    ]


def test_only_the_first_chunk_of_a_run_starts_the_walk():
    sbuf = _buffer()
    _feed(sbuf, ["prefill_text", "prefill_audio", "thinker_decode",
                 "thinker_decode", "prefill_audio", "thinker_decode"])
    assert [ctx for ctx, _ in _drain(sbuf)] == [
        WalkTransitionCtx("prefill_text", True),
        WalkTransitionCtx("prefill_audio", True),
        WalkTransitionCtx("thinker_decode", True),
        WalkTransitionCtx("thinker_decode", False),
        WalkTransitionCtx("prefill_audio", True),
        # Re-entering a walk is a new run
        WalkTransitionCtx("thinker_decode", True),
    ]


def test_an_incomplete_run_waits_for_more_items():
    sbuf = _buffer(chunk_size=3)
    _feed(sbuf, ["thinker_decode", "thinker_decode"])
    assert not sbuf.has_chunk_ready()
    assert sbuf.peek_walk() == WalkTransitionCtx("thinker_decode", True)


def test_an_untagged_buffer_is_unchanged():
    sbuf = _buffer(chunk_size=3, tagged=False)
    _feed(sbuf, ["a", "a", "b", "b"])
    assert sbuf.peek_walk() is None
    assert _drain(sbuf) == [(None, 3)]


# --- Qwen3-Omni's Talker -----------------------------------------------------

def test_the_thinker_stream_walks_the_talker_through_its_prefill():
    """No count of the Thinker's prefill walks is needed: the stream says when
    the prefill is over. Both edges into the Talker agree item by item."""
    thinker = ["prefill_text", "prefill_audio", "prefill_text",
               "thinker_decode", "thinker_decode", "thinker_decode"]
    talker = ["talker_prefill"] * 3 + [
        "talker_last_prefill", "talker_decode", "talker_decode",
    ]
    for _edge in ("thinker_states", "thinker_mask"):
        sbuf = _buffer()
        _feed(sbuf, thinker)
        assert [_talker_walk(ctx) for ctx, _ in _drain(sbuf)] == talker


# --- topology checks ---------------------------------------------------------

def _conn(edge, policy=None, walk=_talker_walk):
    return Connection(
        from_partition="Thinker", to_partition="Talker", edge_name=edge,
        chunk_policy_factory=policy or (lambda: FixedChunkPolicy(1)),
        consumer_walk=walk,
    )


def test_every_edge_into_a_driven_partition_must_map_walks():
    topology = PartitionTopology(
        partitions=["Thinker", "Talker"],
        connections=[_conn("thinker_states"), _conn("thinker_mask", walk=None)],
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
    w._consumer_walk_fns = {"thinker_states": _talker_walk}
    w._consumer_node_cache = {"thinker_states": "Talker"}
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


@pytest.mark.parametrize("idle, allow, switched", [
    (True, True, True),
    (False, True, False),   # mid-pass: the chunk waits
    (True, False, False),   # speculation: the step was built for this walk
])
def test_a_chunk_for_another_walk_switches_only_an_idle_partition(idle, allow, switched):
    w = _worker(idle=idle)
    sbuf = _buffer()
    _feed(sbuf, ["thinker_decode"])
    assert w._enter_chunk_walk(sbuf, "thinker_states", 0, allow) is switched
    walk = w.request_state.get_fwd_info(0, "Talker").graph_walk
    assert walk == ("talker_last_prefill" if switched else "talker_prefill")


def test_a_chunk_for_the_current_walk_needs_no_switch():
    w = _worker(idle=False)
    sbuf = _buffer()
    _feed(sbuf, ["prefill_text"])
    assert w._enter_chunk_walk(sbuf, "thinker_states", 0, allow_walk_change=False)
    assert w.walks == []


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
