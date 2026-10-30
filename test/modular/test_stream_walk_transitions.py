"""Producer-triggered graph-walk transitions.

A connection with ``consumer_walk`` lets the producer decide its consumer
partition's walk: the producer worker assigns each emitting pass's streamed
items a consumer walk, a chunk never spans two assigned walks, and the consumer
switches walk before popping a chunk, but only when the partition is idle.
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
from mstar.streaming.chunk_policy import FixedChunkPolicy, LeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.streaming.topology import Connection, PartitionTopology, ProducerWalkCtx
from mstar.utils.ipc_format import WorkerGraphsDone
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.stream_buffers import StreamBufferManager
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


def _talker_walk(ctx: ProducerWalkCtx) -> str:
    """Qwen3-Omni's Talker rule, restated: importing the model needs flashinfer,
    which the CPU suite does not have (its own copy is tested in
    test_qwen3_omni_talker_walk.py)."""
    if ctx.producer_walk != "thinker_decode":
        return "talker_prefill"
    first = ctx.loop_iters.get("thinker_decode_loop", 0) == 0
    return "talker_last_prefill" if first else "talker_decode"


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


def _streams(idle=True):
    request_state = RequestStateManager(node_to_partition={"Talker": "Talker"})
    request_state.add_request(0, _fwd("talker_prefill"))
    walks = []
    runtime = SimpleNamespace(
        is_partition_idle=lambda rid, partition: idle,
        set_walk=lambda rid, partition, walk: walks.append(walk),
    )
    streams = StreamBufferManager(
        request_state, None, [], graph_runtime=runtime,
        walk_driven_edges={"thinker_states": "Talker"},
    )
    return streams, request_state, walks


@pytest.mark.parametrize("idle, allow, switched", [
    (True, True, True),
    (False, True, False),   # mid-pass: the chunk waits
    (True, False, False),   # speculation: the step was built for this walk
])
def test_a_chunk_for_another_walk_switches_only_an_idle_partition(idle, allow, switched):
    streams, request_state, _ = _streams(idle=idle)
    sbuf = _buffer()
    _feed(sbuf, ["talker_last_prefill"])
    assert streams._enter_chunk_walk(sbuf, "thinker_states", 0, allow) is switched
    walk = request_state.get_fwd_info(0, "Talker").graph_walk
    assert walk == ("talker_last_prefill" if switched else "talker_prefill")


def test_a_chunk_for_the_current_walk_needs_no_switch():
    streams, _, walks = _streams(idle=False)
    sbuf = _buffer()
    _feed(sbuf, ["talker_prefill"])
    assert streams._enter_chunk_walk(sbuf, "thinker_states", 0, allow_walk_change=False)
    assert walks == []


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
    w._walk_ctx = ProducerWalkCtx()
    w._graph_runtime = SimpleNamespace(
        get_output_signals=lambda node, walk: (
            ["thinker_mask", "thinker_states"] if (node, walk) in _EMITS else ["audio_embeds"]
        ),
    )
    w.request_state = RequestStateManager()
    w.request_state.add_request(0, _fwd("prefill_text", partition="Thinker"))
    return w


def _batch(node, rows):
    """A Thinker step of ``rows``: rid -> (its walk, its loop iterations)."""
    walks = {rid: walk for rid, (walk, _) in rows.items()}
    graph_walk = next(iter(walks.values())) if len(set(walks.values())) == 1 else "thinker_mixed"
    return SimpleNamespace(
        node_name=node, graph_walk=graph_walk, partition="Thinker",
        batch=ScheduledBatch(
            node_name=node, graph_walk=graph_walk,
            request_to_worker_graph=dict.fromkeys(rows, 0),
            request_walks=walks if graph_walk == "thinker_mixed" else {},
        ),
        node_batch=SimpleNamespace(per_request_input_metadata={
            rid: SimpleNamespace(dynamic_loop_iter_counts=iters or {})
            for rid, (_, iters) in rows.items()
        }),
    )


def _pass(w, node, walk, loop_iters=None):
    """The Talker walk this pass's states were assigned, or None if it emitted none."""
    fwd = w.request_state.get_fwd_info(0, "Thinker")
    fwd.stream_consumer_walks.clear()
    w._assign_consumer_walks(_batch(node, {0: (walk, loop_iters)}))
    return fwd.stream_consumer_walks.get("Talker")


def _decode(i):
    return {"thinker_decode_loop": i}


def test_the_thinker_walks_the_talker_through_its_prefill():
    """The Talker needs no count of the Thinker's prefill walks: the Thinker,
    which knows when its prefill is over, assigns each pass's states a walk."""
    w = _producer()
    passes = [
        ("Thinker", "prefill_text", None),
        ("audio_encoder", "prefill_audio", None),  # emits nothing to the Talker
        ("Thinker", "prefill_audio", None), ("Thinker", "prefill_text", None),
        ("Thinker", "thinker_decode", _decode(0)),
        ("Thinker", "thinker_decode", _decode(1)),
        ("Thinker", "thinker_decode", _decode(2)),
    ]
    got = [_pass(w, node, walk, iters) for node, walk, iters in passes]
    assert got == [
        "talker_prefill", None, "talker_prefill", "talker_prefill",
        "talker_last_prefill", "talker_decode", "talker_decode",
    ]


def test_a_mixed_step_asks_each_row_under_its_own_walk():
    """Chunked prefill batches prefill and decode rows as ``thinker_mixed``,
    which no worker graph declares: each row's states take the walk it ran."""
    w = _producer()
    w.request_state.add_request(1, _fwd("thinker_decode", partition="Thinker"))
    asked = []
    get_output_signals = w._graph_runtime.get_output_signals

    def signals(node, walk):
        assert walk != "thinker_mixed", "the combined label is no walk of the graph's"
        asked.append(walk)
        return get_output_signals(node, walk)

    w._graph_runtime.get_output_signals = signals
    w._assign_consumer_walks(_batch("Thinker", {
        0: ("prefill_text", None), 1: ("thinker_decode", _decode(4)),
    }))
    assert sorted(asked) == ["prefill_text", "thinker_decode"]
    walks = [
        w.request_state.get_fwd_info(rid, "Thinker").stream_consumer_walks["Talker"]
        for rid in (0, 1)
    ]
    assert walks == ["talker_prefill", "talker_decode"]


def test_the_hook_sees_the_pass_and_not_a_count_of_emissions():
    """Two producer nodes streaming in one walk see the graph's own loop
    iteration, so neither shifts what the other is told."""
    seen = []

    def record(ctx: ProducerWalkCtx) -> str:
        # The context is reused across calls: copy what is needed during the call
        seen.append((ctx.producer_walk, ctx.node, dict(ctx.loop_iters), ctx.fwd_info))
        return "talker_decode"

    w = _producer()
    for conn in w.partition_topology.connections:
        conn.consumer_walk = record
    w._walk_drivers = w.partition_topology.walk_drivers()
    w._graph_runtime.get_output_signals = lambda node, walk: ["thinker_states"]
    fwd = w.request_state.get_fwd_info(0, "Thinker")
    fwd.step_metadata = {"is_last_prefill": True}
    _pass(w, "Thinker", "thinker_decode", _decode(3))
    _pass(w, "Projector", "thinker_decode", _decode(3))
    _pass(w, "Thinker", "thinker_decode", _decode(4))
    assert seen == [
        ("thinker_decode", "Thinker", _decode(3), fwd),
        ("thinker_decode", "Projector", _decode(3), fwd),
        ("thinker_decode", "Thinker", _decode(4), fwd),
    ]
    assert seen[0][3].step_metadata == {"is_last_prefill": True}


def test_a_pass_allocates_nothing_per_row():
    w = _producer()
    walks = w.request_state.get_fwd_info(0, "Thinker").stream_consumer_walks
    ctx = w._walk_ctx
    _pass(w, "Thinker", "thinker_decode", _decode(0))
    _pass(w, "Thinker", "thinker_decode", _decode(1))
    assert w.request_state.get_fwd_info(0, "Thinker").stream_consumer_walks is walks
    assert w._walk_ctx is ctx


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
