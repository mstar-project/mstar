"""A fresh rid merged into a speculative step keeps its final stream chunk.

A stream-consuming node in a loop with no stop of its own (nemotron_duplex's
``nano_llm``) ends when it consumes its stream's final chunk:
``ExecutingBatch.final_stream_rids`` stops the loop in postprocess and
``stream_partition_done_rids`` reports the partition done. A rid that left the
speculation chain (it caught up with its stream, or waited in the backlog) has
its final chunk ingested by the plain stream poll, and can then join another
chain's speculative step as a FRESH rid. Its final flag rides on the fresh
batch's edges, not on the chunks the speculation polls itself, so the step has
to take it from there; dropping it leaves the loop waiting forever for a chunk
the stream will never send.

Drives the real ``_try_speculate_next`` / ``_try_follow_speculation``,
``SpeculationBuilder.assemble`` and ``StreamBufferManager.settle_final``; the scheduler, graph
runtime and tensor store are stubs that hand back one continuing and one fresh
rid.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.graph.runtime.base import (  # noqa: E402
    ColumnarEdgeSpecs,
    SpeculationOutput,
    SpeculationPrepOutput,
)
from mstar.streaming.chunk_policy import FixedChunkPolicy  # noqa: E402
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.micro_scheduler import MicroScheduler, PopReadyResult, ScheduledBatch  # noqa: E402
from mstar.worker.speculation import SpeculationBuilder, SpeculationContext, SpeculationPreparer  # noqa: E402
from mstar.worker.stream_buffers import StreamBufferManager  # noqa: E402
from mstar.worker.worker import SpeculationTarget, Worker  # noqa: E402

NODE, WALK, LOOP, PART, STREAM = "nano_llm", "decode", "decode_loop", "LLM", "audio_frame"
CONTINUING, FRESH = 1, 2


def _edges(rows) -> ColumnarEdgeSpecs:
    """``rows``: (rid, signal, is_final) per edge, one tensor each."""
    block = ColumnarEdgeSpecs.empty()
    for i, (rid, signal, is_final) in enumerate(rows):
        block.add(rid, signal, [i + 1], is_final)
    return block


def _fresh_edges(final: bool) -> ColumnarEdgeSpecs:
    # what the plain poll ingested for FRESH: its fed-back token and its
    # stream's last chunk
    return _edges([(FRESH, "prev_text", False), (FRESH, STREAM, final)])


def _prep() -> SpeculationPrepOutput:
    # CONTINUING's loop-back; its own next chunk is not in yet
    return SpeculationPrepOutput(
        consumed_streaming_edge_idxs=[], ready_rids=[CONTINUING], wg_ids=[0],
        input_edges=_edges([(CONTINUING, "prev_text", False)]),
    )


def _worker(scheduler, runtime):
    def stream_state():
        sbuf = SimpleNamespace(policy=FixedChunkPolicy(chunk_size=1))
        return SimpleNamespace(stream_buffers={STREAM: sbuf}, ended_streams=set())

    fwd = {
        rid: CurrentForwardPassInfo(
            request_id=rid, graph_walk=WALK, fwd_index=0, random_seed=0, max_tokens=0,
        ) for rid in (CONTINUING, FRESH)
    }
    worker = SimpleNamespace(
        worker_id="w0",
        scheduler=scheduler,
        _graph_runtime=runtime,
        tensor_manager=SimpleNamespace(get_tensor=lambda uuid: torch.tensor([uuid])),
        request_state=SimpleNamespace(
            per_request_info={rid: stream_state() for rid in fwd},
            get_fwd_info=lambda rid, partition: fwd[rid],
        ),
        _in_flight_rids=set(),
        chunk_loops=ChunkLoops(),
        _is_tearing_down=lambda rid: False,
        _teardowns_pending=lambda: False,
    )
    streams = StreamBufferManager(worker.request_state, worker.tensor_manager, [])
    streams.consumer_node = {STREAM: NODE}
    streams._partition = {STREAM: PART}
    streams._rids = {CONTINUING, FRESH}
    streams.poll_for = lambda rid, node: []
    streams.chunks_for = lambda rid, node, inputs: None
    worker.streams = streams
    for name in ("_try_speculate_next", "_try_follow_speculation"):
        setattr(worker, name, getattr(Worker, name).__get__(worker))
    ctx = SpeculationContext(
        graph_runtime=runtime, tensor_manager=worker.tensor_manager,
        request_state=worker.request_state, scheduler=scheduler, streams=streams,
    )
    worker._spec_context = ctx
    worker._spec_preparer = SpeculationPreparer(ctx)
    worker._spec_builder = SpeculationBuilder(ctx)
    return worker


def _runtime():
    target = SpeculationOutput(
        node_name=NODE, graph_walk=WALK, is_new_loop_iter=True, loop_name=LOOP,
        output_signals=("prev_text",),
    )
    return SimpleNamespace(
        prep_spec_rids=lambda inp: _prep(),
        prep_follow_spec_rids=lambda inp: _prep(),
        get_spec_target=lambda *a: target,
        get_consumed_edges=lambda *a: [("prev_text", NODE)],
    ), target


def _pending():
    return SimpleNamespace(
        batch=ScheduledBatch(
            node_name=NODE, graph_walk=WALK, request_to_worker_graph={CONTINUING: 0},
        ),
        node_name=NODE, graph_walk=WALK, partition=PART, tp_seq=5,
        node_batch=SimpleNamespace(commit_done=SimpleNamespace(is_set=lambda: True)),
    )


def _leader_speculation(final: bool):
    fresh = ScheduledBatch(
        node_name=NODE, graph_walk=WALK, request_to_worker_graph={FRESH: 0},
        input_edges=_fresh_edges(final),
    )
    scheduler = SimpleNamespace(
        backlog_splits_from=lambda *a: False,
        room_for_continuing=lambda target: None,
        get_next_batch=lambda *a, **k: fresh,
    )
    runtime, target = _runtime()
    worker = _worker(scheduler, runtime)
    return worker._try_speculate_next(_pending(), SpeculationTarget(target, [CONTINUING]))


def _follower_speculation(final: bool):
    head = SimpleNamespace(
        speculative=True, spec_from_seq=5, spec_seq=6, node_name=NODE, graph_walk=WALK,
        walks=[], walk_idx=[], request_ids=[CONTINUING, FRESH], read_check=False,
    )
    scheduler = SimpleNamespace(
        peek_tp_follow=lambda: head,
        settle_tp_follow_delta=lambda caught_up: True,
        unread_rows=lambda h, rids: set(),
        tp_rids=lambda h: list(h.request_ids),
        pop_ready_rids=lambda *a, **k: PopReadyResult(
            {FRESH: 0}, _fresh_edges(final), ("prev_text",),
        ),
        pop_tp_follow_head=lambda: head,
        combined_walk_of={},
    )
    scheduler._key = MicroScheduler._key.__get__(scheduler)
    runtime, _ = _runtime()
    worker = _worker(scheduler, runtime)
    return worker._try_follow_speculation(_pending())


def test_leader_fresh_rid_final_chunk_ends_its_stream():
    spec = _leader_speculation(final=True)

    assert list(spec.scheduled_batch.request_to_worker_graph) == [CONTINUING, FRESH]
    assert spec.node_batch.final_stream_rids == {FRESH}
    assert spec.node_batch.stream_partition_done_rids == {FRESH}


def test_follower_fresh_rid_final_chunk_ends_its_stream():
    spec = _follower_speculation(final=True)

    assert list(spec.scheduled_batch.request_to_worker_graph) == [CONTINUING, FRESH]
    assert spec.node_batch.final_stream_rids == {FRESH}
    assert spec.node_batch.stream_partition_done_rids == {FRESH}


def test_fresh_rid_without_a_final_chunk_keeps_its_stream_open():
    for build in (_leader_speculation, _follower_speculation):
        spec = build(final=False)
        assert FRESH in spec.scheduled_batch.request_to_worker_graph
        assert not spec.node_batch.final_stream_rids
        assert not spec.node_batch.stream_partition_done_rids
