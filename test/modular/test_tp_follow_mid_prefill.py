"""A TP follower's speculative step runs the leader's chunks: a row of N that
needs another chunk takes it from the input held since its first chunk, not as
a loop-back row."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

from mstar.engine.chunking import ChunkRange
from mstar.graph.runtime.base import ColumnarEdgeSpecs, SpeculationPrepOutput
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.worker import Worker

NODE, MIXED = "LLM", "mixed"


def _follower(head: ScheduleTPNode) -> Worker:
    """A real ``Worker`` with only what ``_try_follow_speculation`` reads, as
    far as assembling the step."""
    worker = Worker.__new__(Worker)
    worker.worker_id = "worker_1"
    worker.scheduler = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: None), parallel_leader_nodes=set(),
    )
    worker.scheduler.register_tp_follow(head)
    worker.scheduler.chunk_loop_walks = {NODE: frozenset({"prefill_text"})}
    worker.capped = []
    worker.scheduler.runtime = SimpleNamespace(
        set_loop_caps=lambda node, rids, walks, loops, caps: worker.capped.extend(zip(rids, caps, strict=True)),
    )
    worker.prepped = []

    def prep(prep_input):
        worker.prepped.append(list(prep_input.rids))
        return SpeculationPrepOutput(
            consumed_streaming_edge_idxs=[], ready_rids=list(prep_input.rids),
            wg_ids=[0] * len(prep_input.rids), input_edges=ColumnarEdgeSpecs.empty(),
        )

    worker._graph_runtime = SimpleNamespace(
        get_spec_target=lambda *args: SimpleNamespace(node_name=NODE, graph_walk="decode"),
        prep_follow_spec_rids=prep,
    )
    worker.tensor_manager = SimpleNamespace(get_tensor=lambda uuid: uuid)
    worker.request_state = SimpleNamespace(per_request_info={})
    worker._in_flight_rids = set()
    worker._poll_stream_buffers_for_speculation = lambda rid, node: []
    worker.assembled = {}

    worker.streams = SimpleNamespace(poll_for=lambda rid, node: [])

    def assemble(pending, target, rid_to_wg, inputs, consumed, continuing, fresh_final, **kwargs):
        worker.assembled = dict(rid_to_wg=rid_to_wg, inputs=inputs, continuing=list(continuing), **kwargs)
        batch = ScheduledBatch(
            node_name=NODE, graph_walk=MIXED, request_to_worker_graph=dict(rid_to_wg),
            request_walks=kwargs["request_walks"] or {}, chunk_ranges=kwargs["chunk_ranges"],
        )
        return SimpleNamespace(continuing_rids=set(continuing), node_batch=SimpleNamespace(), scheduled_batch=batch)

    worker._spec_builder = SimpleNamespace(assemble=assemble)
    return worker


def _pending(batch_n: ScheduledBatch) -> SimpleNamespace:
    return SimpleNamespace(
        batch=batch_n, node_name=NODE, graph_walk=MIXED, tp_seq=5,
        node_batch=SimpleNamespace(commit_done=SimpleNamespace(is_set=lambda: True)),
    )


def _batch_n() -> ScheduledBatch:
    """N ran d0's decode and p0's first chunk."""
    return ScheduledBatch(
        node_name=NODE, graph_walk=MIXED, request_to_worker_graph={"d0": 0, "p0": 0},
        request_walks={"d0": "decode", "p0": "prefill_text"},
        chunk_ranges={"p0": ChunkRange(0, 4, 10, 4)}, incomplete_node_rids={"p0"},
    )


def test_a_follower_runs_a_row_mid_prefill_as_the_leader_does():
    """The leader's head for N+1 holds both rows. p0 loops back on nothing of
    N's, so it is not prepped: an all-or-none prep would wait on a loop-back
    input N never produced."""
    head = ScheduleTPNode(
        node_name=NODE, graph_walk=MIXED, request_ids=["d0", "p0"],
        speculative=True, spec_seq=6, spec_from_seq=5, walks=["decode", "prefill_text"], walk_idx=[0, 1],
        chunk_starts=[-1, 4], chunk_ends=[-1, 8], chunk_totals=[-1, 10], chunk_strides=[-1, 4],
    )
    follower = _follower(head)

    speculation = follower._try_follow_speculation(_pending(_batch_n()))

    assert follower.prepped == [["d0"]]
    assert follower.assembled["continuing"] == ["d0"], "the leader advances no loop for a row mid-prefill"
    assert follower.assembled["inputs"]["p0"] == {}
    assert follower.assembled["rid_to_wg"] == {"d0": 0, "p0": 0}
    assert follower.assembled["chunk_ranges"] == {"p0": (4, 8, 10, 4)}
    assert follower.assembled["incomplete_node_rids"] == {"p0"}
    assert speculation.continuing_rids == {"d0", "p0"}, "a cleared step must not ready p0 mid-chunk"
    assert follower.capped == [], "p0's Loop was capped on its first chunk, in N"


def test_a_followers_last_chunk_ends_its_prompt_as_the_leaders_does():
    head = ScheduleTPNode(
        node_name=NODE, graph_walk=MIXED, request_ids=["d0", "p0"],
        speculative=True, spec_seq=6, spec_from_seq=5, walks=["decode", "prefill_text"], walk_idx=[0, 1],
        chunk_starts=[-1, 4], chunk_ends=[-1, 10], chunk_totals=[-1, 10], chunk_strides=[-1, 6],
    )
    follower = _follower(head)

    follower._try_follow_speculation(_pending(_batch_n()))

    assert follower.assembled["chunk_ranges"] == {"p0": (4, 10, 10, 6)}
    assert follower.assembled["incomplete_node_rids"] == set()


def test_a_follower_refuses_a_speculative_head_whose_chunks_do_not_match_its_rows():
    """The ranks disagree on the step, so nothing is built and the head stays queued."""
    head = ScheduleTPNode(
        node_name=NODE, graph_walk=MIXED, request_ids=["d0", "p0"],
        speculative=True, spec_seq=6, spec_from_seq=5, walks=["decode", "prefill_text"], walk_idx=[0, 1],
        chunk_starts=[4], chunk_ends=[8], chunk_totals=[10], chunk_strides=[4],
    )
    follower = _follower(head)

    with pytest.raises(ValueError):
        follower._try_follow_speculation(_pending(_batch_n()))

    assert follower.prepped == []
    assert follower.scheduler.peek_tp_follow() is head


def test_a_follower_caps_a_fresh_rows_loop_on_its_first_chunk_as_the_leader_does():
    """The cap decides on which step the Loop ends, so every rank of the
    instance sets the leader's. p1 starts past a 2-token cached prefix: its
    stride runs from 0, which is how a follower tells a first chunk."""
    head = ScheduleTPNode(
        node_name=NODE, graph_walk=MIXED, request_ids=["d0", "p0", "p1"],
        speculative=True, spec_seq=6, spec_from_seq=5, walks=["decode", "prefill_text"], walk_idx=[0, 1, 1],
        chunk_starts=[-1, 4, 2], chunk_ends=[-1, 8, 6], chunk_totals=[-1, 10, 12], chunk_strides=[-1, 4, 6],
    )
    follower = _follower(head)
    follower.scheduler.pop_ready_rids = lambda *args, **kwargs: SimpleNamespace(
        wg_ids={"p1": 0}, edge_specs=ColumnarEdgeSpecs.empty(),
    )

    follower._try_follow_speculation(_pending(_batch_n()))

    assert follower.assembled["chunk_ranges"]["p1"] == (2, 6, 12, 6)
    assert follower.capped == [("p1", 12)]
