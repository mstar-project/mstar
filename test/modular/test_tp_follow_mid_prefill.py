"""A TP follower's speculative step: a row of N that needs another chunk takes
it from the input held since its first chunk, not as a loop-back row."""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.graph.runtime.base import ColumnarEdgeSpecs, SpeculationPrepOutput
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.worker import Worker

NODE, MIXED = "LLM", "mixed"


class _Follower:
    """``Worker._try_follow_speculation`` over stubs, as far as assembling the step."""

    _try_follow_speculation = Worker._try_follow_speculation

    def __init__(self, head: ScheduleTPNode):
        self.worker_id = "worker_1"
        self.scheduler = MicroScheduler(
            engine_manager=SimpleNamespace(get_engine=lambda node: None), parallel_leader_nodes=set(),
        )
        self.scheduler.register_tp_follow(head)
        self.prepped: list[list] = []

        def prep(prep_input):
            self.prepped.append(list(prep_input.rids))
            return SpeculationPrepOutput(
                consumed_streaming_edge_idxs=[], ready_rids=list(prep_input.rids),
                wg_ids=[0] * len(prep_input.rids), input_edges=ColumnarEdgeSpecs.empty(),
            )

        self._graph_runtime = SimpleNamespace(
            get_spec_target=lambda *a: SimpleNamespace(node_name=NODE, graph_walk="decode"),
            prep_follow_spec_rids=prep,
        )
        self.tensor_manager = SimpleNamespace(get_tensor=lambda uuid: uuid)
        self.request_state = SimpleNamespace(per_request_info={})
        self._in_flight_rids: set = set()
        self.assembled: dict = {}

    def _poll_stream_buffers_for_speculation(self, rid, node):
        return []

    def _assemble_speculation(self, pending, target, rid_to_wg, inputs, consumed, continuing, **kw):
        self.assembled = dict(rid_to_wg=rid_to_wg, inputs=inputs, continuing=list(continuing), **kw)
        return SimpleNamespace(continuing_rids=set(continuing))


def test_a_follower_runs_a_row_mid_prefill_as_the_leader_does():
    """N ran d0's decode and p0's first chunk; the leader's head for N+1 holds
    both. p0 loops back on nothing of N's, so it is not prepped: an all-or-none
    prep would expect a loop-back input it never produced."""
    batch_n = ScheduledBatch(
        node_name=NODE, graph_walk=MIXED, request_to_worker_graph={"d0": 0, "p0": 0},
        request_walks={"d0": "decode", "p0": "prefill_text"},
        chunk_ranges={"p0": (0, 4, 10)}, incomplete_node_rids={"p0"},
    )
    pending = SimpleNamespace(
        batch=batch_n, node_name=NODE, graph_walk=MIXED, tp_seq=5,
        node_batch=SimpleNamespace(commit_done=SimpleNamespace(is_set=lambda: True)),
    )
    head = ScheduleTPNode(
        node_name=NODE, graph_walk=MIXED, request_ids=["d0", "p0"],
        speculative=True, spec_seq=6, spec_from_seq=5,
        chunk_starts=[-1, 4], chunk_ends=[-1, 8], chunk_totals=[-1, 10], incomplete_node_rids=["p0"],
    )
    follower = _Follower(head)

    speculation = follower._try_follow_speculation(pending)

    assert follower.prepped == [["d0"]]
    assert follower.assembled["continuing"] == ["d0"]
    assert follower.assembled["inputs"]["p0"] == {}
    assert follower.assembled["chunk_ranges"] == {"p0": (4, 8, 10)}
    assert speculation.continuing_rids == {"d0", "p0"}, "a cleared step must not ready p0 mid-chunk"
