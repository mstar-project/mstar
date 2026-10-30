"""A speculative step carries a request onto the next node only once that node's readiness check passes.

Otherwise the request reaches that node's admit without a readiness check, so it never joins
the node's KV pool queue and holds no allotment there. A request left out of the
speculative step joins that queue like any other.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.worker.batch_builder import StepPick
from mstar.worker.micro_scheduler import ChainPick
from mstar.worker.worker import SpeculationTarget, Worker


class _Runtime:
    """Records the rids speculation asked the graph runtime to prep."""

    def prep_spec_rids(self, prep_input):
        self.prepped = list(prep_input.rids)
        return SimpleNamespace(
            consumed_streaming_edge_idxs=[], ready_rids=[], wg_ids=[],
            input_edges=SimpleNamespace(to_input_tensors=lambda get_tensor, rids: SimpleNamespace(by_rid={})),
        )


def test_a_request_the_next_node_holds_back_is_not_continued_onto_it():
    worker = Worker.__new__(Worker)
    worker._graph_runtime = _Runtime()
    worker._is_tearing_down = lambda rid: False
    worker._stream_buffer_rids = set()
    worker.scheduler = SimpleNamespace(
        backlog_splits_from=lambda *_: False,
        # an uncapped target: the chain keeps every row it brings
        pick_chain_step=lambda state, target, continuing: ChainPick(target, StepPick(
            taken=list(continuing), continuing=list(continuing), room=None,
            held_back=set(), ordered=False,
        ), []),
        # the decoder isn't ready to run request 2
        _check_ready=lambda node, rid, info: rid != 2,
    )
    worker.tensor_manager = SimpleNamespace(get_tensor=None)
    worker.request_state = SimpleNamespace(get_partition_for_node=lambda node: 0, get_fwd_info=lambda rid, p: None)
    target = SpeculationTarget(SimpleNamespace(node_name="decoder", graph_walk="walk"), [1, 2, 3])

    worker._try_speculate_next(SimpleNamespace(batch=SimpleNamespace(node_name="audio_encoder")), target)

    assert worker._graph_runtime.prepped == [1, 3], "a request the decoder would not run was carried onto it anyway"
