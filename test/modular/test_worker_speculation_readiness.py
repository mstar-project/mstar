"""A speculative step carries a request onto the next node only once that node's readiness check passes.

Otherwise the request reaches that node's admit without a readiness check, so it never joins
the node's KV pool queue and holds no allotment there. A request left out of the
speculative step joins that queue like any other.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.engine.resources.step import FULL_ADMIT_NOT_READY, FULL_ADMIT_OK
from mstar.graph.runtime.base import SpeculationOutput
from mstar.worker.batch_builder import StepPick
from mstar.worker.micro_scheduler import ChainPick, ScheduledBatch
from mstar.worker.testing import make_worker
from mstar.worker.worker import SpeculationTarget


class _Preparer:
    """Records the rids speculation asked to prep."""

    def prepare(self, batch_N, spec_node_name, graph_walk, candidates, max_continuing, laps, loop_strides=None):
        self.prepped = list(candidates)
        return SimpleNamespace(ready_rids=[])


def test_a_request_the_next_node_holds_back_is_not_continued_onto_it():
    # the decoder isn't ready to run request 2
    engine = SimpleNamespace(
        check_ready=lambda node, rid, info, allow_reload=True: FULL_ADMIT_NOT_READY if rid == 2 else FULL_ADMIT_OK,
    )
    worker = make_worker(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        request_state=SimpleNamespace(get_partition_for_node=lambda node: 0, get_fwd_info=lambda rid, p: None),
        scheduler=SimpleNamespace(
            failed_rids=set(),
            backlog_splits_from=lambda *_, **__: False,
            # an uncapped target: the chain keeps every row it brings
            pick_chain_step=lambda target, continuing, mid_prefill=(), next_iter_of=None: ChainPick(target, StepPick(
                taken=list(continuing), continuing=list(continuing), room=None,
                held_back=set(), ordered=False,
            ), []),
        ),
    )
    worker._spec_preparer = _Preparer()
    target = SpeculationTarget(SpeculationOutput(node_name="decoder", graph_walk="walk"), [1, 2, 3])

    batch = ScheduledBatch(node_name="audio_encoder", graph_walk="walk")
    worker._try_speculate_next(SimpleNamespace(batch=batch), target)

    assert worker._spec_preparer.prepped == [1, 3], "a request the decoder would not run was carried onto it anyway"
