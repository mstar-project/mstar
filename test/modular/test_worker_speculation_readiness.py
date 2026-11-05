"""A speculative step carries a request onto the next node only once that node's readiness check passes.

Otherwise the request reaches that node's admit without a readiness check, so it never joins
the node's KV pool queue and holds no allotment there. A request left out of the
speculative step joins that queue like any other.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, ".")

from mstar.graph.runtime.base import SpeculationOutput
from mstar.model.submodule_base import BatchedModelOutput
from mstar.worker.batch_builder import StepPick
from mstar.worker.micro_scheduler import ChainPick, ScheduledBatch
from mstar.worker.worker import SpeculationTarget, Worker


class _Preparer:
    """Records the rids speculation asked to prep."""

    def prepare(self, batch_N, spec_node_name, graph_walk, candidates, max_continuing, laps, loop_strides=None):
        self.prepped = list(candidates)
        return SimpleNamespace(ready_rids=[])


def test_a_request_the_next_node_holds_back_is_not_continued_onto_it():
    worker = Worker.__new__(Worker)
    worker._spec_preparer = _Preparer()
    # nothing is tearing down
    worker._pending_removes, worker._pending_drains, worker._draining_rids = {}, {}, set()
    worker.scheduler = SimpleNamespace(
        failed_rids=set(),
        backlog_splits_from=lambda *_: False,
        # an uncapped target: the chain keeps every row it brings
        pick_chain_step=lambda state, target, continuing, *_: ChainPick(target, StepPick(
            taken=list(continuing), continuing=list(continuing), room=None,
            held_back=set(), ordered=False,
        ), []),
        # the decoder isn't ready to run request 2
        _check_ready=lambda node, rid, info: rid != 2,
    )
    worker.request_state = SimpleNamespace(get_partition_for_node=lambda node: 0, get_fwd_info=lambda rid, p: None)
    target = SpeculationTarget(SpeculationOutput(node_name="decoder", graph_walk="walk"), [1, 2, 3])

    batch = ScheduledBatch(node_name="audio_encoder", graph_walk="walk")
    worker._try_speculate_next(SimpleNamespace(batch=batch), target)

    assert worker._spec_preparer.prepped == [1, 3], "a request the decoder would not run was carried onto it anyway"


def test_a_speculated_step_takes_the_last_iteration_of_a_signal_that_repeated():
    """Orpheus's token loops back and goes to SNAC, so a run keeps it a column
    an iteration: the step after the run reads the last one, not the row."""
    worker = Worker.__new__(Worker)
    node_batch = SimpleNamespace(request_ids=[1], per_request_input_tensors={1: {"text_inputs": []}})
    speculation = SimpleNamespace(
        node_batch=node_batch, continuing_rids={1}, scheduled_batch=SimpleNamespace(chunk_ranges={}),
        consumed_edges=[("text_inputs", "LLM")], dropped=set(),
    )
    outputs = BatchedModelOutput(per_rid_outputs={1: {"text_inputs": [torch.tensor([[5, 6, 7, 8]])]}})
    outputs.run_columns = frozenset({"text_inputs"})
    worker._thread_outputs_to_speculative(speculation, outputs)
    (token,) = node_batch.per_request_input_tensors[1]["text_inputs"]
    assert token.tolist() == [8]
