"""A walk the cache holds whole runs nothing, and its node still completes.

When a hit covers every token a walk would write, the walk has no forward to
run, but the graph still waits on its node: the conductor moves a request to
its next walk only once every worker graph reports done. A rid left out of the
step the way a vetoed one is would also be left out of routing, and the
request would sit there until the client gave up.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from mstar.communication.tensors import StoredOutputs
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.graph.runtime.base import FreedTensors, RouteOutput
from mstar.worker.worker import PendingBatch, Worker

RID = 7
NODE = "LLM"
WALK = "prefill"


def _batch(rids=(RID,)) -> ExecutingBatch:
    return ExecutingBatch(
        node_name=NODE,
        per_request_info={rid: None for rid in rids},
        step_context=StepContext(
            request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False,
        ),
    )


class _StubGraphRuntime:
    """Records what it was asked to complete, and routes nothing onward."""

    def __init__(self):
        self.routed = None

    def cleanup_consumed_inputs(self, node_name, rids, wg_ids):
        return FreedTensors.none()

    def clear_pending_loop_stops(self):
        pass

    def get_dynamic_loop_iters(self, rids, partition):
        return []

    def complete_and_route_batch(self, route_input):
        self.routed = route_input
        return RouteOutput(
            completion_id=0, register_uuids=[], register_rids=[],
            new_token_output_idxs=[], local_streaming_by_signal={},
        )

    def send_outputs(self, send_input):
        pass


class _StubTensorManager:
    """Counts each rid's outputs and stores none of them."""

    def cleanup_collectable(self, *freed):
        pass

    def store_and_return_tensor_info_batch(self, rids, outputs, signals, **kwargs):
        return StoredOutputs(
            flat_uuids=[], flat_rids=[], signal_idxs=[],
            num_tensors=[
                len((outputs.get(rid) or {}).get(signal, ()))
                for rid in rids for signal in signals
            ],
        )

    def increment_ref_batch_uniform(self, uuids, count):
        pass

    def dereference_batch_uniform(self, uuids):
        pass


def _worker() -> Worker:
    w = Worker.__new__(Worker)
    w._graph_runtime = _StubGraphRuntime()
    w.tensor_manager = _StubTensorManager()
    w.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        check_stop_for_batch=lambda batch, outputs: {},
        extend_prefix_chains=lambda batch, outputs: None,
    ))
    w.request_state = SimpleNamespace(
        per_request_info={}, get_fwd_info=lambda rid, partition: None,
    )
    w.device = torch.device("cpu")
    w.enable_nvtx = False
    w.enable_prof = False
    w._phase_period = 0
    w._last_active = {}
    return w


def test_a_served_walk_is_routed_with_no_tensors():
    worker = _worker()
    node_batch = _batch(rids=())
    node_batch.cached_rids = {RID}
    pending = PendingBatch(
        batch=SimpleNamespace(
            request_to_worker_graph={RID: 0}, node_name=NODE,
            output_signals=["new_token"],
        ),
        node_batch=node_batch, node_name=NODE, partition="default",
        graph_walk=WALK, future=None,
    )

    worker._postprocess_batch(pending, {})

    routed = worker._graph_runtime.routed
    assert routed is not None and list(routed.wg_ids.keys) == [RID], (
        "a walk the cache served was never routed, so its node never "
        "completes and the request never reaches its next walk"
    )
    assert routed.num_tensors == [0], "a served walk routed tensors it never produced"
