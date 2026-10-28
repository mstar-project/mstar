"""A step's host copies must not outlive its sends.

The stop check copies a step's outputs into pinned host buffers that the next
step reuses, and a host-memory transport (SHM) sends from those copies. A
loop's accumulated output is stored on its own iteration but sent when the loop
ends, after later steps have rewritten the buffer, so every iteration was sent
with the last one's values (V-JEPA2-AC rollouts came back with four equal
steps).

Drives the real ``Worker._postprocess_batch`` with the SHM transport on CPU.
The graph runtime is a stub that holds each iteration's output and registers
them all on the last one, as a loop's accumulated edge to the client does, and
the stop check hands out one reused host buffer, as the pinned ring does on a
GPU (on CPU the real one returns the outputs themselves).
"""

from types import SimpleNamespace

import torch

from mstar.communication.communicator import BaseCommunicator
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.graph.base import GraphEdge
from mstar.graph.runtime.base import FreedTensors, RouteOutput
from mstar.model.submodule_base import BatchedModelOutput
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker

STEPS, RID, NODE, WALK = 4, 7, "rollout_predictor", "rollout"


class _Communicator(BaseCommunicator):
    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []


class _AccumulatingRuntime:
    """Holds every iteration's output and registers all of them on the last."""

    def __init__(self):
        self.held: list[int] = []

    def cleanup_consumed_inputs(self, *args):
        return FreedTensors.none()

    def clear_pending_loop_stops(self):
        pass

    def update_dynamic_loop_iters(self, per_request_info, partition):
        pass

    def send_outputs(self, send_input):
        return []

    def complete_and_route_batch(self, route_input):
        self.held += list(route_input.tensors)
        send = len(self.held) == STEPS
        return RouteOutput(
            completion_id=len(self.held),
            register_uuids=list(self.held) if send else [],
            register_rids=[RID] * len(self.held) if send else [],
            new_token_output_idxs=[], local_streaming_by_signal={},
        )


def _manager(tmp_path, entity_id):
    return SharedMemoryCommunicationManager(
        my_entity_id=entity_id, hostname="localhost", device="cpu",
        communicator=_Communicator(), shm_dir=str(tmp_path),
    )


def _worker(tmp_path):
    w = Worker.__new__(Worker)
    w._phase_period = 0
    w.enable_nvtx = w.enable_prof = False
    w.device = torch.device("cpu")
    w._last_active = {}
    w._kept_ids = {}
    w._copied_inputs = {}
    w._innermost_loop_by_node = {}
    w._graph_runtime = _AccumulatingRuntime()
    w.scheduler = SimpleNamespace(failed_rids=set())
    w.tensor_manager = _manager(tmp_path, "worker_0")
    w.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        check_stop_for_batch=lambda batch, outputs, **kwargs: {},
        extend_prefix_chains=lambda batch, outputs, **kwargs: None,
        materialize_publish=lambda batch, rids: None,
    ))
    w.request_state = SimpleNamespace(
        per_request_info={},
        buffer_publish_info=lambda *args: None,
        get_fwd_info=lambda rid, partition: None,
        get_pending_publish_info=lambda rid, partition: {},
    )
    reused = torch.empty(1, 4)

    def stop_check_copies(outputs, completion_event, request_ids=None):
        reused.copy_(outputs.per_rid_outputs[RID]["pred"][0])
        return {RID: {"pred": [reused]}}, None

    w._prematerialize_for_check_stop = stop_check_copies
    return w


def _step(w, value):
    pending = SimpleNamespace(
        batch=ScheduledBatch(
            node_name=NODE, graph_walk=WALK, request_to_worker_graph={RID: 0},
            output_signals=["pred"],
        ),
        node_batch=ExecutingBatch(
            node_name=NODE, per_request_info={}, cached_rids={RID},
            step_context=StepContext(request_ids=(), graph_walk=WALK, slot=0, capture=False),
        ),
        node_name=NODE, partition="default", graph_walk=WALK, loop_name=None,
        speculative_new_iter=False,
    )
    outputs = BatchedModelOutput(per_rid_outputs={RID: {"pred": [torch.full((1, 4), value)]}})
    Worker._postprocess_batch(w, pending, outputs)


def test_accumulated_outputs_are_sent_with_their_own_values(tmp_path):
    w = _worker(tmp_path)
    for step in range(STEPS):
        _step(w, float(step))
        assert not w.tensor_manager.tensor_store._tensors_cpu, "a step left host copies behind"

    uuids = w._graph_runtime.held
    infos = [w.tensor_manager.tensor_store.get_info(uuid) for uuid in uuids]
    reader = _manager(tmp_path, "api_server")
    edges = [GraphEdge(next_node="client", name="pred", tensor_info=infos)]
    reader.start_read_tensors(RID, edges, graph_walk=WALK)
    reader.get_ready_tensors(graph_walk=WALK)
    got = [reader.get_tensor(uuid).flatten().tolist() for uuid in uuids]
    assert got == [[float(step)] * 4 for step in range(STEPS)]
