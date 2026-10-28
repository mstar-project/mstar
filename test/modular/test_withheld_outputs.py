"""A step's output that the stop check finds is not data is never routed.

``check_stop`` returns a ``StopCheck`` (``check_stop_batched`` a
``BatchedStopCheck``) that names it beside the loops to stop, off the same
read; ``Engine.check_stop_for_batch`` records it on the batch and the worker
leaves those outputs out of the store, so the route carries zero tensors for
them (a streaming edge then gets no item). Drives the real engine method and
the real ``Worker._postprocess_batch`` on CPU, with the graph runtime stubbed
to record what it was asked to route.
"""

from types import SimpleNamespace

import torch

from mstar.communication.communicator import BaseCommunicator
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.graph.runtime.base import FreedTensors, RouteOutput
from mstar.model.submodule_base import BatchedModelOutput, BatchedStopCheck, HostRows, NodeSubmodule, StopCheck
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker, _LazyPerRid

NODE, WALK, STOP_TOKEN, LOOP = "producer", "decode", 9, "loop"
KEEP, DROP = 1, 2


class _Producer(NodeSubmodule):
    """Streams a frame every step; the frame of a stop-token step is not data.
    With ``stop=False`` (``ignore_eos``, say) the loop runs on past it."""

    def __init__(self, fail_on=None, stop=True, batched=False):
        super().__init__()
        self.fail_on, self.stop, self.batched = fail_on, stop, batched
        self.reads = []

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _check(self, request_id, token):
        self.reads.append(request_id)
        if request_id == self.fail_on:
            raise RuntimeError("cannot tell")
        if token != STOP_TOKEN:
            return set()
        return StopCheck({LOOP} if self.stop else set(), withheld=("frame",))

    def check_stop(self, request_id, request_info, outputs):
        return self._check(request_id, int(outputs["token"][0].item()))

    def check_stop_batched(self, request_ids, request_infos, host_rows, step_walk=None):
        if not self.batched:
            return None
        stops, withheld = {}, {}
        for rid, token in host_rows.first_values("token", request_ids):
            result = self._check(rid, token)
            if isinstance(result, StopCheck):
                if result.stops:
                    stops[rid] = result.stops
                withheld[rid] = result.withheld
        return BatchedStopCheck(stops, withheld)


class _Plain(_Producer):
    """A stop check that only stops, the way every other model's does."""

    def _check(self, request_id, token):
        self.reads.append(request_id)
        return {LOOP} if token == STOP_TOKEN else set()


def _batch(rids=(KEEP, DROP)):
    return ExecutingBatch(
        node_name=NODE, per_request_info={rid: SimpleNamespace() for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
    )


def _outputs():
    return {
        KEEP: {"token": [torch.tensor(3)], "frame": [torch.tensor([3, 0])]},
        DROP: {"token": [torch.tensor(STOP_TOKEN)], "frame": [torch.tensor([STOP_TOKEN, 0])]},
    }


def _engine(submodule):
    engine = Engine.__new__(Engine)
    engine._submodules = {NODE: SimpleNamespace(submodule=submodule)}
    return engine


def test_the_stop_check_records_what_the_submodule_withholds():
    submodule, batch = _Producer(), _batch()
    assert _engine(submodule).check_stop_for_batch(batch, _outputs()) == {DROP: {LOOP}}
    assert batch.withheld_outputs == {DROP: ("frame",)}
    assert submodule.reads == [KEEP, DROP]   # one read per row


def test_a_row_can_be_withheld_without_stopping():
    batch = _batch()
    assert _engine(_Producer(stop=False)).check_stop_for_batch(batch, _outputs()) == {}
    assert batch.withheld_outputs == {DROP: ("frame",)}


def test_a_plain_set_withholds_nothing():
    batch = _batch()
    assert _engine(_Plain()).check_stop_for_batch(batch, _outputs()) == {DROP: {LOOP}}
    assert batch.withheld_outputs == {}


def test_a_raise_fails_only_that_request():
    batch = _batch()
    _engine(_Producer(fail_on=KEEP)).check_stop_for_batch(batch, _outputs())
    assert set(batch.failed_requests) == {KEEP}
    assert batch.withheld_outputs == {DROP: ("frame",)}


def test_the_batched_check_withholds_off_the_rows_without_building_per_row_dicts():
    host = {"token": torch.tensor([[3], [STOP_TOKEN]]), "frame": torch.tensor([[3, 0], [STOP_TOKEN, 0]])}
    per_rid = _LazyPerRid(host, [KEEP, DROP])
    submodule, batch = _Producer(batched=True), _batch()

    stops = _engine(submodule).check_stop_for_batch(batch, per_rid, host_rows=HostRows((KEEP, DROP), host))

    assert stops == {DROP: {LOOP}} and batch.withheld_outputs == {DROP: ("frame",)}
    assert submodule.reads == [KEEP, DROP]
    assert per_rid._built is None


def test_a_batched_check_that_declines_falls_back_per_row():
    host = {"token": torch.tensor([[3], [STOP_TOKEN]])}
    batch = _batch()
    stops = _engine(_Producer()).check_stop_for_batch(batch, _outputs(), host_rows=HostRows((KEEP, DROP), host))
    assert stops == {DROP: {LOOP}} and batch.withheld_outputs == {DROP: ("frame",)}


class _Communicator(BaseCommunicator):
    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []


class _RecordingRuntime:
    def __init__(self):
        self.routes = []
        self.stops = []

    def cleanup_consumed_inputs(self, *args):
        return FreedTensors.none()

    def clear_pending_loop_stops(self):
        pass

    def stop_loops_batched(self, partition, graph_walk, last_node_run, loop_names):
        self.stops.append(dict(zip(loop_names.keys, loop_names.values, strict=True)))
        return []

    def update_dynamic_loop_iters(self, per_request_info, partition):
        pass

    def send_outputs(self, send_input):
        return []

    def complete_and_route_batch(self, route_input):
        self.routes.append(route_input)
        return RouteOutput(
            completion_id=len(self.routes), register_uuids=[], register_rids=[],
            new_token_output_idxs=[], local_streaming_by_signal={},
        )


def test_the_worker_routes_no_tensor_for_a_withheld_output(tmp_path):
    engine = _engine(_Producer())
    w = Worker.__new__(Worker)
    w._phase_period = 0
    w.enable_nvtx = w.enable_prof = False
    w.device = torch.device("cpu")
    w._last_active = {}
    w._innermost_loop_by_node = {}
    w._graph_runtime = _RecordingRuntime()
    w.scheduler = SimpleNamespace(failed_rids=set())
    w.tensor_manager = SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=_Communicator(), shm_dir=str(tmp_path),
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        check_stop_for_batch=engine.check_stop_for_batch,
        extend_prefix_chains=lambda batch, outputs, **kwargs: None,
        materialize_publish=lambda batch, rids: None,
    ))
    w.request_state = SimpleNamespace(
        per_request_info={},
        buffer_publish_info=lambda *args: None,
        get_fwd_info=lambda rid, partition: None,
        get_pending_publish_info=lambda rid, partition: {},
    )
    pending = SimpleNamespace(
        batch=ScheduledBatch(
            node_name=NODE, graph_walk=WALK, request_to_worker_graph={KEEP: 0, DROP: 0},
            output_signals=["token", "frame"],
        ),
        node_batch=_batch(),
        node_name=NODE, partition="default", graph_walk=WALK, loop_name=None,
        speculative_new_iter=False,
    )

    Worker._postprocess_batch(w, pending, BatchedModelOutput(per_rid_outputs=_outputs()))

    assert w._graph_runtime.stops == [{DROP: [LOOP]}]
    (route,) = w._graph_runtime.routes
    assert list(route.wg_ids.keys) == [KEEP, DROP]
    # (KEEP: token, frame), (DROP: token, frame): the stop step's frame carries nothing
    assert route.num_tensors == [1, 1, 1, 0]
    assert len(route.tensors) == 3
