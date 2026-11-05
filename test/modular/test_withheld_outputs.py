"""A step's output that its declared stop withholds is never routed.

A ``TokenStop.withhold`` (``StopPolicy.withhold``) names outputs that are not
data on an iteration whose token is a stop id, ``ignore_eos`` or not. The
worker reads that off the stop check's host copy, whichever check decided the
stops, and leaves those items out of the store, so the route carries none of
them (a streaming edge then gets no item). A step that repeated routes each
row the iterations its stop kept. Drives the real ``Worker._postprocess_batch``
on CPU, with the graph runtime stubbed to record what it was asked to route.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.communication.communicator import BaseCommunicator
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.graph.runtime.base import FreedTensors, PhasedPostprocess, PostprocessNeed, PostprocessStep, RouteOutput
from mstar.model.submodule_base import BatchedModelOutput, HostRows, NodeSubmodule, StopKind, StopPolicy
from mstar.utils.containers import ParallelList
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker

NODE, WALK, STOP_TOKEN, LOOP = "producer", "decode", 9, "loop"
KEEP, DROP = 1, 2
# the producer's declared stop: STOP_TOKEN ends the loop and its frame is no data
WITHHOLDING = {(NODE, WALK): StopPolicy(
    StopKind.EOS, token_signal="token", eos_tokens=(STOP_TOKEN,), withhold=("frame",),
)}


class _Producer(NodeSubmodule):
    """Streams a frame every step; its own check stops on a stop token, or
    with ``stop=False`` (``ignore_eos``, say) runs on past it."""

    def __init__(self, fail_on=None, stop=True):
        super().__init__()
        self.fail_on, self.stop = fail_on, stop

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def check_stop(self, request_id, request_info, outputs):
        if request_id == self.fail_on:
            raise RuntimeError("cannot tell")
        return {LOOP} if self.stop and int(outputs["token"][0].item()) == STOP_TOKEN else set()


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


def test_a_raise_fails_only_that_request():
    batch = _batch()
    stops = _engine(_Producer(fail_on=KEEP)).check_stop_for_batch(batch, _outputs())
    assert set(batch.failed_requests) == {KEEP} and stops == {DROP: {LOOP}}


class _Communicator(BaseCommunicator):
    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []

    def register_event_for_poll(self, event):
        pass

    def poll_for_messages(self, timeout_ms=20):
        return False


class _RecordingRuntime(PhasedPostprocess):
    """Recording phase stubs under the reference begin / resume."""

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


@pytest.mark.parametrize("stop", [True, False], ids=["stops", "ignore_eos"])
def test_the_worker_routes_no_tensor_for_a_withheld_output(tmp_path, stop):
    """The model's own check decides the stops here (no runtime policies), and
    the frame of the stop-token row is withheld whether or not it stops."""
    engine = _engine(_Producer(stop=stop))
    w = Worker.__new__(Worker)
    w._phase_period = 0
    w.enable_nvtx = w.enable_prof = False
    w.device = torch.device("cpu")
    w._innermost_loop_by_node = {}
    w._walk_drivers = {}  # no producer-driven walk transitions here
    w._stop_token_signal = {}  # no declared stop policies: check_stop decides
    w._withholding = WITHHOLDING
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
        after_output_copy=lambda batch: None,
        step_finished=lambda batch, outputs, **kwargs: None,
    ))
    w.request_state = SimpleNamespace(
        per_request_info={},
        buffer_publish_info=lambda *args: None,
        get_fwd_info=lambda rid, partition: None,
        get_pending_publish_info=lambda rid, partition: {},
    )
    w.streams = StreamBufferManager(w.request_state, w.tensor_manager, [])  # no streams
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

    assert w._graph_runtime.stops == ([{DROP: [LOOP]}] if stop else [])
    (route,) = w._graph_runtime.routes
    assert list(route.wg_ids.keys) == [KEEP, DROP]
    # (KEEP: token, frame), (DROP: token, frame): the stop-token step's frame carries nothing
    assert route.num_tensors == [1, 1, 1, 0]
    assert len(route.tensors) == 3


def _cut(width, tokens, kept=None, emitted=None, columns=("frame",), spans=None):
    """``Worker._cut_run_items`` over two rows whose "frame" holds a column
    per replay (KEEP's values 0.., DROP's 10..) beside each replay's
    ``tokens``; the frames a row routes."""
    def column(base):
        if emitted is not None:  # two items a replay along dim 1, as many as it emitted real
            return torch.arange(base, base + width * 2, dtype=torch.float32).reshape(1, width, 2)
        values = torch.arange(base, base + width, dtype=torch.float32).reshape(1, width, 1)
        return values if width > 1 else values[:, 0]

    w = Worker.__new__(Worker)
    w._withholding = WITHHOLDING
    outputs = BatchedModelOutput(
        per_rid_outputs={KEEP: {"frame": [column(0)]}, DROP: {"frame": [column(10)]}},
        output_spans=None if spans is None else {"frame": torch.tensor(spans)},
        span_request_ids=None if spans is None else (KEEP, DROP),
    )
    buffers = {"token": torch.tensor(tokens)}
    if emitted is not None:
        buffers["emitted_count"] = torch.tensor(emitted)
    pending = SimpleNamespace(
        node_name=NODE, graph_walk=WALK,
        batch=ScheduledBatch(node_name=NODE, graph_walk=WALK, request_to_worker_graph={KEEP: 0, DROP: 0}),
    )
    pending.batch.repeats = width
    step = PostprocessStep(PostprocessNeed.STORE, ParallelList([KEEP, DROP], [0, 0]), kept=kept or {})
    engine = SimpleNamespace(run_columns=lambda node, walk: frozenset(columns))
    Worker._cut_run_items(w, pending, outputs, step, HostRows((KEEP, DROP), buffers), None, engine)
    return {
        rid: [t.reshape(-1)[0].item() for t in outs.get("frame", [])]
        for rid, outs in outputs.per_rid_outputs.items()
    }


def test_a_run_routes_each_row_its_kept_iterations_less_the_withheld_eos_column():
    """A row that stopped on an EOS at iteration 1 routes iteration 0 alone:
    the run past its stop goes nowhere, and the EOS column's frame is withheld."""
    eos_at_1 = [[1, 2, 3], [4, STOP_TOKEN, 5]]
    assert _cut(3, eos_at_1, kept={DROP: 2}) == {KEEP: [0, 1, 2], DROP: [10]}
    # a stop by the loop's cap withholds nothing: its last frame is data
    assert _cut(3, [[1, 2, 3], [4, 5, 6]], kept={DROP: 2}) == {KEEP: [0, 1, 2], DROP: [10, 11]}
    # one iteration: the stop-token row routes no frame
    assert _cut(1, [[1], [STOP_TOKEN]]) == {KEEP: [0], DROP: []}


def test_a_replay_that_emits_two_items_routes_both():
    """``EMITTED_COUNT`` says how many of a replay's items along dim 1 are
    real: KEEP emits 2 then 1, DROP 1 then 2."""
    assert _cut(2, [[1, 2], [3, 4]], emitted=[[2, 1], [1, 2]]) == {KEEP: [0, 1, 2], DROP: [10, 12, 13]}


def test_a_run_cuts_each_replays_item_to_its_own_span():
    """A spanned per-iteration output holds a span a replay: each item is cut
    to its own, and an empty one routes nothing."""
    spans = [[[0, 1], [0, 1]], [[0, 1], [1, 1]]]  # DROP's second replay is empty
    assert _cut(2, [[1, 2], [3, 4]], spans=spans) == {KEEP: [0, 1], DROP: [10]}


def test_a_runs_column_hands_on_a_tensor_of_its_own():
    """A row's iteration, ``column[:, i]``, is a view with a stride of the
    run's width even at one element, where ``contiguous`` leaves it as it is:
    a routed edge's byte view refused it (Canary at N=4), and the step after
    the run took the whole row (Orpheus)."""
    from mstar.communication.tensors import _serialize_tensor
    from mstar.worker.worker import run_item

    column = torch.tensor([[5, 6, 7, 8]])
    item = run_item(column, 2)
    assert item.tolist() == [7] and item.stride() == (1,)
    assert _serialize_tensor(item) == _serialize_tensor(torch.tensor([7]))
