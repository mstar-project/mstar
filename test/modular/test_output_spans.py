"""Per-row valid spans on a step's outputs (``NodeSubmodule.spanned_outputs``).

A submodule returns a fixed-width output and a ``[bs, 2]`` span per row; the
worker cuts each row to its span once the step has run, before the stop check
and the store see it. Drives the real ``Engine`` (declare, admit, plan,
forward, commit, collect) and the real ``Worker._postprocess_batch`` up to the
store, which is stubbed to capture what it would send. CPU only.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mstar.distributed.communication import WorkerParallelGroups  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import (  # noqa: E402
    Resource,
    ResourceStep,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
)
from mstar.graph.base import GraphEdge, GraphNode, Loop  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, NodeSubmodule  # noqa: E402
from mstar.worker.engine_manager import _refuse_speculated_spans  # noqa: E402
from mstar.worker.micro_scheduler import ScheduledBatch  # noqa: E402
from mstar.worker.worker import PendingBatch, Worker  # noqa: E402

NODE, WALK = "vocoder", "synth"
WIDTH = 12  # every row is padded to this, as a captured graph would be


class _Recorder(Resource):
    """Records the lifecycle calls the step drives."""

    def __init__(self):
        self.calls: list[str] = []

    @classmethod
    def build(cls, spec, device, comm_group, **engine_kwargs):
        raise NotImplementedError

    def admit(self, step, ctx):
        self.calls.append("admit")
        return super().admit(step, ctx)

    def plan(self, step, ctx):
        self.calls.append("plan")

    def commit(self, step, ctx):
        self.calls.append("commit")


class _SpanningVocoder(NodeSubmodule):
    """Emits each request's samples zero-padded to ``WIDTH``, with the span of
    its nonzero samples computed on the device, and a second output that is
    not spanned."""

    def __init__(self, batched: bool = True):
        super().__init__()
        self.batched = batched
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.stop_check_lengths: dict[int, int] = {}

    def spanned_outputs(self, graph_walk):
        return frozenset({"audio"})

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(tensor_inputs={"samples": inputs["samples"][0]}, input_seq_len=1)

    def can_batch(self, batch, model_inputs):
        return self.batched

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        segments = tuple(Segment(rid, "main", 1) for rid in request_ids)
        return SubmoduleStep(steps={"rec": ResourceStep(segments=segments)})

    def preprocess(self, graph_walk, engine_inputs, inputs):
        padded = torch.zeros(len(inputs), WIDTH)
        for row, inp in enumerate(inputs):
            samples = inp.tensor_inputs["samples"]
            padded[row, : samples.shape[0]] = samples
        return {"padded": padded}

    def _output(self, rids, padded):
        loud = (padded != 0).to(torch.uint8)
        first = loud.argmax(dim=1)
        last = WIDTH - 1 - loud.flip(1).argmax(dim=1)
        return BatchedModelOutput(
            per_rid_outputs={
                rid: {"audio": [padded[row]], "level": [padded[row].abs().amax(dim=-1, keepdim=True)]}
                for row, rid in enumerate(rids)
            },
            output_spans={"audio": torch.stack([first, last + 1], dim=1)},
        )

    def forward(self, graph_walk, engine_inputs, padded, **kwargs):
        return self._output(engine_inputs.request_ids, padded)

    def forward_batched(self, graph_walk, engine_inputs, padded, **kwargs):
        return self._output(engine_inputs.request_ids, padded)

    def check_stop(self, request_id, request_info, outputs):
        self.stop_check_lengths[request_id] = outputs["audio"][0].shape[-1]
        return set()


class _ReachedStoreError(Exception):
    pass


def _engine(sub: NodeSubmodule) -> tuple[Engine, _Recorder]:
    engine = Engine(graph_runtime=None)
    engine.load_model(
        {NODE: sub}, [], WorkerParallelGroups(num_workers=1, global_rank=0),
        torch.device("cpu"), transfer_engine_info=None,
    )
    recorder = _Recorder()
    engine._resources = {"rec": recorder}
    engine._runner = StepRunner({"rec": recorder}, node_resources={NODE: ["rec"]})
    engine._submodules[NODE].resources = {"rec": recorder}
    return engine, recorder


def _batch(samples: dict[int, torch.Tensor]) -> ExecutingBatch:
    rids = list(samples)
    return ExecutingBatch(
        node_name=NODE,
        per_request_info={
            rid: SimpleNamespace(rid=rid, update_publish_info=lambda published: None) for rid in rids
        },
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
        per_request_input_tensors={rid: {"samples": [row]} for rid, row in samples.items()},
    )


def _worker(engine: Engine) -> tuple[Worker, dict]:
    stored: dict = {}

    def store(rids, outputs, signals, **kwargs):
        stored.update({rid: outputs.get(rid) for rid in rids}, signals=signals)
        raise _ReachedStoreError

    worker = Worker.__new__(Worker)
    worker._phase_period = 0
    worker.enable_nvtx = False
    worker.enable_prof = False
    worker.device = torch.device("cpu")
    worker._last_active = {}
    worker._graph_runtime = SimpleNamespace(
        cleanup_consumed_inputs=lambda node, rids, wgs: ([], []),
        clear_pending_loop_stops=lambda: None,
        stop_loops_batched=lambda **kwargs: [],
        supports_inline_emit=False,
    )
    worker.tensor_manager = SimpleNamespace(
        cleanup_collectable=lambda *args: None,
        store_and_return_tensor_info_batch=store,
    )
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    worker._innermost_loop_by_node = {}
    worker.request_state = SimpleNamespace(buffer_publish_info=lambda *args: None)
    worker.scheduler = SimpleNamespace(failed_rids=set())
    return worker, stored


def _run_step(sub: _SpanningVocoder, samples: dict[int, torch.Tensor]):
    """One step through the engine, then the worker's postprocess up to the store."""
    engine, recorder = _engine(sub)
    batch = _batch(samples)
    engine.prepare_inputs(batch)
    outputs = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None and not batch.failed_requests
    worker, stored = _worker(engine)
    pending = PendingBatch(
        batch=ScheduledBatch(
            node_name=NODE, graph_walk=WALK, request_to_worker_graph=dict.fromkeys(samples, 0),
            output_signals=["audio", "level"],
        ),
        node_batch=batch, node_name=NODE, partition="p", graph_walk=WALK, future=Future(),
    )
    uncut = {rid: out["audio"][0] for rid, out in outputs.per_rid_outputs.items()}
    with pytest.raises(_ReachedStoreError):
        worker._postprocess_batch(pending, outputs)
    return stored, uncut, recorder


SAMPLES = {
    # silence on both sides
    1: torch.tensor([0.0, 0.0, 0.5, -0.25, 0.0, 0.75, 0.0]),
    # fills the padded width: its span is the whole row, so it is not cut
    2: torch.arange(1, WIDTH + 1, dtype=torch.float32),
    # one loud sample at the very start
    3: torch.tensor([0.3]),
}
EXPECTED = {1: [0.5, -0.25, 0.0, 0.75], 2: list(range(1, WIDTH + 1)), 3: [0.3]}


@pytest.mark.parametrize("batched", [True, False], ids=["forward_batched", "per_request"])
def test_rows_reach_the_stop_check_and_the_store_cut_to_their_spans(batched):
    sub = _SpanningVocoder(batched=batched)
    stored, uncut, recorder = _run_step(sub, SAMPLES)
    # the step ran through the resource lifecycle; unbatched, every request
    # admits before any plans
    rows = 1 if batched else len(SAMPLES)
    assert recorder.calls == ["admit"] * rows + ["plan", "commit"] * rows
    for rid, expected in EXPECTED.items():
        audio = stored[rid]["audio"][0]
        assert audio.tolist() == pytest.approx(expected), rid
        assert sub.stop_check_lengths[rid] == len(expected)
        # a view of the step's output, not a copy
        assert audio.untyped_storage().data_ptr() == uncut[rid].untyped_storage().data_ptr()
        # an output the step did not span keeps its width
        assert stored[rid]["level"][0].shape == (1,)
    # a span covering the whole row leaves the tensor as it was
    assert stored[2]["audio"][0] is uncut[2]


def test_a_span_on_an_undeclared_output_fails_the_step():
    class _Undeclared(_SpanningVocoder):
        def spanned_outputs(self, graph_walk):
            return frozenset()

    engine, _ = _engine(_Undeclared())
    batch = _batch({1: SAMPLES[1]})
    engine.prepare_inputs(batch)
    with pytest.raises(ValueError, match="spanned_outputs does not declare"):
        engine.exec_and_postprocess(batch)


def test_a_span_without_a_row_per_request_fails_the_step():
    class _Short(_SpanningVocoder):
        def _output(self, rids, padded):
            out = super()._output(rids, padded)
            out.output_spans["audio"] = out.output_spans["audio"][:1]
            return out

    engine, _ = _engine(_Short())
    batch = _batch({1: SAMPLES[1], 3: SAMPLES[3]})
    engine.prepare_inputs(batch)
    with pytest.raises(ValueError, match=r"not \[2, 2\]"):
        engine.exec_and_postprocess(batch)


def test_spans_are_merged_in_request_order():
    a = BatchedModelOutput(output_spans={"audio": torch.tensor([[1, 4]])}, span_request_ids=(7,))
    a.update(BatchedModelOutput(output_spans={"audio": torch.tensor([[0, 2]])}, span_request_ids=(9,)))
    assert a.span_request_ids == (7, 9)
    assert a.output_spans["audio"].tolist() == [[1, 4], [0, 2]]


def test_a_merge_of_spans_on_different_outputs_is_refused():
    """A later forward's span on another name must not reach the store uncut,
    nor fail the step with a KeyError on a name the first forward lacks."""
    a = BatchedModelOutput(output_spans={"audio": torch.tensor([[1, 4]])}, span_request_ids=(7,))
    b = BatchedModelOutput(
        output_spans={"audio": torch.tensor([[0, 2]]), "level": torch.tensor([[0, 1]])},
        span_request_ids=(9,),
    )
    with pytest.raises(ValueError, match="every forward of a step spans the same outputs"):
        a.update(b)


def test_a_step_spanning_only_some_declared_outputs_fails():
    """Declaring two spanned outputs and spanning one would leave the other
    uncut on some rows once per-request forwards are merged by name."""
    class _Partial(_SpanningVocoder):
        def spanned_outputs(self, graph_walk):
            return frozenset({"audio", "level"})

    engine, _ = _engine(_Partial())
    batch = _batch({1: SAMPLES[1]})
    engine.prepare_inputs(batch)
    with pytest.raises(ValueError, match=r"but none for \['level'\]"):
        engine.exec_and_postprocess(batch)


class _Graph:
    def __init__(self, section):
        self.section = section

    def get_graph_walk_graphs(self):
        return {WALK: self.section}


def _loop(async_scheduling: bool, next_node: str = NODE):
    node = GraphNode(
        name=NODE, input_names=["samples"],
        outputs=[GraphEdge(next_node=next_node, name="audio")],
        enable_async_scheduling=async_scheduling,
    )
    return Loop(name="loop", section=node, max_iters=4, outputs=[])


def test_startup_refuses_a_spanned_output_that_a_speculated_step_reads():
    submodules = {NODE: _SpanningVocoder()}
    with pytest.raises(ValueError, match="enable_async_scheduling=False"):
        _refuse_speculated_spans(submodules, _Graph(_loop(async_scheduling=True)))
    # no speculation, or nothing on this worker reading it: nothing to refuse
    _refuse_speculated_spans(submodules, _Graph(_loop(async_scheduling=False)))
    _refuse_speculated_spans(submodules, _Graph(_loop(async_scheduling=True, next_node=EMIT_TO_CLIENT)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stages a CUDA span to pinned host memory")
def test_gpu_thread_stages_spans_to_the_host_behind_the_step():
    from mstar.utils.device import DeviceCapability

    worker = Worker.__new__(Worker)
    worker.device = torch.device("cuda", torch.cuda.current_device())
    worker.device_capability = DeviceCapability(worker.device)
    worker._host_rows_slot = 0
    worker._pinned_d2h_buffers = __import__("collections").defaultdict(list)
    spans = torch.tensor([[3, 9], [0, 5]], device=worker.device)
    outputs = BatchedModelOutput(output_spans={"audio": spans}, span_request_ids=(1, 2))
    worker._stage_host_rows(outputs)
    torch.cuda.current_stream().synchronize()
    host = outputs.host_spans["audio"]
    assert host.device.type == "cpu" and host.is_pinned()
    assert host.tolist() == [[3, 9], [0, 5]]
    assert outputs.host_rows is None
