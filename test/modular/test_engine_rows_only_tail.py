"""A rows-only step runs only a submodule's batch tail, and only when it has
one; a step on the unbatched forward (per request or a single row) is not
rows-only."""
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.engine.engine import Engine  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.model.submodule_base import BatchedModelOutput, NodeSubmodule  # noqa: E402


class _Plain(NodeSubmodule):
    def prepare_inputs(self, *a, **kw):
        return None

    def forward(self, *a, **kw):
        return None


class _WithTail(_Plain):
    def batch_postprocess(self, outputs, request_infos, inputs, piecewise_runners=None):
        self.seen = (outputs, request_infos, inputs, piecewise_runners)


RUNNERS = {"region": object()}


def _engine(sub):
    eng = Engine.__new__(Engine)
    eng._submodules = {"llm": SimpleNamespace(submodule=sub, piecewise_runners=RUNNERS, device_stops={})}
    eng._loopback_steps = {}
    eng.per_request_tails = []
    eng.postprocess_batch = lambda batch, outputs: eng.per_request_tails.append(batch)
    return eng


def _batch(rows_only):
    return SimpleNamespace(
        node_name="llm", admit_error=None, rows_only=rows_only, cached_rids=set(),
        per_request_info_wrapped={1: object()}, inputs=[object()], run_started=None,
    )


@pytest.mark.parametrize("sub_cls", [_Plain, _WithTail])
def test_rows_only_runs_the_batch_tail_only_when_overridden(sub_cls):
    sub = sub_cls.__new__(sub_cls)
    eng = _engine(sub)
    out = BatchedModelOutput()
    eng.exec = lambda batch: out
    batch = _batch(rows_only=True)
    assert eng.exec_and_postprocess(batch) is out
    assert eng.per_request_tails == []
    if sub_cls is _WithTail:
        # the node's captured regions come along, as the forward gets them
        assert sub.seen == (out, batch.per_request_info_wrapped, batch.inputs, RUNNERS)
    eng.exec_and_postprocess(_batch(rows_only=False))
    assert len(eng.per_request_tails) == 1


@pytest.mark.parametrize("rids", [[1, 2], [1]])
def test_an_unbatched_step_clears_rows_only(rids):
    eng = _engine(_Plain.__new__(_Plain))
    eng._enable_nvtx = False
    eng._enable_profile = False
    eng._device = SimpleNamespace(type="cpu")
    eng._autocast_dtype_for = lambda sub: None
    eng.reserve_replay_slot = lambda batch: None
    seen = {}

    def _per_request(batch):
        seen["rows_only"] = batch.rows_only
        return BatchedModelOutput()

    eng._exec_per_request = _per_request
    eng._exec_single = _per_request
    batch = SimpleNamespace(
        node_name="llm", request_ids=rids, rows_only=True, running_batched=False,
        step_context=StepContext(request_ids=rids, graph_walk="decode", slot=0, capture=False),
        exec_timings=SimpleNamespace(start=None), admit_error=None,
        outputs_ready=threading.Event(), release_waiters=lambda: None,
        preplanned_rids=None, preplan_event=None,
    )
    eng.exec(batch)
    assert seen["rows_only"] is False and batch.rows_only is False
    assert eng._loopback_steps == {}


@pytest.mark.parametrize("executions", [1, 2])
def test_run_row_collection_uses_descriptor_order_and_the_late_requests_metadata(executions):
    import torch

    from mstar.engine.device_run import RUN_COUNTS, RUN_DESCRIPTORS, RUN_REPLAYS
    from mstar.engine.engine import ExecutingBatch
    from mstar.engine.resources.iteration_tables import STREAM_COUNTS, STREAM_TRANSITION

    def raw(values):
        values = torch.tensor([values])
        spans = torch.stack((values, values + 1), dim=-1)
        return BatchedModelOutput(row_outputs={"frame": values, "final": values + 100}, output_spans={"frame": spans},
                                  check_stop_buffers={"token": values, RUN_COUNTS: torch.tensor([99]),
                                                      RUN_REPLAYS: torch.tensor([99]),
                                                      STREAM_COUNTS: torch.tensor([3]),
                                                      STREAM_TRANSITION: torch.tensor([7])})

    first, second = raw([11, 12, 21]), raw([22, 31, 32])
    first.check_stop_buffers[RUN_DESCRIPTORS] = torch.tensor([[7, 7, 8]])
    second.check_stop_buffers[RUN_DESCRIPTORS] = torch.tensor([[8, 9, 9]])
    records = [(first, [[7, 7, 8]]), (second, [[8, 9, 9]])]
    late = Engine._run_row_output(records, 8, 4)
    assert late.row_outputs["frame"].tolist() == [[21, 22, 22, 22]]
    assert late.output_spans["frame"].tolist() == [[[21, 22], [22, 23], [22, 23], [22, 23]]]
    assert late.check_stop_buffers["token"].tolist() == [[21, 22, 22, 22]]
    assert late.check_stop_buffers[RUN_COUNTS].tolist() == [2]
    compacted, counts = Engine._run_outputs(records, [8, 10], 4, {"frame"})
    assert counts == {8: 2, 10: 0}
    assert compacted.row_outputs["frame"].tolist() == [[21, 22, 22, 22], [0, 0, 0, 0]]
    assert compacted.row_outputs["final"].tolist() == [122, 0]
    assert compacted.check_stop_buffers[RUN_COUNTS].tolist() == [2, 0]
    assert compacted.check_stop_buffers[RUN_REPLAYS].tolist() == [2, 2]
    assert not {RUN_DESCRIPTORS, STREAM_COUNTS, STREAM_TRANSITION} & compacted.check_stop_buffers.keys()
    empty, counts = Engine._run_outputs(records, [10], 4, {"frame"})
    assert counts == {10: 0} and not empty.row_outputs
    assert empty.check_stop_buffers[RUN_COUNTS].tolist() == [0]
    assert Engine._run_row_output(records, 10, 4) is None
    with pytest.raises(ValueError, match="admitted logical horizon"):
        Engine._run_row_output(records, 8, 1)

    class Filtered(_Plain):
        def filter_batched_output(self, info, outputs):
            return {info.signal: outputs["frame"]}

        def spanned_outputs(self, walk):
            return {"frame"}

    sub = Filtered.__new__(Filtered)
    eng = _engine(sub)
    pending = _batch(rows_only=False)
    eng.exec = lambda batch: first
    assert eng.exec_and_postprocess(pending) is first
    assert eng._collect_step(pending, eng._submodules["llm"], None, first, [], {}) is first
    assert eng.per_request_tails == []
    collected = eng._collect_outputs(
        eng._submodules["llm"], None, late, [None], {44: SimpleNamespace(signal="late")},
        request_ids=[44], step_request_ids=(44,), graph_walk="decode",
    )
    assert collected.per_rid_outputs[44]["late"][0].tolist() == [[21, 22, 22, 22]]
    assert collected.row_request_ids == (44,) and collected.span_request_ids == (44,)
    eng._postprocess_outputs(pending, collected)
    assert eng.per_request_tails == [pending]
    # The real deferred tail sees this receipt's inputs/metadata and only its
    # actual columns, including an accepted row that ran exactly once.
    eng._submodules["llm"].rings = {}
    eng._submodules["llm"].run_columns = {"decode": frozenset({"frame"})}
    ctx = StepContext((44,), "decode", 0, False, repeats=executions)
    ctx.run_counts = {44: executions}
    receipt = ExecutingBatch("llm", {}, ctx, inputs=[None])
    receipt.per_request_info_wrapped = {44: SimpleNamespace(signal="late")}
    receipt.run_rows = {44: SimpleNamespace(handle=8)}
    receipt.run_started = torch.tensor([1])
    selected = records if executions == 2 else records[:1]
    resolved = eng.collect_run_outputs(receipt, selected)
    assert resolved.per_rid_outputs[44]["late"][0].tolist() == [[21, 22][:executions]]
    assert resolved.run_columns == frozenset({"frame"})
    assert RUN_COUNTS not in resolved.check_stop_buffers and RUN_REPLAYS not in resolved.check_stop_buffers
    assert eng.per_request_tails == [pending, receipt]
    ctx.run_counts = {44: 0}
    with pytest.raises(ValueError, match="settled, nonempty"):
        eng.collect_run_outputs(receipt, selected)
    first.row_outputs["frame"].fill_(-1)
    second.row_outputs["frame"].fill_(-2)
    assert collected.per_rid_outputs[44]["late"][0].tolist() == [[21, 22, 22, 22]]


@pytest.mark.parametrize("captured,rows_only", [(False, False), (True, False), (True, True)])
def test_logical_subset_collection_projects_original_rows_before_tail(captured, rows_only):
    import torch

    class Spanned(_Plain):
        def spanned_outputs(self, walk):
            return {"frame"}

    eng = _engine(Spanned.__new__(Spanned))
    mgmt = eng._submodules["llm"]
    mgmt.accelerator_graph_runner = SimpleNamespace(slot_for=lambda lease: SimpleNamespace(dummy_rids=(-1, -2, -3)))
    values = torch.tensor([[10, 11], [20, 21], [30, 31]])
    raw = BatchedModelOutput(
        row_outputs={"frame": values}, check_stop_buffers={"token": values, "semantic": torch.tensor([0, 1, 2])},
        output_spans={"frame": torch.tensor([[0, 2], [1, 2], [0, 1]])}, row_request_ids=(40, 50, 60),
        per_rid_outputs={} if rows_only else {(-3 if captured else 60): {"keyed": [torch.tensor([600])]}},
    )
    result = eng._collect_outputs(
        mgmt, object() if captured else None, raw, [None], {60: None},
        request_ids=[60], step_request_ids=(40, 50, 60), graph_walk="decode", rows_only=rows_only,
    )
    assert result.row_request_ids == (60,)
    assert result.check_stop_buffers["token"].tolist() == [[30, 31]]
    assert result.check_stop_buffers["semantic"].tolist() == [2]
    if not rows_only:
        assert result.per_rid_outputs[60]["frame"][0].tolist() == [[30, 31]]
        assert result.per_rid_outputs[60]["keyed"][0].tolist() == [600]
        assert result.output_spans["frame"].tolist() == [[0, 1]]
    assert raw.row_request_ids == (40, 50, 60) and raw.check_stop_buffers["semantic"].tolist() == [0, 1, 2]


@pytest.mark.parametrize("producer", [False, True])
def test_connected_tails_select_positive_rows_but_settle_the_complete_receipt(producer):
    import torch

    from mstar.engine.device_run import RUN_COUNTS, RUN_REPLAYS
    from mstar.engine.engine import ExecutingBatch
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.iteration_tables import STREAM_COUNTS, STREAM_TRANSITION
    from mstar.engine.resources.runner import StepRunner
    from mstar.engine.resources.step import ResourceStep
    from mstar.model.submodule_base import NodeInputs, SubmoduleStep

    rids, requested = [40, 50, 60], 4 if producer else 1
    counts = {40: 0, 50: 2 if producer else 1, 60: 1}
    lengths = dict.fromkeys(rids, requested)
    tails, batch_tails, rewinds = [], [], []

    class Owner(Resource):
        @classmethod
        def build(cls, spec, info):
            return cls()

        def rewind_run(self, step, ctx, delta, selected):
            rewinds.append((delta, set(selected)))
            for rid in selected:
                lengths[rid] -= delta

        def step_host_rows(self, repeats):
            return {"owned": torch.tensor([400, 500, 600])}

    class Tail(_Plain):
        def postprocess(self, request_id, request_info, outputs, inputs):
            assert request_id != 40, "semantic completion cannot invoke an ordinary model tail"
            assert inputs.kwargs["rid"] == request_id and request_info.tag == request_id
            expected = requested if producer else counts[request_id]
            assert lengths[request_id] == expected
            assert outputs["frame"][0].flatten().tolist() == [request_id] * (requested if producer else 1)
            tails.append(request_id)
            if request_id == 60:
                raise ValueError("tail failed after actual computation")

        def batch_postprocess(self, outputs, request_infos, inputs, piecewise_runners=None):
            batch_tails.append(tuple(request_infos))

    eng = _engine(Tail.__new__(Tail))
    eng._enable_nvtx = False
    eng.postprocess_batch = lambda batch, outputs: Engine._postprocess_batch(eng, batch, outputs)
    owner = Owner()
    eng._runner = StepRunner({"owned": owner})
    eng._runner._generations = dict.fromkeys(rids, 1)
    mgmt = eng._submodules["llm"]
    mgmt.resources, mgmt.carries, mgmt.rings = {"owned": owner}, {}, {}
    mgmt.accelerator_graph_runner = None
    mgmt.run_columns = {"decode": frozenset({"frame"})}
    ctx = StepContext(rids, "decode", 0, False, repeats=requested, counted=dict.fromkeys(rids, 1))
    batch = ExecutingBatch("llm", dict.fromkeys(rids, SimpleNamespace()), ctx,
                           inputs=[NodeInputs(kwargs={"rid": rid}) for rid in rids], prepared_rids=tuple(rids))
    batch.per_request_info_wrapped = {rid: SimpleNamespace(tag=rid) for rid in rids}
    batch.step = SubmoduleStep(steps={"owned": ResourceStep()})
    batch.step.set_ctx(ctx)
    if producer:
        batch.ring_bases = {"frame": dict.fromkeys(rids, 0)}
    values = torch.tensor(rids).view(3, 1, 1).expand(3, requested, 1)
    batch.outputs = BatchedModelOutput(
        row_outputs={"frame": values}, row_request_ids=tuple(rids),
        check_stop_buffers={"token": values[..., 0], RUN_REPLAYS: torch.tensor([max(counts.values())]),
                            RUN_COUNTS: torch.tensor(list(counts.values())),
                            STREAM_COUNTS: torch.tensor([1, 1, 1]), STREAM_TRANSITION: torch.tensor([0, 1, 2])},
    )
    result = eng.collect_group_outputs(batch, max(counts.values()), counts)
    assert tails == [50, 60] and batch_tails == [(50,)]
    assert lengths == counts and batch.step_context.run_counts == counts
    assert set(batch.failed_requests) == {60}
    assert 40 not in result.per_rid_outputs
    assert result.row_request_ids == tuple(rids)
    assert result.check_stop_buffers[STREAM_COUNTS].tolist() == [1, 1, 1]
    assert result.check_stop_buffers[STREAM_TRANSITION].tolist() == [0, 1, 2]
    assert result.check_stop_buffers["owned"].tolist() == [0, 500, 600]
    assert sum(delta * len(rows) for delta, rows in rewinds) == 3 * requested - sum(counts.values())
    assert batch.request_ids == rids and len(batch.inputs) == 3 and not batch.run_rows
