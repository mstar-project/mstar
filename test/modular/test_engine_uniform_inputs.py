"""Engine.prepare_inputs: a walk whose rows all get the same inputs is
prepared with one call, the object shared by the rows; any other walk still
prepares row by row. A replay pads such a step with that object too."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig  # noqa: E402
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner  # noqa: E402
from mstar.engine.chunking import ChunkingManagement  # noqa: E402
from mstar.engine.engine import Engine  # noqa: E402
from mstar.engine.resources.step import BucketKey, SlotLease  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs  # noqa: E402


def _connected_engine(transform=None, *, head_state=False, mixed=False):
    """Real ordinary preparation/plan/collect, with replay standing in for CUDA."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphSlot
    from mstar.engine.engine import ExecutingBatch, ExecutionGroup, SubmoduleManagement
    from mstar.engine.resources.runner import StepRunner
    from mstar.engine.resources.step import StepContext
    from mstar.graph.runtime.base import ExecutionUnit
    from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, NodeSubmodule
    from mstar.worker.micro_scheduler import ScheduledBatch

    events = []

    class Node(NodeSubmodule):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def prepare_inputs(self, inputs, fwd_info, **kw):
            if head_state and self.name == "a":
                self.request_state(fwd_info.request_id).kwargs["count"] = 1
            x = inputs["x"][0]
            if self.name == "b" and transform is not None:
                x = transform(x)
                if x is None:
                    return None
            return NodeInputs(tensor_inputs={"x": x})

        def can_batch(self, **kw):
            return True

        def preprocess(self, graph_walk, engine_inputs, inputs):
            values = [inp.tensor_inputs["x"] for inp in inputs]
            if mixed and self.name == "b":
                # Real-only work must stay real; its values precede future rows.
                values.insert(0, torch.ones(1, 2).mul(7))
            return {"x": torch.cat(values)}

        def forward(self, *a, **kw):
            raise AssertionError("connected execution must replay the captures")

    eng = Engine.__new__(Engine)
    eng._enable_nvtx = eng._enable_profile = False
    eng._device = torch.device("cpu")
    eng._device_capability = SimpleNamespace(is_accelerator=False)
    eng._keyed_walks = eng._loopback_steps = {}
    eng._runner = StepRunner({})
    eng._submodules = {}
    members = []
    leases = {}
    for name in ("a", "b"):
        # Three captured rows, two real rows: stale producer padding must
        # never replace the consumer template's padding.
        config = BatchedAcceleratorGraphConfig(
            capture_graph_walk="prefill",
            single_request_inputs=NodeInputs(tensor_inputs={"x": torch.ones(1, 2).mul(9)}),
        )
        runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
        bucket = BucketKey("prefill", 3, 0)
        leases[name] = SlotLease(0, bucket)
        inputs = torch.zeros(4 if name == "b" and mixed else 3, 2)
        output = torch.full_like(inputs, -999)

        def replay(name=name, inputs=inputs, output=output):
            events.append(name)
            output.copy_(inputs + (10 if name == "a" else 100))

        slot = AcceleratorGraphSlot(
            graph=SimpleNamespace(replay=replay), static_inputs={"x": inputs}, static_input_keys=("x",),
            static_outputs=BatchedModelOutput(row_outputs={"x": output}),
            dummy_rids=[-1, -2, -3], dummy_metadata={-3: SimpleNamespace()}, config_idx=0,
        )
        runner._buckets = {bucket: SimpleNamespace(config=config, slots=[slot])}
        runner._static_buffer_seq_dims = {}
        runner._submodule_name = name
        runner._device = torch.device("cpu")
        runner._generator_walks = set()
        runner.release = lambda lease, bs, name=name: events.append(f"release:{name}")
        mgmt = SubmoduleManagement(Node(name), None, None, None, {}, accelerator_graph_runner=runner)
        eng._submodules[name] = mgmt
        batch = ExecutingBatch(
            node_name=name, per_request_info={rid: SimpleNamespace(request_id=rid) for rid in (2, 1)},
            step_context=StepContext((2, 1), "prefill", 0, False),
            per_request_input_tensors=(
                {rid: {"x": [torch.full((1, 2), rid * 1.)]} for rid in (2, 1)} if name == "a" else {}
            ),
        )
        members.append((ScheduledBatch(name, "prefill", {2: 1, 1: 1}), batch))
    eng._chunking = ChunkingManagement(eng._submodules)

    def reserve(batch):
        lease = leases[batch.node_name]
        batch.lease_slot(lease)
        return lease

    eng.reserve_replay_slot = reserve
    group = ExecutionGroup(ExecutionUnit(1, ("a", "b"), (("a", "b", "x"),), ((), ("x",))), 1, tuple(members))
    return eng, group, events


@pytest.mark.parametrize("mixed", [False, True])
def test_connected_members_launch_once_with_ordered_real_and_future_rows(mixed):
    eng, group, events = _connected_engine(mixed=mixed)
    runner = eng._submodules["a"].accelerator_graph_runner
    runs = []
    run = runner._run
    runner._run = lambda ops: (runs.append(list(ops)), run(ops))[-1]
    outputs = eng.exec_group(group)
    assert group.submitted and group.refusal is None and len(runs) == 1
    assert events == ["a", "b", "release:b", "release:a"]
    assert outputs[0].per_rid_outputs[2]["x"][0].tolist() == [[12., 12.]]
    b_slot = group.parts[1].slot
    expected = [[107., 107.], [112., 112.], [111., 111.], [109., 109.]] if mixed else [
        [112., 112.], [111., 111.], [109., 109.],
    ]
    assert b_slot.static_outputs.row_outputs["x"].tolist() == expected
    # Output collection made durable copies before either slot was released.
    group.parts[0].slot.static_outputs.row_outputs["x"].fill_(-1)
    assert outputs[0].per_rid_outputs[2]["x"][0].tolist() == [[12., 12.]]


@pytest.mark.parametrize("transform", [
    lambda x: x + 1,
    lambda x: torch.empty(2).copy_(x),
    lambda x: x[0].item(),
    lambda x: x.tolist(),
    lambda x: bool(x[0]),
    lambda x: None,
])
def test_connected_actual_preparation_refuses_before_launch_without_failing_rows(transform):
    eng, group, events = _connected_engine(transform)
    assert eng.exec_group(group) is None
    assert group.refusal and not group.submitted
    assert not any(event in {"a", "b"} for event in events)
    assert all(not batch.failed_requests for _, batch in group.members)


def test_connected_head_cannot_mutate_request_state_before_a_later_refusal():
    eng, group, events = _connected_engine(head_state=True)
    assert eng.exec_group(group) is None
    assert not eng._submodules["a"].submodule.request_states
    assert events == []


class _Batch:
    def __init__(self, rids, walk="decode"):
        self.node_name = "llm"
        self.request_ids = list(rids)
        self.step_context = SimpleNamespace(
            graph_walk=walk, request_walks={}, walk_of=lambda _r: walk, repeats=1,
        )
        self.per_request_info_wrapped = {r: SimpleNamespace() for r in rids}
        self.per_request_input_tensors = {}
        self.per_request_input_metadata = {}
        self.final_stream_rids = set()
        self.skipped_rids = set()
        self.failed_requests = {}
        self.inputs = None
        self.uniform_inputs = None
        self.running_batched = None
        self.rows_only = False
        self.loopback_rids = frozenset()
        self.chunk_ranges = {}

    def register_prepare_batch(self, inputs, uniform=None):
        self.inputs = inputs
        self.uniform_inputs = uniform

    def drop_rids(self, rids):
        self.request_ids = [r for r in self.request_ids if r not in rids]

    def register_failure(self, rid, error):
        self.failed_requests[rid] = str(error)


class _Sub:
    def __init__(self, uniform=None):
        self.uniform = uniform
        self.calls = []
        self.row_loopback = []
        self.uniform_loopback = None

    def uniform_row_inputs(self, graph_walk, has_loopback=False):
        self.uniform_loopback = has_loopback
        return self.uniform

    def prepare_inputs(self, **kw):
        self.calls.append(kw["graph_walk"])
        self.row_loopback.append(kw["has_loopback"])
        self.row_inputs = getattr(self, "row_inputs", []) + [kw["inputs"]]
        return SimpleNamespace(input_seq_len=1)

    def can_batch(self, batch, model_inputs):
        return len(model_inputs) > 0


def _engine(sub, rows=None, row_signals=None):
    eng = Engine.__new__(Engine)
    eng._submodules = {"llm": SimpleNamespace(
        submodule=sub, resources={}, loopback_rows=rows, row_signals=row_signals or {},
    )}
    eng._enable_nvtx = False
    eng._loopback_steps = {}
    # no policies: these steps are not chunked
    eng._chunking = ChunkingManagement(eng._submodules)
    return eng


def test_a_uniform_walk_is_prepared_with_one_object():
    row = object()
    sub = _Sub(uniform=row)
    batch = _Batch([1, 2, 3])
    _engine(sub).prepare_inputs(batch)
    assert batch.inputs == [row, row, row] and all(x is row for x in batch.inputs)
    assert batch.uniform_inputs is row
    assert sub.calls == [] and batch.running_batched is True


def test_other_walks_still_prepare_row_by_row():
    sub = _Sub(uniform=None)
    batch = _Batch([1, 2])
    _engine(sub).prepare_inputs(batch)
    assert sub.calls == ["decode", "decode"] and len(batch.inputs) == 2


def test_loopback_is_per_row_and_tracks_each_request_s_last_step():
    sub = _Sub(uniform=None)
    eng = _engine(sub)
    ran = _Batch([1, 2])
    ran.rows_only = True
    eng._record_loopback(ran)
    eng.prepare_inputs(_Batch([1, 2, 3]))
    # 3 never ran rows-only here (e.g. its prefill ran on another rank)
    assert sub.uniform_loopback is False and sub.row_loopback == [True, True, False]
    eng.prepare_inputs(_Batch([1, 2]))
    assert sub.uniform_loopback is True
    # a step that is not rows-only clears the record; so does another walk's
    other = _Batch([2])
    other.rows_only = False
    eng._record_loopback(other)
    sub.row_loopback.clear()
    eng.prepare_inputs(_Batch([1, 2]))
    assert sub.row_loopback == [True, False]
    sub.row_loopback.clear()
    eng.prepare_inputs(_Batch([1], walk="prefill"))
    assert sub.row_loopback == [False]


def test_a_mixed_step_records_loopback_under_each_row_s_own_walk():
    sub = _Sub(uniform=None)
    eng = _engine(sub)
    walks = {1: "decode", 2: "decode", 3: "prefill"}
    mixed = _Batch([1, 2, 3], walk="mixed")
    mixed.step_context.request_walks = walks
    mixed.step_context.walk_of = walks.__getitem__
    # rows 1 and 2 kept their token on the device; 3's prompt row routed a tensor
    mixed.loopback_rids = frozenset({1, 2})
    eng._record_loopback(mixed)
    eng.prepare_inputs(_Batch([1, 2, 3]))
    assert sub.row_loopback == [True, True, False]


def _runner(config, bs=4):
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    bucket = BucketKey(graph_walk="decode", bs=bs, num_tokens=bs)
    runner._buckets = {bucket: SimpleNamespace(config=config)}
    return runner, SlotLease(slot=0, bucket=bucket)


def _decode_config():
    return BatchedAcceleratorGraphConfig(
        capture_graph_walk="decode",
        single_request_inputs=ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=1),
    )


def test_a_uniform_step_pads_with_its_shared_row():
    """The row any request of the walk gets is as good a padding row as a
    clone of the template, and costs nothing per row."""
    runner, lease = _runner(_decode_config())
    row = ARNodeInputs(input_seq_len=1)
    padded = runner.pad_inputs(lease, [row, row], uniform=row)
    assert len(padded) == 4 and all(x is row for x in padded)


def test_other_steps_pad_with_template_clones():
    runner, lease = _runner(_decode_config())
    row = ARNodeInputs(input_seq_len=1)
    other = ARNodeInputs(input_ids=torch.ones(1, dtype=torch.long), input_seq_len=1)
    for inputs, uniform in (
        ([row, row], None),
        # a row that is not the shared one: the step is not uniform
        ([row, other], row),
        # a shared row of another span would change what the padding declares
        ([ARNodeInputs(input_seq_len=2)] * 2, ARNodeInputs(input_seq_len=2)),
    ):
        if uniform is not None and inputs[0].input_seq_len == 2:
            inputs = [uniform] * 2
        padded = runner.pad_inputs(lease, inputs, uniform=uniform)
        assert padded[:2] == inputs
        assert all(p.input_ids is not None and p.input_seq_len == 1 for p in padded[2:])
        assert padded[2] is not padded[3]


def test_a_row_the_node_kept_reads_its_kept_signal_off_its_row():
    """A loop-back signal kept in the node's rows (``Engine.rows_refusal``)
    arrives empty on a row whose last step here kept it; the engine hands
    ``prepare_inputs`` its row as the routed value would be."""
    from mstar.engine.resources.loopback_rows import LoopbackRows

    rows = LoopbackRows(torch.device("cpu"))
    for rid in (1, 2):
        rows.add(rid)
    # the last step's tail
    rows.write("h", [1, 2], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    sub = _Sub(uniform=None)
    eng = _engine(sub, rows, {"decode": {"h": "h"}})
    ran = _Batch([1])
    ran.rows_only = True
    eng._record_loopback(ran)
    batch = _Batch([1, 2])
    batch.per_request_input_tensors = {2: {"h": [torch.tensor([9.0, 9.0])]}}
    eng.prepare_inputs(batch)
    assert sub.row_loopback == [True, False]
    (kept,), (routed,) = sub.row_inputs[0]["h"], sub.row_inputs[1]["h"]
    assert kept.tolist() == [1.0, 2.0] and routed.tolist() == [9.0, 9.0]



def test_connected_capture_can_run_when_the_eager_forward_does_not_batch():
    eng, group, events = _connected_engine()
    for mgmt in eng._submodules.values():
        mgmt.submodule.can_batch = lambda **kw: False
    outputs = eng.exec_group(group)
    assert group.submitted and len(outputs) == 2
    assert outputs[1].per_rid_outputs[2]["x"][0].tolist() == [[112., 112.]]
    assert events == ["a", "b", "release:b", "release:a"]


@pytest.mark.parametrize("kept", [frozenset(), frozenset({1})])
def test_connected_completion_keeps_ordinary_last_member_metadata(kept):
    eng, group, _ = _connected_engine()
    eng._loopback_steps = {1: ("a", "prefill"), 2: ("a", "prefill")}
    group.members[0][1].rows_only = True
    group.members[1][1].loopback_rids = kept
    outputs = eng.exec_group(group)
    assert group.submitted
    assert outputs[1].loopback_rids == kept
    assert eng._loopback_steps == {rid: ("b", "prefill") for rid in kept}
    assert eng._loopback_rows(group.members[0][1]) == ([False, False] if kept else None)
    assert eng._loopback_rows(group.members[1][1]) == ([False, True] if kept else None)


@pytest.mark.parametrize("unsupported", ["run", "resident", "row_heads", "generator"])
def test_connected_refuses_unstaged_captured_state_before_launch(unsupported):
    eng, group, events = _connected_engine()
    batch = group.members[0][1]
    runner = eng._submodules["a"].accelerator_graph_runner
    slot = next(iter(runner._buckets.values())).slots[0]
    if unsupported == "resident":
        batch.step_context.resident = object()
    elif unsupported == "run":
        slot.run = object()
    elif unsupported == "generator":
        runner._generator_walks.add("prefill")
    else:
        slot.row_heads = {"x": "x"}
    assert eng.exec_group(group) is None
    assert not group.submitted
    assert events == ["release:a"]


@pytest.mark.parametrize("replacement", ["per_request", "unpacked"])
def test_connected_preserves_ordinary_output_precedence(replacement, monkeypatch):
    eng, group, events = _connected_engine()
    mgmt = eng._submodules["a"]
    slot = next(iter(mgmt.accelerator_graph_runner._buckets.values())).slots[0]
    changed = {2: {"x": [torch.full((1, 2), 42.)]}}
    if replacement == "per_request":
        slot.static_outputs.per_rid_outputs = {slot.dummy_rids[0]: changed[2]}
    else:
        monkeypatch.setattr(type(mgmt.submodule), "unpack_packed_outputs", lambda *a, **kw: changed)
    assert eng.exec_group(group) is None
    assert not group.submitted and events == ["release:a"]
    batch = group.members[0][1]
    ordinary = eng._collect_step(batch, mgmt, batch.step_context.slot_lease, slot.static_outputs,
                                 batch.inputs, batch.per_request_info_wrapped)
    assert ordinary.per_rid_outputs[2]["x"][0].tolist() == [[42., 42.]]


def test_composed_unit_reuses_capture_bindings_across_new_real_inputs():
    from dataclasses import replace

    eng, original, events = _connected_engine(mixed=True)
    runner = eng._submodules["a"].accelerator_graph_runner
    runner._device = torch.device("cuda")
    compositions, launches = [], []

    def compose(ops):
        compositions.append(tuple(ops))

        def replay():
            launches.append(1)
            for op in ops:
                if isinstance(op, tuple):
                    op[0].copy_(op[1])
                else:
                    op.replay()

        return SimpleNamespace(replay=replay)

    runner._graph_backend = SimpleNamespace(compose=compose)
    retained = []
    for offset in (0, 20):
        group = replace(original, parts=[], initial_copies=[])
        head = group.members[0][1]
        head.per_request_input_tensors = {
            rid: {"x": [torch.full((1, 2), float(rid + offset))]} for rid in (2, 1)
        }
        outputs = eng.exec_group(group)
        assert group.submitted and not group.refusal
        retained.append(outputs[1].per_rid_outputs[1]["x"][0].clone())
    assert len(compositions) == 1 and len(launches) == 2
    assert [out.tolist() for out in retained] == [[[112., 112.]], [[132., 132.]]]
    assert events == ["a", "b", "release:b", "release:a"] * 2
