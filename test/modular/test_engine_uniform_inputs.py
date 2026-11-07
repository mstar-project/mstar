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


def _connected_engine(transform=None, *, head_state=False, mixed=False, device="cpu", width=2,
                      walks=("prefill", "prefill")):
    """Real ordinary preparation/plan/collect, with replay standing in for CUDA."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphSlot
    from mstar.engine.engine import ExecutingBatch, ExecutionGroup, SubmoduleManagement
    from mstar.engine.resources.runner import StepRunner
    from mstar.engine.resources.step import StepContext
    from mstar.graph.runtime.base import ExecutionUnit
    from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, NodeSubmodule
    from mstar.worker.micro_scheduler import ScheduledBatch

    events = []
    device = torch.device(device)

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
            if mixed and self.name == "b":
                return ARNodeInputs(input_embeds=x)
            return NodeInputs(tensor_inputs={"x": x})

        def can_batch(self, **kw):
            return True

        def preprocess(self, graph_walk, engine_inputs, inputs):
            values = [inp.input_embeds if mixed and self.name == "b" else inp.tensor_inputs["x"] for inp in inputs]
            if mixed and self.name == "b":
                # Real-only work must stay real; its values precede future rows.
                values.insert(0, torch.ones(1, width, device=device).mul(7))
            return {"x": torch.cat(values)}

        def forward(self, *a, **kw):
            raise AssertionError("connected execution must replay the captures")

    eng = Engine.__new__(Engine)
    eng._enable_nvtx = eng._enable_profile = False
    eng._device = device
    eng._device_capability = SimpleNamespace(is_accelerator=device.type != "cpu")
    eng._keyed_walks = eng._loopback_steps = {}
    eng._pre_plan_done = {}
    eng._plan_peers = {"a": (), "b": ()}
    eng._runner = StepRunner({})
    eng._submodules = {}
    members = []
    leases = {}
    for name, walk in zip(("a", "b"), walks, strict=True):
        # Three captured rows, two real rows: stale producer padding must
        # never replace the consumer template's padding.
        config = BatchedAcceleratorGraphConfig(
            capture_graph_walk=walk,
            single_request_inputs=(ARNodeInputs(input_embeds=torch.ones(1, width, device=device).mul(9))
                                   if mixed and name == "b" else
                                   NodeInputs(tensor_inputs={"x": torch.ones(1, width, device=device).mul(9)})),
        )
        runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
        bucket = BucketKey(walk, 3, 0)
        leases[name] = SlotLease(0, bucket, name)
        inputs = torch.zeros(4 if name == "b" and mixed else 3, width, device=device)
        output = torch.full_like(inputs, -999)

        def replay(name=name, inputs=inputs, output=output):
            events.append(name)
            output.copy_(inputs + (10 if name == "a" else 100))

        graph = SimpleNamespace(replay=replay)
        if device.type == "cuda":
            from mstar.engine.accelerator_graph_backend import CUDAGraphBackend
            from mstar.engine.accelerator_graph_runner import capture_into_graph

            runner._graph_backend = CUDAGraphBackend(device)
            replay()
            graph, _ = capture_into_graph(
                lambda inputs=inputs, output=output, name=name: output.copy_(inputs + (10 if name == "a" else 100)),
                runner._graph_backend.graph_pool_handle(), device, None, keep_graph=True,
            )
            events.clear()
        slot = AcceleratorGraphSlot(
            graph=graph, static_inputs={"x": inputs}, static_input_keys=("x",),
            static_outputs=BatchedModelOutput(row_outputs={"x": output}),
            dummy_rids=[-1, -2, -3], dummy_metadata={-3: SimpleNamespace()}, config_idx=0,
            raw_launch=device.type == "cuda",
        )
        runner._buckets = {bucket: SimpleNamespace(config=config, slots=[slot])}
        runner._static_buffer_seq_dims = {}
        runner._submodule_name = name
        runner._device = device
        runner._generator_walks = set()
        runner.release = lambda lease, bs, name=name: events.append(f"release:{name}")
        mgmt = SubmoduleManagement(Node(name), None, None, None, {}, accelerator_graph_runner=runner)
        eng._submodules[name] = mgmt
        batch = ExecutingBatch(
            node_name=name, per_request_info={rid: SimpleNamespace(request_id=rid) for rid in (2, 1)},
            step_context=StepContext((2, 1), walk, 0, False),
            per_request_input_tensors=(
                {rid: {"x": [torch.full((1, width), rid * 1., device=device)]} for rid in (2, 1)}
                if name == "a" else {}
            ),
        )
        members.append((ScheduledBatch(name, walk, {2: 1, 1: 1}), batch))
    eng._chunking = ChunkingManagement(eng._submodules)

    def reserve(batch):
        lease = leases[batch.node_name]
        batch.lease_slot(lease)
        return lease

    eng.reserve_replay_slot = reserve
    unit = ExecutionUnit((1, 1), walks, ("a", "b"), (("a", "b", "x"),), ((), ("x",)))
    group = ExecutionGroup(unit, 1, tuple(members))
    return eng, group, events


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("trigger", [0, 1])
def test_connected_members_launch_once_with_ordered_real_and_future_rows(mixed, trigger):
    import threading

    eng, group, events = _connected_engine(mixed=mixed)
    group.trigger = trigger
    for _, member in group.members:
        member.launch_started_event = threading.Event()
    a, b = (eng._submodules[name] for name in ("a", "b"))
    b.slot_rotation = a.slot_rotation
    a.slot_rotation.count = 2

    def reserve(batch):
        mgmt = eng._submodules[batch.node_name]
        if batch.slot is None:
            batch.set_slot(mgmt.lease_slot())
        lease = SlotLease(batch.slot, BucketKey("prefill", 3, 0), batch.node_name)
        batch.lease_slot(lease)
        return lease

    eng.reserve_replay_slot = reserve
    runner = eng._submodules["a"].accelerator_graph_runner
    runs = []
    run = runner._run
    def launch(ops):
        assert group.members[trigger][1].launch_started_event.is_set()
        assert not group.members[1 - trigger][1].launch_started_event.is_set()
        runs.append(list(ops))
        return run(ops)
    runner._run = launch
    outputs = eng.exec_group(group)
    eng.release_group(group)
    assert group.submitted and group.refusal is None and len(runs) == 1
    assert a.next_slot == 1  # one component lease, shared by both members
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



def test_connected_output_binding_uses_the_producers_own_walk():
    from mstar.model.submodule_base import CarryPair

    eng, group, events = _connected_engine(walks=("producer", "consumer"))
    producer = eng._submodules["a"].submodule
    producer.repeat_carry = lambda walk, signals: {
        signal: CarryPair("x" if walk == "producer" else "absent", signal) for signal in signals
    }
    outputs = eng.exec_group(group)
    eng.release_group(group)
    assert group.submitted and group.refusal is None
    assert outputs[1].per_rid_outputs[2]["x"][0].tolist() == [[112., 112.]]
    assert events == ["a", "b", "release:b", "release:a"]

@pytest.mark.parametrize("transform,width", [
    (lambda x: x + 1, 2),
    (lambda x: torch.empty(2).copy_(x), 2),
    (lambda x: x.item(), 1),
    (lambda x: x.tolist(), 2),
    (bool, 1),
    (lambda x: None, 2),
])
def test_connected_actual_preparation_refuses_before_launch_without_failing_rows(transform, width):
    eng, group, events = _connected_engine(transform, width=width)
    assert eng.exec_group(group) is None
    assert group.refusal and not group.submitted
    assert not any(event in {"a", "b"} for event in events)
    assert all(not batch.failed_requests for _, batch in group.members)


def test_connected_head_cannot_mutate_request_state_before_a_later_refusal():
    eng, group, events = _connected_engine(head_state=True)
    assert eng.exec_group(group) is None
    assert not eng._submodules["a"].submodule.request_states
    assert events == []


@pytest.mark.parametrize("opaque", [False, True])
def test_connected_preparation_isolates_request_metadata_before_speculating(opaque):
    eng, group, events = _connected_engine()
    original_infos = [batch.per_request_info_wrapped for _, batch in group.members]
    nested = {"counter": 0}

    class Owned:
        def __deepcopy__(self, memo):
            raise AssertionError("opaque owner copying must never be attempted")

    for _, batch in group.members:
        for info in batch.per_request_info.values():
            info.step_metadata = {"state": Owned() if opaque else nested}
    node = eng._submodules["a"].submodule
    prepare = node.prepare_inputs
    calls = []

    def changing(**kwargs):
        calls.append(1)
        kwargs["fwd_info"].step_metadata["state"]["counter"] += 1
        return prepare(**kwargs)

    node.prepare_inputs = changing
    assert eng.exec_group(group) is None and not group.submitted
    assert nested == {"counter": 0}
    assert not calls if opaque else calls
    assert all(batch.per_request_info_wrapped is old
               for (_, batch), old in zip(group.members, original_infos, strict=True))
    assert not any(event in {"a", "b"} for event in events)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
def test_connected_cuda_launch_copies_future_values_and_preserves_external_outputs(monkeypatch):
    eng, group, events = _connected_engine(mixed=True, device="cuda")
    calls = []
    cuda = eng._submodules["a"].accelerator_graph_runner._graph_backend.while_runtime()
    launch = cuda.launch

    def launch_counted(*args):
        calls.append(args)
        return launch(*args)

    monkeypatch.setattr(cuda, "launch", launch_counted)
    outputs = eng.exec_group(group)
    eng.release_group(group)
    assert group.submitted and group.refusal is None and len(calls) == 1
    torch.cuda.synchronize()
    assert events == ["release:b", "release:a"]
    assert group.parts[1].slot.static_outputs.row_outputs["x"].tolist() == [
        [107., 107.], [112., 112.], [111., 111.], [109., 109.],
    ]
    # Reuse the producer slot after all GPU readers were queued. External
    # outputs retain their values even though the captured storage changes.
    group.parts[0].slot.static_outputs.row_outputs["x"].fill_(-1)
    assert outputs[0].per_rid_outputs[2]["x"][0].tolist() == [[12., 12.]]


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
    eng.release_group(group)
    assert group.submitted and len(outputs) == 2
    assert outputs[1].per_rid_outputs[2]["x"][0].tolist() == [[112., 112.]]
    assert events == ["a", "b", "release:b", "release:a"]


@pytest.mark.parametrize("fail_member", [None, "a", "b"])
def test_connected_worker_keeps_capture_leases_through_all_member_tails(fail_member):
    from contextlib import nullcontext

    from mstar.worker.worker import PendingBatch, Worker

    eng, group, events = _connected_engine()
    group.trigger = 1
    outputs = eng.exec_group(group)
    assert events == ["a", "b"]
    assert all(batch.step_context.slot_lease is not None for _, batch in group.members)
    event = SimpleNamespace(synchronize=lambda: events.append("await"))
    for _, batch in group.members:
        batch.completion_event = event
    worker = Worker.__new__(Worker)
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: eng)
    worker.request_state = SimpleNamespace(get_partition_for_node=lambda node: node)
    worker.scheduler = SimpleNamespace(clear_batch_limit=lambda *a, **kw: None)
    worker._phase_period = 0
    worker._metrics = None
    worker._span = lambda *a, **kw: nullcontext()
    worker._clear_in_flight_flag = lambda scheduled: None

    def finish(step, output, **kwargs):
        assert all(batch.step_context.slot_lease is not None for _, batch in group.members)
        assert output is group.members[0 if step.node_name == "a" else 1][1].outputs
        events.append(f"tail:{step.node_name}")
        if step.node_name == fail_member:
            raise ValueError("output routing failed")

    worker._postprocess_batch = finish
    scheduled, batch = group.members[group.trigger]
    pending = PendingBatch(scheduled, batch, scheduled.node_name, "p", scheduled.graph_walk, None, group=group)
    if fail_member is None:
        worker._finish_step(pending, outputs[group.trigger])
    else:
        with pytest.raises(ValueError, match="output routing failed"):
            worker._finish_step(pending, outputs[group.trigger])
    tails = ["tail:a"] if fail_member == "a" else ["tail:a", "tail:b"]
    assert events == ["a", "b", *tails, "await", "release:b", "release:a"]
    assert all(batch.step_context.slot_lease is None for _, batch in group.members)
    assert not group.parts
    # Main-loop failure cleanup can encounter this same group a second time.
    eng.release_group(group)
    assert events.count("release:a") == events.count("release:b") == 1


@pytest.mark.parametrize("kept", [frozenset(), frozenset({1})])
def test_connected_completion_keeps_ordinary_last_member_metadata(kept):
    eng, group, _ = _connected_engine()
    eng._loopback_steps = {1: ("a", "prefill"), 2: ("a", "prefill")}
    group.members[0][1].rows_only = True
    group.members[1][1].loopback_rids = kept
    outputs = eng.exec_group(group)
    eng.release_group(group)
    assert group.submitted
    assert outputs[1].loopback_rids == kept
    assert eng._loopback_steps == {rid: ("b", "prefill") for rid in kept}
    assert eng._loopback_rows(group.members[0][1]) == ([False, False] if kept else None)
    assert eng._loopback_rows(group.members[1][1]) == ([False, True] if kept else None)


@pytest.mark.parametrize("unsupported", ["run", "resident", "generator", "generator_alias", "early_stop"])
def test_connected_refuses_unstaged_captured_state_before_launch(unsupported):
    eng, group, events = _connected_engine()
    batch = group.members[0][1]
    runner = eng._submodules["a"].accelerator_graph_runner
    slot = next(iter(runner._buckets.values())).slots[0]
    if unsupported == "resident":
        batch.step_context.resident = object()
    elif unsupported == "run":
        slot.run = object()
    elif unsupported == "early_stop":
        eng._submodules["a"].submodule.get_stop_policies = lambda walk: {"loop": object()}
    elif unsupported in {"generator", "generator_alias"}:
        runner._generator_walks.add("prefill")
        if unsupported == "generator_alias":
            batch.step_context.graph_walk = "alias"
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
    assert not group.submitted and events == (["release:a"] if replacement == "per_request" else [])
    batch = group.members[0][1]
    if batch.inputs is None:
        eng.prepare_inputs(batch)
    eng.reserve_replay_slot(batch)
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
        eng.release_group(group)
        assert group.submitted and not group.refusal
        retained.append(outputs[1].per_rid_outputs[1]["x"][0].clone())
    assert len(compositions) == 1 and len(launches) == 2
    assert [out.tolist() for out in retained] == [[[112., 112.]], [[132., 132.]]]
    assert events == ["a", "b", "release:b", "release:a"] * 2


@pytest.mark.parametrize("repeats,device", [(1, "cpu"), (3, "cpu"), (3, "cuda")])
def test_connected_loop_repeats_from_its_first_seed_and_keeps_each_members_outputs(repeats, device):
    from dataclasses import replace

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("real connected CUDA Loop")
    eng, group, events = _connected_engine(device=device)
    group.unit = replace(group.unit, loop_name="loop", loop_back=(("b", "a", "x"),))
    group.capacity = 3
    for _, batch in group.members:
        batch.step_context.repeats = repeats
        eng._submodules[batch.node_name].run_columns["prefill"] = frozenset({"x"})
    outputs = eng.exec_group(group)
    eng.release_group(group)
    assert group.submitted and group.refusal is None
    assert events == (["a", "b"] * repeats if device == "cpu" else []) + ["release:b", "release:a"]
    for i, output in enumerate(outputs):
        got = output.per_rid_outputs[2]["x"][0]
        wanted = torch.tensor([[12. + i * 100. + 110. * n] * 2 for n in range(repeats)])
        if repeats > 1:
            wanted = wanted.unsqueeze(0)
        torch.testing.assert_close(got.cpu(), wanted)
    # The ordinary collection owns these values, beyond reuse of the captures.
    for part in group.parts:
        part.slot.static_outputs.row_outputs["x"].fill_(-999)
    assert outputs[0].per_rid_outputs[2]["x"][0].flatten()[0] == 12


@pytest.mark.parametrize("late", [False, True])
def test_connected_round_count_settles_before_admit_and_late_narrowing_unwinds_every_owner(late):
    from dataclasses import replace

    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.runner import StepRunner
    from mstar.engine.resources.step import ADMIT_OK, ResourceStep, SubmoduleStep

    admitted, rolled_back = [], []
    class Owner(Resource):
        @classmethod
        def build(cls, spec, info):
            return cls()

        def repeat_refusal(self, walk, host_step=None):
            return None

        def max_repeats(self, step, ctx):
            return 3 if late else 2

        def admit(self, step, ctx):
            admitted.append((ctx.unit_member, ctx.repeats))
            if late and ctx.unit_member == 1:
                ctx.repeats = 1
            return ADMIT_OK

        def rollback_admit(self, step, ctx):
            rolled_back.append(ctx.unit_member)

    eng, group, events = _connected_engine()
    group.unit = replace(group.unit, loop_name="loop", loop_back=(("b", "a", "x"),))
    group.capacity = 3
    owners = {name: Owner() for name in ("a", "b")}
    eng._runner = StepRunner(owners)
    for _, batch in group.members:
        name = batch.node_name
        batch.step_context.repeats = 3
        mgmt = eng._submodules[name]
        mgmt.resources = {name: owners[name]}
        mgmt.submodule.declare_step = lambda name=name, **kw: SubmoduleStep(steps={name: ResourceStep()})
    result = eng.exec_group(group)
    eng.release_group(group)
    if late:
        assert result is None and not group.submitted
        assert admitted == [(0, 3), (1, 3)] and rolled_back == [1, 0]
        assert events == ["release:b", "release:a"]
    else:
        assert result is not None and group.submitted
        assert admitted == [(0, 2), (1, 2)] and rolled_back == []
        assert events == ["a", "b"] * 2 + ["release:b", "release:a"]
        assert all(batch.step_context.repeats == 2 for _, batch in group.members)


@pytest.mark.parametrize("repeats,refuse,packed,device", [
    (1, False, False, "cpu"), (3, False, False, "cpu"), (1, True, False, "cpu"), (1, False, True, "cpu"),
    (3, False, False, "cuda"),
])
def test_connected_inputs_reach_captured_row_owners_only_after_submission(repeats, refuse, packed, device):
    from dataclasses import replace

    from mstar.engine.resources.loopback_rows import LoopbackRows

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("real captured row ownership")
    eng, group, events = _connected_engine(device=device)
    owners = []
    for name in ("a", "b"):
        mgmt = eng._submodules[name]
        bucket, captures = next(iter(mgmt.accelerator_graph_runner._buckets.items()))
        slot = captures.slots[0]
        rows = mgmt.loopback_rows = LoopbackRows(torch.device(device), capacity=4)
        for rid in (1, 2):  # allocation order differs from the batch order
            rows.add(rid)
        rows.write("x", (2, 1), torch.full((2, 2), -50., device=device))
        rows.indices(bucket, 0).copy_(torch.tensor([rows._row_of[2], rows._row_of[1], 0], device=device))
        slot.row_heads = {"x": "x"}
        replay = slot.graph.replay

        def replay_rows(rows=rows, bucket=bucket, slot=slot, replay=replay, name=name):
            rows.head(bucket, 0, "x", slot.static_inputs["x"])
            if device == "cuda":
                # Capture the forward with its owner reads/writes. Replaying
                # an already captured graph inside capture is unsupported.
                torch.add(slot.static_inputs["x"], 10 if name == "a" else 100,
                          out=slot.static_outputs.row_outputs["x"])
            else:
                replay()
            rows.tail(bucket, 0, "x", slot.static_outputs.row_outputs["x"])

        if device == "cuda":
            from mstar.engine.accelerator_graph_runner import capture_into_graph

            replay_rows()
            backend = mgmt.accelerator_graph_runner._graph_backend
            slot.graph, _ = capture_into_graph(
                replay_rows, backend.graph_pool_handle(), torch.device(device), None, keep_graph=True,
            )
        else:
            slot.graph.replay = replay_rows
        owners.append(rows)
    for _, batch in group.members:
        batch.step_context.repeats = repeats
        eng._submodules[batch.node_name].run_columns[batch.graph_walk] = frozenset({"x"})
    if repeats > 1:
        group.unit = replace(group.unit, loop_name="loop", loop_back=(("b", "a", "x"),))
        group.capacity = repeats
        # Aliased initial owner rows must be snapshotted before either write.
        owners[0].write("x", (2, 1), torch.tensor([[1., 1.], [2., 2.]], device=device))
        group.members[0][1].per_request_input_tensors = {
            2: {"x": [owners[0].row("x", 1).unsqueeze(0)]},
            1: {"x": [owners[0].row("x", 2).unsqueeze(0)]},
        }
    else:
        # A carried row already lives in the owner; its placeholder is not a seed.
        eng._loopback_steps = {1: ("a", "prefill")}
        group.members[0][1].per_request_input_tensors[1]["x"] = [torch.full((1, 2), -1000., device=device)]
        owners[0].write("x", (1,), torch.ones(1, 2, device=device))
    if packed:
        def preprocess(graph_walk, engine_inputs, inputs):
            a, b = (inp.tensor_inputs["x"].view(-1) for inp in inputs[:2])
            # A future span crosses two owner rows, with real spans at either end.
            padding = torch.full((1,), 9., device=device)
            return {"x": torch.cat((padding, a, b, padding)).view(3, 2)}

        eng._submodules["b"].submodule.preprocess = preprocess
    if refuse:
        node = eng._submodules["b"].submodule
        preprocess = node.preprocess

        def arithmetic(*args, **kwargs):
            values = preprocess(*args, **kwargs)
            return {"x": values["x"] + 1}

        node.preprocess = arithmetic
        before = [owner.read("x", (2, 1)).clone() for owner in owners]
        assert eng.exec_group(group) is None and not group.submitted
        assert not any(event in {"a", "b"} for event in events)
        assert all(torch.equal(owner.read("x", (2, 1)), values) for owner, values in zip(owners, before, strict=True))
        return
    outputs = eng.exec_group(group)
    eng.release_group(group)
    assert group.submitted and group.refusal is None
    for rid in (2, 1):
        expected = torch.tensor([[rid + 110 * (i + 1)] * 2 for i in range(repeats)])
        if packed:
            expected = torch.tensor([[109., 112.]] if rid == 2 else [[112., 111.]])
        assert torch.equal(outputs[1].per_rid_outputs[rid]["x"][0].reshape(repeats, 2).cpu(), expected)
        assert torch.equal(owners[1].row("x", rid).cpu(), expected[-1])


def test_dynamic_run_retains_capture_until_identity_resolved_collection_finishes(monkeypatch):
    from mstar.engine.device_run import RUN_DESCRIPTORS
    from mstar.model.submodule_base import BatchedModelOutput

    eng, group, events = _connected_engine()
    batch = group.members[0][1]
    eng.prepare_inputs(batch)
    eng.reserve_replay_slot(batch)
    raw = BatchedModelOutput(row_outputs={"x": torch.tensor([[[1.], [2.]]])},
                             check_stop_buffers={RUN_DESCRIPTORS: torch.tensor([[7, 8]])})
    monkeypatch.setattr(eng, "_drive_step", lambda *args, **kwargs: (raw, batch.step))
    assert eng._exec_single(batch) is raw
    assert batch.outputs is raw and "release:a" not in events
    assert eng._postprocess_outputs(batch, raw) is raw
    ordinary = BatchedModelOutput(per_rid_outputs={batch.request_ids[0]: {"x": [torch.tensor([[1.]])]}})
    eng.finish_run_outputs(batch, ordinary)
    assert events.count("release:a") == 1 and batch.outputs is ordinary
    with pytest.raises(ValueError, match="finish once"):
        eng.finish_run_outputs(batch, ordinary)
