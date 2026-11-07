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


def test_connected_consumer_drains_without_preparing_or_leasing_its_finished_producer():
    from dataclasses import replace

    eng, group, events = _connected_engine(walks=("producer", "consumer"))
    group.unit = replace(group.unit, streaming_edges=group.unit.edges)
    group.trigger = 1
    group.finished_rids = (frozenset({1, 2}), frozenset())
    producer, consumer = [batch for _, batch in group.members]
    # The completed partition may have no remaining inputs/capture/forward info.
    producer.per_request_input_tensors = {}
    producer.per_request_info = {rid: object() for rid in (2, 1)}
    eng._submodules["a"].accelerator_graph_runner = None
    consumer.per_request_input_tensors = {
        rid: {"x": [torch.full((1, 2), 10. + rid)]} for rid in (2, 1)
    }
    output = eng.exec_group(group)
    eng.release_group(group)
    assert group.submitted and group.refusal is None
    assert output[0] is producer.outputs and not output[0].per_rid_outputs
    assert output[1].per_rid_outputs[2]["x"][0].tolist() == [[112., 112.]]
    assert events == ["b", "release:b"]
    assert producer.step is None and producer.step_context.slot_lease is None

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


def test_connected_native_copy_refuses_strided_packing_before_staging_or_submission():
    eng, group, events = _connected_engine()
    # Exercise accelerator copy eligibility without needing to submit a GPU graph.
    eng._device = torch.device("cuda")
    node = eng._submodules["b"].submodule
    node.preprocess = lambda graph_walk, engine_inputs, inputs: {
        "x": torch.cat([item.tensor_inputs["x"] for item in inputs[:2]]).t(),
    }
    slots = [next(iter(m.accelerator_graph_runner._buckets.values())).slots[0]
             for m in eng._submodules.values()]
    before = [slot.static_inputs["x"].clone() for slot in slots]
    assert eng.exec_group(group) is None and not group.submitted
    assert "copies" in group.refusal
    assert not any(event in {"a", "b"} for event in events)
    assert torch.equal(slots[1].static_inputs["x"], before[1])


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


@pytest.mark.parametrize("change", ["none", "host", "real", "empty"])
def test_streamed_alternatives_preserve_host_meaning_and_do_not_read_future_values(change):
    from mstar.engine.connected_inputs import ConnectedInputs
    from mstar.streaming.chunk_policy import FixedChunkPolicy
    from mstar.streaming.stream_buffer import StreamBuffer, StreamingEdge

    eng, group, events = _connected_engine()
    batch = group.members[1][1]
    batch.request_ids = [2]
    batch.per_request_info = {2: batch.per_request_info[2]}
    source = StreamBuffer(2, "x", "a", FixedChunkPolicy(2))
    if change == "empty":
        source.signal_done(0)
    snapshot = source.reserve_future_window([] if change == "empty" else range(4), 3, may_finish=False)
    batch.stream_windows = {2: [StreamingEdge("x", "b", [], snapshot.info)]}
    node = eng._submodules["b"].submodule
    ordinary = node.prepare_inputs
    seen = []

    def prepare(**kwargs):
        chunk = kwargs["input_metadata"].stream_chunks["x"]
        assert chunk.reservation is None, "model preparation cannot mutate the stream owner's policy"
        seen.append(chunk.start_offset)
        if chunk.num_items == 0:
            return None
        inputs = ordinary(**kwargs)
        if change == "host":
            inputs.kwargs["offset"] = chunk.start_offset
        elif change == "real":
            inputs.tensor_inputs["independent"] = torch.tensor([chunk.start_offset])
        return inputs

    node.prepare_inputs = prepare
    guard = ConnectedInputs({})
    with guard.preparing():
        if change in ("host", "real"):
            with pytest.raises(ValueError, match="prepared host inputs"):
                eng._prepare_stream_choices(batch, "a", "x", torch.empty(4, 2), guard)
        else:
            choices = eng._prepare_stream_choices(batch, "a", "x", torch.empty(4, 2), guard)
            assert all(choice is None for choice in choices) if change == "empty" else all(choices)
    assert seen and events == []
    assert source._executed_chunks == 0 and source.owns_window(snapshot.reservation)
    assert batch.request_ids == [2] and batch.inputs is None
    assert not batch.failed_requests and not batch.skipped_rids


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["live", "full", "yield", "unsupported", "bank_small",
                                  "tail", "tail_error", "tail_zero", "tail_wide", "producer_tail",
                                  "packing", "packing_refuse", "direct", "direct_zero", "direct_eos",
                                  "direct_error", "direct_missing", "direct_walk", "direct_alias",
                                  "direct_spans", "direct_absent", "direct_wide", "direct_shared"])
def test_streamed_engine_composes_independent_bodies_and_owner_reads(device_name, mode, monkeypatch):
    from dataclasses import replace

    from mstar.engine.accelerator_graph_backend import CUDAGraphBackend, GraphConditional
    from mstar.engine.accelerator_graph_runner import capture_into_graph
    from mstar.engine.device_run import RUN_COUNTS, DeviceRun, RunState
    from mstar.engine.resources.iteration_tables import STREAM_COUNTS, IterationTables
    from mstar.engine.resources.loopback_rows import LoopbackRows
    from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, StopKind, StopPolicy
    from mstar.streaming.chunk_policy import FixedChunkPolicy
    from mstar.streaming.stream_buffer import StreamBuffer

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("real Engine streamed composition")
    device = torch.device(device_name)
    direct = mode.startswith("direct")
    full, yielded = mode == "full", mode in {"yield", "tail_zero", "direct_zero"}
    packing = mode.startswith("packing")
    custom = mode.startswith("tail") or mode == "producer_tail" or packing
    tails, batch_tails, producer_tails = [], [], []
    eng, group, events = _connected_engine(device=device)
    group.unit = replace(group.unit, streaming_edges=group.unit.edges)
    source = StreamBuffer(2, "x", "a", FixedChunkPolicy(2))
    ring = LoopbackRows(device, capacity=1)
    ring.add(2)
    ring.ask_ring("x", 8)
    ring.ring_table("x", torch.zeros(1, 2, device=device))
    if custom:
        # A prior refused producer left physical holes. Semantic offset zero
        # starts at position three, not at the ring's first storage element.
        ring.ring_reserve("x", [2], 3)
        ring.ring_write("x", [2], torch.full((1, 3, 2), -999., device=device), [0])
        ring.ring_skip("x", 2, [0, 1, 2])
    tables = IterationTables(device)
    eng._submodules["a"].loopback_rows = ring
    eng._submodules["a"].rings = {"x": {"prefill"}}
    eng._submodules["b"].submodule.iteration_tables = tables
    leases, slots, launches = {}, {}, []

    def cpu_compose(ops):
        def execute(ops):
            for op in ops:
                if isinstance(op, GraphConditional):
                    if op.kind == "if":
                        if op.predicate.item():
                            execute(op.body)
                    else:
                        while op.predicate.item():
                            execute(op.body)
                else:
                    op.replay()
        return SimpleNamespace(replay=lambda: (launches.append(1), execute(ops))[-1])

    backend = (CUDAGraphBackend(device) if device_name == "cuda" else
               SimpleNamespace(compose=cpu_compose, graph_pool_handle=lambda: None))
    if device_name == "cpu":
        monkeypatch.setattr("mstar.engine.accelerator_graph_runner.capture_into_graph",
                            lambda fn, *a, **kw: (SimpleNamespace(replay=fn), None))
    else:
        runtime = backend.while_runtime()
        launch = runtime.launch
        def launch_counted(*args):
            launches.append(1)
            return launch(*args)
        monkeypatch.setattr(runtime, "launch", launch_counted)

    def capture(fn):
        if device_name == "cpu":
            return SimpleNamespace(replay=fn)
        return capture_into_graph(fn, backend.graph_pool_handle(), device, None, keep_graph=True)[0]

    for name, width in (("a", 2), ("b", 4)):
        mgmt = eng._submodules[name]
        if custom and name == ("a" if mode == "producer_tail" else "b"):
            class Tail(type(mgmt.submodule)):
                def postprocess(self, request_id, request_info, outputs, inputs, **kwargs):
                    assert inputs.kwargs["context"] == 1
                    assert request_info.metadata.stream_chunks["x"].num_items == 2
                    actual = inputs.tensor_inputs["x"].reshape(-1)
                    tails.append(actual.tolist())
                    assert source._reservations, "the real input reader survives its ordinary tail"
                    if mode == "tail_error":
                        raise ValueError("ordinary tail failed")
                    outputs["x"] = [actual[1:3]]

                def batch_postprocess(self, outputs, request_infos, inputs, **kwargs):
                    batch_tails.append(tuple(request_infos))
                    assert outputs.get(2)["x"][0].tolist() == [10, 11]
            mgmt.submodule.__class__ = Tail
        if direct and name == "a":
            class Direct(type(mgmt.submodule)):
                DIRECT_OUTPUTS = {"wrong" if mode == "direct_walk" else "prefill": frozenset({"x"})}

                def postprocess(self, request_id, request_info, outputs, inputs, **kwargs):
                    producer_tails.append(request_id)
                    if mode == "direct_error":
                        raise ValueError("producer tail failed")
            mgmt.submodule.__class__ = Direct
            if mode == "direct_missing":
                mgmt.submodule.DIRECT_OUTPUTS = {}
            if mode == "direct_alias":
                from mstar.model.submodule_base import CarryPair
                mgmt.submodule.REPEAT_CARRIES = {"x": CarryPair("other", "x")}
        if mode == "direct_error" and name == "b":
            class DependentTail(type(mgmt.submodule)):
                def postprocess(self, *args, **kwargs):
                    pytest.fail("failed predecessor cannot run a successful dependent tail")
            mgmt.submodule.__class__ = DependentTail
        runner = mgmt.accelerator_graph_runner
        runner._graph_backend = backend
        old_bucket = next(iter(runner._buckets.values()))
        slot = old_bucket.slots[0]
        input_shape = (1, 2, 3) if packing and name == "b" else (1, width)
        slot.static_inputs = {"x": torch.zeros(input_shape, device=device)}
        slot.static_outputs = BatchedModelOutput(
            row_outputs={"x": torch.zeros(1, 2 if name == "a" else 1, device=device)},
            check_stop_buffers={"token": torch.zeros(1, dtype=torch.long, device=device)},
        )
        if direct and name == "a" and mode == "direct_spans":
            slot.static_outputs.output_spans = {"x": torch.ones(1, dtype=torch.long, device=device)}
        slot.dummy_rids, slot.dummy_metadata = [-1], {}
        config = BatchedAcceleratorGraphConfig(
            capture_graph_walk="prefill", single_request_inputs=NodeInputs(
                tensor_inputs={"x": torch.ones(input_shape, device=device)},
            ),
        )
        bucket = BucketKey("prefill", 1, 0)
        leases[name] = SlotLease(0, bucket, name)
        runner._buckets = {bucket: SimpleNamespace(config=config, slots=[slot])}
        mgmt.joint_comm_group = SimpleNamespace(world_size=1)
        mgmt.run_columns["prefill"] = frozenset({"x"})
        mgmt.device_stops["prefill"] = StopPolicy(StopKind.EOS, "token", frozenset({99}),
                                                  withhold=("x",) if name == "a" else ())
        mgmt.refills["prefill"] = None
        state = RunState(1, "token", (99,), device)
        run = slot.run = DeviceRun(state, {}, lambda: None, 4)
        run.rows["x"] = torch.zeros(1, 4, 2 if name == "a" else 1, device=device)
        run.stops["token"] = torch.zeros(1, 4, dtype=torch.long, device=device)
        if name == "b":
            owner = mgmt.loopback_rows = LoopbackRows(device, capacity=1)
            owner.add(2)
            owner.write("x", [2], torch.full(input_shape, -100., device=device))
            owner.indices(bucket, 0).fill_(1)
            slot.row_heads = {"x": "x"}
            ordinary = mgmt.submodule.prepare_inputs
            def prepare(ordinary=ordinary, **kwargs):
                result = ordinary(**kwargs)
                shape = (-1, 2) if packing else (1, -1)
                result.tensor_inputs["x"] = result.tensor_inputs["x"].reshape(shape)
                if custom:
                    result.kwargs["context"] = 1
                return result
            mgmt.submodule.prepare_inputs = prepare
            if packing:
                def preprocess(graph_walk, engine_inputs, inputs):
                    return {"x": torch.stack([
                        torch.nn.functional.pad(item.tensor_inputs["x"], (0, 0, 0, 1)).t() for item in inputs
                    ])}
                mgmt.submodule.preprocess = preprocess

        def entry(state=state, name=name):
            state.index.zero_()
            state.counts.zero_()
            state.marks.zero_()
            state.continuation.fill_(0 if yielded and name == "a" else 1)

        def body(name=name, slot=slot, run=run, state=state, mgmt=mgmt, bucket=bucket):
            if name == "a":
                slot.static_outputs.row_outputs["x"].copy_((state.index + (11 if direct else 10)).expand(1, 2))
                slot.static_outputs.check_stop_buffers["token"].copy_(
                    torch.where(state.index == (0 if mode == "direct_eos" else 2), 99, 1),
                )
            else:
                mgmt.loopback_rows.head(bucket, 0, "x", slot.static_inputs["x"])
                slot.static_outputs.row_outputs["x"].copy_(slot.static_inputs["x"].sum().reshape(1, 1))
                slot.static_outputs.check_stop_buffers["token"].fill_(1)
            run.keep(slot.static_outputs.row_outputs, run.rows)
            run.keep(slot.static_outputs.check_stop_buffers, run.stops)
            state.index.add_(1)
            state.counts.add_(1)
            state.marks.copy_(slot.static_outputs.check_stop_buffers["token"] == 99)
            state.continuation.copy_((state.index < state.limit) & ~state.marks)

        run.entry, slot.graph = capture(entry), capture(body)
        slots[name] = slot
        if name == "a" and mode == "direct_absent":
            slot.static_outputs.row_outputs.pop("x")
        scheduled, batch = next(pair for pair in group.members if pair[1].node_name == name)
        batch.request_ids = [2]
        batch.per_request_info = {2: batch.per_request_info[2]}
        batch.per_request_info_wrapped = {2: batch.per_request_info_wrapped[2]}
        batch.step_context.repeats = scheduled.repeats = (4 if name == "a" else
                                                        1 if custom and mode != "tail_wide" else 3)
        scheduled.request_to_worker_graph = {2: 1}
        if (direct and name == "a") or mode == "direct_error":
            batch.step_context.repeats = scheduled.repeats = 2 if mode == "direct_wide" else 1

    def reserve(batch):
        lease = leases[batch.node_name]
        batch.lease_slot(lease)
        return lease
    eng.reserve_replay_slot = reserve
    a, b = (pair[1] for pair in group.members)
    from mstar.worker.stream_buffers import StreamBufferManager
    from mstar.worker.worker import PendingBatch, Worker

    worker = Worker.__new__(Worker)
    worker.engine_manager = SimpleNamespace(get_engine=lambda name: eng)
    worker.streams = StreamBufferManager(SimpleNamespace(
        per_request_info={2: SimpleNamespace(stream_buffers={"x": source})},
    ), None, [])
    worker.streams.consumer_node = {"x": "b"}
    worker.streams.rings = {"x": ring}
    if full:
        ring.ring_reserve("x", [2], 8)
        ring.ring_write("x", [2], torch.arange(10, 18, device=device).reshape(1, 8, 1).expand(1, 8, 2), [0])
        source.put_positions(list(range(8)))
    if direct:
        n = 2 if mode == "direct_zero" else 1
        ring.ring_reserve("x", [2], n)
        ring.ring_write("x", [2], torch.arange(10, 10 + n, device=device).reshape(1, n, 1).expand(1, n, 2), [0])
        source.put_positions(list(range(n)))
        if mode == "direct_shared":
            from mstar.engine.resources.runner import StepRunner
            eng._runner = StepRunner({"shared": ring})
            eng._submodules["a"].resources = {"shared": ring}
            eng._submodules["b"].resources = {"shared": ring}
    assert worker._reserve_group_stream(group)
    if mode == "unsupported":
        from mstar.engine.resources.base import Resource
        from mstar.engine.resources.runner import StepRunner

        class UnsupportedOwner(Resource):
            @classmethod
            def build(cls, spec, info):
                return cls()
        owner = UnsupportedOwner()
        eng._runner = StepRunner({"unsupported": owner})
        eng._submodules["a"].resources = {"unsupported": owner}
    if mode in {"bank_small", "packing_refuse"}:
        tables.prepare_stream_window(leases["b"], 1, 1, 1)
    output = eng.exec_group(group)
    if mode in {"unsupported", "bank_small", "tail_wide", "producer_tail", "packing_refuse",
                "direct_missing", "direct_walk", "direct_alias", "direct_spans", "direct_absent",
                "direct_wide", "direct_shared"}:
        assert output is None and group.refusal is not None
        assert not group.submitted and not launches
        if packing:
            assert owner.row("x", 2).eq(-100).all(), "unsubmitted padding cannot alter its owning row"
        worker._restore_group_stream(group)
        assert not source._reservations
        assert ring.ring_room("x", 2) == (5 if mode == "direct_wide" else 6 if direct else 8)
        return
    assert output is not None, group.refusal
    if device_name == "cuda":
        backend.synchronize()
    assert launches == [1]
    if packing:
        assert owner.row("x", 2).tolist() == [[10, 11, 0], [10, 11, 0]]
    if not full:
        assert output[0].check_stop_buffers[RUN_COUNTS].tolist() == [0 if yielded else 1 if direct else 3]
    else:
        assert not output[0].check_stop_buffers and not a.step
    b_count = (0 if mode == "direct_eos" else 1) if direct else (0 if yielded else 3 if full else 1)
    assert output[1].check_stop_buffers[RUN_COUNTS].tolist() == [b_count]
    assert output[1].check_stop_buffers[STREAM_COUNTS].tolist() == [b_count]
    if b_count:
        assert output[1].row_outputs["x"][0, 0].item() == 42
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext

    completed, released = [], []
    worker._graph_runtime = SimpleNamespace(release_execution_unit_member=lambda *args: released.append(args))
    worker.scheduler = SimpleNamespace(note_ran=lambda batch: None, clear_batch_limit=lambda *a, **kw: None)
    worker.request_state = SimpleNamespace(get_partition_for_node=lambda node: node)
    worker._phase_period, worker._metrics = 0, None
    worker._span = lambda *a, **kw: nullcontext()
    worker._wait_for_step = lambda step: None  # synchronized above on the accelerator
    worker._stage_host_rows = lambda outputs, lane: None
    worker._clear_in_flight_flag = lambda batch: None
    failed_members = []
    def settle_member(step, outputs):
        if mode == "direct_error" and step.node_batch.failed_requests:
            failed_members.append(step.node_name)
            worker._drop_failed_rids(step, outputs, dict(step.node_batch.failed_requests))
        return False
    worker._settle_step_rows = settle_member
    worker._publishable_request_ids = lambda batch: batch.request_ids
    worker.device_capability = SimpleNamespace(is_accelerator=False)

    def close_member(step, outputs, **kwargs):
        completed.append((step.node_name, step.batch.repeats, dict(step.node_batch.step_context.run_counts)))
        if mode == "direct_error":
            assert not outputs.per_rid_outputs
            assert not step.batch.request_to_worker_graph
            return
        if step.node_name == "a":
            base = a.ring_bases["x"][2]
            if direct:
                assert producer_tails == [2]
                if mode in {"direct_eos", "direct_error"}:
                    ring.ring_skip("x", 2, [base])
                else:
                    source.put_positions([base])
            else:
                assert outputs.per_rid_outputs[2]["x"][0].shape[1] == 3
                source.put_positions([base, base + 1])  # ordinary routing publishes only kept items
                ring.ring_skip("x", 2, [base + 2, base + 3])
        else:
            assert source._consumed == (0 if mode == "direct_error" else 6 if full else 2) and not source._reservations
            if custom:
                assert tails == [[10, 10, 11, 11]]
                assert not outputs.run_columns
                if mode == "tail_error":
                    assert 2 in step.node_batch.failed_requests
                else:
                    # The ordinary ragged output reaches the worker unchanged;
                    # it must not be split as a synthetic width-one run column.
                    worker._withholding = False
                    worker._cut_run_items(step, outputs, SimpleNamespace(), None, {}, eng)
                    assert outputs.per_rid_outputs[2]["x"][0].tolist() == [10, 11]
            else:
                if mode == "direct_error":
                    assert 2 in step.node_batch.failed_requests
                assert outputs.per_rid_outputs[2]["x"][0].flatten().tolist() == ([42, 50, 58] if full else [42])
    worker._postprocess_batch = close_member
    pending = PendingBatch(group.members[0][0], a, "a", "a", "prefill", None, group=group)
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker._finish_step(pending, output[0], gpu_executor=executor)
    if direct:
        assert completed == ([('b', 1, {2: 1})] if mode == "direct_zero" else
                             [('a', 1, {2: 1})] if mode == "direct_eos" else
                             [('a', 1, {2: 1}), ('b', 1, {2: 1})])
        assert producer_tails == ([] if mode == "direct_zero" else [2])
        assert failed_members == (["a", "b"] if mode == "direct_error" else [])
        assert not b.stream_windows and not source._reservations
        assert launches == [1]
        return
    assert completed == ([] if yielded else [("b", 3, {2: 3})] if full else
                         [("a", 3, {2: 3}), ("b", 1, {2: 1})])
    assert len(released) == (2 if yielded else 1 if full else 0)
    assert ring.ring_room("x", 2) == (6 if full else 8) and not b.stream_windows
    assert not source.producer_done, "a stopped producer Loop is not conductor EOF"
    if yielded:
        assert not tails
    if custom:
        assert batch_tails == ([] if yielded or mode == "tail_error" else [(2,)])
