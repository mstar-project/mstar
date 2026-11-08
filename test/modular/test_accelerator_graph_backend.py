import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.accelerator_graph_backend import (
    AcceleratorGraphBackend,
    CUDAGraphBackend,
    GraphConditional,
    XPUGraphBackend,
    get_accelerator_graph_backend,
)
from mstar.engine.accelerator_graph_runner import (
    capture_into_graph,
    capture_with_static_outputs,
)
from mstar.utils.device import DeviceCapability


def test_backend_rejects_non_accelerator_device():
    with pytest.raises(ValueError, match="unsupported"):
        get_accelerator_graph_backend(torch.device("cpu"))
    assert DeviceCapability.of("cpu").graph_backend is None


def test_backend_interface_requires_an_implementation():
    with pytest.raises(TypeError, match="abstract"):
        AcceleratorGraphBackend(torch.device("xpu"))


@pytest.mark.parametrize(
    ("device_type", "backend_type", "graph_type"),
    [
        ("cuda", CUDAGraphBackend, "CUDAGraph"),
        ("xpu", XPUGraphBackend, "XPUGraph"),
    ],
)
def test_backend_dispatches_without_initializing_hardware(
    monkeypatch, device_type, backend_type, graph_type,
):
    runtime = getattr(torch, device_type)
    graph = object()
    monkeypatch.setattr(runtime, graph_type, lambda keep_graph=False: graph, raising=False)
    monkeypatch.setattr(runtime, "is_available", lambda: False)

    def initialize(*args, **kwargs):
        pytest.fail("selecting a graph backend must not initialize hardware")

    monkeypatch.setattr(runtime, "_lazy_init", initialize)
    device = torch.device(device_type, 0)
    backend = get_accelerator_graph_backend(device)
    assert isinstance(backend, backend_type)
    assert backend.device == device
    assert DeviceCapability.of(device).graph_backend is not None
    assert not backend.is_available()
    assert backend.create_graph() is graph


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_runner_retains_failed_graphs_only_for_xpu(monkeypatch, device_type):
    device = torch.device(device_type, 0)
    backend = get_accelerator_graph_backend(device)
    monkeypatch.setattr(backend._runtime, "graph_pool_handle", object, raising=False)
    pool = backend.graph_pool_handle()
    graphs = []

    class Graph:
        pass

    def create_graph(keep_graph=False):
        graph = Graph()
        graphs.append(weakref.ref(graph))
        return graph

    stream = object()
    monkeypatch.setattr(backend, "create_graph", create_graph)
    monkeypatch.setattr(backend, "current_stream", lambda: stream)
    monkeypatch.setattr(backend, "capture", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(backend, "synchronize", lambda: None)
    recovery = Mock()
    monkeypatch.setattr(backend, "recover_failed_capture", recovery)
    monkeypatch.setattr(
        DeviceCapability, "of", lambda device: SimpleNamespace(graph_backend=backend),
    )

    def fails():
        raise RuntimeError("capture interrupted")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="capture interrupted"):
            capture_into_graph(fails, pool, device, None)
        gc.collect()
        assert (graphs[-1]() is not None) == (device_type == "xpu")
    assert graphs[0]() is None
    assert recovery.call_count == 2
    recovery.assert_called_with(pool, stream)

    graph, output = capture_into_graph(lambda: 7, pool, device, None)
    gc.collect()
    assert graphs[-2]() is None
    assert graphs[-1]() is graph
    assert output == 7


@pytest.mark.parametrize("backend_type", [CUDAGraphBackend, XPUGraphBackend])
def test_concrete_backend_rejects_wrong_device(backend_type):
    with pytest.raises(ValueError, match="requires"):
        backend_type(torch.device("cpu"))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
def test_xpu_graph_backend_replays_with_updated_static_input():
    backend = get_accelerator_graph_backend(torch.device("xpu:0"))
    backend.set_device()
    static_input = torch.ones(16, device=backend.device)
    static_output = torch.empty_like(static_input)
    stream = backend.new_stream()
    stream.wait_stream(backend.current_stream())

    graph = backend.create_graph()
    with backend.capture(
        graph,
        pool=backend.graph_pool_handle(),
        stream=stream,
    ):
        static_output.copy_(static_input * 3)
    stream.synchronize()

    static_input.fill_(4)
    graph.replay()
    backend.synchronize()
    torch.testing.assert_close(
        static_output.cpu(),
        torch.full((16,), 12.0),
    )


@pytest.fixture(params=["cuda", "xpu"])
def graph_device(request):
    runtime = getattr(torch, request.param)
    if not runtime.is_available():
        pytest.skip(f"{request.param} is unavailable")
    device = torch.device(request.param, 0)
    runtime.set_device(device)
    return device


def test_runner_capture_recovers_after_failed_capture(graph_device):
    backend = get_accelerator_graph_backend(graph_device)
    value = torch.ones(8, device=graph_device)
    pool = backend.graph_pool_handle()
    stream = backend.current_stream()

    def fails():
        value * 2
        raise RuntimeError("capture interrupted")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="capture interrupted"):
            capture_into_graph(fails, pool, graph_device, None)
        if graph_device.type == "xpu":
            assert pool.failed_graph is not None
        else:
            assert not hasattr(pool, "failed_graph")
    assert backend.current_stream() == stream

    graph, output = capture_into_graph(lambda: value * 2, pool, graph_device, None)
    value.fill_(3)
    graph.replay()
    backend.synchronize()
    if graph_device.type == "xpu":
        assert pool.failed_graph is None, "successful capture must release the failed graph"
    torch.testing.assert_close(output.cpu(), torch.full((8,), 6.0))


def test_sampler_buffers_stage_on_accelerator(graph_device):
    from mstar.engine.resources.sampler.utils import SamplerBuffers, SamplingConfig

    buffers = SamplerBuffers.allocate(2, graph_device, cg_slots=2)
    config = SamplingConfig(temperature=0.5)
    config.set_seed(777)
    buffers.register_request(7, config)
    buffers.gather_static([7], 1, 1)
    buffers.gather_dynamic([7], 1, 1)
    getattr(torch, graph_device.type).synchronize()

    assert buffers._slot_idx_cpu.is_pinned()
    assert buffers.seed.slot_view(1, 1).item() == 777
    assert buffers.temperature.slot_view(1, 1).item() == 0.5


def test_shared_pool_region_outputs_survive_other_graph_replays(graph_device):
    backend = get_accelerator_graph_backend(graph_device)
    value = torch.ones(8, device=graph_device)
    pool = backend.graph_pool_handle()
    graphs, outputs = [], []
    for scale in (2, 3):
        def run(scale=scale):
            return {"x": value * scale}

        graph, output = capture_with_static_outputs(run, run(), pool, graph_device, None)
        graphs.append(graph)
        outputs.append(output)

    value.fill_(5)
    graphs[1].replay()
    graphs[0].replay()
    backend.synchronize()
    torch.testing.assert_close(outputs[0]["x"].cpu(), torch.full((8,), 10.0))
    torch.testing.assert_close(outputs[1]["x"].cpu(), torch.full((8,), 15.0))


@pytest.mark.parametrize("batched", [False, True])
def test_worker_output_copy_on_accelerator(graph_device, batched):
    from collections import defaultdict

    from mstar.model.submodule_base import BatchedModelOutput
    from mstar.worker.worker import Worker

    worker = object.__new__(Worker)
    worker.device = graph_device
    worker._d2h_stream = None
    worker._pinned_d2h_buffers = defaultdict(list)
    value = torch.arange(16, device=graph_device)
    outputs = BatchedModelOutput(per_rid_outputs={7: {"token": [value]}})
    if batched:
        outputs.check_stop_buffers = {"token": value}
        outputs.row_request_ids = (7, 3)
    event = getattr(torch, graph_device.type).Event()
    event.record()
    copied, host_rows = worker._prematerialize_for_check_stop(
        outputs, event, request_ids=[3],
    )
    assert copied[7]["token"][0].device.type == "cpu"
    if batched:
        assert list(copied) == [7, 3]
        assert host_rows.request_ids == (7, 3)
        assert host_rows.buffers["token"].device.type == "cpu"
        torch.testing.assert_close(copied[7]["token"][0], torch.tensor([0]))
        torch.testing.assert_close(copied[3]["token"][0], torch.tensor([1]))
    else:
        assert host_rows is None
        torch.testing.assert_close(copied[7]["token"][0], torch.arange(16))


def test_pinned_staging_keeps_copies_intact_across_reuse_and_growth(graph_device):
    from mstar.utils.h2d import H2DMirror, PinnedStager

    stager = PinnedStager(torch.float32, numel=2, depth=2)
    backend = get_accelerator_graph_backend(graph_device)
    side = backend.new_stream()
    side.wait_stream(backend.current_stream())
    outputs = []
    with backend.stream_context(side):
        for i in range(10):
            size = 2 if i < 3 else 64
            dst = torch.empty(size, device=graph_device)
            stager.copy_(dst, [float(i), -float(i)], pad_value=0)
            outputs.append(dst)
    side.synchronize()
    assert stager._arena.is_pinned()
    for i, dst in enumerate(outputs):
        expected = torch.zeros(dst.numel())
        expected[:2] = torch.tensor([float(i), -float(i)])
        torch.testing.assert_close(dst.cpu(), expected)

    mirror = H2DMirror()
    stager.copy_(outputs[-1], [3., 4.], pad_value=0, mirror=mirror)
    stager.copy_(outputs[-1], [3., 4.], pad_value=0, mirror=mirror)
    outputs[-1].fill_(9)
    mirror.invalidate()
    stager.copy_(outputs[-1], [3., 4.], pad_value=0, mirror=mirror)
    backend.synchronize()
    torch.testing.assert_close(outputs[-1][:2].cpu(), torch.tensor([3., 4.]))
    assert outputs[-1][2:].count_nonzero().item() == 0


def _pinnable(monkeypatch, pin: bool):
    from mstar.utils import h2d

    capability = SimpleNamespace(pin_memory=pin)
    monkeypatch.setattr(h2d.DeviceCapability, "of", staticmethod(lambda device: capability))
    return h2d.to_device_nonblocking


@pytest.mark.parametrize(("device", "pin"), [("xpu:0", True), ("cuda:0", False)])
def test_to_device_nonblocking_pins_where_the_device_capability_says(monkeypatch, device, pin):
    to_device_nonblocking = _pinnable(monkeypatch, pin)
    pinned = Mock(spec=torch.Tensor)
    empty = Mock(return_value=pinned)
    monkeypatch.setattr(torch, "empty", empty)
    host = Mock(spec=torch.Tensor)
    host.device = torch.device("cpu")
    host.is_pinned.return_value = False
    host.is_contiguous.return_value = True
    host.numel.return_value = 0
    to_device_nonblocking(host, device)
    if pin:
        assert empty.call_args.kwargs["pin_memory"] is True
        pinned.to.assert_called_once_with(torch.device(device), non_blocking=True)
    else:
        empty.assert_not_called()
        host.to.assert_called_once_with(torch.device(device))


def test_to_device_nonblocking_builds_a_sequence_in_one_pinned_allocation(monkeypatch):
    to_device_nonblocking = _pinnable(monkeypatch, True)
    host = Mock(spec=torch.Tensor)
    build = Mock(return_value=host)
    monkeypatch.setattr(torch, "tensor", build)
    out = to_device_nonblocking([True, False], "xpu:0", dtype=torch.bool)
    build.assert_called_once_with([True, False], dtype=torch.bool, pin_memory=True)
    host.pin_memory.assert_not_called()
    host.to.assert_called_once_with(torch.device("xpu:0"), non_blocking=True)
    assert out is host.to.return_value


def test_to_device_nonblocking_takes_a_sequence_on_cpu():
    from mstar.utils.h2d import to_device_nonblocking

    torch.testing.assert_close(to_device_nonblocking([3, 1], "cpu"), torch.tensor([3, 1]))
    keep = to_device_nonblocking([True, False], "cpu", dtype=torch.bool)
    assert keep.dtype == torch.bool and keep.tolist() == [True, False]


def test_chatterbox_shape_graphs_keep_results_across_replays(graph_device):
    from mstar.model.chatterbox.components.s3gen_graphs import ShapeGraphs

    graphs = ShapeGraphs(lambda tensors, extra: tensors["x"] * extra[0] + 1)
    source = torch.arange(8, device=graph_device, dtype=torch.float32)
    first = graphs.run({"x": source}, (2,))
    second = graphs.run({"x": source + 10}, (3,))
    third = graphs.run({"x": source + 20}, (2,))
    backend = get_accelerator_graph_backend(graph_device)
    backend.synchronize()

    assert not graphs.disabled
    assert graphs.captures == 2 and graphs.replays == 3
    host_source = torch.arange(8, dtype=torch.float32)
    torch.testing.assert_close(first.cpu(), host_source * 2 + 1)
    torch.testing.assert_close(second.cpu(), (host_source + 10) * 3 + 1)
    torch.testing.assert_close(third.cpu(), (host_source + 20) * 2 + 1)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_composed_cuda_captures_keep_each_round_and_reuse_both_slots(monkeypatch, device_name):
    """Per-member outputs, parallel carries and varying tails use bounded slot buffers."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner, AcceleratorGraphSlot, RunPart
    from mstar.model.submodule_base import BatchedModelOutput

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("real CUDA graph composition")
    device = torch.device(device_name)
    backend = None if device_name == "cpu" else get_accelerator_graph_backend(device)
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._device, runner._graph_backend = device, backend

    def capture(fn):
        if backend is not None:
            return capture_into_graph(fn, backend.graph_pool_handle(), device, None, keep_graph=True)
        value = fn()
        return SimpleNamespace(replay=lambda: value.copy_(fn())), value

    slots = []
    for _ in range(2):
        x = torch.zeros(3, device=device)
        z, delta, b_input = (torch.zeros_like(x) for _ in range(3))
        graph_a, a = capture(lambda x=x, z=z, delta=delta: x + z + delta)
        graph_b, b = capture(lambda b_input=b_input: b_input * 2)
        def slot(graph, output):
            return AcceleratorGraphSlot(graph, {}, (), BatchedModelOutput(row_outputs={"value": output}), [], {}, 0)
        slots.append((x, z, delta, b_input, slot(graph_a, a), slot(graph_b, b), a, b))
    launches = []
    if backend is not None:
        runtime = backend.while_runtime()
        launch = runtime.launch
        def counted_launch(*args):
            launches.append(args[0])
            return launch(*args)
        monkeypatch.setattr(runtime, "launch", counted_launch)
    storage_addresses, retained, expected = {}, [], []
    for wave, width in enumerate((3, 3, 1, 3)):
        for index, (x, z, delta, b_input, a_slot, b_slot, a, b) in enumerate(slots):
            initial = torch.arange(3, device=device, dtype=torch.float32) + 10 * wave + index
            x.copy_(initial)
            z.fill_(2)
            values = torch.arange(3, device=device, dtype=torch.float32).expand(3, 3).clone() + wave
            outputs = runner.launch_parts(
                [RunPart(a_slot, [], [(values, delta)]), RunPart(b_slot, [(a, b_input)], [])],
                width, [("value",), ("value",)], capacity=3, boundary=[(b, x), (x, z)],
            )
            addresses = tuple(t.data_ptr() for t in a_slot.composed_buffers)
            assert storage_addresses.setdefault(index, addresses) == addresses
            retained.append([out.row_outputs["value"].clone() for out in outputs])
            want_a, want_b = [], []
            previous, other = initial, torch.full_like(initial, 2)
            for iteration in range(width):
                value_a = previous + other + values[:, iteration]
                value_b = value_a * 2
                want_a.append(value_a)
                want_b.append(value_b)
                previous, other = value_b, previous
            expected.append([torch.stack(want_a, dim=1), torch.stack(want_b, dim=1)])
    if backend is not None:
        backend.synchronize()
        assert len(launches) == 8 and launches[:2] == launches[2:4]
    for got, want in zip(retained, expected, strict=True):
        for member_got, member_want in zip(got, want, strict=True):
            torch.testing.assert_close(member_got, member_want)


def test_composed_parent_owns_operations_after_intermediate_reset(monkeypatch):
    from test_device_run import _Loop

    from mstar.engine.accelerator_graph_backend import _CUDAComposedGraph

    # Exercise production operation ownership without claiming native lowering.
    # The real CUDA body-reuse case checks the dependency chain and IF-in-WHILE.
    def call(name, *args):
        if name in ("cudaGraphCreate", "cudaGraphInstantiateWithFlags"):
            args[0]._obj.value = 1
    backend = SimpleNamespace(device=torch.device("cpu"), while_runtime=lambda: SimpleNamespace(_call=call))
    monkeypatch.setattr(_CUDAComposedGraph, "_append", lambda *args: None)
    value = torch.zeros(1)
    predicate = torch.ones(1, dtype=torch.long)
    class Leaf:
        def replay(self):
            value.mul_(2)
    leaf = Leaf()
    held = weakref.ref(leaf)
    inner = _CUDAComposedGraph(backend, (leaf,))
    selected = _CUDAComposedGraph(backend, (GraphConditional("if", predicate, (inner,)),))
    parent = _CUDAComposedGraph(backend, (
        SimpleNamespace(replay=lambda: value.add_(3)), selected,
        SimpleNamespace(replay=lambda: value.add_(1)),
    ))
    inner.reset()
    selected.reset()
    del inner, selected, leaf
    gc.collect()
    assert held() is not None
    for enabled, expected in ((True, 7), (False, 4), (True, 7)):
        predicate.fill_(enabled)
        value.zero_()
        _Loop().compose(parent._ops).replay()
        assert value.item() == expected
    # Invalid nested members fail before changing the live source composition.
    other = SimpleNamespace(device=torch.device("cuda:1"), while_runtime=backend.while_runtime)
    with pytest.raises(ValueError, match="same device"):
        _CUDAComposedGraph(other, (parent,))
    assert parent._graph.value
    parent.reset()
    with pytest.raises(ValueError, match="live"):
        _CUDAComposedGraph(backend, (GraphConditional("if", predicate, (parent,)),))
    gc.collect()
    assert held() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA conditional composition")
def test_one_cuda_launch_gates_members_independently_and_drains_without_producing(monkeypatch):
    """Nested IFs are constructed in the WHILE body, never launched as child runs."""
    backend = get_accelerator_graph_backend(torch.device("cuda:0"))
    device = backend.device
    produced, consumed, remaining = (torch.zeros(1, device=device, dtype=torch.int64) for _ in range(3))
    producer_cap, consumer_cap = (torch.ones_like(produced) for _ in range(2))
    active, producer_ready, consumer_ready = (torch.zeros_like(produced) for _ in range(3))
    ring, output, order = (torch.zeros(16, device=device, dtype=torch.int64) for _ in range(3))
    order_count = torch.zeros_like(produced)

    def ready_a():
        producer_ready.copy_(produced < producer_cap)

    def ready_b():
        consumer_ready.copy_((consumed < produced) & (consumed < consumer_cap))

    def more():
        active.copy_(((produced < producer_cap) | ((consumed < produced) & (consumed < consumer_cap)))
                     & (remaining > 0))

    def producer():
        ring.scatter_(0, produced.clamp(0, 15), produced + 10)
        order.scatter_(0, order_count.clamp(0, 15), produced + 100)
        order_count.add_(1)
        produced.add_(1)

    def consumer():
        output.scatter_(0, consumed.clamp(0, 15), ring.gather(0, consumed.clamp(0, 15)) * 2)
        order.scatter_(0, order_count.clamp(0, 15), consumed + 200)
        order_count.add_(1)
        consumed.add_(1)

    def boundary():
        remaining.sub_(1)
        more()

    graphs = [capture_into_graph(fn, backend.graph_pool_handle(), device, None, keep_graph=True)[0]
              for fn in (more, ready_a, producer, ready_b, consumer, boundary)]
    graph = backend.compose((graphs[0], GraphConditional("while", active, (
        graphs[1], GraphConditional("if", producer_ready, (graphs[2],)),
        graphs[3], GraphConditional("if", consumer_ready, (graphs[4],)), graphs[5],
    ))))
    launches = []
    cuda = backend.while_runtime()
    launch = cuda.launch
    monkeypatch.setattr(cuda, "launch", lambda *args: (launches.append(args[0]), launch(*args))[-1])
    # Normal interleaving, independent consumer stop, preexisting consumer-only
    # drain, empty entry, then a short reservation followed by ordinary reuse.
    cases = [(0, 3, 3, 8), (0, 4, 2, 8), (2, 2, 2, 8), (0, 0, 0, 8), (0, 4, 4, 2), (0, 1, 1, 8)]
    for initial, a_limit, b_limit, budget in cases:
        produced.fill_(initial)
        consumed.zero_()
        remaining.fill_(budget)
        producer_cap.fill_(a_limit)
        consumer_cap.fill_(b_limit)
        ring.copy_(torch.arange(16, device=device) + 10)
        output.zero_()
        order.zero_()
        order_count.zero_()
        graph.replay()
        backend.synchronize()
        a, b, turns, expected = initial, 0, 0, []
        while turns < budget and (a < a_limit or (b < a and b < b_limit)):
            if a < a_limit:
                expected.append(100 + a)
                a += 1
            if b < a and b < b_limit:
                expected.append(200 + b)
                b += 1
            turns += 1
        assert (produced.item(), consumed.item(), remaining.item()) == (a, b, budget - turns)
        assert order[:len(expected)].tolist() == expected
        assert order_count.item() == len(expected)
        assert output[:b].tolist() == [2 * (10 + i) for i in range(b)]
        assert not output[b:].any()
    assert len(launches) == len(cases) and len(set(launches)) == 1
    graph.reset()


def test_conditional_warmup_checks_branches_before_priming(monkeypatch):
    from mstar.engine.conditional_capture import ConditionalWarmup

    backend = CUDAGraphBackend(torch.device("cuda", 0))
    primed = []
    monkeypatch.setattr(ConditionalWarmup, "_prime", lambda self, *args: primed.append(args))
    x = torch.arange(4.)
    index = torch.tensor([99])
    def choose(x, index, pred):
        return torch.cond(pred, lambda x, i: x[i], lambda x, i: x[:1] + 2, (x, index))
    mode = ConditionalWarmup(backend)
    with mode:
        result = choose(x, index, torch.tensor(False))
    assert mode.found and result.tolist() == [2] and len(primed) == 1
    with mode, torch.compiler.set_stance("fail_on_recompile"):
        result = choose(x, torch.tensor([2]), torch.tensor(True))
    assert result.tolist() == [2] and len(primed) == 2
    before = len(primed)
    with ConditionalWarmup(backend, world_size=2), pytest.raises(ValueError, match="one joint rank"):
        choose(x, index, torch.tensor(False))
    assert len(primed) == before
    with ConditionalWarmup(backend), pytest.raises(ValueError, match="random"):
        torch.cond(torch.tensor(False), torch.rand_like, lambda x: x + 1, (x,))
    assert len(primed) == before and x.tolist() == [0, 1, 2, 3]
    def local(value):
        result = value.sin()
        result.add_(1)
        return result
    with ConditionalWarmup(backend):
        actual = torch.cond(torch.tensor(True), local, lambda value: value + 2, (x,))
    torch.testing.assert_close(actual, x.sin() + 1)
    assert x.tolist() == [0, 1, 2, 3]
    def nested(value, outer, inner):
        return torch.cond(outer, lambda value, inner: torch.cond(
            inner, lambda value: value * 2, lambda value: value + 3, (value,),
        ), lambda value, inner: value + 4, (value, inner))
    with ConditionalWarmup(backend):
        actual = nested(x, torch.tensor(True), torch.tensor(False))
    torch.testing.assert_close(actual, x + 3)
    embedding = torch.nn.Embedding(4, 3).eval().requires_grad_(False)
    ids = torch.tensor([1, 2])
    with ConditionalWarmup(backend):
        actual = torch.cond(torch.tensor(True), lambda ids: embedding(ids).sin(),
                            lambda ids: embedding(ids).cos(), (ids,))
    torch.testing.assert_close(actual, embedding(ids).sin())
    from mstar.engine.conditional_capture import _check_branches

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    graph.call_function(torch.ops.aten._print.default, ("must not execute",))
    result = graph.call_function(torch.ops.aten.sin.default, (value,))
    graph.output(result)
    with pytest.raises(ValueError, match="unsupported"):
        _check_branches((torch.fx.GraphModule({}, graph),), (x,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tensor conditional capture")
def test_tensor_conditionals_capture_safe_unselected_branches_and_shared_slots():
    from mstar.engine.conditional_capture import ConditionalWarmup

    device = torch.device("cuda", torch.cuda.current_device())
    backend = CUDAGraphBackend(device)
    if reason := backend.while_refusal():
        pytest.skip(reason)
    x = torch.arange(4., device=device)
    index = torch.tensor([99], device=device)
    p = torch.tensor(False, device=device)
    q = torch.tensor(True, device=device)
    def run():
        def selected(x, index, q):
            value = x[index]
            return torch.cond(q, lambda a: a.sin() + 1, lambda a: a.cos() + 2, (value,))
        return torch.cond(p, selected, lambda x, index, q: x[:1] + 3, (x, index, q))
    with ConditionalWarmup(backend) as warm:
        expected = run()
    backend.synchronize()  # the invalid unselected gather must not have launched
    assert warm.found and expected.item() == 3
    pool = backend.graph_pool_handle()
    graphs = [capture_into_graph(run, pool, device, None, conditionals=True, keep_graph=True) for _ in range(2)]
    from mstar.engine.accelerator_graph_runner import ExecutableMemory, _capture_thrown_away

    executable = ExecutableMemory()
    try:
        measured = _capture_thrown_away(
            lambda pool, keep_graph: capture_into_graph(
                run, pool, device, None, conditionals=True, keep_graph=keep_graph,
            ), "nested tensor conditionals", executable,
        )
        assert measured is not None and 0 < measured.kept <= measured.pool and measured.outside > 0
    finally:
        executable.release()
    copies = [torch.empty_like(output) for _, output in graphs]
    parents = [backend.compose((graph, (copy, output))) for (graph, output), copy in zip(graphs, copies, strict=True)]
    turns = torch.ones(1, dtype=torch.int64, device=device)
    turns.sub_(1)
    countdown, _ = capture_into_graph(lambda: turns.sub_(1), pool, device, None, keep_graph=True)
    repeated = [backend.compose((GraphConditional("while", turns, (parent, countdown)),)) for parent in parents]
    for graph, _ in graphs:
        graph.reset()  # parents must retain every internal predicate/merge/operand
    for predicate, nested, position in [(True, False, 2), (False, True, 99), (True, True, 1)]:
        p.fill_(predicate)
        q.fill_(nested)
        index.fill_(position)
        expected = run()
        for parent, copy in zip(reversed(parents), reversed(copies), strict=True):
            parent.replay()
            backend.synchronize()
            torch.testing.assert_close(copy, expected, rtol=0, atol=0)
        for parent, copy in zip(repeated, copies, strict=True):
            turns.fill_(2)
            parent.replay()
            backend.synchronize()
            assert turns.item() == 0
            torch.testing.assert_close(copy, expected, rtol=0, atol=0)

    def interrupted():
        run()
        raise RuntimeError("interrupted after nested branch")
    with pytest.raises(RuntimeError, match="interrupted after nested branch"):
        capture_into_graph(interrupted, pool, device, None, conditionals=True)
    healthy, output = capture_into_graph(run, pool, device, None, conditionals=True)
    healthy.replay()
    backend.synchronize()
    torch.testing.assert_close(output, run(), rtol=0, atol=0)
