import pytest
import torch

from mstar.engine.accelerator_graph_backend import (
    AcceleratorGraphBackend,
    CUDAGraphBackend,
    XPUGraphBackend,
    get_accelerator_graph_backend,
    supports_accelerator_graphs,
)
from mstar.engine.accelerator_graph_runner import (
    capture_into_graph,
    capture_with_static_outputs,
)


def test_backend_rejects_non_accelerator_device():
    with pytest.raises(ValueError, match="unsupported"):
        get_accelerator_graph_backend(torch.device("cpu"))
    assert not supports_accelerator_graphs(torch.device("cpu"))


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
    # CPU-only PyTorch wheels may omit the XPU graph class entirely.
    monkeypatch.setattr(runtime, graph_type, lambda: graph, raising=False)
    monkeypatch.setattr(runtime, "is_available", lambda: False)

    def initialize(*args, **kwargs):
        pytest.fail("selecting a graph backend must not initialize hardware")

    monkeypatch.setattr(runtime, "_lazy_init", initialize)
    device = torch.device(device_type, 0)
    backend = get_accelerator_graph_backend(device)
    assert isinstance(backend, backend_type)
    assert backend.device == device
    assert supports_accelerator_graphs(device)
    assert not backend.is_available()
    assert backend.create_graph() is graph


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
    assert backend.current_stream() == stream

    graph, output = capture_into_graph(lambda: value * 2, pool, graph_device, None)
    value.fill_(3)
    graph.replay()
    backend.synchronize()
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
    assert all(buffer.is_pinned() for buffer in stager._bufs)
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
