import pytest
import torch

from mstar.engine.accelerator_graph_backend import AcceleratorGraphBackend
from mstar.engine.accelerator_graph_runner import (
    capture_into_graph,
    capture_with_static_outputs,
)


def test_backend_rejects_non_accelerator_device():
    with pytest.raises(ValueError, match="unsupported"):
        AcceleratorGraphBackend(torch.device("cpu"))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
def test_xpu_graph_backend_replays_with_updated_static_input():
    backend = AcceleratorGraphBackend(torch.device("xpu:0"))
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
    backend = AcceleratorGraphBackend(graph_device)
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
    backend = AcceleratorGraphBackend(graph_device)
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


def test_worker_output_copy_on_accelerator(graph_device):
    from collections import defaultdict

    from mstar.worker.worker import Worker

    worker = object.__new__(Worker)
    worker.device = graph_device
    worker._d2h_stream = None
    worker._pinned_d2h_buffers = defaultdict(list)
    value = torch.arange(16, device=graph_device)
    event = getattr(torch, graph_device.type).Event()
    event.record()
    copied = worker._prematerialize_for_check_stop({7: {"token": [value]}}, event)
    assert copied[7]["token"][0].device.type == "cpu"
    torch.testing.assert_close(copied[7]["token"][0], torch.arange(16))
