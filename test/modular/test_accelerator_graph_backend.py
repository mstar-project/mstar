import pytest
import torch

from mstar.engine.accelerator_graph_backend import AcceleratorGraphBackend
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BasicBatchedAcceleratorGraphConfig,
)
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.cuda_graph_config import (
    BasicBatchedCudaGraphConfig,
    CudaGraphConfig,
)
from mstar.engine.cuda_graph_runner import CudaGraphRunner


def test_cuda_compatibility_aliases_point_to_accelerator_api():
    assert CudaGraphConfig is AcceleratorGraphConfig
    assert BasicBatchedCudaGraphConfig is BasicBatchedAcceleratorGraphConfig
    assert CudaGraphRunner is AcceleratorGraphRunner


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
