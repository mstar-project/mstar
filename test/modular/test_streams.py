"""``mstar.utils.streams.reset_device_scheduling``."""
import ctypes

import pytest
import torch

from mstar import platforms
from mstar.platforms import RocmPlatform
from mstar.utils.streams import reset_device_scheduling

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_reset_device_scheduling_keeps_cuda_state_usable():
    dev = torch.device("cuda", torch.cuda.current_device())
    x = torch.arange(8, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.mul(x, 2, out=y)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.mul(x, 2, out=y)

    reset_device_scheduling(dev)

    assert torch.cuda.current_device() == dev.index
    x.add_(1)  # tensors allocated before the reset are still valid
    g.replay()  # and so is a graph captured before it
    torch.cuda.synchronize()
    assert y.tolist() == [2.0 * (i + 1) for i in range(8)]


def test_reset_device_scheduling_is_a_noop_off_cuda():
    reset_device_scheduling("cpu")


def test_reset_device_scheduling_is_a_noop_on_rocm(monkeypatch):
    # ROCm: torch.cuda reports a device but there is no libcuda
    def no_libcuda(name, *args, **kwargs):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(platforms, "resolve_platform", RocmPlatform)
    monkeypatch.setattr(ctypes, "CDLL", no_libcuda)
    reset_device_scheduling("cuda:0")
