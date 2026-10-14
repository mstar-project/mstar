"""NVTX is opt-in and only supported for CUDA deployments."""

import pytest
import torch

from mstar.utils.profiler import nvtx_enabled


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("backend", ["cuda", "xpu", "cpu", None])
def test_nvtx_follows_the_selected_accelerator(monkeypatch, requested, backend):
    accelerator = torch.device(backend) if backend else None
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator",
        lambda **kwargs: accelerator,
    )
    assert nvtx_enabled(requested) == (requested and backend == "cuda")


@pytest.mark.parametrize("backend", ["cuda", "xpu", "cpu"])
def test_explicit_worker_device_controls_nvtx(monkeypatch, backend):
    # A CUDA accelerator elsewhere must not enable NVTX for an XPU/CPU worker.
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator",
        lambda **kwargs: torch.device("cuda"),
    )
    assert nvtx_enabled(True, torch.device(backend)) == (backend == "cuda")
    assert not nvtx_enabled(False, torch.device(backend))


@pytest.mark.parametrize("backend", ["cuda", "xpu", "cpu"])
def test_client_nvtx_is_only_enabled_on_cuda(monkeypatch, backend):
    from mstar.client.client import MStarClient

    monkeypatch.setattr(
        torch.accelerator, "current_accelerator",
        lambda **kwargs: torch.device(backend),
    )
    client = MStarClient(enable_nvtx=True)
    assert (client._nvtx is not None) == (backend == "cuda")
