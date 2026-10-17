"""``mstar.platforms``: which hardware the process runs on."""

from types import SimpleNamespace

import pytest
import torch

from mstar import platforms
from mstar.platforms import CudaPlatform, RocmPlatform, current_platform


@pytest.fixture
def resolve(monkeypatch):
    """Resolve afresh for a given torch build and visible devices."""

    def run(*, cuda: bool, hip: str | None = None, xpu: bool = False):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: xpu))
        platforms.resolve_platform.cache_clear()
        return platforms.resolve_platform()

    yield run
    platforms.resolve_platform.cache_clear()


def test_a_rocm_build_is_rocm_though_torch_calls_it_cuda(resolve):
    platform = resolve(cuda=True, hip="7.0")
    assert platform.is_rocm() and not platform.is_cuda()


def test_a_cuda_build_is_cuda(resolve):
    platform = resolve(cuda=True)
    assert platform.is_cuda() and not platform.is_rocm()


def test_xpu_and_cpu(resolve):
    assert resolve(cuda=False, xpu=True).is_xpu()
    assert resolve(cuda=False).is_cpu()


def test_current_platform_follows_resolution(resolve):
    resolve(cuda=False)
    assert current_platform.is_cpu()
    assert repr(current_platform) == "CpuPlatform()"


def test_an_mi300x_is_not_a_hopper(monkeypatch):
    """ROCm reports the GFX version where CUDA reports compute capability."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 4))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-"),
    )
    platform = RocmPlatform()
    assert platform.arch() == "gfx942"
    assert platform.cuda_capability() is None
    assert not platform.has_cuda_capability(9)


@pytest.mark.parametrize(
    ("capability", "arch", "hopper"),
    [((8, 0), "sm_80", False), ((9, 0), "sm_90", True), ((12, 0), "sm_120", True)],
)
def test_cuda_arch_and_capability(monkeypatch, capability, arch, hopper):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    platform = CudaPlatform()
    assert platform.arch() == arch
    assert platform.has_cuda_capability(9) is hopper
    assert platform.has_cuda_capability(*capability)
