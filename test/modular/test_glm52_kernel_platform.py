"""The GLM-5.2 kernels' GPU checks go through ``current_platform``: an AMD GPU reports a
CUDA-style capability (an MI300X is 9.4), which a raw capability read takes for Hopper."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")


def test_an_amd_gpu_gets_neither_pdl_nor_the_sm90_gemm(monkeypatch):
    from mstar.utils import dense_fp8, mla_prep

    rocm = SimpleNamespace(cuda_capability=lambda device=None: None,
                           has_cuda_capability=lambda major, minor=0, device=None: False)
    monkeypatch.setattr(dense_fp8, "current_platform", rocm)
    monkeypatch.setattr(mla_prep, "current_platform", rocm)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    device = torch.device("cuda", 0)
    monkeypatch.setattr(mla_prep, "_PDL", {})

    assert not mla_prep._pdl(device)
    assert not dense_fp8._probe_fi(device)
