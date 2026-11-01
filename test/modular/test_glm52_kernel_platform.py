"""Which GPUs the GLM-5.2 kernels' paths apply to. The GPU checks go through
``current_platform``: an AMD GPU reports a CUDA-style capability (an MI300X is 9.4), which a raw
capability read takes for Hopper."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

# GLM-5.2's o_proj at one TP8 rank, (N, K)
O_PROJ = (6144, 2048)


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


def _paths(monkeypatch, gpu, fi_ok, calls):
    """The kernels ``dense_fp8._linear`` picks for ``calls`` [(N, K, tokens)] on a device whose
    limits are resolved as ``reserve`` resolves them for ``gpu``."""
    from mstar.utils import dense_fp8

    cpu = torch.device("cpu")
    seen = []
    for fn in ("_w8a16", "_fi", "_w8a8"):
        monkeypatch.setattr(dense_fp8, fn, lambda *args, fn=fn: seen.append(fn))
    monkeypatch.setitem(dense_fp8._FI_OK, cpu, fi_ok)
    monkeypatch.setitem(dense_fp8._W8A8_OK, cpu, True)
    monkeypatch.setitem(dense_fp8._W8A16_MAX, cpu, dense_fp8._w8a16_max_tokens(gpu, fi_ok))
    for n, k, tokens in calls:
        dense_fp8._linear(torch.zeros(tokens, k, dtype=torch.bfloat16), torch.empty(n, k, dtype=torch.uint8),
                          torch.ones(n // 128, k // 128), (128, 128), False)
    return seen


@pytest.mark.parametrize("gpu, fi_ok, own_rows", [
    ("NVIDIA H200", True, True), ("NVIDIA H200", False, False),
    ("NVIDIA H200 NVL", True, False), ("NVIDIA H100 80GB HBM3", True, False)])
def test_a_gpus_own_token_limits_apply_only_there(monkeypatch, gpu, fi_ok, own_rows):
    """A GPU model's measured w8a16 limits apply to that model alone, and only where
    flashinfer's GEMM runs: without it those sizes would go to cuBLAS's, slower at decode."""
    from mstar.utils import dense_fp8

    own = dense_fp8._W8A16_MAX_TOKENS_BY_GPU["NVIDIA H200"][(*O_PROJ, False)]
    calls = [(*O_PROJ, own + 1), (*O_PROJ, dense_fp8._W8A16_MAX_TOKENS + 1)]
    assert _paths(monkeypatch, gpu, fi_ok, calls) == [
        "_fi" if own_rows else "_w8a16", "_fi" if fi_ok else "_w8a8"]


def test_an_unusable_cublas_block_gemm_falls_back(monkeypatch):
    """torch 2.10-2.12 have the blockwise scaled_mm API, but it raises off SM90 or on a cuBLASLt
    before 12.9 (the cu128 builds): the probe says so, and the large batches take w8a16."""
    from mstar.utils import dense_fp8

    def raises(*args, **kwargs):
        raise NotImplementedError("blockwise scaling needs cuBLASLt >= 12.9")

    monkeypatch.setattr(dense_fp8, "_W8A8_API", True)
    monkeypatch.setattr(dense_fp8, "_w8a8", raises)
    assert dense_fp8._probe_w8a8(torch.device("cpu")) is False
    taken = []
    monkeypatch.setattr(dense_fp8, "_w8a16", lambda *args: taken.append(1) or None)
    monkeypatch.setattr(dense_fp8, "_W8A8_OK", {torch.device("cpu"): False})
    dense_fp8._linear(torch.zeros(600, 256, dtype=torch.bfloat16),
                      torch.zeros(256, 256, dtype=torch.uint8), torch.ones(2, 2), (128, 128), False)
    assert taken == [1]
