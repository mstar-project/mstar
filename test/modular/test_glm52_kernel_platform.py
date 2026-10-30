"""Which GPUs the GLM-5.2 kernels' paths apply to. The GPU checks go through
``current_platform``: an AMD GPU reports a CUDA-style capability (an MI300X is 9.4), which a raw
capability read takes for Hopper. dense_fp8's per-GPU token limits apply only on the GPU model
they were measured on, and only where flashinfer's GEMM runs."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

# GLM-5.2's linears at one TP8 rank, (N, K): o_proj, dense-layer down, q_b
O_PROJ, DENSE_DOWN, Q_B = (6144, 2048), (6144, 1536), (2048, 2048)


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


def test_the_h200_takes_flashinfer_from_3_tokens_for_glm52s_o_proj_and_dense_down(monkeypatch):
    calls = [(*O_PROJ, 2), (*O_PROJ, 3), (*O_PROJ, 32), (*DENSE_DOWN, 2), (*DENSE_DOWN, 3),
             (*Q_B, 3), (*Q_B, 32), (*Q_B, 33)]
    assert _paths(monkeypatch, "NVIDIA H200", True, calls) == [
        "_w8a16", "_fi", "_fi", "_w8a16", "_fi", "_w8a16", "_w8a16", "_fi"]


@pytest.mark.parametrize("gpu, fi_ok", [("NVIDIA H100 80GB HBM3", True), ("NVIDIA H200 NVL", True),
                                        ("NVIDIA H200", False)])
def test_other_gpus_keep_w8a16_to_32_tokens(monkeypatch, gpu, fi_ok):
    """A GPU model without measured rows keeps the shape table; an H200 without flashinfer's
    GEMM keeps w8a16 rather than sending decode sizes to cuBLAS's GEMM."""
    calls = [(*O_PROJ, 3), (*O_PROJ, 32), (*DENSE_DOWN, 3), (*O_PROJ, 33)]
    assert _paths(monkeypatch, gpu, fi_ok, calls) == [
        "_w8a16", "_w8a16", "_w8a16", "_fi" if fi_ok else "_w8a8"]
