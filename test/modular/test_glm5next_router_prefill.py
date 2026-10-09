"""The prefill router with its serving bf16 weight (a bf16 GEMM with fp32 output) vs the
torch reference in true fp32 — CUDA + triton. The tolerances of test_router_matches_reference:
the same experts, weights to 1e-5."""
import pytest
import torch

from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components.moe import Glm5NextMoEGate

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_decode._HAS_TRITON),
    reason="fused kernels need CUDA + triton",
)


@pytest.fixture
def fp32_matmul():
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


@pytest.mark.parametrize("tokens", [65, 300, 1036])
def test_bf16_router_matches_reference(tokens, monkeypatch, fp32_matmul):
    torch.manual_seed(2)
    gate = Glm5NextMoEGate(4096, 288, 8, routed_scaling_factor=2.5).cuda()
    with torch.no_grad():
        gate.weight.normal_(0.0, 0.02)
        gate.e_score_correction_bias.normal_(0.0, 0.5)
    gate.weight.data = gate.weight.data.to(torch.bfloat16)
    gate.finalize_weights()
    x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
    calls = []
    real_mm = torch.mm
    monkeypatch.setattr(torch, "mm", lambda *a, **k: calls.append(k) or real_mm(*a, **k))
    w, ids = gate(x)
    assert calls and calls[0].get("out_dtype") == torch.float32  # the bf16 path ran
    monkeypatch.setattr(fused_decode, "_ENABLED", False)
    w_r, ids_r = gate(x)
    ids, order = ids.sort(dim=-1)
    ids_r, order_r = ids_r.sort(dim=-1)
    torch.testing.assert_close(ids, ids_r, rtol=0, atol=0)
    torch.testing.assert_close(w.gather(1, order), w_r.gather(1, order_r), rtol=1e-5, atol=1e-6)
