"""GPU: the fused MLA output (per-head product with W_UV, output gate, ``[T, H * V]`` layout) against
the torch expression it replaces, with a strided gate slice."""
import pytest
import torch

from mstar.model.kimi_k3.components.mla_out_kernel import mla_out

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")
H, L, V = 12, 512, 128


@pytest.mark.parametrize("t", [1, 8, 33])
@pytest.mark.parametrize("gated", [True, False])
def test_mla_out_matches_torch(t, gated):
    torch.manual_seed(0)
    o_lat = (torch.randn(t, H, L, device=DEV) * 0.5).to(torch.bfloat16)
    w_uv = (torch.randn(H, L, V, device=DEV) * 0.05).to(torch.bfloat16)
    mixed = torch.randn(t, 300 + H * V, device=DEV).to(torch.bfloat16)
    gate = mixed[:, 300:] if gated else None
    y = mla_out(o_lat, w_uv, gate)
    ref = torch.einsum("thl,hlv->thv", o_lat, w_uv).reshape(t, H * V)
    if gated:
        ref = ref * torch.sigmoid(gate)
    assert y.shape == ref.shape and y.dtype == ref.dtype and y.is_contiguous()
    assert torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=2e-2)
    # the same fp32 sums in a different order: bf16 flips on a small share of the entries at most
    assert (y != ref).float().mean().item() < 0.05
