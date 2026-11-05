"""``fused_decode.swiglu`` against torch's clamped SwiGLU on CUDA: the same roundings, so
the same bits."""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.glm5_next import fused_decode

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_decode._HAS_TRITON),
    reason="fused kernels need CUDA + triton",
)


@pytest.mark.parametrize("tokens", [1, 7, 300])
@pytest.mark.parametrize("inter", [256, 1536])
def test_swiglu_matches_torch(tokens, inter):
    torch.manual_seed(tokens + inter)
    # past the clamp both ways, and deep in silu's tails
    gate_up = (torch.randn(tokens, 2 * inter, device="cuda") * 8).to(torch.bfloat16)
    gate_up[0, :4] = torch.tensor([-100.0, -30.0, 30.0, 0.0])
    gate, up = gate_up.split(inter, dim=-1)
    ref = F.silu(gate.clamp(max=10.0)) * up.clamp(min=-10.0, max=10.0)
    assert torch.equal(fused_decode.swiglu(gate_up, 10.0), ref)
