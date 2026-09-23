"""GPU: the strided decode recurrence (``kda_decode``) against fla's fused recurrent kernel on the
same slots, with q | k | v read out of the conv output and the beta out of a wider buffer, and the
strided gated norm against the torch reference and fla's kernel."""
import pytest
import torch

from mstar.engine.resources.linear_attn.kda_decode import kda_decode
from mstar.model.kimi_k3.components.gated_norm_kernel import gated_rmsnorm
from mstar.model.kimi_k3.reference.kda import gated_rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")
H, D = 4, 128
P = H * D


@pytest.mark.parametrize("n", [1, 5, 64])
def test_strided_decode_matches_fla(n):
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd

    torch.manual_seed(0)
    wide = torch.randn(n, 3 * P + 40, device=DEV).to(torch.bfloat16)  # q | k | v live in a wider buffer
    y = wide[:, 7:7 + 3 * P]
    mixed = torch.randn(n, 2 * P + H + 16, device=DEV).to(torch.bfloat16)  # the beta is a column slice
    beta = mixed[:, P + 3:P + 3 + H]
    g = torch.randn(n, H, D, device=DEV).to(torch.bfloat16)
    A_log = torch.randn(H, device=DEV)
    dt_bias = torch.randn(P, device=DEV) * 0.1
    pool = torch.randn(n + 3, H, D, D, device=DEV)
    slots = torch.randperm(n + 3, device=DEV)[:n].to(torch.int32)
    ref_pool = pool.clone()
    q, k, v = (t.contiguous() for t in y.split([P, P, P], dim=-1))
    cu = torch.arange(n + 1, device=DEV, dtype=torch.int32)
    o_ref = fused_recurrent_kda_fwd(
        q=q.view(1, n, H, D), k=k.view(1, n, H, D), v=v.view(1, n, H, D), g=g.view(1, n, H, D),
        beta=beta.contiguous().view(1, n, H), A_log=A_log, dt_bias=dt_bias, initial_state=ref_pool, scale=D ** -0.5,
        output_final_state=True, inplace_final_state=True, state_v_first=True, cu_seqlens=cu, ssm_state_indices=slots,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, lower_bound=-5.0)[0]
    o = kda_decode(y, g, beta, A_log, dt_bias, pool, slots, D ** -0.5, -5.0)
    assert o.shape == (n, H, D) and o.dtype == torch.bfloat16
    assert torch.allclose(o.float(), o_ref.view(n, H, D).float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(pool, ref_pool, atol=1e-4, rtol=1e-4)
    untouched = torch.ones(n + 3, dtype=torch.bool, device=DEV)
    untouched[slots.long()] = False
    assert torch.equal(pool[untouched], ref_pool[untouched])


@pytest.mark.parametrize("n", [1, 6])
def test_strided_gated_norm_matches_the_reference(n):
    torch.manual_seed(1)
    o = torch.randn(n, H * D, device=DEV).to(torch.bfloat16)
    mixed = torch.randn(n, 3 * P + 5, device=DEV).to(torch.bfloat16)
    g = mixed[:, 2 * P + 1:3 * P + 1].view(n, H, D)  # a strided gate slice
    w = torch.randn(D, device=DEV) * 0.5 + 1.0
    y = gated_rmsnorm(o.view(n, H, D), g, w, 1e-5).view(n, H * D)
    ref = gated_rms_norm(o.view(n, H, D), g, w, 1e-5).view(n, H * D)
    assert torch.allclose(y.float(), ref.float(), atol=1e-2, rtol=1e-2)
    assert (y != ref).float().mean().item() < 0.02  # bf16 rounding of the same fp32 math: a few flips at most
    try:
        from fla.modules.fused_norm_gate import rms_norm_gated
    except ImportError:
        return
    fla = rms_norm_gated(o.view(-1, D), g.reshape(-1, D).contiguous(), w, None, activation="sigmoid", eps=1e-5)
    assert torch.allclose(y.float(), fla.view(n, H * D).float(), atol=1e-2, rtol=1e-2)
