"""GPU tests: the fused elementwise kernels the decode step runs on CUDA against the torch
reference modules (which remain the CPU path)."""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


@cuda
@pytest.mark.parametrize("d", [512, 1536, 3584, 7168])
@pytest.mark.parametrize("t", [1, 7, 64])
def test_rmsnorm_kernel_is_bit_exact(d, t):
    from mstar.model.kimi_k3.components.common import KimiRMSNorm
    from mstar.model.kimi_k3.components.rmsnorm_kernel import kimi_rmsnorm_triton

    torch.manual_seed(0)
    norm = KimiRMSNorm(d).to(DEV, torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.2)
        x = torch.randn(t, d, device=DEV, dtype=torch.bfloat16) * 4
        xf = x.float()
        ref = norm.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)).to(x.dtype)
        assert torch.equal(kimi_rmsnorm_triton(x, norm.weight, norm.variance_epsilon), ref)
        assert torch.equal(norm(x), ref)  # the module takes the kernel path on CUDA
        # leading dims and a strided input
        x3 = torch.randn(2, t, d, device=DEV, dtype=torch.bfloat16)
        assert torch.equal(norm(x3), norm(x3.reshape(-1, d)).view(2, t, d))
        assert torch.equal(norm(x3[:, :, :d]), norm(x3))


@cuda
def test_situ_kernel_matches_reference():
    from mstar.model.kimi_k3.components.common import SiTUAndMul
    from mstar.model.kimi_k3.reference.situ import situ_and_mul

    torch.manual_seed(0)
    act = SiTUAndMul(4.0, 25.0)
    x = torch.randn(37, 2 * 768, device=DEV, dtype=torch.bfloat16) * 6
    out = act(x).float()
    ref = situ_and_mul(x, 4.0, 25.0).float()
    assert out.shape == ref.shape
    # fp32 math in both, rounded once to bf16: at most one ulp apart
    assert (out - ref).abs().max() <= (ref.abs().max() * 2 ** -7)
    assert ((out - ref).abs() > 0).float().mean() < 0.05


@cuda
def test_kda_gated_norm_matches_reference():
    from mstar.model.kimi_k3.components.kda import ParallelKDAAttention
    from mstar.model.kimi_k3.reference.kda import gated_rms_norm

    pytest.importorskip("fla.modules.fused_norm_gate")
    torch.manual_seed(0)
    mod = ParallelKDAAttention(hidden_size=256, num_heads=4, head_dim=128, conv_kernel_size=4).to(DEV, torch.bfloat16)
    with torch.no_grad():
        mod.o_norm.weight.normal_(1.0, 0.2)
        o = torch.randn(9, 4, 128, device=DEV, dtype=torch.bfloat16)
        g = torch.randn(9, 4, 128, device=DEV, dtype=torch.bfloat16) * 2
        out = mod._gated_norm(o, g).float()
        ref = gated_rms_norm(o, g, mod.o_norm.weight, mod.norm_eps).float()
    rel = (out - ref).norm() / ref.norm()
    assert rel < 5e-3, rel


@cuda
@pytest.mark.parametrize("t", [1, 9, 64])
def test_fused_router_matches_reference(t):
    from mstar.model.kimi_k3.components.moe import NoAuxTCRouter
    from mstar.model.kimi_k3.reference.router import noaux_tc_route

    torch.manual_seed(0)
    e, k, h = 224, 16, 512
    router = NoAuxTCRouter(hidden_size=h, num_experts=e, top_k=k).to(DEV, torch.bfloat16)
    with torch.no_grad():
        router.weight.normal_(std=0.02)
        router.e_score_correction_bias.normal_(std=0.05)
        x = torch.randn(t, h, device=DEV, dtype=torch.bfloat16)
        idx, w = router(x)
        ref_idx, ref_w = noaux_tc_route(
            x, router.weight, router.e_score_correction_bias, k, scoring=router.scoring,
            renormalize=router.renormalize, routed_scaling_factor=router.routed_scaling_factor,
            num_expert_group=router.num_expert_group, topk_group=router.topk_group,
        )
    assert idx.shape == (t, k) and w.shape == (t, k) and w.dtype == torch.float32
    for i in range(t):
        assert set(idx[i].tolist()) == set(ref_idx[i].tolist())
    mine = torch.zeros(t, e, device=DEV).scatter_(1, idx.long(), w)
    ref = torch.zeros(t, e, device=DEV).scatter_(1, ref_idx.long(), ref_w)
    torch.testing.assert_close(mine, ref, rtol=1e-5, atol=1e-6)
    # a reloaded gate weight must not be served from the fp32 cache
    with torch.no_grad():
        router.weight.copy_(torch.randn_like(router.weight) * 0.02)
        idx2, _ = router(x)
        ref_idx2, _ = noaux_tc_route(
            x, router.weight, router.e_score_correction_bias, k, scoring=router.scoring,
            renormalize=router.renormalize, routed_scaling_factor=router.routed_scaling_factor,
            num_expert_group=router.num_expert_group, topk_group=router.topk_group,
        )
    assert all(set(idx2[i].tolist()) == set(ref_idx2[i].tolist()) for i in range(t))
