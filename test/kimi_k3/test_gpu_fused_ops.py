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


@cuda
@pytest.mark.parametrize("m,d", [(0, 512), (1, 1024), (3, 7168), (8, 7168)])
@pytest.mark.parametrize("t", [1, 37])
def test_attn_res_folded_add_and_norm_are_exact(m, d, t):
    """The residual add folded into the read's first launch gives the same read as the eager
    add followed by the plain read, returns the bit-identical sum, and the folded output norm
    equals the unfused read followed by the KimiRMSNorm module."""
    from mstar.model.kimi_k3.components.attn_res import AttnResRead
    from mstar.model.kimi_k3.components.attn_res_kernel import attn_res_add_read_triton, attn_res_read_triton
    from mstar.model.kimi_k3.components.common import KimiRMSNorm

    torch.manual_seed(0)
    prefix = torch.randn(t, d, device=DEV, dtype=torch.bfloat16) * 2
    add = torch.randn(t, d, device=DEV, dtype=torch.bfloat16) * 3
    blocks = torch.randn(t, m, d, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(d, device=DEV) * 0.05
    norm = KimiRMSNorm(d).to(DEV, torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.2)
        summed = prefix + add
        x, p = attn_res_add_read_triton(prefix, add, blocks, w, 1e-5)
        assert torch.equal(p, summed) and p.data_ptr() != prefix.data_ptr()
        assert torch.equal(x, attn_res_read_triton(summed, blocks, w, 1e-5))
        xn, pn = attn_res_add_read_triton(prefix, add, blocks, w, 1e-5, out_norm_weight=norm.weight, out_eps=norm.variance_epsilon)
        assert torch.equal(pn, summed)
        assert torch.equal(xn, norm(attn_res_read_triton(summed, blocks, w, 1e-5)))
        # the module's read: CUDA (folded) and CPU (eager) paths agree with each other
        mod = AttnResRead(d).to(DEV, torch.bfloat16)
        mod.norm.weight.normal_(1.0, 0.1)
        mod.proj.weight.normal_(0.0, 0.05)
        x_mod, p_mod = mod.read(prefix, blocks, out_norm=norm, add=add)
        assert torch.equal(p_mod, summed)
        assert torch.equal(x_mod, attn_res_add_read_triton(
            prefix, add, blocks, mod.score_weight(), mod.eps, out_norm_weight=norm.weight, out_eps=norm.variance_epsilon)[0])
        mod_cpu = AttnResRead(d).to(torch.bfloat16)
        mod_cpu.load_state_dict(mod.state_dict())
        norm_cpu = KimiRMSNorm(d).to(torch.bfloat16)
        norm_cpu.load_state_dict(norm.state_dict())
        x_cpu, p_cpu = mod_cpu.read(prefix.cpu(), blocks.cpu(), out_norm=norm_cpu, add=add.cpu())
        assert torch.equal(p_cpu, summed.cpu())
        torch.testing.assert_close(x_mod.float().cpu(), x_cpu.float(), rtol=2e-2, atol=2e-2)
        # no blocks and no norm: the read is the sum itself
        x0, p0 = mod.read(prefix, blocks[:, :0], add=add)
        assert torch.equal(x0, summed) and torch.equal(p0, summed)
