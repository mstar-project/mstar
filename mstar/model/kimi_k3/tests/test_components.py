"""Kimi K3's new pieces one at a time against the reference math (``reference.py``, from the
checkpoint's ``modeling_kimi_linear.py``): SiTU-GLU on every dispatch path, the attention
residual read, and the latent MoE block."""
import pytest
import torch
from torch import nn

from mstar.model.components.moe import dispatch_experts_fused
from mstar.model.kimi_k3.components.attn_res import attn_res_read, push_block, score_weight
from mstar.model.kimi_k3.components.moe import KimiLatentMoE
from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.model.kimi_k3.tests import reference
from mstar.model.kimi_k3.weight_loader import build_stacked_params, kimi_k3_name_remapper
from mstar.model.loader import load_weights_into
from mstar.utils.situ import SiTU, situ_and_mul


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(2)


def test_situ_matches_the_reference_and_bends_the_up_half():
    g = torch.Generator().manual_seed(0)
    gate, up = torch.randn(64, 32, generator=g) * 8, torch.randn(64, 32, generator=g) * 40
    act = SiTU(4.0, 25.0)
    torch.testing.assert_close(act(gate, up), reference.situ(gate, up, 4.0, 25.0), atol=0, rtol=0)
    # without linear_beta the up half passes through: SiLU-like gate times up
    plain = situ_and_mul(gate, up, 4.0, None)
    torch.testing.assert_close(plain, 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * up)
    assert not torch.allclose(plain, act(gate, up))  # the linear beta bites at |up| ~ 40
    # bf16 in, bf16 out, fp32 inside
    assert act(gate.bfloat16(), up.bfloat16()).dtype == torch.bfloat16
    with pytest.raises(ValueError):
        SiTU(0.0)


def test_naive_expert_dispatch_takes_the_activation():
    g = torch.Generator().manual_seed(1)
    e, hidden, inter, tokens = 4, 16, 8, 5
    w13, w2 = torch.randn(e, 2 * inter, hidden, generator=g), torch.randn(e, hidden, inter, generator=g)
    x = torch.randn(tokens, hidden, generator=g)
    ids = torch.tensor([[0, 2], [1, 3], [2, 2], [3, 0], [1, 1]])
    w = torch.rand(tokens, 2, generator=g)
    act = SiTU(4.0, 25.0)
    got = dispatch_experts_fused(x, w13, w2, e, ids, w, activation=act)
    want = torch.zeros(tokens, hidden)
    for t in range(tokens):
        for k in range(2):
            gate, up = (x[t] @ w13[ids[t, k]].T).chunk(2)
            want[t] += w[t, k] * (act(gate, up) @ w2[ids[t, k]].T)
    torch.testing.assert_close(got, want)
    # silu stays the default
    assert not torch.allclose(dispatch_experts_fused(x, w13, w2, e, ids, w), got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernel needs CUDA")
def test_triton_situ_activation_matches_torch():
    from mstar.utils.fused_moe.kernels import act_and_mul_triton

    g = torch.Generator(device="cuda").manual_seed(0)
    gate_up = (torch.randn(37, 2 * 384, device="cuda", generator=g) * 20).bfloat16()
    out = torch.empty(37, 384, device="cuda", dtype=torch.bfloat16)
    act = SiTU(4.0, 25.0)
    act_and_mul_triton(gate_up, out, activation=act)
    gate, up = gate_up.chunk(2, dim=-1)
    torch.testing.assert_close(out, act(gate, up), atol=1e-2, rtol=1e-2)


def test_attn_res_read_matches_the_reference():
    g = torch.Generator().manual_seed(2)
    t, d = 6, 32
    prefix = torch.randn(t, d, generator=g)
    stack = [torch.randn(t, d, generator=g) for _ in range(3)]
    norm_w, proj_w = 1 + 0.1 * torch.randn(d, generator=g), torch.randn(1, d, generator=g)
    blocks = None
    for b in stack:
        blocks = push_block(blocks, b)
    got = attn_res_read(prefix, blocks, score_weight(norm_w, proj_w), 1e-5)
    torch.testing.assert_close(got, reference.attn_res(prefix, stack, norm_w, proj_w, 1e-5))
    # an empty stack is the prefix itself: a softmax over one candidate
    assert attn_res_read(prefix, None, score_weight(norm_w, proj_w), 1e-5) is prefix
    # a mixture of the candidates: equal scores average them
    flat = attn_res_read(prefix, blocks, torch.zeros(d), 1e-5)
    torch.testing.assert_close(flat, torch.stack(stack + [prefix]).mean(0))


def test_latent_moe_matches_the_reference():
    cfg = KimiK3ModelConfig.reduced()
    sd = reference.random_checkpoint(cfg, seed=3)
    prefix = "model.layers.1.block_sparse_moe."
    parent = nn.Module()
    parent.block_sparse_moe = KimiLatentMoE(cfg)
    stream = ((k[len("model.layers.1."):], v) for k, v in sd.items() if k.startswith(prefix))
    loaded = load_weights_into(parent, stream, stacked_params=build_stacked_params(cfg.num_experts),
                               name_remapper=kimi_k3_name_remapper(False), strict=True)
    assert loaded == {n for n, _ in parent.named_parameters()}
    x = torch.randn(7, cfg.hidden_size, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        got = parent.block_sparse_moe(x)
    torch.testing.assert_close(got, reference.latent_moe(sd, prefix, x, cfg), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton MXFP4 experts kernel needs CUDA")
def test_triton_mxfp4_experts_take_situ():
    """The quantization API's MXFP4 experts kernel (``fused_experts_w4a16``, e2m1 dequantized in
    registers) with SiTU's spec, against the dequantized experts run in torch with SiTU."""
    from mstar.model.components.int4_formats import KIND_E2M1
    from mstar.model.components.moe import dispatch_experts_fused
    from mstar.model.kimi_k3.truncate import dequant_mxfp4
    from mstar.utils.fused_moe.w4a16 import fused_experts_w4a16

    g = torch.Generator(device="cuda").manual_seed(0)
    e, hidden, inter, tokens, top_k = 8, 256, 128, 37, 4
    w1 = torch.randint(0, 256, (e, 2 * inter, hidden // 2), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (e, hidden, inter // 2), dtype=torch.uint8, device="cuda", generator=g)
    s1 = torch.randint(120, 125, (e, 2 * inter, hidden // 32), dtype=torch.uint8, device="cuda", generator=g)
    s2 = torch.randint(120, 125, (e, hidden, inter // 32), dtype=torch.uint8, device="cuda", generator=g)
    x = torch.randn(tokens, hidden, device="cuda", generator=g).bfloat16()
    ids = torch.stack([torch.randperm(e, device="cuda", generator=g)[:top_k] for _ in range(tokens)])
    weights = torch.rand(tokens, top_k, device="cuda", generator=g).softmax(-1)
    act = SiTU(4.0, 25.0)
    out = fused_experts_w4a16(x, w1, w2, s1, s2, None, None, weights, ids, KIND_E2M1, 32, act.spec, False)
    d1 = torch.stack([dequant_mxfp4(w1[i].cpu(), s1[i].cpu()) for i in range(e)]).cuda()
    d2 = torch.stack([dequant_mxfp4(w2[i].cpu(), s2[i].cpu()) for i in range(e)]).cuda()
    ref = dispatch_experts_fused(x.float(), d1.float(), d2.float(), e, ids, weights, activation=act)
    torch.testing.assert_close(out.float(), ref, atol=5e-2 * ref.abs().max().item(), rtol=5e-2)
    silu = fused_experts_w4a16(x, w1, w2, s1, s2, None, None, weights, ids, KIND_E2M1, 32, "silu", False)
    assert (silu.float() - ref).abs().max() > 0.1 * ref.abs().max()


def test_blackwell_fp4_moe_takes_situ_with_both_betas():
    """FlashInfer's TRT-LLM FP4 MoE (B200/B300) runs SwiGLU and SiTU with both betas (its
    activation_type Situ); any other activation keeps MXFP4 experts on the Triton kernel. So does
    SiTU on MXFP4 with bf16 activations, Kimi's W4A16: trtllm-gen has no such kernel (B300 04c)."""
    from mstar.utils.fused_moe.fp4_flashinfer import unsupported_activation

    assert unsupported_activation("silu") is None
    assert unsupported_activation(SiTU(4.0, 25.0), "mxfp4", "bf16") is not None
    assert unsupported_activation(SiTU(4.0, 25.0)) is not None
    assert unsupported_activation(SiTU(4.0, 25.0), "mxfp4", "mxfp8") is None
    assert unsupported_activation(SiTU(4.0, 25.0), "nvfp4") is None
    assert unsupported_activation("silu", "mxfp4", "bf16") is None
    assert unsupported_activation(SiTU(4.0, None)) is not None
    assert unsupported_activation("gelu") is not None
    assert SiTU.parse(SiTU(4.0, 25.0).spec) == SiTU(4.0, 25.0) and SiTU.parse("silu") is None
