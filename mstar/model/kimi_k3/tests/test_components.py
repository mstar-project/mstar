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
