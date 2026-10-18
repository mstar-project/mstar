"""mstar's own Triton kernels and compiled activations, against torch references,
on AMD GPUs (wave64, at most 16 warps a block, Inductor's ROCm codegen)."""

import pytest
import torch
import torch.nn.functional as F

from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"


def _close(out, ref, rel=2e-3):
    err = ((out.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-12)).item()
    assert err < rel, f"relative error {err:.2e}"


@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("tokens", [1, 16, 300])
def test_compiled_gated_mlp(activation, tokens):
    """The MLP shape every language model runs, compiled the way the engine does."""
    from mstar.model.components.mlp import GatedMLP

    torch.manual_seed(tokens)
    mlp = GatedMLP(512, 1408, activation=activation).to(DEV, torch.bfloat16)
    x = torch.randn(tokens, 512, device=DEV, dtype=torch.bfloat16)
    act = {"silu": F.silu, "gelu": F.gelu,
           "gelu_tanh": lambda t: F.gelu(t, approximate="tanh")}[activation]
    xf = x.double()
    gate = xf @ mlp.gate_proj.weight.double().T
    up = xf @ mlp.up_proj.weight.double().T
    ref = (act(gate) * up) @ mlp.down_proj.weight.double().T
    torch._dynamo.reset()
    compiled = torch.compile(mlp, mode="max-autotune-no-cudagraphs", dynamic=False)
    _close(compiled(x), ref, rel=1e-2)


def test_sampling_prep_kernels():
    """Fused (small vocab, autotuned) and split (large vocab) softmax prep."""
    from mstar.engine.resources.sampler.utils import fused_temperature_softmax

    for vocab in (3072, 151936):
        logits = torch.randn(6, vocab, device=DEV, dtype=torch.bfloat16) * 4
        temperature = torch.tensor([0.0, 0.5, 1.0, 1.5, 0.0, 0.7], device=DEV)
        penalty = torch.tensor([1.0, 1.3, 1.0, 2.0, 1.1, 1.0], device=DEV)
        seen = torch.rand(6, vocab, device=DEV) < 0.1
        probs = fused_temperature_softmax(logits, temperature, penalty, seen, include_greedy=True)
        lf = logits.double()
        pen = torch.where(lf > 0, lf / penalty[:, None].double(), lf * penalty[:, None].double())
        lf = torch.where(seen, pen, lf)
        ref = torch.softmax(lf / temperature.clamp_min(1e-30)[:, None].double(), -1)
        greedy = temperature == 0
        ref[greedy] = F.one_hot(lf[greedy].argmax(-1), vocab).double()
        torch.testing.assert_close(probs.double(), ref, rtol=1e-4, atol=1e-6)


def test_adarms_norm():
    from mstar.utils.adarms_norm import adarms_norm_fused

    bs, ah, h = 3, 10, 1024
    x = torch.randn(bs * ah, h, device=DEV, dtype=torch.bfloat16)
    scale, shift, gate = (torch.randn(bs, h, device=DEV, dtype=torch.bfloat16) for _ in range(3))
    out, out_gate = adarms_norm_fused(x, scale, shift, gate, 1e-6)
    xf = x.double()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    rep = lambda t: t.double().repeat_interleave(ah, 0)  # noqa: E731
    _close(out, normed * (1 + rep(scale)) + rep(shift), rel=5e-3)
    assert torch.equal(out_gate, gate.repeat_interleave(ah, 0))


def test_pi05_image_normalize():
    from mstar.model.pi05.kernels.image_normalize import normalize_float_images

    unit = torch.rand(2, 3, 224, 224, device=DEV)
    torch.testing.assert_close(normalize_float_images(unit), unit * 2 - 1)
    signed = unit * 2 - 1
    torch.testing.assert_close(normalize_float_images(signed), signed)


def test_qwen3_tts_snake():
    from mstar.model.qwen3_tts.components.snake import FusedSnakeBeta

    class Snake(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.alpha = torch.nn.Parameter(torch.randn(c) * 0.1)
            self.beta = torch.nn.Parameter(torch.randn(c) * 0.1)
            self.no_div_by_zero = 1e-9

    ref_mod = Snake(64)
    fused = FusedSnakeBeta(ref_mod).to(DEV)
    x = torch.randn(2, 64, 5000, device=DEV)
    alpha = torch.exp(ref_mod.alpha.detach().double()).to(DEV)[None, :, None]
    beta = torch.exp(ref_mod.beta.detach().double()).to(DEV)[None, :, None]
    ref = x.double() + 1.0 / (beta + 1e-9) * torch.sin(x.double() * alpha) ** 2
    _close(fused(x), ref, rel=1e-5)


def test_attention_helpers():
    from mstar.utils.attention import (
        apply_rope_pos_ids,
        decode_attn_nhd,
        fused_qk_norm_rope,
        rms_norm,
        sliding_window_attn,
    )

    torch.manual_seed(0)
    # rms_norm
    x = torch.randn(17, 1024, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(1024, device=DEV, dtype=torch.bfloat16)
    xf = x.double()
    _close(rms_norm(x, w), xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.double())

    # rope from position ids (split-half), fp32
    def rope_ref(t, pos, theta):
        d = t.shape[-1]
        inv = 1.0 / theta ** (torch.arange(0, d, 2, device=DEV, dtype=torch.float64) / d)
        ang = pos.double()[:, None] * inv[None, :]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        a, b = t.double()[..., : d // 2], t.double()[..., d // 2:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)

    q = torch.randn(9, 8, 128, device=DEV)
    k = torch.randn(9, 2, 128, device=DEV)
    pos = torch.arange(9, device=DEV) * 37
    rq, rk = rope_ref(q, pos, 1e4), rope_ref(k, pos, 1e4)
    apply_rope_pos_ids(q, k, pos, 1e4)
    _close(q, rq, rel=1e-4)
    _close(k, rk, rel=1e-4)

    # q/k norm + rope
    x = torch.randn(9, 4, 64, device=DEV)
    wn = torch.randn(64, device=DEV)
    xf = x.double()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * wn.double()
    ref = rope_ref(normed, pos, 1e6)
    _close(fused_qk_norm_rope(x, wn, pos, 1e-6, 1e6), ref, rel=1e-4)

    # decode attention over an NHD cache
    b, s, hq, hkv, d = 3, 50, 8, 2, 64
    qd = torch.randn(b, hq, d, device=DEV)
    kc = torch.randn(b, 64, hkv, d, device=DEV)
    vc = torch.randn(b, 64, hkv, d, device=DEV)
    out = decode_attn_nhd(qd, kc, vc, s)
    ref = F.scaled_dot_product_attention(
        qd[:, :, None].double(), kc[:, :s].transpose(1, 2).double().repeat_interleave(hq // hkv, 1),
        vc[:, :s].transpose(1, 2).double().repeat_interleave(hq // hkv, 1),
    )[:, :, 0]
    _close(out, ref, rel=1e-4)

    # causal sliding-window attention
    b, s, window = 2, 200, 33
    qs = torch.randn(b, s, hq, d, device=DEV, dtype=torch.bfloat16)
    ks = torch.randn(b, s, hkv, d, device=DEV, dtype=torch.bfloat16)
    vs = torch.randn(b, s, hkv, d, device=DEV, dtype=torch.bfloat16)
    i = torch.arange(s, device=DEV)
    mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)
    ref = F.scaled_dot_product_attention(
        qs.transpose(1, 2).double(),
        ks.transpose(1, 2).double().repeat_interleave(hq // hkv, 1),
        vs.transpose(1, 2).double().repeat_interleave(hq // hkv, 1),
        attn_mask=mask,
    ).transpose(1, 2)
    _close(sliding_window_attn(qs, ks, vs, window), ref, rel=1e-2)
