"""The packed attention prologue (``attn.prologue``): its reference does what a
Qwen3.5 layer does unfused, and the rotary tables it reads agree with the
step's. CPU only."""

from __future__ import annotations

import torch

from mstar.engine.resources.attn import prologue
from mstar.model.qwen3_5.components.rope import apply_partial_mrope, compute_3d_cos_sin, compute_inv_freq

T, H, KH, D, HALF = 5, 4, 2, 32, 4


def _args(gated=True):
    shape = {"tokens": T, "heads": H, "kv_heads": KH, "head_dim": D, "half": HALF}
    args = list(prologue._inputs(shape, torch.float32, "cpu", torch.Generator().manual_seed(0)))
    if not gated:
        args[0] = args[0][:, H * D:].contiguous()
        args[-1] = False
    return args


def _unfused(qkv, qw, kw, cos, sin, pos, gated):
    q_size = H * D * (2 if gated else 1)
    q, k, v = qkv.split([q_size, KH * D, KH * D], -1)
    gate = None
    if gated:
        q, gate = q.reshape(T, H, 2 * D).chunk(2, -1)
    else:
        q = q.reshape(T, H, D)

    def norm(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)

    c, s = torch.cat([cos[pos], cos[pos]], -1), torch.cat([sin[pos], sin[pos]], -1)
    q, k = apply_partial_mrope(norm(q, qw), norm(k.reshape(T, KH, D), kw), c, s)
    return q, gate, k, v.reshape(T, KH, D)


def test_the_reference_is_the_layers_unfused_math():
    for gated in (True, False):
        args = _args(gated)
        q, gate, k, v = prologue.QKV_NORM_ROPE_CACHE(*args)
        qkv, qw, kw, cos, sin, pos, key_cache, value_cache, slots = args[:9]
        eq, egate, ek, ev = _unfused(qkv, qw, kw, cos, sin, pos, gated)
        torch.testing.assert_close(q, eq, rtol=0, atol=0)
        torch.testing.assert_close(k, ek, rtol=0, atol=0)
        torch.testing.assert_close(v, ev, rtol=0, atol=0)
        if gated:
            torch.testing.assert_close(gate, egate, rtol=0, atol=0)
        else:
            assert gate.numel() == 0
        block = key_cache.shape[1]
        torch.testing.assert_close(key_cache[slots // block, slots % block], ek, rtol=0, atol=0)
        torch.testing.assert_close(value_cache[slots // block, slots % block], ev, rtol=0, atol=0)
        assert v.data_ptr() != qkv.data_ptr()


def test_no_implementation_off_amd_gpus():
    assert prologue.available() is False


def test_the_rope_tables_are_the_steps_tables():
    """One position grid per token: the three MRoPE grids agree, so the 1D
    table rounds to what ``build_cos_sin`` gives a decode step."""
    inv_freq = compute_inv_freq(64, 1e7)
    pos = torch.tensor([0, 1, 17, 4095])
    cos, sin = compute_3d_cos_sin(pos[None].expand(3, -1), inv_freq, [11, 11, 10], torch.bfloat16)
    freqs = inv_freq[None, :].float() * torch.arange(4096).float()[:, None]
    table_cos, table_sin = freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)
    assert torch.equal(cos[:, :32], table_cos[pos]) and torch.equal(cos[:, 32:], table_cos[pos])
    assert torch.equal(sin[:, :32], table_sin[pos])
