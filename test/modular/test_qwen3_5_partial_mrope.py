"""Qwen3.5's partial MRoPE, written over the whole head, against the slice-and-concatenate form.

The whole-head form must give the same bits: the rotated dims run the same ops, and the passed
dims see ``x * 1 + 0 * 0``.
"""

import pytest
import torch

from mstar.model.qwen3_5.components.rope import apply_partial_mrope


def _reference(q, k, cos, sin):
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    r = cos.shape[-1]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    out = []
    for x in (q, k):
        rot = x[..., :r] * cos + rotate_half(x[..., :r]) * sin
        out.append(torch.cat((rot, x[..., r:]), dim=-1))
    return tuple(out)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("tokens", "rotary_dim"), [(1, 64), (7, 64), (33, 128)])
def test_whole_head_form_is_bit_exact(dtype, tokens, rotary_dim):
    g = torch.Generator().manual_seed(tokens + rotary_dim)
    q = torch.randn(tokens, 8, 256, generator=g).to(dtype)
    k = torch.randn(tokens, 2, 256, generator=g).to(dtype)
    angles = torch.rand(tokens, rotary_dim // 2, generator=g) * 100
    emb = torch.cat((angles, angles), dim=-1)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    for got, want in zip(apply_partial_mrope(q, k, cos, sin), _reference(q, k, cos, sin), strict=True):
        assert torch.equal(got, want)
