"""The DiT components that swap in one-pass kernels inside a compiled region
(``modulation.ada_layer_norm``, ``qk_norm_rope.rms_norm_rope``), against their
eager formulations.

The eager formulation is what the reference ports are held bit-exact in, so the
CPU tests pin it to the expressions a DiT block writes inline. The kernels
reduce in a different order and are held to bf16 rounding of the eager result:
at most one bf16 step anywhere, and almost every element equal.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.diffusion import modulation, qk_norm_rope
from mstar.model.components.diffusion.rope import apply_rotary_interleaved

CUDA = torch.cuda.is_available()
needs_kernels = pytest.mark.skipif(
    not CUDA or modulation.triton is None, reason="needs CUDA and Triton"
)

B, FRAMES, HW, D = 2, 3, 40, 384
EPS = 1e-6


def _stream(device, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(B, FRAMES * HW, D, generator=g) * 3).to(device, dtype)
    y = torch.randn(B, FRAMES * HW, D, generator=g).to(device, dtype)
    # strided per-frame views out of one [B, F, 6, D] table, as a block unbinds them
    mod = (torch.randn(B, FRAMES, 6, D, generator=g) * 0.2).to(device)
    w = (torch.randn(D, generator=g) * 0.1 + 1).to(device)
    b = (torch.randn(D, generator=g) * 0.1).to(device)
    return x, y, mod.unbind(2), w, b


def _inline_block_math(x, y, shift, scale, gate, w, b):
    """The adaLN steps as a Wan block writes them on [B, F, HW, D] views."""
    xs, ys = x.unflatten(1, (FRAMES, HW)), y.unflatten(1, (FRAMES, HW))
    gated = (xs.float() + ys * gate.unsqueeze(2)).type_as(xs)
    mod = (F.layer_norm(xs.float(), (D,), None, None, EPS) * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).type_as(xs)
    affine = F.layer_norm(gated.float(), (D,), w, b, EPS).type_as(xs)
    plain = xs + ys
    plain_mod = (F.layer_norm(plain.float(), (D,), None, None, EPS) * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2))
    return [t.flatten(1, 2) for t in (gated, mod, affine, plain, plain_mod.type_as(xs))]


def test_eager_formulation_is_the_inline_block_math_bitwise():
    x, y, (shift, scale, gate, *_), w, b = _stream("cpu")
    gated, mod, affine, plain, plain_mod = _inline_block_math(x, y, shift, scale, gate, w, b)

    _, out = modulation.ada_layer_norm(x, shift, scale, eps=EPS)
    assert torch.equal(out, mod)
    x_out, out = modulation.ada_layer_norm(x, None, None, eps=EPS, weight=w, bias=b, residual=y, gate=gate)
    assert torch.equal(x_out, gated) and torch.equal(out, affine)
    x_out, out = modulation.ada_layer_norm(x, shift, scale, eps=EPS, residual=y)
    assert torch.equal(x_out, plain) and torch.equal(out, plain_mod)


def test_eager_rms_norm_rope_is_the_modules_bitwise():
    g = torch.Generator(device="cpu").manual_seed(0)
    heads, head_dim, seq = 3, 32, 20
    q = torch.randn(1, seq, heads * head_dim, generator=g).bfloat16()
    k = torch.randn(1, seq, heads * head_dim, generator=g).bfloat16()
    q_norm = nn.RMSNorm(heads * head_dim, eps=EPS, dtype=torch.bfloat16)
    k_norm = nn.RMSNorm(heads * head_dim, eps=EPS, dtype=torch.bfloat16)
    ang = torch.rand(seq, head_dim // 2, generator=g) * 6
    cos, sin = ang.cos().repeat_interleave(2, 1), ang.sin().repeat_interleave(2, 1)
    q_out, k_out = qk_norm_rope.rms_norm_rope(q, k, q_norm, k_norm, cos, sin, heads)
    assert torch.equal(q_out, apply_rotary_interleaved(q_norm(q).unflatten(2, (heads, -1)), cos, sin))
    assert torch.equal(k_out, apply_rotary_interleaved(k_norm(k).unflatten(2, (heads, -1)), cos, sin))


def _assert_bf16_close(got: torch.Tensor, want: torch.Tensor, min_equal: float = 0.999):
    """Within one bf16 step of the eager result everywhere, equal almost everywhere.
    The step is taken at unit scale too: an output that cancels toward zero
    keeps the absolute error of its O(1) terms."""
    assert got.shape == want.shape and got.dtype == want.dtype
    step = (want.float().abs() + 1.0) * 2.0**-7
    assert ((got.float() - want.float()).abs() <= step).all()
    assert (got == want).float().mean().item() >= min_equal


@needs_kernels
@pytest.mark.parametrize("case", ["mod", "gated_affine", "plain_mod", "per_sample"])
def test_ada_layer_norm_kernel_matches_eager(case):
    x, y, (shift, scale, gate, *_), w, b = _stream("cuda")
    if case == "per_sample":
        shift, scale = shift[:, :1], scale[:, :1]
    kwargs = {
        "mod": dict(shift=shift, scale=scale),
        "per_sample": dict(shift=shift, scale=scale),
        "gated_affine": dict(shift=None, scale=None, weight=w, bias=b, residual=y, gate=gate),
        "plain_mod": dict(shift=shift, scale=scale, residual=y),
    }[case]
    want_x, want = modulation.reference_ada_layer_norm(x, eps=EPS, **kwargs)
    compiled = torch.compile(
        lambda: modulation.ada_layer_norm(x, eps=EPS, **kwargs), fullgraph=True, dynamic=False
    )
    got_x, got = compiled()
    _assert_bf16_close(got_x, want_x)
    _assert_bf16_close(got, want)


@needs_kernels
def test_rms_norm_rope_kernel_matches_eager():
    g = torch.Generator(device="cpu").manual_seed(1)
    heads, head_dim, seq = 6, 128, 90
    q = torch.randn(2, seq, heads * head_dim, generator=g).cuda().bfloat16()
    k = torch.randn(2, seq, heads * head_dim, generator=g).cuda().bfloat16()
    q_norm = nn.RMSNorm(heads * head_dim, eps=EPS, device="cuda", dtype=torch.bfloat16)
    k_norm = nn.RMSNorm(heads * head_dim, eps=EPS, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        q_norm.weight.copy_(torch.randn(heads * head_dim, generator=g) * 0.1 + 1)
        k_norm.weight.copy_(torch.randn(heads * head_dim, generator=g) * 0.1 + 1)
    ang = torch.rand(seq, head_dim // 2, generator=g).cuda() * 6
    cos, sin = ang.cos().repeat_interleave(2, 1), ang.sin().repeat_interleave(2, 1)
    with torch.inference_mode():
        want = qk_norm_rope.rms_norm_rope(q, k, q_norm, k_norm, cos, sin, heads)
        got = torch.compile(
            lambda: qk_norm_rope.rms_norm_rope(q, k, q_norm, k_norm, cos, sin, heads), fullgraph=True, dynamic=False
        )()
    for g_t, w_t in zip(got, want, strict=True):
        _assert_bf16_close(g_t, w_t, min_equal=0.99)
