"""The DiT components that swap in one-pass kernels inside a compiled region
(``modulation.ada_layer_norm``, ``qk_norm_rope.rms_norm_rope``, ``ffn.gelu_tanh_ffn``),
against their eager formulations.

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
from mstar.model.components.diffusion.ffn import gelu_tanh_ffn
from mstar.model.components.mlp import MLP
from mstar.model.wan22.components.dit import Wan22RoPE3D
from mstar.model.wan22.config import Wan22Config

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


def _slice_write_rotation(x, cos, sin):
    """The reference's interleaved-pair rotation, written the way the Wan port did
    before it moved to the shared stacked form: fp32 math through the promotion
    against the tables, rounded to the stream dtype at the slice assignment."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = cos[None, :, None, 0::2]
    sin = sin[None, :, None, 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="needs CUDA"))]
)
def test_eager_rms_norm_rope_is_the_slice_write_rotation_bitwise(device):
    cfg = Wan22Config()
    heads, head_dim = cfg.num_attention_heads, cfg.attention_head_dim
    cos, sin = Wan22RoPE3D(head_dim, cfg.rope_max_seq_len)((3, 5, 8), torch.device(device))
    seq = cos.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    q = (torch.randn(2, seq, heads * head_dim, generator=g) * 3).to(device, torch.bfloat16)
    k = (torch.randn(2, seq, heads * head_dim, generator=g) * 3).to(device, torch.bfloat16)
    q_norm = nn.RMSNorm(heads * head_dim, eps=EPS, device=device, dtype=torch.bfloat16)
    k_norm = nn.RMSNorm(heads * head_dim, eps=EPS, device=device, dtype=torch.bfloat16)
    q_out, k_out = qk_norm_rope.rms_norm_rope(q, k, q_norm, k_norm, cos, sin, heads)
    assert torch.equal(q_out, _slice_write_rotation(q_norm(q).unflatten(2, (heads, -1)), cos, sin))
    assert torch.equal(k_out, _slice_write_rotation(k_norm(k).unflatten(2, (heads, -1)), cos, sin))


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


def _ffn(device):
    torch.manual_seed(0)
    mlp = MLP(D, 4 * D, D, activation="gelu_tanh", bias=True).to(device, torch.bfloat16)
    with torch.no_grad():
        for param in mlp.parameters():
            param.mul_(0.5)
    return mlp


def test_eager_ffn_is_the_module_bitwise():
    x, *_ = _stream("cpu")
    mlp = _ffn("cpu")
    with torch.no_grad():
        assert torch.equal(gelu_tanh_ffn(x, mlp), mlp(x))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_gelu_epilogue_matches_eager():
    """The epilogue rounds once, after the GELU, where eager rounds the GEMM's
    output and then the GELU's."""
    x, *_ = _stream("cuda")
    mlp = _ffn("cuda")
    with torch.no_grad():
        want = mlp(x)
        hidden = torch._addmm_activation(mlp.linear_in.bias, x.flatten(0, -2), mlp.linear_in.weight.t(), use_gelu=True)
        got = mlp.linear_out(hidden.unflatten(0, x.shape[:-1]))
        compiled = torch.compile(lambda t: gelu_tanh_ffn(t, mlp), dynamic=False)(x)
    # the same GEMM once compiled: the swap took effect and nothing else moved
    assert torch.equal(compiled, got)
    # the down-projection sums thousands of rounded activations, so fewer elements
    # come out equal than for the single-pass kernels
    _assert_bf16_close(got, want, min_equal=0.3)
