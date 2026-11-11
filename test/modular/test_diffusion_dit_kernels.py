"""The DiT components that swap in one-pass kernels inside a compiled region
(``modulation.ada_layer_norm``, ``qk_norm_rope.rms_norm_rope``, ``ffn.gelu_tanh_ffn``),
against their eager formulations.

The eager formulation is what the reference ports are held bit-exact in, so the
CPU tests pin it to the expressions a DiT block writes inline. The kernels
reduce in a different order and are held to bf16 rounding of the eager result:
at most one bf16 step anywhere, and almost every element equal.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.diffusion import modulation, qk_norm_rope
from mstar.model.components.diffusion.ffn import addmm_gelu_tanh, gelu_tanh_ffn
from mstar.model.components.diffusion.qwen import dit_kernels
from mstar.model.components.mlp import MLP
from mstar.model.components.norm import QwenRMSNorm
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
@pytest.mark.parametrize("residual", [False, True], ids=["plain", "residual"])
def test_ada_layer_norm_kernel_compiles_with_a_float_eps(monkeypatch, caplog, torch_flag, residual):
    """torch 2.11 gives Triton every non-tensor argument as a Python constant, and 2.12
    only the constexpr ones, so a kernel that calls ``eps.to`` fails to compile on 2.11
    alone. torch catches that error and treats every input as mutated, and the output
    still matches eager."""
    # caplog listens on the root logger, which torch's loggers don't propagate to
    monkeypatch.setattr(logging.getLogger("torch._dynamo"), "propagate", True)
    # a graph cached by an earlier compile of these shapes would skip the analysis
    torch_flag(torch._inductor.config, "fx_graph_cache", False)
    x, y, (shift, scale, *_), *_ = _stream("cuda")
    kwargs = dict(shift=shift, scale=scale, residual=y if residual else None)
    want_x, want = modulation.reference_ada_layer_norm(x, eps=EPS, **kwargs)
    got_x, got = torch.compile(
        lambda: modulation.ada_layer_norm(x, eps=EPS, **kwargs), fullgraph=True, dynamic=False
    )()
    _assert_bf16_close(got_x, want_x)
    _assert_bf16_close(got, want)
    assert "assuming every input is mutated" not in caplog.text, "the kernel failed to compile in torch's analysis"


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


def test_addmm_gelu_tanh_off_cuda_is_the_tanh_gelu_of_the_linear():
    """``_addmm_activation`` applies the erf GELU on CPU; the helper must still mean tanh."""
    x, *_ = _stream("cpu")
    mlp = _ffn("cpu")
    lin = mlp.linear_in
    with torch.no_grad():
        assert torch.equal(addmm_gelu_tanh(x, lin.weight, lin.bias), F.gelu(lin(x), approximate="tanh"))


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


@pytest.mark.skipif(not CUDA or dit_kernels.triton is None, reason="needs CUDA and Triton")
def test_dit_kernels_round_as_the_eager_ops():
    """The Qwen-Image DiTs' fused kernels (``qwen/dit_kernels.py``) are bit for bit the eager ops they replace:
    the per-head QwenRMSNorm then the complex rotary product, the modulation and gated residual with two parameter
    sets along the tokens, and the LayerNorms with them (the statistics in torch's own order)."""
    dev, bf16 = "cuda", torch.bfloat16
    g = torch.Generator().manual_seed(1)
    # the self-check serving runs on first use, through the same entry points
    assert dit_kernels._kernels_off_eager(torch.device("cuda"), torch.bfloat16) == []

    def rand(*shape, scale=1.0, shift=0.0):
        return (torch.randn(*shape, generator=g) * scale + shift).to(dev, bf16)

    def rope(x, freqs):
        pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(pairs * freqs.unsqueeze(-2)).flatten(-2).type_as(x)

    def freqs(*shape):
        return torch.polar(torch.ones(*shape, 64), torch.randn(*shape, 64, generator=g)).to(dev)

    # three heads: a head count that is not a power of two (the masked lanes); one token axis, then [R, S, H, 128]
    norm = QwenRMSNorm(128, 1e-6).to(dev, bf16)
    with torch.no_grad():
        norm.weight.normal_(std=0.2)
    x, f = rand(40, 3, 128), freqs(40)
    assert torch.equal(dit_kernels.qk_norm_rope(x, norm.weight, f, norm.eps), rope(norm(x), f))
    k = rand(40, 3, 128)
    kn = QwenRMSNorm(128, 1e-6).to(dev, bf16)
    with torch.no_grad():
        kn.weight.normal_(std=0.2)
    got = dit_kernels.qk_pair_norm_rope(x, k, norm.weight, kn.weight, f, norm.eps)
    assert all(map(torch.equal, got, (rope(norm(x), f), rope(kn(k), f))))
    x = rand(2, 40, 3, 128)
    for f in (freqs(40), freqs(2, 40)):
        assert torch.equal(dit_kernels.qk_norm_rope(x, norm.weight, f, norm.eps), rope(norm(x), f))
    # the joint op: text and image parts of q and k written into one tensor each
    q_txt, q_img, k_txt, k_img = rand(2, 5, 3, 128), rand(2, 40, 3, 128), rand(2, 5, 3, 128), rand(2, 40, 3, 128)
    f_txt, f_img = freqs(2, 5), freqs(40)
    got = dit_kernels.joint_qk_pair_norm_rope(
        q_txt, q_img, k_txt, k_img, norm.weight, norm.weight, kn.weight, kn.weight, f_txt, f_img, norm.eps)
    parts = ((q_txt, q_img, norm), (k_txt, k_img, kn))
    want = [torch.cat([rope(n(t), f_txt), rope(n(i), f_img)], 1) for t, i, n in parts]
    assert all(map(torch.equal, got, want))

    # the served width (3072) on rows far from zero mean (the statistics' order shows), an odd token count
    x, y = rand(2, 37, 3072, scale=4, shift=30), rand(2, 37, 3072)
    gate, shift, scale = rand(4, 3072), rand(4, 3072), rand(4, 3072)
    scale1p = 1 + scale
    layer_norm = torch.nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)

    def per_token(p):  # set 0 (rows 0:2) for the first 16 tokens, set 1 (rows 2:4) from there
        return torch.cat([p[:2, None].expand(2, 16, 3072), p[2:, None].expand(2, 21, 3072)], dim=1)

    def modulated(a):
        return a * per_token(scale1p) + per_token(shift)

    assert torch.equal(dit_kernels.modulate(x, scale1p, shift, 16), modulated(x))
    # parameters that are views of a wider tensor (a ``chunk`` of a linear's output), not contiguous
    wide = rand(4, 2 * 3072)
    scale1p_view, shift_view = (1 + wide[:, :3072]), wide[:, 3072:]
    assert not shift_view.is_contiguous()
    assert torch.equal(dit_kernels.modulate(x, scale1p_view, shift_view, 16),
                       x * per_token(scale1p_view) + per_token(shift_view))
    residual = x + per_token(gate) * y
    assert torch.equal(dit_kernels.gated_add(x, gate, y, 16), residual)
    assert torch.equal(dit_kernels.norm_modulate(x, scale1p, shift, 16, layer_norm.eps), modulated(layer_norm(x)))
    got = dit_kernels.gated_norm_modulate(x, gate, y, scale1p, shift, 16, layer_norm.eps)
    assert all(map(torch.equal, got, (residual, modulated(layer_norm(residual)))))


def test_dit_kernels_refuse_inputs_they_would_read_wrongly():
    """The rotation kernel reads x with a unit last stride and freqs with one row or one per x row; anything else
    would read the wrong memory, so it is an error, not a silent mis-rotation."""
    x, w = torch.zeros(2, 5, 3, 128), torch.ones(128)
    strided = x.transpose(-1, -2).contiguous().transpose(-1, -2)  # the same shape, last stride 3
    with pytest.raises(ValueError, match="contiguous"):
        dit_kernels.qk_norm_rope(strided, w, torch.ones(5, 64, dtype=torch.complex64), 1e-6)
    with pytest.raises(ValueError, match="rows 1 or 2"):
        dit_kernels.qk_norm_rope(x, w, torch.ones(3, 5, 64, dtype=torch.complex64), 1e-6)
    # the modulations read x contiguous and the parameters one row per batch row (two sets: twice that)
    a, p = torch.zeros(2, 6, 8), torch.zeros(2, 8)
    with pytest.raises(ValueError, match="contiguous"):
        dit_kernels.modulate(a.transpose(0, 1), p, p, 6)
    with pytest.raises(ValueError, match="contiguous"):
        dit_kernels.gated_add(a, p, a.transpose(1, 2), 6)
    with pytest.raises(ValueError, match="contiguous"):
        dit_kernels.gated_add(a, p, torch.zeros(2, 3, 8), 6)  # y of another shape
    with pytest.raises(ValueError, match="do not fit"):
        dit_kernels.modulate(a, p, p, 3)  # two sets needed: 4 rows
    with pytest.raises(ValueError, match="do not fit"):
        dit_kernels.modulate(a, torch.zeros(4, 8), torch.zeros(4, 8), 6)  # one set: 2 rows


def _mismatch(calls):
    return lambda device, dtype: calls.append(device) or ["modulate"]


def _raises(calls):
    def check(device, dtype):
        calls.append(device)
        raise RuntimeError("no kernel image")
    return check


@pytest.mark.parametrize("check, logged", [(_mismatch, "modulate"), (_raises, "raised RuntimeError")])
def test_dit_kernels_fall_back_to_eager_once_on_a_device_they_do_not_reproduce(monkeypatch, caplog, check, logged):
    """A device where a kernel is not bit-equal to the eager ops, or does not compile, runs the eager ops, with one
    warning, checked once."""
    monkeypatch.setattr(dit_kernels, "_VERIFIED", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)  # no CUDA runtime on a CPU box
    calls = []
    monkeypatch.setattr(dit_kernels, "_kernels_off_eager", check(calls))
    with caplog.at_level(logging.WARNING, logger=dit_kernels.logger.name):
        assert not any(dit_kernels._reproduces_eager(torch.device("cpu"), torch.bfloat16) for _ in range(3))
    assert len(calls) == 1 and len(caplog.records) == 1 and logged in caplog.text
