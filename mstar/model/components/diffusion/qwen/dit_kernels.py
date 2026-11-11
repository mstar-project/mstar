"""Fused elementwise kernels for the Qwen-Image DiTs (Edit-2511/2509, 2.1), bit-exact with the eager modules they
replace.

Each kernel rounds to bf16 exactly where the eager ops do, so the fused and the eager forward give the same bits
(the served output stays identical to the reference pipeline's). Off CUDA, or for a shape a kernel does not take,
the callers run the eager ops (``fits``).

* :func:`qk_norm_rope` (one part) and :func:`joint_qk_norm_rope` (text and image parts into one joint tensor):
  ``QwenRMSNorm`` of each 128-wide head, then the complex rotary product. The variance is
  summed in the order of PyTorch's reduction for a contiguous 128-wide row: 32 lanes of 4 consecutive values,
  ``((x0 + x1) + x2) + x3``, then a shuffle-down tree (offsets 16, 8, 4, 2, 1); the rotation is CUDA's complex
  multiply, ``re = fma(a, c, -(b d))``, ``im = fma(a, d, b c)``. Text and image parts go straight into the joint
  q (or k) tensor, so there is no concatenation either. One program per token reads its rotation once for all
  heads; :func:`joint_qk_pair_norm_rope` does q and k in the same launch.
* :func:`modulate`: ``x * (1 + scale) + shift`` with two parameter sets along the tokens (``zero_cond_t``: the
  target's tokens take the step's, the references' timestep 0's), one pass instead of the per-token select.
* :func:`gated_add`: ``x + gate * y``, the same two parameter sets.
* :func:`norm_modulate`: ``modulate(LayerNorm(x))`` (no affine), and :func:`gated_norm_modulate`: the residual
  ``r = x + gate * y`` and ``modulate(LayerNorm(r))`` from one read of ``x`` and ``y``. The statistics are PyTorch's
  ``vectorized_layer_norm_kernel`` (torch 2.12.1, a bf16 row of D % 512 == 0): one row per 128 threads, thread t
  holding the 4-wide vectors t, t + 128, ...; an online Welford over its values in order, ``mean += delta * (1/n)``
  and ``m2 += delta * (v - mean)`` contracted to FMAs; then the warp's shuffle-down tree (16, 8, 4, 2, 1) and the
  four warps' tree (2, 1) of ``cuWelfordCombine(own, other)``; ``var = m2 / D`` and ``rstd = rsqrt(var + eps)``;
  ``y = rstd * (x - mean)``. Both reciprocal and divide are correctly rounded there (no fast math), so here too
  (``tl.div_rn``). A torch that changes its LayerNorm kernel breaks the CUDA test, not the output's correctness.

Every launch turns Triton's fp fusion off: with it on, the compiler folds a bf16 round trip between a multiply and
an add into one fused multiply-add, which drops the rounding the eager ops make there.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

_BLOCK = 1024

if triton is not None:

    @triton.jit
    def _halves(s, n: tl.constexpr):
        # [R, 2n] -> [:, :n] + [:, n:], a shuffle-down step at offset n
        a, b = tl.split(tl.permute(tl.reshape(s, (s.shape[0], 2, n)), (0, 2, 1)))
        return a + b

    @triton.jit
    def _norm_rope_heads(x_ptr, w_ptr, o_ptr, x_off, o_off, sx_h, so_h, c, sn, H, eps, HB: tl.constexpr):
        # one token's heads (HB lanes, H used): QwenRMSNorm of each 128-wide head, then the rotation by (c, sn)
        h = tl.arange(0, HB).to(tl.int64)
        hmask = (h < H)[:, None]
        d = tl.arange(0, 128)
        x = tl.load(x_ptr + x_off + (h * sx_h)[:, None] + d[None, :], mask=hmask, other=0.0)
        x = x.to(tl.float32)
        sq = x * x
        # lane t: elements 4t..4t+3, summed in order
        e0, e1 = tl.split(tl.reshape(sq, (HB, 32, 2, 2)))
        x0, x2 = tl.split(e0)
        x1, x3 = tl.split(e1)
        lane = ((x0 + x1) + x2) + x3
        lane = _halves(lane, 16)
        lane = _halves(lane, 8)
        lane = _halves(lane, 4)
        lane = _halves(lane, 2)
        lane = _halves(lane, 1)
        mean = tl.reshape(lane, (HB,)) * (1.0 / 128.0)
        rstd = tl.math.rsqrt(mean + eps)
        y = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        y = (tl.load(w_ptr + d).to(tl.float32)[None, :] * y).to(tl.bfloat16).to(tl.float32)
        a, b = tl.split(tl.reshape(y, (HB, 64, 2)))
        c = tl.broadcast_to(c, (HB, 64))
        sn = tl.broadcast_to(sn, (HB, 64))
        re = tl.fma(a, c, -(b * sn))
        im = tl.fma(a, sn, b * c)
        out = tl.reshape(tl.join(re, im), (HB, 128)).to(tl.bfloat16)
        tl.store(o_ptr + o_off + (h * so_h)[:, None] + d[None, :], out, mask=hmask)

    @triton.jit
    def _qk_norm_rope_kernel(x_ptr, w_ptr, f_ptr, o_ptr, x2_ptr, w2_ptr, o2_ptr, S, H, eps,
                             sx_r, sx_s, sx_h, sf_r, sf_s, so_r, so_s, so_h, HB: tl.constexpr, PAIR: tl.constexpr):
        # one program per token: its rotation is read once for every head (and for k beside q when PAIR)
        tok = tl.program_id(0)
        r = (tok // S).to(tl.int64)
        s = (tok % S).to(tl.int64)
        f_off = r * sf_r + s * sf_s + 2 * tl.arange(0, 64)
        c = tl.load(f_ptr + f_off)[None, :]
        sn = tl.load(f_ptr + f_off + 1)[None, :]
        x_off = r * sx_r + s * sx_s
        o_off = r * so_r + s * so_s
        _norm_rope_heads(x_ptr, w_ptr, o_ptr, x_off, o_off, sx_h, so_h, c, sn, H, eps, HB)
        if PAIR:
            _norm_rope_heads(x2_ptr, w2_ptr, o2_ptr, x_off, o_off, sx_h, so_h, c, sn, H, eps, HB)

    @triton.jit
    def _params(p_ptr, rows_per_set, row, token, split, D, col, mask):
        # the parameter row of this token: set 0 (row) before ``split``, set 1 (rows_per_set + row) from it
        prow = tl.where(token < split, row, rows_per_set + row).to(tl.int64)
        return tl.load(p_ptr + prow * D + col, mask=mask, other=0.0).to(tl.float32)

    @triton.jit
    def _modulate_kernel(x_ptr, scale1p_ptr, shift_ptr, o_ptr, L, D, rows_per_set, split, BLOCK: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)          # over rows * L tokens
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        row, token = tok // L, tok % L
        x = tl.load(x_ptr + tok * D + col, mask=mask, other=0.0).to(tl.float32)
        scale1p = _params(scale1p_ptr, rows_per_set, row, token, split, D, col, mask)
        shift = _params(shift_ptr, rows_per_set, row, token, split, D, col, mask)
        m = (x * scale1p).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + tok * D + col, (m + shift).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _gated_add_kernel(x_ptr, gate_ptr, y_ptr, o_ptr, L, D, rows_per_set, split, BLOCK: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        row, token = tok // L, tok % L
        x = tl.load(x_ptr + tok * D + col, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + tok * D + col, mask=mask, other=0.0).to(tl.float32)
        gate = _params(gate_ptr, rows_per_set, row, token, split, D, col, mask)
        gy = (gate * y).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + tok * D + col, (x + gy).to(tl.bfloat16), mask=mask)


    @triton.jit
    def _welford(mean, m2, v, n: tl.constexpr):
        # cuWelfordOnlineSum: the n-th value of each lane
        delta = v - mean
        mean = tl.fma(delta, tl.div_rn(1.0, tl.full([], n, tl.float32)), mean)
        return mean, tl.fma(delta, v - mean, m2)

    @triton.jit
    def _welford_halves(mean, m2, n: tl.constexpr, count: tl.constexpr):
        # [R, 2n] -> [R, n]: cuWelfordCombine(own = [:, :n], other = [:, n:]), ``count`` values on each side
        mo, mx = tl.split(tl.permute(tl.reshape(mean, (mean.shape[0], 2, n)), (0, 2, 1)))
        so, sx = tl.split(tl.permute(tl.reshape(m2, (m2.shape[0], 2, n)), (0, 2, 1)))
        c = tl.full([], count, tl.float32)
        share = c * tl.div_rn(1.0, c + c)
        delta = mo - mx
        return tl.fma(share, mx, share * mo), tl.fma(delta * delta * c, share, sx + so)

    @triton.jit
    def _set_params(p_ptr, rows_per_set, row, token, split, D: tl.constexpr, col, mask):
        # [ROWS] tokens' parameter rows (as _params) at columns col [1, 128, 4] -> [ROWS, 128, 4] fp32
        prow = tl.where(token < split, row, rows_per_set + row)
        return tl.load(p_ptr + (prow * D)[:, None, None] + col, mask=mask, other=0.0).to(tl.float32)

    @triton.jit
    def _norm_modulate_kernel(x_ptr, gate_ptr, y_ptr, r_ptr, scale1p_ptr, shift_ptr, o_ptr, n_tok, L, rows_per_set,
                              split, eps, D: tl.constexpr, GATED: tl.constexpr, ROWS: tl.constexpr):
        # ROWS rows of D per program, each reduced as torch reduces one row per block of 128 threads
        tok = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
        row, token = tok // L, tok % L
        base = (tok * D)[:, None, None]
        lane = (4 * tl.arange(0, 128)[:, None] + tl.arange(0, 4)[None, :])[None, :, :]  # [1, 128 threads, 4]
        mask = (tok < n_tok)[:, None, None] & (lane >= 0)
        mean = tl.zeros((ROWS, 128), tl.float32)
        m2 = tl.zeros((ROWS, 128), tl.float32)
        for j in tl.static_range(D // 512):
            col = 512 * j + lane
            if GATED:
                x = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
                gy = (_set_params(gate_ptr, rows_per_set, row, token, split, D, col, mask)
                      * tl.load(y_ptr + base + col, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16)
                r = (x + gy.to(tl.float32)).to(tl.bfloat16)
                tl.store(r_ptr + base + col, r, mask=mask)
                v = r.to(tl.float32)
            else:
                v = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
            e0, e1 = tl.split(tl.reshape(v, (ROWS, 128, 2, 2)))
            v0, v2 = tl.split(e0)
            v1, v3 = tl.split(e1)
            mean, m2 = _welford(mean, m2, v0, 4 * j + 1)
            mean, m2 = _welford(mean, m2, v1, 4 * j + 2)
            mean, m2 = _welford(mean, m2, v2, 4 * j + 3)
            mean, m2 = _welford(mean, m2, v3, 4 * j + 4)
        # thread t = 32 * warp + lane: each warp's tree, then the warps'
        per = D // 128
        mean, m2 = tl.reshape(mean, (ROWS * 4, 32)), tl.reshape(m2, (ROWS * 4, 32))
        mean, m2 = _welford_halves(mean, m2, 16, per)
        mean, m2 = _welford_halves(mean, m2, 8, 2 * per)
        mean, m2 = _welford_halves(mean, m2, 4, 4 * per)
        mean, m2 = _welford_halves(mean, m2, 2, 8 * per)
        mean, m2 = _welford_halves(mean, m2, 1, 16 * per)
        mean, m2 = tl.reshape(mean, (ROWS, 4)), tl.reshape(m2, (ROWS, 4))
        mean, m2 = _welford_halves(mean, m2, 2, 32 * per)
        mean, m2 = _welford_halves(mean, m2, 1, 64 * per)
        rstd = tl.math.rsqrt(tl.div_rn(m2, tl.full([], D, tl.float32)) + eps)[:, :, None]
        mean = mean[:, :, None]
        if GATED:
            tl.debug_barrier()  # the residual this program stored is read back below
        src = r_ptr if GATED else x_ptr
        for j in tl.static_range(D // 512):
            col = 512 * j + lane
            v = tl.load(src + base + col, mask=mask, other=0.0).to(tl.float32)
            normed = (rstd * (v - mean)).to(tl.bfloat16).to(tl.float32)
            m = (normed * _set_params(scale1p_ptr, rows_per_set, row, token, split, D, col, mask)).to(tl.bfloat16)
            shift = _set_params(shift_ptr, rows_per_set, row, token, split, D, col, mask)
            tl.store(o_ptr + base + col, (m.to(tl.float32) + shift).to(tl.bfloat16), mask=mask)


def fits(*tensors: torch.Tensor) -> bool:
    """Whether the fused kernels take these tensors: CUDA bf16 with Triton available."""
    return triton is not None and all(t.is_cuda and t.dtype == torch.bfloat16 for t in tensors)


def _norm_rope_into(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float, out: torch.Tensor,
                    pair: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> None:
    """``out = rope(norm(x))``, and for ``pair = (x2, weight2, out2)`` (same shape and strides) the same with the
    same rotation in the one launch."""
    rows, seq, heads, _ = x.shape
    f = torch.view_as_real(freqs.contiguous())
    sf_r, sf_s = (0, f.stride(0)) if f.dim() == 3 else (f.stride(0), f.stride(1))
    x2, w2, o2 = pair if pair is not None else (x, weight, out)
    if pair is not None and (x2.stride() != x.stride() or o2.stride() != out.stride()):
        raise ValueError("qk_norm_rope pair: both parts must share the strides")
    _qk_norm_rope_kernel[(rows * seq,)](
        x, weight, f, out, x2, w2, o2, seq, heads, eps, x.stride(0), x.stride(1), x.stride(2), sf_r, sf_s,
        out.stride(0), out.stride(1), out.stride(2), HB=triton.next_power_of_2(heads), PAIR=pair is not None,
        enable_fp_fusion=False,
    )


@torch.library.custom_op("mstar::dit_qk_norm_rope", mutates_args=())
def qk_norm_rope(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    """``apply_rope(QwenRMSNorm(x), freqs)``: ``x [N, H, 128]`` with ``freqs`` complex64 ``[N, 64]``, or
    ``x [R, S, H, 128]`` with ``[S, 64]`` / ``[R, S, 64]``; bf16, last dim contiguous. Returns a new tensor."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # one token axis: a single row of N tokens
    _norm_rope_into(x[None] if x.dim() == 3 else x, weight, freqs, eps, out[None] if x.dim() == 3 else out)
    return out


@qk_norm_rope.register_fake
def _(x, weight, freqs, eps):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@torch.library.custom_op("mstar::qwen_image_joint_qk_norm_rope", mutates_args=())
def joint_qk_norm_rope(
    txt: torch.Tensor, img: torch.Tensor, txt_weight: torch.Tensor, img_weight: torch.Tensor,
    txt_freqs: torch.Tensor, img_freqs: torch.Tensor, eps: float,
) -> torch.Tensor:
    """``cat([rope(norm_txt(txt), txt_freqs), rope(norm_img(img), img_freqs)], dim=1)``: ``txt [R, Lt, H, 128]`` and
    ``img [R, Li, H, 128]`` bf16 (last dim contiguous), ``txt_freqs`` complex64 ``[R, Lt, 64]`` (or ``[Lt, 64]``),
    ``img_freqs`` ``[Li, 64]``; each part written straight into its slice of the joint ``[R, Lt + Li, H, 128]``."""
    rows, lt, heads, dim = txt.shape
    out = torch.empty((rows, lt + img.shape[1], heads, dim), dtype=img.dtype, device=img.device)
    _norm_rope_into(txt, txt_weight, txt_freqs, eps, out[:, :lt])
    _norm_rope_into(img, img_weight, img_freqs, eps, out[:, lt:])
    return out


@joint_qk_norm_rope.register_fake
def _(txt, img, txt_weight, img_weight, txt_freqs, img_freqs, eps):
    rows, lt, heads, dim = txt.shape
    return txt.new_empty((rows, lt + img.shape[1], heads, dim))


@torch.library.custom_op("mstar::dit_joint_qk_pair_norm_rope", mutates_args=())
def joint_qk_pair_norm_rope(
    q_txt: torch.Tensor, q_img: torch.Tensor, k_txt: torch.Tensor, k_img: torch.Tensor,
    q_txt_weight: torch.Tensor, q_img_weight: torch.Tensor, k_txt_weight: torch.Tensor, k_img_weight: torch.Tensor,
    txt_freqs: torch.Tensor, img_freqs: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`joint_qk_norm_rope` of q and of k (each part of k shaped and strided as q's), one launch per part."""
    rows, lt, heads, dim = q_txt.shape
    q = torch.empty((rows, lt + q_img.shape[1], heads, dim), dtype=q_img.dtype, device=q_img.device)
    k = torch.empty_like(q)
    _norm_rope_into(q_txt, q_txt_weight, txt_freqs, eps, q[:, :lt], (k_txt, k_txt_weight, k[:, :lt]))
    _norm_rope_into(q_img, q_img_weight, img_freqs, eps, q[:, lt:], (k_img, k_img_weight, k[:, lt:]))
    return q, k


@joint_qk_pair_norm_rope.register_fake
def _(q_txt, q_img, k_txt, k_img, q_txt_weight, q_img_weight, k_txt_weight, k_img_weight, txt_freqs, img_freqs, eps):
    rows, lt, heads, dim = q_txt.shape
    q = q_txt.new_empty((rows, lt + q_img.shape[1], heads, dim))
    return q, torch.empty_like(q)


@torch.library.custom_op("mstar::dit_modulate", mutates_args=())
def modulate(x: torch.Tensor, scale1p: torch.Tensor, shift: torch.Tensor, split: int) -> torch.Tensor:
    """``x * scale1p + shift`` for ``x [R, L, D]`` contiguous and ``[sets * R, D]`` parameters: a token before
    ``split`` takes set 0's row, from it set 1's (one set: ``split = L``)."""
    rows, length, dim = x.shape
    out = torch.empty_like(x)
    _modulate_kernel[(rows * length, triton.cdiv(dim, _BLOCK))](
        x, scale1p, shift, out, length, dim, rows, split, BLOCK=_BLOCK, enable_fp_fusion=False,
    )
    return out


@modulate.register_fake
def _(x, scale1p, shift, split):
    return torch.empty_like(x)


@torch.library.custom_op("mstar::dit_gated_add", mutates_args=())
def gated_add(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor, split: int) -> torch.Tensor:
    """``x + gate * y`` (``x``, ``y`` ``[R, L, D]`` contiguous), ``gate`` by token as in :func:`modulate`."""
    rows, length, dim = x.shape
    out = torch.empty_like(x)
    _gated_add_kernel[(rows * length, triton.cdiv(dim, _BLOCK))](
        x, gate, y, out, length, dim, rows, split, BLOCK=_BLOCK, enable_fp_fusion=False,
    )
    return out


@gated_add.register_fake
def _(x, gate, y, split):
    return torch.empty_like(x)


def norm_fits(x: torch.Tensor) -> bool:
    """Whether :func:`norm_modulate` / :func:`gated_norm_modulate` take ``x``: a contiguous ``[R, L, D]`` with
    ``D % 512 == 0`` (torch's vectorized LayerNorm then gives every thread the same number of values)."""
    return x.dim() == 3 and x.is_contiguous() and x.shape[-1] % 512 == 0


# rows per program and warps of the LayerNorm kernels (ln_micro's sweep on an H200, [2, 8192, 3072])
_LN_ROWS, _LN_WARPS = 8, 8


def _norm_modulate_launch(x, gate, y, residual, scale1p, shift, out, split, eps, gated: bool):
    rows, length, dim = x.shape
    tokens = rows * length
    _norm_modulate_kernel[(triton.cdiv(tokens, _LN_ROWS),)](
        x, gate, y, residual, scale1p, shift, out, tokens, length, rows, split, eps, D=dim, GATED=gated,
        ROWS=_LN_ROWS, num_warps=_LN_WARPS, enable_fp_fusion=False,
    )


@torch.library.custom_op("mstar::dit_norm_modulate", mutates_args=())
def norm_modulate(x: torch.Tensor, scale1p: torch.Tensor, shift: torch.Tensor, split: int, eps: float) -> torch.Tensor:
    """``modulate(layer_norm(x, eps), scale1p, shift, split)``: no affine, ``x`` as :func:`norm_fits` takes, the
    parameters as :func:`modulate`'s."""
    out = torch.empty_like(x)
    _norm_modulate_launch(x, x, x, x, scale1p, shift, out, split, eps, gated=False)
    return out


@norm_modulate.register_fake
def _(x, scale1p, shift, split, eps):
    return torch.empty_like(x)


@torch.library.custom_op("mstar::dit_gated_norm_modulate", mutates_args=())
def gated_norm_modulate(
    x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor, scale1p: torch.Tensor, shift: torch.Tensor, split: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``r = gated_add(x, gate, y, split)`` and ``norm_modulate(r, scale1p, shift, split, eps)``, returned as
    ``(r, out)``; ``y`` contiguous as ``x``."""
    residual, out = torch.empty_like(x), torch.empty_like(x)
    _norm_modulate_launch(x, gate, y, residual, scale1p, shift, out, split, eps, gated=True)
    return residual, out


@gated_norm_modulate.register_fake
def _(x, gate, y, scale1p, shift, split, eps):
    return torch.empty_like(x), torch.empty_like(x)
