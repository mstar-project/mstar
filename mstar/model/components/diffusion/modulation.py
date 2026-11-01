"""adaLN for DiT blocks: an optional gated residual add, LayerNorm and the
shift/scale modulation in one pass over the hidden stream.

A DiT block normalizes its residual stream three times, each time right after
adding a branch output back into it, and the result feeds a bf16 GEMM:

    x = (x.float() + y.float() * gate).to(x.dtype)          # the residual add
    out = (LN(x.float()) * (1 + scale) + shift).to(x.dtype)  # adaLN

The modulation vectors (``shift``, ``scale``, ``gate``) are per group of
tokens: ``[B, G, D]``, group ``g`` covering tokens ``[g * S/G, (g+1) * S/G)``.
``G = 1`` is a per-sample modulation (FLUX), ``G = frames`` a per-frame one
(Wan's per-token timestep is constant over a latent frame).

Compiled, Inductor lowers this to a looped two-pass reduction once ``D`` is
past its persistent-reduction width (1024), reading the row twice, so inside a
compiled region the kernel below does it in one: the row is read once into
registers, the mean and variance are taken there, and both outputs are
written once. Eager calls run the same math as plain PyTorch ops, which is
what the reference implementations hold bit-exact. The kernel rounds the
residual to the stream dtype before normalizing it, as eager does; its mean and
variance are reduced in a different order, so it agrees with eager to fp32
rounding rather than bitwise.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except ImportError:  # CPU-only installs
    triton = None


def reference_ada_layer_norm(
    x: torch.Tensor,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None,
    *,
    eps: float,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The eager formulation; see ``ada_layer_norm`` for the arguments."""
    batch, seq, dim = x.shape
    groups = next((t.shape[1] for t in (shift, scale, gate) if t is not None), 1)

    def grouped(t: torch.Tensor) -> torch.Tensor:
        return t.unflatten(1, (groups, seq // groups))

    def per_group(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(2)

    xg = grouped(x)
    if residual is not None:
        branch = grouped(residual).float()
        if gate is not None:
            branch = branch * per_group(gate)
        xg = (xg.float() + branch).to(x.dtype)
    normed = torch.nn.functional.layer_norm(
        xg.float(), (dim,), weight.float() if weight is not None else None,
        bias.float() if bias is not None else None, eps,
    )
    if scale is not None:
        normed = normed * (1 + per_group(scale))
    if shift is not None:
        normed = normed + per_group(shift)
    return xg.flatten(1, 2), normed.to(x.dtype).flatten(1, 2)


if triton is not None:

    @triton.jit
    def _ada_layer_norm_kernel(
        x_ptr, res_ptr, gate_ptr, shift_ptr, scale_ptr, w_ptr, b_ptr, x_out_ptr, out_ptr,
        seq, tokens_per_group, groups, mod_batch_stride, mod_group_stride, dim, eps,
        HAS_RES: tl.constexpr, HAS_GATE: tl.constexpr, HAS_SHIFT: tl.constexpr, HAS_SCALE: tl.constexpr,
        HAS_W: tl.constexpr, HAS_B: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < dim
        offs = row * dim + cols
        batch = row // seq
        group = (row % seq) // tokens_per_group
        mod = batch * mod_batch_stride + group * mod_group_stride + cols

        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if HAS_RES:
            y = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_GATE:
                y = y * tl.load(gate_ptr + mod, mask=mask, other=0.0)
            x = (x.to(tl.float32) + y).to(x_ptr.dtype.element_ty)
            tl.store(x_out_ptr + offs, x, mask=mask)
        xf = x.to(tl.float32)
        mean = tl.sum(xf, axis=0) / dim
        diff = tl.where(mask, xf - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / dim
        # eps arrives as fp64 from a compiled caller; keep the row in fp32
        out = diff * tl.math.rsqrt(var + eps.to(tl.float32))
        if HAS_W:
            out = out * tl.load(w_ptr + cols, mask=mask, other=0.0)
        if HAS_B:
            out = out + tl.load(b_ptr + cols, mask=mask, other=0.0)
        if HAS_SCALE:
            out = out * (1 + tl.load(scale_ptr + mod, mask=mask, other=0.0))
        if HAS_SHIFT:
            out = out + tl.load(shift_ptr + mod, mask=mask, other=0.0)
        tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)

    def _launch(x, residual, gate, shift, scale, weight, bias, eps, x_out, out):
        batch, seq, dim = x.shape
        mods = [t for t in (gate, shift, scale) if t is not None]
        groups = mods[0].shape[1] if mods else 1
        for t in mods:
            assert t.shape == mods[0].shape and t.stride() == mods[0].stride() and t.stride(2) == 1
        mod_strides = (mods[0].stride(0), mods[0].stride(1)) if mods else (0, 0)
        # absent tensors go in as None (constexpr), never as an alias of x: a
        # compiled caller treats every pointer the kernel could store through as
        # mutated and copies it
        wrap_triton(_ada_layer_norm_kernel)[(batch * seq,)](
            x, residual, gate, shift, scale, weight, bias,
            x_out, out, seq, seq // groups, groups, mod_strides[0], mod_strides[1], dim, eps,
            HAS_RES=residual is not None, HAS_GATE=gate is not None, HAS_SHIFT=shift is not None,
            HAS_SCALE=scale is not None, HAS_W=weight is not None, HAS_B=bias is not None,
            BLOCK_D=triton.next_power_of_2(dim), num_warps=8,
        )

    @triton_op("mstar::ada_layer_norm", mutates_args=())
    def _ada_layer_norm_op(
        x: torch.Tensor,
        shift: torch.Tensor | None,
        scale: torch.Tensor | None,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        _launch(x, None, None, shift, scale, weight, bias, eps, None, out)
        return out

    @triton_op("mstar::residual_ada_layer_norm", mutates_args=())
    def _residual_ada_layer_norm_op(
        x: torch.Tensor,
        residual: torch.Tensor,
        gate: torch.Tensor | None,
        shift: torch.Tensor | None,
        scale: torch.Tensor | None,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_out = torch.empty_like(x)
        out = torch.empty_like(x)
        _launch(x, residual, gate, shift, scale, weight, bias, eps, x_out, out)
        return x_out, out


def ada_layer_norm(
    x: torch.Tensor,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None,
    *,
    eps: float,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x', LN(x') * (1 + scale) + shift)`` with ``x' = x + residual * gate``.

    ``x`` and ``residual`` are ``[B, S, D]`` in the stream dtype; ``shift``,
    ``scale`` and ``gate`` are fp32 ``[B, G, D]`` with ``G`` dividing ``S``
    (any of them may be None, as may ``residual``); ``weight`` and ``bias`` are
    the LayerNorm affine, if any. Math is fp32; both outputs are in ``x``'s
    dtype, and ``x'`` is ``x`` itself when there is no residual.
    """
    if triton is not None and x.is_cuda and torch.compiler.is_compiling():
        if residual is None:
            return x, _ada_layer_norm_op(x.contiguous(), shift, scale, weight, bias, eps)
        return _residual_ada_layer_norm_op(
            x.contiguous(), residual.contiguous(), gate, shift, scale, weight, bias, eps,
        )
    return reference_ada_layer_norm(
        x, shift, scale, eps=eps, weight=weight, bias=bias, residual=residual, gate=gate,
    )
