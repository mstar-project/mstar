"""Fused AdaRMS normalisation + scale/shift/gate Triton kernel.

Replaces the three-step sequence in Pi05AdaRMSNorm.forward():

    1. _rms_normalize(x)           – two passes over x (cast, square, mean, rsqrt, mul)
    2. modulation.chunk(3, dim=-1) – slices already in registers, free
    3. normed * (1+scale) + shift   – two more passes over x-sized tensors

with a single pass:

    * Load x row → compute RMS in float32 → normalise
    * Load (scale, shift, gate) row from modulation → apply conditioning
    * Store normed output and gate

Falls back to the original eager path on CPU or when Triton is not available.
"""

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _adarms_norm_kernel(
        x_ptr,
        out_ptr,
        gate_ptr,
        scale_ptr,
        shift_ptr,
        gate_mod_ptr,
        AH,
        H,
        eps,
        BLOCK_H: tl.constexpr,
    ):
        """One Triton program per row of x (one (bs_idx, ah_idx) pair).

        pid  — linear index into [BS * AH] rows.
        bs_idx = pid // AH — selects which row of (scale, shift, gate_mod) to load.

        All conditioning tensors are [BS, H]; x and outputs are [BS*AH, H].
        Arithmetic is done in float32; stores are cast back to the pointer dtype.
        """
        pid = tl.program_id(0)
        bs_idx = pid // AH

        offsets = tl.arange(0, BLOCK_H)
        mask = offsets < H

        # Load x row, upcast to float32 for numerics.
        x = tl.load(x_ptr + pid * H + offsets, mask=mask, other=0.0).to(tl.float32)

        # RMS normalisation: var = mean(x^2), normed = x / sqrt(var + eps).
        var = tl.sum(x * x) / H
        rstd = tl.rsqrt(var + eps)
        normed = x * rstd

        # Load conditioning (one row per batch element, broadcast over AH).
        base = bs_idx * H + offsets
        scale = tl.load(scale_ptr + base, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + base, mask=mask, other=0.0).to(tl.float32)
        gate_v = tl.load(gate_mod_ptr + base, mask=mask, other=0.0).to(tl.float32)

        # Apply adaRMS scale/shift.
        result = normed * (1.0 + scale) + shift

        # Store; Triton casts float32 → pointer dtype (bfloat16 under autocast).
        tl.store(out_ptr + pid * H + offsets, result, mask=mask)
        tl.store(gate_ptr + pid * H + offsets, gate_v, mask=mask)


def adarms_norm_fused(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate_mod: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused AdaRMS norm: RMS-normalise x then apply scale/shift conditioning.

    Args:
        x:        float tensor, shape [BS * AH, H], any dtype.
        scale:    shape [BS, H] — the (1 + scale) multiplier.
        shift:    shape [BS, H] — additive shift after norm.
        gate_mod: shape [BS, H] — gate vector returned unchanged.
        eps:      variance epsilon for numerical stability.

    Returns:
        (normed, gate) both shape [BS * AH, H], same dtype as x.

    Falls back to an eager implementation on CPU or without Triton.
    """
    BS_AH, H = x.shape
    BS = scale.shape[0]
    AH = BS_AH // BS

    if not _TRITON_AVAILABLE or not x.is_cuda:
        # Eager fallback.
        x_f32 = x.to(torch.float32)
        var = torch.mean(x_f32 * x_f32, dim=-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(var + eps)
        # scale/shift are [BS, H]; need to broadcast to [BS*AH, H].
        scale_b = scale.unsqueeze(1).expand(BS, AH, H).reshape(BS_AH, H)
        shift_b = shift.unsqueeze(1).expand(BS, AH, H).reshape(BS_AH, H)
        gate_b = gate_mod.unsqueeze(1).expand(BS, AH, H).reshape(BS_AH, H)
        normed = normed * (1.0 + scale_b.to(torch.float32)) + shift_b.to(torch.float32)
        return normed.to(x.dtype), gate_b.to(x.dtype)

    x = x.contiguous()
    scale = scale.contiguous()
    shift = shift.contiguous()
    gate_mod = gate_mod.contiguous()

    out = torch.empty_like(x)
    gate = torch.empty_like(x)

    BLOCK_H = triton.next_power_of_2(H)
    grid = (BS_AH,)
    _adarms_norm_kernel[grid](
        x, out, gate,
        scale, shift, gate_mod,
        AH, H, eps,
        BLOCK_H=BLOCK_H,
    )
    return out, gate
