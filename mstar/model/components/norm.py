"""RMSNorm and AdaRMSNorm.

``RMSNorm`` supports both the standard Llama-style normalization
(``normed * weight``) and Gemma's variant (``normed * (1 + weight)``)
through the ``gemma_mode`` flag. Standard mode dispatches to a fused
accelerator kernel when available; Gemma mode uses an fp32 computation
that matches Hugging Face Gemma exactly.

``AdaRMSNorm`` adds adaRMS conditioning (scale / shift / gate from a
condition vector). Used by pi05's action expert flow-matching path; the
output is consumed by ``GatedDecoderLayer`` rather than a plain residual
add.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.rms_norm import run_rms_norm


class RMSNorm(nn.Module):
    """RMSNorm with optional Gemma-style ``(1 + weight)`` scaling.

    Args:
        hidden_size: feature dimension to normalize over.
        eps: variance epsilon.
        gemma_mode: if True, use ``(1 + weight)`` and a fp32 manual
            implementation (matches HF Gemma exactly; the loaded
            checkpoint weight is centered around zero, not one). If
            False, use ``weight`` and dispatch to the device-appropriate
            implementation.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, gemma_mode: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.gemma_mode = gemma_mode
        # Llama-style starts at 1; Gemma-style starts at 0 because the
        # checkpoint stores (weight - 1).
        init = torch.zeros(hidden_size) if gemma_mode else torch.ones(hidden_size)
        self.weight = nn.Parameter(init)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gemma_mode:
            orig_dtype = hidden_states.dtype
            x = hidden_states.to(torch.float32)
            var = x.square().mean(dim=-1, keepdim=True)
            normed = x * torch.rsqrt(var + self.variance_epsilon)
            normed = normed * (1.0 + self.weight.to(torch.float32))
            return normed.to(orig_dtype)
        # Fused kernels expect 2D input. Reshape/restore so callers can pass
        # arbitrary leading dims, such as [batch, sequence, hidden_size].
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, orig_shape[-1])
        out = run_rms_norm(flat, self.weight, eps=self.variance_epsilon)
        return out.reshape(orig_shape)

    def forward_residual(
        self, hidden_states: torch.Tensor, residual: torch.Tensor,
        comm_group=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``r = hidden_states + residual``; returns ``(norm(r), r)``.

        Keeps add and norm together so inductor fuses them, rather than folding
        the add into the producing GEMM as an ``addmm`` (an extra copy of
        ``residual``). Under TP ``hidden_states`` is a row-parallel partial sum
        that ``comm_group`` reduces in the same fused kernel.
        """
        if comm_group is not None and comm_group.world_size > 1:
            orig_shape = hidden_states.shape
            normed, r = comm_group.allreduce_add_rmsnorm(
                hidden_states.reshape(-1, orig_shape[-1]),
                residual.reshape(-1, orig_shape[-1]),
                self.weight, self.variance_epsilon,
                weight_bias=1.0 if self.gemma_mode else 0.0,
            )
            return normed.reshape(orig_shape), r.reshape(orig_shape)
        if hidden_states.is_cuda and hidden_states.dtype in (torch.bfloat16, torch.float16):
            # One FlashInfer kernel, in place: `residual += hidden_states`, then
            # `hidden_states = norm(residual)`. Inductor's own fusion of the
            # pair is a reduction over a few rows that takes ~3x as long at
            # decode batch sizes. Both inputs are consumed by the caller.
            from flashinfer.norm import fused_add_rmsnorm, gemma_fused_add_rmsnorm

            orig_shape = hidden_states.shape
            h = hidden_states.reshape(-1, orig_shape[-1])
            r = residual.reshape(-1, orig_shape[-1])
            fused = gemma_fused_add_rmsnorm if self.gemma_mode else fused_add_rmsnorm
            fused(h, r, self.weight, self.variance_epsilon)
            return h.reshape(orig_shape), r.reshape(orig_shape)
        r = hidden_states + residual
        return self(r), r

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.variance_epsilon}, gemma_mode={self.gemma_mode}"


class RMSNormGated(nn.Module):
    """RMSNorm scaled by a SiLU gate, for the delta-net family.

    Normalizes first, then gates (order matters). The weight is ``[head_v_dim]``,
    so this runs per head over ``[tokens, num_v_heads, head_v_dim]``. Kept apart
    from ``RMSNorm`` because its casts follow HF's for parity, not FlashInfer's.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self, hidden_states: torch.Tensor, gate: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        # HF's casts, but the weight stays fp32 (see GatedDeltaNet._apply), so
        # this product is fp32 where HF's is model dtype: within a bf16 ulp,
        # and what vLLM's fused gated norm does.
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.variance_epsilon}"


class AdaRMSNorm(nn.Module):
    """RMSNorm with adaRMS conditioning.

    A per-norm ``nn.Linear(cond_dim, hidden_size*3)`` maps a shared
    condition vector to ``(scale, shift, gate)``. The normalization is
    ``rmsnorm(x) * (1 + scale) + shift`` and the gate is returned for the
    enclosing decoder layer to apply at the residual.

    The ``dense.weight`` and ``dense.bias`` are zero-initialized so the
    norm starts as the identity (matches HF Gemma / lerobot openpi).
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.variance_epsilon = eps
        self.dense = nn.Linear(cond_dim, hidden_size * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy import keeps the Triton dependency narrow (only AdaRMSNorm
        # imports it).
        from mstar.utils.adarms_norm import adarms_norm_fused

        BS = cond.shape[0]
        x_flat = x.view(BS * (x.shape[0] // BS), -1).contiguous()

        modulation = self.dense(cond)  # [BS, 3*H]
        H = self.hidden_size
        scale = modulation[:, :H]
        shift = modulation[:, H:2 * H]
        gate = modulation[:, 2 * H:]

        return adarms_norm_fused(x_flat, scale, shift, gate, self.variance_epsilon)
