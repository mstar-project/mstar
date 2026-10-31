"""Fused SnakeBeta activation for the 12 Hz codec decoder.

The reference ``SnakeBeta`` (``x + 1/exp(beta) * sin(x * exp(alpha))**2``) runs
as five elementwise kernels over the decoder's audio-rate activations, about
half of the conv stack's time. This computes it in one pass with the
per-channel ``exp`` terms precomputed (the weights are frozen at inference).
"""

from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


if triton is not None:

    @triton.jit
    def _snake_kernel(x_ptr, out_ptr, alpha_ptr, inv_beta_ptr, channels, length, BLOCK: tl.constexpr):
        row = tl.program_id(0)  # one (batch, channel) row of a contiguous [B, C, T]
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < length
        channel = row % channels
        alpha = tl.load(alpha_ptr + channel)
        inv_beta = tl.load(inv_beta_ptr + channel)
        offsets = row.to(tl.int64) * length + cols
        x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        s = tl.sin(x * alpha)
        tl.store(out_ptr + offsets, (x + inv_beta * (s * s)).to(out_ptr.dtype.element_ty), mask=mask)


    @triton.jit
    def _snake_channels_last_kernel(
        x_ptr, out_ptr, alpha_ptr, inv_beta_ptr, channels, numel, BLOCK: tl.constexpr,
    ):
        # the same pass over a contiguous channels-last [B, C, 1, T]: every
        # element's position in memory says its channel
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        channel = offsets % channels
        alpha = tl.load(alpha_ptr + channel, mask=mask, other=0.0)
        inv_beta = tl.load(inv_beta_ptr + channel, mask=mask, other=0.0)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        s = tl.sin(x * alpha)
        tl.store(out_ptr + offsets, (x + inv_beta * (s * s)).to(out_ptr.dtype.element_ty), mask=mask)


class FusedSnakeBeta(nn.Module):
    """Drop-in for the reference ``SnakeBeta`` built from its loaded parameters."""

    BLOCK = 1024

    def __init__(self, snake: nn.Module):
        super().__init__()
        with torch.no_grad():
            alpha = torch.exp(snake.alpha.detach().float())
            inv_beta = 1.0 / (torch.exp(snake.beta.detach().float()) + snake.no_div_by_zero)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("inv_beta", inv_beta, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]``, or ``[B, C, 1, T]`` in channels-last memory."""
        if triton is None or not hidden_states.is_cuda:
            shape = (1, -1) + (1,) * (hidden_states.dim() - 2)
            alpha = self.alpha.to(hidden_states.dtype).view(shape)
            inv_beta = self.inv_beta.to(hidden_states.dtype).view(shape)
            return hidden_states + inv_beta * torch.sin(hidden_states * alpha).pow(2)
        if hidden_states.dim() == 4:
            out = torch.empty_like(hidden_states)
            numel = hidden_states.numel()
            _snake_channels_last_kernel[(triton.cdiv(numel, self.BLOCK),)](
                hidden_states, out, self.alpha, self.inv_beta, hidden_states.shape[1], numel,
                BLOCK=self.BLOCK, num_warps=8,
            )
            return out
        x = hidden_states.contiguous()
        out = torch.empty_like(x)
        batch, channels, length = x.shape
        grid = (batch * channels, triton.cdiv(length, self.BLOCK))
        _snake_kernel[grid](x, out, self.alpha, self.inv_beta, channels, length, BLOCK=self.BLOCK)
        return out


def fuse_snake_activations(module: nn.Module) -> int:
    """Swap every reference ``SnakeBeta`` under ``module`` for ``FusedSnakeBeta``; returns the count."""
    swapped = 0
    for name, child in list(module.named_children()):
        if type(child).__name__ == "SnakeBeta":
            setattr(module, name, FusedSnakeBeta(child).to(child.alpha.device))
            swapped += 1
        else:
            swapped += fuse_snake_activations(child)
    return swapped
