"""Plain interleaved RoPE for GLM-5.2 MLA."""
from __future__ import annotations

import torch
from torch import nn


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    """Interleaved (GPT-J) rotate: pairs ``x[..., ::2]`` / ``x[..., 1::2]``."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class Glm52RotaryEmbedding(nn.Module):
    """Unscaled interleaved rotary embedding over ``rotary_dim`` dims."""

    def __init__(self, rotary_dim: int, base: float) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.base = base
        # Not a registered buffer: meta->to_empty leaves derived buffers
        # uninitialized and model.to(bf16) would downcast it. Recompute fp32
        # lazily per device instead.
        self._inv_freq_cache: torch.Tensor | None = None

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        cached = self._inv_freq_cache
        if cached is None or cached.device != device:
            exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
            cached = (1.0 / self.base ** (exponent / self.rotary_dim)).to(device)
            self._inv_freq_cache = cached
        return cached

    def cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` for ``position_ids``, each ``(T, 1, rotary_dim)`` fp32."""
        inv_freq = self._get_inv_freq(position_ids.device)
        freqs = torch.outer(position_ids.float(), inv_freq)  # (T, rotary_dim/2)
        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(-2)
        return cos, sin

    def forward(
        self,
        position_ids: torch.Tensor,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate the pe slices."""
        cos, sin = self.cos_sin(position_ids) if cos_sin is None else cos_sin

        q32, k32 = q_pe.float(), k_pe.float()
        q_rot = q32 * cos + rotate_gptj(q32) * sin
        k_rot = k32 * cos + rotate_gptj(k32) * sin
        return q_rot.to(q_pe.dtype), k_rot.to(k_pe.dtype)
