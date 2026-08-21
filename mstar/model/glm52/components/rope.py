"""Plain interleaved RoPE for GLM-5.2 MLA.

GLM-5.2 reaches 1M context with a large base (rope_theta 8e6) and NO
position-interpolation scaling — unlike Kimi's deepseek_yarn there is no
factor/mscale machinery, so the rotation is textbook RoPE over the
qk_rope_head_dim slice. ``rope_interleave`` in the checkpoint config means
GPT-J pairing (``x[..., ::2]`` / ``x[..., 1::2]``).
"""
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
        # lazily per device (kimi rope.py precedent).
        self._inv_freq_cache: torch.Tensor | None = None

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        cached = self._inv_freq_cache
        if cached is None or cached.device != device:
            exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
            cached = (1.0 / self.base ** (exponent / self.rotary_dim)).to(device)
            self._inv_freq_cache = cached
        return cached

    def forward(
        self, position_ids: torch.Tensor, q_pe: torch.Tensor, k_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate the pe slices.

        Args:
            position_ids: ``(tokens,)`` int positions.
            q_pe: ``(tokens, num_heads, rotary_dim)``.
            k_pe: ``(tokens, 1, rotary_dim)`` (shared MQA rope key).
        Returns:
            rotated ``(q_pe, k_pe)`` in the input dtypes.
        """
        inv_freq = self._get_inv_freq(position_ids.device)
        freqs = torch.outer(position_ids.float(), inv_freq)  # (T, rotary_dim/2)
        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(-2)

        q32, k32 = q_pe.float(), k_pe.float()
        q_rot = q32 * cos + rotate_gptj(q32) * sin
        k_rot = k32 * cos + rotate_gptj(k32) * sin
        return q_rot.to(q_pe.dtype), k_rot.to(k_pe.dtype)
