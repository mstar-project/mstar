"""deepseek_yarn RoPE for Kimi-K2.7 / DeepSeek-V3 MLA.

MLA rotates only the decoupled ``qk_rope_head_dim`` slice of q/k, with YARN
(NTK-by-parts) frequency scaling and an ``mscale`` amplitude on cos/sin. mstar's
``cache_manager.apply_rope`` (FlashInfer) does not implement YARN, so this is a
standalone rotary module the MLA attention applies itself (analogous to how
Qwen3-Omni applies its 3D MRoPE outside the cache handle).

Style is **interleaved / GPT-J** (``is_neox_style=False`` in DeepSeek): cos/sin
are ``repeat_interleave(2)`` and adjacent even/odd pairs are rotated. Mirrors
vLLM ``layers/rotary_embedding/deepseek_scaling_rope.py::DeepseekScalingRotaryEmbedding``
and the YARN helpers in ``rotary_embedding/common.py``.

Two ``mscale`` values (both use the 2-arg ``yarn_get_mscale``):
  - **amplitude** on cos/sin (here): ``get_mscale(f, mscale) / get_mscale(f, mscale_all_dim) * attn_factor``.
  - **softmax-scale boost** (in the attention, not here): ``get_mscale(f, mscale_all_dim) ** 2``.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """DeepSeek 2-arg mscale (deepseek_v2.py:428 / deepseek_scaling_rope.py:20)."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    """Interleaved (GPT-J) rotate: pairs ``x[..., ::2]`` / ``x[..., 1::2]``
    (vLLM ``common.py::rotate_gptj``)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _yarn_find_correction_dim(num_rotations, dim, base, max_pos) -> float:
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_pos) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low, high, dim, dtype) -> torch.Tensor:
    if low == high:
        high += 0.001  # avoid singularity
    ramp = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


class KimiYarnRotaryEmbedding(nn.Module):
    """deepseek_yarn rotary embedding over the ``rotary_dim`` (=qk_rope_head_dim) slice."""

    def __init__(
        self,
        rotary_dim: int,
        base: float,
        factor: float,
        original_max_position_embeddings: int,
        beta_fast: float = 32,
        beta_slow: float = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim

        # ``inv_freq`` is NOT a registered buffer. Buffers computed in ``__init__``
        # do not survive the production ``meta`` build -> ``to_empty(device)`` ->
        # ``load_weights`` path: ``to_empty`` allocates uninitialized memory and
        # never re-runs ``__init__``, and ``inv_freq`` is not in the checkpoint
        # (it's derived, skipped by the loader) — so a buffer would be left as
        # garbage after loading, silently corrupting YARN RoPE. Instead keep the
        # scalar recipe and compute ``inv_freq`` lazily in fp32 on the target
        # device (also keeps it fp32 under a bf16 model, matching DeepSeek, rather
        # than being downcast by ``model.to(bf16)``).
        self._inv_freq_args = (
            rotary_dim, base, factor, original_max_position_embeddings,
            beta_fast, beta_slow, extrapolation_factor,
        )
        self._inv_freq_cache: torch.Tensor | None = None

        # cos/sin amplitude (deepseek_scaling_rope.py:56-60).
        self.mscale = float(
            yarn_get_mscale(factor, mscale)
            / yarn_get_mscale(factor, mscale_all_dim)
            * attn_factor
        )

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        """Return the fp32 ``inv_freq`` for ``device``, computing + caching once."""
        cached = self._inv_freq_cache
        if cached is None or cached.device != device:
            cached = self._compute_inv_freq(*self._inv_freq_args).to(device=device)
            self._inv_freq_cache = cached
        return cached

    @staticmethod
    def _compute_inv_freq(
        rotary_dim, base, factor, max_pos, beta_fast, beta_slow, extrapolation_factor,
    ) -> torch.Tensor:
        pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, base, max_pos
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, torch.float)
        ) * extrapolation_factor
        return (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

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
        cos = (freqs.cos() * self.mscale).repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = (freqs.sin() * self.mscale).repeat_interleave(2, dim=-1).unsqueeze(-2)

        q32, k32 = q_pe.float(), k_pe.float()
        q_rot = q32 * cos + rotate_gptj(q32) * sin
        k_rot = k32 * cos + rotate_gptj(k32) * sin
        return q_rot.to(q_pe.dtype), k_rot.to(k_pe.dtype)
