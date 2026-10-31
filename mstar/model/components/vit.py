"""Pieces a ViT tower over packed patches shares (Qwen3.5, Muse Glimmer).

- `bilinear_table_taps`: the 2x2 neighbourhood that resamples a square learned
  position table onto an image's patch grid, as a gather plus weights.
- `AxialVisionRope`: 2D rope, one set of frequencies per axis.
- `RaggedVisionAttention`: attention over the packed patches through the
  ragged attention resource, its rope in fp32; a tower subclasses it with its
  q/k/v projections.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.engine.resources.convenience import RaggedAttentionCallable
from mstar.model.keys import VISION_ATTN


def _axis_taps(src: torch.Tensor, side: int, zero_pad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=src.device)
    taps = floor.long()[:, None] + offsets
    weights = (1 - (src[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)
    if zero_pad:
        weights = weights * ((taps >= 0) & (taps <= side - 1))
    return taps.clamp(0, side - 1), weights


def bilinear_table_taps(
    row_src: torch.Tensor, col_src: torch.Tensor, side: int, zero_pad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[P, 4]`` indices into a ``side x side`` table and their weights,
    bilinear around each patch's fp32 source coordinates in the table.

    The caller maps patch positions to source coordinates, which is where
    ``align_corners`` lives. ``zero_pad``: a tap off the table weighs zero
    (``grid_sample``'s ``padding_mode="zeros"``); otherwise it reads the edge.
    """
    h_taps, h_w = _axis_taps(row_src, side, zero_pad)
    w_taps, w_w = _axis_taps(col_src, side, zero_pad)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return indices, weights


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class AxialVisionRope(nn.Module):
    """Axial 2D rope: one set of frequencies per axis, the two concatenated
    and repeated to cover the whole head, so no partial rotation.

    ``keep_fp32``: the frequencies stay fp32 through a cast of the module
    (the engine's ``.to(bfloat16)`` would otherwise round them).
    """

    def __init__(self, head_dim: int, theta: float, keep_fp32: bool = False):
        super().__init__()
        self._spatial_dim = head_dim // 2
        self._theta = theta
        self._keep_fp32 = keep_fp32
        self.register_buffer("inv_freq", self._inv_freq(None), persistent=False)

    def _inv_freq(self, device) -> torch.Tensor:
        d = self._spatial_dim
        return 1.0 / (self._theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self._keep_fp32:
            self.inv_freq = self._inv_freq(self.inv_freq.device)
        return result

    @torch.no_grad()
    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[P, 2]`` positions in, ``[P, head_dim]`` cos and sin out, computed
        in fp32 and rounded to ``dtype``."""
        freqs = positions[..., None].float() * self.inv_freq.float()  # [P, 2, d/4]
        cos, sin = self._recompose(freqs.cos()), self._recompose(freqs.sin())
        return cos.to(dtype), sin.to(dtype)

    @staticmethod
    def _recompose(freq: torch.Tensor) -> torch.Tensor:
        hw = torch.cat([freq[:, 0], freq[:, 1]], dim=-1)
        return torch.cat([hw, hw], dim=-1)


class RaggedVisionAttention(nn.Module):
    """Non-causal attention over packed patches, through the plan the ragged
    attention resource holds for ``label`` (frames, windows, ...), so the
    forward never sees a boundary. Rope runs in fp32, then back.

    A subclass provides ``project_qkv(x) -> (q, k, v)``, each
    ``[patches, heads, head_dim]``.
    """

    def __init__(self, hidden_size: int, num_heads: int, label: str | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.label = label
        self.attend: RaggedAttentionCallable | None = None

    def bind_resources(self, resources: dict) -> None:
        """``.get``: a deployment that never builds the vision node binds
        nothing here."""
        attn = resources.get(VISION_ATTN)
        self.attend = None if attn is None else RaggedAttentionCallable(attn, self.label)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.attend is None:
            raise ValueError(
                f"{type(self).__module__}.{type(self).__qualname__} has no ragged attention resource; "
                f"declare a RaggedAttentionSpec under {VISION_ATTN!r} for the "
                "vision_encoder node"
            )
        q, k, v = self.project_qkv(x)
        dtype = q.dtype
        c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        qf, kf = q.float(), k.float()
        q = ((qf * c) + (_rotate_half(qf) * s)).to(dtype)
        k = ((kf * c) + (_rotate_half(kf) * s)).to(dtype)
        out = self.attend(q, k, v)
        return self.proj(out.reshape(x.shape[0], -1).contiguous())
