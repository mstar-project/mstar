"""Ling-2.0 partial 3D rotary embeddings (``video_rope`` flavor).

Ling-2.0's attention uses **partial rotary** (only the first
``head_dim * partial_rotary_factor`` dims of each head are rotated; the rest
pass through unchanged) with **3D MRoPE positions** (time / height / width
each get their own position id) in the ``video_rope`` cos/sin layout.

The cos/sin layout is the unusual bit. Standard MRoPE places contiguous
frequency sections per axis:

    [ T T ... T  H H ... H  W W ... W ]   (sizes mrope_section = [Nt, Nh, Nw])

Ling's ``video_rope`` interleaves H and W element-wise in the spatial
section and puts T at the end:

    [ H W H W ... H W   T T ... T ]       (sizes hw_size = Nh + Nw,  Nt at tail)

For pure-text positions (1D position_ids, no T/H/W split) the rotation
degenerates to the standard 1D rotary on the first ``rotary_dim`` dims.

References
----------
* Ming upstream ``apply_3d_rotary_pos_emb``
  ``/tmp/ming_repo/modeling_bailing_moe_v2.py:226-313`` (video_rope branch
  is the ``elif rope_type == "video_rope"`` block).
* vllm-omni ``MingVideoRopeMRotaryEmbedding._remap_video_rope``
  ``/tmp/vllm-omni/vllm_omni/model_executor/models/ming_flash_omni/modeling_bailing_moe_v2.py:79-110``
  — same remap as ours; we port the math without depending on vllm.
"""

from __future__ import annotations

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard neox-style rotary half-rotation: ``[-x2, x1]``."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _build_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    """Standard rotary inverse-frequency table: ``theta ** (-2i / rotary_dim)`` for i in [0, rotary_dim/2)."""
    return 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )


class LingPartialMRotaryEmbedding(nn.Module):
    """Partial rotary + ``video_rope`` 3D MRoPE.

    Args:
        head_dim: full head dim of the attention layer.
        partial_rotary_factor: fraction of head_dim that's actually rotated
            (the rest is concatenated pass-through). The model uses 0.5;
            head_dim=128 → rotary_dim=64.
        mrope_section: per-axis cos/sin section sizes. Released ckpt:
            ``[8, 12, 12]``. The first is Nt (time), the rest are Nh
            (height) and Nw (width); Nh+Nw must equal rotary_dim/2 − Nt
            (i.e. the section sums to rotary_dim/2 — see config invariant).
        rope_theta: rotary base frequency. Released ckpt: ``2_400_000``.
        max_position_embeddings: max sequence length; precomputed cache size.

    The forward expects ``position_ids`` of shape ``(3, num_tokens)`` for
    3D positions or ``(num_tokens,)`` for plain 1D rope (degenerates to
    standard rotary).
    """

    def __init__(
        self,
        head_dim: int,
        partial_rotary_factor: float,
        mrope_section: list[int],
        rope_theta: float,
        max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        if self.rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim must be even (got {self.rotary_dim}); check "
                f"partial_rotary_factor."
            )
        self.mrope_section = list(mrope_section)
        if sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(self.mrope_section)} must equal "
                f"rotary_dim//2={self.rotary_dim // 2}"
            )
        if len(self.mrope_section) != 3:
            raise ValueError(
                f"mrope_section must be length-3 [Nt, Nh, Nw]; got {self.mrope_section}"
            )
        self.hw_size = self.mrope_section[1] + self.mrope_section[2]

        self.rope_theta = float(rope_theta)
        self.max_position_embeddings = int(max_position_embeddings)

        # Cache inv_freq once; cos/sin tables are computed on first forward
        # (lazy so we don't pay for max_position_embeddings * rotary_dim
        # storage on CPU for tests).
        self.register_buffer(
            "inv_freq",
            _build_inv_freq(self.rotary_dim, self.rope_theta),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # cos / sin cache
    # ------------------------------------------------------------------

    def _compute_cos_sin(
        self, position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for ``position_ids``.

        ``position_ids`` is ``(num_tokens,)`` or ``(3, num_tokens)``.
        Returns ``cos, sin`` of shape ``(num_tokens, rotary_dim)`` in the
        video_rope layout (H/W interleaved spatial + T tail).
        """
        if position_ids.dim() == 1:
            return self._cos_sin_1d(position_ids)
        if position_ids.dim() != 2 or position_ids.shape[0] != 3:
            raise ValueError(
                f"position_ids must be (num_tokens,) or (3, num_tokens); "
                f"got shape {tuple(position_ids.shape)}"
            )
        return self._cos_sin_3d_video_rope(position_ids)

    def _cos_sin_1d(
        self, position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard 1D rotary cos/sin — used for pure-text positions."""
        # (num_tokens, rotary_dim/2)
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        # (num_tokens, rotary_dim) — neox style: cat freqs with themselves
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _cos_sin_3d_video_rope(
        self, position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """3D positions → video_rope layout.

        position_ids: ``(3, num_tokens)`` — row 0 = time, row 1 = height,
        row 2 = width.

        Steps:
          1. Compute per-axis freqs: ``(3, num_tokens, rotary_dim/2)``.
          2. Form (cos, sin) of shape ``(3, num_tokens, rotary_dim)`` neox-style.
          3. Remap each rotary_dim/2 frequency-pair index ``i`` into:
                - i < hw_size  →  H if i even, W if i odd
                - i ≥ hw_size  →  T
             Pairs ``(cos[i], cos[i + rotary_dim/2])`` correspond to the
             same frequency, so the same row assignment applies to both
             halves.
        """
        # (3, num_tokens, rotary_dim/2)
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        # (3, num_tokens, rotary_dim) — neox cat
        cos_3d = torch.cat((freqs, freqs), dim=-1).cos()
        sin_3d = torch.cat((freqs, freqs), dim=-1).sin()
        return self._remap_video_rope(cos_3d, sin_3d)

    def _remap_video_rope(
        self, cos_3d: torch.Tensor, sin_3d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remap per-axis cos/sin into the video_rope 2D layout.

        cos_3d, sin_3d: ``(3, num_tokens, rotary_dim)``.
        Returns: ``(num_tokens, rotary_dim)``.

        Mirror of vllm-omni's ``_remap_video_rope`` with one difference:
        we operate on the *full* rotary_dim tables (not the half-tables
        chunked from the cos_sin cache), because we never built a cache —
        we computed freqs in 1:1 correspondence with positions in the
        forward path. The H/W alternation rule still picks the correct
        index because each half of the neox-cat repeats the same
        frequency.
        """
        # Both halves of the rotary_dim (the first and second halves
        # contain the same frequencies after the neox cat) get the same
        # axis-assignment. So a single index i in [0, rotary_dim/2) picks
        # a frequency-pair that should come from one axis.
        half = self.rotary_dim // 2

        result_cos = torch.empty_like(cos_3d[0])
        result_sin = torch.empty_like(sin_3d[0])

        # Spatial half: H on even indices, W on odd indices, capped at hw_size.
        # Then mirror to the second half (which holds the same freqs).
        for offset in (0, half):
            # H rows go on even positions [0, 2, 4, ...] up to hw_size
            result_cos[:, offset : offset + self.hw_size : 2] = cos_3d[
                1, :, offset : offset + self.hw_size : 2
            ]
            result_cos[:, offset + 1 : offset + self.hw_size : 2] = cos_3d[
                2, :, offset + 1 : offset + self.hw_size : 2
            ]
            result_sin[:, offset : offset + self.hw_size : 2] = sin_3d[
                1, :, offset : offset + self.hw_size : 2
            ]
            result_sin[:, offset + 1 : offset + self.hw_size : 2] = sin_3d[
                2, :, offset + 1 : offset + self.hw_size : 2
            ]
            # Temporal tail
            result_cos[:, offset + self.hw_size : offset + half] = cos_3d[
                0, :, offset + self.hw_size : offset + half
            ]
            result_sin[:, offset + self.hw_size : offset + half] = sin_3d[
                0, :, offset + self.hw_size : offset + half
            ]
        return result_cos, result_sin

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate the first ``rotary_dim`` dims of q and k in-place.

        Args:
            q, k: ``(..., num_tokens, head_dim)`` (typical layout from
                ParallelAttention is ``(num_tokens, num_heads, head_dim)``).
                Only the last dim and the per-token axis matter.
            position_ids: ``(num_tokens,)`` for 1D rope or
                ``(3, num_tokens)`` for video_rope.

        Returns:
            ``(q, k)`` with rotation applied to the rotary half.
        """
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(
                f"q/k last dim {q.shape[-1]}/{k.shape[-1]} != "
                f"head_dim {self.head_dim}"
            )

        cos, sin = self._compute_cos_sin(position_ids)
        # Broadcast cos/sin across the leading axes of q (typically a
        # heads axis comes BEFORE the token axis: q is (..., heads, T,
        # head_dim)). cos starts as (T, rotary_dim); we need to insert
        # ones at every leading dim of q so the broadcast aligns
        # (T at the second-to-last position, rotary_dim at the last).
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
        cos_q = cos.to(q.dtype)
        sin_q = sin.to(q.dtype)
        cos_k = cos.to(k.dtype)
        sin_k = sin.to(k.dtype)

        q_rot = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        return (
            torch.cat([q_rot, q_pass], dim=-1),
            torch.cat([k_rot, k_pass], dim=-1),
        )
