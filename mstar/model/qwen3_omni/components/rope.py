"""3D Multimodal RoPE (TM-RoPE) for Qwen3-Omni Thinker.

Qwen3-Omni uses an INTERLEAVED 3D MRoPE layout where the three positional
components (temporal, height, width) are woven into the rotary embedding
dimensions in a [T,H,W,T,H,W,...,T,T] pattern rather than the chunked
[TTT...HHH...WWW] layout used by some earlier models.

Key reference
-------------
``Qwen3OmniMoeThinkerTextRotaryEmbedding`` and ``apply_interleaved_mrope``
from the HuggingFace ``modeling_qwen3_omni_moe.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch

from mstar.model.components.mrope import compute_3d_cos_sin, compute_rope_freqs

__all__ = ["apply_interleaved_mrope", "compute_3d_cos_sin", "compute_rope_freqs"]


def apply_interleaved_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved multimodal RoPE to query and key tensors.

    This applies the standard rotate-half RoPE using cos/sin that have
    *already* been interleaved via :func:`compute_3d_cos_sin`.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor.  Typical shapes:
        - ``(batch, heads, seq_len, head_dim)`` (``unsqueeze_dim=1`` is a no-op
          when cos already has a heads broadcast dim, but the unsqueeze makes
          ``(seq_len, head_dim)`` -> ``(1, seq_len, head_dim)`` broadcastable).
        - ``(tokens, heads, head_dim)`` for disaggregated / packed inputs.
    k : torch.Tensor
        Key tensor, same layout as ``q`` but may have fewer heads (GQA).
    cos : torch.Tensor
        Cosine embeddings from :func:`compute_3d_cos_sin`.
    sin : torch.Tensor
        Sine embeddings from :func:`compute_3d_cos_sin`.
    unsqueeze_dim : int
        Dimension along which to unsqueeze cos/sin so they broadcast with
        q/k.  Default 1 matches the HF convention for
        ``(batch, heads, seq_len, head_dim)`` layout.

    Returns
    -------
    q_embed, k_embed : torch.Tensor
        Rotated query and key tensors, same shape and dtype as inputs.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims -- standard RoPE helper."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# -----------------------------------------------------------------------
# Position-ID construction  (per-modality helpers)
# -----------------------------------------------------------------------
#
# In the disaggregated pipeline each prefill graph walk (prefill_text,
# prefill_audio, prefill_vision) is single-modality, so we do not need
# the full multimodal parser used by HF ``get_rope_index``.  Instead we
# provide three small helpers that each return a ``(3, seq_len)`` tensor
# of 3D position IDs (temporal, height, width).
#
# The callers (``ThinkerSubmodule._preprocess_prefill_*``) track a
# per-request ``start_pos`` offset across walks so the position IDs
# remain monotonic along the full sequence.


def get_rope_index_text(
    seq_len: int,
    start_pos: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build 3D MRoPE position IDs for a pure-text span.

    All three components (temporal, height, width) are identical
    sequential positions ``[start_pos, start_pos + 1, ..., start_pos + seq_len - 1]``.

    Parameters
    ----------
    seq_len : int
        Number of text tokens.
    start_pos : float
        Starting position offset (absolute position of the first token).
    device : torch.device, optional
        Device for the returned tensor.

    Returns
    -------
    pos_ids_3d : torch.Tensor
        Shape ``(3, seq_len)``.  ``dtype=torch.float``.
    """
    positions = torch.arange(seq_len, dtype=torch.float, device=device) + float(start_pos)
    return positions.unsqueeze(0).expand(3, -1).contiguous()


def get_rope_index_audio(
    num_audio_tokens: int,
    start_pos: float,
    device: torch.device | None = None,
    position_id_per_seconds: int = 25,
) -> torch.Tensor:
    """Build 3D MRoPE position IDs for an audio-only span.

    All three components advance together, one per audio token — audio
    tokens are positioned exactly like text.  Without vision inputs, HF's
    ``get_rope_index`` takes its non-spatial branch and returns
    ``(cumsum(attention_mask) - 1).expand(3, -1, -1)``; checked against the
    real implementation, an audio span comes back as t == h == w ==
    2, 3, 4, ….

    Pinning h/w to a constant instead (what this did before) leaves the
    ``mrope_section`` bands that read those components with no positional
    progression across the span, which degrades long audio the most.

    Parameters
    ----------
    num_audio_tokens : int
        Number of audio tokens produced by the audio encoder.
    start_pos : float
        Starting position offset.
    device : torch.device, optional
        Device for the returned tensor.
    position_id_per_seconds : int
        Unused here -- kept for API symmetry.  It applies to HF's spatial
        branch (vision present), where audio position IDs come from
        timestamps rather than token index.

    Returns
    -------
    pos_ids_3d : torch.Tensor
        Shape ``(3, num_audio_tokens)``.  ``dtype=torch.float``.
    """
    del position_id_per_seconds  # kept for API compatibility
    positions = torch.arange(num_audio_tokens, dtype=torch.float, device=device) + float(start_pos)
    return positions.unsqueeze(0).expand(3, -1).contiguous()


def get_rope_index_vision(
    grid_thw: torch.LongTensor,
    start_pos: float,
    position_id_per_seconds: float,
    device: torch.device | None = None,
    spatial_merge_size: int = 2,
    seconds_per_grid: float | None = None,
) -> torch.Tensor:
    """Build 3D MRoPE position IDs for a vision-only span.

    Temporal component is set to the constant ``start_pos`` (single
    image / frame) while the height and width components come from the
    spatial grid after the spatial merge.  For a grid of shape
    ``(T, H, W)`` the resulting sequence length is
    ``T * (H // spatial_merge_size) * (W // spatial_merge_size)`` per
    image, concatenated across images.

    Parameters
    ----------
    grid_thw : torch.LongTensor
        Shape ``(num_images, 3)`` -- temporal, height, width grid sizes.
    start_pos : float
        Starting position offset; applied to all three components.
    device : torch.device, optional
        Device for the returned tensor.
    spatial_merge_size : int
        Spatial merge factor (tokens per merged patch).

    Returns
    -------
    pos_ids_3d : torch.Tensor
        Shape ``(3, total_vision_tokens)``.  ``dtype=torch.float``.
    """
    if grid_thw.dim() == 1:
        grid_thw = grid_thw.unsqueeze(0)

    pos_ids_list: list[torch.Tensor] = []
    for img_idx in range(grid_thw.shape[0]):
        grid_t = int(grid_thw[img_idx, 0].item())
        grid_h = int(grid_thw[img_idx, 1].item())
        grid_w = int(grid_thw[img_idx, 2].item())

        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        num_tokens = grid_t * llm_grid_h * llm_grid_w

        # Temporal is constant per image (= start_pos).  In the full HF
        # multimodal parser the temporal component tracks video time via
        # ``position_id_per_seconds``; for still images (grid_t == 1)
        # that collapses to a single value per image.
        if seconds_per_grid is None:
            temporal = torch.full((num_tokens,), float(start_pos), dtype=torch.float, device=device)
        else:
            temporal = (
                (torch.arange(grid_t, dtype=torch.float, device=device) * seconds_per_grid * position_id_per_seconds)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
                .float()
            )

        h_index = torch.arange(llm_grid_h, dtype=torch.float, device=device).view(1, -1, 1).expand(
            grid_t, -1, llm_grid_w
        ).flatten() + float(start_pos)
        w_index = torch.arange(llm_grid_w, dtype=torch.float, device=device).view(1, 1, -1).expand(
            grid_t, llm_grid_h, -1
        ).flatten() + float(start_pos)

        pos_ids_list.append(torch.stack([temporal, h_index, w_index], dim=0))

    return torch.cat(pos_ids_list, dim=1)
