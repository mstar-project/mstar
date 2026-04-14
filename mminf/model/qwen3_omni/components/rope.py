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


# -----------------------------------------------------------------------
# Inverse frequencies
# -----------------------------------------------------------------------

def compute_rope_freqs(
    head_dim: int,
    rope_theta: float = 1_000_000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute inverse frequencies for standard RoPE.

    Returns
    -------
    inv_freq : torch.Tensor  shape ``(head_dim // 2,)``
        Inverse frequency vector  ``1 / (theta^(2i/d))``.
    """
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float()
            / head_dim
        )
    )
    return inv_freq


# -----------------------------------------------------------------------
# 3-D cos / sin from position IDs
# -----------------------------------------------------------------------

def compute_3d_cos_sin(
    position_ids_3d: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: list[int] | tuple[int, ...] = (24, 20, 20),
    attention_scaling: float = 1.0,
    target_dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin embeddings from 3D position IDs.

    This mirrors the ``Qwen3OmniMoeThinkerTextRotaryEmbedding.forward`` path:
    it computes raw per-component frequencies for all ``head_dim // 2`` dims,
    applies interleaved MRoPE mixing, then doubles up for the standard
    rotate-half convention.

    Parameters
    ----------
    position_ids_3d : torch.Tensor
        Shape ``(3, seq_len)`` -- temporal, height, width positions.
    inv_freq : torch.Tensor
        Shape ``(head_dim // 2,)`` -- inverse frequencies from
        :func:`compute_rope_freqs`.
    mrope_section : list[int]
        Three integers ``[s1, s2, s3]`` with ``s1+s2+s3 == head_dim // 2``.
        Default ``[24, 20, 20]`` for head_dim=128.
    attention_scaling : float
        Multiplicative scaling applied to cos/sin (defaults to 1.0).

    Returns
    -------
    cos : torch.Tensor  shape ``(seq_len, head_dim)``
    sin : torch.Tensor  shape ``(seq_len, head_dim)``
        Ready to be used with :func:`apply_interleaved_mrope`.
    """
    # position_ids_3d: (3, seq_len)  ->  (3, 1, seq_len)
    #   for matmul with inv_freq: (1, head_dim//2, 1)
    # result freqs: (3, 1, head_dim//2, seq_len) -> transpose -> (3, 1, seq_len, head_dim//2)
    pos = position_ids_3d[:, None, None, :].float()       # (3, 1, 1, seq_len)
    ifreq = inv_freq[None, None, :, None].float()          # (1, 1, head_dim//2, 1)

    # Broadcast: (3, 1, head_dim//2, seq_len)
    freqs = (ifreq * pos).transpose(2, 3)                  # (3, 1, seq_len, head_dim//2)

    # Apply interleaved mrope mixing -> (1, seq_len, head_dim//2)
    freqs = _apply_interleaved_mrope_freqs(freqs, mrope_section)

    # Double up for rotate-half: (1, seq_len, head_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scaling).squeeze(0)  # (seq_len, head_dim)
    sin = (emb.sin() * attention_scaling).squeeze(0)  # (seq_len, head_dim)

    # Cast back to the model dtype (e.g. bfloat16) so that the subsequent
    # ``q * cos`` / ``k * cos`` multiplications inside apply_interleaved_mrope
    # don't promote q/k to fp32 -- which would mismatch the paged KV cache
    # dtype that FlashInfer's attention call expects.  HF does the same
    # (modeling_qwen3_omni_moe.py:1329).
    if target_dtype is not None:
        cos = cos.to(target_dtype)
        sin = sin.to(target_dtype)

    return cos, sin


# -----------------------------------------------------------------------
# Interleaved MRoPE helpers
# -----------------------------------------------------------------------

def _apply_interleaved_mrope_freqs(
    freqs: torch.Tensor,
    mrope_section: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Mix the three frequency components into an interleaved layout.

    This is the core of the TM-RoPE interleaving.  Given ``freqs`` of shape
    ``(3, bs, seq_len, head_dim//2)`` where component 0 is temporal, 1 is
    height, and 2 is width, produce a single ``(bs, seq_len, head_dim//2)``
    tensor with the interleaved pattern.

    The HF implementation (``apply_interleaved_mrope``) does::

        freqs_t = freqs[0]   # start from temporal (covers all dims)
        for dim, offset in enumerate((1, 2), start=1):   # H=1, W=2
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]

    So the result starts as all-temporal, then selected interleaved slots
    are overwritten with H and W values.  For ``mrope_section = [24, 20, 20]``
    and ``head_dim // 2 = 64``:
      - Temporal occupies: dims 0,3,6,...,57 and 60,61,62,63  (24 dims)
      - Height occupies:   dims 1,4,7,...,58                   (20 dims)
      - Width occupies:    dims 2,5,8,...,59                   (20 dims)
    """
    # Start from temporal component for all dims
    freqs_t = freqs[0].clone()

    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]

    return freqs_t


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
    positions = torch.arange(seq_len, dtype=torch.float, device=device) + float(
        start_pos
    )
    return positions.unsqueeze(0).expand(3, -1).contiguous()


def get_rope_index_audio(
    num_audio_tokens: int,
    start_pos: float,
    device: torch.device | None = None,
    position_id_per_seconds: int = 25,
) -> torch.Tensor:
    """Build 3D MRoPE position IDs for an audio-only span.

    For audio the temporal component advances per audio frame and the
    height/width components are set to ``start_pos`` (i.e. the same
    base as the temporal component's first position), which matches the
    HF convention for single-modality audio where h/w track the text
    position.  The temporal component is time-based via the Qwen3-Omni
    ``position_id_per_seconds`` constant (default 25 positions/sec).

    Parameters
    ----------
    num_audio_tokens : int
        Number of audio tokens produced by the audio encoder.
    start_pos : float
        Starting position offset.
    device : torch.device, optional
        Device for the returned tensor.
    position_id_per_seconds : int
        Unused here -- kept for API symmetry with the HF implementation
        where audio timestamps map to integer position IDs.  The audio
        encoder already outputs one token per quantized frame, so the
        temporal component simply increments by one per token.

    Returns
    -------
    pos_ids_3d : torch.Tensor
        Shape ``(3, num_audio_tokens)``.  ``dtype=torch.float``.
    """
    del position_id_per_seconds  # kept for API compatibility
    temporal = torch.arange(
        num_audio_tokens, dtype=torch.float, device=device
    ) + float(start_pos)
    height = torch.full(
        (num_audio_tokens,), float(start_pos), dtype=torch.float, device=device
    )
    width = torch.full(
        (num_audio_tokens,), float(start_pos), dtype=torch.float, device=device
    )
    return torch.stack([temporal, height, width], dim=0)


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
            temporal = torch.full(
                (num_tokens,), float(start_pos), dtype=torch.float, device=device
            )
        else:
            temporal = (
                torch.arange(grid_t, dtype=torch.float, device=device) * seconds_per_grid * position_id_per_seconds
            ).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()

        h_index = (
            torch.arange(llm_grid_h, dtype=torch.float, device=device)
            .view(1, -1, 1)
            .expand(grid_t, -1, llm_grid_w)
            .flatten()
            + float(start_pos)
        )
        w_index = (
            torch.arange(llm_grid_w, dtype=torch.float, device=device)
            .view(1, 1, -1)
            .expand(grid_t, llm_grid_h, -1)
            .flatten()
            + float(start_pos)
        )

        pos_ids_list.append(torch.stack([temporal, h_index, w_index], dim=0))

    return torch.cat(pos_ids_list, dim=1)
