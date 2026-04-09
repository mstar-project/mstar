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
# Position-ID construction  (get_rope_index)
# -----------------------------------------------------------------------

def get_rope_index(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    audio_seqlens: torch.LongTensor | None = None,
    second_per_grids: torch.Tensor | None = None,
    *,
    # Token IDs -- passed from config
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    audio_token_id: int = 151646,
    vision_start_token_id: int = 151652,
    audio_start_token_id: int = 151647,
    position_id_per_seconds: int = 25,
    spatial_merge_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D MRoPE position IDs from input tokens and grid info.

    This is a **simplified** version of the full HF ``get_rope_index``.
    It currently supports:
      - Pure text inputs (all 3 components = sequential position)
      - Audio-only inputs (temporal = absolute time position, h/w = 0)

    Full image/video support (spatial grid position computation) is marked
    with TODOs below and will be implemented in a later phase.

    Parameters
    ----------
    input_ids : torch.LongTensor
        Shape ``(batch_size, seq_len)``.
    attention_mask : torch.Tensor, optional
        Shape ``(batch_size, seq_len)``.  ``1`` = real token, ``0`` = pad.
    image_grid_thw : torch.LongTensor, optional
        Shape ``(num_images, 3)`` -- temporal, height, width grid sizes.
    video_grid_thw : torch.LongTensor, optional
        Shape ``(num_videos, 3)`` -- temporal, height, width grid sizes.
    audio_seqlens : torch.LongTensor, optional
        Shape ``(num_audios,)`` -- raw audio lengths (before feature extraction).
    second_per_grids : torch.Tensor, optional
        Shape ``(num_videos,)`` -- time interval per temporal grid for video.

    Returns
    -------
    position_ids : torch.Tensor
        Shape ``(3, batch_size, seq_len)`` -- the 3D position IDs.
    mrope_position_deltas : torch.Tensor
        Shape ``(batch_size, 1)`` -- offset between max position and seq length
        (used for incremental decoding).
    """
    batch_size, seq_len = input_ids.shape

    has_vision = image_grid_thw is not None or video_grid_thw is not None

    if has_vision:
        # ---------------------------------------------------------------
        # Full multimodal path (vision + audio + text interleaved)
        # ---------------------------------------------------------------
        return _get_rope_index_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            audio_seqlens=audio_seqlens,
            second_per_grids=second_per_grids,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            audio_token_id=audio_token_id,
            vision_start_token_id=vision_start_token_id,
            audio_start_token_id=audio_start_token_id,
            position_id_per_seconds=position_id_per_seconds,
            spatial_merge_size=spatial_merge_size,
        )
    else:
        # ---------------------------------------------------------------
        # Text-only (possibly with audio but no images/videos)
        # ---------------------------------------------------------------
        # All 3 position components are identical sequential positions.
        # This matches the HF else-branch.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # Expand to 3 components: (3, batch, seq_len)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
            input_ids.device
        )

        max_position_ids = position_ids.max(0, keepdim=False)[0].max(
            -1, keepdim=True
        )[0]
        mrope_position_deltas = (
            max_position_ids + 1 - attention_mask.sum(dim=-1, keepdim=True)
        )

        return position_ids, mrope_position_deltas


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute audio feature extractor output lengths.

    Mirrors the HF helper that accounts for conv down-sampling and
    chunk-based processing with 100-sample chunks.
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def _get_llm_pos_ids_for_vision(
    start_idx: float,
    t_index: torch.Tensor,
    grid_h: int,
    grid_w: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build 3D position IDs for a single image or video grid.

    Parameters
    ----------
    start_idx : float
        Starting position offset.
    t_index : torch.Tensor
        Temporal indices for each frame, shape ``(num_temporal_patches,)``.
    grid_h, grid_w : int
        Full grid height/width from ``grid_thw`` (before spatial merge).
    spatial_merge_size : int
        Spatial merge factor (tokens per merged patch).

    Returns
    -------
    pos_ids : torch.Tensor  shape ``(3, num_vision_tokens)``
    """
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size
    num_t = len(t_index)

    h_index = (
        torch.arange(llm_grid_h)
        .view(1, -1, 1)
        .expand(num_t, -1, llm_grid_w)
        .flatten()
        .float()
    )
    w_index = (
        torch.arange(llm_grid_w)
        .view(1, 1, -1)
        .expand(num_t, llm_grid_h, -1)
        .flatten()
        .float()
    )
    t_expanded = (
        t_index.view(-1, 1)
        .expand(-1, llm_grid_h * llm_grid_w)
        .flatten()
        .float()
    )

    pos_ids = torch.stack([t_expanded, h_index, w_index])  # (3, N)
    pos_ids = pos_ids + start_idx
    return pos_ids


def _get_rope_index_multimodal(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None,
    image_grid_thw: torch.LongTensor | None,
    video_grid_thw: torch.LongTensor | None,
    audio_seqlens: torch.LongTensor | None,
    second_per_grids: torch.Tensor | None,
    *,
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,
    audio_start_token_id: int,
    position_id_per_seconds: int,
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full multimodal get_rope_index mirroring the HF implementation.

    Handles interleaved text, image, video, and audio tokens in a single
    sequence.  Walks through the token stream, identifies each modality
    span using sentinel tokens, and constructs appropriate 3D position IDs.
    """
    total_input_ids = input_ids
    if attention_mask is not None:
        attention_mask_bool = attention_mask == 1
    else:
        attention_mask_bool = torch.ones_like(input_ids, dtype=torch.bool)

    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=torch.float,
        device=input_ids.device,
    )

    mrope_position_deltas = []
    image_idx, video_idx, audio_idx = 0, 0, 0

    for i, input_ids_row in enumerate(total_input_ids):
        if attention_mask is not None:
            ids = input_ids_row[attention_mask_bool[i]]
        else:
            ids = input_ids_row

        image_nums, video_nums, audio_nums = 0, 0, 0
        vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
        if len(vision_start_indices) > 0:
            vision_tokens = ids[vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum().item())
            video_nums = int((vision_tokens == video_token_id).sum().item())
        audio_nums = int(torch.sum(ids == audio_start_token_id).item())

        input_tokens = ids.tolist()
        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
        multimodal_nums = image_nums + video_nums + audio_nums

        for _ in range(multimodal_nums):
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if len(llm_pos_ids_list) > 0
                else 0
            )

            # Find next vision or audio start
            if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                remain_videos > 0 or remain_images > 0
            ):
                ed_vision_start = input_tokens.index(vision_start_token_id, st)
            else:
                ed_vision_start = len(input_tokens) + 1

            if audio_token_id in input_tokens and remain_audios > 0:
                ed_audio_start = input_tokens.index(audio_start_token_id, st)
            else:
                ed_audio_start = len(input_tokens) + 1

            min_ed = min(ed_vision_start, ed_audio_start)

            # Leading text before this multimodal span
            text_len = min_ed - st
            if text_len != 0:
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )
                st_idx += text_len

            # BOS token(s) -- audio-in-video has 2
            bos_len, eos_len = 1, 1
            llm_pos_ids_list.append(
                torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
            )
            st_idx += bos_len

            if min_ed == ed_audio_start:
                # ----- Audio only -----
                audio_len = _get_feat_extract_output_lengths(
                    audio_seqlens[audio_idx]
                )
                llm_pos_ids = (
                    torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                )
                llm_pos_ids_list.append(llm_pos_ids)
                st += int(text_len + bos_len + audio_len + eos_len)
                audio_idx += 1
                remain_audios -= 1

            elif min_ed == ed_vision_start and ids[ed_vision_start + 1] == image_token_id:
                # ----- Image -----
                grid_t = int(image_grid_thw[image_idx][0].item())
                grid_h = int(image_grid_thw[image_idx][1].item())
                grid_w = int(image_grid_thw[image_idx][2].item())
                t_index = (
                    torch.arange(grid_t).float() * 1 * position_id_per_seconds
                )
                llm_pos_ids = _get_llm_pos_ids_for_vision(
                    st_idx, t_index, grid_h, grid_w, spatial_merge_size
                )
                image_len = (
                    image_grid_thw[image_idx].prod() // (spatial_merge_size ** 2)
                )
                llm_pos_ids_list.append(llm_pos_ids)
                st += int(text_len + bos_len + image_len + eos_len)
                image_idx += 1
                remain_images -= 1

            elif min_ed == ed_vision_start and ids[ed_vision_start + 1] == video_token_id:
                # ----- Video -----
                grid_t = int(video_grid_thw[video_idx][0].item())
                grid_h = int(video_grid_thw[video_idx][1].item())
                grid_w = int(video_grid_thw[video_idx][2].item())
                t_index = (
                    torch.arange(grid_t).float()
                    * second_per_grids[video_idx].cpu().float()
                    * position_id_per_seconds
                )
                llm_pos_ids = _get_llm_pos_ids_for_vision(
                    st_idx, t_index, grid_h, grid_w, spatial_merge_size
                )
                video_len = (
                    video_grid_thw[video_idx].prod() // (spatial_merge_size ** 2)
                )
                llm_pos_ids_list.append(llm_pos_ids)
                st += int(text_len + bos_len + video_len + eos_len)
                video_idx += 1
                remain_videos -= 1

            # EOS token(s)
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if len(llm_pos_ids_list) > 0
                else 0
            )
            llm_pos_ids_list.append(
                torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
            )

        # Trailing text
        if st < len(input_tokens):
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if len(llm_pos_ids_list) > 0
                else 0
            )
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(
            [item.float() for item in llm_pos_ids_list], dim=1
        ).reshape(3, -1)

        position_ids[..., i, attention_mask_bool[i]] = llm_positions.to(
            position_ids.device
        )
        mrope_position_deltas.append(llm_positions.max() + 1 - len(ids))

    mrope_position_deltas = torch.tensor(
        mrope_position_deltas, device=input_ids.device
    ).unsqueeze(1)

    return position_ids, mrope_position_deltas
