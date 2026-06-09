"""3D MRoPE position-id helpers for Ming-flash-omni-2.0.

Ming-flash-omni-2.0 uses partial 3D MRoPE
(`mrope_section=[8, 12, 12]`, `partial_rotary_factor=0.5`) in the
``video_rope`` layout. The cos/sin remap lives in
:class:`mminf.model.ming_omni_flash.components.rope.LingPartialMRotaryEmbedding`;
this module produces the *position-id* tensors that feed into it.

Three helpers cover the modality-specific position layouts used by the
Thinker prefill walks:

  * :func:`get_rope_index_text`   — pure-text span (sentinels included).
  * :func:`get_rope_index_audio`  — audio embeddings (treated as text
    positions per ``modeling_bailing_moe_v2.get_rope_index``, which
    only special-cases ``image_*`` / ``video_*`` tokens).
  * :func:`get_rope_index_vision` — image (or video) embeddings with
    grid-aware T/H/W position ids per
    ``modeling_bailing_moe_v2.get_rope_index:592-647``.

All three return ``(3, seq_len)`` tensors with rows ``[t, h, w]``;
the rope module's ``video_rope`` remap will pick out H/W on even/odd
spatial slots and T on the temporal tail (see
``LingPartialMRotaryEmbedding._cos_sin_3d_video_rope`` for the layout).
"""

from __future__ import annotations

import torch


def get_rope_index_text(
    seq_len: int,
    start_pos: int | float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """3D MRoPE positions for a pure-text span.

    All three (T, H, W) components share the same sequential positions
    ``[start_pos, start_pos+1, ..., start_pos+seq_len-1]``. This matches
    the pure-text branch of ``modeling_bailing_moe_v2.get_rope_index``
    (`./modeling_bailing_moe_v2.py:658-675`).

    Args:
        seq_len: number of tokens in this span.
        start_pos: position offset for the first token.
        device:  target device.
        dtype:   integer dtype for the position ids (rope module
                 casts to float internally; long matches the upstream).

    Returns:
        ``(3, seq_len)`` tensor.
    """
    positions = torch.arange(seq_len, dtype=dtype, device=device) + int(start_pos)
    return positions.unsqueeze(0).expand(3, -1).contiguous()


def get_rope_index_audio(
    num_audio_tokens: int,
    start_pos: int | float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """3D MRoPE positions for an audio span.

    Ming's `get_rope_index` does NOT special-case audio: audio tokens
    advance through the same per-token position counter as text. Each
    audio token contributes one position; T/H/W all match. Audio
    semantics live in the audio encoder + projector (which already
    down-sample to one embedding per LLM-time-step).

    Args:
        num_audio_tokens: number of audio embeddings (after the
            projector's conv1d down-sample).
        start_pos: position offset for the first audio embedding.
        device:  target device.
        dtype:   integer dtype for position ids.

    Returns:
        ``(3, num_audio_tokens)`` tensor, identical rows.
    """
    return get_rope_index_text(num_audio_tokens, start_pos, device=device, dtype=dtype)


def get_rope_index_vision(
    grid_thw: torch.Tensor,
    start_pos: int | float,
    spatial_merge_size: int,
    device: torch.device | str | None = None,
    second_per_grid_t: float | None = None,
    tokens_per_second: int = 2,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """3D MRoPE positions for a vision span (single image or video).

    Mirrors `modeling_bailing_moe_v2.get_rope_index:625-647` for one
    image:

    * Temporal:    ``arange(grid_t)`` expanded across ``H*W``, optionally
                   scaled by ``second_per_grid_t * tokens_per_second``
                   for absolute video timestamps.
    * Height:      ``arange(llm_grid_h)`` expanded across ``T * W``.
    * Width:       ``arange(llm_grid_w)`` expanded across ``T * H``.

    ``llm_grid_h = grid_h // spatial_merge_size`` (same for W). All
    three components are offset by ``start_pos`` so the span fits into
    the global position-id counter the caller is tracking.

    Multi-image / video frames concatenate across images by calling
    this helper per image and stitching the results — see
    :func:`stitch_vision_positions` (or the dispatch in
    `BailingMoeV2ThinkerSubmodule.prepare_inputs`).

    Args:
        grid_thw: ``(3,)`` long tensor of (T, H, W) grid sizes.
        start_pos: position offset for this image's first token.
        spatial_merge_size: from `VisionEncoderConfig.spatial_merge_size`
            (= 2 on the released ckpt).
        device:  target device.
        second_per_grid_t: when set, multiply the temporal component by
            ``second_per_grid_t * tokens_per_second`` for absolute video
            timestamps. None ⇒ raw frame index. Image inputs always pass
            None; video inputs pass the per-clip frame interval.
        tokens_per_second: temporal-resolution multiplier
            (= 2 on the released ckpt; mirrors ``config.tokens_per_second``).
        dtype: integer dtype for position ids.

    Returns:
        ``(3, grid_t * (H/m) * (W/m))`` tensor of T/H/W positions
        offset by ``start_pos``.
    """
    if grid_thw.dim() != 1 or grid_thw.numel() != 3:
        raise ValueError(
            f"grid_thw must be a 1-D tensor of length 3 (T, H, W); "
            f"got shape {tuple(grid_thw.shape)}"
        )
    grid_t = int(grid_thw[0].item())
    grid_h = int(grid_thw[1].item())
    grid_w = int(grid_thw[2].item())
    if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
        raise ValueError(
            f"grid_h={grid_h} / grid_w={grid_w} not divisible by "
            f"spatial_merge_size={spatial_merge_size}."
        )
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size

    # Temporal: arange(grid_t), expanded across H*W, optionally absolute time.
    range_t = torch.arange(grid_t, dtype=dtype, device=device).view(-1, 1)
    expanded_t = range_t.expand(-1, llm_grid_h * llm_grid_w)
    if second_per_grid_t is not None:
        # Float math then back to int (matches modeling_bailing_moe_v2 path).
        t_index = (
            expanded_t.float() * float(second_per_grid_t) * float(tokens_per_second)
        ).to(dtype).flatten()
    else:
        t_index = expanded_t.flatten()

    h_index = (
        torch.arange(llm_grid_h, dtype=dtype, device=device)
        .view(1, -1, 1)
        .expand(grid_t, -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(llm_grid_w, dtype=dtype, device=device)
        .view(1, 1, -1)
        .expand(grid_t, llm_grid_h, -1)
        .flatten()
    )
    return torch.stack([t_index, h_index, w_index], dim=0) + int(start_pos)


def vision_span_max_position(
    grid_thw: torch.Tensor,
    start_pos: int | float,
    spatial_merge_size: int,
    second_per_grid_t: float | None = None,
    tokens_per_second: int = 2,
) -> int:
    """Compute one past the largest position id this vision span produces.

    Useful for advancing the global ``start_pos`` counter past a vision
    span when the next walk needs to know where text positions resume
    (mirrors ``modeling_bailing_moe_v2.get_rope_index``'s
    ``llm_pos_ids_list[-1].max() + 1`` accounting at the end of an
    image span).
    """
    grid_t = int(grid_thw[0].item())
    grid_h = int(grid_thw[1].item())
    grid_w = int(grid_thw[2].item())
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size

    if second_per_grid_t is not None:
        max_t = int((grid_t - 1) * float(second_per_grid_t) * float(tokens_per_second))
    else:
        max_t = grid_t - 1
    max_h = llm_grid_h - 1
    max_w = llm_grid_w - 1
    return int(start_pos) + max(max_t, max_h, max_w) + 1


__all__ = [
    "get_rope_index_text",
    "get_rope_index_audio",
    "get_rope_index_vision",
    "vision_span_max_position",
]
