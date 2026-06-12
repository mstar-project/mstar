"""Tests for Ming's 3D MRoPE position-id helpers (step 5b).

These mirror the math in
``modeling_bailing_moe_v2.get_rope_index:625-647`` (vision span) and
the pure-text branch (`658-675`). Audio is treated as text positions
upstream, so the audio helper is just a thin alias.
"""

from __future__ import annotations

import pytest
import torch

from mstar.model.ming_omni_flash.components.positions import (
    get_rope_index_audio,
    get_rope_index_text,
    get_rope_index_vision,
    vision_span_max_position,
)

# ---------------------------------------------------------------------------
# get_rope_index_text
# ---------------------------------------------------------------------------


def test_text_positions_shape_and_offset() -> None:
    """``(3, T)`` with identical sequential rows offset by start_pos."""
    pos = get_rope_index_text(seq_len=5, start_pos=10)
    assert pos.shape == (3, 5)
    expected = torch.tensor([[10, 11, 12, 13, 14]] * 3)
    torch.testing.assert_close(pos, expected)


def test_text_positions_start_at_zero() -> None:
    pos = get_rope_index_text(seq_len=3, start_pos=0)
    assert pos.tolist() == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]


def test_text_positions_long_dtype_default() -> None:
    pos = get_rope_index_text(seq_len=2, start_pos=0)
    assert pos.dtype == torch.long


# ---------------------------------------------------------------------------
# get_rope_index_audio
# ---------------------------------------------------------------------------


def test_audio_positions_match_text_positions() -> None:
    """Audio is text-positioned upstream — verify the helper aliases."""
    a = get_rope_index_audio(num_audio_tokens=7, start_pos=4)
    t = get_rope_index_text(seq_len=7, start_pos=4)
    torch.testing.assert_close(a, t)


# ---------------------------------------------------------------------------
# get_rope_index_vision
# ---------------------------------------------------------------------------


def test_vision_positions_single_image_no_temporal_scale() -> None:
    """grid_thw=(1, 4, 4), spatial_merge=2 → 1 * 2 * 2 = 4 tokens.

    Temporal row: all 0 (single frame); H row cycles [0,0,1,1];
    W row cycles [0,1,0,1]. All offset by start_pos=10 → [10..].
    """
    pos = get_rope_index_vision(
        grid_thw=torch.tensor([1, 4, 4], dtype=torch.long),
        start_pos=10,
        spatial_merge_size=2,
    )
    assert pos.shape == (3, 4)
    expected = torch.tensor([
        [10, 10, 10, 10],  # T
        [10, 10, 11, 11],  # H
        [10, 11, 10, 11],  # W
    ])
    torch.testing.assert_close(pos, expected)


def test_vision_positions_multi_frame_indexes_t_per_frame() -> None:
    """grid_thw=(3, 2, 2), spatial_merge=2 → 3 frames × 1 × 1 = 3 tokens.

    Temporal row increments per frame; H/W rows are zero (single
    merged token per frame). No abs-time scaling here.
    """
    pos = get_rope_index_vision(
        grid_thw=torch.tensor([3, 2, 2], dtype=torch.long),
        start_pos=0,
        spatial_merge_size=2,
    )
    assert pos.shape == (3, 3)
    expected = torch.tensor([[0, 1, 2], [0, 0, 0], [0, 0, 0]])
    torch.testing.assert_close(pos, expected)


def test_vision_positions_absolute_time_scales_temporal() -> None:
    """``second_per_grid_t * tokens_per_second`` multiplies temporal row.

    Mirrors the video branch of get_rope_index where
    ``time_tensor = expanded * second_per_grid_t * tokens_per_second``.
    """
    pos = get_rope_index_vision(
        grid_thw=torch.tensor([4, 2, 2], dtype=torch.long),
        start_pos=0,
        spatial_merge_size=2,
        second_per_grid_t=0.5,    # half a second per frame
        tokens_per_second=2,
    )
    # T row: (frame_index * 0.5 * 2).long() → [0, 1, 2, 3] across frames,
    # each repeated H*W=1 times.
    assert pos[0].tolist() == [0, 1, 2, 3]
    assert pos[1].tolist() == [0, 0, 0, 0]
    assert pos[2].tolist() == [0, 0, 0, 0]


def test_vision_positions_rejects_bad_grid_thw_shape() -> None:
    with pytest.raises(ValueError, match="grid_thw must be a 1-D tensor of length 3"):
        get_rope_index_vision(
            grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),
            start_pos=0,
            spatial_merge_size=2,
        )


def test_vision_positions_rejects_non_divisible_grid() -> None:
    with pytest.raises(ValueError, match="not divisible by spatial_merge_size"):
        get_rope_index_vision(
            grid_thw=torch.tensor([1, 3, 4], dtype=torch.long),
            start_pos=0,
            spatial_merge_size=2,
        )


# ---------------------------------------------------------------------------
# vision_span_max_position
# ---------------------------------------------------------------------------


def test_vision_span_max_position_no_time_scale() -> None:
    """Largest pos in (1, 4, 4) span at start=10 is max(0, 1, 1) = 1; +1 = 12."""
    nxt = vision_span_max_position(
        grid_thw=torch.tensor([1, 4, 4]),
        start_pos=10,
        spatial_merge_size=2,
    )
    assert nxt == 10 + 1 + 1   # start + max(H,W,T) + 1


def test_vision_span_max_position_with_time_scale() -> None:
    """(4, 2, 2) with 0.5s/frame, 2 tps → T=[0,1,2,3]; max=3; +start+1=4."""
    nxt = vision_span_max_position(
        grid_thw=torch.tensor([4, 2, 2]),
        start_pos=0,
        spatial_merge_size=2,
        second_per_grid_t=0.5,
        tokens_per_second=2,
    )
    assert nxt == 4
