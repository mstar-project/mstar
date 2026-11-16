"""CPU checks for the served LingBot-World chunk timing report."""

import pytest

from test.lingbot_world.serve_first_chunk import summarize_chunks


def test_chunk_cadence_and_playback_stalls():
    result = summarize_chunks([1.0, 2.0, 3.5], [13, 16, 16], [16.0] * 3)

    assert result["first_chunk_s"] == 1.0
    assert result["chunk_gap_s"] == [1.0, 1.5]
    assert result["chunk_gap_s_p50"] == 1.25
    assert result["generated_fps"] == pytest.approx(12.8)
    assert result["playback_ratio"] == pytest.approx(0.8)
    assert result["playback_stall_s"] == pytest.approx(0.6875)
    assert result["generated_frames"] == 45


def test_one_chunk_has_no_sustained_rate():
    result = summarize_chunks([0.5], [13], [16.0])

    assert result["generated_fps"] is None
    assert result["playback_ratio"] is None
    assert result["chunk_gap_s_p50"] is None
    assert result["playback_stall_s"] == 0.0


@pytest.mark.parametrize(
    ("arrivals", "counts", "fps"),
    [([1, 2], [13], [16, 16]), ([1, 0.9], [13, 16], [16, 16]),
     ([1], [0], [16]), ([1], [13], [0]), ([1], [13], [float("nan")])],
)
def test_invalid_chunk_metadata_rejected(arrivals, counts, fps):
    with pytest.raises(ValueError):
        summarize_chunks(arrivals, counts, fps)
