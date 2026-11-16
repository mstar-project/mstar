"""CPU checks for the served LingBot-World chunk timing report."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mstar.client import VideoFrameChunk

# CPU Core imports this file as a standalone test module. Load the harness by
# path so its tests do not depend on `test` resolving as a namespace package.
_HARNESS = Path(__file__).resolve().parents[1] / "lingbot_world" / "serve_first_chunk.py"
_SPEC = importlib.util.spec_from_file_location("serve_first_chunk", _HARNESS)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)

summarize_chunks = probe.summarize_chunks


def _chunk(index: int, count: int) -> VideoFrameChunk:
    return VideoFrameChunk(
        b"\0" * (count * 3),
        {"width": 1, "height": 1, "fps": 16, "pixel_format": "rgb24",
         "frame_index": index, "frame_count": count},
    )


class _Client:
    def __init__(self, events):
        self.events = events

    def stream(self, **_kwargs):
        return iter(self.events)


def _run(events, *, num_frames=29, max_chunks=None):
    poses = np.tile(np.eye(4), (num_frames, 1, 1))
    return probe.run_case(_Client(events), VideoFrameChunk, poses, [1, 1, 1, 1], Path("/tmp/example"),
                          num_frames, "4x4", max_chunks, "cpu")


def test_model_rounds_requested_frames_to_whole_chunks():
    assert probe.expected_chunk_frame_counts(93) == [13] + [16] * 5
    assert len(probe.expected_chunk_frame_counts(3001)) == 187
    assert sum(probe.expected_chunk_frame_counts(3001)) == 2989


def test_full_rollout_requires_expected_frame_coverage():
    complete = _run([_chunk(0, 13), _chunk(13, 16)])
    short = _run([_chunk(0, 13)])

    assert complete["sample_complete"] is True
    assert complete["expected_frames"] == 29
    assert short["sample_complete"] is False
    assert "expected 29 frames" in short["error"]


def test_noncontiguous_frame_indices_fail_even_with_expected_count():
    result = _run([_chunk(0, 13), _chunk(14, 16)])

    assert result["sample_complete"] is False
    assert "frame_index" in result["error"]


def test_capped_sample_is_explicitly_incomplete_without_error():
    result = _run([_chunk(0, 13), _chunk(13, 16)], max_chunks=1)

    assert result["sample_complete"] is False
    assert "error" not in result


def test_main_returns_failure_for_truncated_full_rollout(monkeypatch, tmp_path):
    from mstar import client as client_module

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout):
            pass

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def health(self):
            return True

        def stream(self, **kwargs):
            if kwargs["request_id"] == "case0":
                return iter([_chunk(0, 13)])
            return iter([_chunk(0, 13)] + [_chunk(13 + i * 16, 16) for i in range(5)])

    monkeypatch.setattr(probe.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(client_module, "MStarClient", FakeClient)
    monkeypatch.setattr(probe.np, "load", lambda path: np.tile(np.eye(4), (29, 1, 1))
                        if str(path).endswith("poses.npy") else np.array([[1, 1, 1, 1]]))
    monkeypatch.setattr(probe.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["probe", "--example", str(tmp_path), "--out", str(tmp_path / "out"),
                                       "--full-rollout", "--cases", "29:4x4"])

    assert probe.main() == 1
    report = json.loads((tmp_path / "out" / "first_frame.json").read_text())
    assert report["cases"][0]["sample_complete"] is False


def test_invalid_case_fails_before_server_start(monkeypatch, tmp_path):
    def unexpected_start(*_args, **_kwargs):
        raise AssertionError("server started for an invalid case")

    monkeypatch.setattr(probe.subprocess, "Popen", unexpected_start)
    monkeypatch.setattr(sys, "argv", ["probe", "--example", str(tmp_path), "--out", str(tmp_path / "out"),
                                       "--cases", "0:4x4"])

    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


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
