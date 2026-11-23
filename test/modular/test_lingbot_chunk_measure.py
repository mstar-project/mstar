"""CPU checks for the served LingBot-World chunk timing report."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mstar.client import TextChunk, VideoFrameChunk

# CPU Core imports this file as a standalone test module. Load the harness by
# path so its tests do not depend on `test` resolving as a namespace package.
_HARNESS = Path(__file__).resolve().parents[1] / "lingbot_world" / "serve_first_chunk.py"
_SPEC = importlib.util.spec_from_file_location("serve_first_chunk", _HARNESS)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)

summarize_chunks = probe.summarize_chunks


def _chunk(index: int, count: int, *, fps: float = 16) -> VideoFrameChunk:
    return VideoFrameChunk(
        b"\0" * (count * 3),
        {"width": 1, "height": 1, "fps": fps, "pixel_format": "rgb24",
         "frame_index": index, "frame_count": count},
    )


class _Client:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.events)


def _run(events, *, num_frames=29, layout="4x4", max_chunks=None):
    poses = np.tile(np.eye(4), (num_frames, 1, 1))
    return probe.run_case(_Client(events), VideoFrameChunk, poses, [1, 1, 1, 1], Path("/tmp/example"),
                          num_frames, layout, max_chunks, "cpu")


def _install_fake_clock(monkeypatch, times):
    it = iter(times)
    last = times[-1]

    def fake_perf_counter():
        nonlocal last
        try:
            last = next(it)
        except StopIteration:
            pass
        return last

    monkeypatch.setattr(probe.time, "perf_counter", fake_perf_counter)


def _make_example_dir(tmp_path: Path, *, with_image: bool = True) -> Path:
    example = tmp_path / "example"
    example.mkdir(parents=True, exist_ok=True)
    if with_image:
        (example / "image.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    return example


def _warm_chunks():
    return [_chunk(0, 13)] + [_chunk(13 + i * 16, 16) for i in range(5)]


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
    (tmp_path / "image.jpg").write_bytes(b"\xff\xd8\xff\xd9")
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
    [
        ([], [], []),
        ([1, 2], [13], [16, 16]),
        ([-0.1], [13], [16.0]),
        ([float("inf")], [13], [16.0]),
        ([True], [13], [16.0]),
        ([1.0, 1.0], [13, 16], [16.0, 16.0]),
        ([1, 0.9], [13, 16], [16, 16]),
        ([1], [0], [16]),
        ([1.0], [True], [16.0]),
        ([1], [13], [0]),
        ([1.0], [13], [-16.0]),
        ([1], [13], [float("nan")]),
        ([1.0], [13], [float("inf")]),
        ([1.0], [13], [True]),
        ([1.0, 2.0], [13, 16], [16.0, 24.0]),
    ],
)
def test_invalid_chunk_metadata_rejected(arrivals, counts, fps):
    with pytest.raises(ValueError):
        summarize_chunks(arrivals, counts, fps)


def test_pingpong_short_exact_wrap_and_small_poses():
    poses4 = np.arange(4 * 16, dtype=np.float64).reshape(4, 4, 4)
    assert np.array_equal(probe.pingpong(poses4, 2), poses4[[0, 1]])
    assert np.array_equal(probe.pingpong(poses4, 4), poses4[[0, 1, 2, 3]])
    assert np.array_equal(probe.pingpong(poses4, 10), poses4[[0, 1, 2, 3, 2, 1, 0, 1, 2, 3]])

    poses1 = np.arange(16, dtype=np.float64).reshape(1, 4, 4)
    assert np.array_equal(probe.pingpong(poses1, 5), poses1[[0, 0, 0, 0, 0]])

    poses2 = np.arange(32, dtype=np.float64).reshape(2, 4, 4)
    assert np.array_equal(probe.pingpong(poses2, 5), poses2[[0, 1, 0, 1, 0]])

    with pytest.raises(ValueError, match="non-empty"):
        probe.pingpong(np.empty((0, 4, 4)), 4)
    with pytest.raises(ValueError, match="non-negative"):
        probe.pingpong(poses2, -1)


def test_expected_chunk_frame_counts_boundaries_and_config_parity():
    from mstar.model.lingbot_world.config import lingbot_world_1_3b

    cfg = lingbot_world_1_3b()
    assert probe.expected_chunk_frame_counts(13) == [13]
    max_counts = probe.expected_chunk_frame_counts(4077)
    assert len(max_counts) == cfg.max_chunks == 255
    assert sum(max_counts) == 4077
    assert max_counts == [13] + [16] * 254

    for invalid in (12, 0, -1, 4093, True):
        with pytest.raises(ValueError):
            probe.expected_chunk_frame_counts(invalid)


def test_expected_chunk_frame_counts_matches_model_num_latent_frames():
    from mstar.model.lingbot_world.lingbot_world_model import LingBotWorldModel

    model = LingBotWorldModel(skip_weight_loading=True)
    stride = model.config.vae_stride[0]
    for n in (13, 16, 17, 28, 29, 93, 3001, 4077):
        lat_f = model.num_latent_frames(n, n)
        chunks = lat_f // model.config.chunk_size
        counts = probe.expected_chunk_frame_counts(n)
        assert len(counts) == chunks
        assert sum(counts) == 1 + stride * (lat_f - 1)


def test_summarize_chunks_early_arrival_buffered_zero_stall():
    # Chunk 0: 13 frames @ 16 fps = 0.8125s -> playback_until = 1.0 + 0.8125 = 1.8125
    # Chunk 1: arrives at 1.5s (< 1.8125s), +1.0s buffer -> playback_until = 2.8125
    # Chunk 2: arrives at 2.6s (gap 1.1s > 1.0s, yet < 2.8125s due to surplus carry-over) -> 0 stall
    result = summarize_chunks([1.0, 1.5, 2.6], [13, 16, 16], [16.0] * 3)

    assert result["playback_stall_s"] == 0.0
    assert result["generated_fps"] == pytest.approx(32.0 / 1.6)
    assert result["playback_ratio"] == pytest.approx((32.0 / 1.6) / 16.0)
    assert result["playback_ratio"] >= 1.0


def test_run_case_3x4_layout_slices_bottom_row_and_reduces_payload():
    poses = np.tile(np.eye(4), (29, 1, 1))
    client_4x4 = _Client([_chunk(0, 13), _chunk(13, 16)])
    client_3x4 = _Client([_chunk(0, 13), _chunk(13, 16)])

    res_4x4 = probe.run_case(client_4x4, VideoFrameChunk, poses, [1, 1, 1, 1], Path("/tmp/ex"), 29, "4x4", None, "c4")
    res_3x4 = probe.run_case(client_3x4, VideoFrameChunk, poses, [1, 1, 1, 1], Path("/tmp/ex"), 29, "3x4", None, "c3")

    assert np.asarray(client_4x4.calls[0]["camera_poses"]).shape == (29, 4, 4)
    assert np.asarray(client_3x4.calls[0]["camera_poses"]).shape == (29, 3, 4)
    assert res_3x4["model_kwargs_bytes"] < res_4x4["model_kwargs_bytes"]
    assert res_3x4["sample_complete"] is True


def test_run_case_unexpected_stream_event_fails_and_closes_stream():
    class TrackingStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield TextChunk(text="unexpected")

        def close(self):
            self.closed = True

    stream = TrackingStream()

    class StreamClient:
        def stream(self, **_kwargs):
            return stream

    poses = np.tile(np.eye(4), (29, 1, 1))
    res = probe.run_case(StreamClient(), VideoFrameChunk, poses, [1, 1, 1, 1], Path("/tmp/ex"), 29, "4x4", None, "u")

    assert stream.closed is True
    assert res["sample_complete"] is False
    assert "RuntimeError: unexpected stream event TextChunk" in res["error"]


def test_run_case_midstream_exception_closes_stream_and_preserves_primary_error():
    class FailingStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield _chunk(0, 13, fps=16)
            yield _chunk(13, 16, fps=24)  # would also fail summarize_chunks on FPS mismatch
            raise RuntimeError("mid-stream disconnect")

        def close(self):
            self.closed = True

    stream = FailingStream()

    class StreamClient:
        def stream(self, **_kwargs):
            return stream

    poses = np.tile(np.eye(4), (45, 1, 1))
    res = probe.run_case(StreamClient(), VideoFrameChunk, poses, [1, 1, 1, 1], Path("/tmp/ex"), 45, "4x4", None, "err")

    assert stream.closed is True
    assert res["sample_complete"] is False
    assert res["error"] == "RuntimeError: mid-stream disconnect"


def test_run_case_extra_chunk_and_wrong_frame_count_and_empty_stream():
    extra = _run([_chunk(0, 13), _chunk(13, 16), _chunk(29, 16)], num_frames=29)
    assert extra["sample_complete"] is False
    assert "expected None" in extra["error"]

    wrong_count = _run([_chunk(0, 14)], num_frames=29)
    assert wrong_count["sample_complete"] is False
    assert "expected 13" in wrong_count["error"]

    empty = _run([], num_frames=29)
    assert empty["first_chunk_s"] is None
    assert empty["sample_complete"] is False
    assert empty["error"] == "expected 29 frames; got 0"


def test_run_case_summarize_chunks_failure_forces_incomplete(monkeypatch):
    _install_fake_clock(monkeypatch, [0.0, 1.0, 2.0, 2.1])
    res = _run([_chunk(0, 13, fps=16), _chunk(13, 16, fps=24)], num_frames=29)

    assert res["sample_complete"] is False
    assert "ValueError: chunk playback FPS" in res["error"]


def test_run_case_chunk_gap_s_median_two_vs_three_plus_chunks(monkeypatch):
    _install_fake_clock(monkeypatch, [0.0, 1.0, 2.0, 2.5])
    two_chunks = _run([_chunk(0, 13), _chunk(13, 16)], num_frames=29)
    assert two_chunks["chunk_gap_s"] == [1.0]
    assert two_chunks["chunk_gap_s_p50"] == 1.0
    assert two_chunks["chunk_gap_s_median"] is None

    # Arrivals relative to t0=0.0: [1.0, 5.0, 6.0, 8.0] -> gaps [4.0, 1.0, 2.0]
    # p50(all gaps) = 2.0, chunk_gap_s_median(gaps[1:]) = 1.5
    _install_fake_clock(monkeypatch, [0.0, 1.0, 5.0, 6.0, 8.0, 8.1])
    four_chunks = _run([_chunk(0, 13), _chunk(13, 16), _chunk(29, 16), _chunk(45, 16)], num_frames=61)
    assert four_chunks["chunk_gap_s"] == pytest.approx([4.0, 1.0, 2.0])
    assert four_chunks["chunk_gap_s_p50"] == pytest.approx(2.0)
    assert four_chunks["chunk_gap_s_median"] == pytest.approx(1.5)


@pytest.mark.parametrize(
    "extra_argv",
    [
        ["--chunks", "0"],
        ["--chunks", "-1"],
        ["--cases", "93"],
        ["--cases", "93:4x4:x"],
        ["--cases", "abc:4x4"],
        ["--cases", "93:2x4"],
        ["--cases", "4093:4x4"],
        ["--timeout", "0"],
        ["--timeout", "-5"],
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--port", "0"],
        ["--port", "65536"],
        ["--config", "configs/does_not_exist.yaml"],
    ],
)
def test_main_rejects_invalid_cli_args_before_server_start(monkeypatch, tmp_path, extra_argv):
    example = _make_example_dir(tmp_path)
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Popen")))
    monkeypatch.setattr(sys, "argv", ["probe", "--example", str(example), "--out", str(tmp_path / "out"), *extra_argv])

    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_main_rejects_missing_image_and_invalid_trajectory_before_server_start(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Popen")))

    no_img = _make_example_dir(tmp_path / "no_img", with_image=False)
    monkeypatch.setattr(
        sys, "argv",
        ["probe", "--example", str(no_img), "--out", str(tmp_path / "out1"), "--cases", "29:4x4"],
    )
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2

    for tag, poses_arr, intr_arr in [
        ("empty_poses", np.zeros((0, 4, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)),
        ("wrong_shape_poses", np.tile(np.eye(4)[:3], (4, 1, 1)), np.ones((1, 4), dtype=np.float32)),
        ("nan_poses", np.full((2, 4, 4), float("nan"), dtype=np.float32), np.ones((1, 4), dtype=np.float32)),
        ("bad_intr", np.tile(np.eye(4), (4, 1, 1)), np.ones((1, 3), dtype=np.float32)),
    ]:
        bad_dir = _make_example_dir(tmp_path / tag)
        np.save(bad_dir / "poses.npy", poses_arr)
        np.save(bad_dir / "intrinsics.npy", intr_arr)
        monkeypatch.setattr(
            sys, "argv",
            ["probe", "--example", str(bad_dir), "--out", str(tmp_path / f"out_{tag}"), "--cases", "29:4x4"],
        )
        with pytest.raises(SystemExit) as error:
            probe.main()
        assert error.value.code == 2


def test_main_server_premature_exit_and_timeout_cleanup(monkeypatch, tmp_path):
    from mstar import client as client_module

    example = _make_example_dir(tmp_path)
    np.save(example / "poses.npy", np.tile(np.eye(4), (29, 1, 1)))
    np.save(example / "intrinsics.npy", np.ones((1, 4), dtype=np.float32))
    monkeypatch.setattr(probe.time, "sleep", lambda *_args: None)

    created_workdirs = []
    real_mkdtemp = probe.tempfile.mkdtemp

    def track_mkdtemp(**kwargs):
        d = Path(real_mkdtemp(dir=tmp_path, **kwargs))
        created_workdirs.append(d)
        return str(d)

    monkeypatch.setattr(probe.tempfile, "mkdtemp", track_mkdtemp)

    class DeadProcess:
        def __init__(self):
            self.killed = False
            self.wait_calls = []

        def poll(self):
            return 1

        def terminate(self):
            pass

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if timeout == 60:
                raise subprocess.TimeoutExpired(cmd="server", timeout=60)

        def kill(self):
            self.killed = True

    proc = DeadProcess()

    class UnhealthyClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def health(self):
            return False

    monkeypatch.setattr(probe.subprocess, "Popen", lambda *_args, **_kwargs: proc)
    monkeypatch.setattr(client_module, "MStarClient", UnhealthyClient)
    monkeypatch.setattr(
        sys, "argv",
        ["probe", "--example", str(example), "--out", str(tmp_path / "out"), "--cases", "29:4x4"],
    )

    with pytest.raises(SystemExit, match="server did not come up"):
        probe.main()

    assert proc.killed is True
    assert proc.wait_calls == [60, None]
    assert len(created_workdirs) == 1
    assert not created_workdirs[0].exists()


def test_main_exit_codes_and_config_timeout_forwarding(monkeypatch, tmp_path):
    from mstar import client as client_module

    example = _make_example_dir(tmp_path)
    np.save(example / "poses.npy", np.tile(np.eye(4), (29, 1, 1)))
    np.save(example / "intrinsics.npy", np.ones((1, 4), dtype=np.float32))
    custom_cfg = tmp_path / "custom.yaml"
    custom_cfg.write_text("model: lingbot_world_1_3b\n")
    monkeypatch.setattr(probe.time, "sleep", lambda *_args: None)

    popen_cmds = []
    client_timeouts = []

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def make_popen(cmd, **_kwargs):
        popen_cmds.append(cmd)
        return FakeProcess()

    monkeypatch.setattr(probe.subprocess, "Popen", make_popen)

    def run_with_streams(stream_fn, extra_args):
        class FakeClient:
            def __init__(self, _url, timeout=None):
                client_timeouts.append(timeout)

            def health(self):
                return True

            def stream(self, **kwargs):
                return stream_fn(kwargs["request_id"])

        monkeypatch.setattr(client_module, "MStarClient", FakeClient)
        out_dir = tmp_path / f"out_{len(popen_cmds)}"
        monkeypatch.setattr(sys, "argv", ["probe", "--example", str(example), "--out", str(out_dir), *extra_args])
        rc = probe.main()
        return rc, json.loads((out_dir / "first_frame.json").read_text())

    # 1. Capped mode: complete warmup/warm_93, truncated case (no error) -> exit 0, forwards --config/--timeout
    rc, rep = run_with_streams(
        lambda rid: iter(_warm_chunks() if rid in {"warm", "warm2"} else [_chunk(0, 13), _chunk(13, 16)]),
        ["--config", str(custom_cfg), "--timeout", "123", "--chunks", "1", "--cases", "29:4x4"],
    )
    assert rc == 0
    assert rep["full_rollout"] is False
    assert rep["cases"][0]["sample_complete"] is False
    assert "error" not in rep["cases"][0]
    assert popen_cmds[-1][popen_cmds[-1].index("--config") + 1] == str(custom_cfg)
    assert popen_cmds[-1][popen_cmds[-1].index("--timeout") + 1] == "123.0"
    assert client_timeouts[-1] == 123.0

    # 2. Capped mode: case with error -> exit 1
    rc, _ = run_with_streams(
        lambda rid: iter(_warm_chunks() if rid in {"warm", "warm2"} else []),
        ["--chunks", "1", "--cases", "29:4x4"],
    )
    assert rc == 1

    # 3. Full-rollout mode: complete warmup/warm_93 and multi-case 4x4 + 3x4 -> exit 0
    rc, rep = run_with_streams(
        lambda rid: iter(_warm_chunks() if rid in {"warm", "warm2"} else [_chunk(0, 13), _chunk(13, 16)]),
        ["--full-rollout", "--cases", "29:4x4", "29:3x4"],
    )
    assert rc == 0
    assert rep["full_rollout"] is True
    assert [c["layout"] for c in rep["cases"]] == ["4x4", "3x4"]
    assert all(c["sample_complete"] for c in rep["cases"])

    # 4. Incomplete warmup -> exit 1 even if cases succeed
    rc, _ = run_with_streams(
        lambda rid: iter([_chunk(0, 13)] if rid == "warm"
                         else (_warm_chunks() if rid == "warm2" else [_chunk(0, 13), _chunk(13, 16)])),
        ["--full-rollout", "--cases", "29:4x4"],
    )
    assert rc == 1

    # 5. Incomplete warm_93 -> exit 1 even if cases succeed
    rc, _ = run_with_streams(
        lambda rid: iter([_chunk(0, 13)] if rid == "warm2"
                         else (_warm_chunks() if rid == "warm" else [_chunk(0, 13), _chunk(13, 16)])),
        ["--full-rollout", "--cases", "29:4x4"],
    )
    assert rc == 1
