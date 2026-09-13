from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

from mstar.client import VideoFrameChunk


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(
        str(Path(__file__).parents[1] / "waypoint" / "benchmark_streaming.py")
    )


def test_streaming_metric_math_includes_pacing_jitter_and_stalls(benchmark):
    observation = benchmark["ChunkObservation"]
    metrics = benchmark["_stream_metrics"](
        [
            observation(1.0, 100, 0, 4, 4.0),
            observation(2.0, 100, 4, 4, 4.0),
            observation(4.0, 100, 8, 4, 4.0),
        ],
        request_wall_seconds=4.2,
        stall_threshold_seconds=1.5,
        consumer_pause_seconds=0.25,
        consumer_pause_count=2,
        payload_sha256="abc",
    )

    assert metrics["time_to_first_frame_seconds"] == 1.0
    assert metrics["generated_media_seconds"] == 3.0
    assert metrics["sustained_media_to_wall_ratio"] == pytest.approx(2 / 3)
    assert metrics["overall_media_to_wall_ratio"] == pytest.approx(3 / 4.2)
    assert metrics["inter_chunk_gap_seconds"] == {
        "sample_count": 2,
        "p50": 1.5,
        "p95": 1.95,
        "mean": 1.5,
        "jitter_population_stddev": 0.5,
        "maximum": 2.0,
    }
    assert metrics["stalls"] == {
        "threshold_seconds": 1.5,
        "count": 1,
        "longest_seconds": 2.0,
        "total_excess_seconds": 0.5,
    }
    assert metrics["consumer"]["injected_pause_seconds"] == 0.5


def test_streaming_metric_math_handles_one_chunk_without_fake_gap(benchmark):
    observation = benchmark["ChunkObservation"]
    metrics = benchmark["_stream_metrics"](
        [observation(0.5, 72, 0, 4, 60.0)],
        request_wall_seconds=0.6,
        stall_threshold_seconds=0.25,
        consumer_pause_seconds=0.0,
        consumer_pause_count=0,
        payload_sha256="def",
    )

    assert metrics["sustained_media_to_wall_ratio"] is None
    assert metrics["inter_chunk_gap_seconds"]["sample_count"] == 0
    assert metrics["inter_chunk_gap_seconds"]["p50"] is None
    assert metrics["inter_chunk_gap_seconds"]["jitter_population_stddev"] is None
    assert metrics["stalls"]["count"] == 0


def test_backpressure_summary_reports_deltas_without_a_threshold(benchmark):
    baseline = {
        "request_wall_seconds": 2.0,
        "time_to_first_frame_seconds": 0.5,
        "sustained_media_to_wall_ratio": 0.8,
        "payload_sha256": "same",
        "consumer": {"pause_seconds": 0.0, "injected_pause_seconds": 0.0},
    }
    slow = {
        "request_wall_seconds": 3.2,
        "time_to_first_frame_seconds": 0.6,
        "sustained_media_to_wall_ratio": 0.4,
        "payload_sha256": "same",
        "consumer": {"pause_seconds": 0.25, "injected_pause_seconds": 1.0},
    }
    baseline_memory = {"peak_host_pss_mib": 100.0, "peak_gpu_mib": 1000.0}
    slow_memory = {"peak_host_pss_mib": 112.0, "peak_gpu_mib": 1004.0}

    result = benchmark["_backpressure_metrics"](
        baseline, slow, baseline_memory, slow_memory
    )

    assert result["observed_request_wall_increase_seconds"] == pytest.approx(1.2)
    assert result["wall_increase_beyond_injected_pause_seconds"] == pytest.approx(0.2)
    assert result["peak_host_pss_change_mib"] == 12.0
    assert result["peak_gpu_memory_change_mib"] == 4.0
    assert result["payloads_match"] is True
    assert "threshold" not in result


def test_measurement_loop_consumes_typed_chunks_and_pauses_only_between_them(
    benchmark, tmp_path
):
    variant = benchmark["rollout"].Variant("test", 2, 3, 1, "unused")

    def metadata(frame_index):
        return {
            "width": 3,
            "height": 2,
            "fps": 60.0,
            "pixel_format": "rgb24",
            "frame_index": frame_index,
            "frame_count": 4,
        }

    payloads = [bytes([1]) * 72, bytes([2]) * 72]

    class Client:
        kwargs = None

        def stream(self, **kwargs):
            self.kwargs = kwargs
            return iter(
                [
                    VideoFrameChunk(payloads[0], metadata(0)),
                    VideoFrameChunk(payloads[1], metadata(4)),
                ]
            )

    client = Client()
    clock_values = iter([10.0, 10.5, 11.0, 11.1])
    pauses = []
    metrics, failures = benchmark["_measure_stream"](
        client,
        tmp_path / "seed.png",
        variant,
        num_steps=2,
        request_id="rid",
        rng_seed=7,
        consumer_pause_seconds=0.25,
        stall_threshold_seconds=0.4,
        clock=lambda: next(clock_values),
        sleep=pauses.append,
    )

    assert failures == []
    assert pauses == [0.25]
    assert metrics["time_to_first_frame_seconds"] == 0.5
    assert metrics["inter_chunk_gap_seconds"]["p95"] == 0.5
    assert metrics["consumer"]["pause_count"] == 1
    assert metrics["payload_sha256"] == hashlib.sha256(b"".join(payloads)).hexdigest()
    assert client.kwargs["output_modalities"] == ("video_frame",)
    assert len(client.kwargs["actions"]) == 2


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--steps", "0"], "--steps must be positive"),
        (["--warmup-steps", "-1"], "--warmup-steps cannot be negative"),
        (["--slow-consumer-delay", "-0.1"], "--slow-consumer-delay cannot be negative"),
        (["--stall-threshold", "0"], "--stall-threshold must be positive"),
        (["--memory-sample-interval", "0"], "--memory-sample-interval must be positive"),
    ],
)
def test_benchmark_cli_rejects_invalid_measurement_configuration(
    benchmark, capsys, extra, message
):
    with pytest.raises(SystemExit, match="2"):
        benchmark["_parse_args"](
            ["--variant", "360p", "--physical-gpu", "2", *extra]
        )
    assert message in capsys.readouterr().err


def test_benchmark_cli_rejects_hub_with_local_overrides(benchmark, capsys):
    with pytest.raises(SystemExit, match="2"):
        benchmark["_parse_args"](
            [
                "--variant",
                "720p",
                "--physical-gpu",
                "2",
                "--source",
                "hub",
                "--checkpoint-dir",
                "/tmp/checkpoint",
            ]
        )
    assert "--source hub cannot be combined" in capsys.readouterr().err


def test_benchmark_cli_derives_stall_threshold_and_has_no_release_gate(benchmark):
    args = benchmark["_parse_args"](
        ["--variant", "360p", "--physical-gpu", "2"]
    )

    assert args.stall_threshold is None
    assert benchmark["_resolve_stall_threshold"](args) == pytest.approx(4.0 / 15.0)
    assert not any(
        action.dest.startswith("release")
        for action in benchmark["_build_parser"]()._actions
    )
