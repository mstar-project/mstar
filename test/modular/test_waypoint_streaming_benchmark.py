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


def test_startup_latency_is_none_until_samples_are_asked_for(benchmark):
    """The key is always present, so a consumer never has to guess the shape."""
    assert benchmark["_startup_latency_metrics"]([]) is None


def test_startup_latency_summarizes_every_sample(benchmark):
    metrics = benchmark["_startup_latency_metrics"]([0.40, 0.10, 0.20, 0.30])

    assert metrics == {
        "sample_count": 4,
        "p50": pytest.approx(0.25),
        "p95": pytest.approx(0.385),
        "mean": pytest.approx(0.25),
        "minimum": 0.10,
        "maximum": 0.40,
    }


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--steps", "0"], "--steps must be positive"),
        (["--warmup-steps", "-1"], "--warmup-steps cannot be negative"),
        (["--startup-repeats", "-1"], "--startup-repeats cannot be negative"),
        (["--startup-steps", "0"], "--startup-steps must be positive"),
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


def test_streams_worlds_and_batch_default_to_one_without_the_new_flags(benchmark):
    """--streams 1 (today's only mode) must keep worlds=batch=1, matching the
    hardcoded worlds=1 _run_config call this replaced."""
    args = benchmark["_parse_args"](["--variant", "360p", "--physical-gpu", "2"])

    assert args.streams == 1
    assert args.worlds == 1
    assert args.batch == 1


def test_worlds_and_batch_default_to_streams(benchmark):
    args = benchmark["_parse_args"](
        ["--variant", "360p", "--physical-gpu", "2", "--streams", "4"]
    )

    assert args.worlds == 4
    assert args.batch == 4


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--streams", "0"], "--streams must be positive"),
        (["--streams", "4", "--worlds", "0"], "--worlds must be positive"),
        (["--streams", "4", "--batch", "0"], "--batch must be positive"),
        (["--streams", "4", "--worlds", "2", "--batch", "4"], "--batch must be <= --worlds"),
    ],
)
def test_benchmark_cli_rejects_invalid_concurrent_stream_configuration(
    benchmark, capsys, extra, message
):
    with pytest.raises(SystemExit, match="2"):
        benchmark["_parse_args"](["--variant", "360p", "--physical-gpu", "2", *extra])
    assert message in capsys.readouterr().err


def test_measure_stream_waits_on_start_barrier_before_opening_the_stream(benchmark, tmp_path):
    """The request body (client.stream(...)) must be built before the barrier
    wait, and consumption (the clock start) must not begin until after it."""
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

    calls = []

    class Barrier:
        def wait(self, timeout=None):
            calls.append("barrier_wait")

    class Client:
        def stream(self, **kwargs):
            calls.append("stream_called")
            return iter([VideoFrameChunk(bytes([1]) * 72, metadata(0))])

    clock_values = iter([10.0, 10.5, 10.6])
    metrics, failures = benchmark["_measure_stream"](
        Client(),
        tmp_path / "seed.png",
        variant,
        num_steps=1,
        request_id="rid",
        rng_seed=1,
        consumer_pause_seconds=0.0,
        stall_threshold_seconds=0.4,
        clock=lambda: next(clock_values),
        sleep=lambda _seconds: None,
        start_barrier=Barrier(),
    )

    assert calls == ["stream_called", "barrier_wait"]
    assert failures == []
    assert metrics["chunk_count"] == 1


def _chunk_stats(*, ttff_s, p50_s, p95_s, max_s, sustained, stalls):
    return {
        "time_to_first_frame_seconds": ttff_s,
        "inter_chunk_gap_seconds": {"p50": p50_s, "p95": p95_s, "maximum": max_s},
        "sustained_media_to_wall_ratio": sustained,
        "stalls": {"count": stalls},
    }


def test_stream_is_realtime_requires_sustained_gap_budget_and_no_stalls(benchmark):
    is_realtime = benchmark["_stream_is_realtime"]
    budget_s = benchmark["REALTIME_CHUNK_BUDGET_MS"] / 1000.0

    healthy = _chunk_stats(
        ttff_s=0.05, p50_s=0.05, p95_s=budget_s - 0.001, max_s=budget_s, sustained=1.05, stalls=0
    )
    assert is_realtime(healthy) is True

    under_sustained = {**healthy, "sustained_media_to_wall_ratio": 0.9}
    assert is_realtime(under_sustained) is False

    over_budget = {
        **healthy,
        "inter_chunk_gap_seconds": {**healthy["inter_chunk_gap_seconds"], "p95": budget_s + 0.001},
    }
    assert is_realtime(over_budget) is False

    stalled = {**healthy, "stalls": {"count": 1}}
    assert is_realtime(stalled) is False


def test_concurrent_aggregate_computes_worst_median_realtime_and_delivery_bound(benchmark):
    aggregate = benchmark["_concurrent_aggregate"]
    budget_s = benchmark["REALTIME_CHUNK_BUDGET_MS"] / 1000.0

    fast = _chunk_stats(ttff_s=0.05, p50_s=0.05, p95_s=0.06, max_s=0.07, sustained=1.2, stalls=0)
    slow = _chunk_stats(ttff_s=0.20, p50_s=0.15, p95_s=budget_s + 0.01, max_s=0.20, sustained=0.8, stalls=1)
    per_stream = [fast, slow]

    server_slack = {"step_spacing_ms": {"p50": 200.0, "p95": 220.0}}
    result = aggregate(per_stream, aggregate_fps=8.0, server=server_slack)

    assert result["aggregate_fps"] == 8.0
    assert result["ttff_ms"]["p50"] == pytest.approx(125.0)
    assert result["gap_ms"]["p50_worst"] == pytest.approx(150.0)
    assert result["gap_ms"]["p95_worst"] == pytest.approx((budget_s + 0.01) * 1000.0)
    assert result["gap_ms"]["max_worst"] == pytest.approx(200.0)
    assert result["gap_ms"]["p50_median"] == pytest.approx((50.0 + 150.0) / 2)
    assert result["sustained_min"] == pytest.approx(0.8)
    assert result["stall_count_total"] == 1
    assert result["realtime_per_stream"] == [True, False]
    assert result["all_realtime"] is False
    assert result["server"] is server_slack
    # gap p50_median (100ms) is not > 1.2x a 200ms server step: not delivery-bound.
    assert result["delivery_bound"] is False

    server_fast = {"step_spacing_ms": {"p50": 50.0, "p95": 60.0}}
    result_bound = aggregate(per_stream, aggregate_fps=8.0, server=server_fast)
    # gap p50_median (100ms) > 1.2x a 50ms server step: clients are the bottleneck.
    assert result_bound["delivery_bound"] is True


def test_concurrent_server_metrics_parses_rows_histogram_and_step_spacing(benchmark):
    server_metrics = benchmark["_concurrent_server_metrics"]
    log_text = "\n".join(
        [
            "2026-09-17 21:38:40,000 DEBUG [worker-0] mstar.worker.worker: "
            "Executing: dit graph_walk=rollout ('warmup-0',)",
            "2026-09-17 21:38:40,500 DEBUG [worker-0] mstar.worker.worker: "
            "Executing: dit graph_walk=rollout ('measured-0', 'measured-1')",
            "2026-09-17 21:38:40,600 DEBUG [worker-0] mstar.worker.worker: "
            "Executing: dit graph_walk=rollout ('measured-0', 'measured-1')",
            "2026-09-17 21:38:40,800 DEBUG [worker-0] mstar.worker.worker: "
            "Executing: dit graph_walk=rollout ('measured-0', 'measured-1')",
        ]
    )

    result = server_metrics(log_text, {"measured-0", "measured-1"}, 12.3)

    assert result["rows_per_step_histogram"] == {2: 3}
    assert result["step_spacing_ms"]["p50"] == pytest.approx(150.0)
    assert result["step_spacing_ms"]["p95"] == pytest.approx(195.0)
    assert result["startup_seconds"] == 12.3
