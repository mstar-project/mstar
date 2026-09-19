"""Onset logic for _tools/batch_sweep_summary.py, the pure-stdlib summarizer
for the concurrent-stream batch_sweep.sh sweep. Lives outside this repo (see
CLAUDE.md for _tools/ conventions) so it is loaded by absolute path."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

SUMMARY_SCRIPT = Path("/shared/home/garv901-55613a/waypoint-int/_tools/batch_sweep_summary.py")


def _solo_baseline_artifact(*, ttff_s, gap_p50_s, sustained, fps, gpu_peak_mib):
    frame_count = 40
    wall_seconds = frame_count / fps
    return {
        "status": "completed",
        "server": {"startup_seconds": 5.0},
        "runs": {
            "baseline": {
                "time_to_first_frame_seconds": ttff_s,
                "inter_chunk_gap_seconds": {
                    "p50": gap_p50_s,
                    "p95": gap_p50_s + 0.001,
                    "maximum": gap_p50_s + 0.002,
                },
                "sustained_media_to_wall_ratio": sustained,
                "stalls": {"count": 0},
                "on_time_chunk_fraction": 1.0,
                "frame_count": frame_count,
                "request_wall_seconds": wall_seconds,
                "memory": {"peak_gpu_mib": gpu_peak_mib},
            }
        },
    }


def _concurrent_artifact(
    *, ttff_p50_ms, gap_p50_median_ms, aggregate_fps, all_realtime, delivery_bound, gpu_peak_mib,
    streams=2, realtime_count=None, on_time_chunk_fraction_min=1.0,
):
    if realtime_count is None:
        realtime_count = streams if all_realtime else max(streams - 1, 0)
    return {
        "status": "completed",
        "concurrent": {
            "gpu_peak_mib": gpu_peak_mib,
            "streams": streams,
            "realtime_count": realtime_count,
            "on_time_chunk_fraction_min": on_time_chunk_fraction_min,
            "ttff_ms": {"p50": ttff_p50_ms, "p95": ttff_p50_ms + 5.0},
            "gap_ms": {
                "p50_median": gap_p50_median_ms,
                "p95_worst": gap_p50_median_ms + 5.0,
                "max_worst": gap_p50_median_ms + 10.0,
            },
            "sustained_min": 1.1,
            "aggregate_fps": aggregate_fps,
            "all_realtime": all_realtime,
            "delivery_bound": delivery_bound,
            "server": {
                "startup_seconds": 5.0,
                "step_spacing_ms": {"p50": 50.0, "p95": 55.0},
                "rows_per_step_histogram": {"2": 30},
            },
        },
    }


def _write_sweep(tmp_path: Path) -> Path:
    """B=1..16 grid with a distinct, deliberately placed onset per metric:
    gap regression at B=4, TTFF regression and fps-gain flattening at B=8
    (B=8 is also where delivery_bound flips True), and all_realtime failing
    only at B=16. max realtime B should land on B=4 (the largest B that is
    both realtime and not delivery-bound)."""
    artifacts = {
        1: _solo_baseline_artifact(ttff_s=0.050, gap_p50_s=0.020, sustained=1.2, fps=10.0, gpu_peak_mib=1000.0),
        2: _concurrent_artifact(
            ttff_p50_ms=55.0, gap_p50_median_ms=21.0, aggregate_fps=19.0,
            all_realtime=True, delivery_bound=False, gpu_peak_mib=1500.0, streams=2,
        ),
        4: _concurrent_artifact(
            ttff_p50_ms=58.0, gap_p50_median_ms=23.0, aggregate_fps=27.0,
            all_realtime=True, delivery_bound=False, gpu_peak_mib=2000.0, streams=4,
        ),
        8: _concurrent_artifact(
            ttff_p50_ms=65.0, gap_p50_median_ms=30.0, aggregate_fps=29.0,
            all_realtime=True, delivery_bound=True, gpu_peak_mib=3000.0, streams=8,
        ),
        16: _concurrent_artifact(
            ttff_p50_ms=70.0, gap_p50_median_ms=40.0, aggregate_fps=29.5,
            all_realtime=False, delivery_bound=False, gpu_peak_mib=4000.0,
            streams=16, realtime_count=15, on_time_chunk_fraction_min=0.9,
        ),
    }
    for b, artifact in artifacts.items():
        (tmp_path / f"b{b}.json").write_text(json.dumps(artifact))
    return tmp_path


def test_summary_table_and_onsets_over_a_synthetic_sweep(tmp_path, capsys):
    out_dir = _write_sweep(tmp_path)
    module = runpy.run_path(str(SUMMARY_SCRIPT))

    rows, notes = module["_load_rows"](out_dir)
    assert notes == []
    assert [row["b"] for row in rows] == [1, 2, 4, 8, 16]
    # B=1 falls back to runs.baseline: gap p50 0.020s -> 20ms, fps 40/4s = 10.
    assert rows[0]["gap_p50_median_ms"] == 20.0
    assert rows[0]["aggregate_fps"] == 10.0
    assert rows[0]["delivery_bound"] is False
    # Viability columns: how many streams stayed realtime, and the worst
    # stream's per-chunk in-budget fraction.
    by_b = {row["b"]: row for row in rows}
    assert by_b[1]["realtime_count"] == 1 and by_b[1]["streams"] == 1
    assert by_b[16]["realtime_count"] == 15 and by_b[16]["streams"] == 16
    assert by_b[16]["on_time_chunk_fraction_min"] == 0.9

    onsets = module["_onsets"](rows)
    assert "gap p50_median regression onset (>1.10x B=1): B=4" in onsets
    assert "TTFF p50 regression onset (>1.25x B=1): B=8" in onsets
    assert "aggregate fps gain onset (<10% over previous B): B=8" in onsets
    assert "first B with all_realtime=false: B=16" in onsets
    assert "max realtime B (all_realtime and not delivery_bound): B=4" in onsets

    exit_code = module["main"]([str(SUMMARY_SCRIPT), str(out_dir)])
    assert exit_code == 0
    summary_path = out_dir / "summary.md"
    assert summary_path.exists()
    table = summary_path.read_text()
    assert "| B |" in table
    assert "realtime streams" in table
    assert "on-time chunk % (min)" in table
    assert "15/16" in table
    assert "90.0" in table
    assert "Regression onsets" in table
    printed = capsys.readouterr().out
    assert "max realtime B" in printed


def test_summary_skips_error_status_files_gracefully(tmp_path, capsys):
    out_dir = _write_sweep(tmp_path)
    (out_dir / "b32.json").write_text(
        json.dumps({"status": "error", "error": "RuntimeError: server never became healthy"})
    )
    module = runpy.run_path(str(SUMMARY_SCRIPT))

    rows, notes = module["_load_rows"](out_dir)
    assert [row["b"] for row in rows] == [1, 2, 4, 8, 16]
    assert len(notes) == 1
    assert "b32" in notes[0] and "error" in notes[0]


def test_summary_reports_missing_out_dir_without_a_b1_baseline(tmp_path):
    module = runpy.run_path(str(SUMMARY_SCRIPT))
    artifact = _concurrent_artifact(
        ttff_p50_ms=55.0, gap_p50_median_ms=21.0, aggregate_fps=19.0,
        all_realtime=True, delivery_bound=False, gpu_peak_mib=1500.0,
    )
    (tmp_path / "b2.json").write_text(json.dumps(artifact))

    rows, notes = module["_load_rows"](tmp_path)
    assert notes == []
    onsets = module["_onsets"](rows)
    assert onsets[0] == "no B=1 row: gap/TTFF onsets relative to B=1 cannot be computed"
