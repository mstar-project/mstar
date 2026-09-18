"""The benchmark table renderer over synthetic ``bench_images.py`` results."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from benchmark.flux2_klein.summarize_bench import load_results, render_table  # noqa: E402


def _latency(tag, model, median, p95, vram):
    return {"tag": tag, "model": model, "mode": "latency", "size": "1024x1024", "steps": 4, "n": 20,
            "median_s": median, "p95_s": p95, "peak_vram_mib": vram}


def _throughput(tag, model, rates, vram):
    return {"tag": tag, "model": model, "mode": "throughput", "size": "1024x1024", "steps": 4, "by_concurrency": {
        str(c): {"concurrency": c, "images": 32, "repeats": 3, "images_per_s": r, "images_per_s_min": r - 0.1,
                 "images_per_s_max": r + 0.1, "peak_vram_mib": vram} for c, r in rates.items()}}


def test_table_groups_latency_and_throughput_by_system(tmp_path):
    files = []
    for name, data in {
        "a_lat": _latency("mstar", "flux2_klein", 0.4, 0.45, 20000),
        "a_thr": _throughput("mstar", "flux2_klein", {4: 5.0, 8: 6.0, 16: 6.5}, 30000),
        "b_lat": _latency("sglang", "black-forest-labs/FLUX.2-klein-4B", 0.5, 0.6, 25000),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        files.append(path)
    table = render_table(load_results(files))
    lines = table.splitlines()
    assert lines[0].startswith("| System | model | size / steps | output | B=1 latency median | p95 | images/s @4")
    mstar_row = next(line for line in lines if line.startswith("| mstar |"))
    assert "server default | 0.400 s | 0.450 s | 5.00 (4.90-5.10) | 6.00 (5.90-6.10) | 6.50 (6.40-6.60) | 29.3 GiB" in mstar_row
    assert "n=20; 32 images x 3 repeats per level" in mstar_row
    sglang_row = next(line for line in lines if line.startswith("| sglang |"))
    assert "| n/a | n/a | n/a | 24.4 GiB |" in sglang_row  # no throughput file: nothing invented
