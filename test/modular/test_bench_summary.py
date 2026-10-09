"""The benchmark table renderer over synthetic ``bench_images.py`` results."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from benchmark.flux2_klein.summarize_bench import load_fidelity, load_results, model_key, render_table  # noqa: E402


def _latency(tag, model, median, p95, vram, observed=None):
    return {"tag": tag, "model": model, "mode": "latency", "size": "1024x1024", "steps": 4, "n": 20,
            "median_s": median, "p95_s": p95, "peak_vram_mib": vram, "observed_output_format": observed}


def _throughput(tag, model, rates, vram):
    return {"tag": tag, "model": model, "mode": "throughput", "size": "1024x1024", "steps": 4, "by_concurrency": {
        str(c): {"concurrency": c, "images": 32, "repeats": 3, "images_per_s": r, "images_per_s_min": r - 0.1,
                 "images_per_s_max": r + 0.1, "peak_vram_mib": vram} for c, r in rates.items()}}


def test_table_groups_latency_and_throughput_by_system(tmp_path):
    files = []
    for name, data in {
        "a_lat": _latency("mstar", "flux2_klein", 0.4, 0.45, 20000),
        "a_thr": _throughput("mstar", "flux2_klein", {4: 5.0, 8: 6.0, 16: 6.5}, 30000),
        "b_lat": _latency("sglang", "black-forest-labs/FLUX.2-klein-4B", 0.5, 0.6, 25000, observed="jpeg"),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        files.append(path)
    table = render_table(load_results(files))
    lines = table.splitlines()
    assert lines[0].startswith(
        "| System | model | size / steps | output | B=1 latency median | p95 | images/s @4",
    )
    mstar_row = next(line for line in lines if line.startswith("| mstar |"))
    assert (
        "server default | 0.400 s | 0.450 s | 5.00 (4.90-5.10) | 6.00 (5.90-6.10) | 6.50 (6.40-6.60) | 29.3 GiB"
        in mstar_row
    )
    assert "n=20; 32 images x 3 repeats per level" in mstar_row
    sglang_row = next(line for line in lines if line.startswith("| sglang |"))
    assert "| n/a | n/a | n/a | 24.4 GiB |" in sglang_row  # no throughput file: nothing invented
    assert "| jpeg (default) |" in sglang_row  # the format the server actually returned, unrequested


def test_image_format_detection():
    from benchmark.flux2_klein.bench_images import image_format

    assert image_format(b"\x89PNG\r\n\x1a\n" + b"0" * 8) == "png"
    assert image_format(b"\xff\xd8\xff\xe0" + b"0" * 8) == "jpeg"
    assert image_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert image_format(b"GIF89a") == "unknown" and image_format(None) is None


def test_fidelity_column_reads_the_psnr_files_next_to_the_results(tmp_path):
    files = []
    for name, data in {
        "a_lat": _latency("diffusers", "black-forest-labs/FLUX.2-klein-4B", 0.6, 0.62, 20000, observed="png"),
        "b_lat": _latency("sglang", "black-forest-labs/FLUX.2-klein-9B", 0.7, 0.72, 40000, observed="png"),
        "c_lat": _latency("mstar", "flux2_klein", 0.4, 0.45, 20000, observed="png"),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        files.append(path)
    (tmp_path / "fidelity_diffusers_flux2_klein.json").write_text(json.dumps(
        {"values": {}, "missing": [], "summary": {"n": 20, "exact": 20, "min": None, "median": None, "max": None}},
    ))
    (tmp_path / "fidelity_sglang_flux2_klein_9b.json").write_text(json.dumps(
        {"values": {}, "missing": [], "summary": {"n": 20, "exact": 0, "min": 11.2, "median": 12.6, "max": 14.0}},
    ))
    assert model_key("black-forest-labs/FLUX.2-klein-9B") == "flux2_klein_9b"
    assert model_key("Tongyi-MAI/Z-Image-Turbo") == "z_image_turbo"
    assert model_key("flux2_klein") == "flux2_klein"
    fidelity = load_fidelity(files)
    assert set(fidelity) == {("diffusers", "flux2_klein"), ("sglang", "flux2_klein_9b")}
    lines = render_table(load_results(files), fidelity=fidelity).splitlines()
    assert "| peak VRAM | PSNR vs diffusers | notes |" in lines[0]
    assert "| exact (n=20) |" in next(line for line in lines if line.startswith("| diffusers |"))
    assert "| 12.6 dB (min 11.2, n=20) |" in next(line for line in lines if line.startswith("| sglang |"))
    assert "| n/a | n=20 |" in next(line for line in lines if line.startswith("| mstar |"))  # no file: nothing invented


def test_python_api_rows_note_the_in_process_engine_and_pipeline_only_numbers(tmp_path):
    lat = {**_latency("diffusers_compile", "black-forest-labs/FLUX.2-klein-4B", 0.642, 0.722, 56000, observed="png"),
           "engine": "diffusers", "engine_version": "diffusers 0.40.0", "compile": True, "pipe_median_s": 0.311,
           "batching": "batched call: list of prompts"}
    thr = _throughput("diffusers_compile", "black-forest-labs/FLUX.2-klein-4B", {4: 1.65, 8: 1.64, 16: 1.64}, 56000)
    thr.update({"engine": "diffusers", "engine_version": "diffusers 0.40.0", "compile": True})
    for c in ("4", "8"):
        thr["by_concurrency"][c]["images_per_s_pipeline"] = 3.36
    thr["by_concurrency"]["16"] = {"concurrency": 16, "images": 32, "error": "OutOfMemoryError: CUDA out of memory"}
    files = []
    for name, data in {"lat": lat, "thr": thr}.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        files.append(path)
    lines = render_table(load_results(files)).splitlines()
    row = next(line for line in lines if line.startswith("| diffusers_compile |"))
    assert "| 1.65 (1.55-1.75) | 1.64 (1.54-1.74) | n/a |" in row  # the failed level is n/a, not invented
    assert "in-process diffusers 0.40.0 + torch.compile; pipeline only 0.311 s; pipeline-only img/s 3.36 / 3.36" in row
    assert "failed @16: OutOfMemoryError" in row and "batched call: list of prompts" in row
