#!/usr/bin/env python3
"""Render the benchmark protocol's image table from ``bench_images.py`` result files.

    python benchmark/flux2_klein/summarize_bench.py results/2026-09-18/*.json

Rows are grouped by ``(tag, model)``: the latency file supplies the B=1 median / p95, the
throughput file the images/s at each concurrency and the peak VRAM. Missing pieces print as
``n/a`` rather than being invented.

Seed fidelity: a ``fidelity_<tag>_<model key>.json`` written by ``psnr.py --json`` next to the
result files (``notes/gpu_fidelity.sh``) fills the "PSNR vs diffusers" column with the median and
minimum PSNR of the row's latency images against the bit-exact eager references at the same
seeds; ``exact`` means every compared image was identical.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

MODEL_KEYS = {"9b": "flux2_klein_9b", "klein": "flux2_klein", "z-image": "z_image_turbo", "z_image": "z_image_turbo"}


def load_results(paths: list[str | Path]) -> dict[tuple[str, str], dict]:
    """``{(tag, model): {"latency": ..., "throughput": ...}}`` from result JSON files."""
    rows: dict[tuple[str, str], dict] = defaultdict(dict)
    for path in paths:
        data = json.loads(Path(path).read_text())
        rows[(data["tag"], data["model"])][data["mode"]] = data
    return rows


def model_key(model: str) -> str:
    """The registry key a served model id maps to (``black-forest-labs/FLUX.2-klein-9B`` -> ``flux2_klein_9b``)."""
    lowered = model.lower()
    return next((key for needle, key in MODEL_KEYS.items() if needle in lowered), model)


def load_fidelity(paths: list[str | Path]) -> dict[tuple[str, str], dict]:
    """``{(tag, model key): psnr summary}`` from the ``fidelity_<tag>_<model key>.json`` files next to the results."""
    fidelity: dict[tuple[str, str], dict] = {}
    for directory in {Path(path).resolve().parent for path in paths}:
        for file in sorted(directory.glob("fidelity_*.json")):
            summary = json.loads(file.read_text()).get("summary") or {}
            if not summary.get("n"):
                continue
            stem = file.stem[len("fidelity_"):]
            keys = sorted(set(MODEL_KEYS.values()), key=len, reverse=True)
            key = next((k for k in keys if stem.endswith("_" + k)), None)
            if key is not None:
                fidelity[(stem[: -len(key) - 1], key)] = summary
    return fidelity


def _fmt_fidelity(summary: dict | None) -> str:
    if not summary:
        return "n/a"
    if summary.get("exact") == summary["n"]:
        return f"exact (n={summary['n']})"
    median, minimum = summary["median"], summary["min"]
    text = "exact" if math.isinf(median) else f"{median:.1f} dB"
    return f"{text} (min {'exact' if math.isinf(minimum) else f'{minimum:.1f}'}, n={summary['n']})"


def _fmt_s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} s"


def _fmt_rate(entry: dict | None) -> str:
    if entry is None or entry.get("images_per_s") is None:
        return "n/a"
    spread = ""
    if "images_per_s_min" in entry:
        spread = f" ({entry['images_per_s_min']:.2f}-{entry['images_per_s_max']:.2f})"
    return f"{entry['images_per_s']:.2f}{spread}"


def _fmt_vram(*entries: dict | None) -> str:
    peaks = [e["peak_vram_mib"] for e in entries if e and e.get("peak_vram_mib") is not None]
    return "n/a" if not peaks else f"{max(peaks) / 1024:.1f} GiB"


def render_table(
    rows: dict[tuple[str, str], dict], concurrencies: tuple[int, ...] = (4, 8, 16),
    fidelity: dict[tuple[str, str], dict] | None = None,
) -> str:
    """The protocol's markdown table for the image row set."""
    head = ["System", "model", "size / steps", "output", "B=1 latency median", "p95"]
    head += [f"images/s @{c}" for c in concurrencies] + ["peak VRAM", "PSNR vs diffusers", "notes"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (tag, model), parts in sorted(rows.items()):
        lat, thr = parts.get("latency"), parts.get("throughput")
        by_c = (thr or {}).get("by_concurrency", {})
        any_part = lat or thr or {}
        size = f"{any_part.get('size', 'n/a')} / {any_part.get('steps', 'n/a')}"
        notes = []
        if lat:
            notes.append(f"n={lat['n']}")
        if thr and by_c:
            first = next(iter(by_c.values()))
            notes.append(f"{first.get('images', '?')} images x {first.get('repeats', 1)} repeats per level")
        if any_part.get("engine"):  # pipeline_bench.py rows: in-process, no HTTP, the engine's own PNG path
            compiled = " + torch.compile" if any_part.get("compile") else ""
            notes.append(f"in-process {any_part['engine_version']}{compiled}")
            if lat and lat.get("pipe_median_s") is not None:
                notes.append(f"pipeline only {lat['pipe_median_s']:.3f} s")
            pipeline_rates = [f"{by_c[str(c)]['images_per_s_pipeline']:.2f}" for c in concurrencies
                              if str(c) in by_c and by_c[str(c)].get("images_per_s_pipeline") is not None]
            if pipeline_rates:
                notes.append("pipeline-only img/s " + " / ".join(pipeline_rates))
            failed = [str(c) for c in concurrencies if str(c) in by_c and by_c[str(c)].get("error")]
            if failed:
                notes.append("failed @" + ",".join(failed) + ": " + by_c[failed[0]]["error"].split(":")[0])
            notes.append(any_part.get("batching", ""))
        observed = next(
            (p.get("observed_output_format") for p in (lat, thr) if p and p.get("observed_output_format")), None,
        )
        requested = any_part.get("output_format")
        output = observed or requested or "server default"
        if observed and not requested:
            output = f"{observed} (default)"
        cells = [tag, model, size, output, _fmt_s(lat and lat["median_s"]), _fmt_s(lat and lat["p95_s"])]
        cells += [_fmt_rate(by_c.get(str(c))) for c in concurrencies]
        cells += [_fmt_vram(lat, *by_c.values())]
        cells += [_fmt_fidelity((fidelity or {}).get((tag, model_key(model)))), "; ".join(notes)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="bench_images.py JSON files")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()
    print(render_table(load_results(args.results), tuple(args.concurrency), load_fidelity(args.results)))


if __name__ == "__main__":
    main()
