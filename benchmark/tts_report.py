#!/usr/bin/env python3
"""Assemble the BENCHMARK_PROTOCOL.md TTS table from ``tts_speech_bench`` / ``tts_wer`` / parity JSONs.

    python -m benchmark.tts_report --results results/2026-09-18 --out results/2026-09-18/REPORT.md

Every ``*_c<N>.json`` written by ``benchmark/tts_speech_bench.py`` becomes one row
(label, concurrency, TTFA p50/p95, RTF, audio-seconds per second, errors); a
sibling ``*_c<N>_wer.json`` from ``benchmark/tts_wer.py`` fills the WER column
and ``parity_*.json`` files from ``test/qwen3-tts/parity_qwen3_tts.py`` become
the parity table. The markdown is printed and optionally written to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_rows(results: Path) -> list[str]:
    rows = []
    for path in sorted(results.glob("*_c[0-9]*.json")):
        if path.name.endswith("_wer.json"):
            continue
        report = _load(path)
        med = report["median_over_repeats"]
        wer_path = path.with_name(path.stem + "_wer.json")
        wer = f"{_load(wer_path)['wer_percent']:.2f}" if wer_path.is_file() else "n/a"
        version = report.get("engine_version") or ""
        label = report.get("label") or report["engine"]
        rows.append(
            f"| {label} {version} | {report['concurrency']} | "
            f"{med['ttfa_p50_ms']:.0f} / {med['ttfa_p95_ms']:.0f} | {med['rtf_mean']:.3f} | "
            f"{med['audio_s_per_wall_s']:.1f} | {wer} | {report['errors_total']} | "
            f"{report['num_sentences']} sentences x {report['repeats']} repeats |"
        )
    return rows


def parity_rows(results: Path) -> list[str]:
    rows = []
    for path in sorted(results.glob("parity_*.json")):
        r = _load(path)
        clone = r.get("clone") or {}
        clone_cell = (
            f"cos {clone['xvector_cosine']:.4f}, codes {clone.get('ref_code_agreement', float('nan')):.3f}"
            if clone else "-"
        )
        rows.append(
            f"| {r['repo'].split('/')[-1]} | {re.sub(r'^parity_', '', path.stem)} | {r['frames']} | "
            f"{r['talker']['argmax_agreement']:.4f} | {r['code_predictor']['argmax_agreement']:.4f} | "
            f"{r['greedy_codes']['identical_frames_before_divergence']}/{r['greedy_codes']['frames_compared']} | "
            f"{r['codec']['max_abs_diff']:.2e} | {r['audio']['max_abs_diff']:.3f} | {clone_cell} |"
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="directory with the benchmark / parity JSON files")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    results = Path(args.results)

    lines = ["## Benchmarks (H100, back to back, warmup excluded, median over repeats)", "",
             "| System (version) | concurrency | TTFA p50 / p95 ms | RTF | audio-s / s | WER % | errors | notes |",
             "|---|---|---|---|---|---|---|---|", *benchmark_rows(results), "",
             "## Parity vs qwen-tts (greedy, bf16 Talker, fp32 codec)", "",
             "| checkpoint | mode | frames | Talker argmax agreement | CodePredictor argmax agreement | "
             "identical greedy frames | codec max-abs-diff | greedy audio max-abs-diff | "
             "clone (x-vector cosine, ref-code agreement) |",
             "|---|---|---|---|---|---|---|---|---|", *parity_rows(results)]
    env = results / "environment.txt"
    if env.is_file():
        lines += ["", "## Environment", "", "```", env.read_text(encoding="utf-8").strip(), "```"]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
