"""Build the BENCHMARK_PROTOCOL.md TTS table from ``benchmark/run_tts_benchmark.sh`` runs.

    python benchmark/tts_table.py <out_root> --systems mstar:"M* (abc1234)" kfa:"Kokoro-FastAPI 0.9.1-rc1"

Reads ``<out_root>/<label>/c<C>_r<R>/results.json`` (repeats are reduced by their
median) and, when present, ``<out_root>/wer_<label>/summary.json`` for the WER
column. Prints a markdown table: time to first audio p50/p95 at concurrency 1,
RTF p50 at concurrency 1, audio seconds generated per second at 1/8/32, WER.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _median(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _fmt(value: float | None, digits: int = 3, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def collect(root: Path, label: str) -> dict[int, dict[str, float | None]]:
    """concurrency -> median over repeats of each statistic."""
    by_conc: dict[int, list[dict]] = {}
    for path in sorted((root / label).glob("c*_r*/results.json")):
        concurrency = int(path.parent.name.split("_")[0][1:])
        by_conc.setdefault(concurrency, []).append(json.loads(path.read_text()))
    table = {}
    for concurrency, runs in sorted(by_conc.items()):
        aggs = [r.get("aggregate", {}) for r in runs]
        ttfa = [(a.get("ttft_s") or {}).get("audio") or {} for a in aggs]
        rtf = [a.get("rtf") or {} for a in aggs]
        table[concurrency] = {
            "ttfa_p50": _median([t.get("p50") for t in ttfa]),
            "ttfa_p95": _median([t.get("p95") for t in ttfa]),
            "rtf_p50": _median([r.get("p50") for r in rtf]),
            "audio_s_per_s": _median([a.get("audio_seconds_throughput") for a in aggs]),
            "completed": _median([r.get("completed") for r in runs]),
            "failed": _median([r.get("failed") for r in runs]),
            "repeats": len(runs),
        }
    return table


def wer_of(root: Path, label: str) -> float | None:
    path = root / f"wer_{label}" / "summary.json"
    return json.loads(path.read_text())["wer"] if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_root")
    parser.add_argument("--systems", nargs="+", required=True, help="label:display name, e.g. mstar:'M* (abc1234)'")
    args = parser.parse_args()
    root = Path(args.out_root)

    print("| System (version) | TTFA p50 / p95 (s, c=1) | RTF p50 (c=1) | audio-s/s c=1 | c=8 | c=32 | WER | notes |")
    print("|---|---|---|---|---|---|---|---|")
    for spec in args.systems:
        label, _, name = spec.partition(":")
        table = collect(root, label)
        c1 = table.get(1, {})
        cells = [
            name or label,
            f"{_fmt(c1.get('ttfa_p50'))} / {_fmt(c1.get('ttfa_p95'))}",
            _fmt(c1.get("rtf_p50"), 4),
            *[_fmt(table.get(c, {}).get("audio_s_per_s"), 1) for c in (1, 8, 32)],
            _fmt(wer_of(root, label), 4),
            f"{c1.get('repeats', 0)} repeats; failed {_fmt(c1.get('failed'), 0)}",
        ]
        print("| " + " | ".join(str(c) for c in cells) + " |")


if __name__ == "__main__":
    main()
