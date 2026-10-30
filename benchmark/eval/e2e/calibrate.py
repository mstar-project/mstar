"""Cross-process noise floors from A/A (base == head) ``compare`` runs.

The passes inside one server session underestimate the spread between two server processes (kernel selection and
batching differ per process). Every A/A run records, per row, metric and GPU type, how far two processes of the
same code drifted; the report then requires a delta to clear ``report.FLOOR_FACTOR`` times the largest recent drift
before it calls the row faster or a regression (``report.floor_from``: the last ``FLOOR_WINDOW`` drifts, without a
lone drift far above the others, so one noisy A/A run cannot widen a floor for good). The same runs record how far
each row's correctness number moved (WER points, identical-text share, audio length, PSNR ...), and the report
widens a manifest limit that this noise alone would trip (``report.correct_limit``).

``calibration.yaml`` next to this file is the shared baseline; ``--calibration PATH`` keeps a private one (YAML,
or JSON for a ``.json`` path).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from benchmark.eval.e2e import report
from benchmark.eval.e2e.manifest import Model

DEFAULT = Path(__file__).with_name("calibration.yaml")
HEADER = ("# A/A noise floors and correctness swings per GPU type (benchmark/eval/e2e/calibrate.py); written by\n"
          "# `python -m benchmark.eval calibrate`.\n")


def load(path: str | Path = DEFAULT) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text()
    return (json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)) or {}


def save(cal: dict, path: str | Path = DEFAULT) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(cal, indent=1, sort_keys=True) + "\n")
    else:
        path.write_text(HEADER + yaml.safe_dump(cal, sort_keys=True, default_flow_style=None, width=120))


def floor_pct(cal: dict, key: str) -> float | None:
    return report.floor_from((cal.get(key) or {}).get("deltas") or [])


def noise_row(r: report.Row) -> bool:
    """Whether a row's perf drift is A/A noise: correct, finished and without request errors on either arm. The
    timings of a failed or errored row (requests that errored, a server that died, a job cut short) would widen the
    floor until a real regression of that size reads as noise."""
    return r.correct == "pass" and not any(r.errors.values())


def add_run(run_dir: Path, models: list[Model], path: str | Path = DEFAULT,
            excluded: dict[str, int] | None = None) -> int:
    """Fold one A/A run into the calibration; returns the number of entries recorded (0 when already folded in).
    Failed or errored rows stay out of the perf floors; ``excluded`` gets how many per model.

    Perf drifts (``_runs``), correctness swings (``_runs_correct``) and quality-score swings (``_runs_quality``) are
    tracked apart, so a run folded in before one kind was recorded can still add it."""
    path = Path(path)
    run = json.loads((run_dir / "run.json").read_text())
    if run["base"]["sha"] != run["head"]["sha"]:
        raise ValueError(f"{run_dir.name} is not an A/A run (base and head are different commits)")
    gpu = run.get("gpu", "unknown")
    cal = load(path)
    seen, seen_c = set(cal.get("_runs", [])), set(cal.get("_runs_correct", []))
    seen_q = set(cal.get("_runs_quality", []))
    if run["id"] in seen and run["id"] in seen_c and run["id"] in seen_q:
        return 0
    n = 0
    if run["id"] not in seen:
        # The one-session drift that triggered an ABBA rerun is cross-process noise too: keep it beside the rerun's.
        first = report.build_rows(run_dir, models, calibration={}, reruns=False)
        reruns = [r for r in report.build_rows(run_dir, models, calibration={}) if r.rerun]
        left_out = {(r.model, r.case) for r in first + reruns if not noise_row(r)}
        if excluded is not None:
            for model, _ in left_out:
                excluded[model] = excluded.get(model, 0) + 1
        for r in first + reruns:
            if (r.model, r.case) in left_out:
                continue
            for p in [r.latency, r.throughput, *r.extra]:
                if p and p.delta_pct is not None:
                    key = report.calibration_key(gpu, r.model, r.case, p.metric, p.conc)
                    cal.setdefault(key, {"deltas": []})["deltas"].append(round(p.delta_pct, 3))
                    n += 1
    for m in models:
        for name in (m.name, m.name + report.RERUN_TAG):
            f = run_dir / "results" / name / "compare.json"
            compare = json.loads(f.read_text()) if f.exists() else {}
            for c in m.cases:
                if run["id"] not in seen_c:
                    swings = [s for s in (report.aa_swing(c.check["kind"], x) for x in compare.get(c.id, []))
                              if s is not None and math.isfinite(s)]
                    if swings:
                        key = report.correct_key(gpu, m.name, c.id)
                        entry = cal.setdefault(key, {"kind": c.check["kind"], "swings": []})
                        entry["swings"] += [round(s, 4) for s in swings]
                        n += 1
                if run["id"] not in seen_q:
                    for x in compare.get(c.id, []):
                        for kind, s in report.quality_swings(x).items():
                            if kind in c.quality and math.isfinite(s):
                                key = report.quality_key(gpu, m.name, c.id, kind)
                                cal.setdefault(key, {"kind": kind, "swings": []})["swings"].append(round(s, 4))
                                n += 1
    cal["_runs"] = sorted(seen | {run["id"]})
    cal["_runs_correct"] = sorted(seen_c | {run["id"]})
    cal["_runs_quality"] = sorted(seen_q | {run["id"]})
    save(cal, path)
    return n
