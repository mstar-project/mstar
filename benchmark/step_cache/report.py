"""A run's records as a report: ``report.md`` (the table, the frontier, the default's verdict), ``summary.json`` and
``pareto.svg``, next to the ``records.jsonl`` the run wrote as it went."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

from benchmark.step_cache.pareto import (
    METRICS,
    DefaultVerdict,
    Point,
    aggregate,
    check_default,
    frontier,
    pareto_svg,
)

RECORDS = "records.jsonl"


class RunConfig(NamedTuple):
    model: str
    backend: str
    size: str
    frames: int
    steps: int

    @property
    def line(self) -> str:
        return f"{self.model}, {self.size} x {self.frames} frames, {self.steps} steps"


class Report(NamedTuple):
    path: Path
    points: list[Point]
    frontier: list[Point]
    verdict: DefaultVerdict


def load_records(out_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(out_dir) / RECORDS
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_record(out_dir: str | Path, record: Mapping[str, Any]) -> None:
    """One line per clip, flushed as it is written, so a killed run keeps what it finished."""
    with open(Path(out_dir) / RECORDS, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def _fmt(value: float | None, digits: int, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


def _signed(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def floor_line(records: Iterable[Mapping[str, Any]]) -> str:
    """The dense run-to-run floor: a second dense clip of the first prompts against the first."""
    floor = [r for r in records if r.get("kind") == "floor"]
    if not floor:
        return "not measured"
    same = sum(bool(r.get("identical")) for r in floor)
    if same == len(floor):
        return f"bit-identical on {same} of {len(floor)} prompts"
    worst = min(r["psnr"] for r in floor if not r.get("identical"))
    return f"bit-identical on {same} of {len(floor)} prompts; the others at PSNR {worst:.1f} dB or better"


_HEADER = (
    "method", "s/video", "speedup", "steps computed", "branch forwards", "PSNR dB", "worst frame dB", "SSIM", "LPIPS",
    "CLIP delta", "flicker (dense)", "flicker excess", "motion PSNR dB", "frontier",
)


def _dense_row(records: Sequence[Mapping[str, Any]], dense: str) -> list[str] | None:
    rows = [r for r in records if r.get("kind") == "dense" and r["method"] == dense]
    if not rows:
        return None

    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    flicker = mean("flicker")
    return [
        dense, _fmt(mean("seconds"), 2), "1.00x", _fmt(mean("computed_steps"), 0), _fmt(mean("branch_forwards"), 1),
        "-", "-", "-", "-", "-", _fmt(flicker, 4), "-", "-", "reference",
    ]


def _row(point: Point, on_frontier: bool) -> list[str]:
    mark = "default" if point.is_default else ""
    if on_frontier:
        mark = f"{mark}, frontier".lstrip(", ")
    return [
        point.name, _fmt(point.seconds, 2), _fmt(point.speedup, 2, "x"), _fmt(point.computed_steps, 0),
        _fmt(point.branch_forwards, 1), _fmt(point.psnr, 2), _fmt(point.psnr_min, 2), _fmt(point.ssim, 3),
        _fmt(point.lpips, 3), _signed(point.clip_delta, 4),
        f"{_fmt(point.flicker, 4)} ({_fmt(point.flicker_ref, 4)})", _signed(point.flicker_excess, 3),
        _fmt(point.motion_psnr, 2), mark,
    ]


def _env_line(env: Mapping[str, Any]) -> str:
    return "; ".join(f"{key} {value}" for key, value in env.items() if value not in (None, ""))


def write_report(
    out_dir: str | Path, records: Sequence[Mapping[str, Any]], *, env: Mapping[str, Any], config: RunConfig,
    metric: str = "psnr", default: str | None = None, notes: Sequence[str] = (), dense: str = "dense",
) -> Report:
    """Write the report files for ``records`` into ``out_dir`` and return what they say. ``default`` names the
    method the frontier check is about (else the one the records flag)."""
    out_dir = Path(out_dir)
    axis = METRICS[metric]
    points = aggregate(records, dense)
    if default is not None:
        points = [replace(p, is_default=p.name == default) for p in points]
    edge = frontier(points, axis)
    on_edge = {p.name for p in edge}
    verdict = check_default(points, axis)
    prompts = sorted({r["prompt_index"] for r in records if r.get("kind") == "dense"})

    lines = [
        f"# Step cache: {config.line}",
        "",
        f"**Hardware:** {_env_line(env) or 'not recorded'}",
        f"**Config:** {config.line}; {len(prompts)} prompts (seeds 1000 + index), each method against the "
        f"{config.backend} backend's own dense clip of the same prompt and seed.",
        f"**Dense run-to-run floor:** {floor_line(records)}.",
        f"**Default:** {'PASS' if verdict.ok else 'FAIL'}: {verdict.reason}.",
    ]
    lines += [f"**Note:** {note}" for note in notes]
    lines += ["", "| " + " | ".join(_HEADER) + " |", "|" + "---|" * len(_HEADER)]
    dense_row = _dense_row(records, dense)
    if dense_row is not None:
        lines.append("| " + " | ".join(dense_row) + " |")
    lines += ["| " + " | ".join(_row(p, p.name in on_edge)) + " |" for p in points]
    lines += [
        "",
        "Means over prompts; worst frame is the lowest per-frame PSNR of any prompt. CLIP delta is the clip's "
        "text-video score minus the dense clip's. Flicker is VBench's temporal-flickering form (the dense clip's in "
        "brackets), flicker excess the second temporal difference's energy over the dense clip's, minus 1, and "
        "motion PSNR compares the two clips' frame differences.",
        "",
        f"## Frontier ({axis.key})",
        "",
    ]
    lines += [
        f"- `{p.name}`: {p.speedup:.2f}x at {getattr(p, axis.key):.{axis.digits}f} ({p.label})" for p in edge
    ] or ["- no scored method"]
    lines += ["", "## Methods", ""]
    specs = {r["method"]: r.get("step_cache") for r in records if r.get("kind") == "method"}
    lines += [f"- `{p.name}`: {p.label}. `step_cache: {json.dumps(specs.get(p.name))}`" for p in points]
    worst = [p for p in points if p.worst_prompt is not None]
    if worst:
        lines += ["", "## Worst prompt per method", ""]
        lines += [f"- `{p.name}`: prompt {p.worst_prompt} at {p.worst_prompt_psnr:.2f} dB" for p in worst]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")

    scored = [p for p in points if getattr(p, axis.key) is not None]
    if scored:
        where = [str(env[key]) for key in ("gpu", "device") if env.get(key)][:1]
        where += [f"torch {env['torch']}"] if env.get("torch") else []
        subtitle = "; ".join([config.line, f"{len(prompts)} prompts", *where])
        (out_dir / "pareto.svg").write_text(pareto_svg(points, axis, subtitle=subtitle))
    summary = {
        "config": config._asdict(), "env": dict(env), "metric": axis.key, "prompts": prompts,
        "floor": floor_line(records), "notes": list(notes),
        "default": verdict._asdict(), "frontier": [p.name for p in edge], "points": [p.as_dict() for p in points],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return Report(path, points, edge, verdict)
