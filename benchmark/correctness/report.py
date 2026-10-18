"""Verdicts and the per-run ``report.md`` / ``report.json``."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# Best to worst. A run gets the verdict of its worst check.
VERDICTS = ("PASS", "WARN", "NO_REFERENCE", "FAIL", "ERROR")


def worst(*verdicts: str) -> str:
    return max(verdicts, key=VERDICTS.index) if verdicts else "PASS"


@dataclass
class Check:
    """One check with its value, verdict and the rule it was judged by."""

    name: str
    value: float | int | str | None
    verdict: str
    rule: str
    detail: str = ""


@dataclass
class FamilyScore:
    family: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)  # per item
    artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return worst(*(c.verdict for c in self.checks)) if self.checks else "ERROR"

    def check(self, name, value, verdict, rule, detail="") -> None:
        self.checks.append(Check(name, value, verdict, rule, detail))


def band(value: float | None, pass_if, warn_if=None) -> str:
    """PASS, WARN or FAIL from two predicates. None or NaN is ERROR."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "ERROR"
    if pass_if(value):
        return "PASS"
    if warn_if is not None and warn_if(value):
        return "WARN"
    return "FAIL"


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        return f"{v:.4g}"
    return str(v)


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return str(o)


def write_reports(run_dir: Path, header: dict, score: FamilyScore | None, error: str | None = None,
                  max_rows: int = 40) -> str:
    verdict = "ERROR" if score is None else score.verdict
    payload = {
        "verdict": verdict,
        "error": error,
        **header,
        "family": score.family if score else None,
        "checks": [c.__dict__ for c in score.checks] if score else [],
        "metrics": score.metrics if score else {},
        "notes": score.notes if score else [],
        "artifacts": score.artifacts if score else [],
        "rows": score.rows if score else [],
    }
    (run_dir / "report.json").write_text(json.dumps(payload, indent=1, default=_json_default, ensure_ascii=False))

    lines = [f"# Correctness: `{header.get('config')}` — **{verdict}**", ""]
    lines += [f"- run `{header.get('run_id')}`, {header.get('started')} to {header.get('finished')}",
              f"- family {score.family if score else '?'}, set `{header.get('set')}` with {header.get('n_items')} "
              f"items, {header.get('restarts')} server instance(s), concurrency {header.get('concurrency')}"]
    v = header.get("versions") or {}
    stack = v.get("stack", {})
    lines += [f"- code `{v.get('git_branch')}` at `{v.get('git_describe')}` in `{v.get('worktree')}`"
              f"{' (DIRTY)' if v.get('git_dirty') else ''}, server imports it: {v.get('serves_worktree')}",
              f"- torch {stack.get('torch')}, hip {stack.get('hip')}, {stack.get('devices')} ({stack.get('arch')}) on "
              f"{stack.get('host')}, transformers {stack.get('transformers')}, aiter {stack.get('aiter')}, "
              f"flashinfer {stack.get('flashinfer')}",
              f"- config overrides: {header.get('overrides') or 'none'}",
              f"- protocol: {header.get('protocol')}",
              f"- reference: {header.get('reference')}"
              + (" (BUILD FAILED, see report.json reference_build_error)" if header.get("reference_build_error")
                 else ""),
              f"- server startup in seconds, per instance (includes any cold compile): {header.get('startup_s')}", ""]
    if error:
        lines += ["## Error", "", "```", error[-6000:], "```", ""]
    if score:
        lines += ["## Checks", "", "| check | value | verdict | rule |", "|---|---|---|---|"]
        for c in score.checks:
            lines.append(f"| {c.name} | {_fmt(c.value)} | {c.verdict} | {c.rule}"
                         + (f" ({c.detail})" if c.detail else "") + " |")
        lines.append("")
        if score.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in score.notes] + [""]
        if score.metrics:
            lines += ["## Metrics", "", "```json", json.dumps(score.metrics, indent=1, default=_json_default,
                                                              ensure_ascii=False)[:20000], "```", ""]
        if score.rows:
            keys = [k for k in score.rows[0] if not isinstance(score.rows[0][k], (dict, list))][:9]
            lines += [f"## Per item (first {min(max_rows, len(score.rows))} of {len(score.rows)}, the rest are in "
                      "report.json)", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
            for row in score.rows[:max_rows]:
                lines.append("| " + " | ".join(_fmt(row.get(k)).replace("|", "/").replace("\n", " ")[:80]
                                               for k in keys) + " |")
            lines.append("")
        if score.artifacts:
            lines += ["## Artifacts", ""] + [f"- `{a}`" for a in score.artifacts] + [""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict
