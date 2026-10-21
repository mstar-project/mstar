"""Tables, charts and the judge agreement report for ``fdb`` results.

The tables are the two of the reference runner's ``RESULTS.md`` (a row per category and ``all``):
pairs, timed overlap sessions, pooled stop and response latency, label shares and judged
pairs, one table per judge; then the semantic quality and coverage table. Cells are mean
+- sample std over repeats; counts show one value or a ``low-high`` range.

``agreement`` pools two judges' labels over every repeat (each repeat's pair is one
observation) into raw agreement, Cohen's kappa and a confusion matrix, plus the per-repeat
mean +- std that the results files already hold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.eval import stats
from benchmark.eval.llm_judge import C_LABELS
from benchmark.eval.report import provenance
from benchmark.eval.schema import RunResult
from benchmark.eval.suites.duplex_fdb import GROUPS, SEMANTIC_AXES, valid_label_pairs

SHORT = {"C_RESPOND": "RESPOND", "C_RESUME": "RESUME", "C_UNCERTAIN_HANDLING": "UNCERTAIN", "C_UNKNOWN": "UNKNOWN"}
# The label that should be high in each category (the reference runner's reading guide).
DESIRED = {"user_interruption": "RESPOND", "user_backchannel": "RESUME", "talking_to_other": "RESUME",
           "background_speech": "RESUME"}


def judges_of(r: RunResult) -> list[str]:
    """Judges that labeled this run, in the order they were given."""
    return [j["name"] for j in r.config.get("suite", {}).get("judges") or [] if not j.get("skipped")]


def skipped_judges(r: RunResult) -> list[tuple[str, str]]:
    return [(j["name"], j["skipped"]) for j in r.config.get("suite", {}).get("judges") or [] if j.get("skipped")]


def _cell(summary: dict | None, unit: str) -> str:
    if not summary or summary.get("mean") is None:
        return "n/a"
    if unit == "count":
        lo, hi = int(summary["min"]), int(summary["max"])
        return str(lo) if lo == hi else f"{lo}-{hi}"
    if unit == "percent":
        return stats.fmt_mean_std(summary, 1, 100) + "%"
    return stats.fmt_mean_std(summary, 3)


def _table(r: RunResult, columns: list[tuple[str, str, str]]) -> list[str]:
    lines = ["| Category | " + " | ".join(h for _, h, _ in columns) + " |", "|---|" + "---:|" * len(columns)]
    for g in GROUPS:
        if f"{g}.pairs" not in r.summary:
            continue
        lines.append(f"| {g} | " + " | ".join(_cell(r.summary.get(f"{g}.{k}"), u) for k, _, u in columns) + " |")
    return lines


def render(results: list[RunResult]) -> str:
    out: list[str] = []
    for r in results:
        judges = judges_of(r)
        sem_cfg = r.config.get("suite", {}).get("semantic") or {}
        sem = sem_cfg.get("judge")
        sem_model = (sem_cfg.get("identity") or {}).get("model")
        out.append(f"**fdb {r.label}** ({provenance([r])}). Latencies are pooled whole-file overlap intervals; "
                   "label shares are over valid judge labels.\n")
        timing = [("pairs", "Pairs", "count"), ("timed_overlap_sessions", "Timed overlap sessions", "count"),
                  ("stop_latency_s", "Stop latency (s)", "seconds"),
                  ("response_latency_s", "Response latency (s)", "seconds")]
        if not judges:
            out += _table(r, timing) + [""]
        for j in judges:
            labels = [(f"{j}.{lab}", lab, "percent") for lab in C_LABELS]
            out += [f"Judge `{j}`:", ""] + _table(r, timing + labels + [(f"{j}.judged_pairs", "Judged pairs", "count")])
            out.append("")
        for name, why in skipped_judges(r):
            out.append(f"Judge `{name}` skipped: {why}.\n")
        if sem:
            cols = [(f"semantic.{ax}_{m}", f"{ax} {m}", "percent")
                    for ax in SEMANTIC_AXES for m in ("quality", "coverage")]
            who = f"`{sem}` ({sem_model})" if sem_model else f"`{sem}`"
            out += [f"Semantic A/F/U judge {who}: quality = A/(A+F), coverage = (A+F)/N; joint fails if any axis "
                    "fails and accepts only if all three accept.", ""] + _table(r, cols) + [""]
        for i, a in enumerate(judges):
            for b in judges[i + 1:]:
                ag, ka = r.summary.get(f"agreement.{a}.{b}.agreement"), r.summary.get(f"agreement.{a}.{b}.kappa")
                out.append(f"Judge agreement `{a}` vs `{b}` per repeat: {_cell(ag, 'percent')} labels agree, "
                           f"kappa {stats.fmt_mean_std(ka, 2)}.\n")
    return "\n".join(out)


def render_chart(results: list[RunResult], path: str | Path, judges: list[str] | None = None,
                 title: str | None = None) -> Path:
    """Per-category stop and response latency (mean +- std over repeats), then the label
    shares of each judge (default: every judge of the run), one row of panels per file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = [g for g in GROUPS if g != "all"]
    per_row = [judges or judges_of(r) or [None] for r in results]
    cols = 1 + max(len(js) for js in per_row)
    fig, axes = plt.subplots(len(results), cols, figsize=(6 * cols, 4.2 * len(results)), squeeze=False)
    colors = {"C_RESPOND": "#4C72B0", "C_RESUME": "#55A868", "C_UNCERTAIN_HANDLING": "#DD8452", "C_UNKNOWN": "#C44E52"}
    for row, r, js in zip(axes, results, per_row, strict=True):
        ax = row[0]
        width = 0.38
        for k, (kind, color) in enumerate((("stop", "#8172B3"), ("response", "#937860"))):
            s = [r.summary.get(f"{c}.{kind}_latency_s") or {} for c in cats]
            xs = [i + (k - 0.5) * width for i in range(len(cats))]
            ax.bar(xs, [v.get("mean") or 0 for v in s], width, yerr=[v.get("std") or 0 for v in s], capsize=3,
                   color=color, label=f"{kind} latency")
            for x, v in zip(xs, s, strict=True):
                label = "n/a" if v.get("mean") is None else f"{v['mean']:.2f}"
                ax.text(x, (v.get("mean") or 0) + (v.get("std") or 0) + 0.03, label, ha="center", fontsize=7)
        ax.set_xticks(range(len(cats)), [c.replace("_", "\n") for c in cats], fontsize=8)
        tops = [(r.summary.get(f"{c}.{k}_latency_s") or {}) for c in cats for k in ("stop", "response")]
        ax.set_ylim(0, 1.3 * max([(t.get("mean") or 0) + (t.get("std") or 0) for t in tops] + [0.1]))
        ax.set_ylabel("seconds (pooled intervals)")
        ax.set_title(f"{r.label}: stop and response latency")
        ax.legend(fontsize=8, ncol=2, loc="upper center")
        ax.grid(axis="y", alpha=0.3)
        for ax in row[len(js) + 1:]:
            ax.set_axis_off()
        for ax, j in zip(row[1:], js, strict=False):
            if j is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no judge labels", ha="center")
                continue
            bottoms = [0.0] * len(cats)
            for lab in C_LABELS:
                vals = [100 * ((r.summary.get(f"{c}.{j}.{lab}") or {}).get("mean") or 0) for c in cats]
                ax.bar(range(len(cats)), vals, 0.6, bottom=bottoms, color=colors[lab], label=SHORT[lab])
                for i, (v, b) in enumerate(zip(vals, bottoms, strict=True)):
                    if v >= 6:
                        ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white")
                bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
            ax.set_xticks(range(len(cats)), [c.replace("_", "\n") + f"\n(want {DESIRED[c]})" for c in cats],
                          fontsize=8)
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of judged pairs")
            ax.set_title(f"{r.label}: behavior labels (judge {j})")
        row[len(js)].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title or provenance(results), fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def agreement(results: list[RunResult], a: str | None = None, b: str | None = None) -> dict[str, Any]:
    """Pooled agreement of judges ``a`` and ``b`` (default: the first two judges)."""
    if a is None or b is None:
        names = judges_of(results[0])
        if len(names) < 2:
            skipped = "; ".join(f"{n} ({w})" for n, w in skipped_judges(results[0]))
            raise SystemExit(f"agreement needs two judges' labels; this run has {names}"
                             + (f", skipped: {skipped}" if skipped else ""))
        a, b = names[:2]
    for name in (a, b):
        same = {json.dumps(_judge_identity(r, name), sort_keys=True) for r in results}
        if len(same) > 1:
            raise SystemExit(f"judge {name!r} is a different model, revision or decoding across these files; "
                             "pool only results labeled by the same judge")
    rows = [row | {"_repeat": rep.index} for r in results for rep in r.repeats for row in rep.requests]
    xs, ys = valid_label_pairs(rows, a, b)
    out: dict[str, Any] = {"a": a, "b": b, "pairs": len(rows), **stats.label_agreement(xs, ys),
                           "confusion": stats.confusion_matrix(xs, ys, list(C_LABELS)),
                           "shares": {j: {lab: (v.count(lab) / len(v) if v else None) for lab in C_LABELS}
                                      for j, v in ((a, xs), (b, ys))},
                           "per_category": {}}
    for c in (g for g in GROUPS if g != "all"):
        cx, cy = valid_label_pairs([r for r in rows if r.get("category") == c], a, b)
        out["per_category"][c] = stats.label_agreement(cx, cy)
    per_rep = [r.summary.get(f"agreement.{a}.{b}.{k}") for r in results for k in ("agreement", "kappa")]
    out["per_repeat"] = {"agreement": per_rep[0::2], "kappa": per_rep[1::2]}
    return out


def _judge_identity(r: RunResult, name: str) -> dict[str, Any] | None:
    """What decides a judge's labels in one file (``None``: the file has no such judge)."""
    j = next((j for j in r.config.get("suite", {}).get("judges") or [] if j["name"] == name), None)
    return None if j is None else {k: j.get(k) for k in ("kind", "model", "revision", "root", "behavior", "skipped")}


def render_agreement(ag: dict[str, Any]) -> str:
    a, b, n = ag["a"], ag["b"], ag["n"]
    if not n:
        return f"No pair has valid labels from both `{a}` and `{b}`."
    lines = [f"**Judge agreement: `{a}` vs `{b}`** over {n} pairs both labeled validly ({ag['pairs']} pair "
             "sessions over all repeats).", "",
             f"- Label agreement: {100 * ag['agreement']:.1f}%",
             f"- Cohen's kappa: {ag['kappa']:.2f}"]
    for k, (ra, rk) in enumerate(zip(ag["per_repeat"]["agreement"], ag["per_repeat"]["kappa"], strict=True)):
        if ra:
            lines.append(f"- File {k + 1} per repeat: {_cell(ra, 'percent')} agree, kappa {stats.fmt_mean_std(rk, 2)}")
    labels, m = ag["confusion"]["labels"], ag["confusion"]["matrix"]
    lines += ["", f"Confusion matrix (rows `{a}`, columns `{b}`):", "",
              "| | " + " | ".join(SHORT[x] for x in labels) + f" | `{a}` share |",
              "|---|" + "---:|" * (len(labels) + 1)]
    for lab, counts in zip(labels, m, strict=True):
        share = ag["shares"][a][lab]
        lines.append(f"| {SHORT[lab]} | " + " | ".join(map(str, counts)) + f" | {100 * (share or 0):.1f}% |")
    lines.append(f"| `{b}` share | " + " | ".join(f"{100 * (ag['shares'][b][x] or 0):.1f}%" for x in labels) + " | |")
    lines += ["", "| Category | pairs | agreement | kappa |", "|---|---:|---:|---:|"]
    for c, v in ag["per_category"].items():
        if v["n"]:
            lines.append(f"| {c} | {v['n']} | {100 * v['agreement']:.1f}% | {v['kappa']:.2f} |")
    return "\n".join(lines) + "\n"


def render_confusion_chart(ag: dict[str, Any], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, m = ag["confusion"]["labels"], ag["confusion"]["matrix"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(m, cmap="Blues")
    for i, counts in enumerate(m):
        for k, v in enumerate(counts):
            ax.text(k, i, str(v), ha="center", va="center", color="white" if v > max(map(max, m)) / 2 else "black")
    ax.set_xticks(range(len(labels)), [SHORT[x] for x in labels], fontsize=8)
    ax.set_yticks(range(len(labels)), [SHORT[x] for x in labels], fontsize=8)
    ax.set_xlabel(f"judge {ag['b']}")
    ax.set_ylabel(f"judge {ag['a']}")
    ax.set_title(f"{100 * ag['agreement']:.1f}% agree, kappa {ag['kappa']:.2f} (n={ag['n']})", fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
