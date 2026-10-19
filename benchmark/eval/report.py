"""Results files to a markdown table (mean +- std over repeats) and PNG charts.

The table is what goes into a PR description; the charts plot each metric
against concurrency (or arrival rate), one line per label, with the repeat
std as error bars. matplotlib is only needed for charts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from benchmark.eval.schema import RunResult
from benchmark.eval.stats import fmt_mean_std


@dataclass(frozen=True)
class Col:
    metric: str
    header: str
    scale: float = 1.0
    digits: int = 2


COLUMNS: dict[str, list[Col]] = {
    "tts": [
        Col("ttfa_ms_p50", "TTFA p50 (ms)", digits=0),
        Col("ttfa_ms_p90", "TTFA p90 (ms)", digits=0),
        Col("rtf_mean", "RTF", digits=3),
        Col("audio_s_per_s", "audio s/s", digits=1),
        Col("stall_rate", "stalled req %", 100, 1),
        Col("wer", "WER %", 100, 2),
        Col("sim_mean", "SIM", digits=3),
        Col("utmos_mean", "UTMOS", digits=2),
        Col("audio_s_mean", "audio s/req", digits=2),
        Col("errors", "errors", digits=0),
    ],
    "asr": [
        Col("wer", "WER %", 100, 2),
        Col("e2e_s_p50", "latency p50 (s)", digits=3),
        Col("e2e_s_p90", "latency p90 (s)", digits=3),
        Col("rtfx", "RTFx", digits=1),
        Col("req_per_s", "req/s", digits=2),
        Col("errors", "errors", digits=0),
    ],
    "extra": [
        Col("wer_utt_mean", "WER utt-mean %", 100, 2),
        Col("arrival_lateness_s_p99", "arrival late p99 (s)", digits=3),
        Col("slot_wait_s_p99", "slot wait p99 (s)", digits=3),
        Col("ttfa_ms_p99", "TTFA p99 (ms)", digits=0),
        Col("first_text_s_p50", "first text p50 (s)", digits=3),
    ],
    "duplex": [
        Col("response_latency_s_p50", "response p50 (s)", digits=2),
        Col("text_latency_s_p50", "text p50 (s)", digits=2),
        Col("take_turn_rate", "took turn %", 100, 0),
        Col("stop_latency_s_p50", "stop p50 (s)", digits=2),
        Col("yield_rate", "yielded %", 100, 0),
        Col("wall_ms_per_frame_p50", "ms/frame p50", digits=1),
        Col("on_time_rate", "on time %", 100, 0),
        Col("errors", "errors", digits=0),
    ],
}


def load_results(paths: Iterable[str | Path]) -> list[RunResult]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    return [RunResult.load(f) for f in files]


def load_point(r: RunResult) -> str:
    load = r.config.get("load", {})
    if load.get("mode") == "open":
        return f"{load.get('rate'):g} req/s"
    return f"c={load.get('concurrency')}"


def x_value(r: RunResult) -> float:
    load = r.config.get("load", {})
    return float(load.get("rate") if load.get("mode") == "open" else load.get("concurrency"))


def _short_gpu(r: RunResult) -> str:
    gpus = r.meta.get("machine", {}).get("gpus") or []
    if not gpus:
        return "no GPU"
    name = gpus[0].get("name", "GPU")
    return f"{len(gpus)}x {name}" if len(gpus) > 1 else name


def provenance(results: list[RunResult]) -> str:
    """One line: commits, GPUs, repeats, so a table never travels without its context."""
    commits = sorted({(r.meta.get("git", {}).get("commit") or "unknown")[:10] for r in results})
    dirty = any(r.meta.get("git", {}).get("dirty") for r in results)
    gpus = sorted({_short_gpu(r) for r in results})
    reps = sorted({r.config.get("repeats") for r in results})
    return (
        f"commit {', '.join(commits)}{' (dirty)' if dirty else ''}; {', '.join(gpus)}; "
        f"{'/'.join(map(str, reps))} repeats, mean +- std over repeats"
    )


def render_table(results: list[RunResult], columns: list[Col] | None = None) -> str:
    """A markdown table per suite, rows sorted by label then load."""
    out: list[str] = []
    by_suite: dict[str, list[RunResult]] = {}
    for r in results:
        by_suite.setdefault(r.suite, []).append(r)
    for suite, rs in by_suite.items():
        cols = columns or COLUMNS.get(suite) or [Col(k, k) for k in rs[0].summary]
        cols = [c for c in cols if any((r.summary.get(c.metric) or {}).get("n") for r in rs)]
        rs = sorted(rs, key=lambda r: (r.label, x_value(r)))
        out.append(f"**{suite}**: {provenance(rs)}\n")
        out.append("| label | load | n | " + " | ".join(c.header for c in cols) + " |")
        out.append("|---|---|---|" + "---|" * len(cols))
        for r in rs:
            n = r.repeats[0].metrics.get("requests") if r.repeats else None
            cells = [fmt_mean_std(r.summary.get(c.metric), c.digits, c.scale) for c in cols]
            out.append(f"| {r.label} | {load_point(r)} | {n} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def render_chart(results: list[RunResult], metrics: list[str], path: str | Path, title: str | None = None) -> Path:
    """One subplot per metric against concurrency/rate, a line per label, std error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    headers = {c.metric: (c.header, c.scale) for cols in COLUMNS.values() for c in cols}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 3.8), squeeze=False)
    open_loop = any(r.config.get("load", {}).get("mode") == "open" for r in results)
    labels = sorted({r.label for r in results})
    for ax, metric in zip(axes[0], metrics, strict=True):
        header, scale = headers.get(metric, (metric, 1.0))
        for label in labels:
            pts = sorted((x_value(r), r.summary.get(metric) or {}) for r in results if r.label == label)
            pts = [(x, s) for x, s in pts if s.get("mean") is not None]
            if not pts:
                continue
            ax.errorbar([x for x, _ in pts], [s["mean"] * scale for _, s in pts],
                        yerr=[(s.get("std") or 0.0) * scale for _, s in pts], marker="o", capsize=3, label=label)
        ax.set_xlabel("arrival rate (req/s)" if open_loop else "concurrency")
        ax.set_ylabel(header)
        ax.set_title(header)
        xs = sorted({x_value(r) for r in results})
        if not open_loop:
            ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [f"{x:g}" for x in xs])
        ax.minorticks_off()
        means = [s["mean"] * scale for r in results if (s := r.summary.get(metric) or {}).get("mean")]
        if means and min(means) > 0 and max(means) / min(means) > 20:  # lines an order of magnitude apart
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.set_ylim(bottom=0)  # differences read at their true size
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title or provenance(results), fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
