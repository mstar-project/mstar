"""The best-so-far chart of a loop's headline cell, in the style of the VibeSys reports.

A round counts as kept when it passes the keep rule on the headline cell, and moves the line by
its own ratio of candidate to baseline. Those ratios come from separate servers hours apart, so
between two checkpoint sweeps the kept steps are scaled together, in log space, to end on the
ratio the checkpoint measured.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from benchmark.vibesys.rows import Cell, Row, arms, keep, ratio, sweeps


class Round(NamedTuple):
    hypothesis: str
    started: datetime
    ratio: float
    kept: bool


class Checkpoint(NamedTuple):
    started: datetime
    ratio: float
    base: float


def rounds(rows: Iterable[Row], headline: Cell, metric: str = "throughput") -> list[Round]:
    """The loop rounds that measured both arms on ``headline``, in the order they started."""
    out = []
    for (_, hypothesis), group in sweeps(r for r in rows if r.status == "probe").items():
        arm = arms(group, metric).get(headline)
        if arm and arm.base and arm.cand:
            out.append(Round(hypothesis, _start(group), ratio(arm), keep(group, headline, metric).kept))
    return sorted(out, key=lambda r: r.started)


def checkpoints(rows: Iterable[Row], headline: Cell, metric: str = "throughput") -> list[Checkpoint]:
    """The checkpoint sweeps that measured both arms on ``headline``, in the order they started."""
    out = []
    for group in sweeps(r for r in rows if r.status == "counted").values():
        arm = arms(group, metric).get(headline)
        if arm and arm.base and arm.cand:
            out.append(Checkpoint(_start(group), ratio(arm), sum(arm.base) / len(arm.base)))
    return sorted(out, key=lambda c: c.started)


def position(checkpoint: Checkpoint, rounds: list[Round]) -> int:
    """The number of rounds that started before ``checkpoint``, which is where it sits on the line."""
    return sum(r.started < checkpoint.started for r in rounds)


def levels(rounds: list[Round], checkpoints: list[Checkpoint]) -> list[float]:
    """The best-so-far ratio to the baseline before the first round and after each round.

    The line meets every checkpoint whose segment of rounds holds a kept round. A checkpoint that
    measured below the line pulls the segment's kept steps down with it.
    """
    ends = {position(c, rounds): c.ratio for c in checkpoints}
    out = [1.0]
    start = 0
    for end in sorted(ends) + [len(rounds)]:
        segment = rounds[start:end]
        raw = sum(math.log(r.ratio) for r in segment if r.kept)
        scale = math.log(ends[end] / out[-1]) / raw if end in ends and raw else 1.0
        for r in segment:
            out.append(out[-1] * math.exp(scale * math.log(r.ratio)) if r.kept else out[-1])
        start = end
    return out


def chart(rows: Iterable[Row], headline: Cell, out: str | Path, title: str, metric: str = "throughput") -> None:
    """Draw ``headline``'s best-so-far line to ``out``, in ``metric`` units of the first checkpoint's
    baseline, or in multiples of the baseline when there is no checkpoint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(rows)
    found = rounds(rows, headline, metric)
    measured = checkpoints(rows, headline, metric)
    unit = measured[0].base if measured else 1.0
    line = [unit * v for v in levels(found, measured)]
    x = list(range(len(line)))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axhline(line[0], ls="--", color="#7a7a7a", lw=1.2)
    ax.step(x, line, where="post", color="#2f68a8", lw=2.2, label="Best so far")
    ax.fill_between(x, line, step="post", color="#b4cbe6", alpha=0.45)
    kept = [i for i, r in enumerate(found, 1) if r.kept]
    ax.plot(kept, [line[i] for i in kept], "o", color="#2f68a8", ms=6)
    dropped = [i for i, r in enumerate(found, 1) if not r.kept]
    dropped_at = [line[i - 1] * found[i - 1].ratio for i in dropped]
    ax.plot(dropped, dropped_at, "o", color="#b9b9b9", ms=6, label="Round not kept (its own ratio)")
    sweep_at = [unit * c.ratio for c in measured]
    sweep_x = [position(c, found) for c in measured]
    ax.plot(sweep_x, sweep_at, "D", ms=8, mfc="white", mec="#2f68a8", mew=1.8, ls="", label="Checkpoint sweep")
    ax.set_ylim(0, max(line + sweep_at + dropped_at) * 1.15)
    ax.set_xticks(x)
    ax.set_xlabel("Round")
    units = {r.unit for r in rows if r.cell == headline}
    label = f"throughput ({', '.join(sorted(units))})" if metric == "throughput" else metric
    ax.set_ylabel(label if measured else f"{label} / baseline")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#e6e6e6")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _start(group: list[Row]) -> datetime:
    return min(datetime.fromisoformat(r.started) for r in group)
