"""Fit JCT = a + b·K over a step-count sweep to isolate per-step dispatch cost.

The dummy models' forward is an identity op, so a step's cost is essentially
all runtime: scheduler decision, batch assembly, conductor→worker dispatch,
engine entry/exit, output routing. Sweeping K and fitting a line drops the
constant part (HTTP ingress, preprocessing, one-time setup) into the intercept
``a``; the slope ``b`` is the per-step overhead.

    mstar serve dummy_loop --log-stats-file loop.log     # in another shell
    python -m benchmark.dummy.sweep -K 1 -K 10 -K 100 -B 1 -B 16

Run it against ``dummy_walks`` too: the delta between the two slopes is the
cost of the conductor round trip that ``dummy_loop`` avoids by keeping its
loop on the worker.

``--reference-step-ms`` expresses the slope as a percentage of a real decode
step — take the number from an ``--log-stats`` graph-timings table (e.g. the
Qwen3 talker's ``fwd (ms/exec)``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass

from benchmark.dummy.runner import (
    RunResult,
    add_common_args,
    config_from_args,
    run,
)
from mstar.profile.aggregate import summarize


@dataclass
class Fit:
    """Least-squares fit of y = a + b·x."""

    a: float  # intercept, ms
    b: float  # slope, ms per step
    r2: float | None
    n: int

    @property
    def b_us(self) -> float:
        return self.b * 1e3


def least_squares(xs: list[float], ys: list[float]) -> Fit:
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two K values to fit a slope")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("all K values identical; nothing to fit")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx
    a = mean_y - b * mean_x
    syy = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - sum(
        (y - (a + b * x)) ** 2 for x, y in zip(xs, ys, strict=True)
    ) / syy if syy > 0 else None
    return Fit(a=a, b=b, r2=r2, n=n)


@dataclass
class Point:
    steps: int
    batch_size: int
    jct_mean_ms: float
    jct_p50_ms: float
    makespan_mean_ms: float
    num_ok: int
    num_failed: int


def point_from_many(results: list[RunResult]) -> Point:
    """Pool every repeat of one (K, B) cell into a single point.

    Pooling the raw per-request samples — rather than averaging each repeat's
    mean — keeps repeats of unequal size correctly weighted.
    """
    jct = summarize([
        r.jct_ms for res in results for r in res.ok_requests()
        if r.jct_ms is not None
    ])
    makespans = [
        w.makespan_ms for res in results for w in res.measured()
    ]
    makespan = summarize(makespans)
    first = results[0]
    return Point(
        steps=first.config.steps,
        batch_size=first.config.batch_size,
        jct_mean_ms=jct.mean if jct.mean is not None else float("nan"),
        jct_p50_ms=jct.p50 if jct.p50 is not None else float("nan"),
        makespan_mean_ms=(
            makespan.mean if makespan.mean is not None else float("nan")
        ),
        num_ok=sum(len(res.ok_requests()) for res in results),
        num_failed=sum(len(res.errors()) for res in results),
    )


#: Below this, the points are not on a line and the slope is meaningless.
MIN_R2 = 0.95


def diagnose(points: list[Point], fit: Fit | None) -> list[str]:
    """Reasons the fit should not be believed.

    A per-step cost must be monotonic in K — more steps cannot take less time
    — so a dip is drift between runs, not a slope. And when the intercept
    dwarfs the range the fit spans, there is very little signal to fit.
    """
    warnings: list[str] = []
    ordered = sorted(points, key=lambda p: p.steps)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur.jct_mean_ms < prev.jct_mean_ms:
            warnings.append(
                f"NON-MONOTONIC: K={cur.steps} ({cur.jct_mean_ms:.1f} ms) is "
                f"faster than K={prev.steps} ({prev.jct_mean_ms:.1f} ms)"
            )
    if fit is None:
        return warnings
    if fit.r2 is not None and fit.r2 < MIN_R2:
        warnings.append(
            f"POOR FIT: R^2={fit.r2:.3f} < {MIN_R2}; these points are not on a line"
        )
    span = max(p.jct_mean_ms for p in ordered) - min(p.jct_mean_ms for p in ordered)
    if fit.a > 0 and span < fit.a * 0.5:
        warnings.append(
            f"LOW SIGNAL: K sweep moves JCT by {span:.1f} ms against a "
            f"{fit.a:.1f} ms intercept. Shrink the fixed cost "
            f"(--tensor-size '[1,1]') or widen the K range."
        )
    return warnings


def render(
    by_batch: dict[int, list[Point]],
    fits: dict[int, Fit],
    reference_step_ms: float | None,
) -> str:
    lines = ["=" * 88, " Per-step dispatch overhead  (JCT = a + b·K)", "=" * 88]
    for bs in sorted(by_batch):
        points = sorted(by_batch[bs], key=lambda p: p.steps)
        # Closed-loop has no wave structure, so there is no makespan to show.
        has_makespan = any(p.makespan_mean_ms == p.makespan_mean_ms for p in points)
        lines.append("")
        lines.append(f" B = {bs}")
        header = f"   {'K':>6}  {'mean JCT':>12}  {'p50 JCT':>12}  "
        if has_makespan:
            header += f"{'wave makespan':>14}  "
        lines.append(header + f"{'ok':>5}  {'fail':>5}")
        lines.append("   " + "-" * (66 if has_makespan else 50))
        for p in points:
            row = (
                f"   {p.steps:>6}  {p.jct_mean_ms:>10.2f} ms  "
                f"{p.jct_p50_ms:>10.2f} ms  "
            )
            if has_makespan:
                row += f"{p.makespan_mean_ms:>12.2f} ms  "
            lines.append(row + f"{p.num_ok:>5}  {p.num_failed:>5}")
        fit = fits.get(bs)
        if fit is None:
            lines.append("   (need >= 2 K values to fit)")
            continue
        r2 = f"{fit.r2:.5f}" if fit.r2 is not None else "n/a"
        lines.append("")
        lines.append(
            f"   fit: JCT = {fit.a:.2f} ms + {fit.b_us:.1f} us x K     (R^2={r2})"
        )

        problems = diagnose(points, fit)
        if problems:
            # Printing a slope next to its own refutation invites someone to
            # quote the number and drop the caveat, so withhold it entirely.
            lines.append("")
            for problem in problems:
                lines.append(f"   !! {problem}")
            lines.append("   !! slope withheld — this fit does not measure a per-step cost")
            continue

        lines.append(f"   per-step dispatch overhead      b = {fit.b_us:8.1f} us")
        lines.append(
            f"   amortized per request in batch  b/B = {fit.b_us / bs:8.1f} us"
        )
        if reference_step_ms:
            pct = fit.b / reference_step_ms * 100
            pct_amortized = fit.b / bs / reference_step_ms * 100
            lines.append(
                f"   vs a {reference_step_ms:.2f} ms real decode step: "
                f"{pct:.1f}%  ({pct_amortized:.1f}% amortized)"
            )
    lines.append("=" * 88)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.dummy.sweep",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps", "-K", type=int, action="append", default=None,
        help="step count to measure; repeat for the sweep (default: 1 10 100)",
    )
    parser.add_argument(
        "--batch-sizes", "-B", type=int, action="append", default=None,
        help="batch size to measure; repeat (default: 1 16)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="passes over the K list, interleaved round-robin rather than "
             "run-to-completion per K. Server state drifts over a long sweep; "
             "interleaving spreads that drift across all K instead of letting "
             "it masquerade as slope. Results are pooled per (K, B).",
    )
    parser.add_argument(
        "--reference-step-ms", type=float, default=None,
        help="a real model's per-step forward time, to report b as a percentage",
    )
    parser.add_argument("--json", default=None, help="write full results here")
    # B is swept here, so the shared single-value --batch-size is left out and
    # args.batch_size is set per point below.
    add_common_args(parser, include_batch_size=False)
    args = parser.parse_args(argv)

    step_values = args.steps or [1, 10, 100]
    batch_values = args.batch_sizes or [1, 16]

    # Round-robin over K so drift in server state spreads across the sweep
    # rather than accumulating within one K.
    results_by_cell: dict[tuple[int, int], list[RunResult]] = {}
    for rep in range(max(args.repeat, 1)):
        for bs in batch_values:
            args.batch_size = bs
            for k in step_values:
                config = config_from_args(args, k)
                print(
                    f"--> pass {rep + 1}/{args.repeat}  B={bs} K={k} ...",
                    file=sys.stderr, flush=True,
                )
                result = asyncio.run(run(config, verbose=args.verbose))
                results_by_cell.setdefault((bs, k), []).append(result)

    by_batch: dict[int, list[Point]] = {}
    raw: list[dict] = []
    for (bs, k), results in sorted(results_by_cell.items()):
        point = point_from_many(results)
        by_batch.setdefault(bs, []).append(point)
        raw.append(asdict(point))
        if point.num_failed:
            print(
                f"    warning: {point.num_failed} requests failed at B={bs} K={k}",
                file=sys.stderr,
            )

    fits: dict[int, Fit] = {}
    for bs, points in by_batch.items():
        usable = [p for p in points if p.num_ok]
        if len(usable) < 2:
            continue
        fits[bs] = least_squares(
            [float(p.steps) for p in usable],
            [p.jct_mean_ms for p in usable],
        )

    sys.stdout.write(render(by_batch, fits, args.reference_step_ms))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "url": args.url,
                    "steps": step_values,
                    "batch_sizes": batch_values,
                    "repeat": args.repeat,
                    "mode": args.mode.value,
                    "points": raw,
                    "fits": {str(bs): asdict(f) for bs, f in fits.items()},
                    "fit_warnings": {
                        str(bs): diagnose(by_batch[bs], fits.get(bs))
                        for bs in by_batch
                    },
                    "reference_step_ms": args.reference_step_ms,
                },
                fh,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
