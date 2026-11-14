"""Calibration tables from dense in-process runs: what a skipping step would have got wrong, for every choice of the
steps it could be predicted from.

For each step ``k`` of a dense run, with ``r_k`` the block stack's residual (every branch, every token):

* ``mag_ratios[k]``: the mean over tokens of ``|r_k| / |r_(k-1)|``, MagCache's table;
* ``reuse_error[k][d - 1]``: ``|r_k - r_(k-d)| / |r_k|``, the error of reusing the residual of ``d`` steps back;
* ``linear_error[k][d - 1][g - 1]``: the same for the order-1 forecast through the steps ``d`` and ``d + g`` back;
* ``sensitivity[k]`` (optional, one more run per step): the final latents' relative error when step ``k`` alone
  reuses step ``k - 1``, over ``reuse_error[k][0]``: how much the rest of the loop amplifies an error made there.

A run also keeps the residuals' inner products (``sq[k] = <r_k, r_k>``, ``dots[k][m - 1] = <r_k, r_(k-m)>``), from
which :func:`fit` solves, over all runs at once, the weights that predict ``r_k`` best from one or two earlier
residuals (least squares on the relative error) and the error they leave:

* ``fit_gain[k][d - 1]`` and ``fit_reuse_error[k][d - 1]``: the one weight on ``r_(k-d)``;
* ``fit_weights[k][d - 1][g - 1]`` and ``fit_linear_error[...]``: the two weights on ``r_(k-d)`` and ``r_(k-d-g)``.

They have the shapes of ``reuse_error`` and ``linear_error``, so the same search prices a plan under them, and
:func:`fitted_forecast` lists a plan's weights for the policy (``Forecast.fitted``).

``plan_search.py`` turns the tables into a plan for a budget of computed steps.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from mstar.model.components.diffusion.step_cache import RESIDUAL, ExplicitPlan, FittedWeights, Forecast, Interval

_INF = float("inf")


@dataclass
class Calibration:
    steps: int
    depth: int
    # what the forecast's steps are spaced by: "step" or "sigma"
    time: str
    mag_ratios: list[float]
    reuse_error: list[list[float]]
    linear_error: list[list[list[float]]]
    sensitivity: list[float] | None = None
    # the tables of each run averaged above, for a per-prompt search
    runs: list[dict[str, Any]] = field(default_factory=list)
    # the fitted weights and what they leave (``fit``); None for a calibration read from before they existed
    fit_gain: list[list[float]] | None = None
    fit_reuse_error: list[list[float]] | None = None
    fit_weights: list[list[list[list[float]]]] | None = None
    fit_linear_error: list[list[list[float]]] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Calibration:
        return cls(**data)


def _mean(tables: list) -> Any:
    """Entry-wise mean of equally nested lists of floats (padding, infinite in every run, stays infinite)."""
    if isinstance(tables[0], (int, float)):
        return float(sum(tables) / len(tables))
    return [_mean([table[i] for table in tables]) for i in range(len(tables[0]))]


def _dense_tables(backend, prompt: str, seed: int, depth: int, forecast: Forecast, **request) -> dict[str, Any]:
    """One dense run's tables, and its final latents."""
    backend.use(Interval(every=1))  # keeps every step's residual, skips none: dense output
    kept: deque[torch.Tensor] = deque(maxlen=2 * depth)  # newest first
    ratios, reuse, linear, sq, dots = [], [], [], [], []
    times = backend.step_times(forecast, seed, **request)

    def observe(k: int, state, _outputs) -> None:
        residual = state.tensors[RESIDUAL]
        norm = residual.norm().clamp_min(1e-12)
        # summed in fp64: a fit's error is a small difference of these
        sq.append((residual * residual).sum(dtype=torch.float64))
        dots.append([(residual * older).sum(dtype=torch.float64) for older in kept])
        if kept:
            ratios.append((residual.norm(dim=-1) / kept[0].norm(dim=-1).clamp_min(1e-12)).mean())
        reuse.append([(residual - kept[d - 1]).norm() / norm for d in range(1, min(depth, len(kept)) + 1)])
        row = []
        for d in range(1, min(depth, len(kept)) + 1):
            entries = []
            for g in range(1, min(depth, len(kept) - d) + 1):
                w = forecast.weights((times[k - d], times[k - d - g]), times[k])
                entries.append((residual - w[0] * kept[d - 1] - w[1] * kept[d + g - 1]).norm() / norm)
            row.append(entries)
        linear.append(row)
        kept.appendleft(residual)

    loop = backend.run_loop(prompt, seed, **request, observer=observe)
    return {
        "mag_ratios": [1.0, *(float(r) for r in ratios)],
        "reuse_error": [[float(e) for e in row] for row in reuse],
        "linear_error": [[[float(e) for e in entries] for entries in row] for row in linear],
        "sq": [float(v) for v in sq],
        "dots": [[float(v) for v in row] for row in dots],
        "latents": loop.latents,
    }


def _pad(table: list[list], depth: int, fill) -> list[list]:
    """Rows of early steps have fewer entries (nothing kept that far back); pad them so the tables are rectangular.
    The search never reads the padding: a step cannot be predicted from before step 0."""
    return [row + [fill() for _ in range(depth - len(row))] for row in table]


def calibrate(
    backend, prompts: list[tuple[str, int]], *, size: str, frames: int, steps: int, depth: int = 6,
    forecast: Forecast = Forecast(order=1), sensitivity: bool = False,
) -> Calibration:
    """Tables averaged over ``prompts`` (``(prompt, seed)`` pairs), on ``backend`` (the in-process one)."""
    request = {"size": size, "frames": frames, "steps": steps}
    runs = []
    for prompt, seed in prompts:
        tables = _dense_tables(backend, prompt, seed, depth, forecast, **request)
        latents = tables.pop("latents")
        tables["reuse_error"] = _pad(tables["reuse_error"], depth, lambda: float("inf"))
        tables["linear_error"] = [
            _pad(row, depth, lambda: float("inf")) + [[float("inf")] * depth for _ in range(depth - len(row))]
            for row in tables["linear_error"]
        ]
        if sensitivity:
            tables["sensitivity"] = _sensitivity(backend, prompt, seed, latents, tables["reuse_error"], **request)
        runs.append({"prompt": prompt, "seed": seed, **tables})
    backend.use(None)
    return Calibration(
        steps=steps, depth=depth, time=forecast.time,
        mag_ratios=_mean([run["mag_ratios"] for run in runs]),
        reuse_error=_mean([run["reuse_error"] for run in runs]),
        linear_error=_mean([run["linear_error"] for run in runs]),
        sensitivity=_mean([run["sensitivity"] for run in runs]) if sensitivity else None,
        runs=runs,
        **fit(runs, steps, depth),
    )


def _left(a: float, b: float, rx: float, ry: float, xx: float, xy: float, yy: float) -> float:
    """``|r - a x - b y| / |r|`` from inner products already divided by ``<r, r>``."""
    return max(0.0, 1.0 - 2 * a * rx - 2 * b * ry + a * a * xx + 2 * a * b * xy + b * b * yy) ** 0.5


def fit(runs: list[dict[str, Any]], steps: int, depth: int, held_out: int | None = None) -> dict[str, Any]:
    """The fitted weights over ``runs`` (each run's inner products divided by its ``<r_k, r_k>``, so every run
    counts alike) and the error they leave, averaged over the runs; ``held_out`` leaves that run out of the fit and
    reports the error on it alone."""
    fitted = [run for i, run in enumerate(runs) if i != held_out]
    scored = runs if held_out is None else [runs[held_out]]

    def terms(run, k: int, d: int, g: int | None) -> tuple[float, ...]:
        rr = max(run["sq"][k], 1e-300)
        one = (run["dots"][k][d - 1] / rr, run["sq"][k - d] / rr)
        if g is None:
            return one
        return (*one, run["dots"][k][d + g - 1] / rr, run["dots"][k - d][g - 1] / rr, run["sq"][k - d - g] / rr)

    gain = [[1.0] * depth for _ in range(steps)]
    reuse_error = [[_INF] * depth for _ in range(steps)]
    weights = [[[[1.0, 0.0] for _ in range(depth)] for _ in range(depth)] for _ in range(steps)]
    linear_error = [[[_INF] * depth for _ in range(depth)] for _ in range(steps)]
    for k in range(steps):
        for d in range(1, min(depth, k) + 1):
            rx, xx = (sum(col) for col in zip(*(terms(run, k, d, None) for run in fitted), strict=True))
            a = rx / xx if xx > 0 else 1.0
            gain[k][d - 1] = a
            left = []
            for run in scored:
                rx, xx = terms(run, k, d, None)
                left.append(_left(a, 0.0, rx, 0.0, xx, 0.0, 0.0))
            reuse_error[k][d - 1] = sum(left) / len(left)
            for g in range(1, min(depth, k - d) + 1):
                rx, xx, ry, xy, yy = (sum(col) for col in zip(*(terms(run, k, d, g) for run in fitted), strict=True))
                det = xx * yy - xy * xy
                # two residuals too alike to tell apart (or a zero one) leave the one-weight fit
                pair = [a, 0.0] if det <= 1e-9 * xx * yy else [(rx * yy - ry * xy) / det, (ry * xx - rx * xy) / det]
                weights[k][d - 1][g - 1] = pair
                left = []
                for run in scored:
                    rx, xx, ry, xy, yy = terms(run, k, d, g)
                    left.append(_left(*pair, rx, ry, xx, xy, yy))
                linear_error[k][d - 1][g - 1] = sum(left) / len(left)
    return {"fit_gain": gain, "fit_reuse_error": reuse_error, "fit_weights": weights, "fit_linear_error": linear_error}


def fitted_forecast(cal: Calibration, computed: Sequence[int], restarts: Sequence[int] = ()) -> Forecast:
    """An order-1 forecast carrying ``cal``'s fitted weights for every step the plan ``computed`` skips: two
    weights where the two computed steps before it are within the tables' reach and after the last restart, else
    one. The search prices a plan under ``fit_reuse_error`` / ``fit_linear_error`` the same way."""
    if cal.fit_weights is None:
        raise ValueError("this calibration has no fitted weights; run calibrate() again")
    done = sorted(set(computed) | {0})
    entries = []
    for k in range(cal.steps):
        if k in done:
            continue
        earlier = [c for c in done if c < k]
        last, prev = earlier[-1], (earlier[-2] if len(earlier) > 1 else -1)
        floor = max((r for r in restarts if r <= last), default=0)
        d = k - last
        if d > cal.depth:
            continue  # beyond the tables: the forecast's own weights apply
        if prev < floor or prev < 0:
            entries.append((k, (last,), (cal.fit_gain[k][d - 1],)))
        elif last - prev <= cal.depth:
            entries.append((k, (last, prev), tuple(cal.fit_weights[k][d - 1][last - prev - 1])))
        else:
            entries.append((k, (last, prev), (cal.fit_gain[k][d - 1], 0.0)))
    return Forecast(order=1, time=cal.time, fitted=FittedWeights(cal.steps, tuple(entries)))


def _sensitivity(backend, prompt: str, seed: int, dense: torch.Tensor, reuse_error, **request) -> list[float]:
    """Per step: the final latents' relative error when that step alone reuses the one before, per unit of the
    step's own residual error. Steps 0 and the last always compute and get 0."""
    steps = request["steps"]
    norm = dense.float().norm().clamp_min(1e-12)
    out = [0.0] * steps
    for k in range(1, steps - 1):
        backend.use(ExplicitPlan(computed=tuple(j for j in range(steps) if j != k), num_steps=steps))
        latents = backend.run_loop(prompt, seed, **request).latents
        out[k] = float((latents.float() - dense.float()).norm() / norm) / max(reuse_error[k][0], 1e-12)
    return out
