"""Calibration tables from dense in-process runs: what a skipping step would have got wrong, for every choice of the
steps it could be predicted from.

For each step ``k`` of a dense run, with ``r_k`` the block stack's residual (every branch, every token):

* ``mag_ratios[k]``: the mean over tokens of ``|r_k| / |r_(k-1)|``, MagCache's table;
* ``reuse_error[k][d - 1]``: ``|r_k - r_(k-d)| / |r_k|``, the error of reusing the residual of ``d`` steps back;
* ``linear_error[k][d - 1][g - 1]``: the same for the order-1 forecast through the steps ``d`` and ``d + g`` back;
* ``sensitivity[k]`` (optional, one more run per step): the final latents' relative error when step ``k`` alone
  reuses step ``k - 1``, over ``reuse_error[k][0]``: how much the rest of the loop amplifies an error made there.

``plan_search.py`` turns the tables into a plan for a budget of computed steps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from mstar.model.components.diffusion.step_cache import RESIDUAL, ExplicitPlan, Forecast, Interval


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
    ratios, reuse, linear = [], [], []
    times = backend.step_times(forecast, seed, **request)

    def observe(k: int, state, _outputs) -> None:
        residual = state.tensors[RESIDUAL]
        norm = residual.norm().clamp_min(1e-12)
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
    )


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
