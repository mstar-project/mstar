"""Per-deployment normalization of the robot state and actions.

openpi's policy server quantile-normalizes the raw robot state before it is
binned into the prompt, and maps the model's actions back to robot units
(``transforms.Normalize`` / ``Unnormalize`` with ``use_quantiles=True``, which
pi0.5 uses). A deployment that gives the same statistics in its serving config
takes the raw state from clients and returns raw actions, like openpi does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

_EPS = 1e-6  # openpi's _normalize_quantile / _unnormalize_quantile


class QuantileStats(NamedTuple):
    """Per-dimension 1st and 99th percentiles, which normalization maps to -1 and 1."""

    q01: torch.Tensor  # float64 [D]
    q99: torch.Tensor


@dataclass(frozen=True)
class Pi05NormStats:
    state: QuantileStats | None = None
    actions: QuantileStats | None = None

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Quantile-normalize a raw state. Like openpi, the stats are cut to the
        state's length, so a robot with fewer dims than the stats is fine."""
        if self.state is None:
            return state
        n, width = state.shape[-1], self.state.q01.shape[-1]
        if n > width:
            raise ValueError(f"robot_state has {n} values, but the deployment's norm_stats cover {width}")
        q01, q99 = self.state.q01[:n], self.state.q99[:n]
        return (state.to(torch.float64) - q01) / (q99 - q01 + _EPS) * 2.0 - 1.0

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Map the dims the stats cover back to robot units. The rest, the padding
        up to action_dim, pass through as openpi leaves them."""
        if self.actions is None:
            return actions
        d = min(self.actions.q01.shape[-1], actions.shape[-1])
        q01, q99 = self.actions.q01[:d], self.actions.q99[:d]
        out = actions.to(torch.float64, copy=True)
        out[..., :d] = (out[..., :d] + 1.0) / 2.0 * (q99 - q01 + _EPS) + q01
        return out


def load_norm_stats(spec: dict | str | None) -> Pi05NormStats | None:
    """Parse the ``norm_stats`` model kwarg.

    ``spec`` is either a mapping laid out like openpi's ``norm_stats.json``
    (``state`` and/or ``actions``, each with ``q01`` and ``q99`` lists, other
    fields such as ``mean`` / ``std`` ignored, optionally under a top-level
    ``norm_stats`` key) or the path of such a JSON file.
    """
    if spec is None:
        return None
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).expanduser().read_text())
    if not isinstance(spec, dict):
        raise TypeError(f"norm_stats must be a mapping or a file path, got {type(spec).__name__}")
    spec = spec.get("norm_stats", spec)
    unknown = sorted(set(spec) - {"state", "actions"})
    if unknown or not spec:
        raise ValueError(f"norm_stats needs 'state' and/or 'actions' entries, got {sorted(spec)}")
    return Pi05NormStats(state=_quantiles(spec, "state"), actions=_quantiles(spec, "actions"))


def _quantiles(spec: dict, key: str) -> QuantileStats | None:
    entry = spec.get(key)
    if entry is None:
        return None
    if "q01" not in entry or "q99" not in entry:
        raise ValueError(f"norm_stats.{key} needs q01 and q99 lists (pi0.5 uses quantile normalization)")
    q01 = torch.tensor(entry["q01"], dtype=torch.float64)
    q99 = torch.tensor(entry["q99"], dtype=torch.float64)
    if q01.ndim != 1 or q01.shape != q99.shape or not q01.numel():
        raise ValueError(f"norm_stats.{key}: q01 and q99 must be lists of one length, got "
                         f"{tuple(q01.shape)} and {tuple(q99.shape)}")
    return QuantileStats(q01, q99)
