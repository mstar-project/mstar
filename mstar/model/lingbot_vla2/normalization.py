"""Quantile (q01/q99) normalization and 55-dim padding utilities for LingBot-VLA 2.0.

Matches ``bounds_99_woclip`` normalization from ``Robbyant/lingbot-vla-v2``:
active robot state/action dimensions are linearly mapped between ``q01`` ->
``-1.0`` and ``q99`` -> ``+1.0`` (without clipping), and zero-padded to the
action expert's ``max_state_dim`` / ``max_action_dim`` (default 55).
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def pad_vector(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Zero-pad or validate trailing dimension to ``target_dim``."""
    cur = vector.shape[-1]
    if cur == target_dim:
        return vector
    if cur > target_dim:
        raise ValueError(f"Vector dim {cur} exceeds max model dim {target_dim}")
    pad_shape = (*vector.shape[:-1], target_dim - cur)
    pad = torch.zeros(pad_shape, dtype=vector.dtype, device=vector.device)
    return torch.cat([vector, pad], dim=-1)


def slice_vector(vector: torch.Tensor, active_dim: int) -> torch.Tensor:
    """Extract the leading ``active_dim`` channels from a padded action tensor."""
    if active_dim > vector.shape[-1]:
        raise ValueError(f"Requested active_dim={active_dim} exceeds tensor dim {vector.shape[-1]}")
    return vector[..., :active_dim]


@dataclass(frozen=True)
class QuantileStats:
    """Per-dimension ``q01``/``q99`` quantile statistics for ``bounds_99_woclip``."""

    q01: tuple[float, ...]
    q99: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.q01) != len(self.q99):
            raise ValueError(f"q01 len ({len(self.q01)}) != q99 len ({len(self.q99)})")
        if not self.q01:
            raise ValueError("QuantileStats cannot be empty")

    @property
    def dim(self) -> int:
        return len(self.q01)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Map active dims ``[q01, q99]`` -> ``[-1, +1]`` without clipping."""
        if values.shape[-1] < self.dim:
            raise ValueError(f"Expected >= {self.dim} dims to normalize, got {values.shape[-1]}")
        q01 = torch.tensor(self.q01, dtype=torch.float32, device=values.device)
        q99 = torch.tensor(self.q99, dtype=torch.float32, device=values.device)
        span = (q99 - q01).clamp_min(1e-6)
        head = (values[..., : self.dim].to(torch.float32) - q01) / span * 2.0 - 1.0
        # Zero out constant dims (where q99 == q01) to match openpi/lingbot bounds_99_woclip.
        const_mask = (q99 - q01).abs() < 1e-12
        if const_mask.any():
            head = torch.where(const_mask, torch.zeros_like(head), head)
        if values.shape[-1] > self.dim:
            return torch.cat([head.to(values.dtype), values[..., self.dim :]], dim=-1)
        return head.to(values.dtype)

    def unnormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Map normalized dims ``[-1, +1]`` -> ``[q01, q99]``."""
        if values.shape[-1] < self.dim:
            raise ValueError(f"Expected >= {self.dim} dims to unnormalize, got {values.shape[-1]}")
        q01 = torch.tensor(self.q01, dtype=torch.float32, device=values.device)
        q99 = torch.tensor(self.q99, dtype=torch.float32, device=values.device)
        span = q99 - q01
        head = (values[..., : self.dim].to(torch.float32) + 1.0) * 0.5 * span + q01
        if values.shape[-1] > self.dim:
            return torch.cat([head.to(values.dtype), values[..., self.dim :]], dim=-1)
        return head.to(values.dtype)


@dataclass(frozen=True)
class LingBotVLA2NormStats:
    """Joint state + action normalization statistics."""

    state: QuantileStats | None = None
    actions: QuantileStats | None = None

    def normalize_state(self, state: torch.Tensor, max_state_dim: int) -> torch.Tensor:
        if self.state is not None:
            state = self.state.normalize(state)
        return pad_vector(state, max_state_dim)

    def unnormalize_actions(self, actions: torch.Tensor, active_dim: int | None = None) -> torch.Tensor:
        if self.actions is not None:
            out = self.actions.unnormalize(actions)
            dim = self.actions.dim if active_dim is None else active_dim
            return slice_vector(out, dim)
        return slice_vector(actions, active_dim) if active_dim is not None else actions


def _parse_entry(raw: Mapping[str, Sequence[float]] | None) -> QuantileStats | None:
    if raw is None:
        return None
    if "q01" not in raw or "q99" not in raw:
        raise ValueError(f"Quantile norm stats entry must have q01 and q99, got {list(raw)}")
    return QuantileStats(
        q01=tuple(float(v) for v in raw["q01"]),
        q99=tuple(float(v) for v in raw["q99"]),
    )


def load_norm_stats(spec: Mapping[str, Any] | str | Path | None) -> LingBotVLA2NormStats | None:
    """Load quantile stats from a dict or JSON/YAML file path.

    Supports both M*/openpi keys (``state``, ``actions``) and RoboTwin keys
    (``observation.state``, ``action``), optionally nested under ``norm_stats``.
    """
    if spec is None:
        return None
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    else:
        data = dict(spec)

    if "norm_stats" in data and isinstance(data["norm_stats"], Mapping):
        data = dict(data["norm_stats"])

    state_entry = data.get("state") or data.get("observation.state")
    action_entry = data.get("actions") or data.get("action")
    state = _parse_entry(state_entry)
    actions = _parse_entry(action_entry)
    if state is None and actions is None:
        return None
    return LingBotVLA2NormStats(state=state, actions=actions)
