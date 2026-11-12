"""Opt-in step caching for the DiT scaffold: skip a step's transformer blocks and reuse the last computed residual.

A cached step still runs the model's input embedding and output head; only the block stack, where nearly all the
time goes, is replaced by ``hidden + residual``, ``residual`` being what the blocks added at the last step that ran
them. Which steps reuse is the policy's decision, from host values alone (the step index and the request's step
count), so it costs no host sync and every rank of a lockstep instance derives the same answer from the same config
(AGENTS.md invariants 7 and 11).

A model opts in by passing a :class:`StepCachePolicy` to ``DenoiseLoopSubmodule(step_cache=...)`` and routing its
block stack through the :class:`BlockCache` the base hands it (:meth:`DenoiseLoopSubmodule.block_cache`). The
residuals are per-request state the engine owns and frees with the request (``request_state``), each rank holding
its own: its CFG branch, its sequence-parallel shard.

Methods:

* :class:`MagCache` (Ma et al., arXiv 2506.09045): the magnitude ratio of successive residuals is nearly the same
  for every prompt, so one calibration run gives a per-step table; a step reuses while the accumulated error of
  that ratio stays under a threshold. Published as better quality per skipped step than TeaCache on Wan2.1 and
  Open-Sora, and its decision needs no device value at run time.
* :class:`MagCacheCalibration`: runs every step and logs the table for :class:`MagCache` at the request's end.

A policy whose decision depends on a device value (TeaCache's modulated-input distance, DBCache's first-block
residual) would need a reduce across the instance and a host read per step; none is implemented.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Where a request's cached residual and calibration record live in its ``request_state``.
RESIDUAL = "step_cache_residual"
_RATIOS = "step_cache_ratios"


class StepCachePolicy:
    """Which steps of a request reuse the cached residual. A pure function of the step count, memoized."""

    def reuse_plan(self, num_steps: int) -> tuple[bool, ...]:
        """One flag per step, True where the step reuses. Step 0 never does, as nothing is cached yet."""
        raise NotImplementedError

    def reuses(self, step: int, num_steps: int) -> bool:
        return self.reuse_plan(num_steps)[step]

    # Calibration policies record each computed step's residual magnitude ratio.
    records = False


def _nearest(table: Sequence[float], num_steps: int) -> tuple[float, ...]:
    """``table`` resampled to ``num_steps`` entries by nearest index (MagCache's ``nearest_interp``)."""
    if len(table) == num_steps:
        return tuple(table)
    if num_steps == 1:
        return (table[0],)
    last = len(table) - 1
    return tuple(table[round(i * last / (num_steps - 1))] for i in range(num_steps))


@dataclass(frozen=True)
class MagCache(StepCachePolicy):
    """MagCache's skip rule over a calibrated table of residual magnitude ratios.

    ``mag_ratios[k]`` is the mean over tokens of ``|r_k| / |r_(k-1)|``, ``r_k`` the block stack's residual at step
    ``k`` of a dense run (:class:`MagCacheCalibration` logs it); a table calibrated at another step count is resampled
    by nearest index. Past the first ``retention_ratio`` of the steps, a step reuses while the error accumulated since
    the last computed step, ``sum |1 - prod(ratios)|``, stays at or under ``threshold`` and no more than
    ``max_skip_steps`` steps in a row have reused. The last step always computes."""

    mag_ratios: tuple[float, ...]
    threshold: float = 0.06
    max_skip_steps: int = 2
    retention_ratio: float = 0.2

    def __post_init__(self):
        if len(self.mag_ratios) < 2:
            raise ValueError("MagCache needs a calibrated mag_ratios table of at least two steps")
        if self.threshold < 0 or self.max_skip_steps < 0 or not 0 <= self.retention_ratio <= 1:
            raise ValueError(
                f"MagCache: threshold and max_skip_steps must be >= 0 and retention_ratio in [0, 1]; got "
                f"{self.threshold}, {self.max_skip_steps}, {self.retention_ratio}"
            )
        object.__setattr__(self, "mag_ratios", tuple(float(r) for r in self.mag_ratios))

    @functools.lru_cache(maxsize=16)  # noqa: B019 - frozen and hashable; a handful of step counts per deployment
    def reuse_plan(self, num_steps: int) -> tuple[bool, ...]:
        ratios = _nearest(self.mag_ratios, num_steps)
        retained = max(1, int(self.retention_ratio * num_steps + 0.5))
        plan, ratio, err, skipped = [], 1.0, 0.0, 0
        for k in range(num_steps):
            reuse = False
            if retained <= k < num_steps - 1:
                ratio *= ratios[k]
                err += abs(1.0 - ratio)
                reuse = err <= self.threshold and skipped < self.max_skip_steps
            if reuse:
                skipped += 1
            else:
                ratio, err, skipped = 1.0, 0.0, 0
            plan.append(reuse)
        return tuple(plan)


@dataclass(frozen=True)
class MagCacheCalibration(StepCachePolicy):
    """Computes every step and logs the request's ``mag_ratios`` table on its last step (one host read per request,
    so a calibration run only; on one GPU, as a rank sees only its branch and shard)."""

    records = True

    def reuse_plan(self, num_steps: int) -> tuple[bool, ...]:
        return (False,) * num_steps


def step_cache_policy(spec: Mapping[str, Any] | None, calibrated: Mapping[str, Sequence[float]] | None = None):
    """A policy from a serving config's ``step_cache`` mapping; None (the default) or ``method: none`` is no cache.

    ``{method: magcache, threshold, max_skip_steps, retention_ratio, mag_ratios}``: ``mag_ratios`` is a table or the
    name of one of the model's own ``calibrated`` tables (``default`` when omitted). ``{method: magcache_calibrate}``
    logs a table instead."""
    if spec is None:
        return None
    spec = dict(spec)
    method = str(spec.pop("method", "magcache")).lower()
    if method == "none":
        return None
    if method == "magcache_calibrate":
        if spec:
            raise ValueError(f"step_cache: magcache_calibrate takes no options, got {sorted(spec)}")
        return MagCacheCalibration()
    if method != "magcache":
        raise ValueError(f"step_cache: unknown method {method!r} (magcache, magcache_calibrate, none)")
    unknown = set(spec) - {"mag_ratios", "threshold", "max_skip_steps", "retention_ratio"}
    if unknown:
        raise ValueError(f"step_cache: unknown magcache options {sorted(unknown)}")
    ratios = spec.pop("mag_ratios", "default")
    if isinstance(ratios, str):
        if not calibrated or ratios not in calibrated:
            raise ValueError(f"step_cache: no calibrated mag_ratios table {ratios!r} for this model")
        ratios = calibrated[ratios]
    return MagCache(mag_ratios=tuple(ratios), **spec)


@dataclass
class BlockCache:
    """One step's view of its rows' cached residuals, handed to the model's forward.

    ``reuse`` is the step's decision, the same for every row (it is part of the step's key). A reusing step adds
    :attr:`residual` to the stream in place of its blocks; a computing step hands what its blocks added to
    :meth:`keep`. Both are ``[rows, ...]``, row-leading, in whatever per-row layout the model chooses, as long as it
    is the same at both ends."""

    reuse: bool
    _states: list[Any]
    _final: list[bool]
    _records: bool = False
    _ratios: list[list[torch.Tensor]] = field(default_factory=list)

    @property
    def residual(self) -> torch.Tensor:
        cached = [state.tensors.get(RESIDUAL) for state in self._states]
        if any(r is None for r in cached):
            raise RuntimeError("step cache: a reusing step found no residual; step 0 must compute")
        return cached[0].unsqueeze(0) if len(cached) == 1 else torch.stack(cached)

    def keep(self, residual: torch.Tensor) -> None:
        """Cache each row of ``residual`` (``[rows, ...]``, what the block stack added; fp32 recommended) for the
        row's later reusing steps."""
        for i, state in enumerate(self._states):
            row = residual[i]
            if self._records:
                self._record(state, row, self._final[i])
            state.tensors[RESIDUAL] = row

    def _record(self, state, row: torch.Tensor, final: bool) -> None:
        prev = state.tensors.get(RESIDUAL)
        ratios = state.kwargs.setdefault(_RATIOS, [])
        if prev is not None:
            # per-token magnitude ratio over the hidden dim, averaged
            ratios.append((row.norm(dim=-1) / prev.norm(dim=-1).clamp_min(1e-12)).mean())
        if final and ratios:
            table = [round(float(r), 5) for r in torch.stack(ratios).tolist()]
            logger.info("step cache calibration (%d steps): mag_ratios = [1.0, %s]", len(table) + 1,
                        ", ".join(str(r) for r in table))
