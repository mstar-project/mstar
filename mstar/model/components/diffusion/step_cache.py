"""Opt-in step caching for the DiT scaffold: skip a step's transformer blocks and reuse, or extrapolate, the
residuals of the steps that ran them.

A cached step still runs the model's input embedding and output head; only the block stack, where nearly all the
time goes, is replaced by ``hidden + residual``, ``residual`` being what the blocks added at the last step that ran
them, or the policy's :class:`Forecast` from the last few such steps. Which steps reuse is the policy's decision, from
host values alone (the step index, the request's step count and schedule), so it costs no host sync and every rank of
a lockstep instance derives the same answer from the same config (AGENTS.md invariants 7 and 11).

A model opts in by passing a :class:`StepCachePolicy` to ``DenoiseLoopSubmodule(step_cache=...)`` and routing its
block stack through the :class:`BlockCache` the base hands it (:meth:`DenoiseLoopSubmodule.block_cache`). The
residuals are per-request state the engine owns and frees with the request (``request_state``), each rank holding
its own: its CFG branch, its sequence-parallel shard.

Methods:

* :class:`MagCache` (Ma et al., arXiv 2506.09045): the magnitude ratio of successive residuals is nearly the same
  for every prompt, so one calibration run gives a per-step table; a step reuses while the accumulated error of
  that ratio stays under a threshold. Published as better quality per skipped step than TeaCache on Wan2.1 and
  Open-Sora, and its decision needs no device value at run time.
* :class:`Interval`: one step in ``every`` computes (FORA; TaylorSeer's interval).
* :class:`ExplicitPlan`: the computed steps listed outright, as a search over dense runs picks them
  (``benchmark/step_cache``).
* :class:`MagCacheCalibration`: runs every step and logs the table for :class:`MagCache` at the request's end.

Any of the first three takes a :class:`Forecast`: reuse the last residual (the default), extrapolate the last two or
three (TaylorSeer), or weigh them with weights fitted on dense runs (:class:`FittedWeights`).

A policy whose decision depends on a device value (TeaCache's modulated-input distance, DBCache's first-block
residual) would need a reduce across the instance and a host read per step; none is implemented.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

import torch

logger = logging.getLogger(__name__)

# Where a request's cached residuals and calibration record live in its ``request_state``: the newest residual under
# the name itself, older ones under ``<name>_<age>``, and the steps they were kept at in ``kwargs[<name>_steps]``.
RESIDUAL = "step_cache_residual"
_RATIOS = "step_cache_ratios"


class FittedWeights(NamedTuple):
    """Forecast weights fitted offline on dense runs of one schedule (``benchmark/step_cache``'s calibration): for
    a step of a ``num_steps`` schedule predicted from the values kept at ``kept`` steps (newest first), the weights
    over those values that left the least error, in place of a Taylor expansion's."""

    num_steps: int
    # (step, kept steps newest first, weights newest first)
    entries: tuple[tuple[int, tuple[int, ...], tuple[float, ...]], ...]


@dataclass(frozen=True)
class Forecast:
    """What a reusing step adds in place of the block stack, predicted from the residuals of the steps that ran it.

    ``order`` 0 reuses the last residual. Order 1 or 2 extrapolates through the last two or three, a Taylor expansion
    whose derivatives are divided differences (TaylorSeer, Liu et al., arXiv 2503.06923, at equal spacing), each
    derivative term scaled by ``damping`` per order. ``time`` is what the steps are spaced by: their index or the
    schedule's sigma. A request with fewer kept residuals than the order needs uses the order it has.

    ``fitted`` replaces the expansion's weights wherever it lists the step and the kept steps it is predicted from,
    on a request of the table's step count; any other step takes the expansion's."""

    order: int = 0
    damping: float = 1.0
    time: str = "step"
    fitted: FittedWeights | None = None

    def __post_init__(self):
        if self.order not in (0, 1, 2) or not 0.0 <= self.damping <= 1.0 or self.time not in ("step", "sigma"):
            raise ValueError(
                f"step cache forecast: order in (0, 1, 2), damping in [0, 1], time 'step' or 'sigma'; got "
                f"{self.order}, {self.damping}, {self.time!r}"
            )
        index: dict[tuple[int, tuple[int, ...]], tuple[float, ...]] = {}
        if self.fitted is not None:
            num_steps, entries = int(self.fitted[0]), self.fitted[1]
            for step, kept, weights in entries:
                kept, weights = tuple(int(k) for k in kept), tuple(float(w) for w in weights)
                ordered = all(a > b for a, b in zip((num_steps, step, *kept), (step, *kept, -1), strict=True))
                if not (0 < len(kept) == len(weights) <= self.depth and ordered):
                    raise ValueError(
                        f"step cache forecast: a fitted entry is (step, kept steps before it newest first, as many "
                        f"weights), at most {self.depth} of them at order {self.order}; got {(step, kept, weights)}"
                    )
                index[int(step), kept] = weights
            canonical = tuple((step, kept, weights) for (step, kept), weights in sorted(index.items()))
            object.__setattr__(self, "fitted", FittedWeights(num_steps, canonical))
        # a lookup table beside the fields: equality and the hash read ``fitted`` alone
        object.__setattr__(self, "_fitted_index", index)

    @property
    def depth(self) -> int:
        """Kept residuals the forecast reads."""
        return self.order + 1

    def fitted_weights(self, num_steps: int, step: int, kept: tuple[int, ...]) -> tuple[float, ...] | None:
        """The fitted weights for ``step`` of a ``num_steps`` schedule predicted from the residuals kept at ``kept``
        (newest first); None where the table has none, and :meth:`weights` applies."""
        if self.fitted is None or num_steps != self.fitted.num_steps:
            return None
        return self._fitted_index.get((step, kept))

    @functools.lru_cache(maxsize=1024)  # noqa: B019 - frozen and hashable; a few schedules' worth of float tuples
    def weights(self, kept: tuple[float, ...], now: float) -> tuple[float, ...]:
        """The prediction at time ``now`` as weights over the residuals kept at times ``kept``, newest first."""
        order = min(self.order, len(kept) - 1)
        if order == 0 or kept[0] == kept[1]:
            return (1.0,)
        h = (now - kept[0]) * self.damping
        slope = h / (kept[0] - kept[1])
        if order == 1 or kept[1] == kept[2] or kept[0] == kept[2]:
            return (1.0 + slope, -slope)
        # F0 + F'h + F''h^2/2, F' and F'' the first and second divided differences
        older = h / (kept[1] - kept[2])
        curve = h / (kept[0] - kept[2])
        return (1.0 + slope + curve * slope, -slope - curve * (slope + older), curve * older)


@dataclass(frozen=True, kw_only=True)
class StepCachePolicy:
    """Which steps of a request reuse the cached residuals (:meth:`reuse_plan`), and what they add in place of the
    block stack (``forecast``). A pure function of host values, memoized."""

    forecast: Forecast = Forecast()

    # Calibration policies record each computed step's residual magnitude ratio.
    records: ClassVar[bool] = False

    def reuse_plan(self, num_steps: int, segment_starts: tuple[int, ...] = ()) -> tuple[bool, ...]:
        """One flag per step, True where the step reuses. Step 0 never does, as nothing is cached yet, and neither
        does a step in ``segment_starts``, where the model changes weights (``step_cache_segments``)."""
        raise NotImplementedError

    def reuses(self, step: int, num_steps: int, segment_starts: tuple[int, ...] = ()) -> bool:
        return self.reuse_plan(num_steps, segment_starts)[step]


def _nearest(table: Sequence, num_steps: int) -> tuple:
    """``table`` resampled to ``num_steps`` entries by nearest index (MagCache's ``nearest_interp``)."""
    if len(table) == num_steps:
        return tuple(table)
    if num_steps == 1:
        return (table[0],)
    last = len(table) - 1
    return tuple(table[round(i * last / (num_steps - 1))] for i in range(num_steps))


def _retained(retention_ratio: float, num_steps: int) -> int:
    return max(1, int(retention_ratio * num_steps + 0.5))


@dataclass(frozen=True)
class MagCache(StepCachePolicy):
    """MagCache's skip rule over a calibrated table of residual magnitude ratios.

    ``mag_ratios[k]`` is the mean over tokens of ``|r_k| / |r_(k-1)|``, ``r_k`` the block stack's residual at step
    ``k`` of a dense run (:class:`MagCacheCalibration` logs it); a table calibrated at another step count is resampled
    by nearest index. Past the first ``retention_ratio`` of the steps, a step reuses while the error accumulated since
    the last computed step, ``sum |1 - prod(ratios)|``, stays at or under ``threshold`` and no more than
    ``max_skip_steps`` steps in a row have reused. The last step always computes, and so does each step in
    ``segment_starts`` (another expert's first step), where the accumulated error starts over: the table's entry
    there compares two experts' residuals and is never read."""

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
    def reuse_plan(self, num_steps: int, segment_starts: tuple[int, ...] = ()) -> tuple[bool, ...]:
        ratios = _nearest(self.mag_ratios, num_steps)
        retained = _retained(self.retention_ratio, num_steps)
        plan, ratio, err, skipped = [], 1.0, 0.0, 0
        for k in range(num_steps):
            reuse = False
            if retained <= k < num_steps - 1 and k not in segment_starts:
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

    records: ClassVar[bool] = True

    def reuse_plan(self, num_steps: int, segment_starts: tuple[int, ...] = ()) -> tuple[bool, ...]:
        return (False,) * num_steps


@dataclass(frozen=True)
class Interval(StepCachePolicy):
    """Past the first ``retention_ratio`` of the steps, one step in ``every`` computes (FORA; TaylorSeer's interval).
    The last step and each segment start compute."""

    every: int = 2
    retention_ratio: float = 0.2

    def __post_init__(self):
        if self.every < 1 or not 0 <= self.retention_ratio <= 1:
            raise ValueError(
                f"step cache interval: every >= 1 and retention_ratio in [0, 1]; got {self.every}, "
                f"{self.retention_ratio}"
            )

    @functools.lru_cache(maxsize=16)  # noqa: B019 - frozen and hashable; a handful of step counts per deployment
    def reuse_plan(self, num_steps: int, segment_starts: tuple[int, ...] = ()) -> tuple[bool, ...]:
        retained = _retained(self.retention_ratio, num_steps)
        plan, skipped = [], 0
        for k in range(num_steps):
            reuse = retained <= k < num_steps - 1 and k not in segment_starts and skipped < self.every - 1
            skipped = skipped + 1 if reuse else 0
            plan.append(reuse)
        return tuple(plan)


@dataclass(frozen=True)
class ExplicitPlan(StepCachePolicy):
    """The steps that compute, listed outright for a schedule of ``num_steps`` (what a search over a calibration
    run gives, ``benchmark/step_cache``); a request with another step count takes the nearest step's decision.
    Step 0, the last step and each segment start compute whatever the list says."""

    computed: tuple[int, ...]
    num_steps: int

    def __post_init__(self):
        computed = tuple(sorted({int(k) for k in self.computed}))
        if self.num_steps < 1 or any(not 0 <= k < self.num_steps for k in computed):
            raise ValueError(f"step cache plan: computed steps must lie in [0, {self.num_steps}); got {computed}")
        object.__setattr__(self, "computed", computed)

    @functools.lru_cache(maxsize=16)  # noqa: B019 - frozen and hashable; a handful of step counts per deployment
    def reuse_plan(self, num_steps: int, segment_starts: tuple[int, ...] = ()) -> tuple[bool, ...]:
        computed = set(self.computed)
        listed = _nearest([k in computed for k in range(self.num_steps)], num_steps)
        return tuple(
            not listed[k] and 0 < k < num_steps - 1 and k not in segment_starts for k in range(num_steps)
        )


def _options(spec: Mapping[str, Any], what: str, known: set[str]) -> dict[str, Any]:
    unknown = set(spec) - known
    if unknown:
        raise ValueError(f"step_cache: unknown {what} options {sorted(unknown)}")
    return dict(spec)


def _forecast(spec: Any) -> Forecast:
    if spec is None:
        return Forecast()
    if isinstance(spec, Mapping):
        spec = _options(spec, "forecast", {"order", "damping", "time", "fitted"})
        fitted = spec.pop("fitted", None)
        if fitted is not None:
            fitted = _options(fitted, "fitted forecast", {"num_steps", "weights"})
            if set(fitted) != {"num_steps", "weights"}:
                raise ValueError("step_cache: a fitted forecast needs 'num_steps' and 'weights'")
            fitted = FittedWeights(int(fitted["num_steps"]), tuple(
                (int(step), tuple(kept), tuple(weights)) for step, kept, weights in fitted["weights"]
            ))
        return Forecast(**spec, fitted=fitted)
    return Forecast(order=int(spec))


_RULE_OPTIONS = {
    "magcache": {"mag_ratios", "threshold", "max_skip_steps", "retention_ratio"},
    "interval": {"every", "retention_ratio"},
    "plan": {"computed", "num_steps"},
}


def step_cache_policy(spec: Mapping[str, Any] | None, calibrated: Mapping[str, Sequence[float]] | None = None):
    """A policy from a serving config's ``step_cache`` mapping; None (the default) or ``method: none`` is no cache.

    ``method`` picks the steps that skip:

    * ``magcache`` (the default): ``threshold``, ``max_skip_steps``, ``retention_ratio``, ``mag_ratios`` (a table,
      or the name of one of the model's own ``calibrated`` tables; ``default`` when omitted);
    * ``interval``: ``every``, ``retention_ratio``;
    * ``plan``: ``computed`` (the steps that compute) of a ``num_steps`` schedule;
    * ``magcache_calibrate`` logs a ``mag_ratios`` table instead.

    Any of the first three also takes ``forecast``: an order, or ``{order, damping, time, fitted}``, ``fitted`` a
    calibrated ``{num_steps, weights: [[step, [kept steps], [weights]], ...]}``.
    """
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
    if method not in _RULE_OPTIONS:
        raise ValueError(
            f"step_cache: unknown method {method!r} (magcache, interval, plan, magcache_calibrate, none)"
        )
    forecast = _forecast(spec.pop("forecast", None))
    unknown = set(spec) - _RULE_OPTIONS[method]
    if unknown:
        raise ValueError(f"step_cache: unknown {method} options {sorted(unknown)}")
    if method == "interval":
        return Interval(**spec, forecast=forecast)
    if method == "plan":
        if "computed" not in spec or "num_steps" not in spec:
            raise ValueError("step_cache: plan needs 'computed' and 'num_steps'")
        return ExplicitPlan(computed=tuple(spec["computed"]), num_steps=int(spec["num_steps"]), forecast=forecast)
    ratios = spec.pop("mag_ratios", "default")
    if isinstance(ratios, str):
        if not calibrated or ratios not in calibrated:
            raise ValueError(f"step_cache: no calibrated mag_ratios table {ratios!r} for this model")
        ratios = calibrated[ratios]
    return MagCache(mag_ratios=tuple(ratios), **spec, forecast=forecast)


def _push(state: Any, name: str, step: int, value: torch.Tensor, depth: int, floor: int) -> None:
    """Keep ``value`` as the request's newest ``name``, with up to ``depth - 1`` older ones kept at or after step
    ``floor``."""
    steps = [k for k in state.kwargs.get(name + "_steps", ()) if k >= floor][: depth - 1]
    for age in range(len(steps), 0, -1):
        state.tensors[_aged(name, age)] = state.tensors[_aged(name, age - 1)]
    for age in range(len(steps) + 1, depth):
        # what a segment start left behind (another expert's residuals)
        state.tensors.pop(_aged(name, age), None)
    state.tensors[name] = value
    state.kwargs[name + "_steps"] = [step, *steps]


def _aged(name: str, age: int) -> str:
    return name if age == 0 else f"{name}_{age}"


@dataclass
class CacheRow:
    """One row of a step: its request's state and the host values its forecast reads."""

    state: Any
    step: int
    # the schedule's sigma per step, for a forecast spaced by it
    sigmas: tuple[float, ...] | None = None
    # nothing kept before this step is usable (the row's last segment start)
    floor: int = 0
    final: bool = False
    # the request's step count, which a fitted forecast's table is for
    num_steps: int = 0

    def time(self, step: int, forecast: Forecast) -> float:
        return float(step) if forecast.time == "step" else self.sigmas[step]


@dataclass
class BlockCache:
    """One step's view of its rows' cached residuals, handed to the model's forward.

    ``reuse`` is the step's decision, the same for every row (it is part of the step's key). A reusing step adds
    :attr:`residual` to the stream in place of its blocks; a computing step hands what its blocks added to
    :meth:`keep`. Both are ``[rows, ...]``, row-leading, in whatever per-row layout the model chooses, as long as it
    is the same at both ends."""

    reuse: bool
    _rows: list[CacheRow]
    _forecast: Forecast = Forecast()
    _records: bool = False
    _ratios: list[list[torch.Tensor]] = field(default_factory=list)

    @property
    def residual(self) -> torch.Tensor:
        """What each row adds in place of its blocks: the policy's forecast from the row's own kept residuals."""
        forecast, predicted = self._forecast, []
        for row in self._rows:
            steps = [k for k in row.state.kwargs.get(RESIDUAL + "_steps", ()) if k >= row.floor]
            if not steps:
                raise RuntimeError(f"step cache: step {row.step} has no residual; a segment's first step must compute")
            kept = tuple(steps[: forecast.depth])
            weights = forecast.fitted_weights(row.num_steps, row.step, kept) or forecast.weights(
                tuple(row.time(k, forecast) for k in kept), row.time(row.step, forecast),
            )
            value = row.state.tensors[RESIDUAL]
            if weights != (1.0,):
                value = weights[0] * value
                for age in range(1, len(weights)):
                    value = value.add_(row.state.tensors[_aged(RESIDUAL, age)], alpha=weights[age])
            predicted.append(value)
        return predicted[0].unsqueeze(0) if len(predicted) == 1 else torch.stack(predicted)

    def rows(self, index: list[int]) -> BlockCache:
        """The view over a subset of the step's rows (a model running some rows through another expert)."""
        return BlockCache(
            reuse=self.reuse, _rows=[self._rows[i] for i in index], _forecast=self._forecast, _records=self._records,
        )

    def keep(self, residual: torch.Tensor) -> None:
        """Cache each row of ``residual`` (``[rows, ...]``, what the block stack added; fp32 recommended) for the
        row's later reusing steps."""
        for i, cached in enumerate(self._rows):
            row = residual[i]
            if self._records:
                self._record(cached.state, row, cached.final)
            _push(cached.state, RESIDUAL, cached.step, row, self._forecast.depth, cached.floor)

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
