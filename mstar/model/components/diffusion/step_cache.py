"""Opt-in step caching for the DiT scaffold: leave blocks (or a guidance branch) out of a step and predict what they
would have added from the steps that ran them.

A policy is one object with four independent parts, each decided from host values alone (the step index, the
request's step count and schedule, a calibrated table), so no part costs a host sync and every rank of a lockstep
instance derives the same answer from the same config (AGENTS.md invariants 7 and 11):

* **which steps skip** is the policy's class: :class:`MagCache` (a calibrated rule), :class:`Interval` (every n-th
  step computes), :class:`ExplicitPlan` (the steps an offline search picked);
* **which blocks** a skipping step leaves out (:class:`BlockRange`): the whole stack by default, or all but the first
  ``front`` and last ``back`` blocks, optionally another range late in the schedule;
* **what stands in for them** (:class:`Forecast`): the residual those blocks added at the last step that ran them,
  or its extrapolation from the last two or three such steps;
* **which guidance branches run** (:class:`GuidanceCache`): on some computing steps, and on every skipping one, only
  the conditional branch, the unconditional output predicted from the last steps that ran both.

A skipping step still runs the model's input embedding and output head. The model opts in by passing the policy to
``DenoiseLoopSubmodule(step_cache=...)`` and routing its block stack through :func:`run_blocks` with what the step's
:class:`BlockCache` (:meth:`DenoiseLoopSubmodule.block_cache`) hands it. Everything cached is per-request state the
engine owns and frees with the request (``request_state``), each rank holding its own: its CFG branch, its
sequence-parallel shard.

A rule that reads a device value each step (TeaCache's modulated-input distance, DBCache's first-block probe) would
need a reduce across the instance and a host read per step; none is implemented.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

import torch

logger = logging.getLogger(__name__)

# Where a request's cached tensors live in its ``request_state``: the newest under the name itself, older ones under
# ``<name>_<age>``, and the steps they were kept at in ``kwargs[<name>_steps]``.
RESIDUAL = "step_cache_residual"
GUIDANCE = "step_cache_guidance"
_RATIOS = "step_cache_ratios"


class BlockRange(NamedTuple):
    """The blocks a skipping step leaves out: all but the first ``front`` and the last ``back``."""

    front: int = 0
    back: int = 0

    def bounds(self, num_blocks: int) -> tuple[int, int]:
        """``(first skipped, first not skipped)`` in a stack of ``num_blocks``."""
        if self.front < 0 or self.back < 0 or self.front + self.back >= num_blocks:
            raise ValueError(
                f"step cache: front {self.front} and back {self.back} must be >= 0 and leave some of {num_blocks} "
                f"blocks to skip"
            )
        return self.front, num_blocks - self.back


class StepAction(NamedTuple):
    """What one step does with the cache. It joins the step's key, so rows share a forward only at the same one."""

    # the blocks this step leaves out, adding the forecast of their residual instead; None runs every block
    skip: BlockRange | None = None
    # the blocks whose residual this step keeps for later skipping steps; None keeps nothing
    keep: BlockRange | None = None
    # whether the unconditional guidance branch runs; False predicts its output from the conditional one's
    uncond: bool = True

    @property
    def reuse(self) -> bool:
        return self.skip is not None


@dataclass(frozen=True)
class Forecast:
    """How a skipped quantity is predicted from the steps that computed it.

    ``order`` 0 reuses the last value. Order 1 or 2 extrapolates through the last two or three, a Taylor expansion
    whose derivatives are divided differences (TaylorSeer, Liu et al., arXiv 2503.06923, at equal spacing), each
    derivative term scaled by ``damping`` per order. ``time`` is what the steps are spaced by: their index or the
    schedule's sigma. A request with fewer kept values than the order needs uses the order it has."""

    order: int = 0
    damping: float = 1.0
    time: str = "step"

    def __post_init__(self):
        if self.order not in (0, 1, 2) or not 0.0 <= self.damping <= 1.0 or self.time not in ("step", "sigma"):
            raise ValueError(
                f"step cache forecast: order in (0, 1, 2), damping in [0, 1], time 'step' or 'sigma'; got "
                f"{self.order}, {self.damping}, {self.time!r}"
            )

    @property
    def depth(self) -> int:
        """Kept values the forecast reads."""
        return self.order + 1

    @functools.lru_cache(maxsize=1024)  # noqa: B019 - frozen and hashable; a few schedules' worth of float tuples
    def weights(self, kept: tuple[float, ...], now: float) -> tuple[float, ...]:
        """The prediction at time ``now`` as weights over the values kept at times ``kept``, newest first."""
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


@dataclass(frozen=True)
class GuidanceCache:
    """Which steps run the unconditional guidance branch (FasterCache's CFG cache, Lv et al., arXiv 2410.19355,
    without its frequency reweighting).

    The first ``start`` of the steps and the last one run both branches; in between every ``every``-th computing step
    does, and the rest run the conditional branch alone and predict ``uncond - cond`` by ``forecast`` from the steps
    that ran both. A step that skips its blocks never runs the unconditional branch."""

    every: int = 2
    start: float = 0.2
    forecast: Forecast = Forecast()

    def __post_init__(self):
        if self.every < 1 or not 0.0 <= self.start <= 1.0:
            raise ValueError(f"step cache guidance: every >= 1 and start in [0, 1]; got {self.every}, {self.start}")


class _Stage(NamedTuple):
    """A block range and the fraction of the schedule it applies up to."""

    until: float
    blocks: BlockRange


@dataclass(frozen=True, kw_only=True)
class StepCachePolicy:
    """A request's cache plan. The subclass says which steps skip (:meth:`reuse_plan`); the fields say which blocks,
    what stands in for them, and which guidance branches run. A pure function of host values, memoized."""

    forecast: Forecast = Forecast()
    # the blocks a skipping step leaves out: one range, or (fraction of the schedule, range) stages, each up to its
    # fraction (Delta-DiT skips the rear blocks early and the front ones late). A stage's first step computes.
    blocks: BlockRange | tuple[_Stage, ...] = BlockRange()
    guidance: GuidanceCache | None = None

    # Calibration policies record each computed step's residual magnitude ratio.
    records: ClassVar[bool] = False

    def __post_init__(self):
        if not isinstance(self.blocks, BlockRange):
            stages = tuple(_Stage(float(until), BlockRange(*blocks)) for until, blocks in self.blocks)
            if not stages or any(a.until >= b.until for a, b in zip(stages, stages[1:], strict=False)):
                raise ValueError("step cache blocks: stages need increasing 'until' fractions")
            object.__setattr__(self, "blocks", stages)

    def reuse_plan(self, num_steps: int, segment_starts: tuple[int, ...] = ()) -> tuple[bool, ...]:
        """One flag per step, True where the step skips. Step 0 never does, as nothing is cached yet, and neither
        does a step in ``segment_starts``, where the model changes weights (``step_cache_segments``)."""
        raise NotImplementedError

    def reuses(self, step: int, num_steps: int, segment_starts: tuple[int, ...] = ()) -> bool:
        return self.reuse_plan(num_steps, segment_starts)[step]

    @property
    def partial(self) -> bool:
        """Whether a skipping step still runs some blocks, which the model's forward has to support."""
        return self.blocks != BlockRange()

    def _stage_of(self, num_steps: int) -> tuple[BlockRange, ...]:
        if isinstance(self.blocks, BlockRange):
            return (self.blocks,) * num_steps
        return tuple(
            next((s.blocks for s in self.blocks if k < s.until * num_steps), self.blocks[-1].blocks)
            for k in range(num_steps)
        )

    @functools.lru_cache(maxsize=64)  # noqa: B019 - frozen and hashable; a handful of step counts per deployment
    def restarts(self, num_steps: int, segment_starts: tuple[int, ...] = ()) -> tuple[int, ...]:
        """The steps where what was kept before stops being usable: the model's segment starts and the first step of
        each block stage. They compute, and forecasts read nothing kept before them."""
        ranges = self._stage_of(num_steps)
        stage_starts = {k for k in range(1, num_steps) if ranges[k] != ranges[k - 1]}
        return tuple(sorted(stage_starts.union(k for k in segment_starts if 0 < k < num_steps)))

    @functools.lru_cache(maxsize=64)  # noqa: B019 - frozen and hashable; a handful of step counts per deployment
    def plan(self, num_steps: int, segment_starts: tuple[int, ...] = ()) -> tuple[StepAction, ...]:
        """Every step's action."""
        restarts = self.restarts(num_steps, segment_starts)
        reuse = self.reuse_plan(num_steps, restarts)
        ranges = self._stage_of(num_steps)
        both_until = 0 if self.guidance is None else max(1, int(self.guidance.start * num_steps + 0.5))
        actions, since_both = [], 0
        for k in range(num_steps):
            if reuse[k]:
                actions.append(StepAction(skip=ranges[k], uncond=self.guidance is None))
                continue
            uncond = True
            if self.guidance is not None and both_until <= k < num_steps - 1 and k not in restarts:
                since_both += 1
                uncond = since_both >= self.guidance.every
            if uncond:
                since_both = 0
            actions.append(StepAction(keep=ranges[k], uncond=uncond))
        return tuple(actions)


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
    """MagCache's skip rule (Ma et al., arXiv 2506.09045) over a calibrated table of residual magnitude ratios.

    ``mag_ratios[k]`` is the mean over tokens of ``|r_k| / |r_(k-1)|``, ``r_k`` the block stack's residual at step
    ``k`` of a dense run (:class:`MagCacheCalibration` logs it); a table calibrated at another step count is resampled
    by nearest index. Past the first ``retention_ratio`` of the steps, a step skips while the error accumulated since
    the last computed step, ``sum |1 - prod(ratios)|``, stays at or under ``threshold`` and no more than
    ``max_skip_steps`` steps in a row have skipped. The last step always computes, and so does each step in
    ``segment_starts`` (another expert's first step), where the accumulated error starts over: the table's entry
    there compares two experts' residuals and is never read."""

    mag_ratios: tuple[float, ...]
    threshold: float = 0.06
    max_skip_steps: int = 2
    retention_ratio: float = 0.2

    def __post_init__(self):
        super().__post_init__()
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
    ``every`` 1 computes every step, for a policy that only caches the guidance branch. The last step and each
    segment start compute."""

    every: int = 2
    retention_ratio: float = 0.2

    def __post_init__(self):
        super().__post_init__()
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
        super().__post_init__()
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
        return Forecast(**_options(spec, "forecast", {"order", "damping", "time"}))
    return Forecast(order=int(spec))


def _block_range(spec: Mapping[str, Any]) -> BlockRange:
    return BlockRange(int(spec.get("front", 0)), int(spec.get("back", 0)))


def _blocks(spec: Any) -> BlockRange | tuple[_Stage, ...]:
    if spec is None:
        return BlockRange()
    if isinstance(spec, Mapping):
        return _block_range(_options(spec, "blocks", {"front", "back"}))
    return tuple(
        (float(stage.get("until", 1.0)), _block_range(_options(stage, "blocks", {"until", "front", "back"})))
        for stage in spec
    )


def _guidance(spec: Any) -> GuidanceCache | None:
    if spec is None:
        return None
    spec = _options(spec, "guidance", {"every", "start", "forecast"})
    return GuidanceCache(forecast=_forecast(spec.pop("forecast", None)), **spec)


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

    Any of the first three also takes ``forecast`` (an order, or ``{order, damping, time}``), ``blocks``
    (``{front, back}``, or a list of ``{until, front, back}`` stages) and ``guidance`` (``{every, start, forecast}``).
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
    parts = {
        "forecast": _forecast(spec.pop("forecast", None)),
        "blocks": _blocks(spec.pop("blocks", None)),
        "guidance": _guidance(spec.pop("guidance", None)),
    }
    unknown = set(spec) - _RULE_OPTIONS[method]
    if unknown:
        raise ValueError(f"step_cache: unknown {method} options {sorted(unknown)}")
    if method == "interval":
        return Interval(**spec, **parts)
    if method == "plan":
        if "computed" not in spec or "num_steps" not in spec:
            raise ValueError("step_cache: plan needs 'computed' and 'num_steps'")
        return ExplicitPlan(computed=tuple(spec["computed"]), num_steps=int(spec["num_steps"]), **parts)
    ratios = spec.pop("mag_ratios", "default")
    if isinstance(ratios, str):
        if not calibrated or ratios not in calibrated:
            raise ValueError(f"step_cache: no calibrated mag_ratios table {ratios!r} for this model")
        ratios = calibrated[ratios]
    return MagCache(mag_ratios=tuple(ratios), **spec, **parts)


def run_blocks(
    blocks: Sequence[Any], hidden: torch.Tensor, call: Callable[[Any, torch.Tensor], torch.Tensor], *,
    skip: BlockRange | None = None, cached_residual: torch.Tensor | None = None, keep: BlockRange | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """A model's block stack over the stream ``hidden``, ``call(block, hidden)`` running one block.

    With ``skip``, its blocks are left out and ``cached_residual`` (fp32, the stream's shape) is added in their
    place. With ``keep``, also returns what its blocks added, fp32. A pure tensor function of static ranges, so it
    traces into a compiled forward."""
    num_blocks = len(blocks)
    skip_lo, skip_hi = (num_blocks, num_blocks) if skip is None else skip.bounds(num_blocks)
    keep_lo, keep_hi = (-1, -1) if keep is None else keep.bounds(num_blocks)
    before = residual = None
    for i, block in enumerate(blocks):
        if i == keep_lo:
            before = hidden
        if i == skip_lo:
            hidden = (hidden.float() + cached_residual).type_as(hidden)
        if not skip_lo <= i < skip_hi:
            hidden = call(block, hidden)
        if i == keep_hi - 1:
            residual = hidden.float() - before.float()
    return hidden, residual


def _push(state: Any, name: str, step: int, value: torch.Tensor, depth: int, floor: int) -> None:
    """Keep ``value`` as the request's newest ``name``, with up to ``depth - 1`` older ones kept at or after step
    ``floor``."""
    steps = [k for k in state.kwargs.get(name + "_steps", ()) if k >= floor][: depth - 1]
    for age in range(len(steps), 0, -1):
        state.tensors[_aged(name, age)] = state.tensors[_aged(name, age - 1)]
    for age in range(len(steps) + 1, depth):
        # what a restart left behind (another expert's residuals)
        state.tensors.pop(_aged(name, age), None)
    state.tensors[name] = value
    state.kwargs[name + "_steps"] = [step, *steps]


def _aged(name: str, age: int) -> str:
    return name if age == 0 else f"{name}_{age}"


@dataclass
class CacheRow:
    """One row of a step: its request's state and the host values its forecasts read."""

    state: Any
    step: int
    # the schedule's sigma per step, for a forecast spaced by it
    sigmas: tuple[float, ...] | None = None
    # nothing kept before this step is usable (the row's last restart)
    floor: int = 0
    final: bool = False

    def time(self, step: int, forecast: Forecast) -> float:
        return float(step) if forecast.time == "step" else self.sigmas[step]


@dataclass
class BlockCache:
    """One step's view of its rows' cache, handed to the model's forward.

    ``action`` is the step's, the same for every row (it is part of the step's key). A skipping step adds
    :attr:`residual` to the stream in place of ``action.skip``'s blocks; a computing step hands what
    ``action.keep``'s blocks added to :meth:`keep`. Both are ``[rows, ...]``, row-leading, in whatever per-row layout
    the model chooses, as long as it is the same at both ends. Likewise :attr:`guidance_delta` and
    :meth:`keep_guidance` for ``uncond - cond`` of the model's output."""

    action: StepAction
    _rows: list[CacheRow]
    _forecast: Forecast = Forecast()
    _guidance: Forecast = Forecast()
    # whether the policy caches a guidance branch: the model then keeps the conditional branch's residual alone,
    # and ``uncond - cond`` on the steps that run both
    caches_guidance: bool = False
    _records: bool = False
    _ratios: list[list[torch.Tensor]] = field(default_factory=list)

    @property
    def reuse(self) -> bool:
        """Whether the step skips blocks. A model that reads only this takes whole-stack skips."""
        return self.action.skip is not None

    @property
    def residual(self) -> torch.Tensor:
        """The forecast of what the skipped blocks would add, per row."""
        return self._predict(RESIDUAL, self._forecast)

    def keep(self, residual: torch.Tensor) -> None:
        """Cache each row of ``residual`` (``[rows, ...]``, what the kept blocks added; fp32 recommended) for the
        row's later skipping steps."""
        for i, row in enumerate(self._rows):
            value = residual[i]
            if self._records:
                self._record(row, value)
            _push(row.state, RESIDUAL, row.step, value, self._forecast.depth, row.floor)

    @property
    def guidance_delta(self) -> torch.Tensor:
        """The forecast of ``uncond - cond``, per row, on a step that runs the conditional branch alone."""
        return self._predict(GUIDANCE, self._guidance)

    def keep_guidance(self, delta: torch.Tensor) -> None:
        """Cache each row of ``delta`` (``[rows, ...]``, ``uncond - cond``) on a step that ran both branches."""
        for i, row in enumerate(self._rows):
            _push(row.state, GUIDANCE, row.step, delta[i], self._guidance.depth, row.floor)

    def rows(self, index: list[int]) -> BlockCache:
        """The view over a subset of the step's rows (a model running some rows through another expert)."""
        return BlockCache(
            action=self.action, _rows=[self._rows[i] for i in index], _forecast=self._forecast,
            _guidance=self._guidance, caches_guidance=self.caches_guidance, _records=self._records,
        )

    def _predict(self, name: str, forecast: Forecast) -> torch.Tensor:
        predicted = []
        for row in self._rows:
            steps = [k for k in row.state.kwargs.get(name + "_steps", ()) if k >= row.floor]
            if not steps:
                raise RuntimeError(f"step cache: step {row.step} found no {name}; a segment's first step must compute")
            weights = forecast.weights(
                tuple(row.time(k, forecast) for k in steps[: forecast.depth]), row.time(row.step, forecast),
            )
            value = row.state.tensors[name]
            if len(weights) > 1:
                value = weights[0] * value
                for age in range(1, len(weights)):
                    value = value.add_(row.state.tensors[_aged(name, age)], alpha=weights[age])
            predicted.append(value)
        return predicted[0].unsqueeze(0) if len(predicted) == 1 else torch.stack(predicted)

    def _record(self, row: CacheRow, value: torch.Tensor) -> None:
        prev = row.state.tensors.get(RESIDUAL)
        ratios = row.state.kwargs.setdefault(_RATIOS, [])
        if prev is not None:
            # per-token magnitude ratio over the hidden dim, averaged
            ratios.append((value.norm(dim=-1) / prev.norm(dim=-1).clamp_min(1e-12)).mean())
        if row.final and ratios:
            table = [round(float(r), 5) for r in torch.stack(ratios).tolist()]
            logger.info("step cache calibration (%d steps): mag_ratios = [1.0, %s]", len(table) + 1,
                        ", ".join(str(r) for r in table))
