"""The settings the harness compares: one ``step_cache`` mapping per method, per model.

A grid covers each part of a step-cache policy on its own and a few combinations
(``mstar/model/components/diffusion/step_cache.py`` reads the mappings): which steps skip (MagCache's calibrated
rule, a plain interval), what stands in for the skipped blocks (reuse, or a forecast of order 1 or 2), which blocks
skip (all, or all but the first few), and which guidance branches run. Every grid starts with ``dense``, the
reference the others are scored against.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple


class Method(NamedTuple):
    name: str
    label: str
    # the model's ``step_cache`` mapping; None runs every step dense
    step_cache: dict[str, Any] | None
    # the setting a deployment gets when it follows the docs, which the frontier check is about
    is_default: bool = False


DENSE = Method("dense", "dense: every step, every block", None)
# The cache on with nothing to skip: its output must be bit-identical to dense.
NO_SKIP = Method("noskip", "cache on, nothing skipped", {"method": "interval", "every": 1})


class PlanCost(NamedTuple):
    """What a plan computes: the steps that ran blocks, and the block-stack runs in units of one guidance branch
    through every block (a dense guided run of n steps is 2n)."""

    computed_steps: int
    branch_forwards: float


def plan_cost(actions: Sequence[Any], num_blocks: int, guided: bool = True) -> PlanCost:
    """The cost of a policy's ``plan(...)`` (its ``StepAction`` per step) on a stack of ``num_blocks``."""
    computed, forwards = 0, 0.0
    for action in actions:
        branches = 2 if guided and action.uncond else 1
        if action.skip is None:
            computed += 1
            forwards += branches
        else:
            lo, hi = action.skip.bounds(num_blocks)
            forwards += branches * (num_blocks - (hi - lo)) / num_blocks
    return PlanCost(computed, forwards)


class _MagCache(NamedTuple):
    threshold: float
    max_skip_steps: int
    retention_ratio: float = 0.2


class _Model(NamedTuple):
    # the MagCache settings earlier quality tables measured; the one a deployment following the docs gets; a faster
    # one for the combinations; and the ones also tried with a forecast
    magcache: tuple[_MagCache, ...]
    default: _MagCache
    fast: _MagCache
    forecast: tuple[_MagCache, ...]
    # the share of steps an interval rule computes first (A14B's layout forms in its high-noise steps)
    retention_ratio: float = 0.2
    # the DiT's blocks, for a model whose hook takes block ranges and a guidance cache; None without them
    num_blocks: int | None = None


_MODELS: dict[str, _Model] = {
    # MagCache's own defaults; the settings of the 1280x704x81, 50-step tables
    "wan22_ti2v_5b": _Model(
        magcache=(_MagCache(0.04, 2), _MagCache(0.06, 2), _MagCache(0.06, 3), _MagCache(0.12, 4), _MagCache(0.3, 5)),
        default=_MagCache(0.06, 2), fast=_MagCache(0.12, 4),
        forecast=(_MagCache(0.06, 3), _MagCache(0.12, 4), _MagCache(0.3, 5)), num_blocks=30,
    ),
    # the docs' recommendation keeps the first 40% of the steps
    "wan22_t2v_a14b": _Model(
        magcache=(
            _MagCache(0.04, 2), _MagCache(0.06, 3), _MagCache(0.12, 4, 0.4), _MagCache(0.3, 6, 0.4),
            _MagCache(0.3, 6, 0.5),
        ),
        default=_MagCache(0.12, 4, 0.4), fast=_MagCache(0.3, 6, 0.4),
        forecast=(_MagCache(0.06, 3), _MagCache(0.12, 4, 0.4), _MagCache(0.3, 6, 0.4)),
        retention_ratio=0.4, num_blocks=40,
    ),
    # the default is the example in the model's 1-GPU config
    "minimax_h3": _Model(
        magcache=(_MagCache(0.01, 1), _MagCache(0.015, 2), _MagCache(0.03, 2), _MagCache(0.06, 3)),
        default=_MagCache(0.06, 3), fast=_MagCache(0.06, 3),
        forecast=(_MagCache(0.015, 2), _MagCache(0.03, 2), _MagCache(0.06, 3)),
    ),
}
# the in-process backend's toy DiT: TI2V-5B's grid on its four blocks
_MODELS["tiny"] = _MODELS["wan22_ti2v_5b"]._replace(num_blocks=4)
MODELS = tuple(_MODELS)


class _Base(NamedTuple):
    """The rule that picks a method's skipping steps, before any other part is laid over it."""

    name: str
    label: str
    spec: dict[str, Any]


def _mag(setting: _MagCache, mag_ratios: str | None) -> _Base:
    name = f"mag-t{setting.threshold:g}-k{setting.max_skip_steps}"
    label = f"MagCache thr {setting.threshold:g}, K {setting.max_skip_steps}"
    spec: dict[str, Any] = {
        "method": "magcache", "threshold": setting.threshold, "max_skip_steps": setting.max_skip_steps,
    }
    if setting.retention_ratio != 0.2:
        name += f"-r{setting.retention_ratio:g}"
        label += f", first {setting.retention_ratio:.0%} computed"
        spec["retention_ratio"] = setting.retention_ratio
    if mag_ratios is not None:
        spec["mag_ratios"] = mag_ratios
    return _Base(name, label, spec)


def _interval(every: int, retention_ratio: float) -> _Base:
    spec = {"method": "interval", "every": every, "retention_ratio": retention_ratio}
    return _Base(f"int{every}", f"one step in {every} computes", spec)


def _with(base: _Base, *, forecast: int | None = None, front: int | None = None,
          guidance: tuple[int, float] | None = None) -> Method:
    """``base`` with a forecast order, the first ``front`` blocks always run, and a guidance cache
    ``(every, start)`` laid over it."""
    names, labels, spec = [base.name], [base.label], dict(base.spec)
    if forecast is not None:
        spec["forecast"] = forecast
        names.append(f"o{forecast}")
        labels.append(f"forecast order {forecast}" if forecast else "reuse")
    if front is not None:
        spec["blocks"] = {"front": front}
        names.append(f"f{front}")
        labels.append(f"first {front} block{'s' if front > 1 else ''} always run")
    if guidance is not None:
        every, start = guidance
        spec["guidance"] = {"every": every, "start": start}
        names.append(f"cfg{every}s{start:g}")
        labels.append(f"uncond on 1 computed step in {every} past the first {start:.0%}")
    return Method("+".join(n for n in names if n), ", ".join(labels), spec)


def grid(model: str, mag_ratios: str | None = None) -> tuple[Method, ...]:
    """Every method for ``model``, ``dense`` first. ``mag_ratios`` names the model's calibrated table the MagCache
    entries read (default: the model's ``default``)."""
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; the grids are {sorted(_MODELS)}")
    facts = _MODELS[model]
    methods = [DENSE]
    for setting in facts.magcache:
        base = _mag(setting, mag_ratios)
        methods.append(Method(base.name, base.label, base.spec, is_default=setting == facts.default))
    # which steps and what stands in: an interval, reused (FORA) or forecast (TaylorSeer)
    for every in (2, 3, 4):
        for order in (0, 1, 2):
            methods.append(_with(_interval(every, facts.retention_ratio), forecast=order))
    # MagCache's steps with a forecast in place of reuse
    methods += [_with(_mag(setting, mag_ratios), forecast=1) for setting in facts.forecast]
    if facts.num_blocks is not None:
        methods += _part_methods(facts, mag_ratios)
    names = [m.name for m in methods]
    if len(set(names)) != len(names):
        raise AssertionError(f"duplicate method names in the {model} grid: {sorted(names)}")
    return tuple(methods)


def _part_methods(facts: _Model, mag_ratios: str | None) -> list[Method]:
    """The methods that skip a block range or cache a guidance branch, and the combinations."""
    default, fast = _mag(facts.default, mag_ratios), _mag(facts.fast, mag_ratios)
    every3 = _interval(3, facts.retention_ratio)
    # which blocks: the front ones always run
    methods = [_with(every3, forecast=0, front=front) for front in (1, 2, 4) if front < facts.num_blocks]
    methods.append(_with(default, front=2))
    # Delta-DiT's shape: the rear half skips early, the front half late
    half = facts.num_blocks // 2
    methods.append(Method(
        "int3+ddit",
        f"one step in 3 computes; skips the rear {half} blocks in the first half, the front {half} after",
        {**every3.spec, "blocks": [{"until": 0.5, "front": half}, {"back": half}]},
    ))
    # which guidance branches: alone (every step computes), then on top of MagCache
    every_step = _Base("", "every step computes", dict(NO_SKIP.step_cache))
    methods += [_with(every_step, guidance=(2, start)) for start in (0.2, 0.3, 0.4)]
    methods += [_with(default, guidance=(2, 0.2)), _with(fast, guidance=(2, 0.2))]
    # combinations
    methods += [
        _with(default, forecast=1, guidance=(2, 0.2)),
        _with(every3, forecast=2, front=1),
        _with(fast, forecast=1, front=2, guidance=(2, 0.2)),
    ]
    return methods


def select(methods: Iterable[Method], prefixes: Sequence[str] | None) -> tuple[Method, ...]:
    """``dense`` plus the methods whose name starts with one of ``prefixes`` (all of them without any; ``none``
    alone is ``dense`` only, for a run of methods from a file)."""
    methods = tuple(methods)
    if not prefixes:
        return methods
    if list(prefixes) == ["none"]:
        return tuple(m for m in methods if m.step_cache is None)
    chosen = tuple(m for m in methods if m.step_cache is None or any(m.name.startswith(p) for p in prefixes))
    if len(chosen) <= 1:
        raise ValueError(f"--methods {list(prefixes)} matches none of {[m.name for m in methods]}")
    return chosen


def default_of(methods: Iterable[Method], name: str | None = None) -> Method:
    """The default among ``methods``: the one named, else the one the grid flags."""
    methods = tuple(methods)
    if name is not None:
        found = [m for m in methods if m.name == name]
        if not found:
            raise ValueError(f"--default {name!r} is not one of {[m.name for m in methods]}")
        return found[0]
    flagged = [m for m in methods if m.is_default]
    if len(flagged) != 1:
        raise ValueError(f"expected one default among the methods, found {[m.name for m in flagged]}")
    return flagged[0]


def load_methods(path: str | Path) -> tuple[Method, ...]:
    """The methods of a JSON file: a list of ``{"name", "step_cache", "label"?, "is_default"?}``, e.g. the plans a
    search over a calibration run picked."""
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path}: a methods file is a JSON list of methods")
    methods = []
    for entry in entries:
        unknown = set(entry) - {"name", "label", "step_cache", "is_default"}
        if unknown or not isinstance(entry.get("name"), str) or not isinstance(entry.get("step_cache"), dict):
            raise ValueError(
                f"{path}: a method is {{name, step_cache, label?, is_default?}} with a step_cache mapping; got {entry}"
            )
        methods.append(Method(
            entry["name"], str(entry.get("label", entry["name"])), entry["step_cache"], bool(entry.get("is_default")),
        ))
    return tuple(methods)


def combine(methods: Iterable[Method], extra: Iterable[Method]) -> tuple[Method, ...]:
    """``methods`` followed by ``extra``. A name both carry has to mean the same ``step_cache``, and a default
    flagged in ``extra`` takes the flag from ``methods``."""
    methods, extra = list(methods), list(extra)
    if sum(m.is_default for m in extra) > 1:
        raise ValueError(f"more than one default among {[m.name for m in extra if m.is_default]}")
    if any(m.is_default for m in extra):
        methods = [m._replace(is_default=False) for m in methods]
    by_name = {m.name: m for m in methods}
    for method in extra:
        known = by_name.get(method.name)
        if known is None:
            by_name[method.name] = method
            methods.append(method)
        elif known.step_cache != method.step_cache:
            raise ValueError(f"two methods are named {method.name!r} with different step_cache mappings")
        elif method.is_default:
            methods[methods.index(known)] = by_name[method.name] = known._replace(is_default=True)
    return tuple(methods)
