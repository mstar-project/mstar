"""Which steps to compute, searched over a calibration run's error tables.

``calibrate.py`` measures, on dense runs, how far each step's block residual is from what a skipping step would
have added instead: the residual of the step ``d`` back (``reuse_error[k][d - 1]``), or the line through the steps
``d`` and ``d + g`` back (``linear_error[k][d - 1][g - 1]``). Given a budget of computed steps, :func:`search_plan`
finds the set that minimizes the sum (or the largest) of those errors over the skipped steps, exactly, by dynamic
programming over (previous computed step, last computed step). The result is an ``ExplicitPlan`` for
``mstar/model/components/diffusion/step_cache.py``.

The tables are local: they price a skipped step by its own residual's error, not by what that error does to the
final video. ``weights`` scales each step's error (a measured sensitivity, say), :func:`refine_plan` then moves
computed steps one at a time while a measured end-to-end error improves, and the harness is what checks that a
searched plan beats a rule at equal compute, on prompts the search did not see.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

_INF = float("inf")


class Plan(NamedTuple):
    computed: tuple[int, ...]
    # the objective's value over the skipped steps
    cost: float


def _step_cost(
    k: int, last: int, prev: int, reuse_error, linear_error, weights, floor: int,
) -> float:
    """The error of skipping step ``k`` when ``last`` and (before it) ``prev`` were the steps computed; ``prev``
    below ``floor`` (or -1) is not read, as a forecast reads nothing kept before a restart."""
    d = k - last
    if linear_error is not None and prev >= floor and prev >= 0 and last - prev <= len(linear_error[k][d - 1]):
        error = linear_error[k][d - 1][last - prev - 1]
    else:
        error = reuse_error[k][d - 1]
    return error if weights is None else error * weights[k]


def search_plan(
    reuse_error: Sequence[Sequence[float]], linear_error: Sequence[Sequence[Sequence[float]]] | None = None, *,
    budget: int, forced: Sequence[int] = (), restarts: Sequence[int] = (), weights: Sequence[float] | None = None,
    max_skip: int | None = None, objective: str = "sum",
) -> Plan:
    """The ``budget`` computed steps that minimize the skipped steps' error.

    Step 0, the last step, ``forced`` and ``restarts`` always compute (a restart is where the model changes
    weights: nothing kept before it is read after it). At most ``max_skip`` steps in a row skip (default and
    ceiling: the tables' depth). With ``linear_error`` a skipped step is priced as an order-1 forecast whenever two
    computed steps are in reach, else as a reuse. ``objective`` is ``sum`` or ``max``."""
    if objective not in ("sum", "max"):
        raise ValueError(f"objective must be 'sum' or 'max'; got {objective!r}")
    n, depth = len(reuse_error), len(reuse_error[0])
    max_skip = depth if max_skip is None else min(max_skip, depth)
    must = sorted({0, n - 1, *forced, *restarts})
    if any(not 0 <= k < n for k in must):
        raise ValueError(f"forced steps must lie in [0, {n}); got {must}")
    if not len(must) <= budget <= n:
        raise ValueError(f"budget must be in [{len(must)}, {n}] ({len(must)} steps always compute); got {budget}")
    is_forced = [False] * n
    for k in must:
        is_forced[k] = True
    floors = [max((r for r in restarts if r <= k), default=0) for k in range(n)]
    join = (lambda a, b: a + b) if objective == "sum" else max

    def segment(prev: int, last: int, nxt: int) -> float:
        """The cost of skipping every step strictly between ``last`` and ``nxt``."""
        cost = 0.0
        for k in range(last + 1, nxt):
            cost = join(cost, _step_cost(k, last, prev, reuse_error, linear_error, weights, floors[last]))
        return cost

    # best[(prev, last)] after `count` computed steps: (cost, the state it came from)
    layers: list[dict[tuple[int, int], tuple[float, tuple[int, int] | None]]] = [{(-1, 0): (0.0, None)}]
    for _ in range(budget - 1):
        nxt_layer: dict[tuple[int, int], tuple[float, tuple[int, int] | None]] = {}
        for (prev, last), (cost, _) in layers[-1].items():
            for nxt in range(last + 1, min(last + max_skip + 1, n - 1) + 1):
                total = join(cost, segment(prev, last, nxt))
                key = (last, nxt)
                if total < nxt_layer.get(key, (_INF, None))[0]:
                    nxt_layer[key] = (total, (prev, last))
                if is_forced[nxt]:
                    break  # no skip crosses a step that always computes
        layers.append(nxt_layer)
    ends = {key: value for key, value in layers[-1].items() if key[1] == n - 1}
    if not ends:
        raise ValueError(f"no plan of {budget} computed steps skips at most {max_skip} in a row over {n} steps")
    state = min(ends, key=lambda key: ends[key][0])
    cost, computed = ends[state][0], []
    for layer in reversed(layers):
        computed.append(state[1])
        state = layer[state][1]
    return Plan(tuple(reversed(computed)), cost)


def plan_cost(
    computed: Sequence[int], reuse_error, linear_error=None, *, restarts: Sequence[int] = (),
    weights: Sequence[float] | None = None, objective: str = "sum",
) -> float:
    """The objective of a given set of computed steps under the same tables (a rule's plan, for comparison). A skip
    longer than the tables' depth is priced at the deepest entry."""
    n, depth = len(reuse_error), len(reuse_error[0])
    done = sorted(set(computed) | {0})
    join = (lambda a, b: a + b) if objective == "sum" else max
    cost = 0.0
    for k in range(n):
        if k in done:
            continue
        earlier = [c for c in done if c < k]
        last, prev = earlier[-1], (earlier[-2] if len(earlier) > 1 else -1)
        floor = max((r for r in restarts if r <= last), default=0)
        if k - last > depth:
            error = reuse_error[k][depth - 1] * (1.0 if weights is None else weights[k])
        else:
            error = _step_cost(k, last, prev, reuse_error, linear_error, weights, floor)
        cost = join(cost, error)
    return cost


def refine_plan(
    evaluate: Callable[[tuple[int, ...]], float], computed: Sequence[int], num_steps: int, *,
    forced: Sequence[int] = (), max_skip: int | None = None, max_evals: int = 40,
) -> tuple[Plan, int]:
    """Hill-climb ``computed`` on ``evaluate`` (a measured error of running that plan, lower is better): move one
    computed step to a neighbouring skipped step whenever that helps, until no single move does or ``max_evals``
    plans have been run. Step 0, the last step and ``forced`` stay. Returns the plan and the evaluations spent."""
    fixed = {0, num_steps - 1, *forced}

    def allowed(plan: tuple[int, ...]) -> bool:
        return max_skip is None or all(b - a - 1 <= max_skip for a, b in zip(plan, plan[1:], strict=False))

    best = tuple(sorted(set(computed) | fixed))
    best_cost, evals, seen = evaluate(best), 1, {best}
    improved = True
    while improved and evals < max_evals:
        improved = False
        for step in best:
            if step in fixed:
                continue
            for moved in (step - 1, step + 1):
                candidate = tuple(sorted((set(best) - {step}) | {moved}))
                if moved in best or candidate in seen or not allowed(candidate):
                    continue
                seen.add(candidate)
                cost = evaluate(candidate)
                evals += 1
                if cost < best_cost:
                    best, best_cost, improved = candidate, cost, True
                    break
                if evals >= max_evals:
                    break
            if improved or evals >= max_evals:
                break
    return Plan(best, best_cost), evals
