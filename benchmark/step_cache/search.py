"""Calibrate a model in process and search plans for the harness to score.

    python -m benchmark.step_cache.search --model wan22_ti2v_5b --size 832x480 --frames 33 --steps 50 \\
        --prompts 4 --budgets 30,25,21 --out DIR [--sensitivity-prompts 2]

writes ``DIR/calibration.json`` (``calibrate.py``'s tables, per prompt and averaged) and ``DIR/searched.json``, a
methods file for ``run.py --methods-file``: for each budget of computed steps, the plan that minimizes the tables'
error (``plan_search.py``) as a reuse and as an order-1 forecast, free or with the first 20% of the steps kept, and,
with ``--sensitivity-prompts``, weighted by each step's measured effect on the final latents. ``--per-prompt`` also
writes ``DIR/per_prompt.json``: each prompt's own best plan per budget, the upper bound on what a per-sample rule
could gain over one shared plan.

``--refine NAME`` (repeatable) hill-climbs a method of ``searched.json`` or of ``--compare``'s file on the final
latents' measured error over the first ``--refine-prompts`` prompts, moving one computed step at a time, and adds the
result as ``ref_<NAME>``: started from a rule's plan, it says whether that plan is a local optimum.

The search prices a skipped step by its residual's error. Whether a searched plan is better than a rule's at the
same compute is for the harness to say, on decoded frames and on prompts the search did not see.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.step_cache import prompts as prompt_set
from benchmark.step_cache.backends.inproc import InprocBackend
from benchmark.step_cache.calibrate import Calibration, calibrate
from benchmark.step_cache.plan_search import plan_cost, refine_plan, search_plan
from mstar.model.components.diffusion.step_cache import ExplicitPlan, Forecast

_RETENTION = 0.2  # MagCache's default: the first fifth of the steps compute


def _method(name: str, label: str, computed: tuple[int, ...], steps: int, forecast: Forecast | None) -> dict[str, Any]:
    spec: dict[str, Any] = {"method": "plan", "computed": list(computed), "num_steps": steps}
    if forecast is not None:
        spec["forecast"] = {"order": forecast.order, "time": forecast.time}
    return {"name": name, "label": label, "step_cache": spec}


def searched_methods(
    cal: Calibration, budgets: list[int], restarts: tuple[int, ...] = (), forecast: Forecast = Forecast(order=1),
) -> list[dict[str, Any]]:
    """The methods file's entries for ``cal``: per budget, reuse and forecast plans, free and with retained steps,
    unweighted and (when ``cal`` has them) weighted by sensitivity."""
    retained = tuple(range(max(1, int(_RETENTION * cal.steps + 0.5))))
    variants = [("", "", None)]
    if cal.sensitivity:
        variants.append(("s", ", sensitivity-weighted", cal.sensitivity))
    out = []
    for budget in budgets:
        for tag, forced, kept in (("", (), "free"), ("r", retained, f"first {len(retained)} kept")):
            if budget < len(set(forced) | {0, cal.steps - 1, *restarts}):
                continue
            for weight_tag, weight_label, weights in variants:
                common = dict(budget=budget, forced=forced, restarts=restarts, weights=weights)
                try:
                    reuse = search_plan(cal.reuse_error, **common)
                except ValueError:
                    continue  # the budget cannot cover the schedule within the tables' depth
                out.append(_method(
                    f"dp0{tag}{weight_tag}_b{budget}", f"searched reuse, {budget}/{cal.steps}, {kept}{weight_label}",
                    reuse.computed, cal.steps, None,
                ))
                ahead = search_plan(cal.reuse_error, cal.linear_error, **common)
                out.append(_method(
                    f"dp1{tag}{weight_tag}_b{budget}",
                    f"searched order-1 forecast, {budget}/{cal.steps}, {kept}{weight_label}",
                    ahead.computed, cal.steps, forecast,
                ))
    return out


def per_prompt_plans(
    cal: Calibration, budgets: list[int], restarts: tuple[int, ...] = (),
) -> dict[str, dict[str, list[int]]]:
    """``{budget: {prompt index in the calibration: computed steps}}``: each prompt's own best reuse plan."""
    return {
        str(budget): {
            str(i): list(search_plan(run["reuse_error"], budget=budget, restarts=restarts).computed)
            for i, run in enumerate(cal.runs)
        }
        for budget in budgets
    }


def refined_method(
    backend: InprocBackend, method: dict[str, Any], prompts: list[tuple[str, int]], restarts: tuple[int, ...],
    max_evals: int, **request,
) -> dict[str, Any]:
    """``method``'s plan hill-climbed on the final latents' relative error over ``prompts``, as a plan method with
    the same forecast."""
    steps = request["steps"]
    policy = backend.policy_of(method["step_cache"])
    start = tuple(k for k, action in enumerate(policy.plan(steps, restarts)) if not action.reuse)
    backend.use(None)
    dense = [backend.run_loop(prompt, seed, **request).latents.float() for prompt, seed in prompts]

    def evaluate(computed: tuple[int, ...]) -> float:
        backend.use(ExplicitPlan(computed=computed, num_steps=steps, forecast=policy.forecast))
        errors = [
            float((backend.run_loop(prompt, seed, **request).latents.float() - ref).norm() / ref.norm())
            for (prompt, seed), ref in zip(prompts, dense, strict=True)
        ]
        return sum(errors) / len(errors)

    plan, evals = refine_plan(evaluate, start, steps, forced=restarts, max_evals=max_evals)
    backend.use(None)
    moved = len(set(plan.computed) - set(start))
    forecast = policy.forecast if policy.forecast.order else None
    return _method(
        f"ref_{method['name']}",
        f"{method['name']} hill-climbed on latents ({evals} runs, {moved} steps moved, error {plan.cost:.4f})",
        plan.computed, steps, forecast,
    )


def _compare(cal: Calibration, backend: InprocBackend, methods: list[dict[str, Any]], restarts) -> list[str]:
    """One line per method: its plan and the tables' price for it, next to the same-budget searched plans."""
    lines = []
    for method in methods:
        policy = backend.policy_of(method["step_cache"])
        computed = tuple(k for k, action in enumerate(policy.plan(cal.steps, restarts)) if not action.reuse)
        linear = cal.linear_error if policy.forecast.order else None
        price = plan_cost(computed, cal.reuse_error, linear, restarts=restarts)
        plan = "".join("C" if k in computed else "-" for k in range(cal.steps))
        lines.append(f"{method['name']:<16} {len(computed):>3} computed  table error {price:8.4f}  {plan}")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--prompts", type=int, default=4, help="prompts the tables average over")
    parser.add_argument("--budgets", required=True, help="computed-step counts to search plans for, comma-separated")
    parser.add_argument("--depth", type=int, default=6, help="the longest run of skipped steps the tables price")
    parser.add_argument("--time", choices=("step", "sigma"), default="step",
                        help="what the forecast's steps are spaced by")
    parser.add_argument("--sensitivity-prompts", type=int, default=0,
                        help="prompts to measure each step's effect on the final latents on (one run per step each)")
    parser.add_argument("--per-prompt", action="store_true", help="also write each prompt's own best plans")
    parser.add_argument("--compare", help="a methods file whose plans to price under the tables")
    parser.add_argument("--refine", action="append", default=[], help="a method name to hill-climb (repeatable)")
    parser.add_argument("--refine-prompts", type=int, default=2, help="prompts the hill-climb measures on")
    parser.add_argument("--refine-evals", type=int, default=40, help="plans the hill-climb may run per method")
    InprocBackend.add_arguments(parser)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = InprocBackend.from_args(args)
    request = dict(size=args.size, frames=args.frames, steps=args.steps)
    forecast = Forecast(order=1, time=args.time)
    chosen = [(p.text, p.seed) for p in prompt_set.take(args.prompts)]
    cal = calibrate(backend, chosen, depth=args.depth, forecast=forecast, **request)
    if args.sensitivity_prompts:
        measured = calibrate(
            backend, chosen[: args.sensitivity_prompts], depth=args.depth, forecast=forecast, sensitivity=True,
            **request,
        )
        cal.sensitivity = measured.sensitivity
    (out / "calibration.json").write_text(json.dumps(cal.to_json()))
    restarts = backend.restarts(**request)
    budgets = [int(b) for b in args.budgets.split(",")]
    methods = searched_methods(cal, budgets, restarts, forecast)
    compared = json.loads(Path(args.compare).read_text()) if args.compare else []
    by_name = {method["name"]: method for method in methods + compared}
    for name in args.refine:
        methods.append(refined_method(
            backend, by_name[name], chosen[: args.refine_prompts], restarts, args.refine_evals, **request,
        ))
    (out / "searched.json").write_text(json.dumps(methods, indent=1))
    if args.per_prompt:
        (out / "per_prompt.json").write_text(json.dumps(per_prompt_plans(cal, budgets, restarts)))
    print(f"mag_ratios = {[round(r, 5) for r in cal.mag_ratios]}")
    print("\n".join(_compare(cal, backend, methods + compared, restarts)))
    backend.close()


if __name__ == "__main__":
    main()
