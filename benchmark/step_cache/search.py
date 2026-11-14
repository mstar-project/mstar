"""Calibrate a model in process and search plans for the harness to score.

    python -m benchmark.step_cache.search --model wan22_ti2v_5b --size 832x480 --frames 33 --steps 50 \\
        --prompts 4 --budgets 30,25,21 --out DIR [--sensitivity-prompts 2]

writes ``DIR/calibration.json`` (``calibrate.py``'s tables, per prompt and averaged) and ``DIR/searched.json``, a
methods file for ``run.py --methods-file``: for each budget of computed steps, the plan that minimizes the tables'
error (``plan_search.py``) as a reuse (``dp0``), as an order-1 forecast (``dp1``) and as a forecast with the
calibration's fitted weights (``dp2``), free or with the first 20% of the steps kept, and, with
``--sensitivity-prompts``, weighted by each step's measured effect on the final latents. ``--per-prompt`` also
writes ``DIR/per_prompt.json``: each prompt's own best plan per budget, the upper bound on what a per-sample rule
could gain over one shared plan.

``--fit NAME`` (repeatable) adds ``fit_<NAME>``: that method's steps with the fitted weights in place of its own
forecast, which separates what the weights gain from what the plan gains. The run prints how much error the fitted
weights leave on a prompt they were not fitted on.

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
from benchmark.step_cache.calibrate import Calibration, calibrate, fit, fitted_forecast
from benchmark.step_cache.plan_search import plan_cost, refine_plan, search_plan
from mstar.model.components.diffusion.step_cache import ExplicitPlan, Forecast

_RETENTION = 0.2  # MagCache's default: the first fifth of the steps compute


def _kept_tag(ratio: float) -> str:
    """A retained plan's name tag: ``r`` at the default ratio, ``r<percent>`` at another."""
    return "r" if ratio == _RETENTION else f"r{round(ratio * 100)}"


def _method(name: str, label: str, computed: tuple[int, ...], steps: int, forecast: Forecast | None) -> dict[str, Any]:
    spec: dict[str, Any] = {"method": "plan", "computed": list(computed), "num_steps": steps}
    if forecast is not None:
        spec["forecast"] = {"order": forecast.order, "time": forecast.time}
        if forecast.fitted is not None:
            spec["forecast"]["fitted"] = {
                "num_steps": forecast.fitted.num_steps,
                "weights": [[step, list(kept), list(weights)] for step, kept, weights in forecast.fitted.entries],
            }
    return {"name": name, "label": label, "step_cache": spec}


def searched_methods(
    cal: Calibration, budgets: list[int], restarts: tuple[int, ...] = (), forecast: Forecast = Forecast(order=1),
    retentions: tuple[float, ...] = (_RETENTION,),
) -> list[dict[str, Any]]:
    """The methods file's entries for ``cal``: per budget, reuse and forecast plans, free and with the first
    ``retentions`` of the steps kept, unweighted and (when ``cal`` has them) weighted by sensitivity."""
    kept_sets = [((), "", "free")]
    for ratio in retentions:
        retained = tuple(range(max(1, int(ratio * cal.steps + 0.5))))
        kept_sets.append((retained, _kept_tag(ratio), f"first {len(retained)} kept"))
    variants = [("", "", None)]
    if cal.sensitivity:
        variants.append(("s", ", sensitivity-weighted", cal.sensitivity))
    out = []
    for budget in budgets:
        for forced, tag, kept in kept_sets:
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
                if cal.fit_weights is not None:
                    best = search_plan(cal.fit_reuse_error, cal.fit_linear_error, **common)
                    out.append(_method(
                        f"dp2{tag}{weight_tag}_b{budget}",
                        f"searched fitted forecast, {budget}/{cal.steps}, {kept}{weight_label}",
                        best.computed, cal.steps, fitted_forecast(cal, best.computed, restarts),
                    ))
    return out


def fit_method(backend: InprocBackend, cal: Calibration, method: dict[str, Any], restarts) -> dict[str, Any]:
    """``method``'s computed steps as a plan whose skipped steps take the calibration's fitted weights."""
    policy = backend.policy_of(method["step_cache"])
    computed = tuple(k for k, reuse in enumerate(policy.reuse_plan(cal.steps, restarts)) if not reuse)
    return _method(
        f"fit_{method['name']}", f"{method['name']}'s steps with fitted forecast weights", computed, cal.steps,
        fitted_forecast(cal, computed, restarts),
    )


def held_out_fit(cal: Calibration) -> list[str]:
    """How the fitted weights hold on a prompt they were not fitted on: for skips of 1 to 3 steps, the mean error
    of a reuse, of the order-1 forecast, of the weights fitted on every prompt, and of the weights fitted on the
    other prompts, each over the steps and prompts."""
    if cal.fit_weights is None or len(cal.runs) < 2:
        return []
    outs = [fit(cal.runs, cal.steps, cal.depth, held_out=i) for i in range(len(cal.runs))]
    lines = []
    for d in range(1, min(3, cal.depth) + 1):
        ks = range(d + 1, cal.steps)
        mean = lambda values: sum(values) / len(values)  # noqa: E731
        lines.append(
            f"skip {d}, from the two steps before it: reuse {mean([cal.reuse_error[k][d - 1] for k in ks]):.4f}, "
            f"order-1 {mean([cal.linear_error[k][d - 1][0] for k in ks]):.4f}, "
            f"fitted {mean([cal.fit_linear_error[k][d - 1][0] for k in ks]):.4f}, "
            f"fitted on the other prompts {mean([o['fit_linear_error'][k][d - 1][0] for o in outs for k in ks]):.4f}"
        )
    return lines


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
    max_evals: int, cal: Calibration | None = None, **request,
) -> dict[str, Any]:
    """``method``'s plan hill-climbed on the final latents' relative error over ``prompts``, as a plan method with
    the same forecast (a fitted one takes ``cal``'s weights for each plan tried)."""
    steps = request["steps"]
    policy = backend.policy_of(method["step_cache"])
    start = tuple(k for k, reuse in enumerate(policy.reuse_plan(steps, restarts)) if not reuse)
    backend.use(None)
    dense = [backend.run_loop(prompt, seed, **request).latents.float() for prompt, seed in prompts]

    def forecast_of(computed: tuple[int, ...]) -> Forecast:
        if policy.forecast.fitted is None or cal is None:
            return policy.forecast
        return fitted_forecast(cal, computed, restarts)

    def evaluate(computed: tuple[int, ...]) -> float:
        backend.use(ExplicitPlan(computed=computed, num_steps=steps, forecast=forecast_of(computed)))
        errors = [
            float((backend.run_loop(prompt, seed, **request).latents.float() - ref).norm() / ref.norm())
            for (prompt, seed), ref in zip(prompts, dense, strict=True)
        ]
        return sum(errors) / len(errors)

    plan, evals = refine_plan(evaluate, start, steps, forced=restarts, max_evals=max_evals)
    backend.use(None)
    moved = len(set(plan.computed) - set(start))
    forecast = forecast_of(plan.computed) if policy.forecast.order else None
    return _method(
        f"ref_{method['name']}",
        f"{method['name']} hill-climbed on latents ({evals} runs, {moved} steps moved, error {plan.cost:.4f})",
        plan.computed, steps, forecast,
    )


def plan_agreement(cal: Calibration, budgets: list[int], restarts: tuple[int, ...] = ()) -> list[str]:
    """One line per budget: how far each prompt's own best reuse plan is from the shared one, in computed steps
    they share and in the tables' price each prompt pays for running the shared plan instead of its own."""
    lines = []
    for budget in budgets:
        shared = search_plan(cal.reuse_error, budget=budget, restarts=restarts).computed
        overlap, excess = [], []
        for run in cal.runs:
            own = search_plan(run["reuse_error"], budget=budget, restarts=restarts)
            overlap.append(len(set(own.computed) & set(shared)))
            paid = plan_cost(shared, run["reuse_error"], restarts=restarts)
            excess.append(paid / own.cost - 1.0 if own.cost > 0 else 0.0)
        lines.append(
            f"budget {budget}: each prompt's own plan shares {min(overlap)}-{max(overlap)} of {budget} computed steps "
            f"with the shared plan; running the shared plan costs a prompt {100 * sum(excess) / len(excess):.1f}% "
            f"more table error than its own (worst {100 * max(excess):.1f}%)"
        )
    return lines


def _compare(cal: Calibration, backend: InprocBackend, methods: list[dict[str, Any]], restarts) -> list[str]:
    """One line per method: its plan and the tables' price for it, next to the same-budget searched plans."""
    lines = []
    for method in methods:
        policy = backend.policy_of(method["step_cache"])
        computed = tuple(k for k, reuse in enumerate(policy.reuse_plan(cal.steps, restarts)) if not reuse)
        if policy.forecast.fitted is not None:
            price = plan_cost(computed, cal.fit_reuse_error, cal.fit_linear_error, restarts=restarts)
        else:
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
    parser.add_argument("--retention", default=str(_RETENTION),
                        help="fractions of the steps the retained plans keep first, comma-separated")
    parser.add_argument("--compare", help="a methods file whose plans to price under the tables")
    parser.add_argument("--refine", action="append", default=[], help="a method name to hill-climb (repeatable)")
    parser.add_argument("--fit", action="append", default=[],
                        help="a method name whose steps to run with the fitted forecast weights (repeatable)")
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
    retentions = tuple(float(r) for r in args.retention.split(","))
    methods = searched_methods(cal, budgets, restarts, forecast, retentions)
    compared = json.loads(Path(args.compare).read_text()) if args.compare else []
    by_name = {method["name"]: method for method in methods + compared}
    for name in args.fit:
        methods.append(fit_method(backend, cal, by_name[name], restarts))
        by_name[methods[-1]["name"]] = methods[-1]
    for name in args.refine:
        methods.append(refined_method(
            backend, by_name[name], chosen[: args.refine_prompts], restarts, args.refine_evals, cal, **request,
        ))
    (out / "searched.json").write_text(json.dumps(methods, indent=1))
    if args.per_prompt:
        (out / "per_prompt.json").write_text(json.dumps(per_prompt_plans(cal, budgets, restarts)))
        print("\n".join(plan_agreement(cal, budgets, restarts)))
    print("\n".join(held_out_fit(cal)))
    print(f"mag_ratios = {[round(r, 5) for r in cal.mag_ratios]}")
    print("\n".join(_compare(cal, backend, methods + compared, restarts)))
    backend.close()


if __name__ == "__main__":
    main()
