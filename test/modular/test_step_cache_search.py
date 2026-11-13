"""The step cache harness's in-process pieces (``benchmark/step_cache``): the plan search over calibration tables, the
calibration run itself and the in-process backend, on Wan2.2's DiT at toy size. CPU only, random weights."""

from __future__ import annotations

import itertools
import json

import pytest
import torch

from benchmark.step_cache.backends.inproc import InprocBackend
from benchmark.step_cache.calibrate import Calibration, calibrate
from benchmark.step_cache.methods import Method
from benchmark.step_cache.plan_search import plan_cost, refine_plan, search_plan
from mstar.model.components.diffusion.step_cache import ExplicitPlan, Forecast

# --------------------------------------------------------------------- the search


def _tables(n: int, depth: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    reuse = [[float(v) * d for d, v in enumerate(torch.rand(depth, generator=gen), start=1)] for _ in range(n)]
    linear = [[[float(v) for v in torch.rand(depth, generator=gen)] for _ in range(depth)] for _ in range(n)]
    return reuse, linear


def _brute(reuse, linear, budget, objective, forced=(), restarts=()):
    n, depth = len(reuse), len(reuse[0])
    best = None
    for inner in itertools.combinations(range(1, n - 1), budget - 2):
        computed = (0, *inner, n - 1)
        if any(k not in computed for k in (*forced, *restarts)):
            continue
        if any(b - a - 1 > depth for a, b in itertools.pairwise(computed)):
            continue
        cost = plan_cost(computed, reuse, linear, restarts=restarts, objective=objective)
        if best is None or cost < best[1]:
            best = (computed, cost)
    return best


@pytest.mark.parametrize("objective", ["sum", "max"])
@pytest.mark.parametrize("with_forecast", [False, True])
def test_the_search_finds_the_exhaustive_optimum(objective, with_forecast):
    reuse, linear = _tables(9, 3)
    for budget in (3, 4, 6, 9):
        found = search_plan(reuse, linear if with_forecast else None, budget=budget, objective=objective)
        computed, cost = _brute(reuse, linear if with_forecast else None, budget, objective)
        assert found.cost == pytest.approx(cost) and len(found.computed) == budget
        assert plan_cost(found.computed, reuse, linear if with_forecast else None, objective=objective) == (
            pytest.approx(found.cost)
        )
    assert search_plan(reuse, budget=9).cost == 0.0


def test_the_search_computes_forced_steps_and_reads_nothing_across_a_restart():
    reuse, linear = _tables(9, 3, seed=1)
    found = search_plan(reuse, linear, budget=5, forced=(1,), restarts=(5,))
    assert {0, 1, 5, 8} <= set(found.computed)
    assert found.cost == pytest.approx(_brute(reuse, linear, 5, "sum", forced=(1,), restarts=(5,))[1])
    # a forecast across the restart would be cheaper: make it free and check it is not taken
    for k in range(9):
        for d in range(3):
            linear[k][d] = [0.0, 0.0, 0.0]
    restarted = search_plan(reuse, linear, budget=4, restarts=(5,))
    assert restarted.cost > 0.0 and search_plan(reuse, linear, budget=4).cost == 0.0


def test_the_search_puts_computed_steps_where_skipping_costs_most():
    n, depth = 12, 4
    reuse = [[(10.0 if k < 5 else 0.1) * d for d in range(1, depth + 1)] for k in range(n)]
    found = search_plan(reuse, budget=7)
    assert set(range(5)) <= set(found.computed)
    with pytest.raises(ValueError, match="budget"):
        search_plan(reuse, budget=1)
    with pytest.raises(ValueError, match="in a row"):
        search_plan(reuse, budget=2)


# --------------------------------------------------------------------- the in-process backend and calibration

_REQUEST = dict(size="64x64", frames=5, steps=8)


@pytest.fixture(scope="module")
def backend():
    torch.set_num_threads(2)
    return InprocBackend("tiny")


def test_the_backend_runs_a_policy_and_counts_what_it_computed(backend):
    backend.configure(Method("dense", "dense", None))
    dense = backend.generate("a red fox", 7, **_REQUEST)
    assert dense.frames.dtype == torch.uint8 and dense.frames.shape[-1] == 3 and dense.frames.shape[0] == 2
    assert (dense.computed_steps, dense.branch_forwards) == (8, 16.0)
    again = backend.generate("a red fox", 7, **_REQUEST)
    assert torch.equal(dense.frames, again.frames)
    backend.configure(Method("noskip", "cache on, nothing skipped", {"method": "interval", "every": 1}))
    assert torch.equal(backend.generate("a red fox", 7, **_REQUEST).frames, dense.frames)
    spec = {"method": "interval", "every": 2, "retention_ratio": 0.0, "blocks": {"front": 1},
            "guidance": {"every": 2, "start": 0.0}}
    backend.configure(Method("combo", "interval 2, front 1, guidance 2", spec))
    cached = backend.generate("a red fox", 7, **_REQUEST)
    assert cached.extra["plan"] == "C-c-C-cC" and cached.computed_steps == 5
    # both branches on steps 0, 4, 7; one on 2 and 6; a skipping step runs one branch through 1 of 4 blocks
    assert cached.branch_forwards == pytest.approx(3 * 2 + 2 * 1 + 3 * 0.25)
    assert not torch.equal(cached.frames, dense.frames)
    with pytest.raises(ValueError, match="unknown method"):
        backend.configure(Method("bad", "bad", {"method": "teacache"}))


def test_a_calibration_run_is_dense_and_its_tables_price_each_skip(backend):
    prompts = [("a red fox", 7), ("two boats", 8)]
    cal = calibrate(backend, prompts, depth=3, forecast=Forecast(order=1, time="sigma"), sensitivity=True, **_REQUEST)
    assert (cal.steps, cal.depth, len(cal.runs)) == (8, 3, 2) and backend.policy is None
    assert len(cal.mag_ratios) == 8 and cal.mag_ratios[0] == 1.0
    # step k has an entry for each distance d <= k; the rest is padding
    for k in range(8):
        for d in range(1, 4):
            assert (cal.reuse_error[k][d - 1] < float("inf")) == (d <= k)
            for g in range(1, 4):
                assert (cal.linear_error[k][d - 1][g - 1] < float("inf")) == (d + g <= k)
    assert cal.sensitivity[0] == cal.sensitivity[-1] == 0.0 and all(s > 0 for s in cal.sensitivity[1:-1])
    assert Calibration.from_json(cal.to_json()) == cal
    # the tables are the ones the policy's own forecast would have produced: skipping step k alone, from the
    # steps the table names, changes the final latents; a plan from the search runs through the backend
    plan = search_plan(cal.reuse_error, cal.linear_error, budget=5, weights=cal.sensitivity)
    backend.use(ExplicitPlan(computed=plan.computed, num_steps=8, forecast=Forecast(order=1, time="sigma")))
    loop = backend.run_loop("a red fox", 7, **_REQUEST)
    assert tuple(k for k, action in enumerate(loop.plan) if not action.reuse) == plan.computed
    backend.use(None)


def test_the_reuse_table_is_the_error_of_the_residual_a_skipping_step_adds(backend):
    """reuse_error[k][0] is |r_k - r_(k-1)| / |r_k| of the dense run: recompute it from the states directly."""
    from mstar.model.components.diffusion.step_cache import RESIDUAL, Interval

    cal = calibrate(backend, [("a red fox", 7)], depth=2, **_REQUEST)
    kept = []
    backend.use(Interval(every=1))
    backend.run_loop("a red fox", 7, **_REQUEST, observer=lambda k, state, out: kept.append(state.tensors[RESIDUAL]))
    backend.use(None)
    for k in range(1, 8):
        want = float((kept[k] - kept[k - 1]).norm() / kept[k].norm())
        assert cal.reuse_error[k][0] == pytest.approx(want, rel=1e-5)
    line = 2 * kept[4] - kept[3]
    assert cal.linear_error[5][0][0] == pytest.approx(float((kept[5] - line).norm() / kept[5].norm()), rel=1e-5)


def test_the_search_writes_a_methods_file_whose_plans_spend_their_budget(tmp_path, backend):
    from benchmark.step_cache import search

    search.main([
        "--model", "tiny", "--out", str(tmp_path), "--size", "64x64", "--frames", "5", "--steps", "8",
        "--prompts", "2", "--budgets", "5,6", "--depth", "3", "--sensitivity-prompts", "1", "--per-prompt",
    ])
    methods = json.loads((tmp_path / "searched.json").read_text())
    assert {"dp0_b5", "dp1_b5", "dp0r_b6", "dp1rs_b6"} <= {m["name"] for m in methods}
    for method in methods:
        plan = backend.policy_of(method["step_cache"]).plan(8)
        assert sum(not action.reuse for action in plan) == int(method["name"].rsplit("_b", 1)[1])
        if "r" in method["name"].split("_")[0][3:]:
            assert not plan[0].reuse and not plan[1].reuse  # the first fifth of 8 steps is kept
    per_prompt = json.loads((tmp_path / "per_prompt.json").read_text())
    assert set(per_prompt) == {"5", "6"} and set(per_prompt["5"]) == {"0", "1"}
    cal = Calibration.from_json(json.loads((tmp_path / "calibration.json").read_text()))
    agreement = search.plan_agreement(cal, [5, 6])
    assert len(agreement) == 2 and agreement[0].startswith("budget 5: each prompt's own plan shares ")
    assert Calibration.from_json(json.loads((tmp_path / "calibration.json").read_text())).sensitivity is not None


def test_refining_moves_computed_steps_downhill_within_its_budget_of_runs():
    """A measured error with one best place per computed step: the hill-climb walks there, keeps the ends and the
    forced steps, respects the skip cap and stops when its runs are spent."""
    target = (0, 2, 5, 9)
    runs = []

    def evaluate(plan):
        runs.append(plan)
        return float(sum(abs(a - b) for a, b in zip(plan, target, strict=True)))

    plan, evals = refine_plan(evaluate, (0, 4, 7, 9), 10)
    assert plan == (target, 0.0) and evals == len(runs) and len(set(runs)) == len(runs)
    assert refine_plan(evaluate, (0, 4, 7, 9), 10, max_evals=3)[1] == 3
    kept, _ = refine_plan(evaluate, (0, 4, 7, 9), 10, forced=(4,))
    assert 4 in kept.computed and kept.computed[2] == 5
    capped, _ = refine_plan(evaluate, (0, 4, 7, 9), 10, max_skip=3)
    assert all(b - a - 1 <= 3 for a, b in zip(capped.computed, capped.computed[1:], strict=False))


def test_the_search_refines_a_named_method_on_measured_latents(tmp_path, backend):
    from benchmark.step_cache import search

    rule = [{"name": "every2", "label": "interval 2", "step_cache": {"method": "interval", "every": 2}}]
    (tmp_path / "rule.json").write_text(json.dumps(rule))
    search.main([
        "--model", "tiny", "--out", str(tmp_path), "--size", "64x64", "--frames", "5", "--steps", "8",
        "--prompts", "2", "--budgets", "5", "--depth", "3", "--compare", str(tmp_path / "rule.json"),
        "--refine", "every2", "--refine-evals", "6", "--refine-prompts", "1",
    ])
    refined = {m["name"]: m for m in json.loads((tmp_path / "searched.json").read_text())}["ref_every2"]
    start = backend.policy_of(rule[0]["step_cache"]).plan(8)
    assert len(refined["step_cache"]["computed"]) == sum(not action.reuse for action in start)
    assert refined["step_cache"]["computed"][0] == 0 and refined["step_cache"]["computed"][-1] == 7
