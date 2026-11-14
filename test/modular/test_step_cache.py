"""Opt-in step caching in the DiT scaffold (``components/diffusion/step_cache.py``): which steps skip (MagCache, an
interval, a listed plan), what stands in for their blocks (reuse or a forecast), the step key that carries the
decision, and Wan2.2's block stack routed through it. CPU only, tiny random weights."""

from __future__ import annotations

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.diffusion.denoise_loop import DenoiseLoopSubmodule, StepBucket
from mstar.model.components.diffusion.step_cache import (
    RESIDUAL,
    BlockCache,
    CacheRow,
    ExplicitPlan,
    FittedWeights,
    Forecast,
    Interval,
    MagCache,
    MagCacheCalibration,
    step_cache_policy,
)
from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine, PerRequestState
from mstar.model.wan22.components.dit import Wan22DiT
from mstar.model.wan22.config import MAG_RATIOS, WAN22_VARIANT_T2V_A14B, WAN22_VARIANT_TI2V_5B, Wan22Config
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, Wan22DitSubmodule
from mstar.model.wan22.wan22_model import Wan22Model

# --------------------------------------------------------------------- the schedule


def test_flat_ratios_reuse_up_to_the_skip_cap_between_computed_steps():
    """No accumulated error: past the retained 20%, every third step computes (cap 2), and so does the last."""
    plan = MagCache(mag_ratios=(1.0,) * 10, threshold=0.1, max_skip_steps=2, retention_ratio=0.2).reuse_plan(10)
    assert plan == (False, False, True, True, False, True, True, False, True, False)


def test_accumulated_error_past_the_threshold_computes_and_resets():
    # after the retained steps, 0.9 then 0.81 accumulate |1 - 0.9| = 0.1, then 0.1 + 0.19 = 0.29 > 0.15
    ratios = (1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0)
    plan = MagCache(mag_ratios=ratios, threshold=0.15, max_skip_steps=5, retention_ratio=0.25).reuse_plan(8)
    assert plan == (False, False, True, False, True, False, True, False)


@pytest.mark.parametrize("steps", [1, 2, 7, 50])
def test_step_zero_and_the_last_step_always_compute(steps):
    plan = MagCache(mag_ratios=(1.0, 1.0), threshold=10.0, max_skip_steps=100, retention_ratio=0.0).reuse_plan(steps)
    assert len(plan) == steps and not plan[0] and not plan[-1]
    assert all(plan[1:-1])


def test_a_zero_threshold_never_reuses_a_step_that_changes_the_residual():
    assert not any(MagCache(mag_ratios=(0.99,) * 20, threshold=0.0).reuse_plan(20))


def test_a_table_is_resampled_by_nearest_index_to_the_request_step_count():
    """A table calibrated at 4 steps serves 10: step k reads entry round(k / 3)."""
    table = (1.0, 1.0, 0.5, 1.0)  # only entry 2 carries error
    plan = MagCache(mag_ratios=table, threshold=0.4, max_skip_steps=9, retention_ratio=0.0).reuse_plan(10)
    # steps 5-7 read entry 2 (0.5 -> error 0.5 > 0.4: compute); every other inner step reuses
    assert plan == (False, True, True, True, True, False, False, False, True, False)


def test_the_calibration_policy_computes_every_step():
    assert MagCacheCalibration().reuse_plan(5) == (False,) * 5 and MagCacheCalibration.records


@pytest.mark.parametrize("spec,expected", [
    (None, None),
    ({"method": "none"}, None),
    ({"method": "magcache_calibrate"}, MagCacheCalibration()),
    ({"method": "magcache", "threshold": 0.1}, MagCache(mag_ratios=(1.0, 0.9), threshold=0.1)),
    ({"mag_ratios": "other", "max_skip_steps": 3}, MagCache(mag_ratios=(1.0, 0.8), max_skip_steps=3)),
    ({"mag_ratios": [1.0, 0.7, 0.6]}, MagCache(mag_ratios=(1.0, 0.7, 0.6))),
    ({"method": "interval", "every": 3, "forecast": 1}, Interval(every=3, forecast=Forecast(order=1))),
    ({"method": "plan", "computed": [3, 0, 1], "num_steps": 6}, ExplicitPlan(computed=(0, 1, 3), num_steps=6)),
    ({"method": "plan", "computed": [0, 2, 5], "num_steps": 6,
      "forecast": {"order": 1, "fitted": {"num_steps": 6, "weights": [[4, [2, 0], [1.2, -0.1]], [3, [2], [0.9]]]}}},
     ExplicitPlan(computed=(0, 2, 5), num_steps=6, forecast=Forecast(
         order=1, fitted=FittedWeights(6, ((3, (2,), (0.9,)), (4, (2, 0), (1.2, -0.1)))),
     ))),
    ({"mag_ratios": [1.0, 0.9], "forecast": {"order": 2, "damping": 0.5, "time": "sigma"}},
     MagCache(mag_ratios=(1.0, 0.9), forecast=Forecast(order=2, damping=0.5, time="sigma"))),
])
def test_a_serving_config_names_its_policy(spec, expected):
    assert step_cache_policy(spec, {"default": (1.0, 0.9), "other": (1.0, 0.8)}) == expected


@pytest.mark.parametrize("spec,fragment", [
    ({"method": "teacache"}, "unknown method"),
    ({"method": "magcache", "thresh": 0.1}, "unknown magcache options"),
    ({"method": "magcache"}, "no calibrated"),
    ({"method": "magcache_calibrate", "threshold": 0.1}, "takes no options"),
    ({"mag_ratios": [1.0, 0.9], "threshold": -1}, "threshold"),
    ({"method": "interval", "every": 0}, "every"),
    ({"method": "interval", "threshold": 0.1}, "unknown interval options"),
    ({"method": "plan", "computed": [0, 2]}, "num_steps"),
    ({"method": "plan", "computed": [0, 9], "num_steps": 6}, "computed steps"),
    ({"method": "interval", "forecast": 3}, "order"),
    ({"method": "interval", "forecast": {"order": 1, "kind": "hermite"}}, "unknown forecast options"),
    ({"method": "interval", "forecast": {"order": 1, "fitted": {"weights": []}}}, "num_steps"),
    ({"method": "interval",
      "forecast": {"order": 1, "fitted": {"num_steps": 6, "weights": [[4, [2, 1, 0], [1, 0, 0]]]}}}, "fitted entry"),
    ({"method": "interval", "forecast": {"order": 1, "fitted": {"num_steps": 6, "weights": [[2, [3], [1.0]]]}}},
     "fitted entry"),
])
def test_a_bad_serving_config_is_refused(spec, fragment):
    with pytest.raises(ValueError, match=fragment):
        step_cache_policy(spec, {})


def test_a_step_cache_on_a_captured_bucket_is_refused():
    with pytest.raises(ValueError, match="captured"):
        DenoiseLoopSubmodule(loop_name="l", capture_buckets=[("w", 1)], step_cache=MagCacheCalibration())


# --------------------------------------------------------------------- Wan2.2 on the scaffold

_TINY = dict(
    num_attention_heads=2, attention_head_dim=12, num_layers=2, ffn_dim=32,
    in_channels=4, out_channels=4, text_dim=8, freq_dim=16, rope_max_seq_len=16, compile_dit=False,
)
# 5 frames x 64x64 pixels -> a latent grid of (2, 4, 4), 8 tokens after patchify
_META = {"guidance_scale": 5.0, "height": 64, "width": 64, "num_frames": 5}
_STEPS = 6
_FLAT = {"method": "magcache", "mag_ratios": [1.0, 1.0], "threshold": 1.0, "max_skip_steps": 1,
         "retention_ratio": 0.0}
# _FLAT at 6 steps: 0 and 5 compute, 1-4 alternate
_FLAT_PLAN = (False, True, False, True, False, False)


def _tiny_dit() -> Wan22DiT:
    torch.manual_seed(0)
    dit = Wan22DiT(Wan22Config(**_TINY))
    for param in dit.parameters():
        torch.nn.init.normal_(param, std=0.2)
    return dit.eval()


def _info(k: int, rid: str = "r0", steps: int = _STEPS, **meta) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=Wan22Model.VIDEO_GEN_WALK, fwd_index=k, random_seed=7, max_tokens=0,
        step_metadata={"num_inference_steps": steps, **_META, **meta},
        dynamic_loop_iter_counts={DENOISE_LOOP_NAME: k},
    )


def _submodule(step_cache=None, transformer=None, cfg_group=None) -> Wan22DitSubmodule:
    sub = Wan22DitSubmodule(
        transformer=transformer if transformer is not None else _tiny_dit(),
        config=Wan22Config(**_TINY, step_cache=step_cache), cfg_group=cfg_group,
    )
    sub.get_device = lambda: torch.device("cpu")
    return sub


def _text(rid: str = "r0") -> dict:
    seed = sum(map(ord, rid))
    gen = torch.Generator().manual_seed(seed)
    return {name: [torch.randn(1, 5, 8, generator=gen)] for name in ("text_embeds_pos", "text_embeds_neg")}


def _run_loop(sub: Wan22DitSubmodule, rids=("r0",)) -> list[dict[str, torch.Tensor]]:
    """The request's whole denoise loop through the engine contract (prepare_inputs, preprocess, forward), the
    loop-back edges routed by hand; returns each step's latents per request."""
    carried = {rid: {} for rid in rids}
    steps = []
    for k in range(_STEPS):
        infos = {rid: _info(k, rid) for rid in rids}
        inputs = [sub.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, infos[rid], {**_text(rid), **carried[rid]})
                  for rid in rids]
        assert len(rids) == 1 or sub.can_batch(None, inputs)
        engine_inputs = ModelInputsFromEngine(request_ids=list(rids), per_request_info=infos)
        kwargs = sub.preprocess(Wan22Model.VIDEO_GEN_WALK, engine_inputs, inputs)
        if len(rids) == 1:
            outs = {rids[0]: sub.forward(Wan22Model.VIDEO_GEN_WALK, engine_inputs, **kwargs)}
        else:
            outs = sub.forward_batched(Wan22Model.VIDEO_GEN_WALK, engine_inputs, **kwargs)
        for rid in rids:
            carried[rid] = {name: outs[rid][name] for name in sub.loop_back_names}
        steps.append({rid: outs[rid][sub.LATENTS][0] for rid in rids})
    return steps


class _BlockCounter:
    """Counts the block stack's runs (the first block's forward calls)."""

    def __init__(self, dit: Wan22DiT):
        self.calls = 0
        dit.blocks[0].register_forward_hook(lambda *_: setattr(self, "calls", self.calls + 1))


def test_without_a_step_cache_the_step_key_is_the_bucket_key():
    sub = _submodule()
    info = _info(1)
    inputs = sub.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, info, {**_text(), "latents": [torch.zeros(1)],
                                                                   "unipc_model_outputs": [torch.zeros(1)],
                                                                   "unipc_last_sample": [torch.zeros(1)]})
    assert inputs.resource_step_info == sub.bucket_key_for(info) == sub.cg_key_info("w", {"r0": info})
    assert not sub.split_batches_by_capture_key("w") and sub.step_cache is None


def test_the_step_key_carries_the_plan_from_prepare_inputs_and_cg_key_info_alike():
    sub = _submodule(_FLAT)
    key = sub.bucket_key_for(_info(0))
    assert sub.split_batches_by_capture_key("w")
    for k, reuse in enumerate(_FLAT_PLAN):
        expected = StepBucket(key, None, reuse)
        assert sub.step_key(_info(k), key) == expected
        assert sub.cg_key_info("w", {"r0": _info(k)}) == expected
        # a speculated step's rows still carry the step in flight's info; its input metadata has the step it runs
        ahead = {"r0": InputMetadata(dynamic_loop_iter_counts={DENOISE_LOOP_NAME: k})}
        assert sub.cg_key_info("w", {"r0": _info(max(k - 1, 0))}, per_request_input_metadata=ahead) == expected


def test_rows_at_different_decisions_do_not_batch():
    sub = _submodule(_FLAT)
    tail = {"latents": [torch.zeros(1)], "unipc_model_outputs": [torch.zeros(1)], "unipc_last_sample": [torch.zeros(1)]}
    computing = sub.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, _info(2, "a"), {**_text("a"), **tail})
    reusing = sub.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, _info(1, "b"), {**_text("b"), **tail})
    also_reusing = sub.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, _info(3, "c"), {**_text("c"), **tail})
    assert not sub.can_batch(None, [computing, reusing])
    assert sub.can_batch(None, [reusing, also_reusing])


def test_a_policy_that_computes_every_step_is_bit_identical_to_no_cache(caplog):
    """The compute path with the cache on (residual kept, nothing reused) changes no output, and calibration logs a
    table of one ratio per step."""
    dense = _run_loop(_submodule())
    with caplog.at_level("INFO", logger="mstar.model.components.diffusion.step_cache"):
        calibrated = _run_loop(_submodule({"method": "magcache_calibrate"}))
    for a, b in zip(dense, calibrated, strict=True):
        assert torch.equal(a["r0"], b["r0"])
    line = next(r.getMessage() for r in caplog.records if "mag_ratios" in r.getMessage())
    table = [float(v) for v in line.split("[")[1].rstrip("]").split(",")]
    assert len(table) == _STEPS and table[0] == 1.0 and all(r > 0 for r in table)


def test_reusing_steps_skip_the_blocks_and_add_the_cached_residual():
    dit = _tiny_dit()
    counter = _BlockCounter(dit)
    sub = _submodule(_FLAT, transformer=dit)
    cached = _run_loop(sub)
    assert counter.calls == _FLAT_PLAN.count(False)
    dense = _run_loop(_submodule())
    # computing steps before any reuse match exactly; reuse then moves the trajectory, but not far
    assert torch.equal(cached[0]["r0"], dense[0]["r0"])
    assert not torch.equal(cached[1]["r0"], dense[1]["r0"])
    assert torch.allclose(cached[-1]["r0"], dense[-1]["r0"], atol=0.5)
    # the residual is per-request state the engine owns
    assert sub.request_state("r0").tensors[RESIDUAL].shape == (2, 8, 24)  # [branches, tokens, dim]


def test_a_batched_step_keeps_and_reuses_each_rows_own_residual():
    """Two requests in one forward each get back what they get alone."""
    batched = _run_loop(_submodule(_FLAT), rids=("a", "b"))
    for rid in ("a", "b"):
        alone = _run_loop(_submodule(_FLAT), rids=(rid,))
        for x, y in zip(batched, alone, strict=True):
            torch.testing.assert_close(x[rid], y[rid], rtol=1e-5, atol=1e-5)


def test_the_dit_returns_the_residual_its_blocks_added_and_reuses_it():
    dit = _tiny_dit()
    gen = torch.Generator().manual_seed(3)
    x, t, text = torch.randn(2, 4, 2, 8, 8, generator=gen), torch.rand(2, 2, generator=gen) * 999, torch.randn(
        2, 5, 8, generator=gen)
    with torch.no_grad():
        plain = dit(x, t, text)
        out, residual = dit(x, t, text, return_residual=True)
        reused = dit(x, t, text, cached_residual=residual)
    assert torch.equal(out, plain) and residual.dtype == torch.float32 and residual.shape == (2, 32, 24)
    # the same inputs with their own residual give the same output, up to the fp32 round trip of the add
    torch.testing.assert_close(reused, plain)


class _FakeCfgGroup:
    def __init__(self, rank: int):
        self.world_size, self.rank = 2, rank

    def all_gather(self, x, dim):
        return torch.cat([x, x], dim=dim)


def test_cfg_ranks_take_the_same_decisions_and_keep_their_own_branch():
    """Each CFG rank derives every step's decision from the config and the loop counter alone, so the two agree; each
    keeps the residual of the one branch it forwards."""
    ranks = [_submodule(_FLAT, cfg_group=_FakeCfgGroup(r)) for r in (0, 1)]
    key = ranks[0].bucket_key_for(_info(0))
    for k in range(_STEPS):
        assert ranks[0].step_key(_info(k), key) == ranks[1].step_key(_info(k), key)
    for sub in ranks:
        _run_loop(sub)
        assert sub.request_state("r0").tensors[RESIDUAL].shape == (1, 8, 24)
    assert not torch.equal(*(sub.request_state("r0").tensors[RESIDUAL] for sub in ranks))


# --------------------------------------------------------------------- A14B: two experts over the schedule


def test_an_expert_switch_computes_and_starts_the_skip_rule_over():
    """At a segment start (another expert's first step) the step computes, and the skip cap and the accumulated
    error restart there; the table entry at that step, which compares two experts' residuals, is never read."""
    policy = MagCache(mag_ratios=(1.0,) * 8, threshold=1.0, max_skip_steps=9, retention_ratio=0.0)
    assert policy.reuse_plan(8) == (False, True, True, True, True, True, True, False)
    assert policy.reuse_plan(8, (4,)) == (False, True, True, True, False, True, True, False)
    # an error spike in the switch step's entry changes nothing
    spiked = MagCache(mag_ratios=(1.0,) * 4 + (0.1,) + (1.0,) * 3, threshold=1.0, max_skip_steps=9,
                      retention_ratio=0.0)
    assert spiked.reuse_plan(8, (4,)) == policy.reuse_plan(8, (4,))


_A14B_STEPS = 8


def _a14b(step_cache=None):
    config = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B, **_TINY, step_cache=step_cache)
    high, low = _tiny_dit(), _tiny_dit()
    for param in low.parameters():
        torch.nn.init.normal_(param, std=0.2)
    sub = Wan22DitSubmodule(transformer=high, config=config, transformer_2=low)
    sub.get_device = lambda: torch.device("cpu")
    return sub, _BlockCounter(high), _BlockCounter(low)


def _a14b_info(k: int, rid: str = "r0") -> CurrentForwardPassInfo:
    return _info(k, rid, steps=_A14B_STEPS, guidance_scale_2=3.0)


def _a14b_step(sub: Wan22DitSubmodule, steps: dict[str, int], carried: dict[str, dict]) -> dict[str, dict]:
    """One dit step over the requests of ``steps``, each at its own step index; every request's outputs."""
    rids = list(steps)
    infos = {rid: _a14b_info(steps[rid], rid) for rid in rids}
    inputs = [sub.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, infos[rid], {**_text(rid), **carried.get(rid, {})})
              for rid in rids]
    assert len(rids) == 1 or sub.can_batch(None, inputs)
    engine_inputs = ModelInputsFromEngine(request_ids=rids, per_request_info=infos)
    kwargs = sub.preprocess(Wan22Model.VIDEO_GEN_WALK, engine_inputs, inputs)
    if len(rids) == 1:
        return {rids[0]: sub.forward(Wan22Model.VIDEO_GEN_WALK, engine_inputs, **kwargs)}
    return sub.forward_batched(Wan22Model.VIDEO_GEN_WALK, engine_inputs, **kwargs)


def test_a14b_segments_at_the_low_noise_experts_first_step():
    sub, _, _ = _a14b(_FLAT)
    key = sub.bucket_key_for(_a14b_info(0))
    timesteps = sub.schedule_of(_a14b_info(0), key).timesteps.tolist()
    first_low = next(k for k, t in enumerate(timesteps) if t < 875)
    assert 1 < first_low < _A14B_STEPS - 1
    assert sub.step_cache_segments(_a14b_info(0), key) == (first_low,)
    assert sub.step_key(_a14b_info(first_low), key) == StepBucket(key, None, False)
    # TI2V-5B has one DiT, so no segments
    assert _submodule(_FLAT).step_cache_segments(_info(0), key) == ()


def test_a14b_never_reuses_one_experts_residual_in_the_other():
    """Each expert's blocks run on its first step; the low-noise expert never adds the high-noise one's residual."""
    sub, high, low = _a14b(_FLAT)
    key = sub.bucket_key_for(_a14b_info(0))
    first_low = sub.step_cache_segments(_a14b_info(0), key)[0]
    plan = sub.step_cache.reuse_plan(_A14B_STEPS, (first_low,))
    carried = {}
    for k in range(_A14B_STEPS):
        out = _a14b_step(sub, {"r0": k}, {"r0": carried})["r0"]
        carried = {name: out[name] for name in sub.loop_back_names}
    assert not plan[first_low] and any(plan[:first_low]) and any(plan[first_low:])
    assert high.calls == plan[:first_low].count(False)
    assert low.calls == plan[first_low:].count(False)


def test_a_batch_straddling_the_boundary_keeps_each_rows_residual_from_its_own_expert():
    """Rows on both sides of the boundary at one decision share a step: each expert's forward keeps the residuals
    of its own rows (``BlockCache.rows``), as each row keeps alone."""
    def at_the_boundary(batched: bool) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        sub, high, low = _a14b(_FLAT)
        first_low = sub.step_cache_segments(_a14b_info(0), sub.bucket_key_for(_a14b_info(0)))[0]
        carried = {}
        for k in range(first_low):  # request b, up to the low-noise expert's first step
            out = _a14b_step(sub, {"b": k}, {"b": carried})["b"]
            carried = {name: out[name] for name in sub.loop_back_names}
        before = (high.calls, low.calls)
        # a's step 0 runs the high-noise expert and b's step the low-noise one; both compute
        rows = ({"a": 0, "b": first_low},) if batched else ({"a": 0}, {"b": first_low})
        outs = {}
        for steps in rows:
            outs.update(_a14b_step(sub, steps, {"b": carried}))
        assert (high.calls - before[0], low.calls - before[1]) == (1, 1)
        return {rid: (outs[rid][sub.LATENTS][0], sub.request_state(rid).tensors[RESIDUAL]) for rid in ("a", "b")}

    together, alone = at_the_boundary(True), at_the_boundary(False)
    for rid in ("a", "b"):
        for x, y in zip(together[rid], alone[rid], strict=True):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
    assert not torch.equal(together["a"][1], together["b"][1])


def test_a_magcache_without_a_table_takes_its_variants_own():
    """A14B's experts are calibrated on their own table, not TI2V-5B's (different step count, an expert switch)."""
    spec = {"method": "magcache", "threshold": 0.12, "max_skip_steps": 4, "retention_ratio": 0.4}
    five, (a14b, _, _) = _submodule(spec), _a14b(spec)
    assert five.step_cache.mag_ratios == MAG_RATIOS[WAN22_VARIANT_TI2V_5B]["default"]
    assert a14b.step_cache.mag_ratios == MAG_RATIOS[WAN22_VARIANT_T2V_A14B]["default"]
    assert len(five.step_cache.mag_ratios) == 50 and len(a14b.step_cache.mag_ratios) == 40
    # the setting measured on B300: the first 16 of 40 steps computed, 24 in all
    plan = a14b.step_cache.reuse_plan(40, (26,))
    assert not any(plan[:16]) and plan.count(False) == 24


def test_a_step_cache_with_a_distilled_sampler_is_refused():
    """The calibrated table is a UniPC run's: resampled to FastWan's 3 steps it would skip the middle one."""
    table = MAG_RATIOS[WAN22_VARIANT_TI2V_5B]
    assert step_cache_policy({"method": "magcache"}, table).reuse_plan(3) == (False, True, False)
    with pytest.raises(ValueError, match="unipc sampler only"):
        Wan22Config(sampler="dmd", step_cache=_FLAT)


# --------------------------------------------------------------------- which steps skip: an interval, a listed plan


def test_an_interval_computes_one_step_in_every_past_the_retained_ones():
    assert Interval(every=3, retention_ratio=0.2).reuse_plan(10) == (
        False, False, True, True, False, True, True, False, True, False,
    )
    # a segment start computes and the count restarts there; every 1 computes all
    assert Interval(every=3, retention_ratio=0.0).reuse_plan(8, (4,)) == (
        False, True, True, False, False, True, True, False,
    )
    assert not any(Interval(every=1).reuse_plan(12))


def test_a_listed_plan_computes_its_steps_and_always_the_ends_and_segment_starts():
    plan = ExplicitPlan(computed=(0, 1, 4, 5), num_steps=8)
    assert plan.reuse_plan(8) == (False, False, True, True, False, False, True, False)
    assert plan.reuse_plan(8, (3,)) == (False, False, True, False, False, False, True, False)
    # another step count takes the nearest listed step's decision
    assert plan.reuse_plan(4) == (False, True, False, False)
    assert ExplicitPlan(computed=(), num_steps=5).reuse_plan(5) == (False, True, True, True, False)


# --------------------------------------------------------------------- what stands in for the blocks: a forecast


def test_a_forecast_of_order_one_or_two_is_exact_on_a_line_at_any_spacing():
    line = lambda t: 3.0 - 2.0 * t  # noqa: E731
    for kept, now in [((5.0, 3.0), 6.0), ((5.0, 3.0), 8.0), ((0.4, 0.7, 0.9), 0.1), ((7.0, 6.0, 2.0), 9.0)]:
        for order in (1, 2):
            weights = Forecast(order=order).weights(kept, now)
            assert len(weights) == min(order, len(kept) - 1) + 1
            assert sum(w * line(t) for w, t in zip(weights, kept, strict=False)) == pytest.approx(line(now))


def test_a_forecast_at_equal_spacing_is_taylorseers():
    """Kept every N steps, d steps past the newest: F0 + (F0 - F1) d/N + (F0 - 2 F1 + F2) d^2 / (2 N^2)."""
    n, d = 3.0, 2.0
    assert Forecast(order=1).weights((6.0, 3.0), 8.0) == pytest.approx((1 + d / n, -d / n))
    assert Forecast(order=2).weights((6.0, 3.0, 0.0), 8.0) == pytest.approx(
        (1 + d / n + d * d / (2 * n * n), -d / n - d * d / (n * n), d * d / (2 * n * n))
    )


def test_a_forecast_uses_the_order_it_has_values_for_and_damping_scales_the_step():
    assert Forecast(order=2).weights((4.0,), 6.0) == (1.0,)
    assert Forecast(order=2).weights((4.0, 2.0), 6.0) == Forecast(order=1).weights((4.0, 2.0), 6.0)
    assert Forecast(order=0).weights((4.0, 2.0), 6.0) == (1.0,)
    assert Forecast(order=1, damping=0.0).weights((4.0, 2.0), 6.0) == pytest.approx((1.0, 0.0))
    assert Forecast(order=1, damping=0.5).weights((4.0, 2.0), 6.0) == pytest.approx((1.5, -0.5))


def test_a_block_cache_predicts_each_row_from_its_own_kept_steps():
    """Two rows at different steps of different histories share a reusing step; each gets its own extrapolation."""
    forecast = Forecast(order=1)
    a, b = PerRequestState(), PerRequestState()
    for step, value in [(0, 1.0), (2, 2.0)]:
        BlockCache(False, [CacheRow(a, step)], forecast).keep(torch.full((1, 3), value))
    for step, value in [(4, 10.0), (5, 20.0)]:
        BlockCache(False, [CacheRow(b, step)], forecast).keep(torch.full((1, 3), value))
    cache = BlockCache(True, [CacheRow(a, 3), CacheRow(b, 7)], forecast)
    assert cache.residual.shape == (2, 3)
    # a: 2 + (2 - 1) / 2 * 1; b: 20 + (20 - 10) / 1 * 2
    assert torch.equal(cache.residual, torch.tensor([[2.5] * 3, [40.0] * 3]))
    # order 0 hands back the kept tensor itself, and predicting changes nothing kept
    assert torch.equal(BlockCache(True, [CacheRow(a, 3)]).residual[0], torch.full((3,), 2.0))
    assert torch.equal(a.tensors[RESIDUAL], torch.full((3,), 2.0)) and a.kwargs[RESIDUAL + "_steps"] == [2, 0]
    # a subset of the rows (another expert's) reads only its own
    assert torch.equal(cache.rows([1]).residual, torch.full((1, 3), 40.0))


def test_a_forecast_spaced_by_sigma_reads_the_rows_schedule():
    forecast = Forecast(order=1, time="sigma")
    state, sigmas = PerRequestState(), (1.0, 0.9, 0.5, 0.4)
    for step, value in [(0, 0.0), (1, 1.0)]:
        BlockCache(False, [CacheRow(state, step, sigmas)], forecast).keep(torch.full((1, 2), value))
    predicted = BlockCache(True, [CacheRow(state, 2, sigmas)], forecast).residual
    # 1 + (1 - 0) / (0.9 - 1.0) * (0.5 - 0.9) = 5; by step index it would be 2
    torch.testing.assert_close(predicted, torch.full((1, 2), 5.0))


def test_a_fitted_forecast_takes_its_tables_weights_where_the_table_has_the_step():
    """The table is for one step count and names the kept steps each entry was fitted for; anywhere else the
    forecast's own expansion applies."""
    fitted = FittedWeights(8, ((3, (2, 0), (1.5, -0.25)), (5, (4,), (0.5,))))
    forecast = Forecast(order=1, fitted=fitted)
    assert forecast.fitted_weights(8, 3, (2, 0)) == (1.5, -0.25) and forecast.fitted_weights(8, 5, (4,)) == (0.5,)
    assert forecast.fitted_weights(6, 3, (2, 0)) is None and forecast.fitted_weights(8, 3, (2, 1)) is None
    assert Forecast(order=1).fitted_weights(8, 3, (2, 0)) is None
    assert forecast == Forecast(order=1, fitted=fitted) and hash(forecast) == hash(Forecast(order=1, fitted=fitted))
    assert forecast != Forecast(order=1)

    def predicted(step: int, num_steps: int, kept: list[tuple[int, float]]) -> torch.Tensor:
        state = PerRequestState()
        for at, value in kept:
            BlockCache(False, [CacheRow(state, at, num_steps=num_steps)], forecast).keep(torch.full((1, 2), value))
        out = BlockCache(True, [CacheRow(state, step, num_steps=num_steps)], forecast).residual
        # predicting leaves what was kept as it was
        assert torch.equal(state.tensors[RESIDUAL], torch.full((2,), kept[-1][1]))
        return out

    # the table's weights: 1.5 * 4 - 0.25 * 8
    assert torch.equal(predicted(3, 8, [(0, 8.0), (2, 4.0)]), torch.full((1, 2), 4.0))
    # another step count, or other kept steps: the line through the two kept values, 4 + (4 - 8) / 2
    assert torch.equal(predicted(3, 6, [(0, 8.0), (2, 4.0)]), torch.full((1, 2), 2.0))
    assert torch.equal(predicted(3, 8, [(1, 8.0), (2, 4.0)]), torch.full((1, 2), 0.0))
    # one kept value and a fitted gain on it
    assert torch.equal(predicted(5, 8, [(4, 6.0)]), torch.full((1, 2), 3.0))


def test_nothing_kept_before_a_segment_start_is_read_or_left_allocated():
    forecast = Forecast(order=2)
    state = PerRequestState()
    for step in (0, 1, 2):
        BlockCache(False, [CacheRow(state, step)], forecast).keep(torch.full((1, 2), float(step)))
    assert sorted(state.tensors) == [RESIDUAL, RESIDUAL + "_1", RESIDUAL + "_2"]
    # step 5 starts a segment (another expert's first step): its residual is the only one a later step reads
    BlockCache(False, [CacheRow(state, 5, floor=5)], forecast).keep(torch.full((1, 2), 9.0))
    assert sorted(state.tensors) == [RESIDUAL] and state.kwargs[RESIDUAL + "_steps"] == [5]
    assert torch.equal(BlockCache(True, [CacheRow(state, 6, floor=5)], forecast).residual, torch.full((1, 2), 9.0))
    with pytest.raises(RuntimeError, match="must compute"):
        _ = BlockCache(True, [CacheRow(state, 9, floor=8)], forecast).residual


# --------------------------------------------------------------------- a forecast through Wan2.2


@pytest.mark.parametrize("spec", [
    {"method": "interval", "every": 1, "forecast": 2},
    {"method": "plan", "computed": list(range(_STEPS)), "num_steps": _STEPS,
     "forecast": {"order": 2, "time": "sigma", "fitted": {"num_steps": _STEPS, "weights": [[3, [2, 1], [1.5, -0.5]]]}}},
])
def test_a_policy_with_nothing_to_skip_is_bit_identical_to_no_cache(spec):
    """A forecast is idle until a step skips: the compute path keeps what later steps would read and changes no
    output."""
    sub = _submodule(spec)
    assert not any(sub.step_cache.reuse_plan(_STEPS))
    for a, b in zip(_run_loop(_submodule()), _run_loop(sub), strict=True):
        assert torch.equal(a["r0"], b["r0"])


@pytest.mark.parametrize("spec", [
    {"method": "interval", "every": 3, "retention_ratio": 0.0, "forecast": 2},
    # step 2 takes the fitted weights, step 4 the sigma-spaced line
    {"method": "plan", "computed": [0, 1, 3, 5], "num_steps": _STEPS,
     "forecast": {"order": 1, "time": "sigma", "fitted": {"num_steps": _STEPS, "weights": [[2, [1, 0], [1.5, -0.4]]]}}},
])
def test_rows_of_a_batch_are_independent_under_a_forecast(spec):
    """Two requests in one forward each get back what they get alone: a forecast reads the row's own kept residuals
    and schedule."""
    batched = _run_loop(_submodule(spec), rids=("a", "b"))
    for rid in ("a", "b"):
        alone = _run_loop(_submodule(spec), rids=(rid,))
        for x, y in zip(batched, alone, strict=True):
            torch.testing.assert_close(x[rid], y[rid], rtol=1e-4, atol=1e-4)


def test_a_forecast_changes_what_a_skipping_step_adds_once_two_steps_are_kept():
    every3 = {"method": "interval", "every": 3, "retention_ratio": 0.0}
    reuse, forecast = _run_loop(_submodule(every3)), _run_loop(_submodule({**every3, "forecast": 1}))
    # steps 1 and 2 skip on one kept residual, so both add it; step 3 computes; step 4 extrapolates steps 0 and 3
    for k in range(4):
        assert torch.equal(reuse[k]["r0"], forecast[k]["r0"])
    assert not torch.equal(reuse[4]["r0"], forecast[4]["r0"])
    # weights fitted for step 4 from steps 3 and 0 take the line's place there: all on step 3 is a reuse again
    fitted = {"order": 1, "fitted": {"num_steps": _STEPS, "weights": [[4, [3, 0], [1.0, 0.0]]]}}
    for a, b in zip(reuse, _run_loop(_submodule({**every3, "forecast": fitted})), strict=True):
        assert torch.equal(a["r0"], b["r0"])


def test_a14b_forecasts_nothing_from_before_the_low_noise_experts_first_step():
    """The low-noise expert's first step computes, and its skipping steps read nothing the high-noise expert kept."""
    sub, high, low = _a14b({"method": "interval", "every": 2, "retention_ratio": 0.0, "forecast": 2})
    key = sub.bucket_key_for(_a14b_info(0))
    first_low = sub.step_cache_segments(_a14b_info(0), key)[0]
    plan = sub.step_cache.reuse_plan(_A14B_STEPS, (first_low,))
    assert not plan[first_low] and any(plan[first_low:])
    carried, state = {}, sub.request_state("r0")
    for k in range(_A14B_STEPS):
        out = _a14b_step(sub, {"r0": k}, {"r0": carried})["r0"]
        carried = {name: out[name] for name in sub.loop_back_names}
        if k >= first_low:
            assert min(state.kwargs[RESIDUAL + "_steps"]) >= first_low
        if k == first_low:
            assert sorted(state.tensors) == [RESIDUAL]
    assert high.calls == plan[:first_low].count(False)
    assert low.calls == plan[first_low:].count(False)
