"""Opt-in step caching in the DiT scaffold (``components/diffusion/step_cache.py``): MagCache's skip schedule, the
step key that carries the decision, and Wan2.2's block stack routed through it. CPU only, tiny random weights."""

from __future__ import annotations

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.diffusion.denoise_loop import DenoiseLoopSubmodule, StepBucket
from mstar.model.components.diffusion.step_cache import (
    RESIDUAL,
    MagCache,
    MagCacheCalibration,
    step_cache_policy,
)
from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine
from mstar.model.wan22.components.dit import Wan22DiT
from mstar.model.wan22.config import Wan22Config
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
])
def test_a_serving_config_names_its_policy(spec, expected):
    assert step_cache_policy(spec, {"default": (1.0, 0.9), "other": (1.0, 0.8)}) == expected


@pytest.mark.parametrize("spec,fragment", [
    ({"method": "teacache"}, "unknown method"),
    ({"method": "magcache", "thresh": 0.1}, "unknown magcache options"),
    ({"method": "magcache"}, "no calibrated"),
    ({"method": "magcache_calibrate", "threshold": 0.1}, "takes no options"),
    ({"mag_ratios": [1.0, 0.9], "threshold": -1}, "threshold"),
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


def _tiny_dit(sp_group=None) -> Wan22DiT:
    torch.manual_seed(0)
    dit = Wan22DiT(Wan22Config(**_TINY), sp_group=sp_group)
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
