"""BAGEL image_gen runs only the CFG branches a step needs.

Guidance on, the old step always ran main, cfg_text and cfg_img as one
batched forward, though the combine never reads ``v_cfg_img`` when the image
scale is <= 1 (the text-to-image default), and outside the CFG interval it
reduces to ``v_main``. ``cfg_branches`` decides 1/2/3 on the host from the
loop counter; ``declare_step`` plans that many labels and the forward runs
that many. These pin the decision table, its agreement with the old
device-side interval test, the combine reducing to what the old one computed,
and the declaration matching the branch count. CPU only.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch

from mstar.engine.resources import KVStep, PositionStep
from mstar.model.bagel.config import BagelModelConfig
from mstar.model.bagel.submodules import (
    CFG_BRANCHES,
    IMAGE_GEN_LOOP,
    LLMSubmodule,
    active_labels,
    cfg_branches,
    combine_cfg_velocity,
    image_gen_timestep,
)
from mstar.model.submodule_base import ARNodeInputs

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(BagelModelConfig)}
N = _DEFAULTS["num_timesteps"]
SHIFT = _DEFAULTS["timestep_shift"]
T2I = dict(  # text-to-image request defaults
    cfg_interval=tuple(_DEFAULTS["cfg_interval"]),
    text_scale=_DEFAULTS["cfg_text_scale"],
    img_scale=_DEFAULTS["cfg_img_scale"],
)
EDIT = dict(cfg_interval=(0.0, 1.0), text_scale=4.0, img_scale=2.0)  # bagel_model edit overrides
STEPS = range(N - 1)  # time_index 0 .. N-2


def _device_timestep(k: int) -> torch.Tensor:
    """What `_forward_image_gen` computes from its time_index input."""
    time_index = torch.full((8,), float(k), dtype=torch.float32)
    t_uniform = 1.0 - time_index / (N - 1)
    return LLMSubmodule._apply_timestep_shift(t=t_uniform, shift=SHIFT)


def _old_in_interval(k: int, interval) -> bool:
    t_val = _device_timestep(k)[0]
    lo, hi = interval
    return bool((t_val > lo) & (t_val <= hi))


def _branches(k: int, cfg_interval, text_scale, img_scale) -> int:
    return cfg_branches(image_gen_timestep(k, N, SHIFT), cfg_interval, text_scale, img_scale)


# --- the decision ---------------------------------------------------------


def test_host_timestep_matches_the_device_formula():
    for k in STEPS:
        assert image_gen_timestep(k, N, SHIFT) == _device_timestep(k)[0].item()


def test_text_to_image_defaults_run_two_branches_in_interval_and_one_outside():
    counts = [_branches(k, **T2I) for k in STEPS]
    assert set(counts) == {1, 2}, "t2i (img scale 1.0) must never run cfg_img"
    # the schedule is decreasing in t, so the interval is a prefix of the steps
    n_in = counts.count(2)
    assert counts == [2] * n_in + [1] * (len(counts) - n_in)
    assert 0 < n_in < len(counts)


def test_image_edit_runs_all_three_branches_every_step():
    assert [_branches(k, **EDIT) for k in STEPS] == [3] * len(STEPS)


def test_guidance_off_and_text_off_cases():
    assert _branches(0, cfg_interval=(0.0, 1.0), text_scale=1.0, img_scale=1.0) == 1
    # image guidance alone still needs cfg_text (labels are a prefix)
    assert _branches(0, cfg_interval=(0.0, 1.0), text_scale=1.0, img_scale=1.5) == 3
    assert _branches(0, cfg_interval=(0.0, 1.0), text_scale=4.0, img_scale=1.0) == 2


@pytest.mark.parametrize("interval", [
    (0.4, 1.0), (0.0, 1.0), (0.1, 0.7), (0.3333333333, 0.6666666666),
    # bounds landing exactly on schedule values: (lo, hi] is open below
    "on_grid",
])
def test_interval_agrees_with_the_old_device_test(interval):
    if interval == "on_grid":
        interval = (image_gen_timestep(30, N, SHIFT), image_gen_timestep(5, N, SHIFT))
    for k in STEPS:
        in_old = _old_in_interval(k, interval)
        n = _branches(k, interval, 4.0, 2.0)
        assert (n > 1) == in_old, (k, interval)


# --- the combine ----------------------------------------------------------


def _old_combine(k, v_main, v_text, v_img, interval, text_scale, img_scale, renorm, rmin):
    """The always-three-branch step: scales gated by the interval on device."""
    t_val = _device_timestep(k)[0]
    lo, hi = interval
    in_cfg = ((t_val > lo) & (t_val <= hi)).float()
    eff_text = text_scale * in_cfg + 1.0 * (1 - in_cfg)
    eff_img = img_scale * in_cfg + 1.0 * (1 - in_cfg)
    return combine_cfg_velocity(v_main, v_text, v_img, eff_text, eff_img, renorm, rmin)


def _pruned_combine(k, v_main, v_text, v_img, interval, text_scale, img_scale, renorm, rmin):
    n = _branches(k, interval, text_scale, img_scale)
    if n == 1:
        return v_main
    return combine_cfg_velocity(
        v_main, v_text, v_img if n == 3 else None, text_scale, img_scale, renorm, rmin,
    )


@pytest.mark.parametrize("renorm", ["global", "channel", "text_channel"])
@pytest.mark.parametrize("rmin", [0.0, 0.5])
@pytest.mark.parametrize("request_kind", ["t2i", "edit", "img_only_in_window"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pruned_combine_matches_the_old_combine(renorm, rmin, request_kind, dtype):
    params = {
        "t2i": T2I,
        "edit": EDIT,
        "img_only_in_window": dict(cfg_interval=(0.4, 1.0), text_scale=4.0, img_scale=1.5),
    }[request_kind]
    gen = torch.Generator().manual_seed(0)
    v_main, v_text, v_img = (
        torch.randn(64, 16, generator=gen).to(dtype) for _ in range(3)
    )
    args = (params["cfg_interval"], params["text_scale"], params["img_scale"], renorm, rmin)
    for k in STEPS:
        old = _old_combine(k, v_main, v_text, v_img, *args)
        new = _pruned_combine(k, v_main, v_text, v_img, *args)
        assert new.dtype == old.dtype
        if _branches(k, *args[:3]) > 1:
            # in the interval: the same ops on the same branches
            assert torch.equal(new, old), (k, renorm)
        else:
            # outside: the old combine was v_main up to rounding
            tol = 1e-6 if dtype == torch.float32 else 1e-2
            torch.testing.assert_close(new, old, rtol=tol, atol=tol)


# --- the declaration and the forward's inputs -----------------------------


def _stub(**overrides):
    return types.SimpleNamespace(
        node_name="LLM",
        CFG_BATCHED_LABEL=LLMSubmodule.CFG_BATCHED_LABEL,
        _get_active_labels=lambda walk, cfg: active_labels(walk, cfg, "LLM"),
        **overrides,
    )


def _image_gen_row(n_branches: int | None, span: int = 6) -> ARNodeInputs:
    row = ARNodeInputs(input_seq_len=span)
    row.resource_step_info = True  # requires_cfg
    row.custom_pos_ids = {
        label: torch.full((span,), 10 + i, dtype=torch.int32)
        for i, label in enumerate(("main", "cfg_text", "cfg_img"))
    }
    if n_branches is not None:
        row.kwargs[CFG_BRANCHES] = n_branches
    return row


@pytest.mark.parametrize("n_branches,labels", [
    (None, ("main", "cfg_text", "cfg_img")),  # pruning off / unstamped
    (3, ("main", "cfg_text", "cfg_img")),
    (2, ("main", "cfg_text")),
    (1, ("main",)),
])
def test_declared_step_plans_exactly_the_branches_run(n_branches, labels):
    span = 6
    step = LLMSubmodule.declare_step(
        _stub(), "image_gen", ["r0"], [_image_gen_row(n_branches, span)],
    )
    kv: KVStep = step.steps["kv"]
    rope: PositionStep = step.steps["rope"]
    assert tuple(seg.label for seg in kv.segments) == labels
    assert kv.commit is False
    assert rope.advance == (0,) * len(labels)
    if len(labels) > 1:
        assert kv.combined_labels == {labels: LLMSubmodule.CFG_BATCHED_LABEL}
        pos = rope.pos_ids[LLMSubmodule.CFG_BATCHED_LABEL]
        assert pos.tolist() == [10 + i for i in range(len(labels)) for _ in range(span)]
    else:
        assert not kv.combined_labels
        assert rope.pos_ids["main"].tolist() == [10] * span
    assert step.steps["attn"].causal is False


@pytest.mark.parametrize("n_branches", [None, 1, 2, 3])
def test_preprocess_hands_the_forward_the_declared_count(n_branches):
    row = _image_gen_row(n_branches)
    out = LLMSubmodule.preprocess(
        _stub(), "image_gen",
        engine_inputs=types.SimpleNamespace(
            per_request_info={"r0": types.SimpleNamespace(step_metadata={})},
        ),
        inputs=[row],
    )
    assert "kwargs" not in out
    assert out.get(CFG_BRANCHES) == n_branches
    assert out["requires_cfg"] is True


def test_image_gen_shape_inputs_stay_bounded_across_output_sizes():
    """The output size is the client's, so the per-size device inputs are an LRU, not a dict
    that grows with every size requested."""
    from mstar.model.bagel.submodules import _IMAGE_GEN_SHAPES_CACHED

    stub = _stub(
        config=types.SimpleNamespace(latent_downsample=16, max_latent_size=64),
        _image_gen_shape_cache={},
        _get_text_vae_idxs=lambda n, device: LLMSubmodule._get_text_vae_idxs(None, n, device),
    )
    first = LLMSubmodule._image_gen_shape_inputs(stub, 256, 256, "cpu")
    for i in range(3 * _IMAGE_GEN_SHAPES_CACHED):
        LLMSubmodule._image_gen_shape_inputs(stub, 256, 256 + 16 * (i + 1), "cpu")
        # the size in use stays cached while others come and go
        assert LLMSubmodule._image_gen_shape_inputs(stub, 256, 256, "cpu") is first
    assert len(stub._image_gen_shape_cache) == _IMAGE_GEN_SHAPES_CACHED


def _request(k, first_iter, prune=True, **meta):
    cfg = types.SimpleNamespace(
        num_timesteps=N, timestep_shift=SHIFT,
        cfg_interval=_DEFAULTS["cfg_interval"],
        cfg_text_scale=_DEFAULTS["cfg_text_scale"],
        cfg_img_scale=_DEFAULTS["cfg_img_scale"],
    )
    stub = _stub(config=cfg, _cfg_prune=prune, _warned_loop_counter=False)
    info = types.SimpleNamespace(
        step_metadata={"requires_cfg": True, **meta},
        dynamic_loop_iter_counts={} if k is None else {IMAGE_GEN_LOOP: k},
    )
    return LLMSubmodule._image_gen_cfg_branches(stub, info, first_iter)


def test_branch_count_follows_the_loop_counter_and_request_kwargs():
    last = N - 2
    assert _request(0, True) == 2
    assert _request(last, False) == 1
    # per-request kwargs (as step metadata) override the config defaults
    assert _request(last, False, cfg_interval=[0.0, 1.0], cfg_img_scale=2.0) == 3
    assert _request(0, True, cfg_img_scale=2.0, cfg_interval=[0.0, 1.0]) == 3


def test_falls_back_to_all_branches_when_off_or_counter_is_not_tracking():
    assert _request(5, False, prune=False) is None
    assert _request(None, False) is None  # loop counter missing on a later step
    assert _request(3, True) is None      # seeded latents but counter says step 3
    assert _request(0, True, requires_cfg=False) is None


def test_image_gen_walk_names_the_loop_the_counter_is_read_from():
    """An unnamed Loop gets a random uuid, and `dynamic_loop_iter_counts`
    would never carry IMAGE_GEN_LOOP: every step would fall back."""
    from mstar.model.bagel.bagel_model import BagelModel

    model = BagelModel.__new__(BagelModel)
    model.config = types.SimpleNamespace(num_timesteps=N, max_position_embeddings=32768)
    model._has_cfg_parallel = False
    model._has_llm_disaggregation = False
    model._image_gen_remote_handoff = False
    loops = model.get_graph_walk_graphs()["image_gen"].get_loops()
    assert IMAGE_GEN_LOOP in loops
    assert loops[IMAGE_GEN_LOOP].max_iters == N - 1


# --- parallel CFG (image_gen_cfg) ---------------------------------------------


def test_parallel_branch_nodes_sit_out_the_branches_a_step_does_not_read():
    from mstar.model.bagel.submodules import cfg_branch_skipped

    # main always runs; cfg_text runs from 2 branches; cfg_img only at 3
    assert [cfg_branch_skipped("main", n) for n in (1, 2, 3)] == [False, False, False]
    assert [cfg_branch_skipped("cfg_text", n) for n in (1, 2, 3)] == [True, False, False]
    assert [cfg_branch_skipped("cfg_img", n) for n in (1, 2, 3)] == [True, True, False]
    # no count stamped (pruning off): every branch runs
    assert not any(cfg_branch_skipped(label, None) for label in ("main", "cfg_text", "cfg_img"))


def test_parallel_combine_reduces_to_the_old_combine():
    """combine_cfg with a host branch count gives what the device-side interval test gave."""
    from mstar.model.bagel.submodules import CombineCFGSubmodule

    cfg = types.SimpleNamespace(**{k: _DEFAULTS[k] for k in (
        "num_timesteps", "timestep_shift", "cfg_text_scale", "cfg_img_scale",
        "cfg_interval", "cfg_renorm_type", "cfg_renorm_min",
    )})
    node = CombineCFGSubmodule(cfg)
    g = torch.Generator().manual_seed(0)
    v = [torch.randn(16, 8, generator=g) for _ in range(3)]
    latents = torch.randn(16, 8, generator=g)
    for k in (0, N - 2):
        time_index = torch.full((16,), float(k))
        n = _branches(k, **T2I)
        tensors = dict(
            v_main=v[0], v_cfg_text=v[1], v_cfg_img=v[2], latents=latents, time_index=time_index,
        )
        old = node.forward("image_gen_cfg", None, **tensors)["latents"][0]
        new = node.forward("image_gen_cfg", None, **tensors, cfg_branches=n)["latents"][0]
        torch.testing.assert_close(new, old, rtol=0, atol=1e-6)


def _combine_inputs(k: int, sat_out: tuple[str, ...], counter: int | None = None):
    """combine_cfg's prepare_inputs at step ``k`` of a t2i request, with the named
    branch nodes having sat the step out and the combine's loop counter reading
    ``counter`` (default ``k``)."""
    from mstar.model.bagel.submodules import CombineCFGSubmodule, sat_out_velocity

    cfg = types.SimpleNamespace(cfg_prune=True, **{k_: _DEFAULTS[k_] for k_ in (
        "num_timesteps", "timestep_shift", "cfg_text_scale", "cfg_img_scale",
        "cfg_interval", "cfg_renorm_type", "cfg_renorm_min",
    )})
    g = torch.Generator().manual_seed(0)
    latents = torch.randn(16, 8, generator=g)
    inputs = {
        name: [sat_out_velocity(latents) if name[2:] in sat_out else torch.randn(16, 8, generator=g)]
        for name in ("v_main", "v_cfg_text", "v_cfg_img")
    }
    inputs.update(latents=[latents], time_index=[torch.full((16,), float(k))])
    info = types.SimpleNamespace(
        step_metadata={"requires_cfg": True},
        dynamic_loop_iter_counts={IMAGE_GEN_LOOP: k if counter is None else counter},
    )
    node = CombineCFGSubmodule(cfg)
    return node, node.prepare_inputs("image_gen_cfg", info, inputs)


def test_combine_never_reads_a_branch_that_sat_the_step_out():
    """A branch node counts from its own worker's loop counter: if it sat out a step
    whose combine reads its velocity, the request fails rather than combining a
    placeholder as the unconditional velocity."""
    k_out = N - 2
    assert _branches(k_out, **T2I) == 1
    node, prepared = _combine_inputs(k_out, sat_out=("cfg_text", "cfg_img"))
    assert set(prepared.tensor_inputs) == {"v_main", "latents", "time_index"}
    out = node.forward("image_gen_cfg", None, **prepared.tensor_inputs, **prepared.kwargs)
    assert out["latents"][0].shape == (16, 8)

    k_in = next(k for k in STEPS if k > 0 and _branches(k, **T2I) == 2)
    # t2i never reads cfg_img, so it sitting out is consistent
    _, prepared = _combine_inputs(k_in, sat_out=("cfg_img",))
    assert "v_cfg_img" not in prepared.tensor_inputs
    with pytest.raises(ValueError, match="cfg_text sat out"):
        _combine_inputs(k_in, sat_out=("cfg_text", "cfg_img"))

    # The combine's own counter isn't tracking the loop (reads 0 on a later step): it takes
    # the count from the branches that ran instead of failing the request.
    node, prepared = _combine_inputs(k_in, sat_out=("cfg_img",), counter=0)
    assert prepared.kwargs["cfg_branches"] == 2
    out = node.forward("image_gen_cfg", None, **prepared.tensor_inputs, **prepared.kwargs)
    _, reference = _combine_inputs(k_in, sat_out=("cfg_img",))
    expected = node.forward("image_gen_cfg", None, **reference.tensor_inputs, **reference.kwargs)
    torch.testing.assert_close(out["latents"][0], expected["latents"][0])


def test_image_gen_cfg_walk_names_the_loop_too():
    from mstar.model.bagel.bagel_model import BagelModel

    model = BagelModel.__new__(BagelModel)
    model.config = types.SimpleNamespace(num_timesteps=N, max_position_embeddings=32768)
    model._has_cfg_parallel = True
    model._has_llm_disaggregation = False
    model._image_gen_remote_handoff = False
    loops = model.get_graph_walk_graphs()["image_gen_cfg"].get_loops()
    assert IMAGE_GEN_LOOP in loops
