"""BAGEL image_gen runs only the CFG branches a step needs.

Guidance on, the step used to run main, cfg_text and cfg_img as one batched
forward, though the combine never reads ``v_cfg_img`` when the image scale is
<= 1 (the text-to-image default), and outside the CFG interval it reduces to
``v_main``. ``cfg_branches`` decides 1/2/3 on the host; a request's guided
steps are one window (``cfg_window``), which runs as its own walk with a fixed
branch count (``image_gen_phases``). These pin the
decision table, its agreement with the device-side interval test the step used
to run, the phases reproducing the per-step count, the conductor's walk through
them, the combine reducing to what the always-three-branch one computed, and
the declaration matching the branch count. CPU only.
"""

from __future__ import annotations

import itertools
import types

import pytest
import torch

from mstar.conductor.request_info import DEFAULT_PARTITION, CurrentForwardPassInfo
from mstar.engine.resources import KVStep, PositionStep
from mstar.model.bagel.config import BagelAutoEncoderConfig, BagelModelConfig, BagelViTConfig
from mstar.model.bagel.submodules import (
    CFG_WALKS,
    DENOISE_WALKS,
    IMAGE_DECODE,
    IMAGE_GEN,
    IMAGE_GEN_HEAD,
    LLMSubmodule,
    apply_timestep_shift,
    cfg_branches,
    cfg_window,
    combine_cfg_velocity,
    denoise_loop,
    image_gen_phases,
    image_gen_timestep,
)
from mstar.model.submodule_base import ARNodeInputs

IMAGE_GEN_LOOP = denoise_loop(IMAGE_GEN)

CONFIG = BagelModelConfig(vae_config=BagelAutoEncoderConfig(), vit_config=BagelViTConfig())
N = CONFIG.num_timesteps
SHIFT = CONFIG.timestep_shift
T2I = dict(  # text-to-image request defaults
    cfg_interval=tuple(CONFIG.cfg_interval),
    text_scale=CONFIG.cfg_text_scale,
    img_scale=CONFIG.cfg_img_scale,
)
EDIT = dict(cfg_interval=(0.0, 1.0), text_scale=4.0, img_scale=2.0)  # bagel_model edit overrides
STEPS = range(N - 1)  # time_index 0 .. N-2


def _device_timestep(k: int) -> torch.Tensor:
    """What `_forward_image_gen` computes from its time_index input."""
    time_index = torch.full((8,), float(k), dtype=torch.float32)
    t_uniform = 1.0 - time_index / (N - 1)
    return apply_timestep_shift(t=t_uniform, shift=SHIFT)


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


def _llm() -> LLMSubmodule:
    return LLMSubmodule(
        language_model=types.SimpleNamespace(model=types.SimpleNamespace(embed_tokens=None), lm_head=None),
        llm2vae=None, vae2llm=None, time_embedder=None, latent_pos_embed=None, config=CONFIG,
    )


def _info(graph_walk: str = "image_gen", **fields) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(request_id="r0", graph_walk=graph_walk, fwd_index=0, random_seed=0, max_tokens=0,
                                  **fields)


def _image_gen_row(span: int = 6) -> ARNodeInputs:
    row = ARNodeInputs(input_seq_len=span)
    row.resource_step_info = True  # requires_cfg
    row.custom_pos_ids = {
        label: torch.full((span,), 10 + i, dtype=torch.int32)
        for i, label in enumerate(("main", "cfg_text", "cfg_img"))
    }
    return row


@pytest.mark.parametrize("walk,labels", [
    ("image_gen_guided3", ("main", "cfg_text", "cfg_img")),
    ("image_gen_guided2", ("main", "cfg_text")),
    (IMAGE_GEN, ("main",)),
    (IMAGE_GEN_HEAD, ("main",)),
])
def test_declared_step_plans_exactly_the_branches_run(walk, labels):
    span = 6
    step = _llm().declare_step(walk, ["r0"], [_image_gen_row(span)])
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


def test_image_gen_shape_inputs_stay_bounded_across_output_sizes():
    """The output size is the client's, so the per-size device inputs are an LRU, not a dict
    that grows with every size requested."""
    from mstar.model.bagel.submodules import _IMAGE_GEN_SHAPES_CACHED

    llm = _llm()
    first = llm._image_gen_shape_inputs(256, 256, "cpu")
    for i in range(3 * _IMAGE_GEN_SHAPES_CACHED):
        llm._image_gen_shape_inputs(256, 256 + 16 * (i + 1), "cpu")
        # the size in use stays cached while others come and go
        assert llm._image_gen_shape_inputs(256, 256, "cpu") is first
    assert len(llm._image_gen_shape_cache) == _IMAGE_GEN_SHAPES_CACHED


def _old_image_gen_step_branches(config, fwd_info: CurrentForwardPassInfo, first_iter: bool) -> int:
    """The per-step branch count before the denoise phases (a copy of the old
    `image_gen_step_branches`): the oracle the phase schedule must reproduce."""
    meta = fwd_info.step_metadata
    if not meta.get("requires_cfg", False):
        return 1
    k = fwd_info.dynamic_loop_iter_counts.get(IMAGE_GEN_LOOP, 0)
    if (k == 0) != first_iter:
        raise ValueError(
            f"BAGEL image_gen: loop counter {IMAGE_GEN_LOOP!r} reads {k} on a "
            f"{'first' if first_iter else 'later'} iteration"
        )
    t = image_gen_timestep(k, config.num_timesteps, config.timestep_shift)
    return cfg_branches(
        t,
        tuple(meta.get("cfg_interval", config.cfg_interval)),
        meta.get("cfg_text_scale", config.cfg_text_scale),
        meta.get("cfg_img_scale", config.cfg_img_scale),
    )


@pytest.mark.parametrize("num_timesteps,shift", [(50, 3.0), (30, 1.0)])
@pytest.mark.parametrize("scales", [(4.0, 1.0), (4.0, 2.0), (1.0, 1.0)], ids=["t2i", "edit", "no_cfg"])
@pytest.mark.parametrize("interval", [(0.4, 1.0), (0.0, 1.0), (0.1, 0.7)])
@pytest.mark.parametrize("cfg_parallel", [False, True])
def test_phase_schedule_runs_the_old_branch_count_at_every_step(num_timesteps, shift, scales, interval, cfg_parallel):
    """The walks tile the steps in order, only the last decodes, and every step
    runs the branches the old per-step count gave it, as its walk's fixed count,
    on one node or on the branch nodes."""
    config = BagelModelConfig(vae_config=BagelAutoEncoderConfig(), vit_config=BagelViTConfig())
    config.num_timesteps, config.timestep_shift = num_timesteps, shift
    text_scale, img_scale = scales
    meta = {"requires_cfg": text_scale > 1.0 or img_scale > 1.0, "cfg_interval": list(interval),
            "cfg_text_scale": text_scale, "cfg_img_scale": img_scale}
    window = cfg_window(num_timesteps, shift, interval, text_scale, img_scale)
    phases = image_gen_phases(window, num_timesteps, cfg_parallel)

    assert sum(p.steps for p in phases) == num_timesteps - 1
    decodes = [p.walk == IMAGE_DECODE or DENOISE_WALKS[p.walk].decode for p in phases]
    assert decodes == [False] * (len(phases) - 1) + [True]
    windows = [DENOISE_WALKS[p.walk] for p in phases if p.walk in DENOISE_WALKS and DENOISE_WALKS[p.walk].branches > 1]
    assert all(walk.split == cfg_parallel for walk in windows)
    starts = list(itertools.accumulate((p.steps for p in phases), initial=0))
    old = []
    new = []
    for k in range(num_timesteps - 1):
        info = _info(step_metadata=meta, dynamic_loop_iter_counts={IMAGE_GEN_LOOP: k})
        old.append(_old_image_gen_step_branches(config, info, first_iter=k == 0))
        phase = next(p for i, p in enumerate(phases) if starts[i] <= k < starts[i + 1])
        new.append(DENOISE_WALKS[phase.walk].branches)
    assert new == old
    if set(old) == {1}:
        # nothing guided: image_gen alone
        assert phases == ((IMAGE_GEN, num_timesteps - 1),)


def _pointer(uuid):
    from mstar.graph.base import TensorPointerInfo

    return TensorPointerInfo(dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=[1], uuid=uuid,
                             source_session_id="", source_entity="worker_0")


def _image_request_intake(kwargs, cfg_parallel=True):
    """A text-to-image request through BAGEL's intake: the model, the request's
    `CFGWindow`, the kwargs process_prompt hands on, and its first forward."""
    from mstar.model.bagel.bagel_model import CFG_WINDOW, BagelModel

    model = BagelModel.__new__(BagelModel)
    model.config = CONFIG
    model._has_cfg_parallel = cfg_parallel
    model._has_llm_disaggregation = False
    model._image_gen_remote_handoff = False
    params = model._request_params(["text"], ["image"], kwargs)
    window = cfg_window(N, SHIFT, tuple(params["cfg_interval"]), params["cfg_text_scale"], params["cfg_img_scale"])
    model_kwargs = {**kwargs, CFG_WINDOW: tuple(window)}
    fwd = model.get_initial_forward_pass_args(
        DEFAULT_PARTITION, ["text"], ["image"], {"text_inputs": [_pointer(0)]}, model_kwargs,
    )
    return model, window, model_kwargs, fwd


@pytest.mark.parametrize("cfg_parallel", [True, False], ids=["cfg_parallel", "one_node"])
@pytest.mark.parametrize("kwargs,walks", [
    ({}, ["image_gen_cfg2", IMAGE_GEN]),
    ({"cfg_interval": [0.0, 1.0], "cfg_img_scale": 2.0}, ["image_gen_cfg3", IMAGE_DECODE]),
    ({"cfg_interval": [0.1, 0.7]}, [IMAGE_GEN_HEAD, "image_gen_cfg2", IMAGE_GEN]),
    ({"cfg_text_scale": 1.0}, [IMAGE_GEN]),
], ids=["t2i", "edit_scales", "window_inside", "no_cfg"])
def test_conductor_walks_a_request_through_its_phases(kwargs, walks, cfg_parallel):
    """Every rank follows the conductor's walk: after prefill the request runs
    its phases in order, each loop capped at its steps, each walk after the
    first fed the previous one's persisted latents and time_index on every
    node that reads them, and the request ends after the decode. On one node
    the window is a walk of its own too, so its branch count is fixed."""
    if not cfg_parallel:
        walks = [walk.replace("_cfg", "_guided") for walk in walks]
    model, window, model_kwargs, fwd = _image_request_intake(kwargs, cfg_parallel)
    graphs = model.get_graph_walk_graphs()
    metadata, persist, seen, loops = fwd.full_metadata, {}, [], []
    while not (fwd := model.get_partition_forward_pass_args(DEFAULT_PARTITION, metadata, persist)).request_done:
        walk = metadata.graph_walk
        fed = {(edge.next_node, edge.name): [info.uuid for info in edge.tensor_info] for edge in fwd.inputs}
        # the conductor drops what a walk unpersists from the request's persist signals
        unpersisted = {info.uuid for info in fwd.unpersist_tensors}
        assert {uuid for uuids in fed.values() for uuid in uuids} <= unpersisted
        persist = {name: [i for i in infos if i.uuid not in unpersisted] for name, infos in persist.items()}
        expected = [] if not seen else [100 * len(seen) + i for i in range(2)]
        if walk == IMAGE_DECODE:
            assert fed == {("vae_decoder", "latents"): expected[:1]}
        else:
            readers = ("LLM",) if walk not in CFG_WALKS else (
                ("LLM", "LLM_cfg_text", "LLM_cfg_img")[:CFG_WALKS[walk]] + ("combine_cfg",)
            )
            assert fed == {(node, name): expected[i:i + 1] for node in readers
                           for i, name in enumerate(("latents", "time_index"))}
            assert set(graphs[walk].get_nodes()) - {"vae_decoder"} == set(readers)
            assert denoise_loop(walk) in graphs[walk].get_loops()
            loops.append(denoise_loop(walk))
        seen.append(walk)
        # the walk's loop persists where it left off, appended as the conductor does
        persist.setdefault("latents", []).append(_pointer(100 * len(seen)))
        persist.setdefault("time_index", []).append(_pointer(100 * len(seen) + 1))
    assert seen == walks
    phases = model._image_gen_phases(window)
    assert [p.walk for p in phases] == walks
    caps = model.get_loop_caps(**model_kwargs)
    assert {name: caps[name] for name in loops} == {denoise_loop(p.walk): p.steps for p in phases if p.steps}


_TEXT_ONLY = {("LLM", "prefill_text"): ["cfg_text"], ("LLM", "prefill_vit"): ["cfg_text"],
              ("LLM", "prefill_vae"): ["cfg_text"]}
_BOTH = {("LLM", "prefill_text"): ["cfg_text", "cfg_img"], ("LLM", "prefill_vit"): ["cfg_text"],
         ("LLM", "prefill_vae"): ["cfg_text"]}


@pytest.mark.parametrize("kwargs,cfg_parallel,published,final", [
    ({}, True, _TEXT_ONLY, {}),
    ({"cfg_interval": [0.0, 1.0], "cfg_img_scale": 2.0}, True, _BOTH, {("LLM", "decode"): ["cfg_img"]}),
    ({"cfg_interval": [0.5, 0.5]}, True, {}, {}),
    # one node runs every branch: what it publishes is unchanged
    ({}, False, _BOTH, {("LLM", "decode"): ["cfg_img"]}),
], ids=["t2i", "edit_scales", "empty_window", "single_node"])
def test_a_guided_request_publishes_only_the_kv_its_cfg_walk_reads(kwargs, cfg_parallel, published, final):
    """On a CFG-parallel deployment the branch labels' readers are the CFG
    walk's branch nodes, so the main LLM exports only those, and nothing for a
    guided request whose window has no step."""

    model, _, model_kwargs, fwd = _image_request_intake(kwargs, cfg_parallel)
    # a window with no step runs no guidance anywhere
    assert fwd.full_metadata.kwargs["requires_cfg"] == bool(published)
    kv = model.get_request_resource_configs({DEFAULT_PARTITION: fwd}, model_kwargs)["kv"]
    assert kv.publish_labels_per_node_walk == published
    assert kv.final_publish_labels_per_node_walk == final
