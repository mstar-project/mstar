"""CPU checks for the video / windowed denoise CUDA-graph capture: one graph
per latent shape built with every frame declared noisy, the request's
clean/noisy layout carried as a per-token mask input. Covers the bucket
declaration, the capture key (video, chained windows; never kv windows),
the captured inputs a windowed request stages, the captured-step tail in
``postprocess`` (masked velocity, pinned frames, window boundary), and the
mask's exact equivalence to the eager layout on a tiny transformer.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mstar.engine.windowing import WindowSchedule
from mstar.model.cosmos3 import constants as C
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.submodules import (
    COND_LABEL,
    SESSION_TAIL,
    UNCOND_LABEL,
    VIDEO_GEN_AR_LOOP,
    WINDOW_COND,
    Cosmos3DiTSubmodule,
)
from mstar.model.cosmos3.tests.stubs import StubSampler
from mstar.model.cosmos3.tests.test_engine_cache import _SdpaResources, declaring
from mstar.model.cosmos3.tests.test_rotary_table import stand_in_rotary
from mstar.model.submodule_base import ARNodeInputs


def _fake_transformer(cfg):
    return SimpleNamespace(
        proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
        comm_group=SimpleNamespace(world_size=1),
        _rotary=stand_in_rotary(cfg),
    )


def _dit_with_buckets(monkeypatch, **config):
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_RES", "64x64")
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_VIDEO", "64x64x9")
    monkeypatch.setenv("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH", "1")
    monkeypatch.delenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", raising=False)
    cfg = Cosmos3Config(compile_denoise=False, **config)
    sub = Cosmos3DiTSubmodule(transformer=_fake_transformer(cfg), config=cfg, scheduler=None)
    configs = sub.get_accelerator_graph_configs("cpu")
    return sub, configs


def test_video_capture_buckets_and_keys(monkeypatch) -> None:
    sub, configs = _dit_with_buckets(monkeypatch)
    ch = sub.config.latent_channel
    image_shape, video_shape = (1, ch, 1, 4, 4), (1, ch, 3, 4, 4)  # 9 frames -> 3 latent frames
    image_key, video_key = (image_shape, "cfg"), (video_shape, "cfg")
    assert [c.capture_graph_walk for c in configs] == [C.IMAGE_GEN_WALK, C.VIDEO_GEN_WALK]
    assert [c.additional_key_info for c in configs] == [image_key, video_key]
    assert configs[1].replay_graph_walks == [C.VIDEO_GEN_WALK, C.VIDEO_GEN_AR_WALK]
    video_inputs = configs[1].single_request_inputs.tensor_inputs
    assert video_inputs["noisy_token_mask"].shape == (12,) and video_inputs["noisy_token_mask"].all()
    assert video_inputs["vision_timesteps"].shape == (12,) and video_inputs["latents"].shape == video_shape
    assert set(sub._capture_layout) == {image_shape, video_shape}
    assert len(sub._capture_layout[video_shape]["vision_noisy_frame_indexes"][0]) == 3  # all frames noisy

    window = {"window_mode": "chained", "total_latent_units": 6, "window_latent_units": 3, "overlap_latent_units": 0}
    infos = {
        rid: SimpleNamespace(step_metadata={"height": 64, "width": 64, "guidance_scale": 6.0, **md})
        for rid, md in (
            ("img", {}), ("vid", {"num_frames": 9}), ("ch", window), ("kv", {**window, "window_mode": "kv"}),
            ("nocfg", {"num_frames": 9, "guidance_scale": 1.0}), ("odd", {"num_frames": 17}),
        )
    }

    def key(walk, *rids):
        return sub.cg_key_info(walk, {f"{rid}{i}": infos[rid] for i, rid in enumerate(rids)})

    assert key(C.IMAGE_GEN_WALK, "img") == image_key
    assert key(C.VIDEO_GEN_WALK, "vid") == video_key
    assert key(C.VIDEO_GEN_AR_WALK, "ch") == video_key
    assert key(C.VIDEO_GEN_AR_WALK, "kv") is None
    assert key(C.VIDEO_GEN_WALK, "nocfg") is None
    assert key(C.VIDEO_GEN_WALK, "odd") is None
    assert key(C.VIDEO_GEN_AR_WALK, "ch", "kv") is None  # mixed batch
    assert key(C.VIDEO_GEN_AR_WALK, "ch", "ch") == video_key
    assert key(C.PREFILL_WALK, "vid") == "cfg"
    assert key(C.PREFILL_WALK, "nocfg") == "cond"  # keyed apart; no such bucket here
    assert key(C.PREFILL_WALK, "vid", "nocfg") is None  # mixed regimes


def test_conditional_only_capture_when_guidance_is_off(monkeypatch) -> None:
    """A deployment whose default guidance is off (the distilled checkpoints)
    captures the conditional-only step: one branch's inputs, one plan's
    capacity, bs=1 only, and a guided request keys to no bucket."""
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_BS", "1,4")
    sub, configs = _dit_with_buckets(monkeypatch, guidance_scale=1.0)
    monkeypatch.setattr(sub, "get_device", lambda: torch.device("cpu"))
    ch = sub.config.latent_channel
    image_key = ((1, ch, 1, 4, 4), "cond")
    image = configs[0]
    assert image.additional_key_info == image_key
    assert image.capture_forward_method == "forward_captured_cond"
    assert image.total_tokens_multiplier == 1 and list(image.capture_batch_sizes) == [1]
    assert set(image.single_request_inputs.tensor_inputs) == {
        "latents", "vision_timesteps", "cos_cond", "sin_cond", "noisy_token_mask",
    }
    info = image.single_request_inputs.resource_step_info
    assert (info.cfg, info.cfg_active, info.capture_key) == (False, False, image_key)

    off, on = (
        SimpleNamespace(step_metadata={"height": 64, "width": 64, "guidance_scale": scale}) for scale in (1.0, 6.0)
    )
    assert sub.cg_key_info(C.IMAGE_GEN_WALK, {"off": off}) == image_key
    assert sub.cg_key_info(C.IMAGE_GEN_WALK, {"on": on}) is None
    # the captured declaration plans the conditional label alone
    step = sub.declare_step(
        C.IMAGE_GEN_WALK, ["off"], [image.single_request_inputs], slot_lease=object(),
    )
    assert step.cg_key_info == image_key
    assert {seg.label for seg in step.segments} == {COND_LABEL}
    # guided rows can still arrive in this deployment and batch, so its caps stay
    assert sub.max_batch_size(C.IMAGE_GEN_WALK) == sub.max_gen_batch_size
    assert sub.max_batch_size(C.PREFILL_WALK) is None
    # a distilled checkpoint's rows are all conditional-only: one per step where a
    # walk has a graph to replay (this deployment captures no prefill)
    sub.config.distilled_sigmas = (1.0, 0.5)
    assert sub.max_batch_size(C.IMAGE_GEN_WALK) == 1 and sub.max_batch_size(C.PREFILL_WALK) is None


def test_distilled_cap_only_on_captured_walks(monkeypatch) -> None:
    """A distilled deployment takes one row per step on the walks it captures;
    a walk with no capture (Super I2V's video steps) keeps its rows together."""
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_RES", "64x64")
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_VIDEO", "")
    monkeypatch.delenv("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH", raising=False)
    monkeypatch.delenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", raising=False)
    cfg = Cosmos3Config(compile_denoise=False, guidance_scale=1.0)
    sub = Cosmos3DiTSubmodule(transformer=_fake_transformer(cfg), config=cfg, scheduler=None)
    sub.config.distilled_sigmas = (1.0, 0.5)
    # before the capture configs are declared, nothing replays, so nothing is capped
    assert sub.max_batch_size(C.IMAGE_GEN_WALK) == sub.max_gen_batch_size
    sub.get_accelerator_graph_configs("cpu")
    assert sub.max_batch_size(C.IMAGE_GEN_WALK) == 1 and sub.max_batch_size(C.PREFILL_WALK) == 1
    assert sub.max_batch_size(C.VIDEO_GEN_WALK) == sub.max_gen_batch_size


def test_two_branch_deployment_keeps_its_batch_caps(monkeypatch) -> None:
    sub, _ = _dit_with_buckets(monkeypatch)
    assert sub.max_batch_size(C.IMAGE_GEN_WALK) == sub.max_gen_batch_size
    assert sub.max_batch_size(C.PREFILL_WALK) is None


def test_windowed_captured_inputs_carry_the_window_layout(monkeypatch) -> None:
    """A chained window past the first stages the graph's inputs: timesteps
    from the within-window step over every token, and the token mask zero on
    the re-pinned overlap frames."""
    sub, _ = _dit_with_buckets(monkeypatch)
    monkeypatch.setattr(sub, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: StubSampler(torch.tensor([900.0, 500.0, 100.0])))
    sub._solver_state = ()
    ch = sub.config.latent_channel
    md = {
        "window_mode": "chained", "total_latent_units": 6, "window_latent_units": 3, "overlap_latent_units": 1,
        "height": 64, "width": 64, "num_inference_steps": 3, "guidance_scale": 6.0, C.PROMPT_LENGTHS: (7, 9),
    }
    x = torch.zeros(1, ch, 3, 4, 4)
    counts = {VIDEO_GEN_AR_LOOP: 4}
    fwd = SimpleNamespace(
        request_id="r", rid_handle="r", random_seed=0, step_metadata=md, dynamic_loop_iter_counts=counts,
    )
    edges = {"latents": [x], WINDOW_COND: [x], SESSION_TAIL: [torch.zeros(0)]}
    ni = sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, edges)
    t = ni.tensor_inputs
    assert torch.equal(t["noisy_token_mask"], torch.tensor([0.0] * 4 + [1.0] * 8))
    assert t["vision_timesteps"].shape == (12,) and torch.all(t["vision_timesteps"] == 500.0)  # window 1, step 1
    assert ni.resource_step_info.capture_key == ((1, ch, 3, 4, 4), "cfg")
    # The per-frame mask rides alongside, in the window's statics.
    assert torch.equal(ni.kwargs["settings"]["cond"]["noisy_frame_mask"].flatten(), torch.tensor([0.0, 1.0, 1.0]))
    # A kv request at the same shape stages no captured inputs (it never leases).
    fwd.step_metadata = {**md, "window_mode": "kv", "overlap_latent_units": 0}
    counts[VIDEO_GEN_AR_LOOP] = 1
    ni_kv = sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, edges)
    assert "noisy_token_mask" not in ni_kv.tensor_inputs and ni_kv.resource_step_info.capture_key is None


def test_postprocess_captured_video_step() -> None:
    """The captured tail zeroes the clean frames' velocity, re-pins masked
    frames, and ends a chained window at its last step."""
    cfg = Cosmos3Config(compile_denoise=False)
    sub = Cosmos3DiTSubmodule(transformer=_fake_transformer(cfg), config=cfg, scheduler=None)
    # No VAE here: the next window's head is the raw tail (test_windowed pins the real head).
    sub._window_head = lambda w, units: w[:, :, -units:]
    ch = sub.config.latent_channel
    sub._solver_state = ()
    cond = {"vision_token_shapes": [(3, 2, 2)], "vision_noisy_frame_indexes": [torch.tensor([1, 2])],
            "num_vision_tokens": 12}
    sub._add_layout_masks(cond, "cpu")
    pinned = torch.full((1, ch, 3, 4, 4), 7.0)
    vmask = torch.zeros(1, 1, 3, 1, 1)
    vmask[:, :, 0] = 1.0
    settings = {
        "cond": cond, "uncond": cond, "gs": 1.0, "scheduler": StubSampler(torch.tensor([900.0, 500.0])),
        "latent_shape": (1, ch, 3, 4, 4), "ar_schedule": WindowSchedule(6, 3, overlap_units=1), "ar_steps": 2,
        "ar_iters_per_window": 2, "ar_kv_mode": False, "window_index": 0, "cond_units": 1, "vmask": vmask,
        "cond_video_latents": pinned, "session_id": None, "session_tail": None,
    }
    lat = torch.zeros(1, ch, 3, 4, 4)  # the loop's latents carry the batch dim, like the velocities
    inputs = ARNodeInputs(tensor_inputs={"latents": lat}, kwargs={"settings": settings})
    velocity = torch.ones(1, ch, 3, 4, 4)
    counts = {VIDEO_GEN_AR_LOOP: 0}
    info = SimpleNamespace(graph_walk=C.VIDEO_GEN_AR_WALK, dynamic_loop_iter_counts=counts)
    # Window 0, step 0 (global 0): frame 0 keeps the pinned latents, noisy
    # frames move by the velocity.
    out = {"cond_v": [velocity], "uncond_v": [velocity]}
    sub.postprocess("r", info, out, inputs=inputs)
    new = out["latents"][0]  # [1, C, T, H, W]
    assert new.shape == (1, ch, 3, 4, 4)
    assert torch.all(new[:, :, 0] == 7.0) and torch.all(new[:, :, 1:] == -1.0)
    assert "window_latents" not in out
    # Window 0, step 1 (global 1) is the window's last step: it emits the
    # window and the next window's pinned head.
    out = {"cond_v": [velocity], "uncond_v": [velocity]}
    counts[VIDEO_GEN_AR_LOOP] = 1
    sub.postprocess("r", info, out, inputs=inputs)
    assert torch.equal(out["window_latents"][0], out["latents"][0])
    assert out[WINDOW_COND][0].shape == (1, ch, 3, 4, 4)


def test_masked_capture_matches_eager_layout() -> None:
    """On a tiny transformer: the all-frames-noisy graph layout with the
    clean/noisy mask reproduces the eager i2v layout's velocity on the noisy
    frames exactly (the clean frame's output is what postprocess zeroes)."""
    from mstar.model.cosmos3.components.packing import build_static_inputs
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
    from mstar.model.cosmos3.tests.test_edge import _init_all, _tiny_edge_config

    cfg = _tiny_edge_config()
    model = Cosmos3OmniTransformer(cfg).eval()
    _init_all(model)
    res = _SdpaResources()
    res.bind(declaring(cfg), model)
    ids = [3, 5, 7, 11, 13]
    latents = torch.randn(1, cfg.latent_channel, 3, 4, 4)
    shape = tuple(latents.shape)
    eager = build_static_inputs(ids, shape, cfg, 4, 24.0, "cpu", has_image_condition=True)
    graph = build_static_inputs(ids, shape, cfg, 4, 24.0, "cpu", has_image_condition=False)
    n_all = graph["num_vision_tokens"]
    assert eager["num_noisy_vision_tokens"] == n_all - 4  # frame 0 clean
    per_frame = n_all // 3
    mask = torch.cat([torch.zeros(per_frame), torch.ones(n_all - per_frame)])
    both = [("r", COND_LABEL, n_all), ("r", UNCOND_LABEL, n_all)]
    with torch.no_grad():
        for label in (COND_LABEL, UNCOND_LABEL):
            res.step(C.PREFILL_WALK, [("r", label, eager["und_len"])],
                     lambda: model.prefill_und(eager["input_ids"], eager["text_mrope_ids"]))
        t_noisy = torch.full((eager["num_noisy_vision_tokens"],), 600.0)
        rot_e = model._rotary(eager["gen_position_ids"], latents.device, latents.dtype)
        rot_g = model._rotary(graph["gen_position_ids"], latents.device, latents.dtype)
        cond_e, uncond_e = res.step(C.VIDEO_GEN_WALK, both, lambda: model.denoise_step_batched_cfg(
            latents, t_noisy, rot_e, rot_e,
            eager["vision_token_shapes"], eager["vision_noisy_frame_indexes"],
            eager["vision_mse_loss_indexes"] - eager["und_len"],
        ), keep=False)
        t_all = torch.full((n_all,), 600.0)
        cond_g, uncond_g = res.step(C.VIDEO_GEN_WALK, both, lambda: model.denoise_step_batched_cfg(
            latents, t_all, rot_g, rot_g,
            graph["vision_token_shapes"], graph["vision_noisy_frame_indexes"],
            graph["vision_mse_loss_indexes"] - graph["und_len"], noisy_token_mask=mask,
        ), keep=False)
    for e, g in ((cond_e, cond_g), (uncond_e, uncond_g)):  # [1, C, T, H, W]
        assert torch.all(e[:, :, 0] == 0)  # the eager unpatchify leaves the clean frame at zero velocity
        assert torch.allclose(e[:, :, 1:], g[:, :, 1:], atol=1e-5, rtol=1e-4)
        assert not torch.all(g[:, :, 0] == 0)  # the graph predicts it too; postprocess masks it away


def test_conditional_capture_matches_eager_single_branch() -> None:
    """``forward_captured_cond`` on a tiny transformer, at the all-frames-noisy
    graph layout with the clean/noisy mask, gives the eager single-branch
    ``denoise_step``'s velocity on the noisy frames, for its one row."""
    from mstar.model.cosmos3.components.packing import build_static_inputs
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
    from mstar.model.cosmos3.tests.test_edge import _init_all, _tiny_edge_config

    cfg = _tiny_edge_config()
    model = Cosmos3OmniTransformer(cfg).eval()
    _init_all(model)
    res = _SdpaResources()
    res.bind(declaring(cfg), model)
    ids = [3, 5, 7, 11, 13]
    latents = torch.randn(1, cfg.latent_channel, 3, 4, 4)
    shape = tuple(latents.shape)
    eager = build_static_inputs(ids, shape, cfg, 4, 24.0, "cpu", has_image_condition=True)
    graph = build_static_inputs(ids, shape, cfg, 4, 24.0, "cpu", has_image_condition=False)
    n_all = graph["num_vision_tokens"]
    per_frame = n_all // 3
    mask = torch.cat([torch.zeros(per_frame), torch.ones(n_all - per_frame)])
    layout = {
        "vision_token_shapes": graph["vision_token_shapes"],
        "vision_noisy_frame_indexes": graph["vision_noisy_frame_indexes"],
        "mse_gen_indexes": graph["vision_mse_loss_indexes"] - graph["und_len"],
    }
    sub = SimpleNamespace(transformer=model, _capture_layout={shape: layout})
    cond = [("r", COND_LABEL, n_all)]
    with torch.no_grad():
        res.step(C.PREFILL_WALK, [("r", COND_LABEL, eager["und_len"])],
                 lambda: model.prefill_und(eager["input_ids"], eager["text_mrope_ids"]))
        rot_e = model._rotary(eager["gen_position_ids"], latents.device, latents.dtype)
        cos_g, sin_g = model._rotary(graph["gen_position_ids"], latents.device, latents.dtype)
        want = res.step(C.VIDEO_GEN_WALK, cond, lambda: model.denoise_step(
            latents, torch.full((eager["num_noisy_vision_tokens"],), 600.0), rot_e,
            eager["vision_token_shapes"], eager["vision_noisy_frame_indexes"],
            eager["vision_mse_loss_indexes"] - eager["und_len"],
        ), keep=False)
        # the captured inputs carry a leading batch dimension of one row
        out = res.step(C.VIDEO_GEN_WALK, cond, lambda: Cosmos3DiTSubmodule.forward_captured_cond(
            sub, C.IMAGE_GEN_WALK, SimpleNamespace(request_ids=["r"]),
            latents[None], torch.full((1, n_all), 600.0), cos_g[None], sin_g[None], mask[None],
        ), keep=False)
    assert list(out) == ["r"] and list(out["r"]) == ["cond_v"]
    got = out["r"]["cond_v"][0]  # [1, C, T, H, W]
    assert torch.all(want[:, :, 0] == 0)  # the eager unpatchify leaves the clean frame at zero velocity
    assert torch.allclose(want[:, :, 1:], got[:, :, 1:], atol=1e-5, rtol=1e-4)


def test_prefill_capture_inputs_follow_the_regime(monkeypatch) -> None:
    """The captured prefill's rows declare the labels of their regime, so the
    capture's own `declare_step` plans exactly those."""
    for scale, labels in ((6.0, {"main", "uncond"}), (1.0, {"main"})):
        monkeypatch.setenv("COSMOS3_GEN_CAPTURE_RES", "64x64")
        monkeypatch.delenv("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH", raising=False)
        monkeypatch.delenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", raising=False)
        cfg = Cosmos3Config(compile_denoise=False, guidance_scale=scale)
        sub = Cosmos3DiTSubmodule(transformer=_fake_transformer(cfg), config=cfg, scheduler=None)
        prefill = next(c for c in sub.get_accelerator_graph_configs("cpu") if c.capture_graph_walk == C.PREFILL_WALK)
        assert prefill.additional_key_info == ("cfg" if len(labels) == 2 else "cond")
        (row,) = prefill.get_node_inputs(1, 63)
        assert row.kwargs["cfg"] == (len(labels) == 2) and set(row.kwargs["seq_lens"]) == labels
        assert sum(row.kwargs["seq_lens"].values()) == 63
        step = sub.declare_step(C.PREFILL_WALK, ["r"], [row])
        assert {seg.label for seg in step.segments} == labels
        assert step.cg_key_info == prefill.additional_key_info
