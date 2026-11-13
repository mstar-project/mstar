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


def _dit_with_buckets(monkeypatch):
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_RES", "64x64")
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_VIDEO", "64x64x9")
    monkeypatch.setenv("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH", "1")
    monkeypatch.delenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", raising=False)
    cfg = Cosmos3Config(compile_denoise=False)
    sub = Cosmos3DiTSubmodule(transformer=_fake_transformer(cfg), config=cfg, scheduler=None)
    configs = sub.get_accelerator_graph_configs("cpu")
    return sub, configs


def test_video_capture_buckets_and_keys(monkeypatch) -> None:
    sub, configs = _dit_with_buckets(monkeypatch)
    ch = sub.config.latent_channel
    image_shape, video_shape = (1, ch, 1, 4, 4), (1, ch, 3, 4, 4)  # 9 frames -> 3 latent frames
    assert [c.capture_graph_walk for c in configs] == [C.IMAGE_GEN_WALK, C.VIDEO_GEN_WALK]
    assert [c.additional_key_info for c in configs] == [image_shape, video_shape]
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

    assert key(C.IMAGE_GEN_WALK, "img") == image_shape
    assert key(C.VIDEO_GEN_WALK, "vid") == video_shape
    assert key(C.VIDEO_GEN_AR_WALK, "ch") == video_shape
    assert key(C.VIDEO_GEN_AR_WALK, "kv") is None
    assert key(C.VIDEO_GEN_WALK, "nocfg") is None
    assert key(C.VIDEO_GEN_WALK, "odd") is None
    assert key(C.VIDEO_GEN_AR_WALK, "ch", "kv") is None  # mixed batch
    assert key(C.VIDEO_GEN_AR_WALK, "ch", "ch") == video_shape
    assert key(C.PREFILL_WALK, "vid") is True


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
    assert ni.resource_step_info.capture_key == (1, ch, 3, 4, 4)
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
