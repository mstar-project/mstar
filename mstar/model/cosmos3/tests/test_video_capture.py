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
from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule
from mstar.model.submodule_base import ARNodeInputs


def _fake_transformer():
    return SimpleNamespace(
        proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
        sp_group=SimpleNamespace(world_size=1), comm_group=SimpleNamespace(world_size=1),
    )


def _dit_with_buckets(monkeypatch):
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_RES", "64x64")
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_VIDEO", "64x64x9")
    monkeypatch.setenv("COSMOS3_DISABLE_PREFILL_CUDA_GRAPH", "1")
    monkeypatch.delenv("COSMOS3_DISABLE_CUDA_GRAPH", raising=False)
    sub = Cosmos3DiTSubmodule(
        transformer=_fake_transformer(), config=Cosmos3Config(compile_denoise=False), scheduler=None,
    )
    configs = sub.get_cuda_graph_configs("cpu")
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

    def st(shape, kv=None, cfg=True):
        d = {"latent_shape": shape, "uncond": {} if cfg else None}
        if kv is not None:
            d.update(ar_schedule=WindowSchedule(6, 3), ar_kv_mode=kv)
        return d

    sub.request_states.clear()
    for rid, state in (("img", st(image_shape)), ("vid", st(video_shape)), ("ch", st(video_shape, kv=False)),
                       ("kv", st(video_shape, kv=True)), ("nocfg", st(video_shape, cfg=False)),
                       ("odd", st((1, ch, 5, 4, 4)))):
        sub.request_state(rid).add_all(**state)
    info = lambda *rids: {r: None for r in rids}  # noqa: E731
    assert sub.cg_key_info(C.IMAGE_GEN_WALK, info("img")) == image_shape
    assert sub.cg_key_info(C.VIDEO_GEN_WALK, info("vid")) == video_shape
    assert sub.cg_key_info(C.VIDEO_GEN_AR_WALK, info("ch")) == video_shape
    assert sub.cg_key_info(C.VIDEO_GEN_AR_WALK, info("kv")) is None
    assert sub.cg_key_info(C.VIDEO_GEN_WALK, info("nocfg")) is None
    assert sub.cg_key_info(C.VIDEO_GEN_WALK, info("odd")) is None
    assert sub.cg_key_info(C.VIDEO_GEN_AR_WALK, info("ch", "kv")) is None  # mixed batch
    assert sub.cg_key_info(C.VIDEO_GEN_AR_WALK, info("ch", "ch")) == video_shape
    assert sub.cg_key_info(C.PREFILL_WALK, info("vid")) is True


def test_windowed_captured_inputs_carry_the_window_layout(monkeypatch) -> None:
    """A chained window past the first stages the graph's inputs: timesteps
    from the within-window step over every token, and the token mask zero on
    the re-pinned overlap frames."""
    sub, _ = _dit_with_buckets(monkeypatch)
    monkeypatch.setattr(sub, "get_device", lambda: torch.device("cpu"))
    ch = sub.config.latent_channel
    cond, uncond = sub._build_window_statics(
        list(range(7)), list(range(9)), 64, 64, 3, 24.0,
        has_image_condition=False, cond_units=1, device="cpu", first_window=False,
    )
    sched = SimpleNamespace(timesteps=torch.tensor([900.0, 500.0, 100.0]))
    st = sub.request_state("r")
    st.add_all(
        cond=cond, uncond=uncond, gs=6.0, guidance_interval=None, scheduler=sched,
        latent_shape=(1, ch, 3, 4, 4), ar_schedule=WindowSchedule(6, 3, overlap_units=1),
        ar_steps=3, ar_iters_per_window=3, ar_total_iters=9, ar_kv_mode=False,
    )
    x = torch.zeros(1, ch, 3, 4, 4)
    fwd = SimpleNamespace(request_id="r", random_seed=0)
    ni = sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, {"latents": [x], "time_index": [torch.tensor([4])]})
    t = ni.tensor_inputs
    assert torch.equal(t["noisy_token_mask"], torch.tensor([0.0] * 4 + [1.0] * 8))
    assert t["vision_timesteps"].shape == (12,) and torch.all(t["vision_timesteps"] == 500.0)  # window 1, step 1
    assert ni.resource_step_info.capture_key == (1, ch, 3, 4, 4)
    # The per-frame mask is cached alongside, keyed on the window's statics.
    assert torch.equal(sub._noisy_masks(st, "cpu")[1].flatten(), torch.tensor([0.0, 1.0, 1.0]))
    # A kv request at the same shape stages no captured inputs (it never leases).
    st.add("ar_kv_mode", True)
    ni_kv = sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, {"latents": [x], "time_index": [torch.tensor([1])]})
    assert "noisy_token_mask" not in ni_kv.tensor_inputs and ni_kv.resource_step_info.capture_key is None


def test_postprocess_captured_video_step(monkeypatch) -> None:
    """The captured tail zeroes the clean frames' velocity, re-pins masked
    frames, and closes a chained window at its last step."""
    sub = Cosmos3DiTSubmodule(
        transformer=_fake_transformer(), config=Cosmos3Config(compile_denoise=False), scheduler=None,
    )
    ch = sub.config.latent_channel
    sched = SimpleNamespace(timesteps=torch.tensor([900.0, 500.0]),
                            step=lambda v, t, lat, return_dict=False: (lat - v,))
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: sched)
    monkeypatch.setattr(sub, "_window_statics_for", lambda st, plan, dev: (st["cond"], st["uncond"]))
    cond = {"vision_token_shapes": [(3, 2, 2)], "vision_noisy_frame_indexes": [torch.tensor([1, 2])],
            "num_vision_tokens": 12}
    st = sub.request_state("r")
    pinned = torch.full((1, ch, 3, 4, 4), 7.0)
    vmask = torch.zeros(1, 1, 3, 1, 1)
    vmask[:, :, 0] = 1.0
    st.add_all(
        cond=cond, uncond=cond, gs=1.0, scheduler=sched, latent_shape=(1, ch, 3, 4, 4),
        ar_schedule=WindowSchedule(6, 3, overlap_units=1), ar_steps=2, ar_iters_per_window=2,
        ar_total_iters=4, ar_kv_mode=False, ar_size=(64, 64), ar_generator=torch.Generator().manual_seed(0),
        vmask=vmask, cond_video_latents=pinned,
    )
    lat = torch.zeros(1, ch, 3, 4, 4)  # the loop's latents carry the batch dim, like the velocities
    velocity = torch.ones(1, ch, 3, 4, 4)
    info = SimpleNamespace(graph_walk=C.VIDEO_GEN_AR_WALK)
    # Window 0, step 0 (global 0): frame 0 keeps the pinned latents, noisy
    # frames move by the velocity.
    out = {"cond_v": [velocity], "uncond_v": [velocity]}
    step0 = ARNodeInputs(tensor_inputs={"latents": lat, "time_index": torch.tensor([0])})
    sub.postprocess("r", info, out, inputs=step0)
    new = out["latents"][0]  # [1, C, T, H, W]
    assert new.shape == (1, ch, 3, 4, 4)
    assert torch.all(new[:, :, 0] == 7.0) and torch.all(new[:, :, 1:] == -1.0)
    assert "window_latents" not in out and int(out["time_index"][0]) == 1
    # Window 0, step 1 (global 1) is the window's last step: emits the window
    # and stages the next one.
    out = {"cond_v": [velocity], "uncond_v": [velocity]}
    step1 = ARNodeInputs(tensor_inputs={"latents": lat, "time_index": torch.tensor([1])})
    sub.postprocess("r", info, out, inputs=step1)
    assert "window_latents" in out and int(out["time_index"][0]) == 2
    assert out["latents"][0].shape == (1, ch, 3, 4, 4)


def test_masked_capture_matches_eager_layout() -> None:
    """On a tiny transformer: the all-frames-noisy graph layout with the
    clean/noisy mask reproduces the eager i2v layout's velocity on the noisy
    frames exactly (the clean frame's output is what postprocess zeroes)."""
    from mstar.model.cosmos3.components.packing import build_static_inputs
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
    from mstar.model.cosmos3.tests.test_edge import _init_all, _OverwriteKV, _tiny_edge_config

    cfg = _tiny_edge_config()
    model = Cosmos3OmniTransformer(cfg).eval()
    _init_all(model)
    res = _OverwriteKV()
    for child in model.modules():
        bind = getattr(child, "bind_resources", None)
        if bind is not None:
            bind({"kv": res, "attn": res})
    ids = [3, 5, 7, 11, 13]
    latents = torch.randn(1, cfg.latent_channel, 3, 4, 4)
    shape = tuple(latents.shape)
    eager = build_static_inputs(ids, shape, cfg, 4, 24.0, "cpu", has_image_condition=True)
    graph = build_static_inputs(ids, shape, cfg, 4, 24.0, "cpu", has_image_condition=False)
    n_all = graph["num_vision_tokens"]
    assert eager["num_noisy_vision_tokens"] == n_all - 4  # frame 0 clean
    per_frame = n_all // 3
    mask = torch.cat([torch.zeros(per_frame), torch.ones(n_all - per_frame)])
    with torch.no_grad():
        res.causal = True
        model.prefill_und(eager["input_ids"], eager["text_mrope_ids"], "main")
        res.commit()
        res.causal = False
        t_noisy = torch.full((eager["num_noisy_vision_tokens"],), 600.0)
        cond_e, uncond_e = model.denoise_step_batched_cfg(
            latents, t_noisy, eager["vision_mrope_ids"], eager["vision_mrope_ids"],
            eager["vision_token_shapes"], eager["vision_noisy_frame_indexes"],
            eager["vision_mse_loss_indexes"] - eager["und_len"], "main", res,
        )
        t_all = torch.full((n_all,), 600.0)
        cond_g, uncond_g = model.denoise_step_batched_cfg(
            latents, t_all, graph["vision_mrope_ids"], graph["vision_mrope_ids"],
            graph["vision_token_shapes"], graph["vision_noisy_frame_indexes"],
            graph["vision_mse_loss_indexes"] - graph["und_len"], "main", res,
            noisy_token_mask=mask,
        )
    for e, g in ((cond_e, cond_g), (uncond_e, uncond_g)):  # [1, C, T, H, W]
        assert torch.all(e[:, :, 0] == 0)  # the eager unpatchify leaves the clean frame at zero velocity
        assert torch.allclose(e[:, :, 1:], g[:, :, 1:], atol=1e-5, rtol=1e-4)
        assert not torch.all(g[:, :, 0] == 0)  # the graph predicts it too; postprocess masks it away
