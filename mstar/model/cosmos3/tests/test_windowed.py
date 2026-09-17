"""CPU checks for Cosmos3 windowed autoregressive video (the streaming
rollout walk): request-knob quantization and walk selection, the per-window
state machine in the DiT submodule (chained boundaries, kv commit iterations
and their step declarations, the retention hand-off to the KV pool), batching
rules, the streaming VAE decoder, and the NDJSON video surface. No GPU, no
weights. Ported from #198's serving tests onto the resource-pool engine.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.windowing import WindowSchedule
from mstar.model.cosmos3 import constants as C
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import (
    ATTN,
    CFG_BATCHED_LABEL,
    COND_LABEL,
    KV_CACHE,
    UNCOND_LABEL,
    VIDEO_GEN_AR_LOOP,
    Cosmos3DiTSubmodule,
    Cosmos3VAEDecoderARSubmodule,
)


def _windowed_model():
    return Cosmos3Model(model_path_hf="unused", skip_weight_loading=True, enable_windowed_video=True)


def _dit(cfg=None):
    cfg = cfg or Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).config
    return Cosmos3DiTSubmodule(transformer=None, config=cfg, scheduler=None)


def test_windowed_gen_params_and_walk_selection() -> None:
    """Windowed knobs quantize to latent units, the AR walk is selected at the
    prefill transition, and every rejection fires at request resolution."""
    from mstar.conductor.request_info import CurrentForwardConductorMetadata

    model = _windowed_model()
    p = model._resolve_gen_params(
        {"window_mode": "chained", "num_frames": 189, "size": "832x480"}, [], ["video"],
    )
    # Default overlap = 2 latent units (the V2V pin count); 48 requested units
    # pad to 50 so every window is full-size (stride 6).
    assert p["window_latent_units"] == 8 and p["overlap_latent_units"] == 2
    assert p["total_latent_units"] == 50 and p["num_windows"] == 8
    assert p["num_frames"] == 189
    assert p["flow_shift"] == C.V2V_DEFAULT_FLOW_SHIFT

    md = CurrentForwardConductorMetadata(
        input_modalities=[], output_modalities=["video"],
        graph_walk=C.PREFILL_WALK, is_prefill=True, kwargs=p,
    )
    args = model.get_partition_forward_pass_args("default", md, {})
    assert args.full_metadata.graph_walk == C.VIDEO_GEN_AR_WALK
    assert [e.name for e in args.inputs] == ["latents", "time_index", "cond_latents"]
    done = model.get_partition_forward_pass_args("default", args.full_metadata, {})
    assert done.request_done

    dec = model.get_partition_forward_pass_args(
        C.WINDOW_DECODER_PARTITION,
        CurrentForwardConductorMetadata(
            input_modalities=[], output_modalities=["video"],
            graph_walk=C.VIDEO_DECODE_AR_WALK, is_prefill=False, kwargs=p,
        ),
        {},
    )
    assert dec.full_metadata.graph_walk == C.VIDEO_DECODE_AR_WALK
    assert not dec.request_done and dec.inputs == []
    init = model.get_initial_forward_pass_args(
        C.WINDOW_DECODER_PARTITION, ["text"], ["video"], {}, {"window_mode": "kv", "num_frames": 61},
    )
    assert init.full_metadata.graph_walk == C.VIDEO_DECODE_AR_WALK
    assert init.step_metadata["num_windows"] == 2 and init.inputs == []

    for bad in (
        {"window_mode": "bogus", "num_frames": 189},
        {"window_mode": "chained", "num_frames": 1},
        {"window_mode": "chained"},  # image output below
        {"window_mode": "chained", "num_frames": 189, "context_frames": 61},
        {"window_mode": "kv", "num_frames": 189, "overlap_frames": 8},
        {"window_mode": "kv", "num_frames": 189, "context_frames": -1},
        {"window_mode": "chained", "num_frames": 189, "window_frames": 3},
    ):
        with pytest.raises(ValueError):
            out_mod = ["video"] if bad.get("num_frames", 0) > 1 else ["image"]
            model._resolve_gen_params(bad, [], out_mod)
    with pytest.raises(ValueError):
        model._resolve_gen_params(
            {"window_mode": "chained", "num_frames": 189, "generate_sound": True}, [], ["video"],
        )
    with pytest.raises(ValueError):
        model._resolve_gen_params({"window_mode": "chained", "num_frames": 9999999}, [], ["video"])
    with pytest.raises(ValueError):
        model._resolve_gen_params({"window_mode": "chained", "num_frames": 189}, ["video", "text"], ["video"])
    plain = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    with pytest.raises(ValueError):
        plain._resolve_gen_params({"window_mode": "chained", "num_frames": 189}, [], ["video"])
    # A windowed-enabled deployment serves non-windowed requests unchanged.
    q = model._resolve_gen_params({"num_frames": 189}, [], ["video"])
    assert "window_mode" not in q

    # kv mode: overlap-free windows advancing by the full window, committed
    # context capped at the (quantized) horizon; context_frames=0 retains all.
    kv = model._resolve_gen_params({"window_mode": "kv", "num_frames": 189}, [], ["video"])
    assert kv["window_mode"] == "kv"
    assert kv["overlap_latent_units"] == 0
    assert kv["context_latent_units"] == 16  # 61 px default -> 16 units
    assert kv["total_latent_units"] == 48 and kv["num_windows"] == 6
    keep_all = model._resolve_gen_params(
        {"window_mode": "kv", "num_frames": 189, "context_frames": 0}, [], ["video"],
    )
    assert keep_all["context_latent_units"] == 0


def test_windowed_walks_partitions_and_topology() -> None:
    """Enabling windowed serving adds the AR walk + decoder walk, splits the
    decoder into its own partition fed by the window stream, and builds the
    streaming node; a plain deployment is untouched."""
    from mstar.graph.base import Loop

    model = _windowed_model()
    walks = model.get_graph_walk_graphs()
    assert C.VIDEO_GEN_AR_WALK in walks and C.VIDEO_DECODE_AR_WALK in walks
    loop = walks[C.VIDEO_GEN_AR_WALK].sections[0]
    assert isinstance(loop, Loop)
    assert loop.max_iters == model.config.max_windows * (model.config.max_inference_steps + 1)
    streaming = [e for e in loop.section.outputs if getattr(e, "is_streaming", False)]
    assert [e.name for e in streaming] == ["window_latents"]
    assert streaming[0].target_partition == C.WINDOW_DECODER_PARTITION

    parts = {p.name: p for p in model.get_partitions()}
    assert set(parts) == {"default", C.WINDOW_DECODER_PARTITION}
    assert parts[C.WINDOW_DECODER_PARTITION].graph_walks == {C.VIDEO_DECODE_AR_WALK}
    assert parts[C.WINDOW_DECODER_PARTITION].initial_walk == C.VIDEO_DECODE_AR_WALK
    assert C.VIDEO_DECODE_AR_WALK not in parts["default"].graph_walks
    topo = model.get_partition_topology()
    (conn,) = topo.connections
    assert conn.edge_name == "window_latents" and conn.chunk_policy_factory().next_chunk_size(3) == 1
    assert isinstance(model._create_submodule("vae_decoder_ar", "cpu"), Cosmos3VAEDecoderARSubmodule)

    plain = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    assert C.VIDEO_GEN_AR_WALK not in plain.get_graph_walk_graphs()
    assert [p.name for p in plain.get_partitions()] == ["default"]
    assert plain.get_partition_topology().connections == []


def test_windowed_check_stop_counts_all_windows() -> None:
    """The AR loop stops at num_windows x iterations, not at one scheduler's
    length."""
    sub = _dit()
    st = sub.request_state("r")
    st.add_all(
        ar_schedule=WindowSchedule(48, 8, overlap_units=1),
        ar_total_iters=7 * 5,
        scheduler=SimpleNamespace(timesteps=list(range(5))),
    )
    info = SimpleNamespace(graph_walk=C.VIDEO_GEN_AR_WALK, dynamic_loop_iter_counts={VIDEO_GEN_AR_LOOP: 4})
    assert sub.check_stop("r", info, {}) == set()
    info.dynamic_loop_iter_counts[VIDEO_GEN_AR_LOOP] = 33
    assert sub.check_stop("r", info, {}) == set()
    info.dynamic_loop_iter_counts[VIDEO_GEN_AR_LOOP] = 34
    assert sub.check_stop("r", info, {}) == {VIDEO_GEN_AR_LOOP}


def test_windowed_finish_window_bookkeeping(monkeypatch) -> None:
    """The last step of a window emits the window's latents on the streaming
    edge, stages the next window (fresh noise, overlap tail pinned clean), and
    the final window emits without staging."""
    sub = _dit()
    sub.transformer = SimpleNamespace(proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)))
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: "fresh-sched")
    monkeypatch.setattr(sub, "_window_statics_for", lambda st, plan, dev: ({"u": plan.units}, None))

    st = sub.request_state("r")
    schedule = WindowSchedule(total_units=12, window_units=8, overlap_units=1)
    assert schedule.num_windows == 2
    st.add_all(
        ar_schedule=schedule, ar_steps=5, ar_total_iters=10, ar_iters_per_window=5,
        ar_flow_shift=None, ar_karras=None, ar_size=(64, 64),
        ar_generator=torch.Generator().manual_seed(0),
        cond={"u": 8}, uncond=None, scheduler="w0-sched",
    )
    w0_shape = sub._window_latent_shape(64, 64, 8)
    x0 = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1).expand(w0_shape).contiguous()
    ti = torch.tensor([4])
    out = sub._finish_window(st, x0, ti, window_index=0)
    assert torch.equal(out["window_latents"][0], x0)
    assert int(out["time_index"][0].item()) == 5
    # Next window: 5 latent units (12 total - 8 + 1 overlap), head pinned to
    # the previous tail value (7.0), rest fresh noise.
    nxt = out["latents"][0]
    assert nxt.shape[2] == 5
    assert torch.all(nxt[:, :, 0] == 7.0)
    assert st["scheduler"] == "fresh-sched" and st["cond"] == {"u": 5}
    assert st["vmask"].shape[2] == 5 and float(st["vmask"][0, 0, 0]) == 1.0

    # Final window: emit only, no staging.
    st.add("cond", {"u": 5})
    x1 = torch.zeros(sub._window_latent_shape(64, 64, 5))
    out = sub._finish_window(st, x1, torch.tensor([9]), window_index=1)
    assert torch.equal(out["window_latents"][0], x1)
    assert torch.equal(out["latents"][0], x1)


def test_windowed_kv_prefill_state_and_pacing(monkeypatch) -> None:
    """A kv request runs steps + 1 loop iterations per window (the +1 is the
    commit pass); the prefill records the pacing, the schedule and the
    per-unit token stride, and check_stop fires after the final commit."""
    sub = _dit()
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: "sched")
    md = {
        "window_mode": "kv", "total_latent_units": 12, "window_latent_units": 4,
        "overlap_latent_units": 0, "context_latent_units": 6,
    }
    fwd = SimpleNamespace(request_id="r")
    ni = sub._prepare_windowed_prefill(fwd, md, list(range(7)), list(range(9)), 64, 64, 24.0, 6.0, 4, "cpu")
    assert ni.kwargs["cfg"] and ni.kwargs["seq_lens"] == {COND_LABEL: 7, UNCOND_LABEL: 9}
    st = sub.request_states["r"]
    assert st["ar_kv_mode"] and st["ar_iters_per_window"] == 5
    assert st["ar_total_iters"] == 15
    # 64x64 -> 4x4 latent -> 2x2 patchify -> 4 tokens per latent frame.
    assert st["ar_tokens_per_unit"] == 4
    assert st["ar_schedule"].context_units == 6
    assert sub._window_step(st, 4) == (0, 4, True) and sub._window_step(st, 7) == (1, 2, False)

    info = SimpleNamespace(graph_walk=C.VIDEO_GEN_AR_WALK, dynamic_loop_iter_counts={VIDEO_GEN_AR_LOOP: 13})
    assert sub.check_stop("r", info, {}) == set()
    info.dynamic_loop_iter_counts[VIDEO_GEN_AR_LOOP] = 14
    assert sub.check_stop("r", info, {}) == {VIDEO_GEN_AR_LOOP}


def test_windowed_kv_statics_absolute_positions() -> None:
    """kv window statics carry absolute temporal mRoPE positions: a window
    starting at latent frame 8 positions its tokens exactly where the full
    clip would, the image anchor applies only to the first window, and
    chained statics keep per-window positions from 0."""
    sub = _dit()
    cfg = sub.config
    ids = list(range(7))
    kw = dict(height=64, width=64, units=4, fps=24.0, has_image_condition=True, cond_units=0, device="cpu")
    w0, _ = sub._build_window_statics(ids, None, first_window=True, start_unit=0, **kw)
    w2, _ = sub._build_window_statics(ids, None, first_window=False, start_unit=8, **kw)

    full = sub._build_static(
        ids, 64, 64, 1 + (12 - 1) * cfg.vae.scale_factor_temporal, 24.0,
        has_image_condition=True, device="cpu",
    )
    stride = full["num_vision_tokens"] // 12
    assert torch.equal(w2["vision_mrope_ids"], full["vision_mrope_ids"][:, 8 * stride: 12 * stride])
    # Anchor only on the first window: window 0 keeps latent frame 0 clean,
    # a later window predicts every frame.
    assert w0["num_noisy_vision_tokens"] == 3 * stride
    assert w2["num_noisy_vision_tokens"] == 4 * stride
    # Chained (start_unit 0) restarts each window's positions at frame 0.
    ch, _ = sub._build_window_statics(ids, None, first_window=False, start_unit=0, **kw)
    assert torch.equal(ch["vision_mrope_ids"], full["vision_mrope_ids"][:, : 4 * stride])


class _FakeKV:
    """The pool surface the windowed request touches: retention hand-off."""

    def __init__(self):
        self.policies = []

    def set_retention(self, request_id, policy, label=None):
        self.policies.append((request_id, label, policy.protected_prefix, policy.context_budget))


def test_windowed_kv_commit_iteration_declares_and_commits(monkeypatch) -> None:
    """The commit iteration is prepared as a committing span of exactly the
    window's new units, declared as a paged, non-causal, committing step over
    both guidance branches, hands the pool each branch's retention (prefix +
    horizon) once, runs the transformer's commit pass with the window's
    absolute positions, and stages the next window. Denoise iterations keep
    the plain (non-committing) declaration."""
    sub = _dit()
    commits = []
    sub.transformer = SimpleNamespace(
        proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
        commit_window=lambda latents, positions, label, attn: commits.append(
            (latents.clone(), [p.clone() for p in positions], label, attn)
        ),
    )
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: "fresh")
    monkeypatch.setattr(sub, "get_device", lambda: torch.device("cpu"))  # no parameters to read it off
    kv = _FakeKV()

    stride = 64  # tokens per latent frame (256p tier: pages hold 2 frames)

    def fake_statics(plan):
        n = plan.units * stride
        base = plan.start * stride
        ids = (torch.arange(n) + base).view(1, -1).expand(3, -1).contiguous()
        return (
            {"vision_mrope_ids": ids, "und_len": 7, "num_vision_tokens": n},
            {"vision_mrope_ids": ids + 100000, "und_len": 9, "num_vision_tokens": n},
        )

    monkeypatch.setattr(sub, "_window_statics_for", lambda st, plan, dev: fake_statics(plan))
    schedule = WindowSchedule(total_units=12, window_units=4, context_units=6, overlap_units=0)
    st = sub.request_state("r")
    cond0, uncond0 = fake_statics(schedule.window(0))
    st.add_all(
        ar_schedule=schedule, ar_steps=4, ar_iters_per_window=5, ar_total_iters=15,
        ar_kv_mode=True, ar_rid="r", ar_tokens_per_unit=stride, ar_size=(64, 64),
        ar_generator=torch.Generator().manual_seed(0),
        cond=cond0, uncond=uncond0, scheduler=SimpleNamespace(timesteps=torch.arange(4)),
        latent_shape=sub._window_latent_shape(64, 64, 4), gs=6.0, guidance_interval=None,
    )
    x0 = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1).expand(
        sub._window_latent_shape(64, 64, 4)).contiguous()
    fwd = SimpleNamespace(request_id="r", random_seed=0)

    # Commit iteration (global 4 = window 0's 5th iteration).
    ni = sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, {"latents": [x0], "time_index": [torch.tensor([4])]})
    assert ni.input_seq_len == 4 * stride and ni.resource_step_info.commit
    step = sub.declare_step(C.VIDEO_GEN_AR_WALK, ["r"], [ni])
    assert step.steps[KV_CACHE].commit
    assert step.steps[KV_CACHE].combined_labels == {(COND_LABEL, UNCOND_LABEL): CFG_BATCHED_LABEL}
    assert ATTN in step.steps and not step.steps[ATTN].causal
    assert [(s.label, s.span) for s in step.segments] == [(COND_LABEL, 4 * stride), (UNCOND_LABEL, 4 * stride)]

    # A denoise iteration declares the usual non-committing step, and the
    # retention is not handed over twice.
    ni2 = sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, {"latents": [x0], "time_index": [torch.tensor([2])]})
    assert ni2.input_seq_len == 4 * stride and not ni2.resource_step_info.commit
    assert not sub.declare_step(C.VIDEO_GEN_AR_WALK, ["r"], [ni2]).steps[KV_CACHE].commit
    # Past the last iteration the loop's extra dispatch is vetoed.
    assert sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, {"latents": [x0], "time_index": [torch.tensor([15])]}) is None

    # Window 0 commit forward, driven through forward() so the step's
    # resources reach it: the retention is handed to the pool once (prefix +
    # horizon per branch), the full window appended under the combined label
    # through the paged attention, next window staged with absolute positions
    # and a fresh scheduler.
    ei = SimpleNamespace(
        request_ids=["r"], per_request_states=None, resources={KV_CACHE: kv, ATTN: "paged"}, step={ATTN: None},
    )
    out = sub.forward(C.VIDEO_GEN_AR_WALK, ei, latents=x0, time_index=torch.tensor([4]))
    assert kv.policies == [("r", COND_LABEL, 7, 6 * stride), ("r", UNCOND_LABEL, 9, 6 * stride)]
    assert torch.equal(out["window_latents"][0], x0)
    latents_c, positions_c, label, attn = commits[-1]
    assert label == CFG_BATCHED_LABEL and attn == "paged"
    assert torch.equal(latents_c, x0)
    assert torch.equal(positions_c[0], cond0["vision_mrope_ids"])
    assert torch.equal(positions_c[1], uncond0["vision_mrope_ids"])
    assert st["scheduler"] == "fresh"
    assert int(st["cond"]["vision_mrope_ids"][0, 0]) == 4 * stride
    assert out["latents"][0].shape == sub._window_latent_shape(64, 64, 4)

    # Sequential guidance commits per label; the retention is not re-bound.
    sub.batched_cfg = False
    sub._commit_window("paged", st, x0, torch.tensor([9]), window_index=1)
    assert [c[2] for c in commits[-2:]] == [COND_LABEL, UNCOND_LABEL]
    assert len(kv.policies) == 2
    # Final window: emit only.
    out = sub._commit_window("paged", st, x0, torch.tensor([14]), window_index=2)
    assert torch.equal(out["latents"][0], x0) and torch.equal(out["window_latents"][0], x0)


def test_windowed_can_batch_and_batched_boundary(monkeypatch) -> None:
    """Windowed denoise steps batch across requests (loop counters mapped to
    within-window steps per request); any request at a kv commit iteration
    drops the batch to the sequential path; and a chained window boundary
    inside the batched forward emits the window and stages the next one like
    the single-request path."""
    sub = _dit()
    schedule = WindowSchedule(total_units=14, window_units=8, overlap_units=2)
    latent_shape = sub._window_latent_shape(64, 64, 8)
    sched = SimpleNamespace(
        timesteps=torch.arange(4, 0, -1),
        step=lambda v, t, lat, return_dict=False: (lat * 0.5,),
    )

    def add_windowed(rid):
        st = sub.request_state(rid)
        n = 8 * 4
        ids = torch.arange(n).view(1, -1).expand(3, -1).contiguous()
        static = {
            "num_vision_tokens": n, "num_noisy_vision_tokens": n,
            "vision_mrope_ids": ids, "und_len": 7,
            "vision_token_shapes": [(8, 2, 2)],
            "vision_noisy_frame_indexes": [torch.arange(8)],
            "mse_gen_indexes": torch.arange(n),
        }
        st.add_all(
            ar_schedule=schedule, ar_steps=4, ar_iters_per_window=4,
            ar_kv_mode=False, ar_size=(64, 64), gs=6.0,
            ar_generator=torch.Generator().manual_seed(0),
            cond=static, uncond=dict(static), scheduler=sched,
            latent_shape=latent_shape,
        )
        return st

    add_windowed("a")
    add_windowed("b")
    batch = SimpleNamespace(graph_walk=C.VIDEO_GEN_AR_WALK, request_ids=["a", "b"])
    inp = lambda t: SimpleNamespace(tensor_inputs={"time_index": torch.tensor([t])})  # noqa: E731
    assert sub.can_batch(batch, [inp(1), inp(1)])
    assert sub.can_batch(batch, [inp(3), inp(5)])  # different windows, both denoise

    # A kv request at its commit iteration vetoes the batch.
    st_kv = add_windowed("c")
    st_kv.add("ar_kv_mode", True)
    st_kv.add("ar_iters_per_window", 5)
    batch_kv = SimpleNamespace(graph_walk=C.VIDEO_GEN_AR_WALK, request_ids=["a", "c"])
    assert sub.can_batch(batch_kv, [inp(1), inp(1)])
    assert not sub.can_batch(batch_kv, [inp(1), inp(4)])  # local 4 == steps

    # A prefill batch carries no time_index; with a windowed request in it the
    # batch must fall to the sequential path (not raise), while plain-only
    # prefill batches still batch.
    noti = SimpleNamespace(tensor_inputs={})
    batch_pre = SimpleNamespace(graph_walk=C.PREFILL_WALK, request_ids=["a", "p"])
    sub.request_state("p").add_all(cond={}, uncond={})
    assert not sub.can_batch(batch_pre, [noti, noti])
    batch_pp = SimpleNamespace(graph_walk=C.PREFILL_WALK, request_ids=["p", "q"])
    sub.request_state("q").add_all(cond={}, uncond={})
    assert sub.can_batch(batch_pp, [noti, noti])

    # Batched forward: request "a" at its window-0 boundary (local 3),
    # request "b" mid-window. The boundary request emits + stages.
    sub.transformer = SimpleNamespace(
        proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
        denoise_step_batched=lambda reqs, label, attn: [
            (torch.zeros(latent_shape), torch.zeros(latent_shape)) for _ in reqs
        ],
    )
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: sched)
    monkeypatch.setattr(
        sub, "_window_statics_for",
        lambda st, plan, dev: (dict(sub.request_states["a"]["cond"]), None),
    )
    ei = SimpleNamespace(
        request_ids=["a", "b"], per_request_states=None, per_request_info={},
        resources={ATTN: "paged"}, step=None,
    )
    lat = {r: torch.full(latent_shape, 2.0) for r in ("a", "b")}
    ti = {"a": torch.tensor([3]), "b": torch.tensor([1])}
    out = sub.forward_batched(C.VIDEO_GEN_AR_WALK, ei, latents=lat, time_index=ti)
    assert "window_latents" in out["a"] and torch.equal(out["a"]["window_latents"][0], lat["a"] * 0.5)
    assert out["a"]["latents"][0].shape == latent_shape  # staged next window
    assert "window_latents" not in out["b"]
    assert torch.equal(out["b"]["latents"][0], lat["b"] * 0.5)


def _tracing_vae():
    """Stub VAE for the AR-decoder tests: decodes latent value v at latent
    index i to pixel frames of value v — frame 0 from latent 0, then 4 frames
    per later latent, the Wan VAE's temporal contract — so every output frame
    identifies its source latent."""

    class _TracingVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])

        def decode(self, z):
            vals = z[0, 0, :, 0, 0]
            frames = [vals[0].expand(1)]
            for i in range(1, z.shape[2]):
                frames.append(vals[i].expand(4))
            t = torch.cat(frames)
            sample = t.view(1, 1, -1, 1, 1).expand(1, 3, t.numel(), 4, 4)
            # forward maps [-1, 1] -> uint8; keep values identity-recoverable.
            return SimpleNamespace(sample=sample / 127.5 - 1.0)

    return _TracingVAE()


def _ar_decoder(monkeypatch):
    monkeypatch.setenv("COSMOS3_COMPILE_VAE", "0")
    cfg = Cosmos3Config()
    cfg.windowed_decode_context_latents = 3
    sub = Cosmos3VAEDecoderARSubmodule(_tracing_vae(), cfg)
    sub._decode_dtype_cached = torch.float32
    return sub


def test_windowed_decoder_assembles_stream(monkeypatch) -> None:
    """The AR decoder's context-re-decode + trim reproduces the whole-clip
    decode of the same latent stream, chunk counts drive completion, and the
    stream's empty terminal flush is skipped."""
    sub = _ar_decoder(monkeypatch)
    # Latent stream of 12 units, value = absolute unit index; windows of 8
    # units with 1-unit overlap: [0..8) then [7..12).
    stream = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1).expand(1, 16, 12, 4, 4).contiguous()
    md = {"num_windows": 2, "overlap_latent_units": 1, "num_frames": 45}
    engine_inputs = SimpleNamespace(request_ids=["r"], per_request_info={"r": SimpleNamespace(step_metadata=md)})

    flush = sub.prepare_inputs(C.VIDEO_DECODE_AR_WALK, None, {"window_latents": []})
    assert flush.tensor_inputs["latents"].numel() == 0
    assert sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, flush.tensor_inputs["latents"]) == {}
    first = sub.prepare_inputs(C.VIDEO_DECODE_AR_WALK, None, {"window_latents": [stream[:, :, 0:8]]})
    assert sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, first.tensor_inputs["latents"]) == {}
    out = sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, stream[:, :, 7:12])
    video = out["video_output"][0]

    expected = sub._decode_pixels(stream)
    assert video.shape == expected.shape  # 1 + 11*4 = 45 frames
    assert torch.equal(video, expected)


def test_windowed_decoder_streams_chunks(monkeypatch) -> None:
    """stream_video emits each window's pixels as its own chunk with nothing
    accumulated, the tail chunk is capped at the requested frame count, and
    the chunks concatenate to the non-streaming assembly."""
    sub = _ar_decoder(monkeypatch)
    stream = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1).expand(1, 16, 12, 4, 4).contiguous()
    # 43 requested frames inside the 45-frame padded schedule: the trim lands
    # in the tail chunk.
    md = {"num_windows": 2, "overlap_latent_units": 1, "num_frames": 43, "stream_video": True}
    engine_inputs = SimpleNamespace(request_ids=["r"], per_request_info={"r": SimpleNamespace(step_metadata=md)})

    out0 = sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, stream[:, :, 0:8])
    out1 = sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, stream[:, :, 7:12])
    c0 = out0["video_output"][0]
    c1 = out1["video_output"][0]
    assert c0.shape[2] == 29 and c1.shape[2] == 14
    expected = sub._decode_pixels(stream)
    assert torch.equal(torch.cat([c0, c1], dim=2), expected[:, :, :43])
    assert sub.request_state("r")["ar_pixels"] == []
    # The stream's empty terminal flush stays a no-op.
    assert sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, torch.empty(0)) == {}


def test_windowed_stream_video_gen_params() -> None:
    """stream_video resolves only alongside a windowed mode."""
    model = _windowed_model()
    p = model._resolve_gen_params(
        {"window_mode": "chained", "num_frames": 189, "stream_video": True}, [], ["video"],
    )
    assert p["stream_video"] is True
    q = model._resolve_gen_params({"window_mode": "kv", "num_frames": 189}, [], ["video"])
    assert q["stream_video"] is False
    with pytest.raises(ValueError):
        model._resolve_gen_params({"num_frames": 189, "stream_video": True}, [], ["video"])


def test_video_streaming_ndjson_lines() -> None:
    """stream_video turns the video handler into an NDJSON generator: indexed
    video lines closed by a done line, an in-band error line terminal instead
    on failure, and the non-streaming path collecting as before."""
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import VideoGenerationRequest
    from mstar.api_server.openai.serving_videos import create_videos
    from mstar.api_server.request_types import ResultChunk

    class _Api:
        upload_dir = "/tmp"

        def __init__(self, chunks):
            self._chunks = chunks
            self.submits = []

        def submit_request(self, **kw):
            self.submits.append(kw)
            return kw["request_id"]

        async def iter_result_chunks(self, request_id):  # noqa: ARG002
            for c in self._chunks:
                yield c

        async def collect_results(self, request_id, raw_request=None):  # noqa: ARG002
            return list(self._chunks)

    adapter = get_adapter("cosmos3")
    streaming_req = VideoGenerationRequest(prompt="x", num_frames=57, window_mode="chained", stream_video=True)

    async def _lines(api):
        gen = await create_videos(api, "cosmos3", adapter, streaming_req)
        return [json.loads(line) async for line in gen]

    api = _Api([
        ResultChunk(request_id="r", modality="video", data=b"w0"),
        ResultChunk(request_id="r", modality="video", data=b"w1"),
    ])
    lines = asyncio.run(_lines(api))
    assert api.submits[0]["streaming"] is True
    assert api.submits[0]["model_kwargs"]["stream_video"] is True
    assert [ln["modality"] for ln in lines] == ["video", "video", "done"]
    assert [ln["metadata"]["index"] for ln in lines[:2]] == [0, 1]
    assert base64.b64decode(lines[0]["data"]) == b"w0"
    assert base64.b64decode(lines[1]["data"]) == b"w1"
    assert lines[2]["metadata"]["chunks"] == 2

    api = _Api([
        ResultChunk(request_id="r", modality="video", data=b"w0"),
        ResultChunk(request_id="r", modality="error", data=b"boom", metadata={"status": 500}),
    ])
    lines = asyncio.run(_lines(api))
    assert [ln["modality"] for ln in lines] == ["video", "error"]
    assert base64.b64decode(lines[1]["data"]) == b"boom"

    api = _Api([ResultChunk(request_id="r", modality="video", data=b"v")])
    out = asyncio.run(create_videos(api, "cosmos3", adapter, VideoGenerationRequest(prompt="x", num_frames=57)))
    assert api.submits[0]["streaming"] is False
    assert "stream_video" not in api.submits[0]["model_kwargs"]
    assert base64.b64decode(out["data"][0]["b64_json"]) == b"v"
