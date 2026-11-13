"""CPU checks for Cosmos3 windowed autoregressive video (the streaming
rollout walk): request-knob quantization and walk selection, the per-window
state machine in the DiT submodule (chained boundaries, kv commit iterations
and their step declarations, sessions), batching rules, the streaming VAE
decoder, and the NDJSON video surface. No GPU, no weights. Ported from #198's
serving tests onto the resource-pool engine.
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
from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.sessions import SessionStore
from mstar.model.cosmos3.submodules import (
    COND_LABEL,
    DECODE_CACHE,
    DECODE_FRAMES,
    DECODE_SENT,
    DECODE_TAIL,
    KV_CACHE,
    SESSION_TAIL,
    UNCOND_LABEL,
    VIDEO_DECODE_AR_LOOP,
    VIDEO_GEN_AR_LOOP,
    WINDOW_COND,
    Cosmos3DiTSubmodule,
    Cosmos3VAEDecoderARSubmodule,
    GenStepInfo,
)
from mstar.model.cosmos3.tests.stubs import StubSampler, bind_progress
from mstar.model.cosmos3.tests.test_rotary_table import stand_in_rotary


def _windowed_model():
    return Cosmos3Model(model_path_hf="unused", skip_weight_loading=True, enable_windowed_video=True)


def _stand_in(sub, **methods):
    """A transformer stand-in with the dtype and the rotary table the statics
    read, plus whatever ``methods`` a test records."""
    return SimpleNamespace(
        proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
        _rotary=stand_in_rotary(sub.config), **methods,
    )


def _dit(cfg=None):
    cfg = cfg or Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).config
    sub = Cosmos3DiTSubmodule(transformer=None, config=cfg, scheduler=None)
    # No VAE without weights: the boundary bookkeeping tests pin the raw tail
    # instead of a re-encoded head (test_chained_window_head_layout pins that).
    sub._window_head = lambda w, units: w[:, :, -units:]
    return sub


def test_windowed_gen_params_and_walk_selection() -> None:
    """Windowed knobs quantize to latent units, the plan runs the AR walk after
    the prefill, and every rejection fires at request resolution."""
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

    plan = model.walk_plan("default", [], ["video"], {}, {"window_mode": "chained", "num_frames": 189})
    assert [step.walk for step in plan] == [C.PREFILL_WALK, C.VIDEO_GEN_AR_WALK]
    (dec,) = model.walk_plan(
        C.WINDOW_DECODER_PARTITION, ["text"], ["video"], {}, {"window_mode": "kv", "num_frames": 61},
    )
    assert dec.walk == C.VIDEO_DECODE_AR_WALK and dec.step_metadata["num_windows"] == 2

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
    assert C.VIDEO_DECODE_AR_WALK not in parts["default"].graph_walks
    topo = model.get_partition_topology()
    (conn,) = topo.connections
    assert conn.edge_name == "window_latents" and conn.chunk_policy_factory().next_chunk_size(3) == 1
    assert isinstance(model._create_submodule("vae_decoder_ar", "cpu"), Cosmos3VAEDecoderARSubmodule)

    plain = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    assert C.VIDEO_GEN_AR_WALK not in plain.get_graph_walk_graphs()
    assert [p.name for p in plain.get_partitions()] == ["default"]
    assert plain.get_partition_topology().connections == []


def test_windowed_end_window_bookkeeping(monkeypatch) -> None:
    """The last step of a window emits the window's latents on the streaming
    edge and on ``latents``, with the head built from the finished window as
    the next window's pinned frames; that window starts from fresh noise
    behind the head, and the final window pins nothing."""
    sub = _dit()
    sub.transformer = _stand_in(sub)
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: StubSampler(()))
    heads = []
    sub._window_head = lambda w, units: heads.append((w, units)) or torch.full_like(w[:, :, -units:], -3.0)
    md = {
        "window_mode": "chained", "total_latent_units": 12, "window_latent_units": 8, "overlap_latent_units": 1,
        "height": 64, "width": 64, "num_inference_steps": 5, "guidance_scale": 1.0, C.PROMPT_LENGTHS: (7, 9),
    }
    cpu = torch.device("cpu")
    st = sub._settings(md, 4, {}, cpu)
    assert st["ar_schedule"].num_windows == 2
    x0 = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1).expand(st["latent_shape"]).contiguous()
    out = sub._end_window(st, x0, {})
    assert torch.equal(out["window_latents"][0], x0) and torch.equal(out["latents"][0], x0)
    # Next window: 5 latent units (12 total - 8 + 1 overlap); its head is the
    # one built from the finished window.
    pinned = out[WINDOW_COND][0]
    assert pinned.shape[2] == 5
    assert len(heads) == 1 and heads[0][0] is x0 and heads[0][1] == 1
    assert torch.all(pinned[:, :, 0] == -3.0) and out[SESSION_TAIL][0].numel() == 0

    nxt_st = sub._settings(md, 5, {WINDOW_COND: [pinned], SESSION_TAIL: [torch.zeros(0)]}, cpu)
    nxt = sub._window_start(nxt_st, torch.Generator().manual_seed(0), 1, cpu)
    assert nxt.shape[2] == 5
    assert torch.all(nxt[:, :, 0] == -3.0) and not torch.any(nxt[:, :, 1:] == -3.0)
    assert nxt_st["vmask"].shape[2] == 5 and float(nxt_st["vmask"][0, 0, 0]) == 1.0

    # Final window: emit only, nothing pinned.
    x1 = torch.zeros(nxt_st["latent_shape"])
    out = sub._end_window(nxt_st, x1, {})
    assert torch.equal(out["window_latents"][0], x1) and torch.equal(out["latents"][0], x1)
    assert out[WINDOW_COND][0].numel() == 0 and len(heads) == 1


class _CausalVAE(torch.nn.Module):
    """Stand-in for the Wan VAE's temporal layout (factor 4, latent stats 0/1):
    a clip's first latent decodes to one frame and every later one to the
    four after it. Each decoded frame carries its latent's channel-0 value;
    encode averages a latent's frames and adds 100 to a clip's first latent
    (the single-frame latent a window's head position expects). Records the
    latents it decodes and the frames it encodes."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(latents_mean=[0.0] * channels, latents_std=[1.0] * channels)
        self.decoded: list[torch.Tensor] = []
        self.encoded: list[torch.Tensor] = []

    def decode(self, z):
        self.decoded.append(z.clone())
        later = [z[:, :1, t:t + 1].expand(-1, -1, 4, -1, -1) for t in range(1, z.shape[2])]
        return SimpleNamespace(sample=torch.cat([z[:, :1, :1], *later], 2).expand(-1, 3, -1, -1, -1))

    def encode(self, x):
        self.encoded.append(x.clone())
        c = len(self.config.latents_mean)
        later = [x[:, :1, 1 + 4 * i:5 + 4 * i].mean(2, keepdim=True) for i in range((x.shape[2] - 1) // 4)]
        mode = torch.cat([x[:, :1, :1] + 100, *later], 2).expand(-1, c, -1, -1, -1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mode))


@pytest.mark.parametrize("ctx", [8, 1])
def test_chained_window_head_layout(ctx, monkeypatch) -> None:
    """A chained window's head is the finished window's last 1 + 4(u-1)
    frames encoded as a clip of their own, not its tail latents: the decode
    sees the tail behind ``chained_head_context_latents`` latents, the encode
    sees those frames quantized as served (``_to_uint8``), and the head's
    first latent is the encoder's clip-start latent."""
    import mstar.model.cosmos3.submodules as cosmos3_submodules
    from mstar.model.cosmos3.submodules import ChainedWindowHead, Cosmos3VAEDecoderSubmodule

    monkeypatch.setattr(cosmos3_submodules, "wan_vae_decode_dtype", lambda: torch.float32)
    monkeypatch.setenv("COSMOS3_COMPILE_VAE", "0")
    cfg = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).config
    cfg.chained_head_context_latents = ctx
    vae = _CausalVAE(cfg.latent_channel)
    # Latent l holds (100 + 10 l) / 127.5 - 1, inside [-1, 1].
    values = (100 + 10 * torch.arange(8, dtype=torch.float32)) / 127.5 - 1
    window = values.view(1, 1, 8, 1, 1).expand(1, cfg.latent_channel, 8, 4, 4).contiguous()
    head = ChainedWindowHead(vae, vae, cfg)(window, 2)

    assert torch.equal(vae.decoded[0], window[:, :, -min(8, 2 + ctx):])
    frames = vae.encoded[0]
    assert frames.shape[2] == 5
    # The last frame of latent 6, then latent 7's four frames, at 8 bits.
    served = Cosmos3VAEDecoderSubmodule._to_uint8(values).float() / 127.5 - 1
    assert torch.allclose(frames[0, 0, :, 0, 0], torch.stack([served[6]] + [served[7]] * 4), atol=1e-6)
    assert head.shape == (1, cfg.latent_channel, 2, 4, 4)
    assert torch.allclose(head[:, :, 0], served[6] + 100) and torch.allclose(head[:, :, 1], served[7])
    # The raw tail is what the window would have pinned before.
    assert not torch.allclose(head, window[:, :, -2:])


def test_chained_window_head_encodes_in_the_encode_vaes_dtype(monkeypatch) -> None:
    """The head encodes in the encode VAE's own dtype (bf16 as built) and never
    casts it: the instance is the encoder node's when the two share a process."""
    import mstar.model.cosmos3.submodules as cosmos3_submodules
    from mstar.model.cosmos3.submodules import ChainedWindowHead

    monkeypatch.setattr(cosmos3_submodules, "wan_vae_decode_dtype", lambda: torch.float32)
    monkeypatch.setenv("COSMOS3_COMPILE_VAE", "0")
    cfg = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).config
    encode_vae = _CausalVAE(cfg.latent_channel).to(torch.bfloat16)
    window = torch.zeros(1, cfg.latent_channel, 8, 4, 4)
    head = ChainedWindowHead(_CausalVAE(cfg.latent_channel), encode_vae, cfg)(window, 2)
    assert encode_vae.encoded[0].dtype == torch.bfloat16
    assert encode_vae.weight.dtype == torch.bfloat16
    assert head.dtype == window.dtype


def test_windowed_kv_settings_and_pacing(monkeypatch) -> None:
    """A kv request runs steps + 1 loop iterations per window (the +1 is the
    commit pass); its settings carry the pacing, the schedule and the
    per-unit token stride, and its prefill runs both branches alone."""
    sub = _dit()
    sub.transformer = _stand_in(sub)
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: StubSampler(()))
    md = {
        "window_mode": "kv", "total_latent_units": 12, "window_latent_units": 4,
        "overlap_latent_units": 0, "context_latent_units": 6, "height": 64, "width": 64,
        "num_inference_steps": 4, "guidance_scale": 6.0, C.PROMPT_LENGTHS: (7, 9),
    }
    fwd = SimpleNamespace(request_id="r", rid_handle="r", step_metadata=md)
    ni = sub._prepare_prefill(fwd, {"text_inputs": [torch.arange(7), torch.arange(9)]}, torch.device("cpu"))
    assert ni.kwargs["cfg"] and ni.kwargs["windowed"]
    assert ni.kwargs["seq_lens"] == {COND_LABEL: 7, UNCOND_LABEL: 9}
    st = sub._settings(md, 0, {}, torch.device("cpu"))
    assert st["ar_kv_mode"] and st["ar_iters_per_window"] == 5
    # 64x64 -> 4x4 latent -> 2x2 patchify -> 4 tokens per latent frame.
    assert st["ar_tokens_per_unit"] == 4
    assert st["ar_schedule"].context_units == 6 and st["ar_schedule"].num_windows == 3
    assert sub._window_step(st, 4) == (0, 4, True) and sub._window_step(st, 7) == (1, 2, False)


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


def test_windowed_kv_commit_iteration_declares_and_commits(monkeypatch) -> None:
    """The commit iteration is prepared as a span of exactly the window's new
    units, declared as an add that keeps over both guidance branches, and runs
    the transformer's commit pass with the window's absolute rotary rows.
    Denoise iterations keep nothing. The window the request's layout keeps
    bounds what stays (test_model_kv_layouts)."""
    sub = _dit()
    commits = []
    sub.transformer = _stand_in(
        sub, commit_window=lambda latents, rotary: commits.append(
            (latents.clone(), [(c.clone(), s.clone()) for c, s in rotary])
        ),
    )
    sub._solver_state = ()
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: StubSampler(torch.arange(4)))
    monkeypatch.setattr(sub, "get_device", lambda: torch.device("cpu"))  # no parameters to read it off
    stride = 64  # tokens per latent frame (256p tier: pages hold 2 frames)

    def fake_statics(cond_ids, uncond_ids, height, width, units, fps, has_image_condition, cond_units, device,
                     first_window=True, start_unit=0):
        # Row i of a window's table holds its absolute token index.
        n = units * stride
        rows = (torch.arange(n, dtype=torch.float32) + start_unit * stride).view(-1, 1)
        return (
            {"rotary": (rows, -rows), "num_vision_tokens": n, "num_noisy_vision_tokens": n},
            {"rotary": (rows + 100000, -rows), "num_vision_tokens": n, "num_noisy_vision_tokens": n},
        )

    monkeypatch.setattr(sub, "_window_statics", fake_statics)
    md = {
        "window_mode": "kv", "total_latent_units": 12, "window_latent_units": 4, "overlap_latent_units": 0,
        "context_latent_units": 6, "height": 64, "width": 64, "num_inference_steps": 4, "guidance_scale": 6.0,
        C.PROMPT_LENGTHS: (7, 9),
    }
    x0 = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1).expand(
        sub._window_latent_shape(64, 64, 4)).contiguous()
    counts = {VIDEO_GEN_AR_LOOP: 4}
    fwd = SimpleNamespace(
        request_id="r", rid_handle="r", random_seed=0, step_metadata=md, dynamic_loop_iter_counts=counts,
    )
    edges = {"latents": [x0], WINDOW_COND: [torch.zeros(0)], SESSION_TAIL: [torch.zeros(0)]}

    # Commit iteration (global 4 = window 0's 5th iteration).
    ni = sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, edges)
    assert ni.input_seq_len == 4 * stride and ni.resource_step_info.commit
    step = sub.declare_step(C.VIDEO_GEN_AR_WALK, ["r"], [ni])
    assert step.state[KV_CACHE].keep, "the next window would read no context"
    assert [(s.label, s.span) for s in step.segments] == [(COND_LABEL, 4 * stride), (UNCOND_LABEL, 4 * stride)]

    # A denoise iteration keeps nothing.
    counts[VIDEO_GEN_AR_LOOP] = 2
    ni2 = sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, edges)
    assert ni2.input_seq_len == 4 * stride and not ni2.resource_step_info.commit
    assert not sub.declare_step(C.VIDEO_GEN_AR_WALK, ["r"], [ni2]).state[KV_CACHE].keep

    # Window 0's commit forward: the whole window under both branches, at their absolute rotary rows.
    counts[VIDEO_GEN_AR_LOOP] = 4
    ei = SimpleNamespace(request_ids=["r"], per_request_info={"r": fwd})
    out = sub.forward(C.VIDEO_GEN_AR_WALK, ei, **sub.preprocess(C.VIDEO_GEN_AR_WALK, ei, [ni]))
    assert torch.equal(out["window_latents"][0], x0) and torch.equal(out["latents"][0], x0)
    latents_c, rotary_c = commits[-1]
    assert torch.equal(latents_c, x0)
    for got, static in zip(rotary_c, fake_statics(None, None, 64, 64, 4, 24.0, False, 0, None), strict=True):
        assert torch.equal(got[0], static["rotary"][0]) and torch.equal(got[1], static["rotary"][1])
    # The next window's statics carry its absolute positions.
    nxt = sub._settings(md, 5, edges, torch.device("cpu"))
    assert int(nxt["cond"]["rotary"][0][0, 0]) == 4 * stride


def test_windowed_can_batch_and_batched_boundary() -> None:
    """Windowed denoise steps batch across requests (loop counters mapped to
    within-window steps per request); any request at a kv commit iteration
    drops the batch to the sequential path; and a chained window boundary
    inside the batched forward ends the window like the single-request path."""
    sub = _dit()
    batch = SimpleNamespace(graph_walk=C.VIDEO_GEN_AR_WALK, request_ids=["a", "b"])

    def inp(commit=False, tokens=32):
        info = GenStepInfo(cfg=True, cfg_active=True, capture_key=None, commit=commit, vision_tokens=tokens)
        return SimpleNamespace(resource_step_info=info)

    assert sub.can_batch(batch, [inp(), inp()])  # at any two steps, in any two windows

    # With a cap set (an H100 deployment's 16384), a step whose requests total
    # more vision tokens than it runs request by request, even when each is
    # under it. Without one (the default) they batch.
    cap = 16384
    assert sub.config.max_batched_vision_tokens is None
    assert sub.can_batch(batch, [inp(tokens=cap // 2 + 1), inp(tokens=cap // 2 + 1)])
    sub.config.max_batched_vision_tokens = cap
    assert sub.can_batch(batch, [inp(tokens=cap // 4), inp(tokens=cap // 4)])
    assert not sub.can_batch(batch, [inp(tokens=cap // 2 + 1), inp(tokens=cap // 2 + 1)])
    sub.config.max_batched_vision_tokens = None

    # A kv request at its commit iteration vetoes the batch.
    assert not sub.can_batch(batch, [inp(), inp(commit=True)])

    # With a windowed request in it a prefill batch must fall to the sequential
    # path (not raise), while plain-only prefill batches still batch.
    prefill = SimpleNamespace(graph_walk=C.PREFILL_WALK, request_ids=["a", "p"])
    assert not sub.can_batch(prefill, [SimpleNamespace(kwargs={"cfg": True, "windowed": w}) for w in (True, False)])
    assert sub.can_batch(prefill, [SimpleNamespace(kwargs={"cfg": True, "windowed": False})] * 2)

    # Batched forward: request "a" at its window-0 boundary (local 3),
    # request "b" mid-window. The boundary request ends its window.
    schedule = WindowSchedule(total_units=14, window_units=8, overlap_units=2)
    latent_shape = sub._window_latent_shape(64, 64, 8)
    sched = StubSampler(torch.arange(4, 0, -1), lambda lat, v: lat * 0.5)
    sub._solver_state = ()
    n = 8 * 4
    rows = torch.arange(n, dtype=torch.float32).view(-1, 1)
    static = {
        "num_vision_tokens": n, "num_noisy_vision_tokens": n, "rotary": (rows, -rows),
        "vision_token_shapes": [(8, 2, 2)], "vision_noisy_frame_indexes": [torch.arange(8)],
        "mse_gen_indexes": torch.arange(n),
    }
    settings = {
        rid: {
            "ar_schedule": schedule, "ar_steps": 4, "ar_iters_per_window": 4, "ar_kv_mode": False, "gs": 6.0,
            "guidance_interval": None, "cond": dict(static), "uncond": dict(static, rotary=(rows + 1, -rows)),
            "scheduler": sched, "latent_shape": latent_shape, "window_index": 0, "cond_units": 0,
            "session_id": None, "session_tail": None,
        }
        for rid in ("a", "b")
    }
    batched = []
    sub.transformer = SimpleNamespace(
        proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
        denoise_step_batched=lambda reqs: batched.append(reqs) or [
            (torch.zeros(latent_shape), torch.zeros(latent_shape)) for _ in reqs
        ],
        denoise_step_row=Cosmos3OmniTransformer.denoise_step_row,
    )
    ei = SimpleNamespace(
        request_ids=["a", "b"],
        per_request_info={
            r: SimpleNamespace(dynamic_loop_iter_counts={VIDEO_GEN_AR_LOOP: k}) for r, k in (("a", 3), ("b", 1))
        },
    )
    lat = {r: torch.full(latent_shape, 2.0) for r in ("a", "b")}
    out = sub.forward_batched(C.VIDEO_GEN_AR_WALK, ei, latents=lat, settings=settings)
    # Each request's dict carries its statics' tables per branch.
    for req, rid in zip(batched[-1], ("a", "b"), strict=True):
        assert req["rotary_cond"] is settings[rid]["cond"]["rotary"]
        assert req["rotary_uncond"] is settings[rid]["uncond"]["rotary"]
    assert torch.equal(out["a"]["window_latents"][0], lat["a"] * 0.5)
    assert out["a"][WINDOW_COND][0].shape[2] == schedule.window(1).units, "the next window pins the boundary's head"
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
    """The AR decoder on its re-decode path, default layout (the tracing stub
    has no causal decoder; test_windowed_decode.py covers the warm path)."""
    monkeypatch.setenv("COSMOS3_COMPILE_VAE", "0")
    cfg = Cosmos3Config()
    cfg.windowed_decode_context_latents = 3
    cfg.windowed_decode_warm_cache = False
    cfg.vae_decode_channels_last = False
    sub = Cosmos3VAEDecoderARSubmodule(_tracing_vae(), cfg)
    sub._decode_dtype_cached = torch.float32
    return sub


def _loop_edges(outputs: dict) -> dict:
    return {name: outputs[name] for name in (DECODE_TAIL, DECODE_SENT, DECODE_FRAMES, DECODE_CACHE) if name in outputs}


def _decode(sub, engine_inputs, *windows) -> list[dict]:
    """The decoder's outputs for each of a request's windows, its loop-back edges carried from one to the next."""
    outs, edges = [], {}
    for i, latents in enumerate(windows):
        outs.append(sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, latents, window=i, edges=edges))
        edges = _loop_edges(outs[-1])
    return outs


def test_windowed_decoder_assembles_stream(monkeypatch) -> None:
    """The AR decoder's context-re-decode + trim reproduces the whole-clip
    decode of the same latent stream, chunk counts drive completion, and the
    stream's empty terminal flush is declined."""
    sub = _ar_decoder(monkeypatch)
    # Latent stream of 12 units, value = absolute unit index; windows of 8
    # units with 1-unit overlap: [0..8) then [7..12).
    stream = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1).expand(1, 16, 12, 4, 4).contiguous()
    md = {"num_windows": 2, "overlap_latent_units": 1, "num_frames": 45}
    engine_inputs = SimpleNamespace(request_ids=["r"], per_request_info={"r": SimpleNamespace(step_metadata=md)})

    def fwd(window):
        return SimpleNamespace(dynamic_loop_iter_counts={VIDEO_DECODE_AR_LOOP: window})

    assert sub.prepare_inputs(
        C.VIDEO_DECODE_AR_WALK, fwd(2), {"window_latents": []}, is_final_stream_chunk=True,
    ) is None
    first = sub.prepare_inputs(C.VIDEO_DECODE_AR_WALK, fwd(0), {"window_latents": [stream[:, :, 0:8]]})
    out = sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, **first.tensor_inputs, **first.kwargs)
    assert "video_output" not in out
    second = sub.prepare_inputs(
        C.VIDEO_DECODE_AR_WALK, fwd(1), {"window_latents": [stream[:, :, 7:12]], **_loop_edges(out)},
    )
    out = sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, **second.tensor_inputs, **second.kwargs)
    video = out["video_output"][0]

    expected = sub._decode_pixels(stream)[0].transpose(0, 1)  # [T, C, H, W] frames
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

    out0, out1 = _decode(sub, engine_inputs, stream[:, :, 0:8], stream[:, :, 7:12])
    c0 = out0["video_output"][0]
    c1 = out1["video_output"][0]
    assert c0.shape[0] == 29 and c1.shape[0] == 14
    expected = sub._decode_pixels(stream)[0].transpose(0, 1)
    assert torch.equal(torch.cat([c0, c1]), expected[:43])
    assert out1[DECODE_FRAMES][0].numel() == 0


def test_windowed_stream_video_gen_params() -> None:
    """stream_video always runs windowed: chained unless a mode is named."""
    model = _windowed_model()
    p = model._resolve_gen_params(
        {"window_mode": "chained", "num_frames": 189, "stream_video": True}, [], ["video"],
    )
    assert p["stream_video"] is True
    q = model._resolve_gen_params({"window_mode": "kv", "num_frames": 189}, [], ["video"])
    assert q["stream_video"] is False
    d = model._resolve_gen_params({"num_frames": 189, "stream_video": True}, [], ["video"])
    assert d["window_mode"] == "chained" and d["stream_video"] is True
    assert (d["window_latent_units"], d["overlap_latent_units"]) == (8, 2)
    plain = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    with pytest.raises(ValueError, match="disabled"):
        plain._resolve_gen_params({"num_frames": 189, "stream_video": True}, [], ["video"])


def test_video_streaming_ndjson_lines() -> None:
    """stream_video turns the video handler into an NDJSON generator: indexed
    video lines closed by a done line, an in-band error line terminal instead
    on a failure after the first window, a refusal before it raised with its
    status, and the non-streaming path collecting as before."""
    from fastapi import HTTPException

    from mstar.api_server.entrypoint import APIServer
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import VideoGenerationRequest
    from mstar.api_server.openai.serving_videos import create_videos
    from mstar.api_server.request_types import ResultChunk

    class _Api:
        upload_dir = "/tmp"
        open_result_stream = APIServer.open_result_stream

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
        response = await create_videos(api, "cosmos3", adapter, streaming_req)
        return [json.loads(line) async for line in response.body_iterator]

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

    api = _Api([ResultChunk(request_id="r", modality="error", data=b"bad image", metadata={"status": 400})])
    with pytest.raises(HTTPException) as refused:
        asyncio.run(_lines(api))
    assert (refused.value.status_code, refused.value.detail) == (400, "bad image")

    api = _Api([ResultChunk(request_id="r", modality="video", data=b"v")])
    out = asyncio.run(create_videos(api, "cosmos3", adapter, VideoGenerationRequest(prompt="x", num_frames=57)))
    assert api.submits[0]["streaming"] is False
    assert "stream_video" not in api.submits[0]["model_kwargs"]
    assert base64.b64decode(out["data"][0]["b64_json"]) == b"v"


def test_session_gen_params() -> None:
    """A session id rides along; resuming grows the schedule by the pinned
    head (at least two latent frames) and keeps num_frames the new-frame
    count; resume without a session or with an image is rejected."""
    model = _windowed_model()
    p = model._resolve_gen_params(
        {"window_mode": "kv", "num_frames": 189, "session_id": "world-7"}, [], ["video"],
    )
    assert p["session_id"] == "world-7" and p["resume_latent_units"] == 0
    assert p["total_latent_units"] == 48 and p["num_windows"] == 6
    r = model._resolve_gen_params(
        {"window_mode": "kv", "num_frames": 189, "session_id": "world-7", "resume_session": True},
        [], ["video"],
    )
    # 48 units + a 2-unit head = 50, padded to whole 8-unit windows -> 56 / 7 windows.
    assert r["resume_latent_units"] == 2 and r["total_latent_units"] == 56 and r["num_windows"] == 7
    assert r["num_frames"] == 189
    c = model._resolve_gen_params(
        {"window_mode": "chained", "num_frames": 189, "session_id": "s", "resume_session": True},
        [], ["video"],
    )
    assert c["resume_latent_units"] == c["overlap_latent_units"] == 2
    with pytest.raises(ValueError):
        model._resolve_gen_params({"window_mode": "kv", "num_frames": 189, "resume_session": True}, [], ["video"])
    with pytest.raises(ValueError):
        model._resolve_gen_params(
            {"window_mode": "kv", "num_frames": 189, "session_id": "s", "resume_session": True},
            ["image", "text"], ["video"],
        )
    # An id on a non-windowed request is simply not a session.
    assert "session_id" not in model._resolve_gen_params({"num_frames": 189, "session_id": "s"}, [], ["video"])
    # end_session and the timeout ride along; the timeout is validated against its maximum.
    assert p["end_session"] is False and "session_timeout_s" not in p
    e = model._resolve_gen_params(
        {"window_mode": "kv", "num_frames": 189, "session_id": "s", "end_session": True, "session_timeout_s": 3600},
        [], ["video"],
    )
    assert e["end_session"] is True and e["session_timeout_s"] == 3600.0
    with pytest.raises(ValueError, match=r"session_timeout_s must be in \(0, 3600\]"):
        model._resolve_gen_params(
            {"window_mode": "kv", "num_frames": 189, "session_id": "s", "session_timeout_s": 1e6}, [], ["video"],
        )
    with pytest.raises(ValueError, match="end_session requires"):
        model._resolve_gen_params({"window_mode": "kv", "num_frames": 189, "end_session": True}, [], ["video"])
    with pytest.raises(ValueError, match="session_timeout_s"):
        model._resolve_gen_params(
            {"window_mode": "kv", "num_frames": 189, "session_id": "s", "session_timeout_s": 0}, [], ["video"],
        )


def test_session_tail_store_and_resume(monkeypatch) -> None:
    """The DiT keeps a session's last window and decode context (bounded,
    most recent kept); a resumed request pins the head built from it as
    window 0's clean head (statics with the head's frames clean, the vmask and
    pinned latents in place for the first denoise iteration), sends the
    context ahead of its first window, and unknown or mismatched sessions are
    refused at the prefill."""
    sub = _session_dit(monkeypatch)
    sub._sessions.capacity = 2
    # The head is the stored window's re-encoded tail (see
    # test_chained_window_head_layout); here, its last latents plus 10.
    sub._window_head = lambda w, units: w[:, :, -units:] + 10.0
    shape = sub._window_latent_shape(64, 64, 4)
    for sid, value in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        _finish(sub, {**_SESSION_MD, "session_id": sid}, torch.full(shape, value))
    assert sub._sessions.ids() == ["b", "c"]  # "a" evicted (store size 2)
    stored, context = sub._sessions.get("c")
    assert float(stored[0, 0, 0, 0, 0]) == 3.0 and context.shape[2] == 3

    resume = {**_SESSION_MD, "session_id": "c", "resume_latent_units": 2}
    with pytest.raises(ValueError, match="unknown or expired"):
        _prefill(sub, "r-a", {**resume, "session_id": "a"})
    with pytest.raises(ValueError, match="cannot seed"):
        _prefill(sub, "r-c", {**resume, "height": 128, "width": 128})
    for rid in ("r-a", "r-c"):
        sub.cleanup_request(rid)  # the engine removes a refused request

    # Resume "c": window 0's statics carry two clean head frames, and the
    # first iteration's noise keeps the head.
    _prefill(sub, "r", resume)
    assert sub._sessions.ids() == ["b", "c"] and sub._sessions.live("c") == "r"
    bind_progress(sub, "r")
    fwd = SimpleNamespace(
        request_id="r", rid_handle="r", random_seed=0, step_metadata=resume,
        dynamic_loop_iter_counts={VIDEO_GEN_AR_LOOP: 0},
    )
    ni = sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, {"cond_latents": [], WINDOW_COND: [], SESSION_TAIL: []})
    st = ni.kwargs["settings"]
    stride = st["ar_tokens_per_unit"]
    assert st["cond"]["num_noisy_vision_tokens"] == 2 * stride and st["cond"]["num_vision_tokens"] == 4 * stride
    assert st["vmask"].shape[2] == 4 and st["vmask"][0, 0, :2].sum() == 2 and st["vmask"][0, 0, 2:].sum() == 0
    lat = ni.tensor_inputs["latents"]
    assert torch.all(lat[:, :, :2] == 13.0) and not torch.all(lat[:, :, 2:] == 13.0)
    assert ni.input_seq_len == 4 * stride and not ni.resource_step_info.commit

    # The first window goes to the decoder behind the session's decode
    # context, which gains the window's new latents.
    out = sub._end_window(st, lat, {})
    assert torch.equal(out["window_latents"][0], torch.cat([context, lat], dim=2)), "decoded without its context"
    keep = max(1, int(sub.config.windowed_decode_context_latents))
    assert torch.equal(out[SESSION_TAIL][0], torch.cat([context, lat[:, :, 2:]], dim=2)[:, :, -keep:])


def test_session_decoder_resumes_with_context(monkeypatch) -> None:
    """A resumed request's first window reaches the streaming decoder behind
    the session's decode context, which the DiT keeps and sends ahead of the
    window's latents: the decoder trims the pinned head and decodes the first
    new frames behind that context, so two requests of one session
    concatenate exactly to the whole-stream decode."""
    sub = _ar_decoder(monkeypatch)
    stream = torch.arange(16, dtype=torch.float32).view(1, 1, 16, 1, 1).expand(1, 16, 16, 4, 4).contiguous()

    def infos(md):
        return SimpleNamespace(request_ids=[md["rid"]], per_request_info={md["rid"]: SimpleNamespace(step_metadata=md)})

    # Request 1: one 8-unit window.
    md1 = {"rid": "r1", "num_windows": 1, "overlap_latent_units": 0, "num_frames": 29}
    out1 = _decode(sub, infos(md1), stream[:, :, 0:8])[0]
    # Request 2 resumes: its window re-pins units 6..8 and generates 8..16
    # (10 units), asking for the 32 new frames = 8 new units x 4, behind the
    # session's last 3 latents.
    md2 = {
        "rid": "r2", "num_windows": 1, "overlap_latent_units": 0, "num_frames": 32, "resume_latent_units": 2,
        "window_latent_units": 10, "total_latent_units": 10,
    }
    out2 = _decode(sub, infos(md2), torch.cat([stream[:, :, 5:8], stream[:, :, 6:16]], dim=2))[0]
    video = torch.cat([out1["video_output"][0], out2["video_output"][0]])
    assert torch.equal(video, sub._decode_pixels(stream)[0].transpose(0, 1))


def _session_dit(monkeypatch):
    sub = _dit()
    sub.transformer = _stand_in(sub)
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: StubSampler(()))
    monkeypatch.setattr(sub, "get_device", lambda: torch.device("cpu"))
    return sub


_SESSION_MD = {
    "window_mode": "kv", "total_latent_units": 8, "window_latent_units": 4, "overlap_latent_units": 0,
    "context_latent_units": 0, "session_id": "w", "height": 64, "width": 64, "num_inference_steps": 3,
    "guidance_scale": 1.0, C.PROMPT_LENGTHS: (7, 0),
}


def _prefill(sub, rid, md):
    fwd = SimpleNamespace(request_id=rid, rid_handle=rid, random_seed=0, step_metadata=md)
    sub.prepare_inputs(C.PREFILL_WALK, fwd, {"text_inputs": [torch.arange(7)]})


def _finish(sub, md, latents):
    """``postprocess`` of the last step of a ``_SESSION_MD`` rollout, which keeps or ends the session ``md`` names."""
    last = 2 * 4 - 1  # two windows of three steps and a commit
    settings = sub._settings(md, last, {}, torch.device("cpu"))
    info = SimpleNamespace(graph_walk=C.VIDEO_GEN_AR_WALK, dynamic_loop_iter_counts={VIDEO_GEN_AR_LOOP: last})
    outputs = {"latents": [latents], "window_latents": [latents], SESSION_TAIL: [latents[:, :, -3:]]}
    sub.postprocess("r", info, outputs, SimpleNamespace(kwargs={"settings": settings}))


def test_session_in_flight_is_pinned_and_exclusive(monkeypatch) -> None:
    """A request that names a session pins it from its prefill: the store
    never evicts it while other sessions churn, a second request on it is
    refused with a clear error, and the pin goes when the rollout keeps its
    last window."""
    sub = _session_dit(monkeypatch)
    sub._sessions.capacity = 1
    _prefill(sub, "r1", _SESSION_MD)
    assert sub._sessions.live("w") == "r1"
    with pytest.raises(ValueError, match="in use by request 'r1'"):
        _prefill(sub, "r2", _SESSION_MD)
    shape = sub._window_latent_shape(64, 64, 4)
    for sid in ("x", "y"):
        _finish(sub, {**_SESSION_MD, "session_id": sid}, torch.zeros(shape))
    assert sub._sessions.live("w") == "r1" and sub._sessions.ids() == ["w", "y"]
    # r1's final window stores the window and lets go.
    _finish(sub, _SESSION_MD, torch.full(shape, 5.0))
    assert sub._sessions.live("w") is None and float(sub._sessions.get("w")[0][0, 0, 0, 0, 0]) == 5.0
    _prefill(sub, "r2", {**_SESSION_MD, "resume_latent_units": 2})
    assert sub._sessions.live("w") == "r2"


def test_session_released_when_the_request_is_removed(monkeypatch) -> None:
    sub = _session_dit(monkeypatch)
    _prefill(sub, "r1", _SESSION_MD)
    assert sub._sessions.live("w") == "r1"
    sub.cleanup_request("r1")  # cancelled mid-rollout
    assert sub._sessions.live("w") is None and "w" not in sub._sessions


def test_session_end_and_timeout(monkeypatch) -> None:
    """``end_session`` drops the session at the final window; an idle session
    outlives its request only for ``session_timeout_s``."""
    clock = {"t": 0.0}
    sub = _session_dit(monkeypatch)
    sub._sessions = SessionStore(4, 10.0, 60.0, clock=lambda: clock["t"])
    shape = sub._window_latent_shape(64, 64, 4)
    _finish(sub, {**_SESSION_MD, "session_id": "gone", "end_session": True}, torch.zeros(shape))
    assert "gone" not in sub._sessions
    _finish(sub, {**_SESSION_MD, "session_id": "brief", "session_timeout_s": 5.0}, torch.zeros(shape))
    resume = {**_SESSION_MD, "session_id": "brief", "resume_latent_units": 2}
    clock["t"] = 4.0
    _prefill(sub, "r1", resume)
    sub.cleanup_request("r1")
    clock["t"] = 6.0
    with pytest.raises(ValueError, match="unknown or expired"):
        _prefill(sub, "r2", resume)



def test_long_rollout_state_stays_flat(monkeypatch) -> None:
    """A long rollout (24 windows, both modes) rebuilds each step's settings
    from the node's memo, where chained windows past the first share one
    entry; the layout masks are the step's window's, and every window is
    emitted."""
    sub = _dit()
    sub.transformer = _stand_in(sub)
    monkeypatch.setattr(sub, "_new_scheduler", lambda *a, **k: StubSampler(torch.arange(3)))
    for mode, kv in (("chained", False), ("kv", True)):
        md = {
            "window_mode": mode, "window_latent_units": 8, "height": 64, "width": 64, "num_inference_steps": 3,
            "guidance_scale": 6.0, C.PROMPT_LENGTHS: (7, 9),
        }
        if kv:
            md.update(total_latent_units=8 * 24, overlap_latent_units=0, context_latent_units=16)
        else:
            md.update(total_latent_units=8 + 6 * 23, overlap_latent_units=2)
        per = 4 if kv else 3
        x = torch.zeros(sub._window_latent_shape(64, 64, 8))
        pinned = torch.zeros(0)
        for w in range(24):
            edges = {WINDOW_COND: [pinned], SESSION_TAIL: [torch.zeros(0)]}
            st = sub._settings(md, w * per, edges, torch.device("cpu"))
            assert st["ar_schedule"].num_windows == 24 and st["window_index"] == w
            token_mask, frame_mask = st["cond"]["noisy_token_mask"], st["cond"]["noisy_frame_mask"]
            assert token_mask.shape == (8 * 4,) and frame_mask.shape == (1, 8, 1, 1)
            assert int(frame_mask.sum()) == (8 if kv or w == 0 else 6)
            out = sub._end_window(st, x, {})
            assert out["window_latents"][0].shape[2] == 8
            pinned = out[WINDOW_COND][0]
        # the first window's statics, chained windows' one more, and each kv window's own
        assert sum(key[0] == "window" for key in sub._statics.entries) == (2 + 23 if kv else 2)
