"""CPU checks for the generation tower's mRoPE table: the DiT statics build
``rotary = (cos, sin)`` once per request and per window (sliced with the
window), every denoise step and the kv commit take it without building a
table of their own, and a captured step's inputs stack it per guidance branch.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import torch

from mstar.engine.windowing import WindowSchedule
from mstar.model.cosmos3 import constants as C
from mstar.model.cosmos3.components.packing import build_static_inputs
from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer, Cosmos3RotaryEmbedding
from mstar.model.cosmos3.submodules import (
    ATTN,
    CAPTURED_GEN_INPUTS,
    CFG_BATCHED_LABEL,
    COND_LABEL,
    UNCOND_LABEL,
    Cosmos3DiTSubmodule,
)
from mstar.model.cosmos3.tests.test_edge import _init_all, _tiny_edge_config
from mstar.model.cosmos3.tests.test_engine_cache import _SdpaResources

POSITION_FIELDS = ("position_ids", "gen_position_ids", "vision_mrope_ids", "sound_mrope_ids", "action_mrope_ids")
STRIDE = 4  # generation tokens per latent frame at 64x64 (4x4 latents, 2x2 patches)


def stand_in_rotary(cfg):
    """The real ``Cosmos3OmniTransformer._rotary`` for a stand-in transformer,
    so tests that fake the tower still get the statics' table."""
    holder = SimpleNamespace(rotary_emb=Cosmos3RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.rope_axes_dim))
    return functools.partial(Cosmos3OmniTransformer._rotary, holder)


def _tiny(dtype=torch.float32):
    cfg = _tiny_edge_config(
        compile_denoise=False, action_gen=True, max_action_dim=12, num_embodiment_domains=6,
    )
    model = Cosmos3OmniTransformer(cfg).eval()
    _init_all(model)
    model.to(dtype)
    res = _SdpaResources()
    res.bind(model)
    sub = Cosmos3DiTSubmodule(model, cfg, scheduler=None)
    sub.get_device = lambda: torch.device("cpu")
    return sub, model, cfg, res


def _equal(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_statics_hold_the_table() -> None:
    """Per request: the table is ``_rotary`` over the generation band (vision,
    then sound or action) in the tower's dtype, it outlives the slimmed-away
    position ids, and its layout does not depend on the prompt length (the
    compiled step guards on it)."""
    sub, model, cfg, _ = _tiny(torch.bfloat16)
    video = sub._build_static(list(range(5)), 64, 64, 9, 24.0, True, "cpu")
    sound = sub._build_static(list(range(5)), 64, 64, 9, 24.0, False, "cpu", sound_latent_frames=4)
    action = sub._build_action_static(list(range(5)), 64, 64, 9, 4, "policy", 10.0, 10.0, 1, "cpu")
    for s, n_gen in (
        (video, video["num_vision_tokens"]),
        (sound, sound["num_vision_tokens"] + sound["num_sound_tokens"]),
        (action, action["num_vision_tokens"] + action["num_action_tokens"]),
    ):
        cos, sin = s["rotary"]
        assert cos.shape == sin.shape == (n_gen, cfg.head_dim)
        assert cos.dtype == sin.dtype == torch.bfloat16
        assert _equal(s["rotary"], model._rotary(s["gen_position_ids"], "cpu", torch.bfloat16))
    sub._slim_statics(video, action)
    sub._slim_statics(sound, None)
    for s in (video, sound, action):
        assert "rotary" in s and not set(POSITION_FIELDS) & set(s)

    for build in (
        lambda ids: sub._build_static(ids, 64, 64, 9, 24.0, False, "cpu", sound_latent_frames=4),
        lambda ids: sub._build_action_static(ids, 64, 64, 9, 4, "policy", 10.0, 10.0, 1, "cpu"),
    ):
        layouts = {
            (tuple(t.shape), t.stride(), t.storage_offset())
            for n in (4, 9, 23) for t in build(list(range(n)))["rotary"]
        }
        assert len(layouts) == 1


def test_window_tables_slice_with_the_window(monkeypatch) -> None:
    """Per window: a kv window's table is the full clip's table at the
    window's rows; chained windows past the first share one cached table;
    a commit hands the transformer the rows past the window's pinned head."""
    sub, model, cfg, _ = _tiny()
    ids, unc_ids = list(range(6)), list(range(9))
    full = sub._build_static(ids, 64, 64, 1 + 11 * 4, 24.0, False, "cpu")  # 12 latent frames
    st = sub.request_state("r")
    st.add_all(
        ar_kv_mode=True, ar_statics={}, ar_size=(64, 64), ar_cond_ids=ids, ar_uncond_ids=unc_ids,
        ar_fps=24.0, ar_has_image_condition=False,
    )
    kv = WindowSchedule(total_units=12, window_units=4, context_units=6, overlap_units=0)
    for w in range(kv.num_windows):
        plan = kv.window(w)
        cond, uncond = sub._window_statics_for(st, plan, "cpu")
        rows = slice(plan.start * STRIDE, plan.end * STRIDE)
        assert _equal(cond["rotary"], (full["rotary"][0][rows], full["rotary"][1][rows]))
        assert uncond["rotary"][0].shape == cond["rotary"][0].shape
        assert not set(POSITION_FIELDS) & set(cond)
    assert st["ar_statics"] == {}  # kv windows are built on demand, never cached

    st.add("ar_kv_mode", False)
    chained = WindowSchedule(total_units=13, window_units=4, overlap_units=1)  # four full windows
    tables = [sub._window_statics_for(st, chained.window(w), "cpu")[0]["rotary"] for w in range(chained.num_windows)]
    assert all(t is tables[1] for t in tables[2:]) and tables[0] is not tables[1]

    commits = []
    monkeypatch.setattr(model, "commit_window", lambda lat, rot, label, attn: commits.append((rot, label)))
    monkeypatch.setattr(sub, "_finish_window", lambda *a: {})
    plan = chained.window(1)
    cond, uncond = sub._window_statics_for(st, plan, "cpu")
    st.add_all(cond=cond, uncond=uncond, ar_schedule=chained, ar_tokens_per_unit=STRIDE)
    x = torch.zeros(sub._window_latent_shape(64, 64, plan.units))
    head = plan.cond_units * STRIDE
    sub._commit_window("paged", st, x, torch.tensor([0]), window_index=1)
    (rot, label), = commits
    assert label == CFG_BATCHED_LABEL and len(rot) == 2
    assert _equal(rot[0], tuple(t[head:] for t in cond["rotary"]))
    assert _equal(rot[1], tuple(t[head:] for t in uncond["rotary"]))
    sub.batched_cfg = False
    sub._commit_window("paged", st, x, torch.tensor([0]), window_index=1)
    assert [c[1] for c in commits[1:]] == [COND_LABEL, UNCOND_LABEL]
    assert _equal(commits[2][0][0], tuple(t[head:] for t in uncond["rotary"]))


def _prefill(model, res, statics):
    for label, s in statics.items():
        res.plan(label, [(label, s["und_len"])], causal=True)
        model.prefill_und(s["input_ids"], s["text_mrope_ids"], label)
        res.kv.promote()


def _run(res, plan_label, groups, fn):
    res.plan(plan_label, groups, causal=False)
    out = fn()
    res.kv.pending = {}  # a denoise step's own K/V is never committed
    return out


def test_denoise_steps_take_the_table(monkeypatch) -> None:
    """Every step runs with no table built inside it, and its result is the
    fused reference's (which builds its own table from the joint positions):
    single, batched CFG and cross-request batched video steps, the action
    steps, and the kv commit."""
    sub, model, cfg, res = _tiny()
    torch.manual_seed(0)
    cond_ids, unc_ids = [3, 5, 7, 11, 13], [2, 4, 6, 8, 10, 12, 14]
    latent = torch.randn(1, cfg.latent_channel, 3, 4, 4)
    ref = build_static_inputs(cond_ids, tuple(latent.shape), cfg, 4, 24.0, "cpu", has_image_condition=True)
    ts = torch.full((ref["num_noisy_vision_tokens"],), 500.0)
    fields = ("input_ids", "text_indexes", "position_ids", "und_len", "sequence_length", "vision_token_shapes",
              "vision_sequence_indexes", "vision_mse_loss_indexes", "vision_noisy_frame_indexes")
    with torch.no_grad():
        fused, _ = model(vision_tokens=[latent], vision_timesteps=ts, **{k: ref[k] for k in fields})

    cond = sub._build_static(cond_ids, 64, 64, 9, 24.0, True, "cpu")
    uncond = sub._build_static(unc_ids, 64, 64, 9, 24.0, True, "cpu")
    a_cond = sub._build_action_static(cond_ids, 64, 64, 9, 4, "policy", 10.0, 10.0, 1, "cpu")
    a_unc = sub._build_action_static(unc_ids, 64, 64, 9, 4, "policy", 10.0, 10.0, 1, "cpu")
    with torch.no_grad():
        _prefill(model, res, {COND_LABEL: cond, UNCOND_LABEL: uncond})
    for s in (cond, uncond, a_cond, a_unc):
        sub._slim_statics(s, None)

    def no_table(*args, **kwargs):
        raise AssertionError("a denoise step built a rotary table")

    monkeypatch.setattr(model.rotary_emb, "forward", no_table)
    attn, n = res.attn, cond["num_vision_tokens"]
    both = [(COND_LABEL, n), (UNCOND_LABEL, n)]
    with torch.no_grad():
        v_c = _run(res, COND_LABEL, [(COND_LABEL, n)], lambda: sub._denoise(attn, cond, latent, ts, COND_LABEL))
        v_u = _run(res, UNCOND_LABEL, [(UNCOND_LABEL, n)],
                   lambda: sub._denoise(attn, uncond, latent, ts, UNCOND_LABEL))
        assert torch.allclose(v_c, fused[0], atol=1e-4), (v_c - fused[0]).abs().max()
        b_c, b_u = _run(res, CFG_BATCHED_LABEL, both, lambda: model.denoise_step_batched_cfg(
            latent, ts, cond["rotary"], uncond["rotary"], cond["vision_token_shapes"],
            cond["vision_noisy_frame_indexes"], cond["mse_gen_indexes"], CFG_BATCHED_LABEL, attn,
        ))
        req = {
            "latents": latent, "vision_timesteps": ts, "rotary_cond": cond["rotary"],
            "rotary_uncond": uncond["rotary"], "vision_token_shapes": cond["vision_token_shapes"],
            "vision_noisy_frame_indexes": cond["vision_noisy_frame_indexes"],
            "vision_mse_loss_indexes": cond["mse_gen_indexes"],
        }
        ((x_c, x_u),) = _run(
            res, CFG_BATCHED_LABEL, both, lambda: model.denoise_step_batched([req], CFG_BATCHED_LABEL, attn),
        )
    for got in (b_c, x_c):
        assert torch.allclose(got, v_c, atol=1e-5)
    for got in (b_u, x_u):
        assert torch.allclose(got, v_u, atol=1e-5)

    action = torch.randn(1, 4, cfg.max_action_dim)
    domain = torch.tensor([2])
    vts = torch.full((a_cond["num_noisy_vision_tokens"],), 500.0)
    ats = torch.full((a_cond["num_noisy_action_tokens"],), 500.0)
    m = a_cond["num_vision_tokens"] + a_cond["num_action_tokens"]
    with torch.no_grad():
        seq = [
            _run(res, label, [(label, m)], lambda s=s, label=label: sub._denoise_action(
                attn, s, latent, action, vts, ats, domain, label))
            for label, s in ((COND_LABEL, a_cond), (UNCOND_LABEL, a_unc))
        ]
        areq = {
            "latents": latent, "action_latents": action, "vision_timesteps": vts, "action_timesteps": ats,
            "rotary_cond": a_cond["rotary"], "rotary_uncond": a_unc["rotary"],
            "vision_token_shapes": a_cond["vision_token_shapes"],
            "vision_noisy_frame_indexes": a_cond["vision_noisy_frame_indexes"],
            "vision_mse_loss_indexes": a_cond["mse_gen_indexes"],
            "action_token_shapes": a_cond["action_token_shapes"],
            "action_noisy_frame_indexes": a_cond["action_noisy_frame_indexes"],
            "action_mse_gen_indexes": a_cond["action_mse_gen_indexes"], "action_domain_id": domain,
        }
        (batched,) = _run(res, CFG_BATCHED_LABEL, [(COND_LABEL, m), (UNCOND_LABEL, m)],
                          lambda: model.denoise_step_action_batched([areq], CFG_BATCHED_LABEL, attn, with_cfg=True))
        for (sv, sa), (bv, ba) in zip(seq, batched, strict=True):
            assert torch.allclose(bv, sv, atol=1e-5) and torch.allclose(ba, sa, atol=1e-5)

        res.plan(CFG_BATCHED_LABEL, both, causal=False)
        model.commit_window(latent, [cond["rotary"], uncond["rotary"]], CFG_BATCHED_LABEL, attn)
        assert {label for label, _ in res.kv.pending} == {COND_LABEL, UNCOND_LABEL}


def test_captured_inputs_stack_the_tables(monkeypatch) -> None:
    """The capture bucket's static inputs are each branch's cos and sin; a
    request at the captured shape stages its statics' tables as they are, the
    replay-side preprocess stacks them on the batch dim, and
    ``forward_captured`` hands every row its own pair per branch."""
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_RES", "64x64")
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_VIDEO", "64x64x9")
    monkeypatch.setenv("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH", "1")
    monkeypatch.delenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", raising=False)
    sub, model, cfg, _ = _tiny(torch.bfloat16)
    video_cfg = sub.get_accelerator_graph_configs("cpu")[1]
    assert video_cfg.capture_graph_walk == C.VIDEO_GEN_WALK
    single = video_cfg.single_request_inputs.tensor_inputs
    assert set(single) == set(CAPTURED_GEN_INPUTS)
    for key in ("cos_cond", "sin_cond", "cos_uncond", "sin_uncond"):
        assert single[key].shape == (3 * STRIDE, cfg.head_dim) and single[key].dtype == torch.bfloat16

    sched = SimpleNamespace(timesteps=torch.tensor([900.0, 500.0]), step_index=None)
    rids, inputs = ["a", "b"], []
    x = torch.zeros(1, cfg.latent_channel, 3, 4, 4, dtype=torch.bfloat16)
    for i, rid in enumerate(rids):
        cond = sub._build_static(list(range(5 + i)), 64, 64, 9, 24.0, True, "cpu")
        uncond = sub._build_static(list(range(9 + 2 * i)), 64, 64, 9, 24.0, True, "cpu")
        sub._slim_statics(cond, uncond)
        sub.request_state(rid).add_all(
            cond=cond, uncond=uncond, gs=6.0, guidance_interval=None, scheduler=sched,
            latent_shape=(1, cfg.latent_channel, 3, 4, 4),
        )
        fwd = SimpleNamespace(request_id=rid, rid_handle=rid, random_seed=0)
        inp = sub.prepare_inputs(C.VIDEO_GEN_WALK, fwd, {"latents": [x], "time_index": [torch.tensor([1])]})
        t = inp.tensor_inputs
        assert t["cos_cond"] is cond["rotary"][0] and t["sin_cond"] is cond["rotary"][1]
        assert t["cos_uncond"] is uncond["rotary"][0] and t["sin_uncond"] is uncond["rotary"][1]
        inputs.append(inp)
    assert not torch.equal(inputs[0].tensor_inputs["cos_uncond"], inputs[1].tensor_inputs["cos_uncond"])

    packed = sub.preprocess(C.VIDEO_GEN_WALK, SimpleNamespace(captured=True), inputs)
    assert set(packed) == set(CAPTURED_GEN_INPUTS)
    for key in ("cos_cond", "sin_cond", "cos_uncond", "sin_uncond"):
        assert packed[key].shape == (2, 3 * STRIDE, cfg.head_dim)
        assert torch.equal(packed[key], torch.stack([inp.tensor_inputs[key] for inp in inputs]))

    calls = []
    zero = torch.zeros(1)
    monkeypatch.setattr(model, "denoise_step_batched", lambda reqs, label, attn: calls.append(reqs) or [
        (zero, zero) for _ in reqs])
    monkeypatch.setattr(model, "denoise_step_batched_cfg", lambda *a, **k: calls.append(a) or (zero, zero))
    ei = SimpleNamespace(request_ids=rids, step=None, resources={ATTN: "paged"})
    sub.forward_captured(C.VIDEO_GEN_WALK, ei, **packed)
    for i, req in enumerate(calls[-1]):
        assert _equal(req["rotary_cond"], (packed["cos_cond"][i], packed["sin_cond"][i]))
        assert _equal(req["rotary_uncond"], (packed["cos_uncond"][i], packed["sin_uncond"][i]))
    one = {key: value[:1] for key, value in packed.items()}
    sub.forward_captured(C.VIDEO_GEN_WALK, SimpleNamespace(request_ids=rids[:1], step=None,
                                                           resources={ATTN: "paged"}), **one)
    args = calls[-1]
    assert _equal(args[2], (one["cos_cond"][0], one["sin_cond"][0]))
    assert _equal(args[3], (one["cos_uncond"][0], one["sin_uncond"][0]))
