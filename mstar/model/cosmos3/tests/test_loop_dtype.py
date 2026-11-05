"""The DiT's denoise loop dtype, on CPU with a stand-in transformer.

diffusers 0.41's Cosmos3OmniPipeline keeps the denoise state of every band in
fp32 (the noise, the latents between steps, the guidance combine and the
scheduler step) and casts to bf16 only for the transformer input. The served
loop does the same for image, video, sound and action requests (windowed
rollouts keep bf16), and the DiT's forwards run with the engine's CUDA autocast
off so the fp32 step is not cast back down. The stand-in returns fixed bf16
velocities per guidance branch, so a guidance combine or a scheduler step run
in bf16 shows in the result.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import UniPCMultistepScheduler  # noqa: E402

from mstar.model.cosmos3 import constants as C  # noqa: E402
from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer  # noqa: E402
from mstar.model.cosmos3.config import Cosmos3Config  # noqa: E402
from mstar.model.cosmos3.submodules import (  # noqa: E402
    ACTION_GEN_LOOP,
    ATTN,
    COND_LABEL,
    IMAGE_GEN_LOOP,
    LATENT_DTYPE,
    VIDEO_GEN_LOOP,
    VIDEO_SOUND_GEN_LOOP,
    Cosmos3DiTSubmodule,
)
from mstar.model.cosmos3.tests.test_dit_step_guard import EDGE_SCHEDULER  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs  # noqa: E402

GS = 6.0
CHUNK, RAW = 4, 3
VIDEO_SHAPE = (1, 8, 2, 4, 4)
IMAGE_SHAPE = (1, 8, 1, 4, 4)


def _velocity(shape, branch: int) -> torch.Tensor:
    """A fixed bf16 velocity per shape and guidance branch (1 cond, 2 uncond)."""
    gen = torch.Generator().manual_seed(1000 * branch + sum(shape))
    return torch.randn(shape, generator=gen).to(torch.bfloat16)


class _Tower:
    """The transformer surface the loop touches: bf16 weights, bf16 velocities."""

    denoise_step_row = staticmethod(Cosmos3OmniTransformer.denoise_step_row)

    def __init__(self, cfg):
        self.proj_in = SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16))
        self.action_dim = cfg.max_action_dim

    @staticmethod
    def _branch(latents, branch, sound_latents=None, action_latents=None, **kwargs):
        v = _velocity(latents.shape, branch)
        if action_latents is not None:
            return v, _velocity(action_latents.shape, branch)
        if sound_latents is not None:
            return v, _velocity(sound_latents.shape, branch)
        return v

    def denoise_step(self, latents, vts, rotary, shapes, noisy, mse, label, attn, **kwargs):
        return self._branch(latents, 1 if label == COND_LABEL else 2, **kwargs)

    def denoise_step_batched_cfg(self, latents, vts, rot_c, rot_u, shapes, noisy, mse, label, attn, **kwargs):
        kwargs.pop("noisy_token_mask", None)
        return self._branch(latents, 1, **kwargs), self._branch(latents, 2, **kwargs)

    def denoise_step_batched(self, reqs, label, attn):
        out = []
        for r in reqs:
            extra = {"sound_latents": r["sound_latents"]} if r.get("sound_latents") is not None else {}
            out.append((self._branch(r["latents"], 1, **extra), self._branch(r["latents"], 2, **extra)))
        return out

    def denoise_step_action_batched(self, reqs, label, attn, with_cfg):
        branches = (1, 2) if with_cfg else (1,)
        return [
            [self._branch(r["latents"], b, action_latents=r["action_latents"]) for b in branches] for r in reqs
        ]


def _guided(cond, uncond):
    return uncond.float() + GS * (cond.float() - uncond.float())


def _dit() -> Cosmos3DiTSubmodule:
    cfg = Cosmos3Config(
        compile_denoise=False, latent_channel=8, sound_dim=4, max_action_dim=12, use_native_flow_schedule=True,
    )
    sub = Cosmos3DiTSubmodule(
        transformer=_Tower(cfg), config=cfg, scheduler=UniPCMultistepScheduler.from_config(EDGE_SCHEDULER),
    )
    sub.node_resources = {ATTN: None}
    return sub


def _static(shape, **extra) -> dict:
    t = shape[2]
    tokens = t * (shape[3] // 2) * (shape[4] // 2)
    return {
        "num_vision_tokens": tokens, "num_noisy_vision_tokens": tokens, "rotary": (None, None),
        "vision_token_shapes": [(t, shape[3] // 2, shape[4] // 2)],
        "vision_noisy_frame_indexes": [torch.arange(t)], "mse_gen_indexes": None, **extra,
    }


def _gen_request(sub, rid, shape, num_sound=None):
    st = sub.request_state(rid)
    extra = {"sound_token_shapes": None, "sound_noisy_frame_indexes": None, "sound_mse_gen_indexes": None}
    cond = _static(shape, **(extra if num_sound else {}))
    st.add_all(
        cond=cond, uncond=cond, gs=GS, scheduler=sub._new_scheduler(4, "cpu", flow_shift=12.0),
        latent_shape=shape, num_sound=num_sound,
    )
    return st


def _action_request(sub, rid, cfg_branch=True):
    st = sub.request_state(rid)
    shape = VIDEO_SHAPE
    vmask = torch.zeros((1, 1, shape[2], 1, 1), dtype=torch.bfloat16)
    vmask[:, :, 0] = 1.0
    cond = _static(
        shape, num_action_tokens=CHUNK, num_noisy_action_tokens=CHUNK, action_token_shapes=[(CHUNK, 1, 1)],
        action_noisy_frame_indexes=[torch.arange(CHUNK)], action_mse_gen_indexes=None,
    )
    st.add_all(
        cond=cond, uncond=cond if cfg_branch else None, gs=GS,
        scheduler=sub._new_scheduler(4, "cpu", flow_shift=5.0), latent_shape=shape,
        action_chunk=CHUNK, action_dim=sub.transformer.action_dim, raw_action_dim=RAW,
        domain_t=torch.tensor([1]), vmask=vmask,
        action_clean_mask=torch.zeros((1, CHUNK, 1), dtype=torch.bfloat16),
        clean_action=torch.zeros((1, CHUNK, sub.transformer.action_dim), dtype=torch.bfloat16),
    )
    return st


def _fwd(rid, loop, k=0, seed=7):
    return SimpleNamespace(
        request_id=rid, rid_handle=rid, random_seed=seed, dynamic_loop_iter_counts=defaultdict(int, {loop: k}),
    )


def _solver(sub, tensors):
    return {name: tensors[name] for name in sub._solver_state}


def test_sound_request_draws_the_video_requests_noise() -> None:
    """A sound request's video band starts from the same tensor as the same-seed
    video request, and both bands start in fp32 like the reference's."""
    sub = _dit()
    _gen_request(sub, "v", VIDEO_SHAPE)
    _gen_request(sub, "s", VIDEO_SHAPE, num_sound=5)
    video = sub._prepare_image_gen(C.VIDEO_GEN_WALK, _fwd("v", VIDEO_GEN_LOOP), {}, "cpu").tensor_inputs
    sound = sub._prepare_video_sound_gen(_fwd("s", VIDEO_SOUND_GEN_LOOP), {}, "cpu").tensor_inputs
    assert video["latents"].dtype == LATENT_DTYPE
    assert torch.equal(sound["latents"], video["latents"])
    assert sound["sound_latents"].dtype == LATENT_DTYPE and sound["sound_latents"].shape == (1, 4, 5)
    assert all(t.dtype == LATENT_DTYPE for name, t in sound.items() if name in sub._solver_state)


@pytest.mark.parametrize("guided", [True, False])
def test_sound_step_combines_and_steps_in_fp32(guided) -> None:
    sub = _dit()
    st = _gen_request(sub, "s", VIDEO_SHAPE, num_sound=5)
    if not guided:
        st.add("guidance_interval", (0.0, 1.0))  # every step outside: the conditional branch alone
    inp = sub._prepare_video_sound_gen(_fwd("s", VIDEO_SOUND_GEN_LOOP), {}, "cpu").tensor_inputs
    lat, snd = inp["latents"], inp["sound_latents"]
    out = sub._forward_video_sound_gen(None, st, 0, lat, snd, **_solver(sub, inp))
    v_c, s_c = _velocity(lat.shape, 1), _velocity(snd.shape, 1)
    if guided:
        v, s = _guided(v_c, _velocity(lat.shape, 2)), _guided(s_c, _velocity(snd.shape, 2))
    else:
        v, s = v_c.float(), s_c.float()
    ref_lat, ref_snd, ref_solver = sub._sound_scheduler_step(
        st, lat, snd, v, s, 0, _solver(sub, inp),
    )
    assert out["latents"][0].dtype == LATENT_DTYPE and out["sound_latents"][0].dtype == LATENT_DTYPE
    assert torch.equal(out["latents"][0], ref_lat) and torch.equal(out["sound_latents"][0], ref_snd)
    assert all(torch.equal(out[name][0], ref_solver[name]) for name in sub._solver_state)


def test_batched_sound_step_combines_in_fp32() -> None:
    sub = _dit()
    rids = ["a", "b"]
    inps = {}
    for i, rid in enumerate(rids):
        _gen_request(sub, rid, VIDEO_SHAPE, num_sound=5)
        inps[rid] = sub._prepare_video_sound_gen(_fwd(rid, VIDEO_SOUND_GEN_LOOP, seed=i), {}, "cpu").tensor_inputs
    ei = SimpleNamespace(
        request_ids=rids, per_request_states=None, step=None, resources=None,
        per_request_info={rid: _fwd(rid, VIDEO_SOUND_GEN_LOOP) for rid in rids},
    )
    solvers = {name: {rid: inps[rid][name] for rid in rids} for name in sub._solver_state}
    out = sub._forward_batched_sound(
        C.VIDEO_SOUND_GEN_WALK, ei, {r: inps[r]["latents"] for r in rids},
        {r: inps[r]["sound_latents"] for r in rids}, solvers,
    )
    for rid in rids:
        lat, snd = inps[rid]["latents"], inps[rid]["sound_latents"]
        v = _guided(_velocity(lat.shape, 1), _velocity(lat.shape, 2))
        s = _guided(_velocity(snd.shape, 1), _velocity(snd.shape, 2))
        ref_lat, ref_snd, _ = sub._sound_scheduler_step(
            sub.request_states[rid], lat, snd, v, s, 0, _solver(sub, inps[rid]),
        )
        assert torch.equal(out[rid]["latents"][0], ref_lat) and torch.equal(out[rid]["sound_latents"][0], ref_snd)


@pytest.mark.parametrize("cfg_branch", [True, False])
def test_action_step_draws_combines_and_steps_in_fp32(cfg_branch) -> None:
    sub = _dit()
    st = _action_request(sub, "a", cfg_branch=cfg_branch)
    inp = sub._prepare_action_gen(C.ACTION_GEN_WALK, _fwd("a", ACTION_GEN_LOOP), {}, "cpu").tensor_inputs
    lat, act = inp["latents"], inp["action_latents"]
    assert lat.dtype == LATENT_DTYPE and act.dtype == LATENT_DTYPE
    assert all(inp[name].dtype == LATENT_DTYPE for name in sub._solver_state)
    out = sub._forward_action_gen(None, st, 0, lat, act, **_solver(sub, inp))
    v_c, a_c = _velocity(lat.shape, 1), _velocity(act.shape, 1)
    if cfg_branch:
        v, a = _guided(v_c, _velocity(lat.shape, 2)), _guided(a_c, _velocity(act.shape, 2))
    else:
        v, a = v_c.float(), a_c.float()
    ref_lat, ref_act, _ = sub._action_scheduler_step(st, lat, act, v, a, 0, _solver(sub, inp))
    assert out["latents"][0].dtype == LATENT_DTYPE and out["action_latents"][0].dtype == LATENT_DTYPE
    assert torch.equal(out["latents"][0], ref_lat) and torch.equal(out["action_latents"][0], ref_act)


@pytest.mark.parametrize("cfg_branch", [True, False])
def test_batched_action_step_combines_in_fp32(cfg_branch) -> None:
    sub = _dit()
    rids = ["a", "b"]
    inps = {}
    for i, rid in enumerate(rids):
        _action_request(sub, rid, cfg_branch=cfg_branch)
        inps[rid] = sub._prepare_action_gen(C.ACTION_GEN_WALK, _fwd(rid, ACTION_GEN_LOOP, seed=i), {}, "cpu")
        inps[rid] = inps[rid].tensor_inputs
    ei = SimpleNamespace(
        request_ids=rids, per_request_states=None, step=None, resources=None,
        per_request_info={rid: _fwd(rid, ACTION_GEN_LOOP) for rid in rids},
    )
    solvers = {name: {rid: inps[rid][name] for rid in rids} for name in sub._solver_state}
    out = sub._forward_batched_action(
        C.ACTION_GEN_WALK, ei, {r: inps[r]["latents"] for r in rids},
        {r: inps[r]["action_latents"] for r in rids}, solvers,
    )
    for rid in rids:
        lat, act = inps[rid]["latents"], inps[rid]["action_latents"]
        v_c, a_c = _velocity(lat.shape, 1), _velocity(act.shape, 1)
        if cfg_branch:
            v, a = _guided(v_c, _velocity(lat.shape, 2)), _guided(a_c, _velocity(act.shape, 2))
        else:
            v, a = v_c.float(), a_c.float()
        ref_lat, ref_act, _ = sub._action_scheduler_step(
            sub.request_states[rid], lat, act, v, a, 0, _solver(sub, inps[rid]),
        )
        assert torch.equal(out[rid]["latents"][0], ref_lat) and torch.equal(out[rid]["action_latents"][0], ref_act)


def test_embed_action_takes_fp32_action_latents() -> None:
    """The action band reaches the bf16 domain-aware projection from fp32 loop
    latents, cast the way the vision band is in ``_patchify_and_pack_latents``."""
    from mstar.model.cosmos3.tests.test_action import _cfg

    model = Cosmos3OmniTransformer(_cfg()).eval().to(torch.bfloat16)
    lat = torch.randn(1, CHUNK, model.action_dim)
    args = (torch.tensor([1]), torch.full((CHUNK,), 500.0), [(CHUNK, 1, 1)], [torch.arange(CHUNK)], torch.bfloat16)
    with torch.no_grad():
        out = model._embed_action(lat, *args)
        ref = model._embed_action(lat.to(torch.bfloat16), *args)
    assert out.dtype == torch.bfloat16 and torch.equal(out, ref)


def test_eager_image_step_matches_the_captured_tail() -> None:
    """The eager step and the captured step's tail (``postprocess``) combine and
    step the same fp32 state, step after step, given the same velocities."""
    sub = _dit()
    st = _gen_request(sub, "r", IMAGE_SHAPE)
    fwd = _fwd("r", IMAGE_GEN_LOOP)
    inp = sub._prepare_image_gen(C.IMAGE_GEN_WALK, fwd, {}, "cpu").tensor_inputs
    lat, solver = inp["latents"], _solver(sub, inp)
    ei = SimpleNamespace(
        request_ids=["r"], per_request_states=None, step=None, resources=None, per_request_info={"r": fwd},
    )
    info = SimpleNamespace(graph_walk=C.IMAGE_GEN_WALK, dynamic_loop_iter_counts=fwd.dynamic_loop_iter_counts)
    for k in range(len(st["scheduler"].timesteps)):
        fwd.dynamic_loop_iter_counts[IMAGE_GEN_LOOP] = k
        eager = sub.forward(C.IMAGE_GEN_WALK, ei, latents=lat, **solver)
        captured = {"cond_v": [_velocity(lat.shape, 1)], "uncond_v": [_velocity(lat.shape, 2)]}
        sub.postprocess("r", info, captured, inputs=ARNodeInputs(tensor_inputs={"latents": lat, **solver}))
        for name in ("latents", *sub._solver_state):
            assert eager[name][0].dtype == LATENT_DTYPE
            assert torch.equal(eager[name][0], captured[name][0]), (k, name)
        lat, solver = eager["latents"][0], {n: eager[n][0] for n in sub._solver_state}


def _record_autocast(seen):
    def record(*args, **kwargs):
        seen.append(torch.is_autocast_enabled("cuda"))
        return {}

    return record


@pytest.mark.parametrize("walk", [C.PREFILL_WALK, C.IMAGE_GEN_WALK, C.VIDEO_SOUND_GEN_WALK, C.ACTION_GEN_WALK])
def test_forwards_run_with_the_engine_autocast_off(walk, monkeypatch) -> None:
    """The engine runs a node's forward under its bf16 autocast; the DiT's eager
    and batched forwards turn it off for everything they call (on a CPU host
    the flag is set by hand, which is what the engine's scope does on a GPU)."""
    sub = _dit()
    seen: list[bool] = []
    for name in (
        "_forward_prefill", "_forward_image_gen", "_forward_video_sound_gen", "_forward_action_gen",
        "_forward_batched_sound", "_forward_batched_action",
    ):
        monkeypatch.setattr(sub, name, _record_autocast(seen))
    sub.transformer.prefill_und = _record_autocast(seen)
    sub.request_state("r")
    ei = SimpleNamespace(
        request_ids=["r"], per_request_states=None, step=None, resources=None,
        per_request_info={"r": SimpleNamespace(dynamic_loop_iter_counts=defaultdict(int))},
    )
    before = torch.is_autocast_enabled("cuda")
    torch.set_autocast_enabled("cuda", True)
    try:
        sub.forward(walk, ei)
        if walk != C.IMAGE_GEN_WALK:  # the batched image step runs inline, with no helper to record
            sub.forward_batched(walk, ei)
        assert torch.is_autocast_enabled("cuda")
    finally:
        torch.set_autocast_enabled("cuda", before)
    assert seen == [False] * len(seen) and len(seen) == (1 if walk == C.IMAGE_GEN_WALK else 2)
