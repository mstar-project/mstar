"""Denoise step prep reads no tensor on the host (Cosmos3, Wan 2.2, OmniVoice).

A loop-carried step counter (``step_index``) is the previous step's device output,
so an ``.item()`` on it in ``prepare_inputs`` waits for that step to finish and
stalls the speculative preparation of the next one. These loops read the step from
the engine's per-batch loop counter instead, as the DiT scaffold does, and Cosmos3
keeps its timesteps as a host list.

Each test walks a short schedule (including the async-overshoot step the
prepare vetoes) and checks, per step:

* no host read of any tensor (``item`` / ``tolist`` / ``int()`` / ...) happens
  while preparing with the loop counter set, as the engine sets it;
* for OmniVoice, the prepared inputs equal those of the edge-reading path, which
  callers that pass no loop counter still use. Cosmos3 and Wan 2.2 carry no such
  edge, so a missing counter is an error there.

No weights. The whole Wan 2.2 and Cosmos3 steps also run under CUDA's sync-debug
mode, which needs a GPU; everything else is CPU only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers import UniPCMultistepScheduler
from torch import nn
from torch.overrides import TorchFunctionMode

from mstar.engine.windowing import WindowSchedule
from mstar.model.components.diffusion.unipc import SOLVER_STATE
from mstar.model.cosmos3 import constants as C
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.submodules import (
    ACTION_GEN_LOOP,
    ACTION_VIDEO_GEN_LOOP,
    IMAGE_GEN_LOOP,
    VIDEO_GEN_AR_LOOP,
    VIDEO_SOUND_GEN_LOOP,
    Cosmos3DiTSubmodule,
)
from mstar.model.omnivoice.config import OmniVoiceConfig
from mstar.model.omnivoice.submodules import UNMASK_LOOP_NAME, OmniVoiceBackboneSubmodule
from mstar.model.submodule_base import NodeInputs
from mstar.model.wan22.config import Wan22Config
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, Wan22DitSubmodule

_T = torch.Tensor
_HOST_READS = {_T.item, _T.tolist, _T.cpu, _T.numpy, _T.__int__, _T.__float__, _T.__bool__, _T.__index__}


class _HostReads(TorchFunctionMode):
    """Records every host read of a tensor's values made inside the block."""

    def __init__(self):
        super().__init__()
        self.reads: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in _HOST_READS:
            self.reads.append(func.__name__)
        return func(*args, **(kwargs or {}))


def _fwd(rid, walk, step_metadata=None, loop=None, k=None):
    """A forward-pass info as the engine hands it to prepare_inputs: with the
    loop counter (``loop``/``k``), or without one (a direct caller)."""
    return SimpleNamespace(
        request_id=rid, rid_handle=rid, random_seed=7, graph_walk=walk,
        step_metadata=step_metadata or {},
        dynamic_loop_iter_counts={loop: k} if loop is not None else {},
    )


def _assert_same(got: NodeInputs | None, want: NodeInputs | None):
    if want is None:
        assert got is None
        return
    assert got is not None
    assert got.tensor_inputs.keys() == want.tensor_inputs.keys()
    for key, value in want.tensor_inputs.items():
        assert got.tensor_inputs[key].dtype == value.dtype, key
        assert torch.equal(got.tensor_inputs[key], value), key
    assert got.input_seq_len == want.input_seq_len
    assert got.resource_step_info == want.resource_step_info
    assert got.kwargs == want.kwargs


def _check_schedule(prepare, walk_fwd, edge_inputs, loop, steps):
    """Prepare steps 1..steps (the last one is the overshoot) with and without
    the loop counter; the counter path reads nothing and matches the edge path."""
    for k in range(1, steps + 1):
        with _HostReads() as counter_path:
            got = prepare(walk_fwd(loop, k), edge_inputs(k))
        assert counter_path.reads == [], f"step {k}: host reads {counter_path.reads}"
        with _HostReads() as edge_path:
            want = prepare(walk_fwd(None, None), edge_inputs(k))
        # the edge path is the old derivation: it reads the step off the device edge
        assert edge_path.reads, f"step {k}: the edge path should read the step on the host"
        _assert_same(got, want)


# ---------------------------------------------------------------- Wan 2.2

def _wan22_loop_inputs(shape, k):
    return {
        "text_embeds_pos": [torch.zeros(1, 8, 16)],
        "text_embeds_neg": [torch.zeros(1, 8, 16)],
        "latents": [torch.full(shape, float(k))],
        "unipc_model_outputs": [torch.zeros((2, *shape))],
        "unipc_last_sample": [torch.zeros(shape)],
    }


_WAN22_META = {"num_inference_steps": 3, "guidance_scale": 5.0, "height": 32, "width": 32, "num_frames": 5}


def _wan22_sub(transformer=None, **config) -> Wan22DitSubmodule:
    sub = Wan22DitSubmodule(transformer=transformer, config=Wan22Config(**config))
    sub.get_device = lambda: torch.device("cpu")
    return sub


def test_wan22_step_prep_reads_no_tensor():
    sub = _wan22_sub()
    shape = (1, 4, 2, 2, 2)
    steps = _WAN22_META["num_inference_steps"]

    def prepare(k):
        info = _fwd("r", "video_gen", _WAN22_META, DENOISE_LOOP_NAME, k)
        return sub.prepare_inputs("video_gen", info, _wan22_loop_inputs(shape, k))

    for k in range(1, steps):
        with _HostReads() as host:
            got = prepare(k)
        assert host.reads == [], f"step {k}: host reads {host.reads}"
        assert got.tensor_inputs.keys() == {*_wan22_loop_inputs(shape, k), "timestep"}
        assert torch.equal(got.tensor_inputs["latents"], torch.full(shape, float(k)))
    with _HostReads() as host:
        assert prepare(steps) is None  # overshoot vetoed
    assert host.reads == []


def test_wan22_step_prep_fails_without_the_loop_counter():
    sub = _wan22_sub()
    info = _fwd("r", "video_gen", _WAN22_META)
    with pytest.raises(KeyError, match=DENOISE_LOOP_NAME):
        sub.prepare_inputs("video_gen", info, _wan22_loop_inputs((1, 4, 2, 2, 2), 1))


class _WanStubTransformer(nn.Module):
    """The Wan DiT's call signature and output on device ops alone, so the only
    host syncs a step can take are the submodule's and the solver's."""

    dtype = torch.bfloat16

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return hidden_states * (timestep[:, None, :, None, None] * 1e-3 + encoder_hidden_states.mean())


def _wan22_steps(sub, walk, steps, latents, image_latent):
    """A whole schedule as the engine drives it, prepare through forward, the step
    from the loop counter and the solver state over the loop-back edges."""
    device = latents.device
    sub.get_device = lambda: device
    sub.request_states.pop("r", None)
    ring, last = torch.zeros((2, *latents.shape), device=device), torch.zeros_like(latents)
    inputs = {
        "text_embeds_pos": [torch.full((1, 8, 16), 1.0, device=device, dtype=torch.bfloat16)],
        "text_embeds_neg": [torch.full((1, 8, 16), 0.5, device=device, dtype=torch.bfloat16)],
    }
    if image_latent is not None:
        inputs["image_latent"] = [image_latent]
    meta = {**_WAN22_META, "num_inference_steps": steps, "num_frames": 9}
    for k in range(steps):
        info = _fwd("r", walk, meta, DENOISE_LOOP_NAME, k)
        carried = {"latents": [latents], "unipc_model_outputs": [ring], "unipc_last_sample": [last]}
        if k == 0:
            # the seeds are the caller's here, so a device sample is in play from step 0
            sub.initial_loop_back = lambda *a, l=latents, r=ring, s=last: {
                "latents": l, "unipc_model_outputs": r, "unipc_last_sample": s,
            }
        prepared = sub.prepare_inputs(walk, info, {**inputs, **carried})
        engine_inputs = SimpleNamespace(request_ids=["r"], per_request_info={"r": info})
        out = sub.forward(walk, engine_inputs=engine_inputs, **sub.preprocess(walk, engine_inputs, [prepared]))
        latents, ring, last = out["latents"][0], out["unipc_model_outputs"][0], out["unipc_last_sample"][0]
    return latents


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the host sync is a CUDA stream sync")
@pytest.mark.parametrize("walk", ["video_gen", "video_gen_i2v"])
def test_wan22_forward_takes_no_host_sync(walk):
    """A step that waits on the GPU keeps the worker from queueing it behind the one
    still running. Sync-debug mode raises on one: a host read of a device tensor, a
    blocking copy from pageable memory, or ``linalg.solve``'s check of its info.
    Four steps cover the predictor alone, both corrector orders and the last step."""
    sub = _wan22_sub(_WanStubTransformer(), in_channels=4, compile_dit=False)
    gen = torch.Generator().manual_seed(0)
    shape = (1, 4, 3, 2, 2)
    latents = torch.randn(shape, generator=gen)
    image_latent = torch.randn(shape, generator=gen) if walk == sub._I2V_WALK else None
    want = _wan22_steps(sub, walk, 4, latents, image_latent)

    latents = latents.cuda()
    image_latent = None if image_latent is None else image_latent.cuda()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = _wan22_steps(sub, walk, 4, latents, image_latent)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    # the tables staged without a sync are the host's: the run lands where the host's does
    assert torch.allclose(got.cpu(), want, atol=1e-2), (got.cpu() - want).abs().max()


# ---------------------------------------------------------------- OmniVoice

def _omnivoice():
    cfg = OmniVoiceConfig()
    sub = OmniVoiceBackboneSubmodule(backbone=None, config=cfg)
    sub.get_device = lambda: torch.device("cpu")
    return sub, cfg


def _omnivoice_meta(steps):
    return {
        "num_step": steps, "t_shift": 0.1, "guidance_scale": 2.0, "class_temperature": 0.0,
        "layer_penalty_factor": 5.0, "position_temperature": 5.0,
    }


def _omnivoice_inputs(cfg, k, target_len=6, clone=False):
    c = cfg.num_audio_codebook
    gen = torch.Generator().manual_seed(k)
    tokens = torch.randint(0, cfg.audio_mask_id + 1, (c, target_len), generator=gen)
    inputs = {
        "prefix_ids": [torch.arange(c * 5).view(c, 5)],
        "prefix_audio_mask": [torch.zeros(5, dtype=torch.bool)],
        "target_len": [torch.tensor([target_len])],
        "audio_tokens": [tokens],
        "step_index": [torch.tensor([k], dtype=torch.int64)],
    }
    if clone:
        inputs["ref_audio_tokens"] = [torch.arange(c * 3).view(c, 3)]
    return inputs


@pytest.mark.parametrize("clone", [False, True])
def test_omnivoice_step_prep_reads_no_tensor(clone):
    sub, cfg = _omnivoice()
    steps = 4

    def fwd(loop, k):
        return _fwd("r", "speech_gen", _omnivoice_meta(steps), loop, k)

    def prepare(fwd_info, inp):
        return sub.prepare_inputs("speech_gen", fwd_info, inp)

    _check_schedule(prepare, fwd, lambda k: _omnivoice_inputs(cfg, k, clone=clone), UNMASK_LOOP_NAME, steps)
    assert prepare(fwd(UNMASK_LOOP_NAME, steps), _omnivoice_inputs(cfg, steps)) is None


def test_omnivoice_first_step_prep_unchanged():
    # iteration 0 still reads target_len once to size the canvas
    sub, cfg = _omnivoice()
    inputs = _omnivoice_inputs(cfg, 0)
    del inputs["audio_tokens"], inputs["step_index"]
    ni = sub.prepare_inputs("speech_gen", _fwd("r", "speech_gen", _omnivoice_meta(4)), inputs)
    assert ni.tensor_inputs["audio_tokens"].shape == (cfg.num_audio_codebook, 6)
    assert (ni.tensor_inputs["audio_tokens"] == cfg.audio_mask_id).all()
    assert ni.input_seq_len == 5 + 2 * 6


def test_omnivoice_postprocess_reads_no_tensor():
    """The unmask step's postprocess takes the iteration from the loop counter:
    same canvas as reading the step edge, and no host read."""
    from mstar.model.omnivoice.components.unmask import build_reveal_schedule

    sub, cfg = _omnivoice()
    steps, target_len = 4, 6
    shape = (cfg.num_audio_codebook, target_len, cfg.audio_vocab_size)
    meta = _omnivoice_meta(steps)
    # memoized per request shape; its one tolist() is of a CPU ramp it builds, not a sync
    build_reveal_schedule(
        target_len=target_len, num_codebook=cfg.num_audio_codebook, num_step=steps, t_shift=meta["t_shift"],
    )
    for k in range(steps):
        inputs = _omnivoice_inputs(cfg, k, target_len)
        inputs["audio_tokens"] = [torch.full((cfg.num_audio_codebook, target_len), cfg.audio_mask_id)]
        node_inputs = sub.prepare_inputs("speech_gen", _fwd("r", "speech_gen", _omnivoice_meta(steps)), inputs)
        logits = torch.randn(shape, generator=torch.Generator().manual_seed(100 + k))
        results = {}
        for name, info in (
            ("counter", _fwd("r", "speech_gen", _omnivoice_meta(steps), UNMASK_LOOP_NAME, k)),
            ("edge", _fwd("r", "speech_gen", _omnivoice_meta(steps))),
        ):
            outputs = {"c_logits": [logits.clone()], "u_logits": [logits.flip(-1)]}
            with _HostReads() as reads:
                sub.postprocess("r", info, outputs, inputs=node_inputs)
            results[name] = (outputs, reads.reads)
        assert results["counter"][1] == [], f"step {k}: host reads {results['counter'][1]}"
        assert results["edge"][1], "the edge path should read the step on the host"
        assert torch.equal(results["counter"][0]["audio_tokens"][0], results["edge"][0]["audio_tokens"][0])
        assert torch.equal(results["counter"][0]["step_index"][0], results["edge"][0]["step_index"][0])


# ---------------------------------------------------------------- Cosmos3

def _cosmos3(monkeypatch=None, capture=False):
    if capture:
        monkeypatch.setenv("COSMOS3_GEN_CAPTURE_RES", "64x64")
        monkeypatch.setenv("COSMOS3_GEN_CAPTURE_VIDEO", "64x64x9")
        monkeypatch.setenv("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH", "1")
        monkeypatch.delenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", raising=False)
    from mstar.model.cosmos3.tests.test_rotary_table import stand_in_rotary

    config = Cosmos3Config(compile_denoise=False)
    transformer = SimpleNamespace(
        proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
        sp_group=SimpleNamespace(world_size=1), comm_group=SimpleNamespace(world_size=1),
        _rotary=stand_in_rotary(config),
    )
    sub = Cosmos3DiTSubmodule(
        transformer=transformer, config=config, scheduler=UniPCMultistepScheduler(),
    )
    sub.get_device = lambda: torch.device("cpu")
    if capture:
        configs = sub.get_accelerator_graph_configs("cpu")
        assert {config.capture_graph_walk for config in configs} == {
            C.IMAGE_GEN_WALK, C.VIDEO_GEN_WALK,
        }
    return sub


# walk, its loop, the latent edges it carries, and the request state it reads
_COSMOS3_WALKS = [
    (C.IMAGE_GEN_WALK, IMAGE_GEN_LOOP, ("latents",), {"guidance_interval": (400.0, 1000.0)}),
    (C.VIDEO_GEN_WALK, "video_gen_loop", ("latents",), {"guidance_interval": None}),
    (C.VIDEO_SOUND_GEN_WALK, VIDEO_SOUND_GEN_LOOP, ("latents", "sound_latents"), {"num_sound": 3}),
    (C.ACTION_GEN_WALK, ACTION_GEN_LOOP, ("latents", "action_latents"), {}),
    (C.ACTION_VIDEO_GEN_WALK, ACTION_VIDEO_GEN_LOOP, ("latents", "action_latents"), {}),
]


@pytest.mark.parametrize("walk,loop,edges,extra", _COSMOS3_WALKS)
@pytest.mark.parametrize("cfg", [True, False])
def test_cosmos3_step_prep_reads_no_tensor(walk, loop, edges, extra, cfg):
    sub = _cosmos3()
    steps = 8
    ch = sub.config.latent_channel
    cond = {"num_vision_tokens": 16, "num_action_tokens": 4}
    st = sub.request_state("r")
    st.add_all(
        cond=cond, uncond=dict(cond) if cfg else None, gs=6.0,
        scheduler=sub._new_scheduler(steps, "cpu", flow_shift=3.0),
        latent_shape=(1, ch, 1, 4, 4), **extra,
    )
    # the interval splits the default schedule, so cfg_active changes mid-loop
    if walk == C.IMAGE_GEN_WALK and cfg:
        t = st["scheduler"].timesteps
        assert (t < 400).any() and (t >= 400).any()

    def inputs(k):
        out = {name: [torch.full((1, ch, 1, 4, 4), float(k))] for name in edges}
        out.update({name: [torch.zeros(1)] for name in SOLVER_STATE})
        return out

    for k in range(1, steps + 1):
        with _HostReads() as host:
            got = sub.prepare_inputs(walk, _fwd("r", walk, {}, loop, k), inputs(k))
        assert host.reads == [], f"step {k}: host reads {host.reads}"
        if k == steps:
            assert got is None  # overshoot vetoed
            continue
        assert got.tensor_inputs.keys() >= {*edges, *SOLVER_STATE}
        assert torch.equal(got.tensor_inputs["latents"], inputs(k)["latents"][0])
    with pytest.raises(KeyError, match=loop):
        sub.prepare_inputs(walk, _fwd("r", walk), inputs(1))


def test_cosmos3_windowed_captured_step_prep_reads_no_tensor(monkeypatch):
    """A chained windowed request on a captured shape: the global loop counter
    maps to the within-window step and the staged graph inputs match."""
    sub = _cosmos3(monkeypatch, capture=True)
    ch = sub.config.latent_channel
    cond, uncond = sub._build_window_statics(
        list(range(7)), list(range(9)), 64, 64, 3, 24.0,
        has_image_condition=False, cond_units=1, device="cpu", first_window=False,
    )
    st = sub.request_state("r")
    st.add_all(
        cond=cond, uncond=uncond, gs=6.0, guidance_interval=None,
        scheduler=sub._new_scheduler(3, "cpu"),
        latent_shape=(1, ch, 3, 4, 4), ar_schedule=WindowSchedule(6, 3, overlap_units=1),
        ar_steps=3, ar_iters_per_window=3, ar_total_iters=9, ar_kv_mode=False,
    )

    inputs = {"latents": [torch.zeros(1, ch, 3, 4, 4)], **{name: [torch.zeros(1)] for name in SOLVER_STATE}}

    def prepare(k):
        return sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, _fwd("r", C.VIDEO_GEN_AR_WALK, {}, VIDEO_GEN_AR_LOOP, k), inputs)

    prepare(1)  # the per-statics mask cache is filled on first use
    for k in range(1, 10):
        with _HostReads() as host:
            got = prepare(k)
        assert host.reads == [], f"step {k}: host reads {host.reads}"
        assert (got is None) == (k == 9)
    ni = prepare(4)
    assert ni.resource_step_info.capture_key == (1, ch, 3, 4, 4)
    assert torch.equal(ni.tensor_inputs["vision_timesteps"],
                       st["scheduler"].timesteps[1].to(torch.float32).expand(12))


class _CosmosStubTransformer:
    """The generation tower's steps on device ops alone, so the only host syncs a
    step can take are the submodule's and the solver's."""

    proj_in = SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16))

    @staticmethod
    def _branches(latents, vision_timesteps, band):
        cond = latents * (vision_timesteps.float().mean() * 1e-3)
        return (cond, cond * 0.5) if band is None else ((cond, band * 0.1), (cond * 0.5, band * 0.05))

    def denoise_step_batched_cfg(self, latents, ts, *args, sound_latents=None, action_latents=None, **kwargs):
        return self._branches(latents, ts, sound_latents if sound_latents is not None else action_latents)

    def denoise_step_row(self, static, latents, vision_timesteps, *args, **kwargs):
        return {"latents": latents, "vision_timesteps": vision_timesteps, **kwargs}

    def denoise_step_batched(self, reqs, label, attn):
        return [self._branches(r["latents"], r["vision_timesteps"], r.get("sound_latents")) for r in reqs]

    def denoise_step_action_batched(self, reqs, label, attn, with_cfg):
        return [self._branches(r["latents"], r["vision_timesteps"], r["action_latents"]) for r in reqs]


_EDGE_SCHEDULER = {
    "num_train_timesteps": 1000, "solver_order": 2, "prediction_type": "flow_prediction", "predict_x0": True,
    "solver_type": "bh2", "lower_order_final": True, "use_flow_sigmas": True, "use_karras_sigmas": True,
    "flow_shift": 1.0, "final_sigmas_type": "zero", "timestep_spacing": "linspace", "sigma_min": 0.147,
    "sigma_max": 200.0,
}


def _cosmos3_request(sub, walk, rid, steps, device):
    """A request past its prefill: one latent frame of 4 tokens, per walk's extras."""
    ch = sub.config.latent_channel
    bf16 = dict(device=device, dtype=torch.bfloat16)
    cond = {
        "num_vision_tokens": 16, "num_noisy_vision_tokens": 16, "num_action_tokens": 4,
        "num_noisy_action_tokens": 4, "rotary": (None, None), "vision_token_shapes": [(1, 4, 4)],
        "vision_noisy_frame_indexes": [None], "mse_gen_indexes": None, "action_token_shapes": None,
        "action_noisy_frame_indexes": None, "action_mse_gen_indexes": None, "sound_token_shapes": None,
        "sound_noisy_frame_indexes": None, "sound_mse_gen_indexes": None,
    }
    extra = {}
    if walk == C.VIDEO_SOUND_GEN_WALK:
        extra = {"num_sound": 3}
    elif walk == C.ACTION_GEN_WALK:
        extra = {
            "raw_action_dim": 2, "action_chunk": 3, "action_dim": 4, "domain_t": None,
            "vmask": torch.zeros(1, 1, 1, 1, 1, **bf16), "cond_video_latents": torch.zeros(1, ch, 1, 4, 4, **bf16),
            "action_clean_mask": torch.zeros(1, 3, 1, **bf16), "clean_action": torch.zeros(1, 3, 4, **bf16),
        }
    sub.request_state(rid).add_all(
        cond=cond, uncond=dict(cond), gs=6.0, guidance_interval=None, latent_shape=(1, ch, 1, 4, 4),
        scheduler=sub._new_scheduler(steps, device, flow_shift=12.0), **extra,
    )


_COSMOS3_STEP_WALKS = [
    (C.VIDEO_GEN_WALK, "video_gen_loop", None),
    (C.VIDEO_SOUND_GEN_WALK, VIDEO_SOUND_GEN_LOOP, "sound_latents"),
    (C.ACTION_GEN_WALK, ACTION_GEN_LOOP, "action_latents"),
]


def _cosmos3_stepper(walk, loop, band, rows, steps, device, wrap=lambda sampler: sampler):
    """A generator of whole engine passes over a fresh DiT, prepare through forward,
    yielding ``(k, {rid: edges})`` after each."""
    sub = _cosmos3()
    sub._scheduler_template = UniPCMultistepScheduler.from_config(_EDGE_SCHEDULER)
    sub.transformer = _CosmosStubTransformer()
    sub.get_device = lambda: device
    rids = [f"r{i}" for i in range(rows)]
    for rid in rids:
        _cosmos3_request(sub, walk, rid, steps, device)
        st = sub.request_state(rid)
        st.add("scheduler", wrap(st["scheduler"]))
    fwds = {rid: _fwd(rid, walk, {}, loop, 0) for rid in rids}
    for rid in rids:
        fwds[rid].random_seed = int(rid[1:]) + 7
    carried = {rid: {} for rid in rids}
    edges = ("latents", *((band,) if band else ()), *SOLVER_STATE)
    for k in range(steps):
        prepared = []
        for rid in rids:
            fwds[rid].dynamic_loop_iter_counts[loop] = k
            prepared.append(sub.prepare_inputs(walk, fwds[rid], carried[rid]))
        engine_inputs = SimpleNamespace(
            request_ids=rids, per_request_info=fwds, per_request_states=None, resources={"attn": None},
            step=None, captured=False,
        )
        stacked = sub.preprocess(walk, engine_inputs, prepared)
        out = sub.forward_batched(walk, engine_inputs, **stacked) if rows > 1 else {
            rids[0]: sub.forward(walk, engine_inputs, **stacked)
        }
        carried = {rid: {name: out[rid][name] for name in edges} for rid in rids}
        yield k, carried


class _Lockstep:
    """A request's sampler, with diffusers' stateful scheduler taking every step
    beside it from the same inputs. All the reference remembers has to come back
    over the loop-back edges for the two to agree."""

    def __init__(self, sampler, steps):
        self.sampler = sampler
        self.reference = UniPCMultistepScheduler.from_config(_EDGE_SCHEDULER, flow_shift=12.0)
        self.reference.set_timesteps(steps, device="cpu")
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def step(self, model_output, sample, step_index, solver, generator=None):
        got = self.sampler.step(model_output, sample, step_index, solver, generator)
        want = self.reference.step(model_output, self.reference.timesteps[step_index], sample, return_dict=False)[0]
        assert torch.equal(got[0], want), f"step {step_index}"
        self.steps += 1
        return got


@pytest.mark.parametrize("rows", [1, 2], ids=["single", "batched"])
@pytest.mark.parametrize("walk,loop,band", _COSMOS3_STEP_WALKS, ids=["video", "sound", "action"])
def test_cosmos3_step_is_the_reference_scheduler(walk, loop, band, rows):
    """Every path's whole schedule, one row or a batch, in lockstep with diffusers."""
    samplers = []

    def lockstep(sampler):
        samplers.append(_Lockstep(sampler, 6))
        return samplers[-1]

    for _ in _cosmos3_stepper(walk, loop, band, rows, 6, torch.device("cpu"), wrap=lockstep):
        pass
    assert [sampler.steps for sampler in samplers] == [6] * rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the host sync is a CUDA stream sync")
@pytest.mark.parametrize("rows", [1, 2], ids=["single", "batched"])
@pytest.mark.parametrize("walk,loop,band", _COSMOS3_STEP_WALKS, ids=["video", "sound", "action"])
def test_cosmos3_step_takes_no_host_sync(walk, loop, band, rows):
    """A schedule of the eager denoise step as the engine drives it, under sync-debug
    mode from the step after the seeds. Four steps cover the predictor alone, both
    corrector orders and the last step; sound and action step the packed state."""
    device = torch.device("cuda")
    passes = _cosmos3_stepper(walk, loop, band, rows, 4, device)
    next(passes)  # the seeds are host work
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        *_, (_, last) = passes
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert all(torch.isfinite(row["latents"][0]).all() for row in last.values())


def test_cosmos3_cfg_active_uses_host_timesteps():
    """The guidance-interval test reads the host copy, with the same values
    ``.item()`` gave, and builds no new host read per call."""
    sub = _cosmos3()
    sched = sub._new_scheduler(8, "cpu", flow_shift=3.0)
    st = {"uncond": {}, "guidance_interval": (400.0, 1000.0), "scheduler": sched}
    for k in range(9):
        with _HostReads() as reads:
            active = sub._cfg_active(st, k)
        assert reads.reads == []
        want = k < len(sched.timesteps) and 400.0 <= float(sched.timesteps[k].item()) <= 1000.0
        assert active == want
