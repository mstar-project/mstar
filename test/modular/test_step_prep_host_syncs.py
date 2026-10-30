"""Denoise step prep reads no tensor on the host (Cosmos3, Wan 2.2, OmniVoice).

The loop-carried step counter (``time_index`` / ``step_index``) is the
previous step's device output, so an ``.item()`` on it in ``prepare_inputs``
waits for that step to finish and stalls the speculative preparation of the
next one. These loops read the step from the engine's per-batch loop counter
instead, as the DiT scaffold does, and keep Cosmos3's timesteps as a host list.

Each test walks a short schedule (including the async-overshoot step the
prepare vetoes) and checks, per step:

* no host read of any tensor (``item`` / ``tolist`` / ``int()`` / ...) happens
  while preparing with the loop counter set, as the engine sets it;
* the prepared inputs equal those of the edge-reading path, which is the
  derivation the submodules used before (and still use for callers that pass
  no loop counter).

CPU only, no weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.overrides import TorchFunctionMode

from mstar.engine.windowing import WindowSchedule
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

def test_wan22_step_prep_reads_no_tensor():
    sub = Wan22DitSubmodule(transformer=None, config=Wan22Config())
    shape = (1, 4, 2, 2, 2)
    steps = 3

    def inputs(k):
        return {
            "text_embeds_pos": [torch.zeros(1, 8, 16)],
            "text_embeds_neg": [torch.zeros(1, 8, 16)],
            "latents": [torch.full(shape, float(k))],
            "time_index": [torch.tensor([k], dtype=torch.int64)],
            "unipc_model_outputs": [torch.zeros((2, *shape))],
            "unipc_last_sample": [torch.zeros(shape)],
        }

    def fwd(loop, k):
        return _fwd("r", "video_gen", {"num_inference_steps": steps}, loop, k)

    def prepare(fwd_info, inp):
        return sub.prepare_inputs("video_gen", fwd_info, inp)

    _check_schedule(prepare, fwd, inputs, DENOISE_LOOP_NAME, steps)
    assert prepare(fwd(DENOISE_LOOP_NAME, steps), inputs(steps)) is None  # overshoot vetoed


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
    from diffusers import UniPCMultistepScheduler

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
        out["time_index"] = [torch.tensor([k], dtype=torch.long)]
        return out

    def prepare(fwd_info, inp):
        return sub.prepare_inputs(walk, fwd_info, inp)

    _check_schedule(prepare, lambda lp, k: _fwd("r", walk, {}, lp, k), inputs, loop, steps)


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

    def inputs(k):
        return {"latents": [torch.zeros(1, ch, 3, 4, 4)], "time_index": [torch.tensor([k])]}

    def prepare(fwd_info, inp):
        return sub.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd_info, inp)

    # one call outside the counter: the per-statics mask cache is filled on first use
    prepare(_fwd("r", C.VIDEO_GEN_AR_WALK, {}, VIDEO_GEN_AR_LOOP, 1), inputs(1))
    _check_schedule(prepare, lambda lp, k: _fwd("r", C.VIDEO_GEN_AR_WALK, {}, lp, k), inputs, VIDEO_GEN_AR_LOOP, 9)
    ni = prepare(_fwd("r", C.VIDEO_GEN_AR_WALK, {}, VIDEO_GEN_AR_LOOP, 4), inputs(4))
    assert ni.resource_step_info.capture_key == (1, ch, 3, 4, 4)
    assert torch.equal(ni.tensor_inputs["vision_timesteps"],
                       st["scheduler"].timesteps[1].to(torch.float32).expand(12))


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
