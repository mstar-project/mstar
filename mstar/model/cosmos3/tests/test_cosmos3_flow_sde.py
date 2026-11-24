"""CPU-only tests for Cosmos3 Flow-SDE RL rollout, UniPC reset, masking, validation,
LIBERO configs, and batch=8 rollout scheduling."""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("diffusers")
from diffusers import UniPCMultistepScheduler  # noqa: E402

import mstar  # noqa: E402
from mstar.api_server.openpi import unpack  # noqa: E402
from mstar.model.components.diffusion.unipc import LAST_SAMPLE, MODEL_OUTPUTS, initial_state  # noqa: E402
from mstar.model.cosmos3 import constants as C  # noqa: E402
from mstar.model.cosmos3.config import Cosmos3Config  # noqa: E402
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model  # noqa: E402
from mstar.model.cosmos3.submodules import (  # noqa: E402
    ACTION_GEN_LOOP,
    ACTION_ROLLOUT_GEN_LOOP,
    ATTN,
    Cosmos3DiTSubmodule,
    _UniPCSampler,
)
from mstar.model.cosmos3.tests.test_distilled import _fake_distilled_dir  # noqa: E402
from mstar.model.cosmos3.tests.test_dit_step_guard import EDGE_SCHEDULER  # noqa: E402
from mstar.model.cosmos3.tests.test_loop_dtype import _Tower, _velocity  # noqa: E402
from mstar.model.registry import MODEL_REGISTRY  # noqa: E402

CONFIGS_DIR = Path(mstar.__file__).resolve().parent.parent / "configs"
VIDEO_SHAPE = (1, 8, 2, 4, 4)
CHUNK = 4
RAW_DIM = 3
MAX_ACTION_DIM = 12


def _fake_edge_dir(root: Path) -> Path:
    transformer = {
        "_class_name": "Cosmos3OmniTransformer",
        "action_dim": 64, "action_gen": True, "attention_bias": False, "backbone_type": "cosmos3_edge_nemotron_dense",
        "base_fps": 24, "enable_fps_modulation": True, "head_dim": 128, "hidden_act": "relu2", "hidden_size": 2048,
        "intermediate_size": 9216, "latent_channel": 48, "latent_patch_size": 2, "num_attention_heads": 16,
        "num_embodiment_domains": 32, "num_hidden_layers": 28, "num_key_value_heads": 8, "patch_latent_dim": 192,
        "qk_norm_for_text": False, "rms_norm_eps": 1e-05, "rope_scaling": {"mrope_section": [24, 20, 20]},
        "rope_theta": 100000000, "sound_dim": None, "sound_gen": False, "sound_latent_fps": 25,
        "temporal_compression_factor": 4, "timestep_scale": 0.001, "unified_3d_mrope_reset_spatial_ids": True,
        "unified_3d_mrope_temporal_modality_margin": 15000, "use_und_k_norm_for_gen": True, "vocab_size": 131072,
    }
    top = {
        "architectures": ["Cosmos3EdgeForConditionalGeneration"], "model_type": "cosmos3_edge",
        "image_token_id": 19, "video_token_id": 18, "vision_start_token_id": 20, "vision_end_token_id": 21,
        "projector_config": {"input_hidden_size": 1152, "merger_intermediate_size": 11520, "out_hidden_size": 2048,
                             "spatial_merge_size": 2, "use_postshuffle_norm": False},
        "text_config": {"eos_token_id": 11, "hidden_size": 2048, "max_position_embeddings": 131072},
        "vision_config": {"hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
                          "num_hidden_layers": 27, "num_patches": 256, "patch_size": 16, "spatial_merge_size": 2},
    }
    (root / "transformer").mkdir(parents=True)
    (root / "vision_encoder").mkdir()
    (root / "transformer" / "config.json").write_text(json.dumps(transformer))
    (root / "config.json").write_text(json.dumps(top))
    (root / "model_index.json").write_text(json.dumps({"use_native_flow_schedule": True}))
    return root


def _dit(*, use_native_flow_schedule: bool = True, max_action_dim: int = MAX_ACTION_DIM) -> Cosmos3DiTSubmodule:
    cfg = Cosmos3Config(
        compile_denoise=False,
        compile_denoise_batched=False,
        latent_channel=8,
        sound_dim=4,
        max_action_dim=max_action_dim,
        use_native_flow_schedule=use_native_flow_schedule,
    )
    sub = Cosmos3DiTSubmodule(
        transformer=_Tower(cfg),
        config=cfg,
        scheduler=UniPCMultistepScheduler.from_config(EDGE_SCHEDULER),
    )
    sub.node_resources = {ATTN: None}
    return sub


def _static_meta(shape, chunk: int, state_rows: int = 0) -> dict:
    t = shape[2]
    tokens = t * (shape[3] // 2) * (shape[4] // 2)
    noisy_action_idx = torch.arange(state_rows, chunk)
    return {
        "num_vision_tokens": tokens,
        "num_noisy_vision_tokens": tokens,
        "rotary": (None, None),
        "vision_token_shapes": [(t, shape[3] // 2, shape[4] // 2)],
        "vision_noisy_frame_indexes": [torch.arange(t)],
        "mse_gen_indexes": None,
        "num_action_tokens": chunk,
        "num_noisy_action_tokens": chunk - state_rows,
        "action_token_shapes": [(chunk, 1, 1)],
        "action_noisy_frame_indexes": [noisy_action_idx],
        "action_mse_gen_indexes": None,
    }


def _setup_action_request(
    sub: Cosmos3DiTSubmodule,
    rid: str,
    *,
    num_steps: int = 6,
    rollout: dict | None = None,
    cfg_branch: bool = True,
    state_rows: int = 0,
    action_chunk_size: int = CHUNK,
    raw_action_dim: int = RAW_DIM,
    inverse_dynamics: bool = False,
    flow_shift: float = 5.0,
):
    st = sub.request_state(rid)
    shape = VIDEO_SHAPE
    total_chunk = action_chunk_size + state_rows
    adim = sub.transformer.action_dim

    vmask = torch.zeros((1, 1, shape[2], 1, 1), dtype=torch.bfloat16)
    if inverse_dynamics:
        vmask[:] = 1.0
    else:
        vmask[:, :, 0] = 1.0

    cond_video = torch.zeros(shape, dtype=torch.bfloat16)
    if inverse_dynamics:
        cond_video.copy_(torch.linspace(-0.5, 0.5, cond_video.numel()).reshape(shape).to(torch.bfloat16))
    else:
        cond_video[:, :, 0] = 0.25

    action_clean_mask = torch.zeros((1, total_chunk, 1), dtype=torch.bfloat16)
    clean_action = torch.zeros((1, total_chunk, adim), dtype=torch.bfloat16)
    if state_rows > 0:
        action_clean_mask[:, :state_rows] = 1.0
        clean_action[:, :state_rows, :raw_action_dim] = 0.35

    cond = _static_meta(shape, total_chunk, state_rows=state_rows)
    st.add_all(
        cond=cond,
        uncond=copy.deepcopy(cond) if cfg_branch else None,
        gs=3.0 if cfg_branch else 1.0,
        scheduler=sub._new_scheduler(num_steps, "cpu", flow_shift=flow_shift),
        latent_shape=shape,
        action_chunk=total_chunk,
        action_dim=adim,
        raw_action_dim=raw_action_dim,
        domain_t=torch.tensor([5]),
        vmask=vmask,
        cond_video_latents=cond_video,
        action_clean_mask=action_clean_mask,
        clean_action=clean_action,
        num_inference_steps=num_steps,
        state_rows=state_rows,
        rollout=rollout,
    )
    return st


def _fwd(rid: str, loop: str, k: int = 0, seed: int = 7):
    return SimpleNamespace(
        request_id=rid,
        rid_handle=rid,
        random_seed=seed,
        dynamic_loop_iter_counts=defaultdict(int, {loop: k}),
    )


def _run_single_loop(
    sub: Cosmos3DiTSubmodule,
    rid: str,
    *,
    walk: str,
    loop: str,
    num_steps: int,
    seed: int = 7,
) -> tuple[list[dict], dict]:
    carried = {}
    outputs = []
    for k in range(num_steps):
        fwd = _fwd(rid, loop, k=k, seed=seed)
        inp = sub._prepare_action_gen(walk, fwd, carried, "cpu")
        ei = SimpleNamespace(
            request_ids=[rid],
            per_request_states=None,
            step=None,
            resources=None,
            per_request_info={rid: fwd},
        )
        out = sub.forward(walk, ei, **inp.tensor_inputs)
        outputs.append(out)
        carried = out
    return outputs, sub.request_states[rid]


def _ref_sample_mean_var(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    t: float,
    t_next: float,
    t_denom: float,
    noise_level: float,
    is_sde_step: bool,
) -> tuple[torch.Tensor, float]:
    x0_hat = x_t - v_t * t
    x1_hat = x_t + v_t * (1.0 - t)
    delta = t - t_next
    w0 = 1.0 - t_next
    if is_sde_step and noise_level > 0.0:
        sigma_sde = noise_level * math.sqrt(t / (1.0 - t_denom))
        w1 = t_next - (sigma_sde ** 2) * delta / (2.0 * t)
        std_t = math.sqrt(delta) * sigma_sde
    else:
        w1 = t_next
        std_t = 0.0
    mu = x0_hat * w0 + x1_hat * w1
    return mu, std_t


def _ref_gaussian_logprob(sample: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        return torch.zeros_like(sample, dtype=torch.float32)
    return -0.5 * ((sample - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# 1. Plain-torch reference parity (<= 1e-6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_native_flow", [False, True])
@pytest.mark.parametrize("num_steps,denoise_ind", [(4, 0), (4, 3), (10, 1), (30, 0), (30, 15)])
@pytest.mark.parametrize("video_solver", ["euler", "hybrid_unipc"])
def test_flow_sde_matches_plain_torch_reference(use_native_flow, num_steps, denoise_ind, video_solver) -> None:
    sub = _dit(use_native_flow_schedule=use_native_flow)
    rollout = {
        "sde_type": "flow_sde",
        "noise_level": 0.5,
        "sde_step": denoise_ind,
        "return_state": True,
        "ignore_last": False,
        "video_solver": video_solver,
    }
    st = _setup_action_request(sub, "r", num_steps=num_steps, rollout=rollout, cfg_branch=True)
    seed = 42
    outputs, st = _run_single_loop(
        sub, "r", walk=C.ACTION_ROLLOUT_GEN_WALK, loop=ACTION_ROLLOUT_GEN_LOOP, num_steps=num_steps, seed=seed,
    )

    # Check rollout_state emitted only on terminal step & unpacks cleanly.
    for k in range(num_steps - 1):
        assert "rollout_state" not in outputs[k]
    assert "rollout_state" in outputs[-1]
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    unpacked = unpack(model.postprocess(outputs[-1]["rollout_state"][0], "tensor"))
    assert unpacked["denoise_ind"] == denoise_ind
    assert unpacked["seed"] == seed
    assert unpacked["chain"].shape == (num_steps + 1, CHUNK, MAX_ACTION_DIM)
    assert unpacked["chain"].dtype == np.float32
    assert unpacked["logprobs"].shape == (CHUNK, MAX_ACTION_DIM)
    assert unpacked["logprobs"].dtype == np.float32
    assert unpacked["logprobs_mask"].shape == (CHUNK, MAX_ACTION_DIM)
    assert unpacked["logprobs_mask"].dtype == np.bool_

    # Standalone plain-torch reference over the exact same initial noise, velocities, and sigmas.
    from diffusers.utils.torch_utils import randn_tensor

    gen = torch.Generator(device="cpu").manual_seed(seed)
    ref_act = randn_tensor((1, CHUNK, MAX_ACTION_DIM), generator=gen, device="cpu", dtype=torch.float32)
    ref_act[..., RAW_DIM:] = 0.0
    ref_lat = randn_tensor(VIDEO_SHAPE, generator=gen, device="cpu", dtype=torch.float32)
    vmask = st["vmask"].to(torch.float32)
    cond_video = st["cond_video_latents"].to(torch.float32)
    ref_lat = vmask * cond_video + (1.0 - vmask) * ref_lat
    clean_mask = st["action_clean_mask"].to(torch.float32)
    clean_act = st["clean_action"].to(torch.float32)

    ref_sampler = sub._new_scheduler(num_steps, "cpu", flow_shift=5.0)
    ref_solver = ref_sampler.initial_state(sub._packed(ref_lat, ref_act))
    ref_reset_step = 0

    sigmas = ref_sampler.host_sigmas
    s_curr, s_next = sigmas[:-1], sigmas[1:]
    s_denom = torch.where(s_curr >= 1.0 - 1e-3, s_next, s_curr)

    ref_chain = [ref_act[0].clone()]
    ref_logprobs = torch.zeros((CHUNK, MAX_ACTION_DIM), dtype=torch.float32)

    for k in range(num_steps):
        vc = _velocity(ref_lat.shape, 1)
        vu = _velocity(ref_lat.shape, 2)
        ac = _velocity(ref_act.shape, 1)
        au = _velocity(ref_act.shape, 2)
        vv = (vu.float() + 3.0 * (vc.float() - vu.float())) * (1.0 - vmask)
        av = (au.float() + 3.0 * (ac.float() - au.float())) * (1.0 - clean_mask)
        av[..., RAW_DIM:] = 0.0

        t = float(s_curr[k])
        t_next = float(s_next[k])
        t_denom = float(s_denom[k])
        nv = vv.numel()
        if video_solver == "hybrid_unipc":
            packed_next, ref_solver = ref_sampler.step(
                sub._packed(vv, av), sub._packed(ref_lat, ref_act), k, ref_solver, reset_step=ref_reset_step,
            )
            ref_lat = (1.0 - vmask) * packed_next[:, :nv].reshape(ref_lat.shape) + vmask * cond_video
        else:
            ref_lat, _ = _ref_sample_mean_var(ref_lat, vv, t, t_next, t_denom, 0.0, False)
            ref_lat = (1.0 - vmask) * ref_lat + vmask * cond_video

        if k == denoise_ind:
            mu_a, std_t = _ref_sample_mean_var(ref_act, av, t, t_next, t_denom, 0.5, True)
            eps = torch.randn(ref_act.shape, generator=gen, device="cpu", dtype=torch.float32)
            ref_act = mu_a + std_t * eps
            ref_act = (1.0 - clean_mask) * ref_act + clean_mask * clean_act
            ref_act[..., RAW_DIM:] = 0.0
            raw_lp = _ref_gaussian_logprob(ref_act[0], mu_a[0], std_t)
            col_mask = torch.arange(MAX_ACTION_DIM) < RAW_DIM
            active = ((1.0 - clean_mask[0, :, 0]) > 0.5).unsqueeze(-1) & col_mask.unsqueeze(0)
            ref_logprobs = torch.where(active, raw_lp, torch.zeros_like(raw_lp))
            if video_solver == "hybrid_unipc":
                ref_solver = {name: torch.zeros_like(val) for name, val in ref_solver.items()}
                ref_reset_step = k + 1
        else:
            if video_solver == "hybrid_unipc":
                ref_act = packed_next[:, nv:].reshape(1, CHUNK, MAX_ACTION_DIM)
            else:
                ref_act, _ = _ref_sample_mean_var(ref_act, av, t, t_next, t_denom, 0.0, False)
            ref_act = (1.0 - clean_mask) * ref_act + clean_mask * clean_act
            ref_act[..., RAW_DIM:] = 0.0
        ref_chain.append(ref_act[0].clone())

    ref_chain_t = torch.stack(ref_chain, dim=0)
    chain_diff = (torch.from_numpy(unpacked["chain"].copy()) - ref_chain_t).abs().max().item()
    lp_diff = (torch.from_numpy(unpacked["logprobs"].copy()) - ref_logprobs).abs().max().item()
    assert chain_diff <= 1e-6, f"chain diff {chain_diff}"
    assert lp_diff <= 1e-6, f"logprob diff {lp_diff}"


# ---------------------------------------------------------------------------
# 2. Video determinism (zero SDE noise on video band)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("video_solver", ["euler", "hybrid_unipc"])
@pytest.mark.parametrize("inverse_dynamics", [False, True])
@pytest.mark.parametrize("noise_level", [0.0, 0.5, 2.0])
def test_flow_sde_keeps_video_deterministic(noise_level, video_solver, inverse_dynamics) -> None:
    sub_ref = _dit()
    _setup_action_request(
        sub_ref,
        "det",
        num_steps=6,
        inverse_dynamics=inverse_dynamics,
        rollout={
            "sde_type": "flow_sde",
            "noise_level": 0.0,
            "sde_step": 2,
            "return_state": True,
            "ignore_last": False,
            "video_solver": video_solver,
        },
    )
    det_outs, st_det = _run_single_loop(
        sub_ref, "det", walk=C.ACTION_ROLLOUT_GEN_WALK, loop=ACTION_ROLLOUT_GEN_LOOP, num_steps=6, seed=11,
    )

    sub = _dit()
    _setup_action_request(
        sub,
        "r",
        num_steps=6,
        inverse_dynamics=inverse_dynamics,
        rollout={
            "sde_type": "flow_sde",
            "noise_level": noise_level,
            "sde_step": 2,
            "return_state": True,
            "ignore_last": False,
            "video_solver": video_solver,
        },
    )
    outs, st = _run_single_loop(
        sub, "r", walk=C.ACTION_ROLLOUT_GEN_WALK, loop=ACTION_ROLLOUT_GEN_LOOP, num_steps=6, seed=11,
    )

    vmask_bool = st["vmask"].to(torch.bool).expand_as(st["cond_video_latents"])
    cond_f32 = st["cond_video_latents"].to(torch.float32)
    for k in range(6):
        assert torch.equal(outs[k]["latents"][0], det_outs[k]["latents"][0]), f"video changed at step {k}"
        assert torch.equal(outs[k]["latents"][0][vmask_bool], cond_f32[vmask_bool])

    if noise_level > 0.0:
        assert not torch.equal(outs[2]["action_latents"][0], det_outs[2]["action_latents"][0])
    else:
        assert torch.equal(outs[2]["action_latents"][0], det_outs[2]["action_latents"][0])
        assert not st["rollout_logprobs_mask"].any()
        assert torch.count_nonzero(st["rollout_logprobs"]) == 0


# ---------------------------------------------------------------------------
# 3. Hybrid UniPC state + order-ramp reset verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("denoise_ind", [0, 2, 5])
def test_hybrid_unipc_resets_solver_state_after_sde_step(denoise_ind) -> None:
    num_steps = 6
    sub = _dit()
    _setup_action_request(
        sub,
        "r",
        num_steps=num_steps,
        rollout={
            "sde_type": "flow_sde",
            "noise_level": 0.5,
            "sde_step": denoise_ind,
            "return_state": True,
            "ignore_last": False,
            "video_solver": "hybrid_unipc",
        },
    )
    outs, st = _run_single_loop(
        sub, "r", walk=C.ACTION_ROLLOUT_GEN_WALK, loop=ACTION_ROLLOUT_GEN_LOOP, num_steps=num_steps, seed=19,
    )

    # Immediately after denoise_ind, solver tensors must be zeroed and reset_step == denoise_ind + 1.
    assert torch.count_nonzero(outs[denoise_ind][MODEL_OUTPUTS][0]) == 0
    assert torch.count_nonzero(outs[denoise_ind][LAST_SAMPLE][0]) == 0
    assert st["unipc_reset_step"] == denoise_ind + 1

    if denoise_ind + 1 >= num_steps:
        return

    # Construct an independent fresh _UniPCSampler over the tail sigmas full_sigmas[denoise_ind + 1:].
    full_sampler = st["scheduler"]
    tail_sched = UniPCMultistepScheduler.from_config(full_sampler.config)
    tail_sched.sigmas = full_sampler.host_sigmas[denoise_ind + 1 :].clone()
    tail_sched.timesteps = torch.tensor(full_sampler.host_timesteps[denoise_ind + 1 :], dtype=torch.int64)
    tail_sampler = _UniPCSampler(tail_sched, "cpu")

    lat_tail = outs[denoise_ind]["latents"][0].clone()
    act_tail = outs[denoise_ind]["action_latents"][0].clone()
    tail_solver = initial_state(sub._packed(lat_tail, act_tail))
    vmask = st["vmask"].to(torch.float32)
    cond_video = st["cond_video_latents"].to(torch.float32)
    clean_mask = st["action_clean_mask"].to(torch.float32)
    clean_act = st["clean_action"].to(torch.float32)

    for k in range(denoise_ind + 1, num_steps):
        vc = _velocity(lat_tail.shape, 1)
        vu = _velocity(lat_tail.shape, 2)
        ac = _velocity(act_tail.shape, 1)
        au = _velocity(act_tail.shape, 2)
        vv = (vu.float() + 3.0 * (vc.float() - vu.float())) * (1.0 - vmask)
        av = (au.float() + 3.0 * (ac.float() - au.float())) * (1.0 - clean_mask)
        av[..., RAW_DIM:] = 0.0

        nv = vv.numel()
        packed_next, tail_solver = tail_sampler.step(
            sub._packed(vv, av),
            sub._packed(lat_tail, act_tail),
            k - (denoise_ind + 1),
            tail_solver,
        )
        lat_tail = (1.0 - vmask) * packed_next[:, :nv].reshape(lat_tail.shape) + vmask * cond_video
        act_tail = (1.0 - clean_mask) * packed_next[:, nv:].reshape(act_tail.shape) + clean_mask * clean_act
        act_tail[..., RAW_DIM:] = 0.0

        assert torch.equal(outs[k]["latents"][0], lat_tail), f"tail video mismatch at step {k}"
        assert torch.equal(outs[k]["action_latents"][0], act_tail), f"tail action mismatch at step {k}"


# ---------------------------------------------------------------------------
# 4. action_clean_mask, raw_action_dim masking, conditioning re-pinning, state_rows slicing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state_rows", [0, 1])
@pytest.mark.parametrize("sde_type", ["flow_sde", "flow_ode"])
def test_action_clean_mask_and_raw_action_dim_masking(state_rows, sde_type) -> None:
    num_steps = 4
    sub = _dit(max_action_dim=64)
    st = _setup_action_request(
        sub,
        "r",
        num_steps=num_steps,
        state_rows=state_rows,
        action_chunk_size=16,
        raw_action_dim=10,
        rollout={
            "sde_type": sde_type,
            "noise_level": 0.7,
            "sde_step": 1,
            "return_state": True,
            "ignore_last": False,
            "video_solver": "hybrid_unipc",
        },
    )
    outs, st = _run_single_loop(
        sub, "r", walk=C.ACTION_ROLLOUT_GEN_WALK, loop=ACTION_ROLLOUT_GEN_LOOP, num_steps=num_steps, seed=123,
    )

    clean_f32 = st["clean_action"].to(torch.float32)
    for k in range(num_steps):
        act = outs[k]["action_latents"][0]
        # Padding columns 10..63 stay identically zero.
        assert torch.count_nonzero(act[..., 10:]) == 0
        # Clean rows (state_rows) remain bit-identical to clean_action.
        if state_rows:
            assert torch.equal(act[:, :state_rows], clean_f32[:, :state_rows])

    # Unpacked rollout_state slices [state_rows:] so horizon == action_chunk_size == 16.
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    unpacked = unpack(model.postprocess(outs[-1]["rollout_state"][0], "tensor"))
    act_bytes = model.postprocess(
        outs[-1]["action_latents"][0],
        "action",
        {"action_mode": "policy", "raw_action_dim": 10, **({"robot_state": [0.35] * 10} if state_rows else {})},
    )
    act_arr = np.frombuffer(act_bytes, dtype=np.float32).reshape(-1, 64)
    assert act_arr.shape == (16, 64)
    assert unpacked["chain"].shape == (num_steps + 1, 16, 64)
    assert np.array_equal(unpacked["chain"][-1], act_arr)
    assert unpacked["logprobs"].shape == (16, 64)
    assert unpacked["logprobs_mask"].shape == (16, 64)

    if sde_type == "flow_ode":
        assert unpacked["denoise_ind"] == -1
        assert not unpacked["logprobs_mask"].any()
        assert np.count_nonzero(unpacked["logprobs"]) == 0
    else:
        assert unpacked["denoise_ind"] == 1
        expected_mask = np.zeros((16, 64), dtype=np.bool_)
        expected_mask[:, :10] = True
        assert np.array_equal(unpacked["logprobs_mask"], expected_mask)
        assert np.all(unpacked["logprobs"][:, 10:] == 0.0)
        assert np.all(np.isfinite(unpacked["logprobs"][:, :10]))
        assert np.any(unpacked["logprobs"][:, :10] != 0.0)
        if state_rows:
            # Internal pre-slice row 0 has clean mask -> False / 0.0.
            assert not st["rollout_logprobs_mask"][0].any()
            assert torch.count_nonzero(st["rollout_logprobs"][0]) == 0


# ---------------------------------------------------------------------------
# 5. Flag-off bit-identity & return_state=False walk selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_branch", [True, False])
@pytest.mark.parametrize("state_rows", [0, 1])
def test_flag_off_is_bitwise_identical_to_baseline(cfg_branch, state_rows) -> None:
    num_steps = 6
    sub = _dit()
    _setup_action_request(
        sub, "off", num_steps=num_steps, rollout=None, cfg_branch=cfg_branch, state_rows=state_rows,
    )
    outs, _ = _run_single_loop(
        sub, "off", walk=C.ACTION_GEN_WALK, loop=ACTION_GEN_LOOP, num_steps=num_steps, seed=9,
    )

    # Compare against direct baseline _action_scheduler_step calls.
    sub_base = _dit()
    st_base = _setup_action_request(
        sub_base, "base", num_steps=num_steps, rollout=None, cfg_branch=cfg_branch, state_rows=state_rows,
    )
    carried = {}
    for k in range(num_steps):
        assert "rollout_state" not in outs[k]
        inp = sub_base._prepare_action_gen(
            C.ACTION_GEN_WALK, _fwd("base", ACTION_GEN_LOOP, k=k, seed=9), carried, "cpu",
        )
        lat = inp.tensor_inputs["latents"]
        act = inp.tensor_inputs["action_latents"]
        solver = {name: inp.tensor_inputs[name] for name in sub_base._solver_state}
        vc, ac = _velocity(lat.shape, 1), _velocity(act.shape, 1)
        if cfg_branch:
            vv = _velocity(lat.shape, 2).float() + st_base["gs"] * (vc.float() - _velocity(lat.shape, 2).float())
            av = _velocity(act.shape, 2).float() + st_base["gs"] * (ac.float() - _velocity(act.shape, 2).float())
        else:
            vv, av = vc.float(), ac.float()
        ref_lat, ref_act, ref_solver = sub_base._action_scheduler_step(st_base, lat, act, vv, av, k, solver)
        assert torch.equal(outs[k]["latents"][0], ref_lat)
        assert torch.equal(outs[k]["action_latents"][0], ref_act)
        for name in sub_base._solver_state:
            assert torch.equal(outs[k][name][0], ref_solver[name])
        carried = {"latents": [ref_lat], "action_latents": [ref_act], **{n: [t] for n, t in ref_solver.items()}}

    # Verify walk selection & return_state=False routing.
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    signals = {"text_inputs": [], "image_inputs": [object()]}
    base_mk = {"action_mode": "policy", "domain_name": "libero", "raw_action_dim": 10}
    assert model.walk_plan("default", ["image", "text"], ["action"], signals, base_mk)[1].walk == C.ACTION_GEN_WALK
    assert (
        model.walk_plan(
            "default",
            ["image", "text"],
            ["action"],
            signals,
            {**base_mk, "rollout": {"sde_type": "flow_sde", "return_state": False}},
        )[1].walk
        == C.ACTION_GEN_WALK
    )
    assert (
        model.walk_plan(
            "default",
            ["image", "text"],
            ["action"],
            signals,
            {**base_mk, "rollout": {"sde_type": "flow_sde", "return_state": True}},
        )[1].walk
        == C.ACTION_ROLLOUT_GEN_WALK
    )


# ---------------------------------------------------------------------------
# 6. action_view_point, rollout validation, load_decoded_image, non-action/distilled refusals
# ---------------------------------------------------------------------------


def test_action_view_point_and_rollout_validation(tmp_path) -> None:
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    assert {"action_view_point", "rollout"} <= model.request_kwargs()
    assert "action_view_point" not in Cosmos3Model.PREPROCESS_ONLY_KWARGS
    assert "rollout" not in Cosmos3Model.PREPROCESS_ONLY_KWARGS

    # load_decoded_image matches Model.load_image scaling.
    img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    loaded = model.load_decoded_image(img, "cpu")
    assert loaded.data.dtype == torch.float32
    assert torch.equal(loaded.data, img.float() / 255.0)

    # Valid action_view_point + rollout resolves cleanly with zero ignored kwargs.
    mk = {
        "action_mode": "policy",
        "domain_name": "libero",
        "raw_action_dim": 10,
        "action_view_point": "ego_view",
        "num_inference_steps": 4,
        "rollout": {
            "sde_type": "flow_sde",
            "noise_level": 0.5,
            "sde_step": "random",
            "return_state": True,
            "ignore_last": True,
            "video_solver": "hybrid_unipc",
        },
    }
    assert set(mk) - model.request_kwargs() - set(model.ENGINE_KWARGS) - set(model.LENGTH_KWARGS) == set()
    p = model._resolve_gen_params(mk, ["image", "text"], ["action"])
    assert p["action_view_point"] == "ego_view"
    assert p["rollout"]["sde_type"] == "flow_sde"
    assert p["rollout"]["ignore_last"] is True

    # Invalid action_view_point.
    for bad_vp in ("", "   ", 123, True):
        with pytest.raises(ValueError, match="action_view_point"):
            model._resolve_gen_params({**mk, "action_view_point": bad_vp}, ["image", "text"], ["action"])
    with pytest.raises(ValueError, match="action_view_point"):
        model._resolve_gen_params({"action_view_point": "ego_view"}, ["text"], ["image"])

    # Invalid rollout payloads.
    base = {"action_mode": "policy", "domain_name": "libero", "raw_action_dim": 10, "num_inference_steps": 4}
    bad_rollouts = [
        "flow_sde",
        {"sde_type": "flow_sde", "unknown_key": 1},
        {"sde_type": "bogus"},
        {},
        {"sde_type": "flow_sde", "noise_level": -0.1},
        {"sde_type": "flow_sde", "noise_level": True},
        {"sde_type": "flow_sde", "noise_level": float("nan")},
        {"sde_type": "flow_sde", "noise_level": float("inf")},
        {"sde_type": "flow_sde", "noise_level": "0.5"},
        {"sde_type": "flow_sde", "sde_step": -1},
        {"sde_type": "flow_sde", "sde_step": 4},
        {"sde_type": "flow_sde", "sde_step": 3, "ignore_last": True},
        {"sde_type": "flow_sde", "sde_step": True},
        {"sde_type": "flow_sde", "sde_step": 1.5},
        {"sde_type": "flow_sde", "sde_step": "first"},
        {"sde_type": "flow_sde", "return_state": "true"},
        {"sde_type": "flow_sde", "ignore_last": 1},
        {"sde_type": "flow_sde", "video_solver": "ddim"},
    ]
    for bad_r in bad_rollouts:
        with pytest.raises(ValueError):
            model._resolve_gen_params({**base, "rollout": bad_r}, ["image", "text"], ["action"])

    # Refusals on all non-action walks (image, video, forward_dynamics, reasoner text).
    valid_r = {"sde_type": "flow_sde", "noise_level": 0.5}
    with pytest.raises(ValueError, match="rollout"):
        model._resolve_gen_params({"rollout": valid_r}, ["text"], ["image"])
    with pytest.raises(ValueError, match="rollout"):
        model._resolve_gen_params({"num_frames": 17, "rollout": valid_r}, ["text"], ["video"])
    with pytest.raises(ValueError, match="rollout"):
        model._resolve_gen_params(
            {
                "action_mode": "forward_dynamics",
                "domain_name": "libero",
                "raw_action_dim": 10,
                "action": [[0.0] * 10] * 4,
                "rollout": valid_r,
            },
            ["image", "text"],
            ["video"],
        )

    edge_model = Cosmos3Model(model_path_hf=str(_fake_edge_dir(tmp_path / "edge")), skip_weight_loading=True)
    with pytest.raises(ValueError, match="rollout"):
        edge_model.walk_plan("default", ["text"], ["text"], {"text_inputs": []}, {"rollout": valid_r})
    with pytest.raises(ValueError, match="rollout"):
        edge_model.process_prompt("hi", ["text"], ["text"], rollout=valid_r)

    # Refusal on distilled checkpoint.
    distilled_model = Cosmos3Model(
        model_path_hf=str(_fake_distilled_dir(tmp_path / "distilled")), skip_weight_loading=True,
    )
    with pytest.raises(ValueError, match="distilled"):
        distilled_model._resolve_gen_params(
            {"action_mode": "policy", "domain_name": "libero", "raw_action_dim": 10, "rollout": valid_r},
            ["image", "text"],
            ["action"],
        )


# ---------------------------------------------------------------------------
# 7. Cosmos3 LIBERO configs & _resolve_gen_params defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename,expected_model", [
    ("cosmos3_libero.yaml", "cosmos3"),
    ("cosmos3_edge_libero.yaml", "cosmos3_edge"),
])
def test_libero_configs_load_and_resolve(tmp_path, filename, expected_model) -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / filename).read_text())
    assert cfg["model"] == expected_model
    assert cfg["model"] in MODEL_REGISTRY
    assert cfg["max_seq_len"] == 8192
    assert cfg["resources"]["kv"]["max_num_pages"] == 1024

    mk = cfg["model_kwargs"]
    assert "model_path_hf" not in mk
    assert set(mk) <= set(Cosmos3Config.__dataclass_fields__)
    assert mk["enable_sound"] is False
    assert mk["num_inference_steps_action"] == 30
    assert mk["guidance_scale_action"] == 1.0
    assert mk["flow_shift_action"] == 5.0
    assert mk["policy_action_chunk_size"] == 16
    assert mk["policy_conditioning_fps"] == 20.0
    assert mk["policy_domain_name"] == "libero"
    assert mk["inverted_action_dims"] == []
    assert mk["image_size_default"] == [320, 192]
    assert mk["video_size_default"] == [320, 192]
    assert mk["compile_vae_encode_size"] == [320, 192]

    node_names = {n for g in cfg["node_groups"] for n in g["node_names"]}
    assert node_names == {"dit", "vae_encoder", "vae_decoder"}

    if expected_model == "cosmos3_edge":
        assert mk["enable_reasoner"] is False
        assert mk["conditioning_resize"] == "aspect_crop"
        repo = str(_fake_edge_dir(tmp_path / "edge"))
    else:
        repo = "unused"

    model = Cosmos3Model(model_path_hf=repo, skip_weight_loading=True, **mk)
    assert not model._reasoner_enabled()
    assert not model._sound_serving_enabled()

    p = model._resolve_gen_params(
        {"action_mode": "policy", "raw_action_dim": 10},
        ["image", "text"],
        ["action"],
    )
    assert p["domain_id"] == 5
    assert p["action_chunk_size"] == 16
    assert p["num_frames"] == 17
    assert p["num_inference_steps"] == 30
    assert p["flow_shift"] == 5.0
    assert p["fps"] == 20.0
    assert (p["width"], p["height"]) == (320, 192)


# ---------------------------------------------------------------------------
# 8. Batch = 8 concurrent scheduling, isolation & speculative step guard
# ---------------------------------------------------------------------------


def test_batched_eight_rollout_requests() -> None:
    num_steps = 5
    sub = _dit()
    assert sub.max_batch_size(C.ACTION_GEN_WALK) >= 8
    assert sub.max_batch_size(C.ACTION_ROLLOUT_GEN_WALK) >= 8

    seeds = [100 + i for i in range(8)]
    sde_steps = ["random", 0, "random", 2, 4, "random", 1, "random"]
    rids = [f"r{i}" for i in range(8)]

    for rid, sde_step in zip(rids, sde_steps, strict=True):
        _setup_action_request(
            sub,
            rid,
            num_steps=num_steps,
            rollout={
                "sde_type": "flow_sde",
                "noise_level": 0.5,
                "sde_step": sde_step,
                "return_state": True,
                "ignore_last": False,
                "video_solver": "hybrid_unipc",
            },
        )

    carried = {rid: {} for rid in rids}
    batched_outs = {rid: [] for rid in rids}
    for k in range(num_steps):
        prepared = []
        fwds = {}
        for rid, seed in zip(rids, seeds, strict=True):
            fwd = _fwd(rid, ACTION_ROLLOUT_GEN_LOOP, k=k, seed=seed)
            fwds[rid] = fwd
            prepared.append(sub._prepare_action_gen(C.ACTION_ROLLOUT_GEN_WALK, fwd, carried[rid], "cpu"))

        batch = SimpleNamespace(request_ids=rids, graph_walk=C.ACTION_ROLLOUT_GEN_WALK)
        assert sub.can_batch(batch, prepared) is True

        ei = SimpleNamespace(
            request_ids=rids,
            per_request_states=None,
            step=None,
            resources=None,
            per_request_info=fwds,
        )
        pre = sub.preprocess(C.ACTION_ROLLOUT_GEN_WALK, ei, prepared)
        out = sub.forward_batched(C.ACTION_ROLLOUT_GEN_WALK, ei, **pre)
        for rid in rids:
            carried[rid] = out[rid]
            batched_outs[rid].append(out[rid])

    # Verify speculative step guard (k >= num_steps) does not mutate rollout_chain or logprobs.
    saved_chains = {rid: sub.request_states[rid]["rollout_chain"].clone() for rid in rids}
    saved_lps = {rid: sub.request_states[rid]["rollout_logprobs"].clone() for rid in rids}
    spec_fwds = {
        rid: _fwd(rid, ACTION_ROLLOUT_GEN_LOOP, k=num_steps, seed=seed)
        for rid, seed in zip(rids, seeds, strict=True)
    }
    spec_ei = SimpleNamespace(
        request_ids=rids,
        per_request_states=None,
        step=None,
        resources=None,
        per_request_info=spec_fwds,
    )
    spec_pre = {
        "latents": {rid: carried[rid]["latents"][0] for rid in rids},
        "action_latents": {rid: carried[rid]["action_latents"][0] for rid in rids},
        **{name: {rid: carried[rid][name][0] for rid in rids} for name in sub._solver_state},
    }
    spec_out = sub.forward_batched(C.ACTION_ROLLOUT_GEN_WALK, spec_ei, **spec_pre)
    for rid in rids:
        assert "rollout_state" not in spec_out[rid]
        assert torch.equal(spec_out[rid]["action_latents"][0], carried[rid]["action_latents"][0])
        assert torch.equal(sub.request_states[rid]["rollout_chain"], saved_chains[rid])
        assert torch.equal(sub.request_states[rid]["rollout_logprobs"], saved_lps[rid])

    # Each row in the 8-batch must match its solo bs=1 run bit-for-bit, and distinct seeds diverge.
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    unpacked_rows = []
    for rid, seed, sde_step in zip(rids, seeds, sde_steps, strict=True):
        sub_solo = _dit()
        _setup_action_request(
            sub_solo,
            rid,
            num_steps=num_steps,
            rollout={
                "sde_type": "flow_sde",
                "noise_level": 0.5,
                "sde_step": sde_step,
                "return_state": True,
                "ignore_last": False,
                "video_solver": "hybrid_unipc",
            },
        )
        solo_outs, _ = _run_single_loop(
            sub_solo, rid, walk=C.ACTION_ROLLOUT_GEN_WALK, loop=ACTION_ROLLOUT_GEN_LOOP, num_steps=num_steps, seed=seed,
        )
        for k in range(num_steps):
            assert torch.equal(batched_outs[rid][k]["action_latents"][0], solo_outs[k]["action_latents"][0])
            assert torch.equal(batched_outs[rid][k]["latents"][0], solo_outs[k]["latents"][0])
        u_batch = unpack(model.postprocess(batched_outs[rid][-1]["rollout_state"][0], "tensor"))
        u_solo = unpack(model.postprocess(solo_outs[-1]["rollout_state"][0], "tensor"))
        assert u_batch["seed"] == seed == u_solo["seed"]
        assert u_batch["denoise_ind"] == u_solo["denoise_ind"]
        assert np.array_equal(u_batch["chain"], u_solo["chain"])
        assert np.array_equal(u_batch["logprobs"], u_solo["logprobs"])
        assert np.array_equal(u_batch["logprobs_mask"], u_solo["logprobs_mask"])
        unpacked_rows.append(u_batch)

    assert len({tuple(r["chain"].ravel().tolist()) for r in unpacked_rows}) == 8


@pytest.mark.parametrize("raw_seed,expected_i64", [
    (2**63 - 1, 2**63 - 1),
    (2**63, -(2**63)),
    (-(2**63), -(2**63)),
])
def test_pack_rollout_state_wraps_seed_to_signed_int64(raw_seed: int, expected_i64: int) -> None:
    sub = _dit()
    _setup_action_request(
        sub,
        "r-seed",
        num_steps=2,
        rollout={"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": 0, "return_state": True},
    )
    outputs, _ = _run_single_loop(
        sub, "r-seed", walk=C.ACTION_ROLLOUT_GEN_WALK, loop=ACTION_ROLLOUT_GEN_LOOP, num_steps=2, seed=raw_seed,
    )
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    unpacked = unpack(model.postprocess(outputs[-1]["rollout_state"][0], "tensor"))
    assert -(2**63) <= unpacked["seed"] <= 2**63 - 1
    assert unpacked["seed"] == expected_i64
