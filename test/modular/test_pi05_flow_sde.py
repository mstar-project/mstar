"""CPU tests for Pi0.5 opt-in Flow-SDE, fp32 log-probs, ROLLOUT_WALK, CUDA graph safety, and RLinf parity."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from mstar.api_server.openpi import unpack as openpi_unpack
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import TensorPointerInfo
from mstar.model.pi05.components.action_expert import Pi05ActionExpert, Pi05AdaRMSNorm, Pi05TimeMLP
from mstar.model.pi05.components.flow_matching import (
    flow_sde_sigmas,
    flow_sde_timesteps,
    gaussian_logprob,
    normalize_rollout_config,
    sample_mean_var,
    sincos_timestep_embedding,
)
from mstar.model.pi05.config import CHUNK_WALK, LLM_KV, ROLLOUT_WALK, Pi05Config
from mstar.model.pi05.pi05_model import Pi05Model
from mstar.model.pi05.submodules import (
    Pi05ActionSubmodule,
    Pi05LLMSubmodule,
    Pi05ViTEncoderSubmodule,
)
from mstar.model.submodule_base import ModelInputsFromEngine


class _RecordingExpert(torch.nn.Module):
    """Lightweight CPU action-expert stand-in with real Pi05AdaRMSNorms."""

    modulations = Pi05ActionExpert.modulations

    def __init__(self, hidden: int, num_layers: int = 2):
        super().__init__()

        def norm(slot: int) -> Pi05AdaRMSNorm:
            return Pi05AdaRMSNorm(hidden, cond_dim=hidden, eps=1e-6, slot=slot)

        self.layers = torch.nn.ModuleList(
            torch.nn.ModuleDict({"input_layernorm": norm(0), "post_attention_layernorm": norm(1)})
            for _ in range(num_layers)
        )
        self.norm = norm(0)
        for mod in self.modules():
            if isinstance(mod, Pi05AdaRMSNorm):
                torch.nn.init.normal_(mod.dense.weight, std=0.1)
                torch.nn.init.normal_(mod.dense.bias, std=0.1)

    def forward(self, query_sequence, layer_modulations, final_modulation):
        # Mix modulation norm lightly so time conditioning affects velocity non-trivially.
        scale = final_modulation[self.norm.slot].mean()
        return query_sequence * (1.0 + 0.05 * torch.tanh(scale))


def _make_action_submodule(
    *,
    num_flow_steps: int = 3,
    action_horizon: int = 4,
    action_dim: int = 6,
    hidden: int = 16,
    velocity_scale: float = 0.5,
    proj_dtype: torch.dtype = torch.float32,
) -> Pi05ActionSubmodule:
    torch.manual_seed(17)
    config = Pi05Config(
        action_hidden_size=hidden,
        action_horizon=action_horizon,
        action_dim=action_dim,
        num_flow_steps=num_flow_steps,
    )
    in_proj = torch.nn.Linear(action_dim, hidden, bias=True).to(proj_dtype)
    out_proj = torch.nn.Linear(hidden, action_dim, bias=True).to(proj_dtype)
    with torch.no_grad():
        eye_in = torch.eye(hidden, action_dim, dtype=proj_dtype)
        in_proj.weight.copy_(eye_in)
        in_proj.bias.zero_()
        eye_out = velocity_scale * torch.eye(action_dim, hidden, dtype=proj_dtype)
        out_proj.weight.copy_(eye_out)
        out_proj.bias.zero_()
    return Pi05ActionSubmodule(
        _RecordingExpert(hidden),
        in_proj,
        out_proj,
        Pi05TimeMLP(hidden),
        config,
    )


def _make_model(*, num_flow_steps: int = 3, action_horizon: int = 4, action_dim: int = 6) -> Pi05Model:
    model = Pi05Model("unused", skip_weight_loading=True, num_flow_steps=num_flow_steps,
                      action_horizon=action_horizon, action_dim=action_dim)
    return model


def _pointer() -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[1], dtype=torch.float32, nbytes=0, address=0, stride=[], uuid=0,
        source_session_id="", source_entity="",
    )


# ----------------------------------------------------------------------
# Plain-torch RLinf reference (from rlinf/models/embodiment/openpi/sampling/rl_sampler.py)
# ----------------------------------------------------------------------


def _rlinf_ref_sample_mean_var(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    sigma_i: torch.Tensor,
    is_sde: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0_pred = x_t - v_t * t
    x1_pred = x_t + v_t * (1.0 - t)
    if is_sde:
        x0_weight = 1.0 - (t - dt)
        x1_weight = (t - dt) - sigma_i**2 * dt / (2.0 * t)
        x_t_std = torch.sqrt(dt) * sigma_i
    else:
        x0_weight = 1.0 - (t - dt)
        x1_weight = t - dt
        x_t_std = torch.zeros((), dtype=torch.float32)
    x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
    return x_t_mean, x_t_std


def _rlinf_ref_gaussian_logprob(
    x_next: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor
) -> torch.Tensor:
    if float(x_std) == 0.0:
        return torch.zeros_like(x_next, dtype=torch.float32)
    return -torch.log(x_std) - 0.5 * math.log(2.0 * math.pi) - 0.5 * ((x_next - x_mean) / x_std) ** 2


def _rlinf_ref_denoise(
    sub: Pi05ActionSubmodule,
    noisy_actions: torch.Tensor,
    step_noise: torch.Tensor,
    *,
    sde_type: str,
    noise_level: float,
    denoise_ind: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_steps = sub.config.num_flow_steps
    timesteps = torch.cat(
        [
            torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        ]
    )
    denom_timesteps = torch.where(timesteps == 1.0, timesteps[1], timesteps)
    sigmas = noise_level * torch.sqrt(timesteps / (1.0 - denom_timesteps))

    fraction = sub._get_timestep_emb_fraction()
    time_emb = sincos_timestep_embedding(
        timesteps[:num_steps],
        dim=sub.config.action_hidden_size,
        fraction=fraction,
        output_buffer=torch.empty(num_steps, sub.config.action_hidden_size, dtype=torch.float32),
        min_period=sub.config.timestep_min_period,
        max_period=sub.config.timestep_max_period,
    )
    modulations = sub.action_expert.modulations(sub.time_mlp(time_emb))

    x_t = noisy_actions.to(torch.float32)
    chain = [x_t]
    logprobs = torch.zeros_like(x_t, dtype=torch.float32)
    in_w = sub.action_in_proj.weight.dtype
    out_w = sub.action_out_proj.weight.dtype
    for idx in range(num_steps):
        suffix_out = sub.action_expert(
            query_sequence=sub.action_in_proj(x_t.to(in_w)),
            layer_modulations=[mod[idx] for mod in modulations.per_layer],
            final_modulation=modulations.final[idx],
        )
        v_t = sub.action_out_proj(suffix_out.to(out_w)).to(torch.float32)
        t = timesteps[idx]
        dt = t - timesteps[idx + 1]
        is_sde = sde_type == "flow_sde" and idx == denoise_ind
        x_mean, x_std = _rlinf_ref_sample_mean_var(x_t, v_t, t, dt, sigmas[idx], is_sde)
        x_next = x_mean + x_std * step_noise.to(torch.float32)
        logprobs = logprobs + _rlinf_ref_gaussian_logprob(x_next, x_mean, x_std)
        x_t = x_next
        chain.append(x_t)

    mask = torch.full_like(x_t, bool(sde_type == "flow_sde" and noise_level > 0.0), dtype=torch.bool)
    return x_t, torch.stack(chain, dim=0), logprobs, mask


# ----------------------------------------------------------------------
# Contract 1 validation
# ----------------------------------------------------------------------


def test_normalize_rollout_config_defaults_and_passthrough():
    assert normalize_rollout_config(None, 3) is None
    assert normalize_rollout_config({"sde_type": "flow_sde"}, 3) == {
        "sde_type": "flow_sde",
        "noise_level": 0.5,
        "sde_step": "random",
        "return_state": True,
        "ignore_last": False,
    }
    assert normalize_rollout_config(
        {"sde_type": "flow_ode", "noise_level": 0, "sde_step": 2, "return_state": False, "ignore_last": True},
        3,
    ) == {
        "sde_type": "flow_ode",
        "noise_level": 0.0,
        "sde_step": 2,
        "return_state": False,
        "ignore_last": True,
    }


@pytest.mark.parametrize(
    "bad_rollout",
    [
        "flow_sde",
        [],
        {},
        {"sde_type": "euler"},
        {"sde_type": 123},
        {"sde_type": "flow_sde", "unknown_key": 1},
        {"sde_type": "flow_sde", "noise_level": -0.1},
        {"sde_type": "flow_sde", "noise_level": True},
        {"sde_type": "flow_sde", "noise_level": float("nan")},
        {"sde_type": "flow_sde", "noise_level": float("inf")},
        {"sde_type": "flow_sde", "noise_level": "0.5"},
        {"sde_type": "flow_sde", "sde_step": -1},
        {"sde_type": "flow_sde", "sde_step": 3},
        {"sde_type": "flow_sde", "sde_step": True},
        {"sde_type": "flow_sde", "sde_step": 1.5},
        {"sde_type": "flow_sde", "sde_step": "first"},
        {"sde_type": "flow_sde", "return_state": 1},
        {"sde_type": "flow_sde", "ignore_last": "false"},
    ],
)
def test_rollout_config_rejects_invalid_values(bad_rollout):
    with pytest.raises(ValueError):
        normalize_rollout_config(bad_rollout, 3)

    model = _make_model(num_flow_steps=3)
    with pytest.raises(ValueError):
        model.process_prompt(
            prompt=None,
            input_modalities=["image", "text"],
            output_modalities=["action"],
            tensors={"image_inputs": [torch.zeros(3, 4, 4)]},
            tokenized_prompt=[2, 5, 6],
            rollout=bad_rollout,
        )
    with pytest.raises(ValueError):
        model.walk_plan(
            "default",
            ["image", "text"],
            ["action"],
            {"image_inputs": [_pointer()], "text_inputs": [_pointer()]},
            {"rollout": bad_rollout},
        )


# ----------------------------------------------------------------------
# Plain-torch RLinf parity & edge cases
# ----------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [1, 3, 10])
@pytest.mark.parametrize("sde_type", ["flow_sde", "flow_ode"])
@pytest.mark.parametrize("noise_level", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("proj_dtype", [torch.float32, torch.bfloat16])
def test_pi05_flow_sde_matches_rlinf_reference_within_1e_6(
    num_steps, sde_type, noise_level, proj_dtype
):
    sub = _make_action_submodule(num_flow_steps=num_steps, proj_dtype=proj_dtype)
    model = _make_model(num_flow_steps=num_steps, action_horizon=4, action_dim=6)

    for sde_step in range(num_steps):
        rollout_cfg = normalize_rollout_config(
            {"sde_type": sde_type, "noise_level": noise_level, "sde_step": sde_step, "return_state": True},
            num_steps,
        )
        fwd_info = CurrentForwardPassInfo(
            request_id=f"req-{sde_step}",
            graph_walk=ROLLOUT_WALK,
            fwd_index=0,
            random_seed=42 + sde_step,
            max_tokens=1,
            step_metadata={"rollout": rollout_cfg},
        )
        node_inp = sub.prepare_inputs(ROLLOUT_WALK, fwd_info, {"prefix_done": [torch.zeros(1)]})
        engine_inp = ModelInputsFromEngine(
            request_ids=[fwd_info.request_id], per_request_info={fwd_info.request_id: fwd_info}
        )
        pre = sub.preprocess(ROLLOUT_WALK, engine_inp, [node_inp])
        out = sub.forward(ROLLOUT_WALK, engine_inp, **pre)

        actions = out["actions"][0]
        raw_bytes = model.postprocess(out["rollout"][0], modality="tensor")
        unpacked = openpi_unpack(raw_bytes)

        ref_actions, ref_chain, ref_lp, ref_mask = _rlinf_ref_denoise(
            sub,
            node_inp.tensor_inputs["noisy_actions"],
            node_inp.tensor_inputs["step_noise"],
            sde_type=sde_type,
            noise_level=noise_level,
            denoise_ind=sde_step,
        )

        assert actions.dtype == torch.float32
        assert torch.isfinite(actions).all()
        assert torch.isfinite(torch.from_numpy(unpacked["chain"])).all()
        assert torch.isfinite(torch.from_numpy(unpacked["logprobs"])).all()

        torch.testing.assert_close(actions, ref_actions, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.from_numpy(unpacked["chain"]), ref_chain, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.from_numpy(unpacked["logprobs"]), ref_lp, atol=1e-6, rtol=1e-6)
        assert np.array_equal(unpacked["logprobs_mask"], ref_mask.numpy())
        assert unpacked["denoise_ind"] == sde_step
        assert unpacked["seed"] == fwd_info.random_seed

        if sde_type == "flow_ode" or noise_level == 0.0:
            assert np.all(unpacked["logprobs"] == 0.0)
            assert not np.any(unpacked["logprobs_mask"])
        else:
            assert np.all(unpacked["logprobs_mask"])


def test_flow_sde_math_edge_cases_and_ignore_last():
    # t=1.0 (index 0) and N=1 produce finite sigmas, means, stds, and logprobs.
    ts_1 = flow_sde_timesteps(1)
    sig_1 = flow_sde_sigmas(ts_1, 0.5)
    assert torch.isfinite(sig_1).all()
    assert float(sig_1[0]) == pytest.approx(0.5)

    ts_3 = flow_sde_timesteps(3)
    sig_3 = flow_sde_sigmas(ts_3, 0.5)
    assert float(sig_3[0]) == pytest.approx(0.5 * math.sqrt(3.0))

    x = torch.zeros(2, 3)
    v = torch.ones(2, 3)
    mean, std = sample_mean_var(x, v, ts_3[0], ts_3[1], sig_3[0], True)
    assert torch.isfinite(mean).all() and float(std) > 0.0
    lp = gaussian_logprob(mean, mean, std)
    assert torch.isfinite(lp).all()

    # Deterministic step std == 0 yields exact 0.0 logprob.
    mean_ode, std_ode = sample_mean_var(x, v, ts_3[0], ts_3[1], sig_3[0], False)
    assert float(std_ode) == 0.0
    assert torch.equal(gaussian_logprob(mean_ode + 1.0, mean_ode, std_ode), torch.zeros_like(mean_ode))

    # ignore_last=True never picks num_steps - 1 when num_steps > 1.
    sub_3 = _make_action_submodule(num_flow_steps=3)
    rollout_ignore = normalize_rollout_config({"sde_type": "flow_sde", "ignore_last": True}, 3)
    picked = set()
    for seed in range(64):
        info = CurrentForwardPassInfo(
            f"r-{seed}", ROLLOUT_WALK, 0, seed, 1, step_metadata={"rollout": rollout_ignore}
        )
        inp = sub_3.prepare_inputs(ROLLOUT_WALK, info, {})
        picked.add(inp.kwargs["rollout"]["denoise_ind"])
    assert picked <= {0, 1}
    assert 2 not in picked

    # ignore_last=True with num_steps == 1 still falls back safely to 0.
    sub_1 = _make_action_submodule(num_flow_steps=1)
    rollout_ignore_1 = normalize_rollout_config({"sde_type": "flow_sde", "ignore_last": True}, 1)
    inp_1 = sub_1.prepare_inputs(
        ROLLOUT_WALK,
        CurrentForwardPassInfo("r1", ROLLOUT_WALK, 0, 7, 1, step_metadata={"rollout": rollout_ignore_1}),
        {},
    )
    assert inp_1.kwargs["rollout"]["denoise_ind"] == 0


# ----------------------------------------------------------------------
# Flag-off bit-identity & per-request generator determinism
# ----------------------------------------------------------------------


def test_flag_off_is_bit_identical_to_base_denoise():
    sub = _make_action_submodule(num_flow_steps=4)
    model = _make_model(num_flow_steps=4, action_horizon=4, action_dim=6)

    fwd_info = CurrentForwardPassInfo("r-off", CHUNK_WALK, 0, 99, 1, step_metadata={})
    inp = sub.prepare_inputs(CHUNK_WALK, fwd_info, {})
    assert set(inp.tensor_inputs) == {"noisy_actions"}
    assert inp.kwargs == {}

    engine_inp = ModelInputsFromEngine(request_ids=["r-off"], per_request_info={"r-off": fwd_info})
    pre = sub.preprocess(CHUNK_WALK, engine_inp, [inp])
    assert set(pre) == {"noisy_actions", "fraction"}

    out = sub.forward(CHUNK_WALK, engine_inp, **pre)
    assert set(out) == {"actions"}

    ref_actions = sub._denoise(inp.tensor_inputs["noisy_actions"], fraction=pre["fraction"])
    assert torch.equal(out["actions"][0], ref_actions)
    assert model.postprocess(out["actions"][0], "action") == model.postprocess(ref_actions, "action")

    plan = model.walk_plan(
        "default", ["image", "text"], ["action"],
        {"image_inputs": [_pointer()], "text_inputs": [_pointer()]}, {},
    )
    assert [step.walk for step in plan] == [CHUNK_WALK]
    assert plan[0].step_metadata == {}


def test_per_request_generator_determinism_independent_of_global_rng():
    sub = _make_action_submodule(num_flow_steps=5)
    rollout = normalize_rollout_config({"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": "random"}, 5)

    def _run(seed: int, global_seed: int):
        torch.manual_seed(global_seed)
        info = CurrentForwardPassInfo(
            f"r-{seed}", ROLLOUT_WALK, 0, seed, 1, step_metadata={"rollout": rollout}
        )
        inp = sub.prepare_inputs(ROLLOUT_WALK, info, {})
        eng = ModelInputsFromEngine(request_ids=[info.request_id], per_request_info={info.request_id: info})
        pre = sub.preprocess(ROLLOUT_WALK, eng, [inp])
        out = sub.forward(ROLLOUT_WALK, eng, **pre)
        rollout_bytes = _make_model(num_flow_steps=5, action_horizon=4, action_dim=6).postprocess(
            out["rollout"][0], "tensor"
        )
        unpacked = openpi_unpack(rollout_bytes)
        return out["actions"][0], unpacked

    a1, u1 = _run(seed=1234, global_seed=1)
    a2, u2 = _run(seed=1234, global_seed=99999)
    assert torch.equal(a1, a2)
    assert np.array_equal(u1["chain"], u2["chain"])
    assert np.array_equal(u1["logprobs"], u2["logprobs"])
    assert u1["denoise_ind"] == u2["denoise_ind"]
    assert u1["seed"] == u2["seed"] == 1234

    a3, u3 = _run(seed=5678, global_seed=1)
    assert not torch.equal(a1, a3)
    assert not np.array_equal(u1["chain"], u3["chain"])


# ----------------------------------------------------------------------
# Batching & forward_batched vs forward parity
# ----------------------------------------------------------------------


def test_forward_batched_matches_per_request_forward_with_heterogeneous_rollout():
    sub = _make_action_submodule(num_flow_steps=3)
    model = _make_model(num_flow_steps=3, action_horizon=4, action_dim=6)

    cfgs = [
        normalize_rollout_config({"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": 0, "return_state": True}, 3),
        normalize_rollout_config({"sde_type": "flow_sde", "noise_level": 0.8, "sde_step": 2, "return_state": True}, 3),
        normalize_rollout_config({"sde_type": "flow_ode", "noise_level": 0.5, "sde_step": 1, "return_state": False}, 3),
    ]
    infos = [
        CurrentForwardPassInfo(f"r{i}", ROLLOUT_WALK, 0, 100 + i, 1, step_metadata={"rollout": cfg})
        for i, cfg in enumerate(cfgs)
    ]
    inps = [sub.prepare_inputs(ROLLOUT_WALK, info, {}) for info in infos]
    assert sub.can_batch(None, inps)

    eng_batch = ModelInputsFromEngine(
        request_ids=[info.request_id for info in infos],
        per_request_info={info.request_id: info for info in infos},
    )
    pre_batch = sub.preprocess(ROLLOUT_WALK, eng_batch, inps)
    batched_out = sub.forward_batched(ROLLOUT_WALK, eng_batch, **pre_batch)

    for info, inp in zip(infos, inps, strict=True):
        eng_single = ModelInputsFromEngine(request_ids=[info.request_id], per_request_info={info.request_id: info})
        pre_single = sub.preprocess(ROLLOUT_WALK, eng_single, [inp])
        single_out = sub.forward(ROLLOUT_WALK, eng_single, **pre_single)

        rid = info.request_id
        torch.testing.assert_close(batched_out[rid]["actions"][0], single_out["actions"][0], atol=1e-6, rtol=1e-6)
        if info.step_metadata["rollout"]["return_state"]:
            u_b = openpi_unpack(model.postprocess(batched_out[rid]["rollout"][0], "tensor"))
            u_s = openpi_unpack(model.postprocess(single_out["rollout"][0], "tensor"))
            np.testing.assert_allclose(u_b["chain"], u_s["chain"], atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(u_b["logprobs"], u_s["logprobs"], atol=1e-6, rtol=1e-6)
            assert np.array_equal(u_b["logprobs_mask"], u_s["logprobs_mask"])
            assert u_b["denoise_ind"] == u_s["denoise_ind"]
            assert u_b["seed"] == u_s["seed"]
        else:
            assert "rollout" not in batched_out[rid]
            assert "rollout" not in single_out

    # can_batch rejects mixing rollout and non-rollout inputs.
    off_info = CurrentForwardPassInfo("off", CHUNK_WALK, 0, 1, 1, step_metadata={})
    off_inp = sub.prepare_inputs(CHUNK_WALK, off_info, {})
    assert not sub.can_batch(None, [inps[0], off_inp])
    assert sub.can_batch(None, [off_inp])


# ----------------------------------------------------------------------
# CUDA graph safety & end-to-end Pi05Model contract
# ----------------------------------------------------------------------


def test_cuda_graph_safety_structural_invariants():
    cfg = Pi05Config()
    vit = Pi05ViTEncoderSubmodule(torch.nn.Identity(), cfg)
    llm = Pi05LLMSubmodule(torch.nn.Embedding(16, cfg.hidden_size), torch.nn.Identity(), cfg)
    action = _make_action_submodule()

    vit_cfgs = vit.get_accelerator_graph_configs(torch.device("cpu"))
    llm_cfgs = llm.get_accelerator_graph_configs(torch.device("cpu"))
    act_cfgs = action.get_accelerator_graph_configs(torch.device("cpu"))

    assert set(vit_cfgs[0].replay_graph_walks) == {CHUNK_WALK, ROLLOUT_WALK}
    assert set(llm_cfgs[0].replay_graph_walks) == {CHUNK_WALK, ROLLOUT_WALK}
    assert act_cfgs[0].capture_graph_walk == CHUNK_WALK
    assert act_cfgs[0].replay_graph_walks == [CHUNK_WALK]
    assert act_cfgs[0].additional_key_info is None

    assert action.split_batches_by_capture_key(CHUNK_WALK) is True
    assert action.split_batches_by_capture_key(ROLLOUT_WALK) is False

    off = CurrentForwardPassInfo("off", CHUNK_WALK, 0, 1, 1, step_metadata={})
    on = CurrentForwardPassInfo(
        "on", CHUNK_WALK, 0, 1, 1,
        step_metadata={"rollout": normalize_rollout_config({"sde_type": "flow_sde", "return_state": False}, 3)},
    )
    assert action.cg_key_info(CHUNK_WALK, {"off": off}) is None
    assert action.cg_key_info(CHUNK_WALK, {"on": on}) == "rollout"
    assert action.cg_key_info(CHUNK_WALK, {"off": off, "on": on}) == "rollout"
    assert action.cg_key_info(ROLLOUT_WALK, {"on": on}) is None


def test_pi05_model_end_to_end_rollout_walk_and_msgpack_roundtrip():
    model = _make_model(num_flow_steps=3, action_horizon=4, action_dim=6)
    sub = _make_action_submodule(num_flow_steps=3, action_horizon=4, action_dim=6)

    # return_state=True -> ROLLOUT_WALK, Layout spans on ROLLOUT_WALK, emits actions + rollout tensor.
    rollout_true = {"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": 1, "return_state": True}
    prompt_out = model.process_prompt(
        prompt=None,
        input_modalities=["image", "text"],
        output_modalities=["action"],
        tensors={"image_inputs": [torch.zeros(3, 4, 4)] * 3},
        camera_mask=[True, True, False],
        tokenized_prompt=[2, 7, 8, 0],
        tokenized_prompt_mask=[True, True, True, False],
        rollout=rollout_true,
    )
    layout = prompt_out.metadata["state_layout"][LLM_KV]["main"]
    assert [s.walk for s in layout.spans] == [ROLLOUT_WALK, ROLLOUT_WALK]
    assert layout.spans[0].length == 2 * 256
    assert layout.spans[1].length == 3

    conductor_kwargs = {
        "active_camera_mask": prompt_out.metadata["active_camera_mask"],
        "rollout": rollout_true,
    }
    plan = model.walk_plan(
        "default", ["image", "text"], ["action"],
        {"image_inputs": [_pointer()], "text_inputs": [_pointer()]},
        conductor_kwargs,
    )
    assert [s.walk for s in plan] == [ROLLOUT_WALK]

    fwd_info = CurrentForwardPassInfo(
        "e2e", ROLLOUT_WALK, 0, 2026, 1, step_metadata=plan[0].step_metadata
    )
    node_inp = sub.prepare_inputs(ROLLOUT_WALK, fwd_info, {})
    eng = ModelInputsFromEngine(request_ids=["e2e"], per_request_info={"e2e": fwd_info})
    out = sub.forward(ROLLOUT_WALK, eng, **sub.preprocess(ROLLOUT_WALK, eng, [node_inp]))

    action_bytes = model.postprocess(out["actions"][0], "action")
    actions_arr = np.frombuffer(action_bytes, dtype=np.float32).reshape(4, 6)
    assert actions_arr.shape == (4, 6)

    rollout_bytes = model.postprocess(out["rollout"][0], "tensor")
    decoded = openpi_unpack(rollout_bytes)
    assert set(decoded) == {"chain", "denoise_ind", "logprobs", "logprobs_mask", "seed"}
    assert decoded["chain"].shape == (4, 4, 6)
    assert decoded["chain"].dtype == np.float32
    assert decoded["logprobs"].shape == (4, 6)
    assert decoded["logprobs"].dtype == np.float32
    assert decoded["logprobs_mask"].shape == (4, 6)
    assert decoded["logprobs_mask"].dtype == np.bool_
    assert decoded["denoise_ind"] == 1
    assert decoded["seed"] == 2026
    np.testing.assert_allclose(decoded["chain"][-1], actions_arr, atol=1e-6)

    # return_state=False -> CHUNK_WALK, Layout spans on CHUNK_WALK, omits rollout output.
    rollout_false = {"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": 1, "return_state": False}
    prompt_out_false = model.process_prompt(
        prompt=None,
        input_modalities=["image", "text"],
        output_modalities=["action"],
        tensors={"image_inputs": [torch.zeros(3, 4, 4)] * 2},
        tokenized_prompt=[2, 7],
        rollout=rollout_false,
    )
    assert [s.walk for s in prompt_out_false.metadata["state_layout"][LLM_KV]["main"].spans] == [
        CHUNK_WALK,
        CHUNK_WALK,
    ]
    plan_false = model.walk_plan(
        "default", ["image", "text"], ["action"],
        {"image_inputs": [_pointer()], "text_inputs": [_pointer()]},
        {"rollout": rollout_false},
    )
    assert [s.walk for s in plan_false] == [CHUNK_WALK]
    fwd_false = CurrentForwardPassInfo(
        "e2e-false", CHUNK_WALK, 0, 2026, 1, step_metadata=plan_false[0].step_metadata
    )
    inp_false = sub.prepare_inputs(CHUNK_WALK, fwd_false, {})
    eng_false = ModelInputsFromEngine(request_ids=["e2e-false"], per_request_info={"e2e-false": fwd_false})
    out_false = sub.forward(CHUNK_WALK, eng_false, **sub.preprocess(CHUNK_WALK, eng_false, [inp_false]))
    assert set(out_false) == {"actions"}
    torch.testing.assert_close(out_false["actions"][0], out["actions"][0], atol=1e-6, rtol=1e-6)
