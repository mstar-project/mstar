"""Policy requests that send the robot's current state as ``robot_state``, the
way NVIDIA's RoboLab server conditions the released DROID policies: the state is
one more action token, clean at video frame 0 in front of the predicted chunk,
in the policy's convention (DROID's 1 - gripper, set per deployment), and the
reply carries only the predicted rows, back in the robot's convention. CPU only,
no weights."""

from __future__ import annotations

import json
import math
import types

import numpy as np
import pytest
import torch

from mstar.model.base import ProcessPromptOutput
from mstar.model.cosmos3.components.packing import action_start_frame_offset, resolve_action_domain_id
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import ACTION_GEN_WALK, COND_LABEL, Cosmos3DiTSubmodule
from mstar.model.cosmos3.tests.test_rotary_table import stand_in_rotary

STATE = [0.1, -0.2, 0.3, -2.0, 0.05, 1.9, 0.8, 0.25]  # DROID: 7 joint positions, gripper position
POLICY = {"action_mode": "policy", "domain_name": "droid_lerobot"}
MODALITIES = (["image", "text"], ["action"])


def _model(inverted=(7,)) -> Cosmos3Model:
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    model.config.inverted_action_dims = tuple(inverted)
    return model


def _submit(model: Cosmos3Model, mk: dict):
    """What the server does with a request: process_prompt resolves the params
    and puts the conditioning on a tensor edge with only its width in the kwargs,
    the conductor resolves the step metadata from the kwargs without the values,
    and the data worker keeps the full kwargs for postprocess."""
    model._resolve_gen_params(mk, *MODALITIES)
    out = model._action_prompt_output({}, mk)
    tensors, meta = (out.new_input_tensors, out.metadata) if isinstance(out, ProcessPromptOutput) else (out, {})
    kwargs = {**mk, **meta}
    conductor = {k: v for k, v in kwargs.items() if k not in Cosmos3Model.PREPROCESS_ONLY_KWARGS}
    return tensors, model._resolve_gen_params(conductor, *MODALITIES), kwargs


def _dit(model: Cosmos3Model) -> Cosmos3DiTSubmodule:
    dit = Cosmos3DiTSubmodule(transformer=None, config=model.config, scheduler=None)
    dit.transformer = types.SimpleNamespace(
        proj_in=types.SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)),
        action_dim=model.config.max_action_dim,
        _rotary=stand_in_rotary(model.config),
    )
    dit._new_scheduler = lambda *args, **kwargs: None
    return dit


def _prefill(dit: Cosmos3DiTSubmodule, md: dict, tensors: dict, rid: int):
    fwd_info = types.SimpleNamespace(rid_handle=rid, request_id=str(rid))
    dit._prepare_action_prefill(
        fwd_info, md, tensors, [1, 2, 3], [1, 2], md["height"], md["width"], md["fps"],
        md["guidance_scale"], md["num_inference_steps"], "cpu",
    )
    return dit.request_states[rid]


def test_robot_state_request_params() -> None:
    model = _model()
    _, p, _ = _submit(model, {**POLICY, "action_chunk_size": 32, "robot_state": STATE})
    # The chunk is the rows the reply carries; the state is a token in front at frame 0.
    assert p["action_state_rows"] == 1 and p["action_chunk_size"] == 32 and p["num_frames"] == 33
    assert p["raw_action_dim"] == 8  # the state's width
    # A robot_state without a mode is a policy request.
    _, p, _ = _submit(model, {"domain_name": "droid_lerobot", "action_chunk_size": 32, "robot_state": STATE})
    assert p["action_mode"] == "policy" and p["action_state_rows"] == 1
    # Without a state nothing changes.
    p = model._resolve_gen_params({**POLICY, "raw_action_dim": 8, "action_chunk_size": 32}, *MODALITIES)
    assert "action_state_rows" not in p and p["num_frames"] == 33 and p["action_chunk_size"] == 32

    with pytest.raises(ValueError, match="action_chunk_size \\+ 1"):
        model._resolve_gen_params({**POLICY, "action_chunk_size": 32, "num_frames": 32, "robot_state": STATE},
                                  *MODALITIES)
    with pytest.raises(ValueError, match="raw_action_dim=7 values"):
        model._resolve_gen_params({**POLICY, "raw_action_dim": 7, "robot_state": STATE}, *MODALITIES)
    with pytest.raises(ValueError, match="one row"):
        model._resolve_gen_params({**POLICY, "robot_state": [STATE]}, *MODALITIES)
    with pytest.raises(ValueError, match="input of a policy request"):
        model._resolve_gen_params({**POLICY, "action_mode": "inverse_dynamics", "robot_state": STATE}, *MODALITIES)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "0.1", True])
def test_robot_state_must_be_finite_numbers(bad) -> None:
    """A NaN in one joint would come back in every predicted row (they all
    attend to the state token), so it is a 400 instead."""
    state = list(STATE)
    state[3] = bad
    with pytest.raises(ValueError, match="robot_state"):
        _submit(_model(), {**POLICY, "robot_state": state})
    with pytest.raises(ValueError, match="robot_state"):
        _submit(_model(), {**POLICY, "robot_state": "0.1, 0.2"})


def _policy_checkpoint(root, policy):
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text(json.dumps({"_class_name": "Cosmos3OmniTransformer"}))
    (root / "checkpoint.json").write_text(json.dumps({"policy": policy} if policy else {}))
    return Cosmos3Model(model_path_hf=str(root), skip_weight_loading=True)


def test_policy_defaults_come_from_the_checkpoint(tmp_path) -> None:
    """The Edge DROID checkpoint's checkpoint.json carries the chunk, fps and
    domain its policy serves, so a request sends only the observation, the
    prompt and the robot_state."""
    model = _policy_checkpoint(
        tmp_path / "edge", {"action_chunk_size": 32, "conditioning_fps": 15.0, "domain_name": "droid_lerobot"},
    )
    p = model._resolve_gen_params({"size": "736x544", "robot_state": STATE}, *MODALITIES)
    assert p["action_mode"] == "policy" and p["action_chunk_size"] == 32 and p["num_frames"] == 33
    assert p["fps"] == 15.0 and p["domain_id"] == resolve_action_domain_id(None, "droid_lerobot")
    assert p["raw_action_dim"] == 8 and p["action_state_rows"] == 1
    # The request wins.
    p = model._resolve_gen_params(
        {"robot_state": STATE, "action_chunk_size": 16, "fps": 10, "domain_id": 3}, *MODALITIES,
    )
    assert p["action_chunk_size"] == 16 and p["fps"] == 10.0 and p["domain_id"] == 3
    # Only policy requests take them.
    p = model._resolve_gen_params(
        {"action_mode": "inverse_dynamics", "domain_id": 3, "raw_action_dim": 8}, *MODALITIES,
    )
    assert p["action_chunk_size"] == 16 and p["fps"] == model.config.fps
    # A checkpoint without them (Nano DROID's checkpoint.json is empty) keeps the
    # generic defaults, so its yaml sets them.
    nano = _policy_checkpoint(tmp_path / "nano", None)
    assert nano.config.policy_action_chunk_size is None
    with pytest.raises(ValueError, match="domain_id"):
        nano._resolve_gen_params({"robot_state": STATE}, *MODALITIES)


def test_robot_state_reaches_the_dit_in_the_policys_convention() -> None:
    model = _model()
    tensors, p, kwargs = _submit(model, {**POLICY, "action_chunk_size": 32, "robot_state": STATE})
    expected = list(STATE)
    expected[7] = 1.0 - STATE[7]  # the policy's 1 - gripper
    assert torch.allclose(tensors["action_inputs"][0], torch.tensor([expected]))
    assert kwargs["robot_state_dim"] == 8 and "robot_state_dim" in Cosmos3Model.INTERNAL_KWARGS
    # Without the deployment's flip the state goes as sent.
    tensors, _, _ = _submit(_model(inverted=()), {**POLICY, "action_chunk_size": 32, "robot_state": STATE})
    assert torch.allclose(tensors["action_inputs"][0], torch.tensor([STATE]))
    # `action` keeps meaning the action chunk: on a policy request it conditions nothing.
    tensors, p, _ = _submit(model, {**POLICY, "raw_action_dim": 8, "action_chunk_size": 32, "action": [STATE]})
    assert "action_inputs" not in tensors and "action_state_rows" not in p
    # A forward-dynamics `action` goes in the same convention, so a policy reply
    # sent back as `action` reaches the DiT with the gripper the policy predicted.
    fd = model._action_prompt_output({}, {"action_mode": "forward_dynamics", "action": [STATE, STATE]})
    assert torch.allclose(fd.new_input_tensors["action_inputs"][0], torch.tensor([expected, expected]))


def test_prefill_puts_the_state_in_front_of_the_chunk() -> None:
    model = _model()
    dit = _dit(model)
    common = {**POLICY, "size": "64x64", "fps": 15, "action_chunk_size": 32}
    tensors, md, _ = _submit(model, {**common, "robot_state": STATE})
    st = _prefill(dit, md, tensors, 0)
    assert st["action_chunk"] == 33  # the state token and 32 predicted rows
    clean = st["action_clean_mask"][0, :, 0]
    assert clean[0] == 1 and clean[1:].sum() == 0
    assert torch.allclose(st["clean_action"][0, 0, :8], tensors["action_inputs"][0][0])
    assert st["clean_action"][0, 0, 8:].abs().sum() == 0 and st["clean_action"][0, 1:].abs().sum() == 0
    for static in (st["cond"], st["uncond"]):
        assert static["action_noisy_frame_indexes"][0].tolist() == list(range(1, 33))
        assert static["num_noisy_action_tokens"] == 32

    tensors, md, _ = _submit(model, {**common, "raw_action_dim": 8})
    plain = _prefill(dit, md, tensors, 1)
    assert plain["action_chunk"] == 32
    assert plain["action_clean_mask"].sum() == 0 and plain["clean_action"].abs().sum() == 0
    assert plain["cond"]["action_noisy_frame_indexes"][0].tolist() == list(range(32))

    # The state token sits at video frame 0's time, and the predicted rows keep
    # the positions they have without a state (the prefill slims the position
    # ids away, so build the statics again).
    def static(tokens, rows):
        offset = action_start_frame_offset(tokens, 33)
        return dit._build_action_static([1, 2, 3], 64, 64, 33, tokens, "policy", 15.0, 15.0, offset, "cpu", rows)

    with_state, without = static(33, 1), static(32, 0)
    t_state = with_state["action_mrope_ids"][0]
    assert t_state[0] == with_state["vision_mrope_ids"][0, 0]
    assert torch.equal(t_state[1:], without["action_mrope_ids"][0])


def test_kv_layout_counts_the_state_token() -> None:
    model = _model()
    ids = [1, 2, 3]
    _, with_state, _ = _submit(model, {**POLICY, "size": "64x64", "action_chunk_size": 32, "robot_state": STATE})
    _, without, _ = _submit(model, {**POLICY, "size": "64x64", "action_chunk_size": 32, "raw_action_dim": 8})
    scratch = [model._gen_layout(p, ids, ids, MODALITIES[0], {})[COND_LABEL].scratch for p in (with_state, without)]
    assert scratch[0] == scratch[1] + 1


def test_reply_carries_the_predicted_rows_in_the_robots_convention() -> None:
    model = _model()
    chunk = torch.rand(1, 33, 64)
    chunk[..., 8:] = 0  # the action dims past raw_action_dim are padding
    _, _, kwargs = _submit(model, {**POLICY, "action_chunk_size": 32, "robot_state": STATE})
    out = np.frombuffer(model.postprocess(chunk, "action", kwargs), dtype=np.float32).reshape(-1, 64)
    expected = chunk[0, 1:].clone()
    expected[:, 7] = 1 - expected[:, 7]  # the gripper goes back to the robot's convention
    assert np.allclose(out, expected.numpy()) and not out[:, 8:].any()
    # No state: nothing to drop, the rows still go back to the robot's convention.
    _, _, kwargs = _submit(model, {**POLICY, "action_chunk_size": 32, "raw_action_dim": 8})
    out = np.frombuffer(model.postprocess(chunk[:, :32], "action", kwargs), dtype=np.float32).reshape(-1, 64)
    expected = chunk[0, :32].clone()
    expected[:, 7] = 1 - expected[:, 7]
    assert np.allclose(out, expected.numpy())
    # An inverse-dynamics reply goes back to the robot's convention too.
    mk = {**POLICY, "action_mode": "inverse_dynamics", "action_chunk_size": 32, "raw_action_dim": 8}
    out = np.frombuffer(model.postprocess(chunk[:, :32], "action", mk), dtype=np.float32).reshape(-1, 64)
    assert np.allclose(out, expected.numpy())


def test_guidance_interval_is_an_action_default(tmp_path) -> None:
    """The deployment's guidance_interval_action applies to action requests that
    send none, and the Edge DROID warmup takes it, so its first request does not
    compile the conditional-only step."""
    from pathlib import Path

    import yaml

    import mstar

    yaml_path = Path(mstar.__file__).resolve().parent.parent / "configs" / "cosmos3_edge_droid.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())
    mk = cfg["model_kwargs"]
    assert mk["guidance_interval_action"] == [960, 1001] and mk["inverted_action_dims"] == [7]
    assert set(mk) <= set(Cosmos3Config.__dataclass_fields__)
    warmup = cfg["warmup_requests"][0]
    model = _policy_checkpoint(
        tmp_path / "edge", {"action_chunk_size": 32, "conditioning_fps": 15.0, "domain_name": "droid_lerobot"},
    )
    model.config.guidance_interval_action = tuple(mk["guidance_interval_action"])
    p = model._resolve_gen_params(warmup["model_kwargs"], ["image", "text"], warmup["output_modalities"])
    assert p["guidance_interval"] == (960.0, 1001.0) and p["action_state_rows"] == 1
    # A request's interval wins; image and video requests keep their own.
    p = model._resolve_gen_params({"robot_state": STATE, "guidance_interval": [0, 1001]}, *MODALITIES)
    assert p["guidance_interval"] == (0.0, 1001.0)
    assert model._resolve_gen_params({}, ["text"], ["image"])["guidance_interval"] == (400.0, 1000.0)
    assert "guidance_interval" not in model._resolve_gen_params({"num_frames": 17}, ["text"], ["video"])


def test_action_guidance_interval() -> None:
    """A guidance_interval applies to action requests: steps outside it run the
    conditional branch alone, and the declared step agrees."""
    model = _model()
    dit = _dit(model)
    tensors, md, _ = _submit(
        model, {**POLICY, "action_chunk_size": 32, "size": "64x64", "fps": 15,
                "guidance_scale": 3.0, "guidance_interval": [960, 1001], "robot_state": STATE},
    )
    st = _prefill(dit, md, tensors, 0)
    assert st["guidance_interval"] == (960.0, 1001.0)
    st.add("scheduler", types.SimpleNamespace(timesteps=torch.tensor([1000.0, 937.5, 833.3, 625.0])))
    st.add("domain_t", torch.tensor([8]))

    labels = []

    def denoise(attn, static, latents, action_latents, vts, ats, domain, label):
        labels.append(label)
        return torch.zeros_like(latents), torch.zeros_like(action_latents)

    dit._denoise_action = denoise
    dit.batched_cfg = False
    dit._action_scheduler_step = lambda st, lat, act, v, a, step, solver: (lat, act, solver)
    latents, action_latents = torch.zeros(1, 2), torch.zeros(1, 33, 64)
    for step, branches in ((0, 2), (1, 1), (3, 1)):
        labels.clear()
        dit._forward_action_gen(None, st, step, latents, action_latents)
        assert len(labels) == branches, step
        assert dit._step_info(ACTION_GEN_WALK, st, step).cfg_active == (branches == 2)


@pytest.mark.parametrize("steps, guided", [((1, 3), False), ((0, 3), True)])
def test_batched_action_step_outside_the_interval_runs_the_conditional_branch(steps, guided) -> None:
    """A batch none of whose rows is inside its guidance interval plans and runs
    the conditional branch alone, as a single request does; one row inside it
    keeps both branches for the batch, and only that row is guided."""
    from mstar.model.cosmos3.submodules import ACTION_GEN_LOOP, UNCOND_LABEL

    model = _model()
    dit = _dit(model)
    timesteps = types.SimpleNamespace(timesteps=torch.tensor([1000.0, 937.5, 833.3, 625.0]))
    rids = [0, 1]
    for rid in rids:
        tensors, md, _ = _submit(
            model, {**POLICY, "action_chunk_size": 32, "size": "64x64", "fps": 15,
                    "guidance_scale": 3.0, "guidance_interval": [960, 1001], "robot_state": STATE},
        )
        st = _prefill(dit, md, tensors, rid)
        st.add_all(scheduler=timesteps, domain_t=torch.tensor([8]))
    infos = [dit._step_info(ACTION_GEN_WALK, dit.request_states[rid], k) for rid, k in zip(rids, steps, strict=True)]
    declared = dit.declare_step(ACTION_GEN_WALK, rids, [types.SimpleNamespace(input_seq_len=4, resource_step_info=i)
                                                       for i in infos])
    labels = {seg.label for seg in declared.segments}
    assert labels == ({COND_LABEL, UNCOND_LABEL} if guided else {COND_LABEL})

    seen, combined = [], []

    def batched(reqs, label, attn, with_cfg):
        seen.append(with_cfg)
        branch = (torch.zeros(1, 2), torch.ones(1, 33, 64))
        return [[branch, (torch.zeros(1, 2), torch.zeros(1, 33, 64))] if with_cfg else [branch] for _ in reqs]

    def step(st, lat, act, video_v, action_v, k, solver):
        combined.append(float(action_v[0, 1, 0]))
        return lat, act, solver

    dit.transformer.denoise_step_action_batched = batched
    dit._action_scheduler_step = step
    dit._solver_state = ()
    ei = types.SimpleNamespace(
        request_ids=rids, per_request_states=None, step=None, resources={"attn": None},
        per_request_info={rid: types.SimpleNamespace(dynamic_loop_iter_counts={ACTION_GEN_LOOP: k})
                          for rid, k in zip(rids, steps, strict=True)},
    )
    zeros = {rid: torch.zeros(1, 2) for rid in rids}
    dit._forward_batched_action(ACTION_GEN_WALK, ei, zeros, {rid: torch.zeros(1, 33, 64) for rid in rids}, {})
    assert seen == [guided]
    # cond 1, uncond 0, guidance 3: a guided row gets 3, the others the conditional 1
    assert combined == ([3.0, 1.0] if guided else [1.0, 1.0])
