"""Policy requests that send the robot's current state as ``robot_state``, the
way NVIDIA's RoboLab server conditions the released DROID policies: the state
is one more action token, clean at video frame 0 in front of the predicted
rows, and the reply leaves it out. Also the deployment's policy defaults, the
gripper flip and the guidance interval on the action walk. CPU only, no
weights."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch

from mstar.model.base import ProcessPromptOutput
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import (
    ACTION_GEN_WALK,
    ATTN,
    COND_LABEL,
    UNCOND_LABEL,
    Cosmos3DiTSubmodule,
    GenStepInfo,
)
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine

JOINTS = [0.1, -0.2, 0.3, -2.0, 0.05, 1.9, 0.8]
STATE = JOINTS + [0.25]  # the robot's own gripper position
MODALITIES = (["image", "text"], ["action"])
# What configs/cosmos3_droid.yaml sets (Edge reads the first three from its checkpoint.json).
DROID = dict(
    policy_action_chunk_size=32, policy_conditioning_fps=15.0, policy_domain_name="droid_lerobot",
    policy_size=[736, 544], policy_flip_dims=[7],
)
POLICY = {"action_mode": "policy"}


def _model(**overrides) -> Cosmos3Model:
    return Cosmos3Model(model_path_hf="unused", skip_weight_loading=True, **overrides)


def _submit(model: Cosmos3Model, mk: dict):
    """What the server does with a request: process_prompt puts the state on a
    tensor edge and its width in the kwargs, the conductor resolves the step
    metadata from the kwargs without the values, and the data worker keeps the
    full kwargs for postprocess."""
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
    )
    dit._new_scheduler = lambda *args, **kwargs: None
    return dit


def _prefill(dit: Cosmos3DiTSubmodule, md: dict, tensors: dict, rid):
    fwd_info = types.SimpleNamespace(rid_handle=rid, request_id=str(rid))
    dit._prepare_action_prefill(
        fwd_info, md, tensors, [1, 2, 3], [1, 2], md["height"], md["width"], md["fps"],
        md["guidance_scale"], md["num_inference_steps"], "cpu",
    )
    return dit.request_states[rid]


def test_policy_request_takes_the_deployment_defaults() -> None:
    model = _model(**DROID)
    _, p, _ = _submit(model, {**POLICY, "robot_state": STATE})
    assert p["action_chunk_size"] == 32 and p["num_frames"] == 33 and p["action_state_rows"] == 1
    assert p["fps"] == 15.0 and (p["width"], p["height"]) == (736, 544)
    assert p["domain_id"] == 8 and p["raw_action_dim"] == 8

    # Without a state the chunk is the same 32 rows and frames.
    p = model._resolve_gen_params({**POLICY, "raw_action_dim": 8}, *MODALITIES)
    assert p["action_chunk_size"] == 32 and p["num_frames"] == 33 and "action_state_rows" not in p

    # The request's own values win.
    _, p, _ = _submit(model, {**POLICY, "robot_state": STATE, "action_chunk_size": 16, "fps": 24,
                              "size": "832x480", "domain_id": 3})
    assert p["action_chunk_size"] == 16 and p["num_frames"] == 17
    assert p["fps"] == 24.0 and (p["width"], p["height"]) == (832, 480) and p["domain_id"] == 3

    # The defaults are the policy's: other modes and other deployments keep theirs.
    p = model._resolve_gen_params(
        {"action_mode": "inverse_dynamics", "raw_action_dim": 8, "domain_id": 8}, *MODALITIES,
    )
    assert p["action_chunk_size"] == 16 and p["fps"] == 24.0 and (p["width"], p["height"]) == (832, 480)
    bare = _model()
    _, p, _ = _submit(bare, {**POLICY, "robot_state": STATE, "domain_name": "droid_lerobot"})
    assert p["action_chunk_size"] == 16 and p["num_frames"] == 17 and p["raw_action_dim"] == 8
    with pytest.raises(ValueError, match="domain_id"):
        bare._resolve_gen_params({**POLICY, "robot_state": STATE}, *MODALITIES)


def test_checkpoint_json_sets_the_policy_defaults(tmp_path) -> None:
    from mstar.model.cosmos3.config import Cosmos3Config

    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text("{}")
    (tmp_path / "checkpoint.json").write_text(
        '{"policy": {"action_chunk_size": 32, "conditioning_fps": 15.0, "domain_name": "droid_lerobot"}}'
    )
    cfg = Cosmos3Config.from_pretrained(tmp_path)
    assert (cfg.policy_action_chunk_size, cfg.policy_conditioning_fps, cfg.policy_domain_name) == (
        32, 15.0, "droid_lerobot",
    )
    (tmp_path / "checkpoint.json").write_text("{}")
    cfg = Cosmos3Config.from_pretrained(tmp_path)
    assert cfg.policy_action_chunk_size is None and cfg.policy_domain_name is None


@pytest.mark.parametrize("state, match", [
    (JOINTS[:3] + [math.nan] + JOINTS[4:] + [0.25], "finite"),
    (JOINTS + [math.inf], "finite"),
    (JOINTS + [-math.inf], "finite"),
    (JOINTS + [1e39], "finite"),  # overflows float32
    ([STATE], "list of 1 to 64 numbers"),
    ([], "list of 1 to 64 numbers"),
    ([0.0] * 65, "list of 1 to 64 numbers"),
    (JOINTS + [True], "list of 1 to 64 numbers"),
    (JOINTS + ["0.5"], "list of 1 to 64 numbers"),
    (JOINTS + [None], "list of 1 to 64 numbers"),
    ("0.1, 0.2", "list of 1 to 64 numbers"),
])
def test_bad_robot_state_is_a_value_error(state, match) -> None:
    """A ValueError is a 400 from the data worker. A NaN would otherwise reach
    every predicted row through attention to the state token."""
    model = _model(**DROID)
    with pytest.raises(ValueError, match=match):
        model.process_prompt(None, *MODALITIES, **POLICY, robot_state=state)


def test_robot_state_rules() -> None:
    model = _model(**DROID)
    with pytest.raises(ValueError, match="num_frames = action_chunk_size \\+ 1"):
        model._resolve_gen_params(
            {**POLICY, "robot_state": STATE, "action_chunk_size": 32, "num_frames": 32}, *MODALITIES,
        )
    # num_frames alone gives the chunk.
    p = model._resolve_gen_params({**POLICY, "robot_state": STATE, "num_frames": 17}, *MODALITIES)
    assert p["action_chunk_size"] == 16
    with pytest.raises(ValueError, match="robot_state has 8 values, but raw_action_dim is 10"):
        model._resolve_gen_params({**POLICY, "robot_state": STATE, "raw_action_dim": 10}, *MODALITIES)
    for mode in ("inverse_dynamics", "forward_dynamics"):
        with pytest.raises(ValueError, match="policy request"):
            model._resolve_gen_params(
                {"action_mode": mode, "robot_state": STATE, "raw_action_dim": 8, "domain_id": 8}, *MODALITIES,
            )
    with pytest.raises(ValueError, match="policy request"):
        model._resolve_gen_params({"robot_state": STATE}, ["image", "text"], ["video"])


def test_robot_state_rides_the_action_edge_flipped() -> None:
    model = _model(**DROID)
    tensors, p, kwargs = _submit(model, {**POLICY, "robot_state": STATE})
    row = tensors["action_inputs"][0]
    assert row.shape == (1, 8) and row.dtype == torch.float32
    assert torch.allclose(row[0], torch.tensor(JOINTS + [0.75]))
    assert kwargs["robot_state_dim"] == 8 and p["action_state_rows"] == 1
    # The values stay with the data worker; a client can't set the width.
    assert "robot_state" in Cosmos3Model.PREPROCESS_ONLY_KWARGS
    assert "robot_state_dim" in Cosmos3Model.INTERNAL_KWARGS
    # A deployment without a flip sends the state as given.
    tensors, _, _ = _submit(_model(), {**POLICY, "robot_state": STATE, "domain_id": 8})
    assert torch.allclose(tensors["action_inputs"][0][0], torch.tensor(STATE))
    # Without a state nothing goes on the edge, and forward dynamics still sends its chunk.
    tensors, _, _ = _submit(model, {**POLICY, "raw_action_dim": 8})
    assert "action_inputs" not in tensors
    tensors, _, kwargs = _submit(
        model, {"action_mode": "forward_dynamics", "domain_id": 8, "action": [STATE, STATE]},
    )
    assert tensors["action_inputs"][0].shape == (2, 8) and kwargs["action_shape"] == [2, 8]


def test_prefill_keeps_the_state_token_clean_at_frame_0() -> None:
    model = _model(**DROID)
    dit = _dit(model)
    common = {**POLICY, "size": "64x64"}
    tensors, md, _ = _submit(model, {**common, "robot_state": STATE})
    st = _prefill(dit, md, tensors, 0)
    clean = st["action_clean_mask"][0, :, 0]
    assert st["action_chunk"] == 33 and clean[0] == 1 and clean[1:].sum() == 0
    assert torch.allclose(st["clean_action"][0, 0, :8], torch.tensor(JOINTS + [0.75]))
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
    # the positions they have without a state.
    t_state = st["cond"]["action_mrope_ids"][0]
    assert t_state[0] == st["cond"]["vision_mrope_ids"][0, 0]
    assert torch.equal(t_state[1:], plain["cond"]["action_mrope_ids"][0])

    # The prompt layout's scratch counts the state token too.
    with_state = model._gen_layout(md | {"action_state_rows": 1}, [1, 2, 3], [1, 2], ["image", "text"], {})
    without = model._gen_layout(md, [1, 2, 3], [1, 2], ["image", "text"], {})
    assert with_state[COND_LABEL].scratch == without[COND_LABEL].scratch + 1


def test_reply_leaves_out_the_state_row_in_robot_units() -> None:
    model = _model(**DROID)

    def reply(kwargs, rows):
        chunk = torch.rand(1, rows, 64)
        out = np.frombuffer(model.postprocess(chunk, "action", kwargs), dtype=np.float32).reshape(-1, 64)
        return chunk[0].numpy(), out

    _, _, kwargs = _submit(model, {**POLICY, "robot_state": STATE})
    chunk, out = reply(kwargs, 33)
    assert out.shape == (32, 64)
    assert np.array_equal(out[:, :7], chunk[1:, :7]) and np.array_equal(out[:, 8:], chunk[1:, 8:])
    assert np.allclose(out[:, 7], 1 - chunk[1:, 7])

    _, _, kwargs = _submit(model, {**POLICY, "raw_action_dim": 8})
    chunk, out = reply(kwargs, 32)
    assert out.shape == (32, 64) and np.allclose(out[:, 7], 1 - chunk[:, 7])
    # A dim past raw_action_dim is padding and stays as it is, and so do the
    # actions of the other modes.
    _, _, kwargs = _submit(model, {**POLICY, "raw_action_dim": 7})
    chunk, out = reply(kwargs, 32)
    assert np.array_equal(out, chunk)
    chunk, out = reply({"action_mode": "inverse_dynamics", "raw_action_dim": 8}, 32)
    assert np.array_equal(out, chunk)


def _action_state(model, dit, rid, guidance_interval, timesteps):
    tensors, md, _ = _submit(
        model, {**POLICY, "robot_state": STATE, "size": "64x64", "guidance_scale": 3.0,
                "guidance_interval": guidance_interval},
    )
    st = _prefill(dit, md, tensors, rid)
    st.add("scheduler", types.SimpleNamespace(timesteps=torch.tensor(timesteps)))
    st.add("domain_t", torch.tensor([8]))
    return st


def test_action_guidance_interval() -> None:
    """Steps outside the interval run the conditional branch alone, and the
    declared step agrees. The deployment's interval applies when the request
    sends none."""
    model = _model(**DROID, guidance_interval_action=[960, 1001])
    _, p, _ = _submit(model, {**POLICY, "robot_state": STATE})
    assert p["guidance_interval"] == (960.0, 1001.0)
    _, p, _ = _submit(model, {**POLICY, "robot_state": STATE, "guidance_interval": [0, 1000]})
    assert p["guidance_interval"] == (0.0, 1000.0)
    assert "guidance_interval" not in _model(**DROID)._resolve_gen_params(
        {**POLICY, "robot_state": STATE}, *MODALITIES,
    )

    dit = _dit(model)
    st = _action_state(model, dit, 0, None, [999.8, 937.5, 833.3, 624.7])
    assert st["guidance_interval"] == (960.0, 1001.0)
    labels = []

    def denoise(attn, static, latents, action_latents, vts, ats, domain, label):
        labels.append(label)
        return torch.zeros_like(latents), torch.zeros_like(action_latents)

    dit._denoise_action = denoise
    dit.batched_cfg = False
    dit._action_scheduler_step = lambda st, lat, act, v, a, t: (lat, act)
    latents, action_latents = torch.zeros(1, 2), torch.zeros(1, 33, 64)
    for step, branches in ((0, 2), (1, 1), (3, 1)):
        labels.clear()
        dit._forward_action_gen(None, st, latents, action_latents, torch.tensor([step]))
        assert len(labels) == branches, step
        assert dit._step_info(ACTION_GEN_WALK, st, step).cfg_active == (branches == 2)


def test_action_batch_drops_the_unconditional_branch_outside_every_interval() -> None:
    """A cross-request action step declares and runs the unconditional branch
    only when some row is inside its guidance interval, and then only those
    rows combine the two."""
    model = _model(**DROID)
    dit = _dit(model)
    rids = ["a", "b"]
    states = {rid: _action_state(model, dit, rid, [960, 1001], [999.8, 937.5, 833.3, 624.7]) for rid in rids}
    branches = {}

    def batched(reqs, label, attn, with_cfg):
        branches["with_cfg"] = with_cfg
        cond = (torch.full((1, 2), 1.0), torch.full((1, 33, 64), 1.0))
        uncond = (torch.full((1, 2), 3.0), torch.full((1, 33, 64), 3.0))
        return [(cond, uncond) if with_cfg else (cond,) for _ in reqs]

    dit.transformer.denoise_step_action_batched = batched
    dit._action_scheduler_step = lambda st, lat, act, v, a, t: (v, a)

    def step(steps):
        inputs = []
        for rid, s in zip(rids, steps, strict=True):
            info = dit._step_info(ACTION_GEN_WALK, states[rid], s)
            inputs.append(ARNodeInputs(
                tensor_inputs={"latents": torch.zeros(1, 2), "action_latents": torch.zeros(1, 33, 64),
                               "time_index": torch.tensor([s])},
                resource_step_info=info, input_seq_len=10,
            ))
        declared = dit.declare_step(ACTION_GEN_WALK, rids, inputs)
        ei = ModelInputsFromEngine(request_ids=rids, per_request_info={}, resources={ATTN: "attn"},
                                   per_request_states=states)
        pre = dit.preprocess(ACTION_GEN_WALK, ei, inputs)
        out = dit.forward_batched(ACTION_GEN_WALK, ei, **pre)
        return {seg.label for seg in declared.segments}, pre["cfg"], branches["with_cfg"], out

    # Steps 1 and 2 are outside [960, 1001]: one branch, declared and run.
    labels, cfg, with_cfg, out = step([1, 2])
    assert labels == {COND_LABEL} and cfg is False and with_cfg is False
    assert all(float(out[rid]["action_latents"][0][0, 0, 0]) == 1.0 for rid in rids)

    # Row a at step 0 is inside: both branches run, and only a combines them
    # (3 + 3.0 * (1 - 3) = -3).
    labels, cfg, with_cfg, out = step([0, 1])
    assert labels == {COND_LABEL, UNCOND_LABEL} and cfg is True and with_cfg is True
    assert float(out["a"]["action_latents"][0][0, 0, 0]) == -3.0
    assert float(out["b"]["action_latents"][0][0, 0, 0]) == 1.0

    # Without an interval every step guides.
    assert Cosmos3DiTSubmodule._batch_cfg(
        ACTION_GEN_WALK, [GenStepInfo(cfg=True, cfg_active=True, capture_key=None)] * 2,
    )
