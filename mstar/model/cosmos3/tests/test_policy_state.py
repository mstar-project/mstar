"""Policy requests that send the robot's current state as the first action row,
the way NVIDIA's RoboLab server conditions the released DROID policies: the row
is clean conditioning at video frame 0, the rest of the chunk is predicted, and
the reply leaves the row out. CPU only, no weights."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from mstar.model.base import ProcessPromptOutput
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import ACTION_GEN_WALK, Cosmos3DiTSubmodule

STATE = [0.1, -0.2, 0.3, -2.0, 0.05, 1.9, 0.8, 1.0]  # 7 joints, 1 - gripper
POLICY = {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 8}
MODALITIES = (["image", "text"], ["action"])


def _model() -> Cosmos3Model:
    return Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)


def _submit(model: Cosmos3Model, mk: dict):
    """What the server does with a request: process_prompt puts the action on a
    tensor edge and its dimensions in the kwargs, the conductor resolves the step
    metadata from the kwargs without the action values, and the data worker keeps
    the full kwargs for postprocess."""
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


def _prefill(dit: Cosmos3DiTSubmodule, md: dict, tensors: dict, rid: int):
    fwd_info = types.SimpleNamespace(rid_handle=rid, request_id=str(rid))
    dit._prepare_action_prefill(
        fwd_info, md, tensors, [1, 2, 3], [1, 2], md["height"], md["width"], md["fps"],
        md["guidance_scale"], md["num_inference_steps"], "cpu",
    )
    return dit.request_states[rid]


def test_state_row_request_params() -> None:
    model = _model()
    _, p, _ = _submit(model, {**POLICY, "action_chunk_size": 33, "num_frames": 33, "action": [STATE]})
    assert p["action_state_rows"] == 1
    assert p["action_chunk_size"] == 33 and p["num_frames"] == 33

    # The state counts as a row, so the frame count defaults to the chunk, and
    # a bare state row also gives the action width.
    p = model._resolve_gen_params({**POLICY, "action_chunk_size": 33, "action": [STATE]}, *MODALITIES)
    assert p["num_frames"] == 33
    p = model._resolve_gen_params(
        {"action_mode": "policy", "domain_name": "droid_lerobot", "num_frames": 33, "action": STATE},
        *MODALITIES,
    )
    assert p["action_chunk_size"] == 33 and p["raw_action_dim"] == 8 and p["action_state_rows"] == 1

    # Without a state row nothing changes.
    p = model._resolve_gen_params({**POLICY, "action_chunk_size": 32}, *MODALITIES)
    assert "action_state_rows" not in p and p["num_frames"] == 33

    with pytest.raises(ValueError, match="num_frames to equal"):
        model._resolve_gen_params(
            {**POLICY, "action_chunk_size": 33, "num_frames": 34, "action": [STATE]}, *MODALITIES
        )
    with pytest.raises(ValueError, match="raw_action_dim=8 values"):
        model._resolve_gen_params({**POLICY, "action_chunk_size": 33, "action": [STATE[:7]]}, *MODALITIES)
    with pytest.raises(ValueError, match="one row"):
        model._resolve_gen_params({**POLICY, "action_chunk_size": 33, "action": [STATE, STATE]}, *MODALITIES)
    with pytest.raises(ValueError, match="leave a row"):
        model._resolve_gen_params({**POLICY, "action_chunk_size": 1, "action": [STATE]}, *MODALITIES)


def test_state_row_reaches_the_dit_edge() -> None:
    model = _model()
    tensors, _, kwargs = _submit(model, {**POLICY, "action_chunk_size": 33, "action": STATE})
    assert tensors["action_inputs"][0].shape == (1, 8)
    assert kwargs["action_shape"] == [1, 8]
    tensors, p, _ = _submit(model, {**POLICY, "action_chunk_size": 32})
    assert "action_inputs" not in tensors and "action_state_rows" not in p
    # Inverse dynamics predicts the whole chunk, so an action array is not conditioning there.
    tensors, _, _ = _submit(
        model, {**POLICY, "action_mode": "inverse_dynamics", "action_chunk_size": 32, "action": [STATE]}
    )
    assert "action_inputs" not in tensors


def test_prefill_marks_the_state_row_clean() -> None:
    model = _model()
    dit = _dit(model)
    common = {**POLICY, "num_frames": 33, "size": "64x64", "fps": 15}
    tensors, md, _ = _submit(model, {**common, "action_chunk_size": 33, "action": [STATE]})
    st = _prefill(dit, md, tensors, 0)
    clean = st["action_clean_mask"][0, :, 0]
    assert clean[0] == 1 and clean[1:].sum() == 0
    assert torch.allclose(st["clean_action"][0, 0, :8], torch.tensor(STATE))
    assert st["clean_action"][0, 0, 8:].abs().sum() == 0 and st["clean_action"][0, 1:].abs().sum() == 0
    for static in (st["cond"], st["uncond"]):
        assert static["action_noisy_frame_indexes"][0].tolist() == list(range(1, 33))
        assert static["num_noisy_action_tokens"] == 32

    tensors, md, _ = _submit(model, {**common, "action_chunk_size": 32})
    plain = _prefill(dit, md, tensors, 1)
    assert plain["action_clean_mask"].sum() == 0 and plain["clean_action"].abs().sum() == 0
    assert plain["cond"]["action_noisy_frame_indexes"][0].tolist() == list(range(32))

    # The state token sits at video frame 0's time, and the predicted rows keep
    # the positions they have without a state row.
    t_state = st["cond"]["action_mrope_ids"][0]
    assert t_state[0] == st["cond"]["vision_mrope_ids"][0, 0]
    assert torch.equal(t_state[1:], plain["cond"]["action_mrope_ids"][0])


def test_reply_leaves_out_the_state_row() -> None:
    model = _model()
    chunk = torch.arange(33 * 64, dtype=torch.float32).reshape(1, 33, 64)
    _, _, kwargs = _submit(model, {**POLICY, "action_chunk_size": 33, "action": [STATE]})
    out = np.frombuffer(model.postprocess(chunk, "action", kwargs), dtype=np.float32).reshape(-1, 64)
    assert np.array_equal(out, chunk[0, 1:].numpy())
    _, _, kwargs = _submit(model, {**POLICY, "action_chunk_size": 32})
    out = np.frombuffer(model.postprocess(chunk[:, :32], "action", kwargs), dtype=np.float32).reshape(-1, 64)
    assert np.array_equal(out, chunk[0, :32].numpy())


def test_action_guidance_interval() -> None:
    """A guidance_interval applies to action requests: steps outside it run the
    conditional branch alone, and the declared step agrees."""
    model = _model()
    dit = _dit(model)
    tensors, md, _ = _submit(
        model, {**POLICY, "action_chunk_size": 33, "num_frames": 33, "size": "64x64", "fps": 15,
                "guidance_scale": 3.0, "guidance_interval": [960, 1001], "action": [STATE]},
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
    dit._action_scheduler_step = lambda st, lat, act, v, a, t: (lat, act)
    latents, action_latents = torch.zeros(1, 2), torch.zeros(1, 33, 64)
    for step, branches in ((0, 2), (1, 1), (3, 1)):
        labels.clear()
        dit._forward_action_gen(None, st, latents, action_latents, torch.tensor([step]))
        assert len(labels) == branches, step
        assert dit._step_info(ACTION_GEN_WALK, st, step).cfg_active == (branches == 2)
