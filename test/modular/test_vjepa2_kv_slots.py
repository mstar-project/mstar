"""V-JEPA 2-AC says how much its rollout writes to ``main``, because no
prompt tensor does.

The pool admits a request by what it declares. The rollout walk prefills
nothing a prompt tensor measures: each of its ``rollout_horizon`` steps writes
one frame's spatial tokens behind the step's action and state (and extrinsics)
tokens, all kept on ``main``. Declaring the default, a prompt tensor's length,
lets the pool admit a rollout whose later steps find no page, which is the hold
admission exists to end.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from api_uploads import decode_uploads, mp4, preprocess

from mstar.model.vjepa2.config import KV_CACHE
from mstar.model.vjepa2.vjepa2_model import VJepa2ACModel


def _model(use_extrinsics: bool) -> VJepa2ACModel:
    # one block each, not ViT-g's 40 and the predictor's 24: depth changes no
    # token count, only how long the walk takes on CPU
    model = VJepa2ACModel(
        model_path_hf="facebook/vjepa2-ac-vitg", skip_weight_loading=True,
        ac_predictor_config={"depth": 1, "use_extrinsics": use_extrinsics},
    )
    model.config.num_hidden_layers = 1
    return model


def _request(horizon: int, use_extrinsics: bool) -> dict:
    steps = horizon + 4
    request = {
        "rollout_horizon": horizon,
        "actions": torch.rand(steps, 7).tolist(),
        "states": torch.rand(steps, 7).tolist(),
    }
    if use_extrinsics:
        request["extrinsics"] = torch.rand(steps, 6).tolist()
    return request


def _rollout_writes(model: VJepa2ACModel, tensors: dict, kwargs: dict) -> int:
    """Run the rollout walk on CPU and sum the spans each step declares on ``main``.

    The predictor runs without a label (plain SDPA, no cache) only to hand the
    next step its context, as the loop-back edge would.
    """
    signals = {name: [] for name in tensors}
    walk_args = model.get_initial_forward_pass_args("p", ["video"], ["video"], signals, model_kwargs=kwargs)
    walk = walk_args.full_metadata.graph_walk
    encoder = model.get_submodule("video_encoder")
    rollout = model.get_submodule("rollout_predictor")
    with torch.no_grad():
        hidden = encoder._encode(walk, tensors["video_frames"][0].unsqueeze(0))
    conditioning = {name: tensors[name] for name in ("actions", "states", "extrinsics") if name in tensors}
    written = 0
    for it in range(model.config.max_rollout_horizon):
        info = SimpleNamespace(dynamic_loop_iter_counts={"rollout_loop": it}, step_metadata=walk_args.step_metadata)
        inp = rollout.prepare_inputs(walk, info, {"encoder_hidden": [hidden], **conditioning})
        step = rollout.declare_step(walk, ["r"], [inp])
        written += sum(segment.span for segment in step.segments if segment.label == "main")
        with torch.no_grad():
            hidden = model.predictor(
                inp.input_embeds, inp.tensor_inputs["actions"], inp.tensor_inputs["states"],
                extrinsics=inp.tensor_inputs.get("extrinsics"), t_0=it,
            )
        if rollout.check_stop("r", info, {}):
            break
    return written


# a horizon the loop caps runs only the steps under its max_rollout_horizon
@pytest.mark.parametrize(("use_extrinsics", "clips", "horizon", "max_horizon"), [
    (False, 1, 3, None),
    (True, 2, 3, None),
    (False, 1, 5, 2),
], ids=["one_clip", "two_clips_with_extrinsics", "horizon_past_the_loop"])
def test_a_rollout_declares_what_its_steps_write(use_extrinsics, clips, horizon, max_horizon, monkeypatch, tmp_path):
    decode_uploads(monkeypatch)
    model = _model(use_extrinsics)
    if max_horizon is not None:
        model.config.max_rollout_horizon = max_horizon
    with open("configs/vjepa2_ac.yaml") as f:
        deployment = yaml.safe_load(f)
    files = [mp4(8, 240, 320, tmp_path) for _ in range(clips)]
    tensors, model_kwargs = preprocess(
        model, deployment, None, files, ["video"], _request(horizon, use_extrinsics), tmp_path,
    )

    assert (model_kwargs["prompt_slots"], model_kwargs["decode_labels"]) == (
        {KV_CACHE: {"main": _rollout_writes(model, tensors, model_kwargs)}}, {KV_CACHE: []},
    ), "the rollout declared other than the spans its steps write to main"
