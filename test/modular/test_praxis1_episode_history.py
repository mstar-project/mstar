"""CPU checks for the real-frame window carried across Praxis calls."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.api_server.openpi import pack
from mstar.api_server.positronic_input import decode_praxis_frame
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.dense_window import DenseWindowPool
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import build_specs, node_handles, state_steps
from mstar.engine.state.episodes import EpisodeRowBindings
from mstar.model.base import EpisodeView
from mstar.model.praxis1.episode_history import (
    FIRST_ROWS,
    LATER_ROWS,
    STATE_NAME,
    append_count,
    window_size,
)
from mstar.model.praxis1.praxis1_model import Praxis1Model
from mstar.model.praxis1.submodules import Praxis1VaeEncoderSubmodule


class _Tokenizer:
    def __call__(self, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}


def _model(depth=3):
    model = Praxis1Model("runwayml/praxis-1-droid", skip_weight_loading=True, max_context_chunks=depth)
    model.config = replace(
        model.config,
        normalization={
            "proprio_mean": [0.0] * 10,
            "proprio_std": [1.0] * 10,
        },
    )
    model._tokenizer_cache = _Tokenizer()
    return model


def _pose():
    return [0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0, 0.5]


def test_declared_window_is_paired_and_bounded_by_reference_geometry():
    model = _model()
    (array,) = model.declare_state()
    assert array.name == STATE_NAME
    assert array.lifetime == "episode" and array.max_len is None
    assert array.reads["history"].mask.size == window_size(3) == 69
    assert array.entry.tensors["frame"].dtype == torch.uint8
    assert array.entry.tensors["pose"].shape == (8,)
    assert array.entry.tensors["pose_valid"].dtype == torch.uint8


def test_append_plan_tracks_cold_warm_reset_and_eviction():
    model = _model()

    def plan(held, rows, cold=None):
        episode = EpisodeView("session", 0, {STATE_NAME: held} if held else {}, cold)
        steps = model.walk_plan(
            "", ["image"], ["action"], {"image_inputs": [object()]}, {"executed_rows": rows}, episode
        )
        return steps[-1].step_metadata["history_append"]

    assert plan(0, 0) == 1
    assert plan(1, FIRST_ROWS) == 20
    assert plan(21, LATER_ROWS) == 24
    assert plan(45, LATER_ROWS) == 24
    assert plan(69, LATER_ROWS) == 24
    assert plan(0, LATER_ROWS, "evicted") == 1
    assert plan(0, 0, "reset") == 1
    with pytest.raises(ValueError, match="first warm"):
        plan(1, LATER_ROWS)
    with pytest.raises(ValueError, match="later warm"):
        plan(21, FIRST_ROWS)


def test_prompt_pairs_full_camera_stack_with_poses_and_drops_duplicate_signal():
    model = _model()
    current = torch.full((3, 2, 2), 99, dtype=torch.uint8)
    prior = torch.arange(21, dtype=torch.uint8)[:, None, None, None].expand(21, 3, 2, 2)
    pose = torch.tensor(_pose(), dtype=torch.float32)
    tensors = {
        "image_inputs": [current] * model.config.num_cameras,
        "executed_frame_inputs": [prior] * model.config.num_cameras,
        "executed_pose_inputs": [pose.repeat(21, 1)],
    }
    out = model.process_prompt("pick", [], [], robot_state=_pose(), tensors=tensors)
    images = out.new_input_tensors["image_inputs"][0]
    poses = out.new_input_tensors["executed_pose_inputs"][0]
    assert images.shape == (22, 3, 2, 2) and poses.shape == (22, 8)
    assert out.new_input_tensors["executed_pose_valid"][0].all()
    assert torch.equal(images[:-1], prior) and torch.equal(images[-1], current)
    assert "executed_frame_inputs" not in tensors


def test_warm_camera_window_keeps_missing_pose_history_explicit():
    model = _model()
    current = torch.full((3, 2, 2), 99, dtype=torch.uint8)
    prior = torch.arange(21, dtype=torch.uint8)[:, None, None, None].expand(21, 3, 2, 2)
    tensors = {
        "image_inputs": [current] * model.config.num_cameras,
        "executed_frame_inputs": [prior] * model.config.num_cameras,
        "executed_pose_inputs": [torch.empty((0, 8), dtype=torch.float32)],
    }
    out = model.process_prompt("pick", [], [], robot_state=_pose(), tensors=tensors)
    assert out.new_input_tensors["image_inputs"][0].shape[0] == 22
    poses = out.new_input_tensors["executed_pose_inputs"][0]
    valid = out.new_input_tensors["executed_pose_valid"][0]
    assert torch.count_nonzero(poses[:-1]) == 0
    assert valid[:-1].sum() == 0 and bool(valid[-1, 0])


@pytest.mark.parametrize("pose_history", [True, False])
def test_first_warm_25_row_client_stack_keeps_exact_reference_suffix(pose_history):
    model = _model()
    frames = np.stack([np.full((3, 4, 3), i, np.uint8) for i in range(25)])
    obs = {
        "image.exterior": frames,
        "image.wrist": frames,
        "robot_state.ee_pose": (
            np.array([[i, 0, 0.6, 1, 0, 0, 0] for i in range(25)], np.float32)
            if pose_history
            else [24, 0, 0.6, 1, 0, 0, 0]
        ),
        "grip": np.zeros(25, np.float32) if pose_history else 0.5,
        "descriptor": "praxis.sim",
    }
    decoded = decode_praxis_frame(pack({"session_id": "id", "observation": obs}), "single_arm", 21)
    assert not decoded.reset_episode and decoded.executed_rows == 21
    assert decoded.executed_frames[0][:, 0, 0, 0].tolist() == list(range(3, 24))
    assert decoded.executed_poses.shape[0] == (21 if pose_history else 0)
    tensors = {
        "image_inputs": decoded.images,
        "executed_frame_inputs": decoded.executed_frames,
        "executed_pose_inputs": [decoded.executed_poses],
    }
    prepared = model.process_prompt(
        "pick",
        [],
        [],
        robot_state=decoded.robot_state,
        action_format="positronic",
        pose_descriptor=decoded.pose_descriptor,
        tensors=tensors,
    )
    valid = prepared.new_input_tensors["executed_pose_valid"][0]
    assert valid[:-1].sum() == (21 if pose_history else 0)
    assert bool(valid[-1, 0])
    plan = model.walk_plan(
        "",
        ["image"],
        ["action"],
        {"image_inputs": [object()]},
        {"executed_rows": decoded.executed_rows},
        EpisodeView("id", 1, {STATE_NAME: 1}),
    )
    assert plan[-1].step_metadata["history_append"] == 20

    node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), episode_history=True)
    node._tiles = lambda views: views
    node.vae_encode = lambda canvas: canvas
    handle = _Handle()
    node.state_handles = {STATE_NAME: handle}
    views = prepared.new_input_tensors["image_inputs"]
    node.forward(
        "chunk",
        SimpleNamespace(),
        history_append=20,
        history_pose=prepared.new_input_tensors["executed_pose_inputs"][0],
        history_pose_valid=valid,
        **{f"view_{i}": view for i, view in enumerate(views)},
    )
    _, added = handle.added[0]
    assert added["frame"][:, 0, 0, 0].tolist() == list(range(4, 24))
    assert added["pose_valid"].sum() == (20 if pose_history else 0)


class _Handle:
    def __init__(self):
        self.added = []
        self.reads = 0

    def add(self, layer, **fields):
        self.added.append((layer, fields))

    def read(self, layer):
        self.reads += 1


class _Vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(z_dim=None)


@pytest.mark.parametrize("append,expected", [(1, [99]), (20, list(range(1, 21))), (24, list(range(21, 45)))])
def test_vae_state_adds_only_new_executed_rows(append, expected):
    model = _model()
    node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), episode_history=True)
    node._tiles = lambda views: views
    node.vae_encode = lambda canvas: canvas
    handle = _Handle()
    node.state_handles = {STATE_NAME: handle}
    if append == 1:
        values = [99]
    elif append == 20:
        values = list(range(21)) + [99]
    else:
        values = list(range(45)) + [99]
    frames = torch.tensor(values, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 1, 1)
    poses = torch.arange(len(values), dtype=torch.float32)[:, None].expand(-1, 8)
    out = node.forward(
        "chunk",
        SimpleNamespace(),
        history_append=append,
        history_pose=poses,
        history_pose_valid=torch.ones((len(values), 1), dtype=torch.uint8),
        **{f"view_{i}": frames for i in range(model.config.num_cameras)},
    )
    _, fields = handle.added[0]
    assert fields["frame"][:, 0, 0, 0].tolist() == expected
    assert fields["pose"][:, 0].tolist() == ([float(v) for v in expected] if append != 1 else [0.0])
    assert handle.reads == 1
    assert fields["pose_valid"].all()
    assert out["obs_latents"][0].shape[2] == 1


def test_reset_requires_new_stack_before_warm_append():
    assert append_count(0, 24) == 1  # eviction makes held state authoritative
    assert append_count(1, 21) == 20
    assert append_count(21, 24) == 24


def test_executed_window_matches_pinned_reference_after_slide_and_reset():
    reference = pytest.importorskip("praxis.robot_io", reason="reference checkout not on PYTHONPATH")
    stream = reference.ContiguousStreamContext(action_chunk=24, max_context_chunks=3)
    retained = [0]  # cold call's current real frame
    for first, rows in ((0, 21), (21, 24), (45, 24), (69, 24)):
        frame = torch.arange(first, first + rows, dtype=torch.uint8).reshape(1, 1, rows, 1, 1)
        pose = np.arange(first, first + rows, dtype=np.float32).reshape(-1, 1)
        stream.append_executed(frame, pose)
        add = append_count(len(retained), rows)
        retained = (retained + list(range(first + rows - add, first + rows)))[-69:]
        assert retained == stream._clip[0, 0, :, 0, 0].tolist()
        assert retained == stream._poses[:, 0].tolist()
    stream.reset()
    retained = []
    assert stream.n_raws == 0 and retained == []
    assert append_count(len(retained), 24) == 1  # eviction resumes cold


def test_engine_keeps_paired_history_across_request_ids_then_clears_on_reset():
    model = _model()
    model.config = replace(model.config, tile_height=1, tile_width=1)
    (array,) = model.declare_state()
    (spec,) = build_specs([array])
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    handle = node_handles(model, "vae_encoder", {STATE_NAME: pool})[STATE_NAME]
    owners = EpisodeRowBindings()

    def call(rid, episode, values, reset=False):
        pool.ingest_request(rid)
        binding = owners.bind(rid, 0, episode)
        pool.bind_episode(rid, 0, episode, cold=binding.cold, reset=reset)
        step = state_steps(
            SubmoduleStep(
                segments=[Segment(rid, "main", len(values))],
                state={STATE_NAME: ArrayStep()},
            ),
            {STATE_NAME: handle},
            graph_walk="chunk",
        )
        step.set_ctx(StepContext(request_ids=[rid], graph_walk="chunk", slot=0, capture=False))
        assert pool.admit(step.steps[STATE_NAME], step.ctx).ok
        pool.plan(step.steps[STATE_NAME], step.ctx)
        step.begin_calls()
        vals = torch.tensor(values, dtype=torch.uint8)
        handle.add(
            0,
            frame=vals[:, None, None, None].expand(-1, 3, 1, model.config.num_cameras).clone(),
            pose=vals[:, None].expand(-1, 8).float().clone(),
            pose_valid=torch.ones((len(values), 1), dtype=torch.uint8),
        )
        reached = handle.read(0)
        result = reached.fields["frame"][:, 0, 0, 0].tolist()
        assert reached.fields["pose"][:, 0].tolist() == [float(v) for v in result]
        step.check_calls()
        pool.commit(step.steps[STATE_NAME], step.ctx)
        step.commit_calls()
        step.end_calls()
        pool.remove_request(rid)
        owners.unbind(rid)
        return result

    assert call(1, "episode-a", [0]) == [0]
    assert call(2, "episode-a", list(range(1, 21))) == list(range(21))
    assert call(3, "episode-a", list(range(21, 45))) == list(range(45))
    assert call(4, "episode-a", list(range(45, 69))) == list(range(69))
    assert call(5, "episode-a", list(range(69, 93))) == list(range(24, 93))
    assert call(6, "episode-a", [100], reset=True) == [100]
