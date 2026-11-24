"""CPU checks for the real-frame window carried across Praxis calls."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

import mstar.conductor.conductor as conductor_mod
import mstar.conductor.episodes as episodes_mod
from mstar.api_server.openpi import pack
from mstar.api_server.positronic_input import decode_praxis_frame
from mstar.communication.tensors import LocalTransferEngine, TensorPointerInfo
from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import EPISODE_CALLS
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.recurrent.dense_window import DenseWindowPool
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import build_specs, node_handles, state_setup, state_steps
from mstar.engine.state.episodes import EpisodeRowBindings
from mstar.model.base import EpisodeView
from mstar.model.praxis1.components.action_expert import _RESOURCE_MASK
from mstar.model.praxis1.components.encoder import TapFeatures
from mstar.model.praxis1.episode_history import (
    BOUNDARY_STATE_NAME,
    FIRST_ROWS,
    LATER_ROWS,
    STATE_NAME,
    append_count,
    window_size,
)
from mstar.model.praxis1.praxis1_model import TEXT_STATE_NAME, Praxis1Model
from mstar.model.praxis1.submodules import (
    Praxis1ActionExpertSubmodule,
    Praxis1ActionHeadSubmodule,
    Praxis1BackboneSubmodule,
    Praxis1TextEncoderSubmodule,
    Praxis1VaeEncoderSubmodule,
)
from mstar.model.submodule_base import InputMetadata
from mstar.utils.ipc_format import (
    EndEpisode,
    NewRequestConductor,
    ReadsDone,
    WorkerGraphsDone,
    WorkerMessageType,
)


class _Tokenizer:
    def __call__(self, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}


def _model(depth=3):
    model = Praxis1Model(
        "runwayml/praxis-1-droid", skip_weight_loading=True,
        max_context_chunks=depth, max_text_tokens=256,
    )
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
    declared = {array.name: array for array in model.declare_state()}
    array, boundaries, text = (
        declared[STATE_NAME], declared[BOUNDARY_STATE_NAME], declared[TEXT_STATE_NAME],
    )
    assert array.name == STATE_NAME
    assert array.lifetime == "episode" and array.max_len is None
    assert array.reads["history"].mask.size == window_size(3) == 69
    assert array.entry.tensors["frame"].dtype == torch.uint8
    assert array.entry.tensors["pose"].shape == (8,)
    assert array.entry.tensors["pose_valid"].dtype == torch.uint8
    assert boundaries.name == BOUNDARY_STATE_NAME
    assert boundaries.reads["recent"].mask.size == 4
    assert text.name == TEXT_STATE_NAME and text.labels == frozenset({"cond", "blank"})
    assert text.reads["praxis_text_history"].mask.size == 256


def test_episode_text_key_and_walk_follow_cold_warm_changed_prompt_and_reset():
    model = _model()

    class VaryingTokenizer:
        def __call__(self, *, text, **kwargs):
            ids = [1, 2] if text[0] == "pick" else [1, 3]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

    model._tokenizer_cache = VaryingTokenizer()
    tensors = {"image_inputs": [torch.zeros(3, 2, 2, dtype=torch.uint8)] * model.config.num_cameras}

    def prepared(prompt):
        return model.process_prompt(prompt, [], [], robot_state=_pose(), tensors=dict(tensors))

    first, same, changed = prepared("pick"), prepared("pick"), prepared("place")
    assert first.episode_keys == same.episode_keys
    assert first.episode_keys[TEXT_STATE_NAME] != changed.episode_keys[TEXT_STATE_NAME]

    def walks(held):
        episode = EpisodeView("session", 1, {TEXT_STATE_NAME: held} if held else {}, None)
        return [step.walk for step in model.walk_plan(
            "", ["image"], ["action"], {"image_inputs": [object()]},
            {"action_format": "positronic"}, episode,
        )]

    assert walks(0) == ["encode_text", "chunk"]  # first, changed key, reset, or eviction
    assert walks(2) == ["chunk_warm"]


def test_episode_text_reuse_requires_a_deployment_token_cap():
    with pytest.raises(ValueError, match="max_text_tokens"):
        Praxis1Model("runwayml/praxis-1-droid", skip_weight_loading=True, max_context_chunks=1)
    model = _model()
    model.max_text_tokens = 1
    tensors = {"image_inputs": [torch.zeros(3, 2, 2, dtype=torch.uint8)] * model.config.num_cameras}
    with pytest.raises(ValueError, match="exceeds max_text_tokens"):
        model.process_prompt("pick", [], [], robot_state=_pose(), tensors=tensors)


def test_append_plan_tracks_cold_warm_reset_and_eviction():
    model = _model()

    def plan(held, rows, cold=None, calls=0):
        episode = EpisodeView("session", calls, {STATE_NAME: held} if held else {}, cold)
        steps = model.walk_plan(
            "", ["image"], ["action"], {"image_inputs": [object()]},
            {"executed_rows": rows, "action_format": "positronic"}, episode
        )
        return steps[-1].step_metadata["history_append"]

    assert plan(0, 0) == 0
    assert plan(0, FIRST_ROWS, calls=1) == FIRST_ROWS
    assert plan(21, LATER_ROWS) == 24
    assert plan(45, LATER_ROWS) == 24
    assert plan(69, LATER_ROWS) == 24
    assert plan(0, LATER_ROWS, "evicted", calls=3) == 0
    assert plan(0, 0, "reset", calls=3) == 0
    with pytest.raises(ValueError, match="first warm"):
        append_count(0, LATER_ROWS, first_warm=True)
    with pytest.raises(ValueError, match="later warm"):
        plan(21, FIRST_ROWS)


def test_warm_openpi_is_refused_before_any_graph_walk() -> None:
    model = _model()
    episode = EpisodeView("session", 1, {})
    with pytest.raises(ValueError, match="warm Praxis episodes require Positronic"):
        model.walk_plan(
            "", ["image"], ["action"], {"image_inputs": [object()]},
            {"executed_rows": FIRST_ROWS, "action_format": "openpi"}, episode,
        )
    warm = model.walk_plan(
        "", ["image"], ["action"], {"image_inputs": [object()]},
        {"executed_rows": FIRST_ROWS, "action_format": "positronic"}, episode,
    )
    assert warm[-1].step_metadata["action_format"] == "positronic"


def test_attention_layout_uses_episode_depth_after_append() -> None:
    from mstar.conductor.conductor import _with_planned_state_layout
    from mstar.engine.state.adapters import CHUNK_CAUSAL_LAYOUT_KWARG, EXPERT_LAYOUT_KWARG
    from mstar.model.praxis1.submodules import ATTN_READS, ENCODER_COND_KV, EXPERT_TRUNK_KEYS

    model = _model()
    p = (model.config.latent_height // model.config.patch_size[1]) * (
        model.config.latent_width // model.config.patch_size[2]
    )
    g = model.config.num_registers + p + 3

    def layout(held: int, rows: int, calls: int = 0):
        episode = EpisodeView("session", calls, {STATE_NAME: held} if held else {})
        walk = model.walk_plan(
            "", ["image"], ["action"], {"image_inputs": [object()]},
            {"executed_rows": rows, "text_tokens": 3, "blank_tokens": 2,
             "action_format": "positronic"}, episode,
        )[-1]
        assert "state_layout" in walk.step_metadata
        (wire_walk,) = _with_planned_state_layout([walk], model.declare_state())
        assert "state_layout" not in wire_walk.step_metadata
        return wire_walk.step_metadata

    cold, warm, reset = layout(0, 0), layout(0, FIRST_ROWS, 1), layout(0, 0)
    for metadata, k, d in ((cold, 1, 0), (warm, 2, 1), (reset, 1, 0)):
        encoder = metadata[CHUNK_CAUSAL_LAYOUT_KWARG][ENCODER_COND_KV][ENCODER_COND_KV][ATTN_READS[ENCODER_COND_KV]]
        cross = metadata[EXPERT_LAYOUT_KWARG][EXPERT_TRUNK_KEYS][EXPERT_TRUNK_KEYS][ATTN_READS[EXPERT_TRUNK_KEYS]]
        assert encoder == [g, k * 6 * p, d * 6 * p]
        assert cross == [k, g, d * 6, k * 6]


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
        {"executed_rows": decoded.executed_rows, "action_format": "positronic"},
        EpisodeView("id", 1, {}),
    )
    assert plan[-1].step_metadata["history_append"] == FIRST_ROWS

    node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), max_context_chunks=3)
    node._tiles = lambda views: views
    node.vae_encode = lambda canvas: canvas[:, :, ::4]
    # The prior call's observation can differ from the oldest frame in the
    # client's executed stack; equality here hid the original bug.
    seed_pose = torch.tensor([[77.0, 0, 0.6, 1, 0, 0, 0, 0.5]])
    handle = _Handle({
        "frame": torch.empty((0, 3, 3, 8), dtype=torch.uint8),
        "pose": torch.empty((0, 8)),
        "pose_valid": torch.empty((0, 1), dtype=torch.uint8),
    })
    boundary = _Handle({"pose": seed_pose, "history_valid": torch.ones((1, 1), dtype=torch.uint8)})
    node.state_handles = {STATE_NAME: handle, BOUNDARY_STATE_NAME: boundary}
    views = prepared.new_input_tensors["image_inputs"]
    out = node.forward(
        "chunk",
        SimpleNamespace(),
        history_append=FIRST_ROWS,
        history_pose=prepared.new_input_tensors["executed_pose_inputs"][0],
        history_pose_valid=valid,
        **{f"view_{i}": view for i, view in enumerate(views)},
    )
    _, added = handle.added[0]
    assert added["frame"][:, 0, 0, 0].tolist() == list(range(3, 24))
    assert added["pose_valid"].sum() == (21 if pose_history else 0)
    assert out["anchor_latents"][0].shape[2] == 1
    assert out["anchor_latents"][0][0, 0, 0, 0, 0].item() == 3
    assert out["clean_latents"][0].shape[2] == 6
    assert out["proprio_all"][0].shape == (1, 2, 10)


class _Handle:
    def __init__(self, prior=None):
        self.added = []
        self.reads = 0
        self.prior = prior or {}

    def add(self, layer, **fields):
        self.added.append((layer, fields))

    def read(self, layer):
        self.reads += 1
        if not self.added and not self.prior:
            return SimpleNamespace(fields={
                "frame": torch.empty((0, 3, 1, 1), dtype=torch.uint8),
                "pose": torch.empty((0, 8)),
                "pose_valid": torch.empty((0, 1), dtype=torch.uint8),
                "history_valid": torch.empty((0, 1), dtype=torch.uint8),
            })
        fields = dict(self.prior)
        if self.added:
            for name, value in self.added[-1][1].items():
                fields[name] = torch.cat((fields[name], value)) if name in fields else value
        return SimpleNamespace(fields=fields)

    def commit(self):
        if not self.added and not self.prior:
            return
        self.prior = self.read(0).fields
        self.added.clear()


class _Vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(z_dim=None)


def test_cold_seed_ignores_unretained_missing_pose_rows():
    model = _model()
    node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), max_context_chunks=3)
    node._tiles = lambda views: views
    node.vae_encode = lambda canvas: canvas[:, :, ::4]
    frames, boundaries = _Handle(), _Handle()
    node.state_handles = {STATE_NAME: frames, BOUNDARY_STATE_NAME: boundaries}
    images = torch.zeros((22, 3, 2, 2), dtype=torch.uint8)
    poses = torch.zeros((22, 8), dtype=torch.float32)
    poses[-1] = torch.tensor(_pose())
    valid = torch.zeros((22, 1), dtype=torch.uint8)
    valid[-1] = 1

    node.forward(
        "chunk", SimpleNamespace(), history_append=0,
        history_pose=poses, history_pose_valid=valid,
        **{f"view_{i}": images for i in range(model.config.num_cameras)},
    )
    assert not frames.added
    assert boundaries.added[-1][1]["history_valid"].tolist() == [[1]]

    frames.commit()
    boundaries.commit()
    missing = torch.zeros((22, 1), dtype=torch.uint8)
    missing[-1] = 1
    node.forward(
        "chunk", SimpleNamespace(), history_append=21,
        history_pose=torch.tensor(_pose()).repeat(22, 1), history_pose_valid=missing,
        **{f"view_{i}": images for i in range(model.config.num_cameras)},
    )
    assert boundaries.added[-1][1]["history_valid"].tolist() == [[0]]

    frames.commit()
    boundaries.commit()
    full = torch.ones((25, 1), dtype=torch.uint8)
    node.forward(
        "chunk", SimpleNamespace(), history_append=24,
        history_pose=torch.tensor(_pose()).repeat(25, 1), history_pose_valid=full,
        **{f"view_{i}": torch.zeros((25, 3, 2, 2), dtype=torch.uint8)
           for i in range(model.config.num_cameras)},
    )
    assert boundaries.added[-1][1]["history_valid"].tolist() == [[0]]


def test_first_warm_valid_history_does_not_inherit_invalid_cold_pose():
    model = _model()
    node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), max_context_chunks=3)
    node._tiles = lambda views: views
    node.vae_encode = lambda canvas: canvas[:, :, ::4]
    frames, boundaries = _Handle(), _Handle()
    node.state_handles = {STATE_NAME: frames, BOUNDARY_STATE_NAME: boundaries}
    cold_images = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)
    cold_pose = torch.tensor(_pose()).reshape(1, 8)
    node.forward(
        "chunk", SimpleNamespace(), history_append=0,
        history_pose=cold_pose, history_pose_valid=torch.zeros((1, 1), dtype=torch.uint8),
        **{f"view_{i}": cold_images for i in range(model.config.num_cameras)},
    )
    assert boundaries.added[-1][1]["history_valid"].tolist() == [[0]]
    frames.commit()
    boundaries.commit()

    warm_pose = cold_pose.repeat(22, 1)
    warm_pose[:21, 0] = torch.arange(21)
    node.forward(
        "chunk", SimpleNamespace(), history_append=21,
        history_pose=warm_pose, history_pose_valid=torch.ones((22, 1), dtype=torch.uint8),
        **{f"view_{i}": torch.zeros((22, 3, 2, 2), dtype=torch.uint8)
           for i in range(model.config.num_cameras)},
    )
    assert frames.added[-1][1]["pose"][:, 0].tolist() == list(range(21))
    assert boundaries.added[-1][1]["history_valid"].tolist() == [[1]]


def test_missing_pose_fallback_matches_reference_over_three_calls():
    reference = pytest.importorskip("praxis.robot_io", reason="reference checkout not on PYTHONPATH")
    single_arm = pytest.importorskip("praxis.single_arm", reason="reference checkout not on PYTHONPATH")
    stream = reference.ContiguousStreamContext(action_chunk=24, max_context_chunks=3)
    model = _model()
    node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), max_context_chunks=3)
    node._tiles = lambda views: views
    node.vae_encode = lambda canvas: canvas[:, :, ::4]
    frames = _Handle()
    boundaries = _Handle()
    node.state_handles = {STATE_NAME: frames, BOUNDARY_STATE_NAME: boundaries}
    boundary_poses = []

    for call, (current, first, count) in enumerate(((0, 0, 0), (21, 0, 21), (45, 21, 24))):
        pose = [float(current) / 100, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0, 0.5]
        boundary_poses.append(pose)
        prior = torch.arange(first, first + count, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 2, 2)
        cur = torch.full((3, 2, 2), current, dtype=torch.uint8)
        tensors = {
            "image_inputs": [cur] * model.config.num_cameras,
            "executed_frame_inputs": [prior] * model.config.num_cameras,
            "executed_pose_inputs": [torch.empty((0, 8), dtype=torch.float32)],
        }
        prepared = model.process_prompt("pick", [], [], robot_state=pose, tensors=tensors)
        if count:
            clip = prior[None].permute(0, 2, 1, 3, 4)
            stream.append_executed(clip, poses=None)
        model_poses = [
            single_arm.pose8_to_model(np.asarray(row, dtype=np.float32), single_arm.RIG_FRAMES[""])
            for row in boundary_poses[-(stream.depth + 1):]
        ]
        expected = reference.pose8_to_10d(np.asarray(model_poses, dtype=np.float32))
        append = append_count(len(frames.prior.get("frame", [])), count, first_warm=call == 1)
        out = node.forward(
            "chunk", SimpleNamespace(), history_append=append,
            history_pose=prepared.new_input_tensors["executed_pose_inputs"][0],
            history_pose_valid=prepared.new_input_tensors["executed_pose_valid"][0],
            **{f"view_{i}": view for i, view in enumerate(prepared.new_input_tensors["image_inputs"])},
        )
        assert out["clean_latents"][0].shape[2] == stream.depth * 6
        np.testing.assert_allclose(out["proprio_all"][0][0].numpy(), expected, rtol=0, atol=1e-6)
        frames.commit()
        boundaries.commit()
    assert boundaries.prior["history_valid"][-1, 0].item() == 0


@pytest.mark.parametrize("append,expected", [(0, []), (21, list(range(21))), (24, list(range(21, 45)))])
def test_vae_state_adds_only_new_executed_rows(append, expected):
    model = _model()
    node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), max_context_chunks=3)
    node._tiles = lambda views: views
    node.vae_encode = lambda canvas: canvas[:, :, ::4]
    prior_values = [] if append in (0, 21) else list(range(21))
    prior_frame = torch.tensor(prior_values, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 1, 2)
    prior_pose = torch.tensor(prior_values, dtype=torch.float32)[:, None].expand(-1, 8)
    handle = _Handle({"frame": prior_frame, "pose": prior_pose,
                      "pose_valid": torch.ones((len(prior_values), 1), dtype=torch.uint8)})
    boundary_prior = [] if append == 0 else ([0] if append == 21 else [0, 21])
    boundary = _Handle({
        "pose": torch.tensor(boundary_prior, dtype=torch.float32)[:, None].expand(-1, 8),
        "history_valid": torch.ones((len(boundary_prior), 1), dtype=torch.uint8),
    })
    node.state_handles = {STATE_NAME: handle, BOUNDARY_STATE_NAME: boundary}
    if append == 0:
        values = [99]
    elif append == 21:
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
    if append:
        _, fields = handle.added[0]
        assert fields["frame"][:, 0, 0, 0].tolist() == expected
        assert fields["pose"][:, 0].tolist() == [float(v) for v in expected]
        assert fields["pose_valid"].all()
    else:
        assert not handle.added
    assert handle.reads == 1
    assert out["obs_latents"][0].shape[2] == 1


def test_reset_requires_new_stack_before_warm_append():
    assert append_count(0, 24) == 0  # eviction makes held state authoritative
    assert append_count(0, 21, first_warm=True) == 21
    assert append_count(21, 24) == 24


def test_executed_window_matches_pinned_reference_after_slide_and_reset():
    reference = pytest.importorskip("praxis.robot_io", reason="reference checkout not on PYTHONPATH")
    stream = reference.ContiguousStreamContext(action_chunk=24, max_context_chunks=3)
    retained = []  # cold call has not executed any rows
    for first, rows in ((0, 21), (21, 24), (45, 24), (69, 24)):
        frame = torch.arange(first, first + rows, dtype=torch.uint8).reshape(1, 1, rows, 1, 1)
        pose = np.arange(first, first + rows, dtype=np.float32).reshape(-1, 1)
        stream.append_executed(frame, pose)
        add = append_count(len(retained), rows, first_warm=first == 0)
        retained = (retained + list(range(first + rows - add, first + rows)))[-69:]
        assert retained == stream._clip[0, 0, :, 0, 0].tolist()
        assert retained == stream._poses[:, 0].tolist()
    stream.reset()
    retained = []
    assert stream.n_raws == 0 and retained == []
    assert append_count(len(retained), 24) == 0  # eviction resumes cold


def test_engine_keeps_paired_history_across_request_ids_then_clears_on_reset():
    model = _model()
    model.config = replace(model.config, tile_height=1, tile_width=1)
    arrays = [array for array in model.declare_state() if "vae_encoder" in array.nodes]
    pools = {spec.resource_key: DenseWindowPool.build(
        spec, EngineResourceInfo(device=torch.device("cpu"))
    ) for spec in build_specs(arrays)}
    handles = node_handles(SimpleNamespace(declare_state=lambda: arrays), "vae_encoder", pools)
    handle = handles[STATE_NAME]
    boundary = handles[BOUNDARY_STATE_NAME]
    owners = EpisodeRowBindings()

    def call(rid, episode, values, *, boundary_value, reset=False):
        for pool in pools.values():
            pool.ingest_request(rid)
        binding = owners.bind(rid, 0, episode)
        for pool in pools.values():
            pool.bind_episode(rid, 0, episode, cold=binding.cold, reset=reset)
        step = state_steps(
            SubmoduleStep(
                segments=[Segment(rid, "main", len(values))],
                state={STATE_NAME: ArrayStep(),
                       BOUNDARY_STATE_NAME: ArrayStep(segments=(Segment(rid, "main", 1),))},
            ),
            handles,
            graph_walk="chunk",
        )
        step.set_ctx(StepContext(request_ids=[rid], graph_walk="chunk", slot=0, capture=False))
        for name, pool in pools.items():
            assert pool.admit(step.steps[name], step.ctx).ok
            pool.plan(step.steps[name], step.ctx)
        step.begin_calls()
        vals = torch.tensor(values, dtype=torch.uint8)
        previous = boundary.read(0).fields
        boundary.add(0, pose=torch.full((1, 8), float(boundary_value)),
                     history_valid=torch.ones((1, 1), dtype=torch.uint8))
        if values:
            handle.add(
                0,
                frame=vals[:, None, None, None].expand(-1, 3, 1, model.config.num_cameras).clone(),
                pose=vals[:, None].expand(-1, 8).float().clone(),
                pose_valid=torch.ones((len(values), 1), dtype=torch.uint8),
            )
        reached = handle.read(0)
        result = reached.fields["frame"][:, 0, 0, 0].tolist()
        assert reached.fields["pose"][:, 0].tolist() == [float(v) for v in result]
        saved_boundary = boundary.read(0).fields
        assert saved_boundary["pose"][-1, 0].item() == boundary_value
        if not reset and rid > 1:
            assert len(previous["pose"]) >= 1
        step.check_calls()
        for name, pool in pools.items():
            pool.commit(step.steps[name], step.ctx)
        step.commit_calls()
        step.end_calls()
        for pool in pools.values():
            pool.remove_request(rid)
        owners.unbind(rid)
        return result

    assert call(1, "episode-a", [], boundary_value=77) == []
    assert call(2, "episode-a", list(range(21)), boundary_value=21) == list(range(21))
    assert call(3, "episode-a", list(range(21, 45)), boundary_value=45) == list(range(45))
    assert call(4, "episode-a", list(range(45, 69)), boundary_value=69) == list(range(69))
    assert call(5, "episode-a", list(range(69, 93)), boundary_value=93) == list(range(24, 93))
    assert call(6, "episode-a", [], boundary_value=100, reset=True) == []


def test_praxis1_episode_idle_expiry_e2e_with_real_conductor_and_engine(tmp_path, monkeypatch):
    """Real Conductor + real Engine + all 5 Praxis-1 submodules across cold -> warm -> idle expiry -> re-encode."""
    now = 100.0
    monkeypatch.setattr(conductor_mod.time, "perf_counter", lambda: now)
    monkeypatch.setattr(episodes_mod.time, "perf_counter", lambda: now)

    model = Praxis1Model(
        "runwayml/praxis-1-droid", skip_weight_loading=True,
        max_context_chunks=1, max_text_tokens=16, num_steps=1,
    )
    model.config = replace(
        model.config,
        tile_height=32,
        tile_width=16,
        canvas_height=32,
        canvas_width=32,
        latent_channels=4,
        encoder_depth=1,
        decoder_depth=1,
        tap_blocks=(0,),
        expert_layers=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=4,
        expert_heads=2,
        expert_head_dim=4,
        text_hidden_dim=4,
        normalization={
            "proprio_mean": [0.0] * 10,
            "proprio_std": [1.0] * 10,
            "action_q01": [-1.0] * 10,
            "action_q99": [1.0] * 10,
        },
    )

    class _SeqTokenizer:
        def __call__(self, *, text, **kwargs):
            length = 2 if text[0].strip() else 1
            ids = torch.arange(1, length + 1, dtype=torch.long)[None]
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    model._tokenizer_cache = _SeqTokenizer()

    cfg_dict = {
        "max_seq_len": 16384,
        "node_groups": [
            {"node_names": ["text_encoder"], "ranks": [0]},
            {"node_names": ["vae_encoder"], "ranks": [0]},
            {"node_names": ["backbone", "action_expert", "action_head"], "ranks": [0]},
        ],
        "episodes": {"max_episodes": 1, "max_calls": 64, "idle_timeout_s": 10},
        "resources": {
            "executed_observations": {"max_slots": 2},
            "boundary_poses": {"max_slots": 2},
            "text_embeds": {"max_slots": 3},
        },
    }
    cfg_path = tmp_path / "praxis1_e2e.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))

    class _Comm:
        def __init__(self):
            self.sent = []

        def send(self, entity_id, msg):
            self.sent.append((entity_id, msg))

    comm = _Comm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    conductor = Conductor(model, str(cfg_path), socket_path_prefix=str(tmp_path / "sock"))

    setup, problems = state_setup(cfg_dict, model.declare_state())
    assert problems == []
    specs = build_specs(model.declare_state(), setup=setup)
    joint = SimpleNamespace(
        rank=0, world_size=1,
        tp_group=SimpleNamespace(barrier=lambda: None, world_size=1),
        sp_group=SimpleNamespace(world_size=1),
    )
    groups = SimpleNamespace(
        get_joint_group_for_node=lambda name: joint,
        get_instance_world_size_for_node=lambda name: 1,
        get_instance_rank_for_node=lambda name: 0,
        resource_needs_remote_transfer=lambda spec_nodes, relevant: False,
    )

    class _TextEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=3)

        def forward(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            h = torch.full((1, input_ids.shape[1], 4), 0.25, dtype=torch.bfloat16)
            return SimpleNamespace(hidden_states=[h, h, h])

    class _BackboneEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def forward_chunk_causal(self, text, anchor, clean, noised, times, *, attend_for_layer):
            k = times.shape[1]
            f = noised.shape[2] // k
            clean_len = 0 if clean is None else clean.shape[2]
            reg = model.config.num_registers
            anc = anchor.shape[-2] * anchor.shape[-1]
            g = reg + anc + text.shape[1]
            seq = g + noised.shape[2] + clean_len
            dummy = torch.zeros(1, model.config.num_heads, seq, model.config.head_dim)
            attend_for_layer(0)(dummy, dummy, dummy, None)
            emb = torch.zeros((1, seq, model.config.trunk_dim))
            features = TapFeatures(
                taps=[emb.clone()], embeddings=emb,
                frames_per_chunk=f, tokens_per_frame=1, rope_ids=torch.arange(seq)[None],
                global_spans={"register": (0, reg), "anchor": (reg, anc),
                              "text": (reg + anc, text.shape[1])},
                noised_span=(g, k * f), clean_span=(g + k * f, clean_len) if clean_len else None,
                n_chunks=k,
            )
            return torch.zeros((1, k * f, 1, 1, 4)), features

    class _BackboneDec(nn.Module):
        def forward_chunk_causal(self, noised, cond, times, *, attend_for_layer):
            del cond, times
            attend_for_layer(0)(noised, noised, noised, None)
            return torch.zeros_like(noised)

    class _Expert(nn.Module):
        global_order = ("register", "anchor", "text")
        num_future_per_chunk = 8

        def forward_trunk_rope(self, *, noised_action_latents, action_timesteps, proprio_latents,
                               trunk_features, global_tokens, kv_rope_ids, chunk_size,
                               num_clean_frames, num_noised_frames, tokens_per_frame, window_size,
                               cross_mask, self_mask, attend_for_layer):
            del action_timesteps, proprio_latents, num_clean_frames, num_noised_frames, tokens_per_frame, window_size
            del cross_mask, self_mask
            q_len = noised_action_latents.shape[1] // chunk_size * (1 + self.num_future_per_chunk + chunk_size)
            kv_len = global_tokens.shape[1] + sum(t.shape[1] for t in trunk_features)
            q = torch.zeros(1, 2, q_len, 4)
            kv = torch.zeros(1, 2, kv_len, 4)
            attend_for_layer(0, True)(q, kv, kv, _RESOURCE_MASK)
            attend_for_layer(1, False)(q, q, q, _RESOURCE_MASK)
            return torch.zeros_like(noised_action_latents)

    vae_node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), max_context_chunks=1)
    vae_node._tiles = lambda views: views
    vae_node.vae_encode = lambda canvas: torch.zeros(
        (1, 16, (canvas.shape[2] + 3) // 4, 1, 1), dtype=torch.float32,
    )
    submodules = {
        "text_encoder": Praxis1TextEncoderSubmodule(model.config, _TextEnc()),
        "vae_encoder": vae_node,
        "backbone": Praxis1BackboneSubmodule(
            model.config, SimpleNamespace(encoder=_BackboneEnc(), decoder=_BackboneDec()),
            episode_text=True,
        ),
        "action_expert": Praxis1ActionExpertSubmodule(model.config, _Expert()),
        "action_head": Praxis1ActionHeadSubmodule(model.config),
    }
    engine = Engine(None, state_setup=setup, autocast_dtype=None)
    engine.load_model(
        submodules=submodules,
        specs=specs,
        parallel_groups=groups,
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo("worker_0", "worker_0", LocalTransferEngine("localhost")),
        kv_cache_type=torch.float32,
        model=model,
    )

    def _ptr(uuid: int) -> TensorPointerInfo:
        return TensorPointerInfo(
            dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=[1],
            uuid=uuid, source_session_id="s", source_entity="worker_0",
        )

    def _run_node(node_name: str, rid_int: int, info, inp_tensors: dict, merged_held: dict):
        batch = ExecutingBatch(
            node_name,
            {rid_int: info},
            StepContext([rid_int], info.graph_walk, 0, False),
            per_request_input_tensors={rid_int: inp_tensors},
            per_request_input_metadata={
                rid_int: InputMetadata(dynamic_loop_iter_counts=dict(info.dynamic_loop_iter_counts)),
            },
        )
        engine.prepare_inputs(batch)
        assert not batch.failed_requests, batch.failed_requests
        out = engine.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        engine.finalize_batch(batch)
        merged_held.update(batch.episode_held.get(rid_int, {}))
        return out.per_rid_outputs[rid_int]

    def run_call(rid_str: str, rid_int: int, prompt: str, executed_rows: int, *,
                 prior_frames=None, prior_poses=None):
        sent_start = len(comm.sent)
        tensors = {
            "image_inputs": [torch.zeros((3, 32, 16), dtype=torch.uint8)] * 2,
        }
        if prior_frames is not None:
            tensors["executed_frame_inputs"] = prior_frames
        if prior_poses is not None:
            tensors["executed_pose_inputs"] = prior_poses
        prep = model.process_prompt(
            prompt, ["image", "text"], ["action"],
            robot_state=_pose(), action_format="positronic", tensors=tensors,
        )
        model_kwargs = {
            "executed_rows": executed_rows,
            "action_format": "positronic",
            "text_tokens": prep.metadata["text_tokens"],
            "blank_tokens": prep.metadata["blank_tokens"],
            "num_steps": 1,
        }
        initial_signals = {
            name: [_ptr(idx + 1)] for idx, name in enumerate(prep.new_input_tensors)
        }
        conductor._ingest_request(NewRequestConductor(
            rid_str, initial_signals, ["image", "text"], ["action"],
            prep.metadata, model_kwargs, episode="sess-1", episode_keys=prep.episode_keys,
        ))
        new_msg = next(
            msg.body for _, msg in comm.sent[sent_start:]
            if msg.message_type == WorkerMessageType.NEW_REQUEST and msg.body.request_id == rid_str
        )
        engine.add_request(
            rid_int,
            overrides=new_msg.request_info.resource_configs,
            request_info=new_msg.request_info,
            episode_id=new_msg.episode_id,
            episode_slot=new_msg.episode_slot,
            reset_episode=new_msg.reset_episode,
            episode_clear=new_msg.episode_clear,
        )

        walks = [new_msg.request_info.graph_walk]
        step_metas = [dict(new_msg.request_info.step_metadata)]
        merged_held: dict[str, int] = {}
        chunk_info = new_msg.request_info
        text_out: dict = {}

        if new_msg.request_info.graph_walk == model.ENCODE_TEXT_WALK:
            text_out = _run_node("text_encoder", rid_int, new_msg.request_info, prep.new_input_tensors, merged_held)
            sig_idx = len(comm.sent)
            wg_ids = sorted(conductor.requests[rid_str].partition_states["default"].current_worker_graph_ids)
            done_parts = conductor._process_worker_graphs_done(WorkerGraphsDone(
                rid_str, wg_ids, True, partition_name="default",
                persist_signals={"text_embeds": [_ptr(1)], "text_embeds_blank": [_ptr(2)]},
            ))
            for pname in done_parts:
                conductor._process_done_forward(rid_str, pname)
            sig_msg = next(
                msg.body for _, msg in comm.sent[sig_idx:]
                if msg.message_type == WorkerMessageType.INPUT_SIGNALS and msg.body.request_id == rid_str
            )
            chunk_info = sig_msg.request_info
            walks.append(chunk_info.graph_walk)
            step_metas.append(dict(chunk_info.step_metadata))

        vae_out = _run_node("vae_encoder", rid_int, chunk_info, prep.new_input_tensors, merged_held)
        step_info = replace(chunk_info, dynamic_loop_iter_counts={"praxis_denoise": 0})
        backbone_in = {**vae_out, **text_out}
        backbone_out = _run_node("backbone", rid_int, step_info, backbone_in, merged_held)
        expert_in = {**backbone_out, "proprio": [vae_out["proprio_all"][0][:, -1:]],
                     "proprio_all": vae_out["proprio_all"]}
        expert_out = _run_node("action_expert", rid_int, step_info, expert_in, merged_held)
        head_out = _run_node("action_head", rid_int, chunk_info, expert_out, merged_held)

        wg_ids = sorted(conductor.requests[rid_str].partition_states["default"].current_worker_graph_ids)
        done_parts = conductor._process_worker_graphs_done(WorkerGraphsDone(
            rid_str, wg_ids, True, partition_name="default", partition_done=True,
            episode_held=merged_held,
        ))
        conductor._record_episode_held(
            WorkerGraphsDone(rid_str, wg_ids, True, partition_name="default", episode_held=merged_held),
            conductor.requests[rid_str],
        )
        for pname in done_parts:
            conductor._process_done_forward(rid_str, pname)
        conductor._process_request_done(rid_str)
        conductor._handle_reads_done(ReadsDone(rid_str, conductor.PREPROCESS_WORKER))
        engine.remove_request(rid_int)
        return prep, walks, step_metas, merged_held, head_out["actions"][0]

    def _metric_map(counter) -> dict[str, float]:
        snap = counter.snapshot()
        return {labels[0]: val for labels, val in zip(snap.label_values, snap.values, strict=True)}

    # Call 1: cold "new" -> encode_text + chunk, 21 action rows, text & boundary retained.
    prep1, walks1, metas1, held1, act1 = run_call("c1", 1, "pick cup", executed_rows=0)
    assert walks1 == ["encode_text", "chunk"]
    assert [m[EPISODE_CALLS] for m in metas1] == [0, 0]
    assert metas1[-1]["history_append"] == 0
    assert act1.shape == (21, 10)
    assert held1 == {"executed_observations": 0, "boundary_poses": 1, "text_embeds": 3}
    assert _metric_map(conductor._episode_cold_starts) == {"new": 1.0}

    # Call 2: warm -> chunk_warm only (text encoding skipped), 21 frames appended, 24 action rows.
    prior_frames_2 = [torch.zeros((21, 3, 32, 16), dtype=torch.uint8)] * 2
    prior_poses_2 = [torch.tensor(_pose(), dtype=torch.float32).repeat(21, 1)]
    _, walks2, metas2, held2, act2 = run_call(
        "c2", 2, "pick cup", executed_rows=21,
        prior_frames=prior_frames_2, prior_poses=prior_poses_2,
    )
    assert walks2 == ["chunk_warm"]
    assert [m[EPISODE_CALLS] for m in metas2] == [1]
    assert metas2[0]["history_append"] == 21
    assert act2.shape == (24, 10)
    assert held2 == {"executed_observations": 21, "boundary_poses": 2, "text_embeds": 3}
    assert _metric_map(conductor._episode_cold_starts) == {"new": 1.0}

    # Idle expiry -> END_EPISODE, eviction metric "idle", Engine.end_episode releases all 3 pools.
    now = 112.0
    sent_before_expire = len(comm.sent)
    assert conductor._expire_episode_slots(now) is True
    end_msgs = [
        msg.body for _, msg in comm.sent[sent_before_expire:]
        if msg.message_type == WorkerMessageType.END_EPISODE
    ]
    assert end_msgs == [EndEpisode("sess-1", 0)]
    assert _metric_map(conductor._episode_evictions) == {"idle": 1.0}
    assert conductor._episode_slots.used_by_worker() == {}
    view_after_expire = conductor._episode_records.view(
        "sess-1", prep1.episode_keys, identity=(conductor._episode_weight_epoch, None),
    )
    assert view_after_expire.cold == "idle"
    assert dict(view_after_expire.held) == {
        "executed_observations": 0, "boundary_poses": 0, "text_embeds": 0,
    }
    assert engine.end_episode("sess-1", 0) is True

    # Call 3: cold after idle expiry -> encode_text re-runs, history_append resets to 0, 21 action rows.
    prior_frames_3 = [torch.zeros((24, 3, 32, 16), dtype=torch.uint8)] * 2
    prior_poses_3 = [torch.tensor(_pose(), dtype=torch.float32).repeat(24, 1)]
    _, walks3, metas3, held3, act3 = run_call(
        "c3", 3, "pick cup", executed_rows=24,
        prior_frames=prior_frames_3, prior_poses=prior_poses_3,
    )
    assert walks3 == ["encode_text", "chunk"]
    assert [m[EPISODE_CALLS] for m in metas3] == [2, 2]
    assert metas3[-1]["history_append"] == 0
    assert act3.shape == (21, 10)
    assert held3 == {"executed_observations": 0, "boundary_poses": 1, "text_embeds": 3}
    assert _metric_map(conductor._episode_cold_starts) == {"new": 1.0, "idle": 1.0}
