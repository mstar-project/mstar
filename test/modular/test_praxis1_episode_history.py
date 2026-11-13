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
    BOUNDARY_STATE_NAME,
    FIRST_ROWS,
    LATER_ROWS,
    STATE_NAME,
    append_count,
    window_size,
)
from mstar.model.praxis1.praxis1_model import TEXT_STATE_NAME, Praxis1Model
from mstar.model.praxis1.submodules import Praxis1VaeEncoderSubmodule


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

    def plan(held, rows, cold=None):
        episode = EpisodeView("session", 0, {STATE_NAME: held} if held else {}, cold)
        steps = model.walk_plan(
            "", ["image"], ["action"], {"image_inputs": [object()]},
            {"executed_rows": rows, "action_format": "positronic"}, episode
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


def test_warm_openpi_is_refused_before_any_graph_walk() -> None:
    model = _model()
    episode = EpisodeView("session", 1, {STATE_NAME: 1})
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

    def layout(held: int, rows: int):
        episode = EpisodeView("session", 0, {STATE_NAME: held} if held else {})
        walk = model.walk_plan(
            "", ["image"], ["action"], {"image_inputs": [object()]},
            {"executed_rows": rows, "text_tokens": 3, "blank_tokens": 2,
             "action_format": "positronic"}, episode,
        )[-1]
        assert "state_layout" in walk.step_metadata
        (wire_walk,) = _with_planned_state_layout([walk], model.declare_state())
        assert "state_layout" not in wire_walk.step_metadata
        return wire_walk.step_metadata

    cold, warm, reset = layout(0, 0), layout(1, FIRST_ROWS), layout(0, 0)
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
        EpisodeView("id", 1, {STATE_NAME: 1}),
    )
    assert plan[-1].step_metadata["history_append"] == 20

    node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), max_context_chunks=3)
    node._tiles = lambda views: views
    node.vae_encode = lambda canvas: canvas[:, :, ::4]
    seed_pose = torch.tensor([[3.0, 0, 0.6, 1, 0, 0, 0, 0.5]])
    handle = _Handle({
        "frame": torch.full((1, 3, 3, 8), 3, dtype=torch.uint8),
        "pose": seed_pose,
        "pose_valid": torch.ones((1, 1), dtype=torch.uint8),
    })
    boundary = _Handle({"pose": seed_pose, "history_valid": torch.ones((1, 1), dtype=torch.uint8)})
    node.state_handles = {STATE_NAME: handle, BOUNDARY_STATE_NAME: boundary}
    views = prepared.new_input_tensors["image_inputs"]
    out = node.forward(
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
    assert out["anchor_latents"][0].shape[2] == 1
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
                "pose": torch.empty((0, 8)),
                "history_valid": torch.empty((0, 1), dtype=torch.uint8),
            })
        fields = dict(self.prior)
        if self.added:
            for name, value in self.added[-1][1].items():
                fields[name] = torch.cat((fields[name], value)) if name in fields else value
        return SimpleNamespace(fields=fields)

    def commit(self):
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
        "chunk", SimpleNamespace(), history_append=1,
        history_pose=poses, history_pose_valid=valid,
        **{f"view_{i}": images for i in range(model.config.num_cameras)},
    )
    assert frames.added[-1][1]["pose_valid"].tolist() == [[1]]
    assert boundaries.added[-1][1]["history_valid"].tolist() == [[1]]

    frames.commit()
    boundaries.commit()
    missing = torch.zeros((22, 1), dtype=torch.uint8)
    missing[-1] = 1
    node.forward(
        "chunk", SimpleNamespace(), history_append=20,
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

    for current, first, count in ((0, 0, 0), (21, 0, 21), (45, 21, 24)):
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
        append = append_count(len(frames.prior.get("frame", [])), count)
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


@pytest.mark.parametrize("append,expected", [(1, [99]), (20, list(range(1, 21))), (24, list(range(21, 45)))])
def test_vae_state_adds_only_new_executed_rows(append, expected):
    model = _model()
    node = Praxis1VaeEncoderSubmodule(model.config, _Vae(), max_context_chunks=3)
    node._tiles = lambda views: views
    node.vae_encode = lambda canvas: canvas[:, :, ::4]
    prior_values = [] if append == 1 else ([0] if append == 20 else list(range(21)))
    prior_frame = torch.tensor(prior_values, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 1, 2)
    prior_pose = torch.tensor(prior_values, dtype=torch.float32)[:, None].expand(-1, 8)
    handle = _Handle({"frame": prior_frame, "pose": prior_pose,
                      "pose_valid": torch.ones((len(prior_values), 1), dtype=torch.uint8)})
    boundary_prior = [] if append == 1 else ([0] if append == 20 else [0, 21])
    boundary = _Handle({
        "pose": torch.tensor(boundary_prior, dtype=torch.float32)[:, None].expand(-1, 8),
        "history_valid": torch.ones((len(boundary_prior), 1), dtype=torch.uint8),
    })
    node.state_handles = {STATE_NAME: handle, BOUNDARY_STATE_NAME: boundary}
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
    arrays = [array for array in model.declare_state() if "vae_encoder" in array.nodes]
    pools = {spec.resource_key: DenseWindowPool.build(
        spec, EngineResourceInfo(device=torch.device("cpu"))
    ) for spec in build_specs(arrays)}
    handles = node_handles(SimpleNamespace(declare_state=lambda: arrays), "vae_encoder", pools)
    handle = handles[STATE_NAME]
    boundary = handles[BOUNDARY_STATE_NAME]
    owners = EpisodeRowBindings()

    def call(rid, episode, values, reset=False):
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
        boundary.add(0, pose=torch.full((1, 8), float(values[-1])),
                     history_valid=torch.ones((1, 1), dtype=torch.uint8))
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
        assert saved_boundary["pose"][-1, 0].item() == values[-1]
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

    assert call(1, "episode-a", [0]) == [0]
    assert call(2, "episode-a", list(range(1, 21))) == list(range(21))
    assert call(3, "episode-a", list(range(21, 45))) == list(range(45))
    assert call(4, "episode-a", list(range(45, 69))) == list(range(69))
    assert call(5, "episode-a", list(range(69, 93))) == list(range(24, 93))
    assert call(6, "episode-a", [100], reset=True) == [100]
