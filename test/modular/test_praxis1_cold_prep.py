"""Cold Praxis-1 input and sampler preparation on CPU."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from mstar.model.praxis1.components.proprio import normalized_proprio
from mstar.model.praxis1.components.sampler import (
    ACTION_SALT,
    VIDEO_SALT,
    seeded_normal,
    window_time_grid,
)
from mstar.model.praxis1.praxis1_model import DENOISE_LOOP_NAME, Praxis1Model


def _stats(width: int) -> dict[str, list[float]]:
    return {"proprio_mean": [0.0] * width, "proprio_std": [1.0] * width}


@pytest.mark.parametrize("embodiment", ["single_arm", "bimanual"])
def test_cold_pose_has_one_normalized_boundary(embodiment: str) -> None:
    raw = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5]
    if embodiment == "bimanual":
        raw = raw + [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.25]
    width = 10 if embodiment == "single_arm" else 20
    got = normalized_proprio(raw, embodiment, _stats(width))
    assert got.shape == (1, 1, width)
    assert got.dtype == torch.float32
    assert torch.equal(got[0, 0, :3], torch.tensor(raw[:3]))
    assert torch.equal(got[0, 0, 3:9], torch.tensor([1.0, 0, 0, 0, 1, 0]))
    if embodiment == "bimanual":
        assert torch.equal(got[0, 0, 10:13], torch.tensor(raw[8:11]))


@pytest.mark.parametrize(
    "bad", [None, [], [0.0] * 7, [float("nan")] * 8, ["bad"] * 8, [1e300] * 8]
)
def test_cold_pose_rejects_bad_client_state(bad: object) -> None:
    with pytest.raises(ValueError):
        normalized_proprio(bad, "single_arm", _stats(10))


def test_cold_pose_rejects_bad_checkpoint_stats() -> None:
    with pytest.raises(ValueError, match="normalization"):
        normalized_proprio([0.0] * 8, "single_arm", {})


def test_noise_salts_are_distinct_and_reproducible() -> None:
    video = seeded_normal((1, 3, 2), 42, VIDEO_SALT)
    action = seeded_normal((1, 3, 2), 42, ACTION_SALT)
    assert torch.equal(video, seeded_normal((1, 3, 2), 42, VIDEO_SALT))
    assert not torch.equal(video, action)
    assert video.device.type == action.device.type == "cpu"


def test_cold_pose_and_sampler_match_reference() -> None:
    pytest.importorskip("praxis", reason="reference checkout is not on PYTHONPATH")
    from praxis.robot_io import pose8_to_10d, proprio_normalize
    from praxis.sampler import seeded_normal as ref_noise
    from praxis.sampler import window_time_grid as ref_times

    pose = np.array([0.3, -0.7, 1.2, 0.6, -0.2, 0.4, 0.1, 0.35])
    stats = {"proprio_mean": [0.2] * 10, "proprio_std": [0.5] * 10}
    expected = proprio_normalize(
        pose8_to_10d(pose), np.array(stats["proprio_mean"]), np.array(stats["proprio_std"])
    )
    got = normalized_proprio(pose.tolist(), "single_arm", stats).numpy()[0, 0]
    assert np.array_equal(got, expected)
    for salt in (VIDEO_SALT, ACTION_SALT):
        assert torch.equal(seeded_normal((2, 3), 79, salt), ref_noise((2, 3), 79, salt))
    assert window_time_grid(3, 12.5) == tuple(ref_times(3, 12.5))


class _Tokenizer:
    def __call__(self, *, text: list[str], **kwargs):
        assert len(text) == 1
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}


def test_prompt_moves_pose_to_tensor_edge_and_walk_carries_recipe() -> None:
    model = Praxis1Model("runwayml/praxis-1-droid", skip_weight_loading=True)
    model.config = replace(model.config, normalization=_stats(10))
    model._tokenizer_cache = _Tokenizer()
    pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5]
    prompt = model.process_prompt("pick up cup", [], [], robot_state=pose)
    assert prompt.new_input_tensors["proprio"][0].shape == (1, 1, 10)
    assert "robot_state" in model.PREPROCESS_ONLY_KWARGS
    assert "proprio" in model.get_graph_walk_graphs()[model.CHUNK_WALK].get_nodes()["action_expert"].input_names

    plan = model.walk_plan("", ["image"], ["action"], {"image_inputs": [object()]}, {"num_steps": 1})
    assert len(plan) == 2
    assert plan[1].step_metadata == {
        "num_steps": 1,
        "t_shift": 12.5,
        "cfg_scale": 1.0,
        "cfg_video_scale": 4.0,
    }
    assert model.get_loop_caps(num_steps=1)[DENOISE_LOOP_NAME] == 1
