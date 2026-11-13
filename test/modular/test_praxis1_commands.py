"""Pose-composed Positronic commands from relative Praxis rows."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mstar.api_server.openpi import unpack
from mstar.model.praxis1.components.commands import compose_commands, pose_to_model
from mstar.model.praxis1.praxis1_model import MODEL_POSE_KWARG, Praxis1Model


def _pose(arms: int) -> np.ndarray:
    one = np.array([0.4, -0.2, 0.6, 0.8, 0.2, -0.3, 0.4, 0.25], dtype=np.float32)
    return np.concatenate([one + (0.1 if i else 0) for i in range(arms)])


def _rows(dim: int) -> np.ndarray:
    rows = np.random.default_rng(39).normal(size=(21, dim)).astype(np.float32)
    rows[:, 9::10] = np.linspace(0.1, 0.9, 21)[:, None]
    return rows


class _Tokenizer:
    def __call__(self, *, text: list[str], **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}


def _model() -> Praxis1Model:
    model = Praxis1Model("runwayml/praxis-1-droid", skip_weight_loading=True)
    model.config = replace(model.config, normalization={
        "proprio_mean": [0.0] * 10, "proprio_std": [1.0] * 10,
    })
    model._tokenizer_cache = _Tokenizer()
    return model


@pytest.mark.parametrize("descriptor", [
    "", "remote.robolab.droid", "remote.molmo_spaces.droid", "praxis.sim", "flange",
])
def test_client_pose_to_model_matches_reference(descriptor: str) -> None:
    pytest.importorskip("praxis", reason="reference checkout is not on PYTHONPATH")
    from praxis import single_arm

    raw = _pose(1)
    got = pose_to_model(raw, "single_arm", descriptor)
    expected = single_arm.pose8_to_model(raw, single_arm.RIG_FRAMES[descriptor])
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("descriptor", [
    "", "remote.robolab.droid", "remote.molmo_spaces.droid", "praxis.sim", "flange",
])
def test_single_arm_commands_match_reference_in_each_client_frame(descriptor: str) -> None:
    pytest.importorskip("praxis", reason="reference checkout is not on PYTHONPATH")
    from praxis import single_arm

    rows, pose = _rows(10), _pose(1)
    got = compose_commands(rows, pose, "single_arm", descriptor=descriptor)
    expected = single_arm.rel10_to_wire(rows, pose, frame=single_arm.RIG_FRAMES[descriptor])
    assert len(got) == len(expected) == 21
    for actual, want in zip(got, expected, strict=True):
        assert actual.keys() == want.keys()
        assert actual["timestamp"] == want["timestamp"]
        assert actual["target_grip"] == want["target_grip"]
        assert actual["robot_command"]["type"] == want["robot_command"]["type"]
        assert np.array_equal(actual["robot_command"]["pose"], want["robot_command"]["pose"])


def test_bimanual_commands_match_reference() -> None:
    pytest.importorskip("praxis", reason="reference checkout is not on PYTHONPATH")
    from praxis import bimanual

    rows, pose = _rows(20), _pose(2)
    got = compose_commands(rows, pose, "bimanual")
    expected = bimanual.rel20_to_wire(rows, pose)
    assert len(got) == len(expected) == 21
    for actual, want in zip(got, expected, strict=True):
        assert actual.keys() == want.keys()
        assert actual["timestamp"] == want["timestamp"]
        for side in ("left", "right"):
            assert actual[f"target_grip.{side}"] == want[f"target_grip.{side}"]
            assert np.array_equal(
                actual[f"robot_command.{side}"]["pose"], want[f"robot_command.{side}"]["pose"]
            )


def test_single_arm_impedance_mode_matches_reference() -> None:
    pytest.importorskip("praxis", reason="reference checkout is not on PYTHONPATH")
    from praxis import single_arm

    rows, pose = _rows(10), _pose(1)
    got = compose_commands(rows, pose, "single_arm", control_mode="droid_impedance")
    expected = single_arm.rel10_to_wire(
        rows, pose, mode=single_arm.control_mode_payload("droid_impedance"),
        frame=single_arm.RIG_FRAMES[""],
    )
    assert [r["robot_command"]["mode"] for r in got] == [
        r["robot_command"]["mode"] for r in expected
    ]
    assert all(np.array_equal(
        actual["robot_command"]["pose"], want["robot_command"]["pose"],
    ) for actual, want in zip(got, expected, strict=True))


def test_model_postprocess_keeps_openpi_relative_rows_and_packs_positronic_commands() -> None:
    model = _model()
    rows, pose = _rows(10), _pose(1)
    tensor = torch.from_numpy(rows)
    assert model.postprocess(tensor, "action") == rows.tobytes()

    kwargs = {
        "action_format": "positronic", "robot_state": pose.tolist(), "pose_descriptor": "praxis.sim",
        "control_mode": "droid_impedance",
    }
    with pytest.raises(ValueError, match="model-frame pose from process_prompt"):
        model.postprocess(tensor, "action", kwargs)
    prepared = model.process_prompt("pick", [], [], **kwargs)
    assert MODEL_POSE_KWARG in prepared.metadata
    packed = model.postprocess(tensor, "action", kwargs | prepared.metadata)
    commands = unpack(packed)
    assert isinstance(commands, list) and len(commands) == 21
    assert np.array_equal(
        commands[0]["robot_command"]["pose"],
        compose_commands(rows, pose_to_model(pose, "single_arm", "praxis.sim"),
                         "single_arm", descriptor="praxis.sim")[0]["robot_command"]["pose"],
    )
    assert commands[0]["robot_command"]["mode"]["type"] == "impedance"
    assert {"action_format", "pose_descriptor", "robot_state", "control_mode"} <= model.request_kwargs()
    assert {"action_format", "pose_descriptor", "robot_state", "control_mode"} <= set(model.PREPROCESS_ONLY_KWARGS)
    assert MODEL_POSE_KWARG in model.INTERNAL_KWARGS


def test_raw_fr3_hand_pose_round_trips_without_tool_offset() -> None:
    model = _model()
    raw = [0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0, 0.5]
    kwargs = {"robot_state": raw, "pose_descriptor": "", "action_format": "positronic"}
    prepared = model.process_prompt("pick", [], [], **kwargs)
    rows = np.zeros((21, 10), dtype=np.float32)
    rows[:, 3:9] = [1.0, 0, 0, 0, 1.0, 0]
    rows[:, 9] = 0.5
    command = unpack(model.postprocess(torch.from_numpy(rows), "action", kwargs | prepared.metadata))[0]
    assert command["robot_command"]["pose"][2] == np.float32(0.6)
    assert command["target_grip"] == 0.5


@pytest.mark.parametrize("bad_pose", [None, [], [0.0] * 7, [float("nan")] * 8, [0.0] * 8])
def test_commands_refuse_bad_pose(bad_pose) -> None:
    with pytest.raises(ValueError):
        compose_commands(_rows(10), bad_pose, "single_arm")
    with pytest.raises(ValueError):
        pose_to_model(bad_pose, "single_arm")


def test_commands_refuse_unknown_frame_and_action_format() -> None:
    with pytest.raises(ValueError, match="unknown single-arm pose frame"):
        compose_commands(_rows(10), _pose(1), "single_arm", descriptor="unknown")
    with pytest.raises(ValueError, match="rig world frame"):
        compose_commands(_rows(20), _pose(2), "bimanual", descriptor="remote.robolab.droid")
    with pytest.raises(ValueError, match="unknown bimanual control mode"):
        compose_commands(_rows(20), _pose(2), "bimanual", control_mode="droid_impedance")
    model = Praxis1Model("runwayml/praxis-1-droid", skip_weight_loading=True)
    with pytest.raises(ValueError, match="unsupported Praxis-1 action format"):
        model.postprocess(torch.from_numpy(_rows(10)), "action", {"action_format": "unknown"})
