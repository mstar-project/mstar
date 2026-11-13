"""Positronic input fields as the pinned Praxis single-arm and YAM rigs read them."""

import io

import msgpack
import numpy as np
import pytest
from PIL import Image

from mstar.api_server.openpi import pack
from mstar.api_server.positronic import _session_seed
from mstar.api_server.positronic_input import decode_praxis_frame


def test_session_seed_prefers_query_then_deployment_and_validates_int64(monkeypatch):
    monkeypatch.setenv("PRAXIS_SAMPLE_SEED", "23")
    assert _session_seed("17") == (17, "session_param")
    assert _session_seed(" +17 ") == (17, "session_param")
    assert _session_seed(None) == (23, "env")
    monkeypatch.delenv("PRAXIS_SAMPLE_SEED")
    base, source = _session_seed(None)
    assert 0 <= base < 2**31 and source == "rotated"
    for bad in ("1.5", "nan", "1_7", str(2**63)):
        with pytest.raises(ValueError, match="signed 64-bit"):
            _session_seed(bad)


def test_single_arm_uses_the_last_executed_frame_and_raw_client_pose():
    exterior = np.stack([np.full((4, 6, 3), n, np.uint8) for n in (10, 20)])
    wrist = np.stack([np.full((4, 6, 3), n, np.uint8) for n in (30, 40)])
    obs = {
        "image.exterior": exterior, "image.wrist": wrist,
        "robot_state.ee_pose": np.array([[0, 0, 0, 1, 0, 0, 0], [1, 2, 0.6, 1, 0, 0, 0]], np.float32),
        "robot_state.grip": np.array([0.1, 0.25], np.float32),
        "descriptor": "remote.robolab.droid",
    }
    decoded = decode_praxis_frame(pack({"session_id": "id", "observation": obs}), "single_arm")
    assert [tuple(image.shape) for image in decoded.images] == [(3, 4, 6), (3, 4, 6)]
    assert [int(image[0, 0, 0]) for image in decoded.images] == [20, 40]
    assert np.allclose(decoded.robot_state, [1, 2, 0.6, 1, 0, 0, 0, 0.25])
    assert decoded.pose_descriptor == "remote.robolab.droid"


def test_bimanual_uses_three_views_and_left_then_right_raw_poses():
    obs = {name: np.full((3, 4, 3), i, np.uint8) for i, name in enumerate(
        ("image.exterior", "image.wrist_left", "image.wrist_right"), 1
    )}
    for side, x in (("left", 1), ("right", 2)):
        obs[f"robot_state.{side}.ee_pose"] = [x, 0, 0, 1, 0, 0, 0]
        obs[f"grip.{side}"] = [0.1 * x]
    decoded = decode_praxis_frame(pack({"session_id": "id", "observation": obs}), "bimanual")
    assert [int(image[0, 0, 0]) for image in decoded.images] == [1, 2, 3]
    assert np.allclose(decoded.robot_state, [1, 0, 0, 1, 0, 0, 0, 0.1, 2, 0, 0, 1, 0, 0, 0, 0.2])
    assert decoded.pose_descriptor == ""


def test_jpeg_stack_decodes_before_selecting_the_latest_frame():
    def jpeg(value):
        buf = io.BytesIO()
        Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(buf, format="JPEG")
        return buf.getvalue()

    images = {b"__jpeg__": True, b"frames": [jpeg(0), jpeg(255)], b"ndim": 4}
    obs = {
        "image.exterior": images, "image.wrist": images,
        "ee_pose": [0, 0, 0, 1, 0, 0, 0], "grip": 0.5,
    }
    decoded = decode_praxis_frame(msgpack.packb({"session_id": "id", "observation": obs}), "single_arm")
    assert all(int(image[0, 0, 0]) >= 250 for image in decoded.images)


def test_decoded_array_budget_is_checked_before_media_loading(monkeypatch):
    import mstar.api_server.positronic_input as input_module

    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(buf, format="JPEG")
    image = {b"__jpeg__": True, b"frames": [buf.getvalue()], b"ndim": 3}
    obs = {"image.exterior": image, "image.wrist": image}
    frame = msgpack.packb({"session_id": "id", "observation": obs})
    assert len(frame) < 2000  # compressed input is below the cap
    monkeypatch.setattr(input_module, "MAX_DECODED_BYTES", 2000)
    with pytest.raises(ValueError, match="decoded arrays exceed"):
        decode_praxis_frame(frame, "single_arm")


@pytest.mark.parametrize("prior_rows", [0, 21, 24])
def test_reference_executed_window_excludes_the_current_row(prior_rows):
    frames = np.stack([np.full((3, 4, 3), i, np.uint8) for i in range(25)])
    obs = {
        "image.exterior": frames, "image.wrist": frames + 1,
        "robot_state.ee_pose": np.array([[i, 0, 0.6, 1, 0, 0, 0] for i in range(25)], np.float32),
        "grip": np.arange(25, dtype=np.float32) / 25,
    }
    result = decode_praxis_frame(pack({"session_id": "id", "observation": obs}), "single_arm", prior_rows)
    assert not result.reset_episode and result.executed_rows == prior_rows
    assert [int(image[0, 0, 0]) for image in result.images] == [24, 25]
    span = prior_rows
    assert [tuple(stack.shape) for stack in result.executed_frames] == [(span, 3, 3, 4)] * 2
    assert result.executed_poses.shape == (span, 8)
    assert result.executed_frames[0][:, 0, 0, 0].tolist() == list(range(24 - span, 24))
    assert result.executed_poses[:, 0].tolist() == list(range(24 - span, 24))


def test_missing_pose_history_keeps_the_reference_camera_window_warm():
    frames = np.stack([np.full((3, 4, 3), i, np.uint8) for i in range(22)])
    obs = {
        "image.exterior": frames, "image.wrist": frames + 1,
        "robot_state.ee_pose": [0, 0, 0.6, 1, 0, 0, 0], "grip": 0.25,
    }
    result = decode_praxis_frame(pack({"session_id": "id", "observation": obs}), "single_arm", 21)
    assert not result.reset_episode and result.executed_rows == 21
    assert [stack[:, 0, 0, 0].tolist() for stack in result.executed_frames] == [
        list(range(21)), list(range(1, 22)),
    ]
    assert result.executed_poses.shape == (0, 8)
    assert result.robot_state == [0, 0, pytest.approx(0.6), 1, 0, 0, 0, 0.25]


def test_short_executed_window_restarts_before_admission():
    frames = np.stack([np.full((3, 4, 3), i, np.uint8) for i in range(21)])
    obs = {
        "image.exterior": frames, "image.wrist": frames,
        "robot_state.ee_pose": np.array([[i, 0, 0.6, 1, 0, 0, 0] for i in range(21)], np.float32),
        "grip": np.zeros(21, np.float32),
    }
    result = decode_praxis_frame(pack({"session_id": "id", "observation": obs}), "single_arm", 21)
    assert result.reset_episode and result.executed_rows == 0
    assert all(stack.shape[0] == 0 for stack in result.executed_frames)
    assert result.executed_poses.shape == (0, 8)
    assert int(result.images[0][0, 0, 0]) == 20


def test_bimanual_executed_window_keeps_three_views_and_aligned_pose16():
    obs = {
        name: np.stack([np.full((3, 4, 3), i + offset, np.uint8) for i in range(25)])
        for offset, name in enumerate(("image.exterior", "image.wrist_left", "image.wrist_right"))
    }
    for side, offset in (("left", 0), ("right", 100)):
        obs[f"robot_state.{side}.ee_pose"] = np.array(
            [[i + offset, 0, 0.6, 1, 0, 0, 0] for i in range(25)], np.float32,
        )
        obs[f"grip.{side}"] = np.arange(25, dtype=np.float32) / 25
    result = decode_praxis_frame(pack({"session_id": "id", "observation": obs}), "bimanual", 24)
    assert [tuple(stack.shape) for stack in result.executed_frames] == [(24, 3, 3, 4)] * 3
    assert result.executed_poses.shape == (24, 16)
    assert result.executed_poses[0, [0, 8]].tolist() == [0, 100]
    assert result.executed_poses[-1, [0, 8]].tolist() == [23, 123]
    assert [int(image[0, 0, 0]) for image in result.images] == [24, 25, 26]


def test_bimanual_missing_pose_history_keeps_three_camera_windows():
    obs = {
        name: np.stack([np.full((3, 4, 3), i + offset, np.uint8) for i in range(22)])
        for offset, name in enumerate(("image.exterior", "image.wrist_left", "image.wrist_right"))
    }
    for side in ("left", "right"):
        obs[f"robot_state.{side}.ee_pose"] = [0, 0, 0.6, 1, 0, 0, 0]
        obs[f"grip.{side}"] = 0.25
    result = decode_praxis_frame(pack({"session_id": "id", "observation": obs}), "bimanual", 21)
    assert not result.reset_episode and result.executed_rows == 21
    assert [stack.shape[0] for stack in result.executed_frames] == [21] * 3
    assert result.executed_poses.shape == (0, 16)
