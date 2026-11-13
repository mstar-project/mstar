"""Positronic input fields as the pinned Praxis single-arm and YAM rigs read them."""

import io

import msgpack
import numpy as np
import pytest
from PIL import Image

from mstar.api_server.openpi import pack
from mstar.api_server.positronic_input import decode_praxis_frame


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
