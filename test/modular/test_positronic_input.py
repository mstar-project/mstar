"""Positronic input fields as the pinned Praxis single-arm and YAM rigs read them."""

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import numpy as np
import pytest
import yaml
from PIL import Image

from mstar.api_server.openpi import pack
from mstar.api_server.positronic import PositronicPolicy, _session_seed
from mstar.api_server.positronic_input import decode_praxis_frame
from mstar.api_server.rust_frontend import positronic_facts


def test_droid_ready_metadata_declares_the_reference_client_stack():
    config = yaml.safe_load((Path(__file__).parents[2] / "configs/praxis1_droid.yaml").read_text())
    policy = PositronicPolicy.from_config(config["positronic"], SimpleNamespace(POSITRONIC_SESSION_READY=True))
    meta = policy.meta
    assert meta["chunk_rows"] == 24
    assert meta["action_fps"] == 15.0
    assert meta["image_codecs"] == ["h264", "jpeg"]
    stack = meta["local_stack"]["seq"]
    assert [step["name"] for step in stack] == [
        "stop_on_fault", "restrict_image_size", "temporal_stack", "chunked_schedule",
    ]
    assert [step.get("version", 1) for step in stack] == [2, 1, 2, 2]
    assert stack[3]["args"] == {"fps": meta["action_fps"]}
    assert stack[1]["args"] == {"width": 416, "height": 234}
    assert stack[2]["args"]["keys"] == [
        "image.wrist", "image.exterior", "robot_state.ee_pose", "grip",
    ]
    assert stack[2]["args"]["offsets_sec"] == pytest.approx(
        [0.0 if frame == 0 else -frame / 15 + 0.5 / 15 for frame in range(24, -1, -1)]
    )
    assert stack[2]["args"]["pad_start"] is False
    assert positronic_facts(policy)["meta"] == meta


def test_droid_ready_metadata_builds_the_stock_v2_client_stack():
    remote = pytest.importorskip("positronic.policy.remote", reason="stock Positronic client is optional")
    client = pytest.importorskip("positronic.offboard.client", reason="stock Positronic client is optional")
    wire = pytest.importorskip("positronic_wire.wire", reason="stock Positronic client is optional")
    config = yaml.safe_load((Path(__file__).parents[2] / "configs/praxis1_droid.yaml").read_text())
    meta = config["positronic"]["meta"]
    sid = "a" * 32
    connection = MagicMock(spec=wire.ClientConnection)
    connection.recv.return_value = msgpack.packb({
        "status": "ready", "meta": meta, "protocol_version": 2,
        "session_id": sid, "max_in_flight": 1,
    }, use_bin_type=True)
    session = client.InferenceSession(connection)
    assert session.session_id == sid
    stack = remote.declared_stack(session.metadata, session.protocol_version)
    assert isinstance(stack, remote.Processor)


def test_droid_v2_stack_emits_only_robot_commands():
    remote = pytest.importorskip("positronic.policy.remote", reason="stock Positronic client is optional")
    executor = pytest.importorskip("positronic.policy.executor", reason="stock Positronic client is optional")
    base = pytest.importorskip("positronic.policy.base", reason="stock Positronic client is optional")
    offboard = pytest.importorskip("positronic.offboard.protocol", reason="stock Positronic client is optional")
    wire = pytest.importorskip("positronic_wire.wire", reason="stock Positronic client is optional")
    from mstar.model.praxis1.components.commands import compose_commands

    config = yaml.safe_load((Path(__file__).parents[2] / "configs/praxis1_droid.yaml").read_text())
    session = MagicMock()
    session.metadata = config["positronic"]["meta"] | {"compress_images": False}
    session.protocol_version = pytest.importorskip("positronic_model_server.protocol").ProtocolVersion.V2
    session.max_in_flight = 1
    session.session_id = "a" * 32
    relative = np.zeros((21, 10), dtype=np.float32)
    relative[:, 3] = relative[:, 7] = 1
    session.infer.return_value = offboard.typed_commands(compose_commands(
        relative, np.array([0, 0, 0.6, 1, 0, 0, 0, 0.5]), "single_arm",
        control_mode=config["positronic"]["control_mode"],
    ))
    policy = remote.RemotePolicy("websocket", wire.HostPortAddress("localhost", 8000, wire.SESSION_PATH, ""))
    policy._client = MagicMock()
    policy._client.new_session.return_value = session
    runtime = executor.Executor(lambda: 0, simulated=True, charge_inference_time=False)
    run = runtime.start(policy)
    obs = {
        "image.wrist": np.zeros((234, 416, 3), dtype=np.uint8),
        "image.exterior": np.zeros((234, 416, 3), dtype=np.uint8),
        "robot_state.ee_pose": np.array([0, 0, 0.6, 1, 0, 0, 0], dtype=np.float32),
        "grip": np.float32(0.5),
    }
    try:
        assert isinstance(run.send(obs), base.Step)
        assert runtime.wait(timeout_sec=5).status is executor.WaitStatus.ANSWERS_READY
        step = run.send(obs)
        assert isinstance(step, base.Step)
        assert set(step.commands) == {"robot_command", "target_grip"}
        assert session.infer.call_count == 1
    finally:
        runtime.close()
        run.close()


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


@pytest.mark.parametrize("expected_rows,frame_count", [(21, 21), (24, 22)])
def test_short_executed_window_restarts_before_admission(expected_rows, frame_count):
    frames = np.stack([np.full((3, 4, 3), i, np.uint8) for i in range(frame_count)])
    obs = {
        "image.exterior": frames, "image.wrist": frames,
        "robot_state.ee_pose": np.array([[i, 0, 0.6, 1, 0, 0, 0] for i in range(frame_count)], np.float32),
        "grip": np.zeros(frame_count, np.float32),
    }
    result = decode_praxis_frame(pack({"session_id": "id", "observation": obs}), "single_arm", expected_rows)
    assert result.reset_episode and result.executed_rows == 0
    assert all(stack.shape[0] == 0 for stack in result.executed_frames)
    assert result.executed_poses.shape == (0, 8)
    assert int(result.images[0][0, 0, 0]) == frame_count - 1


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
