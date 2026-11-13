"""Pose-compose relative Praxis rows for Positronic command replies.

Ported from ``praxis.robot_io`` and ``praxis.single_arm``/``bimanual`` at
68f4d7a (Apache-2.0). The input pose is already in the model frame.
"""

from __future__ import annotations

import copy

import numpy as np

from mstar.model.praxis1.components.pose_frames import RIG_FRAMES, _no_turn, _quat_matrix
from mstar.model.praxis1.components.pose_frames import pose_to_model as pose_to_model  # noqa: PLC0414

# The single-arm reference's session option; the bimanual rig only accepts none.
CONTROL_MODES = {
    "none": None,
    "droid_impedance": {
        "type": "impedance",
        "kq": [40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0],
        "kqd": [4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0],
        "kx": [750.0, 750.0, 750.0, 15.0, 15.0, 15.0],
        "kxd": [37.0, 37.0, 37.0, 2.0, 2.0, 2.0],
    },
}


def _rot6d_matrix(r6: np.ndarray) -> np.ndarray:
    a0 = r6[:3].astype(np.float64)
    a1 = r6[3:6].astype(np.float64)
    c0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1 = a1 - np.dot(c0, a1) * c0
    c1 = a1 / (np.linalg.norm(a1) + 1e-8)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def compose_commands(
    relative: np.ndarray,
    pose_now: np.ndarray,
    embodiment: str,
    *,
    fps: float = 15.0,
    descriptor: str = "",
    control_mode: str = "none",
) -> list[dict]:
    """Compose model-frame relative rows against the current model-frame pose.

    ``pose_now`` is pose8 for a single arm or pose16 for bimanual. A frontend
    handling a client frame converts the pose to the model frame first and
    passes the client's descriptor back for command output.
    """
    dim = {"single_arm": 10, "bimanual": 20}.get(embodiment)
    if dim is None:
        raise ValueError(f"unsupported robot embodiment {embodiment!r}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("command fps must be finite and positive")
    try:
        rel = np.asarray(relative, dtype=np.float32)
        pose = np.asarray(pose_now, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("actions and pose must be numeric") from exc
    if rel.ndim != 2 or rel.shape[1] != dim or not np.isfinite(rel).all():
        raise ValueError(f"relative actions need finite rows of width {dim}")
    if pose.shape != (dim // 10 * 8,) or not np.isfinite(pose).all():
        raise ValueError(f"current pose needs {dim // 10 * 8} finite values")
    frame = RIG_FRAMES.get(descriptor) if embodiment == "single_arm" else None
    if embodiment == "single_arm" and frame is None:
        raise ValueError(f"unknown single-arm pose frame {descriptor!r}")
    if embodiment == "bimanual" and descriptor:
        raise ValueError("bimanual commands use the rig world frame, not a pose descriptor")
    supported_modes = CONTROL_MODES if embodiment == "single_arm" else {"none": None}
    if control_mode not in supported_modes:
        raise ValueError(f"unknown {embodiment} control mode {control_mode!r}")
    mode = copy.deepcopy(supported_modes[control_mode])

    anchors = [
        (pose[i : i + 3], _quat_matrix(pose[i + 3 : i + 7]))
        for i in range(0, pose.size, 8)
    ]
    dt = 1.0 / fps
    result = []
    for time, row in enumerate(rel):
        command: dict = {}
        for arm, (anchor_xyz, anchor_rot) in enumerate(anchors):
            start = arm * 10
            xyz = row[start : start + 3].astype(np.float64) + anchor_xyz
            rot = _rot6d_matrix(row[start + 3 : start + 9]) @ anchor_rot
            if frame is not None:
                xyz = xyz + frame.tool_offset * rot[:, 2] + np.asarray(frame.base_offset)
                rot = rot @ _no_turn.T @ np.asarray(frame.turn)
            target = {"type": "cartesian_pos", "pose": np.concatenate([xyz, rot.reshape(-1)]).astype(np.float32)}
            if mode is not None:
                target["mode"] = mode
            grip = float(row[start + 9])
            if embodiment == "single_arm":
                command["robot_command"] = target
                command["target_grip"] = grip
            else:
                side = "left" if arm == 0 else "right"
                command[f"robot_command.{side}"] = target
                command[f"target_grip.{side}"] = grip
        command["timestamp"] = float(time * dt)
        result.append(command)
    return result
