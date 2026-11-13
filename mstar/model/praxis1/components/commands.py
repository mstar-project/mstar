"""Pose-compose relative Praxis rows for Positronic command replies.

Ported from ``praxis.robot_io`` and ``praxis.single_arm``/``bimanual`` at
68f4d7a (Apache-2.0). The input pose is already in the model frame.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigFrame:
    tool_offset: float
    turn: np.ndarray
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


_c, _s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
_flange_turn = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])
_no_turn = np.eye(3)
_turned_flange = RigFrame(0.0, _flange_turn)
_fr3_hand = RigFrame(0.1034, _flange_turn)
RIG_FRAMES = {
    "": _fr3_hand,
    "remote.robolab.droid": _fr3_hand,
    "remote.molmo_spaces.droid": RigFrame(0.155, _no_turn, (0.0, 0.0, 0.58)),
    "praxis.sim": _turned_flange,
    "flange": RigFrame(0.0, _no_turn),
}

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


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("robot pose has a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.array([
        [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2],
    ], dtype=np.float64)


def _matrix_quat(matrix: np.ndarray) -> np.ndarray:
    """A float64 rotation matrix to xyzw, with the reference's branch order."""
    diagonal = [matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[0, 0] + matrix[1, 1] + matrix[2, 2]]
    choice = int(np.argmax(diagonal))
    quat = np.empty(4, dtype=np.float64)
    if choice != 3:
        i, j, k = choice, (choice + 1) % 3, (choice + 2) % 3
        quat[i] = 1 - diagonal[3] + 2 * matrix[i, i]
        quat[j] = matrix[j, i] + matrix[i, j]
        quat[k] = matrix[k, i] + matrix[i, k]
        quat[3] = matrix[k, j] - matrix[j, k]
    else:
        quat[0] = matrix[2, 1] - matrix[1, 2]
        quat[1] = matrix[0, 2] - matrix[2, 0]
        quat[2] = matrix[1, 0] - matrix[0, 1]
        quat[3] = 1 + diagonal[3]
    x, y, z, w = quat
    return quat / np.sqrt(x * x + y * y + z * z + w * w)


def pose_to_model(pose_now: object, embodiment: str, descriptor: str = "") -> np.ndarray:
    """Convert a raw client-frame pose8 or world-frame pose16 to model space.

    This is the inverse of the frame conversion in :func:`compose_commands`.
    The source reference is ``praxis.single_arm.pose8_to_model``.
    """
    width = {"single_arm": 8, "bimanual": 16}.get(embodiment)
    if width is None:
        raise ValueError(f"unsupported robot embodiment {embodiment!r}")
    try:
        pose = np.asarray(pose_now, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("robot pose must be numeric") from exc
    if pose.shape != (width,) or not np.isfinite(pose).all():
        raise ValueError(f"robot pose needs {width} finite values")
    if max(abs(pose[3:7])) == 0 or (width == 16 and max(abs(pose[11:15])) == 0):
        raise ValueError("robot pose has a zero quaternion")
    if embodiment == "bimanual":
        if descriptor:
            raise ValueError("bimanual pose uses the rig world frame, not a pose descriptor")
        return pose.astype(np.float32)
    frame = RIG_FRAMES.get(descriptor)
    if frame is None:
        raise ValueError(f"unknown single-arm pose frame {descriptor!r}")
    rot = _quat_matrix(pose[3:7])
    result = pose.copy()
    result[:3] = pose[:3] - np.asarray(frame.base_offset) - frame.tool_offset * rot[:, 2]
    turn = np.asarray(frame.turn).T @ _no_turn
    xyzw = _matrix_quat(rot @ turn)
    result[3:7] = xyzw[[3, 0, 1, 2]]
    result[7] = np.clip(result[7], 0.0, 1.0)
    return result.astype(np.float32)


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
