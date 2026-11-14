"""Convert client robot poses into the frame of the Praxis checkpoint.

The single-arm frame rules follow ``praxis.single_arm`` at 68f4d7a
(Apache-2.0). A missing descriptor names the FR3 hand, not the flange.
"""

from __future__ import annotations

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
_fr3_hand = RigFrame(0.1034, _flange_turn)
RIG_FRAMES = {
    "": _fr3_hand,
    "remote.robolab.droid": _fr3_hand,
    "remote.molmo_spaces.droid": RigFrame(0.155, _no_turn, (0.0, 0.0, 0.58)),
    "praxis.sim": RigFrame(0.0, _flange_turn),
    "flange": RigFrame(0.0, _no_turn),
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
    """Convert a float64 rotation matrix to xyzw in the reference's branch order."""
    diagonal = [
        matrix[0, 0], matrix[1, 1], matrix[2, 2],
        matrix[0, 0] + matrix[1, 1] + matrix[2, 2],
    ]
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
    """Convert a client pose8 or bimanual world-frame pose16 to model space."""
    width = {"single_arm": 8, "bimanual": 16}.get(embodiment)
    if width is None:
        raise ValueError(f"unsupported robot embodiment {embodiment!r}")
    try:
        pose = np.asarray(pose_now, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("robot pose must be numeric") from exc
    if pose.shape != (width,) or not np.isfinite(pose).all():
        raise ValueError(f"robot pose needs {width} finite values")
    if embodiment == "bimanual":
        if descriptor:
            raise ValueError("bimanual pose uses the rig world frame, not a pose descriptor")
        for start in (0, 8):
            norm = float(np.linalg.norm(pose[start + 3 : start + 7]))
            if abs(norm - 1.0) > 1e-2:
                raise ValueError("bimanual pose quaternion norm != 1")
        return pose
    frame = RIG_FRAMES.get(descriptor)
    if frame is None:
        raise ValueError(f"unknown single-arm pose frame {descriptor!r}")
    rot = _quat_matrix(pose[3:7])
    result = pose.astype(np.float64)
    result[:3] = pose[:3] - np.asarray(frame.base_offset) - frame.tool_offset * rot[:, 2]
    turn = np.asarray(frame.turn).T @ _no_turn
    xyzw = _matrix_quat(rot @ turn)
    result[3:7] = xyzw[[3, 0, 1, 2]]
    result[7] = np.clip(result[7], 0.0, 1.0)
    return result.astype(np.float32)
