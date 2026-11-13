"""The current robot pose as one normalized expert proprio row."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def _pose8_to_10d(pose: np.ndarray) -> np.ndarray:
    """Convert ``[xyz, quat_wxyz, grip]`` to ``[xyz, rot6d, grip]``."""
    q = pose[3:7] / (np.linalg.norm(pose[3:7]) + 1e-8)
    w, x, y, z = q
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z)],
            [2 * (x * z - w * y), 2 * (y * z + w * x)],
        ],
        dtype=np.float64,
    )
    return np.concatenate([pose[:3], rotation[:, 0], rotation[:, 1], pose[7:8]]).astype(
        np.float32
    )


def normalized_proprio(
    robot_state: object, embodiment: str, normalization: Mapping[str, object]
) -> torch.Tensor:
    """Return ``(1, 1, 10|20)`` float32 for a cold single-chunk call.

    The client pose is eight numbers per arm. A context window would supply
    earlier boundary poses too; this function deliberately covers only K=1.
    """
    width = {"single_arm": 8, "bimanual": 16}.get(embodiment)
    if width is None:
        raise ValueError(f"unsupported robot embodiment {embodiment!r}")
    if not isinstance(robot_state, (list, tuple)) or len(robot_state) != width:
        raise ValueError(f"robot_state needs {width} pose numbers")
    try:
        raw = np.asarray(robot_state, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("robot_state needs numeric pose values") from exc
    if raw.shape != (width,) or not np.isfinite(raw).all():
        raise ValueError("robot_state needs finite pose values")
    if np.max(np.abs(raw)) > np.finfo(np.float32).max:
        raise ValueError("robot_state cannot be represented as float32 proprio")
    rows = [_pose8_to_10d(raw[start : start + 8]) for start in range(0, width, 8)]
    pose = np.concatenate(rows)
    if not np.isfinite(pose).all():
        raise ValueError("robot_state cannot be represented as finite proprio")
    try:
        mean = np.asarray(normalization["proprio_mean"], dtype=np.float32)
        std = np.asarray(normalization["proprio_std"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint lacks proprio normalization statistics") from exc
    if mean.shape != pose.shape or std.shape != pose.shape or not (
        np.isfinite(mean).all() and np.isfinite(std).all()
    ):
        raise ValueError("checkpoint proprio statistics have the wrong shape or nonfinite values")
    normalized = (pose - mean) / np.maximum(std, np.float32(1e-6))
    if not np.isfinite(normalized).all():
        raise ValueError("robot_state cannot be represented as finite proprio")
    return torch.from_numpy(np.asarray(normalized, dtype=np.float32)).reshape(1, 1, -1)
