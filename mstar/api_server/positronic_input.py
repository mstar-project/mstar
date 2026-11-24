"""Decode a Praxis Positronic observation once in the shared data worker.

Both HTTP frontends forward the original checked msgpack frame here. The
returned pose is still in the client's frame; Praxis converts it in
``process_prompt`` before either proprio or action postprocessing uses it.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import msgpack
import numpy as np
import torch
from PIL import Image

MAX_DECODED_BYTES = 256 * 1024 * 1024
MAX_IMAGE_FRAMES = 25
VIEWS = {
    "single_arm": (
        ("exterior_image", "image.exterior", "observation.images.side"),
        ("wrist_image", "image.wrist", "observation.images.left"),
    ),
    "bimanual": (
        ("image.exterior", "image.top", "image.top_left"),
        ("image.wrist_left", "image.left_wrist"),
        ("image.wrist_right", "image.right_wrist"),
    ),
}


@dataclass(frozen=True)
class PraxisObservation:
    images: list[torch.Tensor]
    robot_state: list[float]
    pose_descriptor: str
    executed_frames: list[torch.Tensor]
    executed_poses: torch.Tensor
    executed_rows: int
    reset_episode: bool


class _Decoder:
    def __init__(self):
        self.decoded_bytes = 0

    def charge(self, size: int) -> None:
        self.decoded_bytes += size
        if self.decoded_bytes > MAX_DECODED_BYTES:
            raise ValueError("decoded arrays exceed the request byte limit")

    def pairs(self, pairs: list[tuple]) -> object:
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate msgpack key {key!r}")
            obj[key] = value
        markers = (b"__ndarray__", b"__npgeneric__", b"__jpeg__", b"__h264__")
        if sum(key in obj for key in markers) > 1:
            raise ValueError("conflicting wire markers")
        if b"__ndarray__" in obj:
            dtype = np.dtype(obj[b"dtype"])
            if dtype.kind not in "biuf":
                raise ValueError(f"unsupported array dtype {dtype}")
            shape, data = obj[b"shape"], obj[b"data"]
            if not isinstance(data, bytes) or not isinstance(shape, list) or not all(
                type(n) is int and n >= 0 for n in shape
            ) or math.prod(shape) * dtype.itemsize != len(data):
                raise ValueError("array data and shape disagree")
            self.charge(len(data))
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        if b"__npgeneric__" in obj:
            dtype = np.dtype(obj[b"dtype"])
            if dtype.kind not in "biuf":
                raise ValueError(f"unsupported scalar dtype {dtype}")
            return dtype.type(obj[b"data"])
        if b"__jpeg__" in obj:
            frames, ndim = obj[b"frames"], obj[b"ndim"]
            if type(ndim) is not int or ndim not in (3, 4) or not isinstance(frames, list) or not (
                1 <= len(frames) <= MAX_IMAGE_FRAMES
            ) or (ndim == 3 and len(frames) != 1):
                raise ValueError("JPEG stack must contain 1 to 25 frames")
            decoded = []
            for data in frames:
                if not isinstance(data, bytes):
                    raise ValueError("JPEG frame must be bytes")
                with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
                    self.charge(image.width * image.height * 3)
                    decoded.append(np.asarray(image.convert("RGB")))
            stacked = np.stack(decoded)
            return stacked if ndim == 4 else stacked[0]
        if b"__h264__" in obj:
            return self.h264(obj)
        return obj

    def h264(self, obj: dict) -> np.ndarray:
        import av  # H.264 is optional; only this codec needs PyAV

        shape, ndim, data = obj[b"shape"], obj[b"ndim"], obj[b"data"]
        if type(ndim) is not int or ndim not in (3, 4) or not isinstance(shape, list) or len(shape) != ndim:
            raise ValueError("H.264 shape must describe positive RGB dimensions")
        if not all(type(n) is int and n > 0 for n in shape) or shape[-1] != 3 or not isinstance(data, bytes):
            raise ValueError("H.264 shape must describe positive RGB dimensions")
        count, height, width = (shape[0], shape[1], shape[2]) if ndim == 4 else (1, shape[0], shape[1])
        if count > MAX_IMAGE_FRAMES:
            raise ValueError("H.264 stack must contain 1 to 25 frames")
        self.charge(count * height * width * 3)
        frames = []
        with av.open(io.BytesIO(data), mode="r", format="h264") as container:
            for frame in container.decode(video=0):
                if len(frames) == count or (frame.height, frame.width) != (
                    height + height % 2, width + width % 2,
                ):
                    raise ValueError("H.264 frame dimensions do not match the declared shape")
                frames.append(frame.to_ndarray(format="rgb24")[:height, :width])
        if len(frames) != count:
            raise ValueError(f"H.264 stack decoded to {len(frames)} frames, expected {count}")
        stacked = np.stack(frames)
        return stacked if ndim == 4 else stacked[0]


def _pick(obs: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in obs:
            return obs[key]
    raise ValueError(f"observation needs one of {keys}")


def _last_pose(obs: dict, keys: tuple[str, ...]) -> list[float]:
    value = np.asarray(_pick(obs, keys), dtype=np.float32)
    if value.ndim >= 2:
        value = value[-1]
    value = value.reshape(-1)
    if value.size < 7 or not np.isfinite(value[:7]).all():
        raise ValueError("ee_pose needs seven finite values")
    return value[:7].astype(np.float64).tolist()


def _last_grip(obs: dict, keys: tuple[str, ...]) -> float:
    value = np.asarray(_pick(obs, keys), dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.isfinite(value[-1]):
        raise ValueError("grip needs a finite value")
    return float(value[-1])


def _pose_rows(obs: dict, pose_keys: tuple[str, ...], grip_keys: tuple[str, ...]) -> np.ndarray | None:
    """Return aligned raw pose rows, or None when a history stack is absent."""
    try:
        pose = np.asarray(_pick(obs, pose_keys), dtype=np.float32)
        grip = np.asarray(_pick(obs, grip_keys), dtype=np.float32)
    except ValueError:
        return None
    if pose.ndim == 1:
        pose = pose[None]
    if pose.ndim != 2 or pose.shape[1] < 7:
        return None
    if grip.ndim == 0:
        grip = grip.reshape(1, 1)
    elif grip.ndim == 1:
        grip = grip.reshape(-1, 1)
    elif grip.ndim == 2:
        grip = grip[:, :1]
    else:
        return None
    if grip.shape[0] != pose.shape[0] or not np.isfinite(pose[:, :7]).all() or not np.isfinite(grip).all():
        return None
    return np.concatenate((pose[:, :7], grip), axis=1)


def decode_praxis_frame(frame: bytes, embodiment: str, expected_prior_rows: int = 0) -> PraxisObservation:
    """Read current inputs and the executed stack preceding the current frame.

    ``expected_prior_rows`` comes from the previous served action count. A
    missing or short stack restarts the episode before conductor admission.
    """
    if embodiment not in VIEWS:
        raise ValueError(f"unsupported Praxis embodiment {embodiment!r}")
    if type(expected_prior_rows) is not int or expected_prior_rows not in (0, 21, 24):
        raise ValueError("expected_prior_rows must be 0, 21 or 24")
    if len(frame) > MAX_DECODED_BYTES:
        raise ValueError("Positronic frame exceeds the request byte limit")
    decoder = _Decoder()
    try:
        request = msgpack.unpackb(frame, object_pairs_hook=decoder.pairs, raw=False, strict_map_key=False)
    except (msgpack.UnpackException, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid Positronic observation: {exc}") from None
    obs = request.get("observation") if isinstance(request, dict) else None
    if not isinstance(obs, dict):
        raise ValueError("a v2 request carries an observation dict")
    images = []
    stacks = []
    for candidates in VIEWS[embodiment]:
        value = np.asarray(_pick(obs, candidates))
        if value.ndim == 3:
            value = value[None]
        if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8 or value.shape[0] < 1:
            raise ValueError(f"{candidates[0]} must be RGB uint8 HWC")
        stacks.append(value)
        images.append(torch.from_numpy(np.array(value[-1].transpose(2, 0, 1), order="C", copy=True)))
    if embodiment == "single_arm":
        robot_state = [
            *_last_pose(obs, ("ee_pose", "robot_state.ee_pose")),
            _last_grip(obs, ("grip", "robot_state.grip", "gripper")),
        ]
    else:
        robot_state = []
        for side in ("left", "right"):
            robot_state.extend(_last_pose(obs, (f"robot_state.{side}.ee_pose", f"ee_pose.{side}")))
            robot_state.append(_last_grip(obs, (f"grip.{side}", f"robot_state.{side}.grip", f"gripper.{side}")))
    if embodiment == "single_arm":
        pose_rows = _pose_rows(
            obs, ("ee_pose", "robot_state.ee_pose"), ("grip", "robot_state.grip", "gripper"),
        )
    else:
        arms = [
            _pose_rows(
                obs, (f"robot_state.{side}.ee_pose", f"ee_pose.{side}"),
                (f"grip.{side}", f"robot_state.{side}.grip", f"gripper.{side}"),
            ) for side in ("left", "right")
        ]
        pose_rows = None if any(rows is None for rows in arms) else np.concatenate(
            [rows[-min(len(rows) for rows in arms):] for rows in arms], axis=1
        )
    # The reference uses exactly the requested suffix before the current row.
    # Missing pose history does not invalidate a complete camera clip.
    reset_episode = any(len(stack) < expected_prior_rows + 1 for stack in stacks)
    selected_rows = 0 if reset_episode else expected_prior_rows
    executed_frames = [
        torch.from_numpy(np.array(stack[-(selected_rows + 1):-1].transpose(0, 3, 1, 2), order="C", copy=True))
        if selected_rows else torch.empty((0, *image.shape), dtype=torch.uint8)
        for stack, image in zip(stacks, images, strict=True)
    ]
    pose_dim = 8 if embodiment == "single_arm" else 16
    executed_poses = (
        torch.from_numpy(np.array(pose_rows[-(selected_rows + 1):-1], dtype=np.float32, copy=True))
        if selected_rows and pose_rows is not None and len(pose_rows) >= selected_rows + 1
        else torch.empty((0, pose_dim), dtype=torch.float32)
    )
    descriptor = obs.get("descriptor") or ""
    if isinstance(descriptor, bytes):
        descriptor = descriptor.decode("utf-8", "replace")
    if not isinstance(descriptor, str):
        raise ValueError("descriptor must be a string")
    return PraxisObservation(
        images, robot_state, descriptor, executed_frames, executed_poses,
        0 if reset_episode else expected_prior_rows, reset_episode,
    )
