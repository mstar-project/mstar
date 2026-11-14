"""Deterministic DROID WebSocket probes for the e2e matrix.

Each Positronic item is one cold, two warm, and one reset call on the same
connection. The saved float32 output contains all four command sets, in order.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
import re
import time
from functools import lru_cache
from typing import Any

import msgpack
import numpy as np

from benchmark.eval.datasets import Item

PHASES = (("cold", 21, False), ("warm1", 24, False), ("warm2", 24, False), ("reset", 21, True))
OPENPI_ROWS = (21, 10)
CAMERA_SIZE = (234, 416)


class ProtocolError(ValueError):
    """The served policy did not satisfy the selected wire contract."""


def _pack_array(obj: Any) -> dict:
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(f"cannot encode {type(obj).__name__}")


def pack(value: Any) -> bytes:
    return msgpack.packb(value, default=_pack_array, use_bin_type=True)


def _unpack_array(value: dict) -> Any:
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        shape = tuple(value[b"shape"])
        if math.prod(shape) * dtype.itemsize != len(value[b"data"]):
            raise ProtocolError("array byte count does not match its shape")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(shape)
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_unpack_array, raw=False, strict_map_key=False)


def _ws_url(url: str, path: str) -> str:
    if url.startswith("http://"):
        return "ws://" + url[7:].rstrip("/") + path
    if url.startswith("https://"):
        return "wss://" + url[8:].rstrip("/") + path
    raise ValueError("the e2e server URL must be HTTP or HTTPS")


async def _recv(ws, timeout_s: float) -> Any:
    data = await asyncio.wait_for(ws.recv(), timeout_s)
    if not isinstance(data, bytes):
        raise ProtocolError(f"expected a binary MessagePack reply, got {type(data).__name__}")
    return unpack(data)


@lru_cache(maxsize=256)
def _camera(index: int, call: int, view: int, frame: int) -> bytes:
    from PIL import Image

    h, w = CAMERA_SIZE
    color = (17 * index + 29 * call + 7 * frame + 53 * view) % 256
    pixels = np.full((h, w, 3), color, dtype=np.uint8)
    pixels[:, ::16, 1] = (color + 40) % 256
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@lru_cache(maxsize=16)
def _openpi_observation(index: int) -> bytes:
    h, w = CAMERA_SIZE
    exterior = np.full((h, w, 3), (index * 17 + 37) % 256, dtype=np.uint8)
    wrist = np.full((h, w, 3), (index * 17 + 91) % 256, dtype=np.uint8)
    return pack({
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/cartesian_position": np.array([0, 0, 0.6, 1, 0, 0, 0], np.float32),
        "observation/gripper_position": np.array([0.5], np.float32),
        "prompt": "pick up the red block", "seed": 1234 + index,
    })


def _positronic_observation(index: int, call: int, reset: bool) -> dict:
    images = {
        name: {b"__jpeg__": True, b"frames": [_camera(index, call, view, frame) for frame in range(25)],
               b"ndim": 4}
        for view, name in enumerate(("image.exterior", "image.wrist"))
    }
    poses = np.tile(np.array([0, 0, 0.6, 1, 0, 0, 0], np.float32), (25, 1))
    poses[:, 0] = index * 0.01 + call * 0.1 + np.arange(25, dtype=np.float32) * 0.001
    return {**images, "robot_state.ee_pose": poses, "grip": np.full(25, 0.5, np.float32),
            "descriptor": "remote.robolab.droid", "task": "pick up the red block",
            **({"reset_context": 1} if reset else {})}


def _ready(value: Any, index: int) -> str:
    if not isinstance(value, dict) or value.get("status") != "ready" or value.get("protocol_version") != 2:
        raise ProtocolError("Positronic did not send protocol v2 ready metadata")
    sid = value.get("session_id")
    if not isinstance(sid, str) or re.fullmatch(r"[0-9a-f]{32}", sid) is None:
        raise ProtocolError("Positronic ready frame has no 32-hex session id")
    meta = value.get("meta")
    if not isinstance(meta, dict) or meta.get("sample_seed_base") != 1234 + index:
        raise ProtocolError("Positronic ready metadata has the wrong session seed")
    if (meta.get("chunk_rows") != 24 or meta.get("action_fps") != 15.0
            or meta.get("image_codecs") != ["h264", "jpeg"] or meta.get("compress_images") is not True):
        raise ProtocolError("Positronic ready metadata has the wrong DROID schedule")
    stack = meta.get("local_stack")
    seq = stack.get("seq") if isinstance(stack, dict) else None
    if not isinstance(seq, list) or [s.get("name") for s in seq if isinstance(s, dict)] != [
        "stop_on_fault", "restrict_image_size", "temporal_stack", "chunked_schedule",
    ]:
        raise ProtocolError("Positronic ready metadata has no DROID client stack")
    if [s.get("version", 1) for s in seq] != [2, 1, 2, 2] or seq[3].get("args") != {"fps": 15.0}:
        raise ProtocolError("Positronic ready metadata has an incompatible v2 client stack")
    resize, temporal = seq[1].get("args"), seq[2].get("args")
    offsets = temporal.get("offsets_sec") if isinstance(temporal, dict) else None
    expected = [0.0 if frame == 0 else -frame / 15 + 0.5 / 15 for frame in range(24, -1, -1)]
    if (resize != {"width": 416, "height": 234} or not isinstance(temporal, dict)
            or temporal.get("keys") != ["image.wrist", "image.exterior", "robot_state.ee_pose", "grip"]
            or temporal.get("pad_start") is not False or not isinstance(offsets, list)
            or len(offsets) != 25 or not np.allclose(offsets, expected)):
        raise ProtocolError("Positronic ready metadata has the wrong DROID client stack")
    return sid


def _commands(reply: Any, expected: int) -> np.ndarray:
    rows = reply.get("result") if isinstance(reply, dict) else None
    if not isinstance(rows, list) or len(rows) != expected:
        count = len(rows) if isinstance(rows, list) else "no"
        raise ProtocolError(f"Positronic returned {count} commands, not {expected}")
    values = []
    for row in rows:
        command = row.get("robot_command") if isinstance(row, dict) else None
        pose = np.asarray(command.get("pose"), dtype=np.float32) if isinstance(command, dict) else np.empty(0)
        if not isinstance(command, dict) or command.get("type") != "cartesian_pos" or pose.shape != (12,):
            raise ProtocolError("Positronic command needs a Cartesian pose of 12 floats")
        if "timestamp" in row:
            raise ProtocolError("Positronic v2 command contains an unsupported timestamp")
        try:
            grip = float(row["target_grip"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Positronic command needs grip") from exc
        vals = np.concatenate((pose, np.array([grip], np.float32)))
        if not np.isfinite(vals).all():
            raise ProtocolError("Positronic command contains nonfinite values")
        values.append(vals)
    return np.stack(values)


def _row(item: Item, t0: float, blob: bytes, chunks: int, **phases: float) -> dict:
    elapsed = time.perf_counter() - t0
    return {"id": item.id, "ok": True, "error": None, "e2e_s": elapsed, "ttft_s": phases.get("_cold_s", elapsed),
            "text": "", "n_chunks": chunks, "audio_s": 0.0, "out_sha": hashlib.sha256(blob).hexdigest()[:16],
            "_audio": b"", "_blob": blob, **phases}


async def run_openpi(_session, url: str, index: int, item: Item, t0: float, timeout) -> dict:
    import websockets

    async with websockets.connect(_ws_url(url, "/"), compression=None, max_size=None,
                                  open_timeout=timeout.total) as ws:
        meta = await _recv(ws, timeout.total)
        if not isinstance(meta, dict) or meta.get("checkpoint") != "runwayml/praxis-1-droid":
            raise ProtocolError("openpi did not send the DROID policy metadata")
        observation = _openpi_observation(index)
        started = time.perf_counter()
        await ws.send(observation)
        reply = await _recv(ws, timeout.total)
        actions = reply.get("actions") if isinstance(reply, dict) else None
        if not isinstance(actions, np.ndarray) or actions.shape != OPENPI_ROWS or not np.isfinite(actions).all():
            raise ProtocolError("openpi actions need 21 finite rows of width 10")
        return _row(item, t0, np.ascontiguousarray(actions, dtype="<f4").tobytes(), 21,
                    _cold_s=time.perf_counter() - started)


async def run_positronic(_session, url: str, index: int, item: Item, t0: float, timeout) -> dict:
    import websockets

    async with websockets.connect(_ws_url(url, f"/api/v1/session?sample_seed={1234 + index}"),
                                  compression=None, max_size=None, open_timeout=timeout.total) as ws:
        loading = await _recv(ws, timeout.total)
        if not isinstance(loading, dict) or loading.get("status") != "loading":
            raise ProtocolError("Positronic did not send a loading frame")
        sid = _ready(await _recv(ws, timeout.total), index)
        outputs, times = [], {}
        for call, (phase, expected, reset) in enumerate(PHASES):
            frame = pack({"session_id": sid, "observation": _positronic_observation(index, call, reset)})
            started = time.perf_counter()
            await ws.send(frame)
            outputs.append(_commands(await _recv(ws, timeout.total), expected))
            times[f"_{phase}_s"] = time.perf_counter() - started
        await ws.send(pack({"session_id": sid, "end_session": True}))
        if await _recv(ws, timeout.total) != {"session_id": sid, "end_session": True}:
            raise ProtocolError("Positronic did not acknowledge end_session")
        blob = np.ascontiguousarray(np.concatenate(outputs), dtype="<f4").tobytes()
        return _row(item, t0, blob, sum(len(out) for out in outputs), **times)
