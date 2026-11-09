"""openpi's WebSocket policy protocol, for robot clients written against openpi.

openpi (Physical Intelligence's reference stack) serves a policy with
``scripts/serve_policy.py``. Its client, ``openpi_client.WebsocketClientPolicy``
(the DROID, LIBERO and ALOHA examples use it), opens a WebSocket at the
server's root, reads one metadata frame, then sends an observation and waits
for its reply, one call at a time::

    server -> client   msgpack(metadata)                         once, on connect
    client -> server   msgpack({"observation/...": ndarray, ..., "prompt": str})
    server -> client   msgpack({"actions": ndarray [H, D], "server_timing": {...}})

Arrays are encoded as ``openpi_client.msgpack_numpy`` encodes them: a map of
``__ndarray__``, ``data`` (the raw bytes), ``dtype`` and ``shape``. A call that
fails is answered with a text frame, which the client raises, and the socket
closes, as openpi's server does.

A deployment opts in with an ``openpi:`` block in its serving config. The block
does the job of openpi's per-robot transforms (``DroidInputs``, ``DroidOutputs``):
it names the observation keys that are the cameras, in slot order, the keys whose
values are joined into the state, and how many action dims a reply keeps. The
images reach the model decoded, with no file in between
(``APIServer.submit_request(raw_inputs=...)``).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import msgpack
import numpy as np
import torch
from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from mstar.model.base import Model
from mstar.utils.client_errors import client_error_text

logger = logging.getLogger(__name__)

# Array dtypes an observation may carry: bool, integers and floats.
_NUMERIC_KINDS = frozenset("biuf")
_CONFIG_KEYS = frozenset({"images", "state", "action_dims", "metadata"})


# ----------------------------------------------------------------------
# Wire format: openpi_client.msgpack_numpy
# ----------------------------------------------------------------------

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(f"cannot encode {type(obj).__name__}")


def _numeric_dtype(name) -> np.dtype:
    # dtype.str, as openpi packs it: np.dtype(None) would quietly be float64
    if not isinstance(name, str):
        raise ValueError(f"an array's dtype must be a dtype string, not {name!r}")
    try:
        dtype = np.dtype(name)
    except TypeError:
        raise ValueError(f"unknown array dtype {name!r}") from None
    if dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"array dtype {dtype.str} is not bool, integer or float")
    return dtype


def _unpack_array(obj: dict):
    if b"__ndarray__" in obj:
        if not {b"data", b"dtype", b"shape"} <= obj.keys():
            raise ValueError("an array needs data, dtype and shape")
        dtype = _numeric_dtype(obj[b"dtype"])
        shape, data = obj[b"shape"], obj[b"data"]
        if not isinstance(data, bytes) or not isinstance(shape, list) or not all(
            type(n) is int and n >= 0 for n in shape
        ):
            raise ValueError("an array needs bytes data and a shape of non-negative ints")
        if math.prod(shape) * dtype.itemsize != len(data):
            raise ValueError(f"an array of shape {tuple(shape)} and dtype {dtype.str} is not {len(data)} bytes")
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if b"__npgeneric__" in obj:
        if not {b"data", b"dtype"} <= obj.keys():
            raise ValueError("a scalar needs data and dtype")
        data = obj[b"data"]
        # what np.generic.item() packs: a missing or None value would be NaN
        if type(data) not in (bool, int, float):
            raise ValueError(f"a scalar's data must be a number, not {type(data).__name__}")
        return _numeric_dtype(obj[b"dtype"]).type(data)
    return obj


def pack(value) -> bytes:
    return msgpack.packb(value, default=_pack_array, use_bin_type=True)


# the two msgpack errors raised with no message
_BLANK_ERRORS = {msgpack.FormatError: "an unknown type byte", msgpack.StackError: "nested too deeply"}


def unpack(data: bytes):
    try:
        return msgpack.unpackb(data, object_hook=_unpack_array, raw=False)
    except (msgpack.UnpackException, ValueError, TypeError, OverflowError) as exc:
        detail = str(exc) or _BLANK_ERRORS.get(type(exc), type(exc).__name__)
        raise ValueError(f"the observation is not openpi's msgpack: {detail}") from None


# ----------------------------------------------------------------------
# Observation -> request, output -> reply
# ----------------------------------------------------------------------

class PolicyRequest(NamedTuple):
    prompt: str
    images: list[torch.Tensor]
    model_kwargs: dict


def _image(value, key: str) -> torch.Tensor:
    """An observation image as RGB uint8 ``[3, H, W]``. Read as openpi's
    ``_parse_image`` reads one: floats are scaled by 255, and a leading axis of
    3 is the channel axis (else it is ``[H, W, 3]``). Unlike openpi, a float
    image must be in [0, 1]: scaled and cast, NaN or 2.0 would become an
    ordinary pixel (2.0 wraps to 254)."""
    if not isinstance(value, np.ndarray) or value.ndim != 3:
        raise ValueError(f"{key} must be an image array of 3 dimensions")
    floating = np.issubdtype(value.dtype, np.floating)
    if not floating and value.dtype != np.uint8:
        raise ValueError(f"{key} must be uint8 or float, not {value.dtype}")
    chw = value if value.shape[0] == 3 else value.transpose(2, 0, 1)
    if chw.shape[0] != 3 or 0 in chw.shape:
        raise ValueError(f"{key} must be an RGB image, [H, W, 3] or [3, H, W]; got {list(value.shape)}")
    if floating:
        # NaN fails both comparisons
        if not (chw.min() >= 0 and chw.max() <= 1):
            raise ValueError(f"{key} is a float image, so its values must be in [0, 1] (no NaN or inf)")
        chw = (255 * chw).astype(np.uint8)
    # a copy: the decoded frame is read-only, and the model gets its own contiguous tensor
    return torch.from_numpy(np.array(chw, order="C"))


def _numbers(value, key: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"{key} must be numeric")
    return value.reshape(-1)


@dataclass(frozen=True)
class OpenpiPolicy:
    """A deployment's ``openpi:`` block, checked against its model."""

    image_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    state_kwarg: str
    # how many values the state keys hold together, when the model says (RobotIO.state_dim)
    state_dim: int | None
    action_horizon: int
    # the model's output width, and the leading dims of it a reply keeps
    action_dim: int
    action_dims: int
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, block, model) -> OpenpiPolicy:
        """Raises ``ValueError`` at startup for a block the model can't serve."""
        if not isinstance(block, dict):
            raise ValueError("openpi: must be a mapping")
        unknown = block.keys() - _CONFIG_KEYS
        if unknown:
            raise ValueError(f"openpi: takes {sorted(_CONFIG_KEYS)}; got {sorted(unknown)}")
        io = model.robot_io() if model is not None else None
        name = type(model).__name__
        if io is None:
            raise ValueError(f"openpi: {name} is not a robot policy (it declares no Model.robot_io)")
        if type(model).load_decoded_image is Model.load_decoded_image:
            raise ValueError(f"openpi: {name} takes images only as files (no Model.load_decoded_image)")
        images, state = block.get("images"), block.get("state")
        for name, keys in (("images", images), ("state", state)):
            if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
                raise ValueError(f"openpi: {name} must be a non-empty list of observation keys")
        cameras = model.max_input_files().get("image")
        if cameras is not None and len(images) > cameras:
            raise ValueError(f"openpi: images names {len(images)} cameras; the model takes at most {cameras}")
        action_dims = block.get("action_dims", io.action_dim)
        if type(action_dims) is not int or not 1 <= action_dims <= io.action_dim:
            raise ValueError(f"openpi: action_dims must be an int in [1, {io.action_dim}]")
        metadata = block.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("openpi: metadata must be a mapping")
        try:
            pack(metadata)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"openpi: metadata cannot be sent as msgpack: {exc}") from None
        return cls(
            image_keys=tuple(images), state_keys=tuple(state), state_kwarg=io.state_kwarg, state_dim=io.state_dim,
            action_horizon=io.action_horizon, action_dim=io.action_dim, action_dims=action_dims,
            metadata=metadata,
        )

    def request(self, obs) -> PolicyRequest:
        """The request an observation makes. Raises ``ValueError`` for one this
        deployment can't read. ``seed``, which openpi doesn't have, pins the
        call's noise, as ``/generate``'s ``model_kwargs`` seed does."""
        if not isinstance(obs, dict):
            raise ValueError("an observation is a map of names to values")
        missing = [k for k in (*self.image_keys, *self.state_keys) if k not in obs]
        if missing:
            raise ValueError(f"the observation has no {', '.join(missing)}")
        prompt = obs.get("prompt")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if not isinstance(prompt, str):
            raise ValueError("the observation needs a prompt string")
        images = [_image(obs[k], k) for k in self.image_keys]
        state = np.concatenate([_numbers(obs[k], k) for k in self.state_keys])
        if not state.size or self.state_dim not in (None, state.size):
            want = f"{self.state_dim} values" if self.state_dim is not None else "at least one value"
            raise ValueError(f"the state ({' + '.join(self.state_keys)}) has {state.size} values; "
                             f"this deployment's robot sends {want}")
        model_kwargs = {self.state_kwarg: state.astype(np.float64).tolist()}
        seed = obs.get("seed")
        if seed is not None:
            if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
                raise ValueError("seed must be an integer")
            model_kwargs["seed"] = int(seed)
        return PolicyRequest(prompt, images, model_kwargs)

    def actions(self, data: bytes) -> np.ndarray:
        """The reply's ``actions``: the model's float32 ``[H, action_dim]``
        output, cut to its first ``action_dims`` dims."""
        actions = np.frombuffer(data, dtype=np.float32)
        if actions.size != self.action_horizon * self.action_dim:
            raise RuntimeError(
                f"the model returned {actions.size} action values, not {self.action_horizon}x{self.action_dim}"
            )
        return np.ascontiguousarray(actions.reshape(self.action_horizon, self.action_dim)[:, :self.action_dims])


# ----------------------------------------------------------------------
# The connection
# ----------------------------------------------------------------------

class _CallFailedError(Exception):
    """The request failed after it was submitted; ``text`` is client-safe."""

    def __init__(self, text: str):
        super().__init__(text)
        self.text = text


async def _infer(api, policy: OpenpiPolicy, request: PolicyRequest) -> np.ndarray:
    request_id = api.submit_request(
        text=request.prompt,
        raw_inputs={"image": request.images},
        input_modalities=["image", "text"],
        output_modalities=["action"],
        model_kwargs=request.model_kwargs,
        streaming=True,
    )
    data = None
    # Leaving early (the handler cancelled) aborts the submission: the iterator owns that.
    async for chunk in api.iter_result_chunks(request_id):
        if chunk.modality == "error":
            raise _CallFailedError(bytes(chunk.data).decode("utf-8", "replace"))
        if chunk.modality == "action":
            data = bytes(chunk.data)
    if data is None:
        raise RuntimeError("the request finished without actions")
    return policy.actions(data)


def _error_text(exc: Exception) -> str:
    if isinstance(exc, _CallFailedError):
        return exc.text
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return client_error_text(exc, "openpi request failed", log=logger, type_name=False)[0]


async def serve_connection(websocket: WebSocket, api) -> None:
    """One robot's connection, served as openpi's ``WebsocketPolicyServer``
    serves one. A server whose config has no ``openpi:`` block refuses the
    handshake."""
    policy = getattr(api, "openpi_policy", None)
    if policy is None:
        # closing before accept refuses the handshake (HTTP 403)
        await websocket.close(code=1008, reason="this server has no openpi policy (no openpi: block in its config)")
        return
    await websocket.accept()
    try:
        await websocket.send_bytes(pack(policy.metadata))
        prev_total_ms = None
        while True:
            # as openpi's server times it: prev_total_ms spans the wait for the observation too
            start = time.monotonic()
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            try:
                if message.get("bytes") is None:
                    raise ValueError("an observation is a binary msgpack frame")
                obs = unpack(message["bytes"])
                infer_start = time.monotonic()
                actions = await _infer(api, policy, policy.request(obs))
                infer_ms = (time.monotonic() - infer_start) * 1000
            except Exception as exc:  # noqa: BLE001 — openpi's contract: the error as text, then close
                await websocket.send_text(_error_text(exc))
                await websocket.close(code=1011, reason="request failed; the error is in the previous frame")
                return
            timing = {"infer_ms": infer_ms}
            if prev_total_ms is not None:
                timing["prev_total_ms"] = prev_total_ms
            await websocket.send_bytes(pack({"actions": actions, "server_timing": timing}))
            prev_total_ms = (time.monotonic() - start) * 1000
    except WebSocketDisconnect:
        pass  # the robot went away; a send is the first to notice once its call has finished
