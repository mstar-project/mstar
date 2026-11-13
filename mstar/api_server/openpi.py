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

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import msgpack
import numpy as np
import torch

from mstar.model.base import Model

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


def unpack(data: bytes):
    try:
        return msgpack.unpackb(data, object_hook=_unpack_array, raw=False)
    except (msgpack.UnpackException, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"the observation is not openpi's msgpack: {exc}") from None


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
    3 is the channel axis (else it is ``[H, W, 3]``)."""
    if not isinstance(value, np.ndarray) or value.ndim != 3:
        raise ValueError(f"{key} must be an image array of 3 dimensions")
    if np.issubdtype(value.dtype, np.floating):
        value = (255 * value).astype(np.uint8)
    elif value.dtype != np.uint8:
        raise ValueError(f"{key} must be uint8 or float, not {value.dtype}")
    chw = value if value.shape[0] == 3 else value.transpose(2, 0, 1)
    if chw.shape[0] != 3 or 0 in chw.shape:
        raise ValueError(f"{key} must be an RGB image, [H, W, 3] or [3, H, W]; got {list(value.shape)}")
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
            image_keys=tuple(images), state_keys=tuple(state), state_kwarg=io.state_kwarg,
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
