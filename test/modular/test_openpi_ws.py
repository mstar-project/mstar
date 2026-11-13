"""The openpi policy (``mstar.api_server.openpi``) behind mstar-server's
WebSocket at ``/``, and the decoded image path it feeds; the socket itself is
tested in test/rust/test_rust_openpi.py.

openpi's own client code is the oracle: ``_openpi_pack`` / ``_openpi_unpack``
are ``openpi_client.msgpack_numpy`` (Physical-Intelligence/openpi,
packages/openpi-client, Apache-2.0).
"""

from __future__ import annotations

import functools
import io

import msgpack
import numpy as np
import pytest
import torch

from mstar.api_server import entrypoint, openpi
from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.model.pi05.pi05_model import Pi05Model

# ----------------------------------------------------------------------
# openpi_client, as the reference
# ----------------------------------------------------------------------


def _openpi_pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _openpi_unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_openpi_pack = functools.partial(msgpack.packb, default=_openpi_pack_array)
_openpi_unpack = functools.partial(msgpack.unpackb, object_hook=_openpi_unpack_array)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------

HORIZON, WIDTH = 15, 32
DROID = {"images": ["observation/exterior_image_1_left", "observation/wrist_image_left"],
         "state": ["observation/joint_position", "observation/gripper_position"],
         "action_dims": 8}


def _droid_model() -> Pi05Model:
    return Pi05Model(model_path_hf="lerobot/pi05_droid", skip_weight_loading=True,
                     num_cameras=2, action_horizon=HORIZON)


def _policy(**overrides) -> openpi.OpenpiPolicy:
    return openpi.OpenpiPolicy.from_config({**DROID, **overrides}, _droid_model())


def _droid_obs(seed: int = 0, **extra) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "observation/exterior_image_1_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": rng.random(7),
        "observation/gripper_position": rng.random(1),
        "prompt": "pick up the marker",
        **extra,
    }


def _model_output(i: int) -> np.ndarray:
    return (np.arange(HORIZON * WIDTH, dtype=np.float32).reshape(HORIZON, WIDTH) + 1000 * i)


# ----------------------------------------------------------------------
# wire format
# ----------------------------------------------------------------------

def test_codec_reads_and_writes_openpi_client_frames():
    obs = _droid_obs()
    obs["scalar"] = np.float32(0.5)
    got = openpi.unpack(_openpi_pack(obs))
    for key, value in obs.items():
        np.testing.assert_array_equal(got[key], value)
        assert type(got[key]) is type(value)
    reply = {"actions": _model_output(0)[:, :8].copy(), "server_timing": {"infer_ms": 1.5}}
    back = _openpi_unpack(openpi.pack(reply))
    assert back["actions"].dtype == np.float32 and back["actions"].shape == (HORIZON, 8)
    np.testing.assert_array_equal(back["actions"], reply["actions"])
    assert back["server_timing"] == {"infer_ms": 1.5}


@pytest.mark.parametrize("frame", [
    # an object array would run arbitrary code if it were unpickled; openpi refuses to send one
    msgpack.packb({"x": {b"__ndarray__": True, b"data": b"\0" * 8, b"dtype": "|O", b"shape": [1]}}),
    msgpack.packb({"x": {b"__ndarray__": True, b"data": b"\0" * 7, b"dtype": "<f4", b"shape": [2]}}),
    msgpack.packb({"x": {b"__ndarray__": True, b"data": b"\0" * 8, b"dtype": "<f4", b"shape": [-2, -1]}}),
    msgpack.packb({"x": {b"__ndarray__": True, b"data": b"", b"dtype": "nonsense", b"shape": [0]}}),
    # a missing key must not fall back to a default: np.dtype(None) is float64, a scalar of None NaN
    msgpack.packb({"x": {b"__ndarray__": True, b"data": b"\0" * 8, b"shape": [1]}}),
    msgpack.packb({"x": {b"__ndarray__": True, b"dtype": "<f8", b"shape": [1]}}),
    msgpack.packb({"x": {b"__ndarray__": True, b"data": b"\0" * 8, b"dtype": "<f8"}}),
    msgpack.packb({"x": {b"__npgeneric__": True, b"dtype": "<f8"}}),
    msgpack.packb({"x": {b"__npgeneric__": True, b"data": None, b"dtype": "<f8"}}),
    msgpack.packb({"x": {b"__npgeneric__": True, b"data": 1.0}}),
    msgpack.packb({"x": {b"__ndarray__": True, b"data": b"\0" * 8, b"dtype": 8, b"shape": [1]}}),
    b"\xc1",
    b"\x92\x01",
])
def test_codec_refuses_what_is_not_a_numeric_array(frame):
    with pytest.raises(ValueError):
        openpi.unpack(frame)


# ----------------------------------------------------------------------
# observation -> request
# ----------------------------------------------------------------------

def test_droid_observation_maps_as_droid_inputs_does():
    obs = _droid_obs()
    request = _policy().request(obs)
    assert request.prompt == "pick up the marker"
    (ext, wrist) = request.images
    for image, key in ((ext, "observation/exterior_image_1_left"), (wrist, "observation/wrist_image_left")):
        assert image.dtype == torch.uint8 and image.shape == (3, 224, 224) and image.is_contiguous()
        assert torch.equal(image, torch.from_numpy(obs[key]).permute(2, 0, 1))
    state = np.concatenate([obs["observation/joint_position"], obs["observation/gripper_position"]])
    assert request.model_kwargs == {"robot_state": state.tolist()}


def test_observation_variants_droid_inputs_accepts():
    obs = _droid_obs(seed=1)
    hwc = obs["observation/exterior_image_1_left"]
    obs["observation/exterior_image_1_left"] = hwc.transpose(2, 0, 1).copy()  # CHW
    obs["observation/wrist_image_left"] = obs["observation/wrist_image_left"].astype(np.float32) / 255
    obs["observation/gripper_position"] = np.float64(0.25)  # a scalar, not [1]
    obs["prompt"] = b"bytes prompt"
    request = _policy().request(obs)
    assert torch.equal(request.images[0], torch.from_numpy(hwc).permute(2, 0, 1))
    expected_wrist = (255 * obs["observation/wrist_image_left"]).astype(np.uint8).transpose(2, 0, 1)
    assert torch.equal(request.images[1], torch.from_numpy(expected_wrist))
    assert request.model_kwargs["robot_state"][-1] == 0.25 and len(request.model_kwargs["robot_state"]) == 8
    assert request.prompt == "bytes prompt"


def test_seed_pins_the_noise():
    assert _policy().request(_droid_obs(seed=np.int64(7)) | {"seed": np.int64(7)}).model_kwargs["seed"] == 7
    with pytest.raises(ValueError, match="seed"):
        _policy().request(_droid_obs() | {"seed": 1.5})


@pytest.mark.parametrize("change,match", [
    ({"observation/wrist_image_left": None}, "wrist_image_left"),
    ({"observation/wrist_image_left": np.zeros((224, 224), np.uint8)}, "3 dimensions"),
    ({"observation/wrist_image_left": np.zeros((224, 224, 4), np.uint8)}, "RGB"),
    ({"observation/wrist_image_left": np.zeros((224, 224, 3), np.int32)}, "uint8 or float"),
    ({"observation/joint_position": "seven"}, "numeric"),
    ({"prompt": 3}, "prompt"),
])
def test_bad_observations_are_refused(change, match):
    obs = _droid_obs()
    for key, value in change.items():
        if value is None:
            del obs[key]
        else:
            obs[key] = value
    with pytest.raises(ValueError, match=match):
        _policy().request(obs)


@pytest.mark.parametrize("block,match", [
    ({**DROID, "action_horizon": 10}, "takes"),
    ({**DROID, "images": DROID["images"] * 2}, "at most 2"),
    ({**DROID, "state": []}, "state"),
    ({**DROID, "action_dims": 33}, "action_dims"),
    ({**DROID, "metadata": {"pose": object()}}, "metadata"),
])
def test_config_block_is_checked_at_startup(block, match):
    with pytest.raises(ValueError, match=match):
        openpi.OpenpiPolicy.from_config(block, _droid_model())


def test_the_policy_shape_comes_from_the_model():
    """The API server names no model: it serves whatever declares Model.robot_io."""
    policy = _policy()
    io = _droid_model().robot_io()
    assert tuple(io) == ("robot_state", HORIZON, WIDTH)
    assert (policy.state_kwarg, policy.action_horizon, policy.action_dim) == tuple(io)


def test_only_a_robot_policy_that_takes_decoded_images_is_served():
    from mstar.model.base import Model

    class _NotAPolicy(Pi05Model):
        def robot_io(self):
            return None

    class _FilesOnly(Pi05Model):
        load_decoded_image = Model.load_decoded_image

    for cls, match in ((_NotAPolicy, "not a robot policy"), (_FilesOnly, "only as files")):
        model = cls(model_path_hf="lerobot/pi05_droid", skip_weight_loading=True, num_cameras=2)
        with pytest.raises(ValueError, match=match):
            openpi.OpenpiPolicy.from_config(DROID, model)


def test_reply_keeps_the_leading_action_dims():
    policy = _policy()
    np.testing.assert_array_equal(policy.actions(_model_output(2).tobytes()), _model_output(2)[:, :8])
    with pytest.raises(RuntimeError, match="action values"):
        policy.actions(b"\0" * 4 * 10)



API_KEY = {"Authorization": "Api-Key k3y"}


# ----------------------------------------------------------------------
# decoded images through the preprocess worker
# ----------------------------------------------------------------------

def _loader(model) -> PreprocessWorkerThread:
    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker.model, worker.device, worker._keyed_labels = model, "cpu", {}
    return worker


def test_a_decoded_image_loads_as_its_lossless_file_does(tmp_path):
    from PIL import Image

    model = _droid_model()
    images = [torch.from_numpy(np.ascontiguousarray(_droid_obs(seed=s)["observation/wrist_image_left"]))
              .permute(2, 0, 1).contiguous() for s in range(2)]
    paths = []
    for i, image in enumerate(images):
        buf = io.BytesIO()
        Image.fromarray(image.permute(1, 2, 0).numpy()).save(buf, format="PNG")
        paths.append(tmp_path / f"cam{i}.png")
        paths[-1].write_bytes(buf.getvalue())
    worker = _loader(model)
    from_files, _ = worker._load_media({"image": [str(p) for p in paths]})
    decoded, _ = worker._load_media(None, {"image": images})
    assert len(decoded["image_inputs"]) == 2
    for a, b in zip(from_files["image_inputs"], decoded["image_inputs"], strict=True):
        assert a.dtype == b.dtype == torch.uint8 and torch.equal(a, b)


def test_decoded_images_are_held_to_the_media_limits(monkeypatch):
    worker = _loader(_droid_model())
    image = torch.zeros(3, 224, 224, dtype=torch.uint8)
    with pytest.raises(ValueError, match="at most 2"):
        worker._load_media(None, {"image": [image] * 3})
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", str(224 * 224 + 1))
    with pytest.raises(ValueError, match="pixels"):
        worker._load_media(None, {"image": [image] * 2})


def test_a_model_without_the_hook_takes_images_only_as_files():
    from mstar.model.base import Model

    worker = _loader(_droid_model())
    worker.model.load_decoded_image = functools.partial(Model.load_decoded_image, worker.model)
    with pytest.raises(NotImplementedError, match="only as files"):
        worker._load_media(None, {"image": [torch.zeros(3, 8, 8, dtype=torch.uint8)]})


def test_a_prefix_keyed_deployment_refuses_decoded_images():
    worker = _loader(_droid_model())
    worker._keyed_labels = {"llm_kv": {"main": object()}}
    request = PreprocessInput(
        request_id="r#1", text="t", file_paths=None, input_modalities=["image", "text"],
        output_modalities=["action"], model_kwargs={},
        raw_inputs={"image": [torch.zeros(3, 8, 8, dtype=torch.uint8)]},
    )
    with pytest.raises(ValueError, match="send the media as files"):
        worker._prepare_input(request)


@pytest.mark.parametrize("raw,file_paths,match", [
    ({"image": [torch.zeros(3, 8, 8, dtype=torch.uint8)]}, {"image": ["a.png"]}, "not both"),
    ({"image": [torch.zeros(3, 8, 8, dtype=torch.uint8)]}, {"image": []}, "not both"),
    ({"image": [torch.zeros(3, 8, 8)]}, None, "uint8"),
    ({"image": [torch.zeros(8, 8, 3, dtype=torch.uint8)]}, None, r"\[3, H, W\]"),
    # a modality with no decoded loader is refused at submit, not in the data worker
    ({"audio": [torch.zeros(16000)]}, None, "only as files"),
])
def test_submit_refuses_malformed_raw_inputs(raw, file_paths, match):
    with pytest.raises(ValueError, match=match):
        entrypoint._check_raw_inputs(raw, file_paths)
