"""The openpi WebSocket at ``/`` (``mstar.api_server.openpi``), and the decoded
image path it feeds.

openpi's own client code is the oracle: ``_openpi_pack`` / ``_openpi_unpack``
are ``openpi_client.msgpack_numpy`` and ``_OpenpiClient`` is
``openpi_client.WebsocketClientPolicy`` (Physical-Intelligence/openpi,
packages/openpi-client, Apache-2.0), cut to what a call uses. The client runs
against a real uvicorn socket; the rest drive the route through Starlette's test
client against a fake API server.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import io
import socket
import threading
import time

import msgpack
import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from mstar.api_server import auth, entrypoint, openpi  # noqa: E402
from mstar.api_server.data_worker import PreprocessWorkerThread  # noqa: E402
from mstar.api_server.request_types import PreprocessInput, ResultChunk  # noqa: E402
from mstar.model.pi05.pi05_model import Pi05Model  # noqa: E402

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


class _OpenpiClient:
    def __init__(self, host: str, port: int, api_key: str | None = None):
        import websockets.sync.client

        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None, additional_headers=headers,
        )
        self.metadata = _openpi_unpack(self._ws.recv())

    def infer(self, obs: dict) -> dict:
        self._ws.send(_openpi_pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return _openpi_unpack(response)

    def close(self):
        self._ws.close()


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


class _FakeServer:
    """Stands in for ``APIServer``: records submissions, answers each with one
    action chunk of the model's ``[15, 32]`` float32 bytes, or fails it."""

    def __init__(self, policy):
        self.openpi_policy = policy
        self.submitted = []
        self.fail_with: str | None = None
        self.fatal_error = None

    def cleanup(self):  # the app's shutdown hook calls it when a real server stops
        pass

    def submit_request(self, **kwargs):
        rid = f"req-{len(self.submitted)}"
        self.submitted.append(kwargs)
        return rid

    async def iter_result_chunks(self, request_id):
        await asyncio.sleep(0)
        if self.fail_with is not None:
            yield ResultChunk(request_id, "error", self.fail_with.encode(), {"status": 400})
            return
        index = int(request_id.split("-")[1])
        yield ResultChunk(request_id, "action", _model_output(index).tobytes())


@pytest.fixture
def served(monkeypatch):
    fake = _FakeServer(_policy())
    monkeypatch.setattr(entrypoint, "api_server", fake)
    return fake


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


# ----------------------------------------------------------------------
# the connection
# ----------------------------------------------------------------------

def test_handshake_then_one_reply_per_observation(served):
    with TestClient(entrypoint.app).websocket_connect("/") as ws:
        assert _openpi_unpack(ws.receive_bytes()) == {}
        replies = []
        for i in range(2):
            ws.send_bytes(_openpi_pack(_droid_obs(seed=i)))
            replies.append(_openpi_unpack(ws.receive_bytes()))
    for i, reply in enumerate(replies):
        assert reply["actions"].shape == (HORIZON, 8) and reply["actions"].dtype == np.float32
        np.testing.assert_array_equal(reply["actions"], _model_output(i)[:, :8])
        assert reply["server_timing"]["infer_ms"] >= 0
    assert "prev_total_ms" not in replies[0]["server_timing"]
    assert replies[1]["server_timing"]["prev_total_ms"] >= 0
    sub = served.submitted[0]
    assert sub["text"] == "pick up the marker"
    assert sub["input_modalities"] == ["image", "text"] and sub["output_modalities"] == ["action"]
    assert len(sub["raw_inputs"]["image"]) == 2 and "file_paths" not in sub
    assert len(sub["model_kwargs"]["robot_state"]) == 8


def test_metadata_frame_is_the_configs(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", _FakeServer(_policy(metadata={"reset_pose": [0, -1.5]})))
    with TestClient(entrypoint.app).websocket_connect("/") as ws:
        assert _openpi_unpack(ws.receive_bytes()) == {"reset_pose": [0, -1.5]}


@pytest.mark.parametrize("send", ["failed request", "bad observation", "text frame"])
def test_a_failed_call_is_a_text_frame_then_a_close(served, send):
    with TestClient(entrypoint.app).websocket_connect("/") as ws:
        ws.receive_bytes()
        if send == "failed request":
            served.fail_with = "ValueError: the state has 8 values; the norm stats have 7"
            ws.send_bytes(_openpi_pack(_droid_obs()))
        elif send == "bad observation":
            ws.send_bytes(_openpi_pack({"prompt": "no cameras"}))
        else:
            ws.send_text("{}")
        text = ws.receive_text()
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_bytes()
    assert closed.value.code == 1011
    expected = {"failed request": "norm stats", "bad observation": "has no", "text frame": "binary msgpack"}[send]
    assert expected in text


def test_a_server_without_an_openpi_block_refuses_the_handshake(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", _FakeServer(None))
    with pytest.raises(WebSocketDisconnect) as refused:
        with TestClient(entrypoint.app).websocket_connect("/"):
            pass
    assert refused.value.code == 1008


def test_healthz_answers_as_openpi_does(served):
    r = TestClient(entrypoint.app).get("/healthz")
    assert r.status_code == 200 and r.text == "OK\n"


def test_healthz_is_not_served_without_an_openpi_block(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", _FakeServer(None))
    assert TestClient(entrypoint.app).get("/healthz").status_code == 404


@pytest.fixture
def keyed():
    entrypoint.api_key_guard.set("k3y")
    yield TestClient(entrypoint.app)
    entrypoint.api_key_guard.set(auth.resolve_api_key())


API_KEY = {"Authorization": "Api-Key k3y"}


def test_openpi_client_conventions_where_the_socket_is_served(served, keyed):
    assert keyed.get("/healthz").status_code == 200  # no key, as openpi's server probes
    with pytest.raises(WebSocketDisconnect):
        with keyed.websocket_connect("/"):
            pass
    for headers in (API_KEY, {"Authorization": "Bearer k3y"}):
        with keyed.websocket_connect("/", headers=headers) as ws:
            assert _openpi_unpack(ws.receive_bytes()) == {}
    with pytest.raises(WebSocketDisconnect):
        with keyed.websocket_connect("/", headers={"Authorization": "Api-Key wrong"}):
            pass


def test_the_api_key_scheme_stays_on_the_openpi_socket(served, keyed):
    """Every other route takes only Bearer, on a server with the socket too."""
    assert keyed.get("/v1/models", headers=API_KEY).status_code == 401
    assert keyed.post("/generate", headers=API_KEY).status_code == 401
    with pytest.raises(WebSocketDisconnect) as refused:
        with keyed.websocket_connect("/generate/ws", headers=API_KEY):
            pass
    assert refused.value.code == 1008


def test_without_an_openpi_block_neither_convention_applies(monkeypatch, keyed):
    monkeypatch.setattr(entrypoint, "api_server", _FakeServer(None))
    assert keyed.get("/healthz").status_code == 401
    with pytest.raises(WebSocketDisconnect) as refused:
        with keyed.websocket_connect("/", headers=API_KEY):
            pass
    assert refused.value.code == 1008 and refused.value.reason == "invalid API key"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _real_server(config: dict):
    """The app on a real uvicorn socket, built as the server builds it."""
    pytest.importorskip("websockets.sync.client")
    import uvicorn

    port = _free_port()
    uv_config = entrypoint._uvicorn_config(entrypoint.app, "127.0.0.1", port, config)
    uv_config.log_level = "warning"
    server = uvicorn.Server(uv_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.02)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_openpi_client_over_a_real_socket(served):
    with _real_server({}) as port:
        client = _OpenpiClient("127.0.0.1", port)
        assert client.metadata == {}
        for i in range(3):
            actions = client.infer(_droid_obs(seed=i))["actions"]
            np.testing.assert_array_equal(actions, _model_output(i)[:, :8])
        served.fail_with = "ValueError: refused"
        with pytest.raises(RuntimeError, match="refused"):
            client.infer(_droid_obs())
        client.close()


def test_an_observation_over_the_message_limit_closes_the_socket(served):
    """openpi's client sets no size limit; the server's ws_max_message_mb does."""
    from websockets.exceptions import ConnectionClosed

    big = _droid_obs()
    big["observation/exterior_image_1_left"] = np.zeros((1024, 1024, 3), np.uint8)  # 3 MiB
    with _real_server({}) as port:  # under the default 16 MiB it is served
        client = _OpenpiClient("127.0.0.1", port)
        assert client.infer(big)["actions"].shape == (HORIZON, 8)
        client.close()
    served.submitted.clear()
    with _real_server({"ws_max_message_mb": 1}) as port:
        client = _OpenpiClient("127.0.0.1", port)
        with pytest.raises(ConnectionClosed) as closed:
            client.infer(big)
        # The server closes with 1009 while the client may still be sending; the
        # reset that closing over unread data causes can drop the close frame.
        assert closed.value.rcvd is None or closed.value.rcvd.code == 1009
        assert not served.submitted
        client.close()


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
