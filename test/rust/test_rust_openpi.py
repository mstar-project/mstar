"""openpi's policy socket at ``/`` on mstar-server, held to the Python route: the
Python route's answers replayed as it gave them before it was removed
(``python_oracle``).

Each case sends frames (openpi_client's ``msgpack_numpy`` encoding) and compares
the metadata frame, every reply, what reached ``submit_request`` (the images as
the model gets them, the state, the seed), and, for a refused call, the text
frame and the close."""

import asyncio
import contextlib
import functools
import json
import os
import subprocess
import tempfile
import threading

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("websockets")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
msgpack = pytest.importorskip("msgpack")

import websockets.exceptions
import websockets.sync.client
from python_oracle import python_side
from test_rust_frontend import BINARY, _free_port, _wait_healthy

from mstar.api_server import openpi
from mstar.api_server.request_types import ResultChunk
from mstar.api_server.rust_frontend import RustFrontendBridge, openpi_facts

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")


# openpi_client.msgpack_numpy, the clients' encoding
def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    return obj


_pack = functools.partial(msgpack.packb, default=_pack_array)
_unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)

HORIZON, WIDTH = 15, 32
DROID = {"images": ["observation/exterior_image_1_left", "observation/wrist_image_left"],
         "state": ["observation/joint_position", "observation/gripper_position"],
         "action_dims": 8, "metadata": {"robot": "droid", "rate_hz": 15, "dims": [1, 2.5]}}


@functools.cache
def _policy() -> openpi.OpenpiPolicy:
    """Pi0.5 DROID's block, as ``OpenpiPolicy.from_config`` checks it against the model."""
    return openpi.OpenpiPolicy(
        image_keys=tuple(DROID["images"]), state_keys=tuple(DROID["state"]), state_kwarg="state",
        action_horizon=HORIZON, action_dim=WIDTH, action_dims=DROID["action_dims"], metadata=DROID["metadata"],
    )


def _droid_obs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "observation/exterior_image_1_left": rng.integers(256, size=(24, 32, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(256, size=(24, 32, 3), dtype=np.uint8),
        "observation/joint_position": rng.random(7),
        "observation/gripper_position": rng.random(1),
        "prompt": "pick up the marker",
    }


def _actions(i: int) -> bytes:
    return (np.arange(HORIZON * WIDTH, dtype=np.float32) + 1000 * i).tobytes()


class _Server:
    """Both frontends' data plane: records each submission (images as the
    tensors the model gets) and answers it with one action chunk, or fails it."""

    def __init__(self):
        self.openpi_policy = _policy()
        self.upload_dir = tempfile.mkdtemp(prefix="mstar_openpi_up_")
        self.metrics = None
        self.fatal_error = None
        self.submitted: list[dict] = []
        self.fail_with: str | None = None
        self.answer: bytes | None = None

    def cleanup(self):
        pass

    def submit_request(self, **kw):
        rid = kw.get("request_id") or f"req-{len(self.submitted)}"
        self.submitted.append({
            "text": kw.get("text"), "in": kw.get("input_modalities"), "out": kw.get("output_modalities"),
            "mk": kw.get("model_kwargs"), "streaming": kw.get("streaming"),
            "images": [(tuple(t.shape), str(t.dtype), t.numpy().tobytes())
                       for t in (kw.get("raw_inputs") or {})["image"]],
        })
        return rid

    async def iter_result_chunks(self, request_id):
        await asyncio.sleep(0)
        if self.fail_with is not None:
            yield ResultChunk(request_id, "error", self.fail_with.encode(), {"status": 400})
            return
        yield ResultChunk(request_id, "action", self.answer or _actions(len(self.submitted)))

    def submission_key(self, request_id):
        return request_id

    def relay_chunks_to(self, key, sink, tag):
        pass

    def abort_submission(self, key):
        pass


@contextlib.contextmanager
def _rust(server: _Server, **extra_env):
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_openpi_")
    env = dict(os.environ, MSTAR_OPENPI=json.dumps(openpi_facts(server.openpi_policy)), **extra_env)
    proc = subprocess.Popen([BINARY, "pi05_droid", str(port), bridge_dir, server.upload_dir], env=env)
    bridge = RustFrontendBridge(server, bridge_dir)
    threading.Thread(target=bridge.run, daemon=True).start()
    try:
        assert _wait_healthy(port)
        yield port
    finally:
        bridge.stop()
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def stack():
    server = _Server()
    with _rust(server) as port:
        yield server, port


def _rust_calls(port, frames):
    out = []
    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/", compression=None, max_size=None) as ws:
        metadata = ws.recv()
        for frame in frames:
            ws.send(frame)
            reply = ws.recv()
            out.append(reply)
            if isinstance(reply, str):
                try:
                    ws.recv()
                except websockets.exceptions.ConnectionClosed as e:
                    out.append(("close", e.rcvd.code if e.rcvd else None))
                break
    return metadata, out


def _same_replies(py, rust):
    assert len(py) == len(rust)
    for p, r in zip(py, rust, strict=True):
        if isinstance(p, bytes):
            pu, ru = _unpack(p), _unpack(r)
            assert np.array_equal(pu["actions"], ru["actions"]) and pu["actions"].dtype == ru["actions"].dtype
            assert pu["actions"].shape == (HORIZON, DROID["action_dims"])
            assert sorted(pu["server_timing"]) == sorted(ru["server_timing"])
        else:
            assert p == r


def _both(stack, monkeypatch, frames, **server_setup):
    """Each frontend's (metadata, replies, submissions), the Python route's as recorded."""
    server, port = stack
    for k, v in server_setup.items():
        setattr(server, k, v)
    try:
        py = python_side()
        server.submitted.clear()
        rust_meta, rust = _rust_calls(port, frames)
        return py, (rust_meta, rust, list(server.submitted))
    finally:
        for k in server_setup:
            setattr(server, k, None)


def test_the_metadata_frame_and_each_call_read_as_pythons(stack, monkeypatch):
    frames = [_pack(_droid_obs(i)) for i in range(3)]
    (pm, py, ps), (rm, rust, rs) = _both(stack, monkeypatch, frames)
    assert rm == pm
    assert _unpack(rm) == DROID["metadata"]
    _same_replies(py, rust)
    assert "prev_total_ms" in _unpack(rust[1])["server_timing"]
    assert rs == ps and len(ps) == 3


def _variant(**changes):
    obs = _droid_obs()
    for key, value in changes.items():
        if value is ...:
            del obs[key]
        else:
            obs[key] = value
    return obs


KEY_A, KEY_B = DROID["images"]
VARIANTS = {
    "chw-image": {KEY_A: np.random.default_rng(1).integers(256, size=(3, 24, 32), dtype=np.uint8)},
    "float32-image": {KEY_A: np.random.default_rng(2).random((24, 32, 3), dtype=np.float32)},
    "float64-image": {KEY_A: np.random.default_rng(3).random((24, 32, 3))},
    "float16-image": {KEY_A: np.random.default_rng(4).random((24, 32, 3)).astype(np.float16)},
    "out-of-range-floats": {KEY_A: np.linspace(-2, 3, 24 * 32 * 3).reshape(24, 32, 3)},
    "state-as-lists": {"observation/joint_position": [0.1, 2, 3.5, 4, 5, 6, 7], "observation/gripper_position": [True]},
    "state-int-array": {"observation/joint_position": np.arange(7, dtype=np.int16)},
    "state-big-endian": {"observation/joint_position": np.arange(7, dtype=">f4")},
    "state-scalar": {"observation/gripper_position": np.float32(0.25)},
    "prompt-bytes": {"prompt": "grasp the cup".encode()},
    "seed-int": {"seed": 42},
    "seed-numpy-int": {"seed": np.int32(7)},
}
REFUSALS = {
    "missing-images": {KEY_A: ..., KEY_B: ...},
    "missing-state": {"observation/gripper_position": ...},
    "two-dimensional": {KEY_B: np.zeros((24, 32), np.uint8)},
    "four-channels": {KEY_B: np.zeros((24, 32, 4), np.uint8)},
    "empty-image": {KEY_B: np.zeros((3, 0, 32), np.uint8)},
    "int-image": {KEY_B: np.zeros((24, 32, 3), np.int32)},
    "bool-image": {KEY_B: np.zeros((24, 32, 3), np.bool_)},
    "image-not-an-array": {KEY_B: [1, 2, 3]},
    "string-state": {"observation/joint_position": "seven"},
    "no-prompt": {"prompt": ...},
    "prompt-not-a-string": {"prompt": 3},
    "seed-bool": {"seed": True},
    "seed-float": {"seed": 1.5},
    "seed-numpy-bool": {"seed": np.bool_(True)},
}


@pytest.mark.parametrize("case", sorted(VARIANTS))
def test_an_observation_reaches_the_model_as_the_python_route_sends_it(stack, monkeypatch, case):
    (_, py, ps), (_, rust, rs) = _both(stack, monkeypatch, [_pack(_variant(**VARIANTS[case]))])
    _same_replies(py, rust)
    assert rs == ps and len(ps) == 1


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_refused_observation_is_the_same_text_then_a_close(stack, monkeypatch, case):
    (_, py, ps), (_, rust, rs) = _both(stack, monkeypatch, [_pack(_variant(**REFUSALS[case]))])
    assert isinstance(py[0], str), py
    assert rust == py
    assert py[-1] == ("close", 1011)
    assert ps == rs == []


def _raw_array(dtype, shape, data):
    return msgpack.packb({"observation": {b"__ndarray__": True, b"data": data, b"dtype": dtype, b"shape": shape}})


@pytest.mark.parametrize("frame", [
    _raw_array("<f4", [2], b"\0" * 4),
    _raw_array("<U3", [1], b"\0" * 12),
    _raw_array("nope", [1], b"\0"),
    _raw_array(5, [1], b"\0"),
    _raw_array("<f4", [-1], b""),
    msgpack.packb({"x": {b"__ndarray__": True, b"data": b""}}),
    msgpack.packb({"x": {b"__npgeneric__": True, b"data": "a", b"dtype": "<f4"}}),
    msgpack.packb({"x": {b"__npgeneric__": True, b"data": 300, b"dtype": "|u1"}}),
], ids=["size", "string-dtype", "unknown-dtype", "dtype-not-a-string", "negative-shape", "no-dtype",
        "scalar-not-a-number", "scalar-out-of-range"])
def test_an_array_the_codec_refuses_is_the_same_text(stack, monkeypatch, frame):
    (_, py, _), (_, rust, _) = _both(stack, monkeypatch, [frame])
    assert rust == py and isinstance(py[0], str)


def test_a_text_frame_and_an_engine_failure_close_alike(stack, monkeypatch):
    (_, py, _), (_, rust, _) = _both(stack, monkeypatch, ["not binary"])
    assert rust == py and py[0] == "an observation is a binary msgpack frame"
    (_, py, _), (_, rust, _) = _both(stack, monkeypatch, [_pack(_droid_obs())], fail_with="the model refused")
    assert rust == py and py[0] == "the model refused"


def test_a_short_action_chunk_is_an_internal_error_on_both(stack, monkeypatch):
    (_, py, _), (_, rust, _) = _both(stack, monkeypatch, [_pack(_droid_obs())], answer=b"\0" * 8)
    assert py[0].startswith("internal error (error id ") and rust[0].startswith("internal error (error id ")
    assert rust[-1] == py[-1] == ("close", 1011)


def test_healthz_answers_as_openpi_does(stack):
    import urllib.request

    _, port = stack
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=10) as r:
        assert (r.status, r.read()) == (200, b"OK\n")



# what uvicorn (websockets) closes a socket with when a frame passes ws_max_message_mb
TOO_BIG = (1009, "frame with 2000 bytes exceeds limit of 1000 bytes")


def test_a_client_that_closes_gets_the_close_handshake(stack):
    """A client's Close is answered with one, as uvicorn answers it; without the
    answer the client sees the connection drop (1006)."""
    _, port = stack
    ws = websockets.sync.client.connect(f"ws://127.0.0.1:{port}/", compression=None, max_size=None)
    ws.recv(timeout=30)  # the metadata frame
    ws.close()
    assert ws.close_code == 1000


def test_a_message_past_the_limit_closes_the_socket_as_uvicorn_does():
    with _rust(_Server(), MSTAR_WS_MAX_BYTES="1000") as port:
        url = f"ws://127.0.0.1:{port}/"
        with websockets.sync.client.connect(url, compression=None, max_size=None) as ws:
            ws.recv(timeout=10)   # the greeting frame
            ws.send(b"\x00" * 2000)
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                ws.recv(timeout=10)
    assert (closed.value.rcvd.code, closed.value.rcvd.reason) == TOO_BIG


def test_the_api_key_takes_openpi_clients_scheme_on_its_socket_only():
    """openpi_client sends ``Authorization: Api-Key <key>``; the openpi socket
    takes it (and Bearer), ``/healthz`` stays open, and every other route
    still wants Bearer."""
    import urllib.error
    import urllib.request

    key = "s3cret"
    with _rust(_Server(), MSTAR_API_KEY=key) as port:
        url = f"ws://127.0.0.1:{port}/"
        for headers in ({}, {"Authorization": "Api-Key wrong"}, {"Authorization": f"Basic {key}"}):
            with pytest.raises(websockets.exceptions.InvalidStatus) as refused:
                websockets.sync.client.connect(url, additional_headers=headers, open_timeout=10)
            assert refused.value.response.status_code == 403, headers
        for scheme in ("Api-Key", "Bearer", "api-key"):
            with websockets.sync.client.connect(url, additional_headers={"Authorization": f"{scheme} {key}"},
                                                compression=None, max_size=None, open_timeout=10) as ws:
                assert ws.recv(timeout=10)  # the metadata frame
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=10) as r:
            assert r.status == 200
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": f"Api-Key {key}"})
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(req, timeout=10)
        assert refused.value.code == 401
