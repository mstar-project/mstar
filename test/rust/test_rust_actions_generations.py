"""Side-by-side Python vs Rust parity tests for ``POST /v1/actions/generations``.

Sends identical raw ``application/msgpack`` HTTP requests to both the Python
FastAPI frontend (``entrypoint.app``) and the compiled ``mstar-server`` Rust
binary + ``RustFrontendBridge``, verifying byte/schema/status-code parity,
recorded ``submit_request`` parity, and ``weight_version`` propagation.
"""

import asyncio
import base64
import contextlib
import json
import os
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from test_rust_frontend import BINARY, _Chunk, _free_port, _wait_healthy

from mstar.api_server import auth, entrypoint
from mstar.api_server.rust_frontend import ACTIONS_ENV, OPENAI_ADAPTER_ENV, RustFrontendBridge, openai_adapter_class

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
TINY_PNG_URI = f"data:image/png;base64,{TINY_PNG_B64}"
TINY_PNG_BYTES = base64.b64decode(TINY_PNG_B64)


def _pack_bytes_keys(value) -> bytes:
    def _default(obj):
        if isinstance(obj, np.ndarray):
            return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
        if isinstance(obj, np.generic):
            return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
        return obj

    return msgpack.packb(value, default=_default, use_bin_type=True)


def _pack_str_keys(value) -> bytes:
    def _default(obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tobytes(), "dtype": obj.dtype.str, "shape": obj.shape}
        if isinstance(obj, np.generic):
            return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
        return obj

    return msgpack.packb(value, default=_default, use_bin_type=True)


def _unpack_strict_bytes(data: bytes):
    def _hook(obj):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]).copy()
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj

    return msgpack.unpackb(data, object_hook=_hook, raw=False)


class _FakePi05Model:
    def __init__(self, action_horizon: int = 10, action_dim: int = 32, num_cameras: int = 3, num_flow_steps: int = 3):
        self.config = SimpleNamespace(
            action_horizon=action_horizon,
            action_dim=action_dim,
            state_dim=action_dim,
            num_cameras=num_cameras,
            num_flow_steps=num_flow_steps,
        )

    def robot_io(self):
        return SimpleNamespace(
            state_kwarg="robot_state",
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
        )

    def max_input_files(self):
        return {"image": self.config.num_cameras}


class _FakeCosmos3Model:
    ACTION_GEN_WALK = "action_gen"

    def __init__(self, action_horizon: int = 16, action_dim: int = 64, num_steps: int = 30):
        self.config = SimpleNamespace(
            action_gen=True,
            is_distilled=False,
            policy_action_chunk_size=action_horizon,
            max_action_dim=action_dim,
            num_inference_steps_action=num_steps,
            policy_domain_name="libero",
        )


class _ActionsStub:
    def __init__(self, model, model_name: str, upload_dir: str):
        self.model = model
        self.model_name = model_name
        self.openpi_policy = None
        self.upload_dir = upload_dir
        self.metrics = None
        self.fatal_error = None
        self.paused = False
        self.submitted: list[dict] = []
        self.chunks_fn = None

    def cleanup(self):
        pass

    def submit_request(self, **kw):
        idx = len(self.submitted)
        rid = kw.get("request_id") or f"act-{idx}"
        file_bytes = {
            mod: [Path(p).read_bytes() for p in paths]
            for mod, paths in (kw.get("file_paths") or {}).items()
        }
        raw_images = [
            (tuple(t.shape), str(t.dtype), t.numpy().tobytes())
            for t in (kw.get("raw_inputs") or {}).get("image", [])
        ]
        self.submitted.append({
            "rid": rid,
            "text": kw.get("text"),
            "in": kw.get("input_modalities"),
            "out": kw.get("output_modalities"),
            "mk": kw.get("model_kwargs"),
            "streaming": kw.get("streaming"),
            "file_bytes": file_bytes,
            "raw_images": raw_images,
        })
        return rid

    async def iter_result_chunks(self, request_id):
        idx, sub = next((i, s) for i, s in enumerate(self.submitted) if s["rid"] == request_id)
        if self.chunks_fn is not None:
            async for chunk in self.chunks_fn(request_id, idx, sub):
                yield chunk
            return
        await asyncio.sleep(0)
        horizon, dim = (16, 64) if isinstance(self.model, _FakeCosmos3Model) else (10, 32)
        arr = np.full((horizon, dim), float(idx + 1), dtype=np.float32)
        yield _Chunk("action", arr.tobytes())

    def submission_key(self, request_id):
        return request_id

    def relay_chunks_to(self, key, sink, tag):
        pass

    def abort_submission(self, key):
        pass


def _comparable_submits(submits: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in s.items() if k != "rid"}
        for s in submits
    ]


@contextlib.contextmanager
def _actions_stack(model_name: str, stub: _ActionsStub, facts: dict | None, extra_env: dict | None = None):
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_act_br_")
    env = dict(os.environ)
    env[OPENAI_ADAPTER_ENV] = openai_adapter_class(model_name)
    if facts is not None:
        env[ACTIONS_ENV] = json.dumps(facts)
    else:
        env.pop(ACTIONS_ENV, None)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen([BINARY, model_name, str(port), bridge_dir, stub.upload_dir], env=env)
    bridge = RustFrontendBridge(stub, bridge_dir)
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    try:
        assert _wait_healthy(port)
        yield port
    finally:
        bridge.stop()
        proc.terminate()
        proc.wait(timeout=10)


def _http_post(port: int, raw_body: bytes, content_type: str | None, headers: dict | None = None):
    hdrs = dict(headers or {})
    if content_type is not None:
        hdrs["Content-Type"] = content_type
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/actions/generations", data=raw_body, headers=hdrs, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.headers.get("content-type", ""), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("content-type", ""), err.read()


def _both(
    monkeypatch,
    tmp_path,
    model_name: str,
    model,
    facts: dict | None,
    raw_body: bytes,
    *,
    content_type: str | None = "application/msgpack",
    headers: dict | None = None,
    chunks_fn=None,
    extra_env: dict | None = None,
):
    stub = _ActionsStub(model, model_name, str(tmp_path / "uploads"))
    os.makedirs(stub.upload_dir, exist_ok=True)
    stub.chunks_fn = chunks_fn

    monkeypatch.setattr(entrypoint, "api_server", stub)
    py_headers = dict(headers or {})
    if content_type is not None:
        py_headers["Content-Type"] = content_type
    py_resp = TestClient(entrypoint.app).post("/v1/actions/generations", content=raw_body, headers=py_headers)
    py_submits = _comparable_submits(list(stub.submitted))
    stub.submitted.clear()

    with _actions_stack(model_name, stub, facts, extra_env=extra_env) as port:
        r_status, r_ctype, r_bytes = _http_post(port, raw_body, content_type, headers=headers)
    rust_submits = _comparable_submits(list(stub.submitted))
    return (
        (py_resp.status_code, py_resp.headers.get("content-type", ""), py_resp.content, py_submits),
        (r_status, r_ctype, r_bytes, rust_submits),
        stub.upload_dir,
    )


def _assert_response_parity(py_bytes: bytes, rust_bytes: bytes, *, expect_rollout: bool):
    py_obj = _unpack_strict_bytes(py_bytes)
    rust_obj = _unpack_strict_bytes(rust_bytes)
    assert len(py_obj["data"]) == len(rust_obj["data"])
    for p_item, r_item in zip(py_obj["data"], rust_obj["data"], strict=True):
        assert p_item["input_index"] == r_item["input_index"]
        assert np.array_equal(p_item["action"]["values"], r_item["action"]["values"])
        assert p_item["action"]["values"].dtype == r_item["action"]["values"].dtype == np.float32
        if expect_rollout:
            p_ro, r_ro = p_item["rollout"], r_item["rollout"]
            assert p_ro["denoise_ind"] == r_ro["denoise_ind"]
            assert p_ro["seed"] == r_ro["seed"]
            assert p_ro["weight_version"] == r_ro["weight_version"]
            for k in ("chain", "logprobs", "logprobs_mask"):
                assert np.array_equal(p_ro[k], r_ro[k])
                assert p_ro[k].dtype == r_ro[k].dtype
        else:
            assert "rollout" not in p_item and "rollout" not in r_item


# ----------------------------------------------------------------------
# Cosmos3 & Pi0.5 parity tests
# ----------------------------------------------------------------------


def test_cosmos3_eval_and_rollout_parity(monkeypatch, tmp_path):
    facts = {
        "family": "cosmos3",
        "action_horizon": 16,
        "action_dim": 64,
        "num_cameras": 1,
        "num_flow_steps": 30,
    }
    eval_body = _pack_bytes_keys({
        "input": {
            "prompt": [f"pick cube {i}" for i in range(4)],
            "input_reference": [TINY_PNG_URI for _ in range(4)],
        },
        "parameters": {
            "raw_action_dim": 10,
            "num_frames": 17,
            "action_chunk_size": 16,
            "fps": 20,
            "action_fps": 20.0,
            "action_view_point": "ego_view",
            "num_inference_steps": 30,
            "seed": 42,
            "height": 192,
            "width": 320,
            "domain_name": "libero",
        },
        "runtime": {"response_format": "envelope", "output_format": "numpy"},
    })
    (ps, pc, pb, psub), (rs, rc, rb, rsub), _ = _both(
        monkeypatch, tmp_path, "cosmos3", _FakeCosmos3Model(), facts, eval_body,
    )
    assert ps == rs == 200
    assert pc.startswith("application/msgpack") and rc.startswith("application/msgpack")
    assert psub == rsub
    assert psub[0]["file_bytes"]["image"] == [TINY_PNG_BYTES]
    _assert_response_parity(pb, rb, expect_rollout=False)

    # Rollout with weight_version="v2" and seeds
    async def rollout_chunks(rid, idx, sub):
        await asyncio.sleep(0)
        yield _Chunk("action", np.full((8, 64), float(idx + 1), dtype=np.float32).tobytes(), {
            "shape": [8, 64], "weight_version": "v2",
        })
        tensor = _pack_bytes_keys({
            "chain": np.full((5, 8, 64), float(idx), dtype=np.float32),
            "denoise_ind": 2,
            "logprobs": np.full((8, 64), -0.125, dtype=np.float32),
            "logprobs_mask": np.ones((8, 64), dtype=np.bool_),
            "seed": sub["mk"]["seed"],
        })
        yield _Chunk("tensor", tensor, {"rollout_state": True, "weight_version": "v2"})

    rollout_body = _pack_str_keys({
        "input": {"prompt": ["a", "b"]},
        "parameters": {
            "seeds": [10, 20],
            "robot_states": [[0.25] * 10, [0.5] * 10],
            "rollout": {"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": 2},
        },
    })
    (ps, _, pb, psub), (rs, _, rb, rsub), _ = _both(
        monkeypatch, tmp_path, "cosmos3", _FakeCosmos3Model(), facts, rollout_body, chunks_fn=rollout_chunks,
    )
    assert ps == rs == 200
    assert psub == rsub
    _assert_response_parity(pb, rb, expect_rollout=True)
    assert _unpack_strict_bytes(rb)["data"][0]["rollout"]["weight_version"] == "v2"


@pytest.mark.parametrize("use_str_keys", [False, True])
def test_pi05_eval_and_rollout_parity(monkeypatch, tmp_path, use_str_keys):
    facts = {
        "family": "pi05",
        "action_horizon": 10,
        "action_dim": 32,
        "state_kwarg": "robot_state",
        "state_dim": 32,
        "num_cameras": 3,
        "num_flow_steps": 3,
    }
    packer = _pack_str_keys if use_str_keys else _pack_bytes_keys
    rng = np.random.default_rng(13)

    async def rollout_chunks(rid, idx, sub):
        await asyncio.sleep(0)
        yield _Chunk("action", np.full((10, 32), float(idx + 3), dtype=np.float32).tobytes())
        tensor = _pack_bytes_keys({
            "chain": np.ones((4, 10, 32), dtype=np.float32) * (idx + 1),
            "denoise_ind": 1,
            "logprobs": np.full((10, 32), -0.5, dtype=np.float32),
            "logprobs_mask": np.ones((10, 32), dtype=np.bool_),
            "seed": sub["mk"]["seed"],
        })
        yield _Chunk("tensor", tensor, {"rollout_state": True})

    body = packer({
        "input": {
            "tokenized_prompt": np.array([[10, 20, 30], [40, 50, 0]], dtype=np.int64),
            "tokenized_prompt_mask": np.array([[True, True, True], [True, True, False]]),
            "images": {
                "base_0_rgb": rng.integers(0, 255, size=(2, 6, 8, 3), dtype=np.uint8),
                "left_wrist_0_rgb": rng.random((2, 6, 8, 3), dtype=np.float32),
                "right_wrist_0_rgb": np.zeros((2, 3, 6, 8), dtype=np.uint8),
            },
            "camera_mask": {
                "right_wrist_0_rgb": [False, True],
                "base_0_rgb": [True, True],
                "left_wrist_0_rgb": [True, False],
            },
            "robot_state": np.full((2, 8), 0.125, dtype=np.float32),
        },
        "parameters": {
            "seeds": [5, 6],
            "rollout": {"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": 1, "return_state": True},
        },
    })
    (ps, _, pb, psub), (rs, _, rb, rsub), _ = _both(
        monkeypatch, tmp_path, "pi05_libero", _FakePi05Model(), facts, body, chunks_fn=rollout_chunks,
    )
    assert ps == rs == 200
    assert psub == rsub
    _assert_response_parity(pb, rb, expect_rollout=True)
    assert _unpack_strict_bytes(rb)["data"][0]["rollout"]["weight_version"] is None


# ----------------------------------------------------------------------
# Error status parity (400, 401, 404, 415, 500, 503)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_body",
    [
        b"",
        b"\xc1",
        msgpack.packb([1, 2]),
        msgpack.packb({"parameters": {}}),
        msgpack.packb({"input": {"prompt": []}}),
        msgpack.packb({"input": {"prompt": [f"p{i}" for i in range(9)]}}),
        msgpack.packb({"input": {"prompt": ["a", "b"], "input_reference": [TINY_PNG_URI]}}),
        msgpack.packb({"input": {"prompt": ["a"], "input_reference": ["/etc/passwd"]}}),
        msgpack.packb({
            "input": {"prompt": "a"},
            "parameters": {"rollout": {"sde_type": "flow_sde", "noise_level": float("nan")}},
        }),
        msgpack.packb({"x": {b"__ndarray__": True, b"data": b"\0", b"dtype": "|O", b"shape": [1]}}),
        msgpack.packb({"__ndarray__": True, "dtype": "<f4", "shape": [2]}, use_bin_type=True),
        msgpack.packb({"__ndarray__": True, "data": b"\x00" * 4, "dtype": "<f4", "shape": [2]}, use_bin_type=True),
        msgpack.packb({"__ndarray__": True, "data": b"", "dtype": "<f4", "shape": [-1]}, use_bin_type=True),
        msgpack.packb({"__ndarray__": True, "data": b"\x00" * 8, "dtype": "|O", "shape": [1]}, use_bin_type=True),
    ],
)
def test_400_validation_parity_and_no_upload_leaks(monkeypatch, tmp_path, bad_body):
    facts = {"family": "cosmos3", "action_horizon": 16, "action_dim": 64, "num_cameras": 1, "num_flow_steps": 30}
    (ps, _, _, psub), (rs, _, _, rsub), updir = _both(
        monkeypatch, tmp_path, "cosmos3", _FakeCosmos3Model(), facts, bad_body,
    )
    assert ps == rs == 400
    assert psub == rsub == []
    assert not list(Path(updir).iterdir())


def test_415_404_401_503_500_parity(monkeypatch, tmp_path):
    facts = {"family": "pi05", "action_horizon": 10, "action_dim": 32, "num_cameras": 3, "num_flow_steps": 3}
    valid = _pack_bytes_keys({"input": {"prompt": "pick", "images": np.zeros((1, 1, 3, 4, 4), dtype=np.uint8)}})

    # 415 Content-Type
    (ps, _, _, _), (rs, _, _, _), _ = _both(
        monkeypatch, tmp_path, "pi05_libero", _FakePi05Model(), facts, valid, content_type="application/json",
    )
    assert ps == rs == 415

    # 404 Non-action model
    (ps, _, _, _), (rs, _, _, _), _ = _both(
        monkeypatch, tmp_path, "qwen3_omni", SimpleNamespace(), None, valid,
    )
    assert ps == rs == 404

    # 401 Auth
    entrypoint.api_key_guard.set("s3cret")
    try:
        (ps, _, _, _), (rs, _, _, _), _ = _both(
            monkeypatch, tmp_path, "pi05_libero", _FakePi05Model(), facts, valid,
            extra_env={"MSTAR_API_KEY": "s3cret"},
        )
        assert ps == rs == 401
        (ps2, _, _, _), (rs2, _, _, _), _ = _both(
            monkeypatch, tmp_path, "pi05_libero", _FakePi05Model(), facts, valid,
            headers={"Authorization": "Bearer s3cret"},
            extra_env={"MSTAR_API_KEY": "s3cret"},
        )
        assert ps2 == rs2 == 200
    finally:
        entrypoint.api_key_guard.set(auth.resolve_api_key())

    # 503 Paused error chunk
    async def err_503(rid, idx, sub):
        await asyncio.sleep(0)
        yield _Chunk("error", b"Server is paused", {"status": 503})

    (ps, _, _, _), (rs, _, _, _), _ = _both(
        monkeypatch, tmp_path, "pi05_libero", _FakePi05Model(), facts, valid, chunks_fn=err_503,
    )
    assert ps == rs == 503

    # 500 Truncated action bytes
    async def bad_action(rid, idx, sub):
        await asyncio.sleep(0)
        yield _Chunk("action", b"\0" * 12)

    (ps, _, _, _), (rs, _, _, _), _ = _both(
        monkeypatch, tmp_path, "pi05_libero", _FakePi05Model(), facts, valid, chunks_fn=bad_action,
    )
    assert ps == rs == 500

    # 500 Missing rollout tensor chunk
    rollout_req = _pack_bytes_keys({
        "input": {"prompt": "pick", "images": np.zeros((1, 1, 3, 4, 4), dtype=np.uint8)},
        "parameters": {"rollout": {"sde_type": "flow_sde", "noise_level": 0.5}},
    })
    (ps, _, _, _), (rs, _, _, _), _ = _both(
        monkeypatch, tmp_path, "pi05_libero", _FakePi05Model(), facts, rollout_req,
    )
    assert ps == rs == 500


def test_f9_pi05_4d_images_and_f10_cosmos3_8env_seeds_parity(monkeypatch, tmp_path):
    pi05_facts = {"family": "pi05", "action_horizon": 10, "action_dim": 32, "num_cameras": 3, "num_flow_steps": 3}

    async def pi05_chunks(rid, idx, sub):
        await asyncio.sleep(0)
        yield _Chunk("action", np.full((10, 32), 1.5, dtype=np.float32).tobytes(), {"shape": [10, 32]})
        tensor = _pack_bytes_keys({
            "chain": np.zeros((4, 10, 32), dtype=np.float32),
            "denoise_ind": 1,
            "logprobs": np.zeros((10, 32), dtype=np.float32),
            "logprobs_mask": np.ones((10, 32), dtype=np.bool_),
            "seed": sub["mk"]["seed"],
        })
        yield _Chunk("tensor", tensor, {"rollout_state": True})

    f9_body = _pack_bytes_keys({
        "model": "pi05",
        "input": {
            "tokenized_prompt": np.array([1, 4, 5, 6], dtype=np.int64),
            "images": np.zeros((3, 8, 8, 3), dtype=np.uint8),
            "camera_mask": np.array([True, True, False], dtype=np.bool_),
        },
        "parameters": {
            "seed": 42,
            "rollout": {"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": 1, "return_state": True},
        },
    })
    (ps, _, pb, psub), (rs, _, rb, rsub), _ = _both(
        monkeypatch, tmp_path / "f9", "pi05", _FakePi05Model(), pi05_facts, f9_body, chunks_fn=pi05_chunks,
    )
    assert ps == rs == 200 and psub == rsub
    _assert_response_parity(pb, rb, expect_rollout=True)

    c3_facts = {"family": "cosmos3", "action_horizon": 16, "action_dim": 64, "num_cameras": 1, "num_flow_steps": 30}

    async def c3_chunks(rid, idx, sub):
        await asyncio.sleep(0)
        yield _Chunk("action", np.full((16, 64), float(idx), dtype=np.float32).tobytes(), {"shape": [16, 64]})
        tensor = _pack_bytes_keys({
            "chain": np.full((3, 16, 64), float(idx), dtype=np.float32),
            "denoise_ind": 0,
            "logprobs": np.zeros((16, 64), dtype=np.float32),
            "logprobs_mask": np.ones((16, 64), dtype=np.bool_),
            "seed": sub["mk"]["seed"],
        })
        yield _Chunk("tensor", tensor, {"rollout_state": True})

    f10_body = _pack_bytes_keys({
        "model": "cosmos3",
        "input": {"prompt": ["pick block"] * 8, "input_reference": [TINY_PNG_URI] * 8},
        "parameters": {
            "action_mode": "policy",
            "raw_action_dim": 7,
            "num_frames": 5,
            "fps": 15.0,
            "action_fps": 15.0,
            "num_inference_steps": 2,
            "seeds": [100 + i for i in range(8)],
            "height": 64,
            "width": 64,
            "action_view_point": "base_and_wrist",
            "domain_name": "libero",
            "rollout": {"sde_type": "flow_sde", "noise_level": 0.4, "sde_step": "random", "return_state": True},
        },
        "runtime": {"response_format": "envelope", "output_format": "numpy"},
    })
    (ps, _, pb, psub), (rs, _, rb, rsub), _ = _both(
        monkeypatch, tmp_path / "f10", "cosmos3", _FakeCosmos3Model(), c3_facts, f10_body, chunks_fn=c3_chunks,
    )
    assert ps == rs == 200 and len(psub) == 8 and psub == rsub
    assert [s["mk"]["seed"] for s in psub] == [100 + i for i in range(8)]
    _assert_response_parity(pb, rb, expect_rollout=True)
