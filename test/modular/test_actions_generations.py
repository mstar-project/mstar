"""Unit and route tests for ``POST /v1/actions/generations`` (Python frontend).

Exercises the shared ``msgpack_numpy`` codec (bytes + str marker keys),
``rust_frontend.actions_facts``, FastAPI middleware integration (auth, body
limit, refusal metrics), RLinf-style Cosmos3 and Pi0.5 payloads, ``weight_version``
composition, and the full HTTP error matrix without GPU weights.
"""

from __future__ import annotations

import asyncio
import functools
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mstar.api_server import auth, entrypoint, media_io, openpi, rust_frontend  # noqa: E402
from mstar.api_server.request_types import ResultChunk  # noqa: E402
from mstar.model.pi05.pi05_model import Pi05Model  # noqa: E402

TINY_PNG_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


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


class _Cosmos3StubModel:
    ACTION_GEN_WALK = "action_gen"

    def __init__(self, *, action_gen: bool = True, is_distilled: bool = False):
        self.config = SimpleNamespace(
            action_gen=action_gen,
            is_distilled=is_distilled,
            policy_action_chunk_size=16,
            max_action_dim=64,
            num_inference_steps_action=30,
            policy_domain_name="libero",
        )


class _FakeActionsServer:
    def __init__(self, model, tmp_path, *, model_name: str = "pi05_libero", openpi_policy=None):
        self.model = model
        self.model_name = model_name
        self.openpi_policy = openpi_policy
        self.upload_dir = tmp_path
        self.fatal_error = None
        self.paused = False
        self.metrics = None
        self.submitted: list[dict] = []
        self.released: list[str] = []
        self.pending_requests: dict = {}
        self.fail_on_submit_idx: int | None = None
        self.chunks_fn = None

    def cleanup(self):
        pass

    def submit_request(self, **kwargs):
        idx = len(self.submitted)
        if self.fail_on_submit_idx is not None and idx == self.fail_on_submit_idx:
            raise ValueError("simulated submit failure")
        rid = kwargs.get("request_id") or f"act-{idx}"
        self.submitted.append(kwargs)
        return rid

    def submission_key(self, request_id: str) -> str:
        return f"{request_id}#0"

    def release_submission(self, key: str) -> None:
        self.released.append(key)

    async def iter_result_chunks(self, request_id: str):
        idx = next(i for i, sub in enumerate(self.submitted) if sub.get("request_id") == request_id)
        if self.chunks_fn is not None:
            async for chunk in self.chunks_fn(request_id, idx, self.submitted[idx]):
                yield chunk
            return
        await asyncio.sleep(0)
        horizon, dim = (16, 64) if isinstance(self.model, _Cosmos3StubModel) else (10, 32)
        arr = np.full((horizon, dim), float(idx + 1), dtype=np.float32)
        yield ResultChunk(request_id, "action", arr.tobytes())


def _pi05_model() -> Pi05Model:
    return Pi05Model(
        model_path_hf="lerobot/pi05_droid",
        skip_weight_loading=True,
        num_cameras=3,
        action_horizon=10,
    )


def _post(
    client: TestClient,
    body: bytes,
    content_type: str | None = "application/msgpack",
    headers: dict | None = None,
):
    hdrs = dict(headers or {})
    if content_type is not None:
        hdrs["Content-Type"] = content_type
    return client.post("/v1/actions/generations", content=body, headers=hdrs)


# ----------------------------------------------------------------------
# Codec & facts unit tests
# ----------------------------------------------------------------------


def test_openpi_codec_accepts_bytes_and_str_keys_and_writes_bytes_keys():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    scalar = np.int32(9)
    for pack_fn in (_pack_bytes_keys, _pack_str_keys):
        decoded = openpi.unpack(pack_fn({"arr": arr, "scalar": scalar, "k": "v"}))
        np.testing.assert_array_equal(decoded["arr"], arr)
        assert decoded["arr"].dtype == np.float32
        assert decoded["scalar"] == scalar and type(decoded["scalar"]) is np.int32
        assert decoded["k"] == "v"

    encoded = openpi.pack({"action": arr})
    unpacked = _unpack_strict_bytes(encoded)
    np.testing.assert_array_equal(unpacked["action"], arr)


@pytest.mark.parametrize(
    "bad_obj",
    [
        np.array(["a", "b"], dtype="<U1"),
        np.array([None], dtype=object),
        np.array([1 + 2j], dtype=np.complex64),
        np.complex64(1 + 2j),
        np.void(b"ab"),
    ],
)
def test_openpi_pack_rejects_non_numeric_dtypes(bad_obj):
    with pytest.raises(ValueError, match="only bool, integer and float"):
        openpi.pack({"bad": bad_obj})


@pytest.mark.parametrize(
    "bad_frame",
    [
        msgpack.packb({"x": {"__ndarray__": True, "data": b"\0" * 4, "dtype": "<f4", "shape": [2]}}),
        msgpack.packb({"x": {"__ndarray__": True, "data": b"\0" * 12, "dtype": "<U3", "shape": [1]}}),
        msgpack.packb({"x": {"__ndarray__": True, "data": b"\0" * 8, "dtype": "|O", "shape": [1]}}),
        msgpack.packb({"x": {"__ndarray__": True, "data": b"\0" * 8, "dtype": "<f4", "shape": [-1]}}),
        msgpack.packb({"x": {"__npgeneric__": True, "data": "bad", "dtype": "<f4"}}),
        msgpack.packb({"x": {b"__ndarray__": True, b"data": b"\0" * 4}}) + b"\x01",
    ],
)
def test_openpi_unpack_preserves_error_prefix_for_str_and_bytes_keys(bad_frame):
    with pytest.raises(ValueError, match="the observation is not openpi's msgpack:"):
        openpi.unpack(bad_frame)


def test_actions_facts_covers_pi05_cosmos3_policy_fallback_and_non_action(tmp_path):
    pi05 = _pi05_model()
    assert rust_frontend.actions_facts(SimpleNamespace(model=pi05)) == {
        "family": "pi05",
        "action_horizon": 10,
        "action_dim": 32,
        "state_kwarg": "robot_state",
        "state_dim": 32,
        "num_cameras": 3,
        "num_flow_steps": 10,
    }
    cosmos = _Cosmos3StubModel(action_gen=True, is_distilled=False)
    assert rust_frontend.actions_facts(SimpleNamespace(model=cosmos)) == {
        "family": "cosmos3",
        "action_horizon": 16,
        "action_dim": 64,
        "num_cameras": 1,
        "num_flow_steps": 30,
        "policy_domain_name": "libero",
    }
    assert rust_frontend.actions_facts(SimpleNamespace(model=_Cosmos3StubModel(action_gen=False))) is None
    assert rust_frontend.actions_facts(SimpleNamespace(model=_Cosmos3StubModel(is_distilled=True))) is None
    assert rust_frontend.actions_facts(SimpleNamespace(model=SimpleNamespace())) is None
    assert rust_frontend.actions_facts(SimpleNamespace()) is None

    policy = SimpleNamespace(action_horizon=15, action_dim=32, state_kwarg="robot_state", image_keys=("a", "b"))
    assert rust_frontend.actions_facts(SimpleNamespace(model=None, openpi_policy=policy)) == {
        "family": "pi05",
        "action_horizon": 15,
        "action_dim": 32,
        "state_kwarg": "robot_state",
        "state_dim": 32,
        "num_cameras": 2,
        "num_flow_steps": 10,
    }


# ----------------------------------------------------------------------
# Cosmos3 eval & rollout tests
# ----------------------------------------------------------------------


def test_cosmos3_batch_8_eval_and_out_of_order_collection(monkeypatch, tmp_path):
    fake = _FakeActionsServer(_Cosmos3StubModel(), tmp_path, model_name="cosmos3")
    monkeypatch.setattr(entrypoint, "api_server", fake)
    scheduled = []
    monkeypatch.setattr(media_io, "schedule_upload_cleanup", lambda fp, ud: scheduled.append((fp, ud)))

    async def chunks_fn(rid, idx, sub):
        # Complete reverse order to verify input_index ordering
        await asyncio.sleep(0.005 * (8 - idx))
        arr = np.full((16, 64), float(idx + 10), dtype=np.float32)
        yield ResultChunk(rid, "action", arr.tobytes())

    fake.chunks_fn = chunks_fn
    body = _pack_bytes_keys({
        "input": {
            "prompt": [f"pick obj {i}" for i in range(8)],
            "input_reference": [TINY_PNG_URI for _ in range(8)],
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
            "guidance_scale": 3.0,
            "flow_shift": 5.0,
            "use_karras_sigmas": True,
        },
        "runtime": {"response_format": "envelope", "output_format": "numpy"},
    })
    resp = _post(TestClient(entrypoint.app), body, content_type="application/msgpack; charset=utf-8")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/msgpack"
    out = _unpack_strict_bytes(resp.content)
    assert len(out["data"]) == 8
    assert len(fake.submitted) == 8
    assert len(scheduled) == 1
    for i, item in enumerate(out["data"]):
        assert item["input_index"] == i
        assert "rollout" not in item
        vals = item["action"]["values"]
        assert vals.shape == (16, 64) and vals.dtype == np.float32
        np.testing.assert_array_equal(vals, np.full((16, 64), float(i + 10), dtype=np.float32))
        sub = fake.submitted[i]
        assert sub["text"] == f"pick obj {i}"
        assert sub["input_modalities"] == ["image", "text"]
        assert sub["output_modalities"] == ["action"]
        assert len(sub["file_paths"]["image"]) == 1
        assert sub["model_kwargs"]["action_mode"] == "policy"
        assert sub["model_kwargs"]["seed"] == 42
        assert "runtime" not in sub["model_kwargs"]
        assert "seeds" not in sub["model_kwargs"]


def test_cosmos3_text_only_seeds_and_rollout(monkeypatch, tmp_path):
    fake = _FakeActionsServer(_Cosmos3StubModel(), tmp_path, model_name="cosmos3")
    monkeypatch.setattr(entrypoint, "api_server", fake)

    async def chunks_fn(rid, idx, sub):
        await asyncio.sleep(0)
        act = np.full((8, 64), float(idx + 1), dtype=np.float32)
        yield ResultChunk(rid, "action", act.tobytes(), {"shape": [8, 64], "weight_version": "v1"})
        rollout_payload = _pack_bytes_keys({
            "chain": np.full((5, 8, 64), float(idx), dtype=np.float32),
            "denoise_ind": 2,
            "logprobs": np.full((8, 64), -0.25, dtype=np.float32),
            "logprobs_mask": np.ones((8, 64), dtype=np.bool_),
            "seed": sub["model_kwargs"]["seed"],
        })
        yield ResultChunk(rid, "tensor", rollout_payload, {"rollout_state": True, "weight_version": "v1"})

    fake.chunks_fn = chunks_fn
    body = _pack_str_keys({
        "input": {"prompt": ["pick", "place"]},
        "parameters": {
            "seeds": [100, 101],
            "robot_states": [[0.1] * 10, [0.2] * 10],
            "rollout": {"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": 2, "return_state": True},
        },
    })
    resp = _post(TestClient(entrypoint.app), body)
    assert resp.status_code == 200
    out = _unpack_strict_bytes(resp.content)
    for i, item in enumerate(out["data"]):
        assert item["input_index"] == i
        assert item["action"]["values"].shape == (8, 64)
        assert fake.submitted[i]["input_modalities"] == ["text"]
        assert fake.submitted[i]["file_paths"] is None
        assert fake.submitted[i]["model_kwargs"]["seed"] == 100 + i
        assert fake.submitted[i]["model_kwargs"]["robot_state"] == pytest.approx([0.1 * (i + 1)] * 10)
        ro = item["rollout"]
        assert ro["denoise_ind"] == 2
        assert ro["seed"] == 100 + i
        assert ro["weight_version"] == "v1"
        assert ro["chain"].shape == (5, 8, 64) and ro["chain"].dtype == np.float32
        assert ro["logprobs"].shape == (8, 64) and ro["logprobs"].dtype == np.float32
        assert ro["logprobs_mask"].shape == (8, 64) and ro["logprobs_mask"].dtype == np.bool_


# ----------------------------------------------------------------------
# Pi0.5 eval, rollout & weight_version composition tests
# ----------------------------------------------------------------------


def test_pi05_batched_tokenized_prompt_dict_images_and_rollout(monkeypatch, tmp_path):
    fake = _FakeActionsServer(_pi05_model(), tmp_path, model_name="pi05_libero")
    monkeypatch.setattr(entrypoint, "api_server", fake)

    async def chunks_fn(rid, idx, sub):
        await asyncio.sleep(0)
        act = np.full((10, 32), float(idx + 2), dtype=np.float32)
        yield ResultChunk(rid, "action", act.tobytes())
        rollout_payload = _pack_str_keys({
            "chain": np.zeros((4, 10, 32), dtype=np.float32),
            "denoise_ind": 1,
            "logprobs": np.ones((10, 32), dtype=np.float32),
            "logprobs_mask": np.ones((10, 32), dtype=np.bool_),
            "seed": sub["model_kwargs"]["seed"],
        })
        yield ResultChunk(rid, "tensor", rollout_payload, {"rollout_state": True})

    fake.chunks_fn = chunks_fn
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, size=(2, 8, 8, 3), dtype=np.uint8)
    wrist = rng.random((2, 8, 8, 3), dtype=np.float32)
    padded = np.zeros((2, 3, 8, 8), dtype=np.uint8)

    body = _pack_bytes_keys({
        "input": {
            "tokenized_prompt": np.array([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=np.int64),
            "tokenized_prompt_mask": np.array([[True, True, True, False], [True, True, False, False]]),
            "images": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": padded,
            },
            # Intentionally shuffled dict key order to verify alignment to images order
            "camera_mask": {
                "right_wrist_0_rgb": [False, True],
                "base_0_rgb": [True, True],
                "left_wrist_0_rgb": [True, False],
            },
            "state": np.full((2, 8), 0.25, dtype=np.float32),
        },
        "parameters": {
            "seeds": [11, 12],
            "rollout": {"sde_type": "flow_sde", "noise_level": 0.5, "sde_step": 1, "return_state": True},
        },
    })
    resp = _post(TestClient(entrypoint.app), body)
    assert resp.status_code == 200
    out = _unpack_strict_bytes(resp.content)
    assert len(out["data"]) == 2

    sub0, sub1 = fake.submitted
    assert sub0["text"] is None
    assert sub0["input_modalities"] == ["image", "text"]
    assert sub0["output_modalities"] == ["action"]
    assert len(sub0["raw_inputs"]["image"]) == 3
    assert all(t.dtype == torch.uint8 and t.shape == (3, 8, 8) for t in sub0["raw_inputs"]["image"])
    assert sub0["model_kwargs"]["camera_mask"] == [True, True, False]
    assert sub1["model_kwargs"]["camera_mask"] == [True, False, True]
    assert sub0["model_kwargs"]["tokenized_prompt"] == [1, 2, 3, 0]
    assert sub1["model_kwargs"]["tokenized_prompt_mask"] == [True, True, False, False]
    assert sub0["model_kwargs"]["robot_state"] == pytest.approx([0.25] * 8)
    assert "state" not in sub0["model_kwargs"]
    assert "seeds" not in sub0["model_kwargs"]
    assert out["data"][0]["rollout"]["weight_version"] is None
    assert out["data"][1]["rollout"]["seed"] == 12


def test_pi05_return_state_false_and_pending_request_weight_version_fallback(monkeypatch, tmp_path):
    fake = _FakeActionsServer(_pi05_model(), tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)

    # return_state=False omits rollout
    body_no_state = _pack_bytes_keys({
        "input": {"prompt": "pick", "images": np.zeros((1, 2, 3, 4, 4), dtype=np.uint8)},
        "parameters": {"rollout": {"sde_type": "flow_ode", "return_state": False}},
    })
    resp = _post(TestClient(entrypoint.app), body_no_state)
    assert resp.status_code == 200
    item = _unpack_strict_bytes(resp.content)["data"][0]
    assert "rollout" not in item

    # PendingRequest.weight_version and chunk_weight_version fallback
    async def chunks_with_pending(rid, idx, sub, attr_name, wv):
        key = fake.submission_key(rid)
        fake.pending_requests[key] = SimpleNamespace(**{attr_name: wv})
        await asyncio.sleep(0)
        yield ResultChunk(rid, "action", np.ones((10, 32), dtype=np.float32).tobytes())
        tensor = _pack_bytes_keys({
            "chain": np.zeros((4, 10, 32), dtype=np.float32),
            "denoise_ind": 0,
            "logprobs": np.zeros((10, 32), dtype=np.float32),
            "logprobs_mask": np.ones((10, 32), dtype=np.bool_),
            "seed": 7,
        })
        yield ResultChunk(rid, "tensor", tensor, {"rollout_state": True})

    for attr, expected in (("weight_version", "v2"), ("chunk_weight_version", "v3")):
        fake.submitted.clear()
        fake.pending_requests.clear()
        fake.chunks_fn = functools.partial(chunks_with_pending, attr_name=attr, wv=expected)
        body = _pack_bytes_keys({
            "input": {"prompt": "pick", "input_reference": TINY_PNG_URI},
            "parameters": {"seed": 7, "rollout": {"sde_type": "flow_sde", "noise_level": 0.5}},
        })
        resp = _post(TestClient(entrypoint.app), body)
        assert resp.status_code == 200
        assert _unpack_strict_bytes(resp.content)["data"][0]["rollout"]["weight_version"] == expected


# ----------------------------------------------------------------------
# Error status, auth, body limit, and cleanup tests
# ----------------------------------------------------------------------


def test_auth_401_and_body_limit_413_on_actions_route(monkeypatch, tmp_path):
    fake = _FakeActionsServer(_pi05_model(), tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    entrypoint.api_key_guard.set("k3y")
    try:
        client = TestClient(entrypoint.app)
        body = _pack_bytes_keys({"input": {"prompt": "pick", "images": np.zeros((1, 1, 3, 4, 4), dtype=np.uint8)}})
        assert _post(client, body).status_code == 401
        assert _post(client, body, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert _post(client, body, headers={"Authorization": "Api-Key k3y"}).status_code == 401
        assert _post(client, body, headers={"Authorization": "Bearer k3y"}).status_code == 200
        assert client.get("/health").status_code == 200
    finally:
        entrypoint.api_key_guard.set(auth.resolve_api_key())

    # Body limit 413
    monkeypatch.setenv("MSTAR_MAX_BODY_MB", "0")
    stack = TestClient(entrypoint.app.build_middleware_stack())
    resp = _post(stack, b"\x80" * 16)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == 413


def test_503_404_415_500_and_partial_submit_cleanup(monkeypatch, tmp_path):
    # 503 api_server None
    monkeypatch.setattr(entrypoint, "api_server", None)
    assert _post(TestClient(entrypoint.app), b"\x80").status_code == 503

    # 503 fatal_error & paused
    fake = _FakeActionsServer(_pi05_model(), tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    fake.fatal_error = "conductor crashed"
    assert _post(TestClient(entrypoint.app), b"\x80").status_code == 503
    fake.fatal_error = None
    fake.paused = True
    assert _post(TestClient(entrypoint.app), b"\x80").status_code == 503
    fake.paused = False

    # 404 non-action model
    fake_non_action = _FakeActionsServer(SimpleNamespace(), tmp_path, model_name="qwen3_omni")
    monkeypatch.setattr(entrypoint, "api_server", fake_non_action)
    r404 = _post(TestClient(entrypoint.app), b"\x80")
    assert r404.status_code == 404 and r404.json()["error"]["code"] == 404

    # 415 wrong content type
    monkeypatch.setattr(entrypoint, "api_server", fake)
    assert _post(TestClient(entrypoint.app), b"{}", content_type="application/json").status_code == 415
    assert _post(TestClient(entrypoint.app), b"{}", content_type=None).status_code == 415

    # 503 from ResultChunk("error", ..., {"status": 503})
    async def err_503(rid, idx, sub):
        await asyncio.sleep(0)
        yield ResultChunk(rid, "error", b"Server is paused", {"status": 503})

    fake.chunks_fn = err_503
    valid_body = _pack_bytes_keys({"input": {"prompt": "pick", "images": np.zeros((1, 1, 3, 4, 4), dtype=np.uint8)}})
    r503 = _post(TestClient(entrypoint.app), valid_body)
    assert r503.status_code == 503 and "paused" in r503.json()["error"]["message"]

    # 500 missing tensor rollout chunk
    async def missing_tensor(rid, idx, sub):
        await asyncio.sleep(0)
        yield ResultChunk(rid, "action", np.zeros((10, 32), dtype=np.float32).tobytes())

    fake.chunks_fn = missing_tensor
    rollout_body = _pack_bytes_keys({
        "input": {"prompt": "pick", "images": np.zeros((1, 1, 3, 4, 4), dtype=np.uint8)},
        "parameters": {"rollout": {"sde_type": "flow_sde", "noise_level": 0.5}},
    })
    assert _post(TestClient(entrypoint.app), rollout_body).status_code == 500

    # 500 wrong action byte length
    async def bad_bytes(rid, idx, sub):
        await asyncio.sleep(0)
        yield ResultChunk(rid, "action", b"\0" * 7)

    fake.chunks_fn = bad_bytes
    assert _post(TestClient(entrypoint.app), valid_body).status_code == 500

    # Partial submission failure releases earlier submitted rows and cleans up uploads
    fake_c3 = _FakeActionsServer(_Cosmos3StubModel(), tmp_path, model_name="cosmos3")
    fake_c3.fail_on_submit_idx = 1
    monkeypatch.setattr(entrypoint, "api_server", fake_c3)
    batch_body = _pack_bytes_keys({
        "input": {"prompt": ["a", "b"], "input_reference": [TINY_PNG_URI, TINY_PNG_URI]},
    })
    r_partial = _post(TestClient(entrypoint.app), batch_body)
    assert r_partial.status_code == 400
    assert len(fake_c3.released) == 1


@pytest.mark.parametrize(
    "raw_body",
    [
        b"",
        b"\xc1",
        msgpack.packb([1, 2, 3]),
        msgpack.packb({"parameters": {}}),
        msgpack.packb({"input": "not-a-map"}),
        msgpack.packb({"input": {"prompt": "pick"}, "runtime": {"response_format": "json"}}),
        msgpack.packb({"input": {"prompt": "pick"}, "runtime": {"output_format": "list"}}),
        msgpack.packb({"input": {"prompt": []}}),
        msgpack.packb({"input": {"prompt": [f"p{i}" for i in range(9)]}}),
        msgpack.packb({"input": {"prompt": ["a", "b"], "input_reference": [TINY_PNG_URI]}}),
        msgpack.packb({"input": {"prompt": ["a"], "input_reference": ["data:image/png;base64,%%%"]}}),
        msgpack.packb({"input": {"prompt": ["a"], "input_reference": ["/etc/passwd"]}}),
        msgpack.packb({"input": {"prompt": ["a"], "input_reference": ["data:audio/wav;base64,AAAA"]}}),
        msgpack.packb({"input": {"prompt": "a"}, "parameters": {"seed": True}}),
        msgpack.packb({"input": {"prompt": "a"}, "parameters": {"seed": 1, "seeds": [1]}}),
        msgpack.packb({"input": {"prompt": ["a", "b"]}, "parameters": {"seeds": [1]}}),
        msgpack.packb({"input": {"prompt": "a"}, "parameters": {"rollout": {"sde_type": "unknown"}}}),
        msgpack.packb({
            "input": {"prompt": "a"},
            "parameters": {"rollout": {"sde_type": "flow_sde", "noise_level": -0.1}},
        }),
        msgpack.packb({
            "input": {"prompt": "a"},
            "parameters": {"rollout": {"sde_type": "flow_sde", "noise_level": float("nan")}},
        }),
        msgpack.packb({"input": {"prompt": "a"}, "parameters": {"rollout": {"sde_type": "flow_sde", "sde_step": 99}}}),
        msgpack.packb({
            "input": {"prompt": "a"},
            "parameters": {"rollout": {"sde_type": "flow_sde", "return_state": "yes"}},
        }),
        msgpack.packb({"input": {"prompt": "a"}, "parameters": {"rollout": {"sde_type": "flow_sde", "extra": 1}}}),
    ],
)
def test_cosmos3_400_validation_and_zero_upload_leaks(monkeypatch, tmp_path, raw_body):
    fake = _FakeActionsServer(_Cosmos3StubModel(), tmp_path, model_name="cosmos3")
    monkeypatch.setattr(entrypoint, "api_server", fake)
    resp = _post(TestClient(entrypoint.app), raw_body)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == 400
    assert not list(tmp_path.iterdir())


def test_pi05_400_validation_cases(monkeypatch, tmp_path):
    fake = _FakeActionsServer(_pi05_model(), tmp_path, model_name="pi05_libero")
    monkeypatch.setattr(entrypoint, "api_server", fake)
    client = TestClient(entrypoint.app)
    img = np.zeros((1, 2, 3, 4, 4), dtype=np.uint8)

    bad_inputs = [
        {"input": {"images": img}},
        {"input": {"prompt": "pick"}},
        {"input": {"prompt": "pick", "images": img, "camera_mask": [False, False]}},
        {"input": {"prompt": "pick", "images": img, "camera_mask": [True]}},
        {"input": {"prompt": "pick", "images": img, "camera_mask": [True, 2]}},
        {"input": {"tokenized_prompt": [1, 2, 3], "tokenized_prompt_mask": [True, False, True], "images": img}},
        {"input": {"tokenized_prompt": [-1, 2], "images": img}},
        {"input": {"tokenized_prompt": np.array([1.5, 2.5], dtype=np.float32), "images": img}},
        {"input": {"tokenized_prompt": np.array([2**63], dtype=np.uint64), "images": img}},
        {"input": {"tokenized_prompt": [1, 2], "images": np.zeros((2, 2, 3, 4, 4), dtype=np.uint8)}},
        {"input": {"prompt": "pick", "images": np.zeros((1, 4, 3, 4, 4), dtype=np.uint8)}},
        {"input": {"prompt": "pick", "images": img, "robot_state": [float("nan")]}},
        {"input": {"prompt": "pick", "images": img}, "parameters": {"seeds": np.array([2**63], dtype=np.uint64)}},
        {"input": {"prompt": "pick", "images": img}, "parameters": {"big": np.array([2**63], dtype=np.uint64)}},
    ]
    for payload in bad_inputs:
        resp = _post(client, _pack_bytes_keys(payload))
        assert resp.status_code == 400, payload


def test_uint64_max_int64_and_cosmos3_explicit_none_robot_state(monkeypatch, tmp_path):
    fake_pi05 = _FakeActionsServer(_pi05_model(), tmp_path / "pi05", model_name="pi05_libero")
    monkeypatch.setattr(entrypoint, "api_server", fake_pi05)
    resp = _post(
        TestClient(entrypoint.app),
        _pack_bytes_keys({
            "input": {
                "tokenized_prompt": np.array([[2**63 - 1]], dtype=np.uint64),
                "images": np.zeros((1, 2, 3, 4, 4), dtype=np.uint8),
            },
            "parameters": {"seeds": np.array([2**63 - 1], dtype=np.uint64)},
        }),
    )
    assert resp.status_code == 200
    assert fake_pi05.submitted[0]["model_kwargs"]["tokenized_prompt"] == [2**63 - 1]
    assert fake_pi05.submitted[0]["model_kwargs"]["seed"] == 2**63 - 1

    fake_c3 = _FakeActionsServer(_Cosmos3StubModel(), tmp_path / "c3", model_name="cosmos3")
    monkeypatch.setattr(entrypoint, "api_server", fake_c3)
    for payload in (
        {"input": {"prompt": "pick", "robot_state": None}, "parameters": {"robot_state": [0.1, 0.2]}},
        {"input": {"prompt": "pick", "robot_state": [0.1, 0.2]}, "parameters": {"robot_state": None}},
    ):
        fake_c3.submitted.clear()
        r = _post(TestClient(entrypoint.app), _pack_bytes_keys(payload))
        assert r.status_code == 200
        assert fake_c3.submitted[0]["model_kwargs"]["robot_state"] == [0.1, 0.2]


def test_data_worker_attaches_action_shape_and_rollout_state_metadata():
    from mstar.api_server.data_worker import PreprocessWorkerThread, RequestOutputState, ResultQueue

    class _Model:
        def text_output(self):
            return None

        def postprocess(self, output, modality, request_kwargs=None):
            if modality == "action":
                # Simulate Cosmos3 stripping the prepended robot_state row from (1, 17, 64) -> (16, 64)
                x = output[0] if output.ndim == 3 else output
                if (request_kwargs or {}).get("robot_state") is not None:
                    x = x[1:]
                return x.to(torch.float32).numpy().tobytes()
            return output.to(torch.uint8).numpy().tobytes()

    worker = object.__new__(PreprocessWorkerThread)
    worker.model = _Model()
    worker.enable_nvtx = False
    worker._turn_tokens = False
    worker.out_queue = ResultQueue(lambda: None)
    worker.request_model_kwargs = {
        "r-rollout": {"robot_state": [0.1] * 7, "rollout": {"sde_type": "flow_sde"}},
        "r-plain": {},
    }
    worker.request_output_state = {
        "r-rollout": RequestOutputState(),
        "r-plain": RequestOutputState(),
    }
    worker.tensor_uuid_to_metadata_per_request = {
        "r-rollout": {1: {"weight_version": "v2"}, 2: {"weight_version": "v2"}},
        "r-plain": {3: {}},
    }

    worker._postprocess_inline("r-rollout", 0, "action", 1, torch.zeros((1, 17, 64), dtype=torch.bfloat16), None)
    worker._postprocess_inline("r-rollout", 1, "tensor", 2, torch.zeros((12,), dtype=torch.uint8), None)
    worker._postprocess_inline("r-plain", 0, "tensor", 3, torch.zeros((4,), dtype=torch.uint8), None)

    c0 = worker.out_queue.get_nowait()
    c1 = worker.out_queue.get_nowait()
    c2 = worker.out_queue.get_nowait()
    assert c0.modality == "action" and c0.metadata == {"weight_version": "v2", "shape": [16, 64]}
    assert c1.modality == "tensor" and c1.metadata == {"weight_version": "v2", "rollout_state": True}
    assert c2.modality == "tensor" and c2.metadata == {}


def test_live_asgi_mid_batch_iter_failure_cancels_and_releases_siblings(monkeypatch, tmp_path):
    import time

    import httpx

    fake = _FakeActionsServer(_Cosmos3StubModel(), tmp_path, model_name="cosmos3")
    monkeypatch.setattr(entrypoint, "api_server", fake)
    state = {"row1_completed": False, "row1_cancelled": False, "row1_finally": False}

    async def chunks_fn(rid, idx, sub):
        if idx == 0:
            await asyncio.sleep(0.01)
            yield ResultChunk(rid, "error", b"row0 failed", {"status": 503})
            return
        try:
            await asyncio.sleep(0.30)
            state["row1_completed"] = True
            yield ResultChunk(rid, "action", np.zeros((16, 64), dtype=np.float32).tobytes())
        except asyncio.CancelledError:
            state["row1_cancelled"] = True
            raise
        finally:
            state["row1_finally"] = True

    fake.chunks_fn = chunks_fn
    body = _pack_bytes_keys({"input": {"prompt": ["a", "b"]}})

    async def _run():
        t0 = time.monotonic()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=entrypoint.app), base_url="http://testserver",
        ) as client:
            resp = await client.post(
                "/v1/actions/generations", content=body, headers={"Content-Type": "application/msgpack"},
            )
        elapsed = time.monotonic() - t0
        await asyncio.sleep(0.35)
        return resp, elapsed

    resp, elapsed = asyncio.run(_run())
    assert resp.status_code == 503
    assert elapsed < 0.20
    assert state["row1_cancelled"] is True
    assert state["row1_finally"] is True
    assert state["row1_completed"] is False
    assert len(fake.released) == 2


def test_live_asgi_mid_batch_failure_with_completed_sibling_on_real_apiserver(monkeypatch, tmp_path):
    import threading

    import httpx

    from mstar.api_server.entrypoint import APIServer, PendingRequest

    api = object.__new__(APIServer)
    api.model = _Cosmos3StubModel()
    api.model_name = "cosmos3"
    api.openpi_policy = None
    api.upload_dir = tmp_path
    api.fatal_error = None
    api.paused = False
    api.metrics = None
    api.enable_nvtx = False
    api.timeout_seconds = 5.0
    api.request_lock = threading.Lock()
    api.pending_requests = {}
    api.recently_completed = {}
    api._live = {}
    aborted: list[str] = []
    api.preprocess_worker = SimpleNamespace(abort_request=aborted.append)
    api._retire = lambda key: None
    api._record_end = lambda req, outcome: None
    api._finalize_profile = lambda req: None

    submitted_keys: list[str] = []

    def submit_request(**kwargs):
        rid = kwargs["request_id"]
        key = f"{rid}#0"
        submitted_keys.append(key)
        req = PendingRequest(
            streaming=False,
            input_modalities=kwargs["input_modalities"],
            output_modalities=kwargs["output_modalities"],
            profile=None,
            request_id=rid,
        )
        with api.request_lock:
            api._live[rid] = key
            api.pending_requests[key] = req
        return rid

    api.submit_request = submit_request
    monkeypatch.setattr(entrypoint, "api_server", api)
    body = _pack_bytes_keys({"input": {"prompt": ["a", "b"]}})

    async def _run():
        async def _drive():
            while len(api.pending_requests) < 2:
                await asyncio.sleep(0.002)
            with api.request_lock:
                keys = list(api.pending_requests.keys())
                req0 = api.pending_requests[keys[0]]
                req1 = api.pending_requests[keys[1]]
            # Row 1 finishes cleanly BEFORE row 0 errors
            req1.chunks.append(ResultChunk(req1.request_id, "action", np.zeros((16, 64), dtype=np.float32).tobytes()))
            req1.event.set()
            req1.notify()
            await asyncio.sleep(0.01)
            req0.chunks.append(ResultChunk(req0.request_id, "error", b"boom", {"status": 503}))
            req0.event.set()
            req0.notify()

        driver = asyncio.create_task(_drive())
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=entrypoint.app), base_url="http://testserver",
            ) as client:
                return await client.post(
                    "/v1/actions/generations", content=body, headers={"Content-Type": "application/msgpack"},
                )
        finally:
            await driver

    resp = asyncio.run(_run())
    assert resp.status_code == 503
    assert resp.json()["error"]["message"] == "boom"
    assert api.pending_requests == {}
    # Completed sibling (row 1) is not aborted; only the erroring row (row 0) aborts on early stream close
    assert aborted == [submitted_keys[0]]


def test_weight_version_chunk_precedence_seed_u64_and_cosmos3_0d_ndarray(monkeypatch, tmp_path):
    fake = _FakeActionsServer(_pi05_model(), tmp_path, model_name="pi05_libero")
    monkeypatch.setattr(entrypoint, "api_server", fake)

    # Chunk metadata weight_version on later tensor chunk must override PendingRequest.weight_version
    async def split_wv_chunks(rid, idx, sub):
        await asyncio.sleep(0)
        key = fake.submission_key(rid)
        fake.pending_requests[key] = SimpleNamespace(weight_version="v1", chunk_weight_version=None)
        yield ResultChunk(rid, "action", np.zeros((10, 32), dtype=np.float32).tobytes(), {})
        tensor = _pack_bytes_keys({
            "chain": np.zeros((4, 10, 32), dtype=np.float32),
            "denoise_ind": 0,
            "logprobs": np.zeros((10, 32), dtype=np.float32),
            "logprobs_mask": np.ones((10, 32), dtype=np.bool_),
            "seed": 42,
        })
        yield ResultChunk(rid, "tensor", tensor, {"rollout_state": True, "weight_version": "v2"})

    fake.chunks_fn = split_wv_chunks
    rollout_req = _pack_bytes_keys({
        "input": {"prompt": "pick", "images": np.zeros((1, 2, 3, 4, 4), dtype=np.uint8)},
        "parameters": {"rollout": {"sde_type": "flow_sde", "noise_level": 0.5}},
    })
    resp = _post(TestClient(entrypoint.app), rollout_req)
    assert resp.status_code == 200
    assert _unpack_strict_bytes(resp.content)["data"][0]["rollout"]["weight_version"] == "v2"

    # First chunk weight_version wins over later chunk weight_version (matching Rust find_map)
    async def multi_wv_chunks(rid, idx, sub):
        await asyncio.sleep(0)
        key = fake.submission_key(rid)
        fake.pending_requests[key] = SimpleNamespace(weight_version="v0", chunk_weight_version=None)
        yield ResultChunk(rid, "action", np.zeros((10, 32), dtype=np.float32).tobytes(), {"weight_version": "v2"})
        tensor = _pack_bytes_keys({
            "chain": np.zeros((4, 10, 32), dtype=np.float32),
            "denoise_ind": 0,
            "logprobs": np.zeros((10, 32), dtype=np.float32),
            "logprobs_mask": np.ones((10, 32), dtype=np.bool_),
            "seed": 2**63 - 1,
        })
        yield ResultChunk(rid, "tensor", tensor, {"rollout_state": True, "weight_version": "v3"})

    fake.chunks_fn = multi_wv_chunks
    resp2 = _post(TestClient(entrypoint.app), rollout_req)
    assert resp2.status_code == 200
    ro2 = _unpack_strict_bytes(resp2.content)["data"][0]["rollout"]
    assert ro2["weight_version"] == "v2"
    assert ro2["seed"] == 2**63 - 1

    # rollout.seed >= 2**63 rejected with 500 (matching Rust check_int64)
    async def u64_over_seed_chunks(rid, idx, sub):
        await asyncio.sleep(0)
        yield ResultChunk(rid, "action", np.zeros((10, 32), dtype=np.float32).tobytes(), {})
        tensor = _pack_bytes_keys({
            "chain": np.zeros((4, 10, 32), dtype=np.float32),
            "denoise_ind": 0,
            "logprobs": np.zeros((10, 32), dtype=np.float32),
            "logprobs_mask": np.ones((10, 32), dtype=np.bool_),
            "seed": 2**63,
        })
        yield ResultChunk(rid, "tensor", tensor, {"rollout_state": True})

    fake.chunks_fn = u64_over_seed_chunks
    resp_bad_seed = _post(TestClient(entrypoint.app), rollout_req)
    assert resp_bad_seed.status_code == 500

    # Cosmos3 0-D ndarray in parameters.{raw_action_dim, num_frames, action_chunk_size} -> 400
    fake_c3 = _FakeActionsServer(_Cosmos3StubModel(), tmp_path / "c3_0d", model_name="cosmos3")
    monkeypatch.setattr(entrypoint, "api_server", fake_c3)
    client = TestClient(entrypoint.app)
    for param_key in ("raw_action_dim", "num_frames", "action_chunk_size"):
        bad_0d = _pack_bytes_keys({
            "input": {"prompt": "pick"},
            "parameters": {param_key: np.array(10, dtype=np.int64)},
        })
        r = _post(client, bad_0d)
        assert r.status_code == 400, param_key

