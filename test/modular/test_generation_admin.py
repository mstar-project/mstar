"""Generation routes share this checked control path in both HTTP frontends."""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from mstar.api_server.auth import APIKeyGuard, APIKeyMiddleware
from mstar.api_server.generation_admin import GenerationAdmin, reload_roots
from mstar.api_server.generation_routes import router
from mstar.api_server.request_types import GenerationControlResult
from mstar.utils.ipc_format import ConductorMessageType


def test_control_messages_and_status(tmp_path):
    root = tmp_path / "weights"
    checkpoint = root / "next"
    checkpoint.mkdir(parents=True)
    sent = []

    def send(message):
        sent.append(message)
        admin.on_result(GenerationControlResult(
            op_id=message.body.op_id, status=200, state="paused", paused=True,
            weight_version="v2", rank_checksums={"worker_0": "a" * 64},
        ))

    admin = GenerationAdmin(send, key_enabled=True, roots=(root,))

    async def run():
        assert (await admin.call("pause_generation", {"mode": "wait"}))["paused"]
        assert (await admin.call("update_weights_from_disk", {
            "model_path": str(checkpoint), "weight_version": "v2",
        }))["weight_version"] == "v2"
        await admin.call("is_paused")
        assert (await admin.call("weight_checker", {"mode": "checksum"}))["rank_checksums"] == {
            "worker_0": "a" * 64,
        }
        await admin.call("continue_generation")

    asyncio.run(run())
    assert [m.message_type for m in sent] == [
        ConductorMessageType.PAUSE_GENERATION,
        ConductorMessageType.UPDATE_WEIGHTS_FROM_DISK,
        ConductorMessageType.GET_GENERATION_STATE,
        ConductorMessageType.CHECK_WEIGHTS,
        ConductorMessageType.CONTINUE_GENERATION,
    ]
    assert sent[0].body.mode == "wait"
    assert sent[1].body.model_path == str(checkpoint.resolve())
    assert sent[3].body.mode == "checksum"


def test_disabled_without_key_or_roots(tmp_path):
    for key, roots in ((False, (tmp_path,)), (True, ())):
        admin = GenerationAdmin(lambda _: pytest.fail("message escaped"), key_enabled=key, roots=roots)
        with pytest.raises(HTTPException, match="Not Found") as exc:
            asyncio.run(admin.call("is_paused"))
        assert exc.value.status_code == 404


def test_checkpoint_allowlist_rejects_symlink_escape(tmp_path):
    root, outside = tmp_path / "allowed", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    admin = GenerationAdmin(lambda _: pytest.fail("message escaped"), key_enabled=True, roots=(root,))

    for path in (str(outside), str(root / "escape"), "relative/path"):
        with pytest.raises(ValueError, match="checkpoint directory"):
            asyncio.run(admin.call("update_weights_from_disk", {"model_path": path, "weight_version": "v2"}))


def test_reload_roots_rejects_relative_or_unrestricted_root(tmp_path):
    assert reload_roots(str(tmp_path)) == (tmp_path.resolve(),)
    for value in ("relative", "/"):
        with pytest.raises(ValueError):
            reload_roots(value)


def test_python_routes_auth_and_path_guard(tmp_path, monkeypatch):
    root, outside = tmp_path / "allowed", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    sent = []

    def send(message):
        sent.append(message)
        admin.on_result(GenerationControlResult(
            op_id=message.body.op_id, status=409, state="paused", paused=True,
            error_message="rank still dirty",
        ))

    admin = GenerationAdmin(send, key_enabled=True, roots=(root,))
    monkeypatch.setattr("mstar.api_server.generation_routes._admin", lambda: admin)
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware, guard=APIKeyGuard("secret"))
    app.include_router(router)
    with TestClient(app) as client:
        assert client.post("/pause_generation", json={"mode": "wait"}).status_code == 401
        assert client.post("/weight_checker", json={"mode": "checksum"}).status_code == 401
        auth = {"Authorization": "Bearer secret"}
        response = client.post("/pause_generation", json={"mode": "wait"}, headers=auth)
        assert response.status_code == 409
        assert response.json()["error_message"] == "rank still dirty"
        assert client.post("/update_weights_from_disk", json={
            "model_path": str(outside), "weight_version": "v2",
        }, headers=auth).status_code == 400
        assert client.get("/is_paused", headers=auth).json()["paused"]
        assert client.post("/weight_checker", json={"mode": "checksum"}, headers=auth).status_code == 409
    assert [m.message_type for m in sent] == [
        ConductorMessageType.PAUSE_GENERATION, ConductorMessageType.GET_GENERATION_STATE,
        ConductorMessageType.CHECK_WEIGHTS,
    ]


def test_late_conductor_answer_after_http_timeout(tmp_path):
    sent = []
    admin = GenerationAdmin(sent.append, key_enabled=True, roots=(tmp_path,), timeout_s=0.001)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(admin.call("is_paused"))
    assert exc.value.status_code == 504
    # The HTTP waiter is gone, but a worker can still finish its step.
    admin.on_result(GenerationControlResult(
        op_id=sent[0].body.op_id, status=200, state="paused", paused=True,
    ))
