"""The HTTP surface for sessions: ``/generate``'s flags and ``/sessions``.

Covers what a client can see — the status a refused session request gets, the
minted id coming back in the response, and the session flags reaching
``submit_request`` — with the engine stubbed out entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.api_server.request_types import ResultChunk
from mstar.api_server.sessions import SessionRegistry
from mstar.model.sessions import SessionResourceConfig, SessionsConfig


class _FakeServer:
    """Just enough APIServer for the session routes."""

    def __init__(self, config=None, teardown=None):
        self.submitted: list[dict] = []
        self.torn_down: list[str] = []
        self.sessions = SessionRegistry(
            config, teardown=teardown or self.torn_down.append,
        )

    def submit_request(self, **kwargs):
        self.submitted.append(kwargs)
        return kwargs["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return []


def _config(**kwargs):
    return SessionsConfig(
        resources={"kv_cache": SessionResourceConfig(max_state=32)}, **kwargs
    )


def _client(monkeypatch, server):
    monkeypatch.setattr(entrypoint, "api_server", server)
    return TestClient(entrypoint.app)


def _generate(client, **data):
    data.setdefault("streaming", "false")
    return client.post("/generate", data=data)


# ── /generate ───────────────────────────────────────────────────────────────

def test_start_session_returns_the_minted_id_and_forwards_it(monkeypatch):
    server = _FakeServer(_config())
    response = _generate(_client(monkeypatch, server), start_session="true")

    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert session_id
    session = server.submitted[0]["session"]
    assert session.session_id == session_id
    assert session.created is True
    assert session.end_session is False


def test_a_request_without_session_flags_carries_no_session(monkeypatch):
    server = _FakeServer(_config())
    response = _generate(_client(monkeypatch, server), text="hi")

    assert response.status_code == 200
    assert "session_id" not in response.json()
    assert server.submitted[0]["session"].session_id is None


def test_resume_of_an_unknown_session_is_a_404(monkeypatch):
    server = _FakeServer(_config())
    response = _generate(
        _client(monkeypatch, server), resume_session="true", session_id="nope",
    )

    assert response.status_code == 404
    assert server.submitted == []


def test_start_on_a_taken_id_is_a_409(monkeypatch):
    server = _FakeServer(_config())
    client = _client(monkeypatch, server)
    _generate(client, start_session="true", session_id="mine")

    response = _generate(client, start_session="true", session_id="mine")

    assert response.status_code == 409


def test_resume_while_the_previous_request_is_in_flight_is_a_409(monkeypatch):
    server = _FakeServer(_config())
    client = _client(monkeypatch, server)
    # the fake server never completes the request, so the session stays busy
    _generate(client, start_session="true", session_id="s")

    response = _generate(client, resume_session="true", session_id="s")

    assert response.status_code == 409


def test_sessions_on_a_deployment_without_them_is_a_400(monkeypatch):
    server = _FakeServer(None)
    response = _generate(_client(monkeypatch, server), start_session="true")

    assert response.status_code == 400
    assert "does not support sessions" in response.json()["detail"]


def test_over_the_concurrency_cap_is_a_429(monkeypatch):
    server = _FakeServer(_config(max_concurrent_sessions=1))
    client = _client(monkeypatch, server)
    _generate(client, start_session="true", session_id="a")

    response = _generate(client, start_session="true", session_id="b")

    assert response.status_code == 429


def test_a_timeout_over_the_maximum_is_a_400(monkeypatch):
    server = _FakeServer(_config(default_timeout_s=30.0, max_timeout_s=60.0))
    response = _generate(
        _client(monkeypatch, server),
        start_session="true", session_timeout_s="600",
    )

    assert response.status_code == 400


def test_end_session_rides_on_the_request_and_holds_the_id(monkeypatch):
    server = _FakeServer(_config())
    client = _client(monkeypatch, server)
    _generate(client, start_session="true", session_id="s")

    assert server.submitted[0]["session"].end_session is False

    server.sessions.finish_request(server.submitted[0]["request_id"])
    _generate(client, resume_session="true", session_id="s", end_session="true")

    assert server.submitted[1]["session"].end_session is True
    # no standalone teardown: it rides on the request's own removal
    assert server.torn_down == []


def test_a_refused_submit_releases_the_session_it_claimed(monkeypatch):
    server = _FakeServer(_config())

    def _boom(**kwargs):
        raise RuntimeError("preprocess exploded")

    server.submit_request = _boom
    client = _client(monkeypatch, server)

    response = _generate(client, start_session="true", session_id="s")

    assert response.status_code == 500
    # the failed request took the session with it, rather than leaving it
    # busy forever with a request that never ran
    assert server.torn_down == ["s"]


# ── /sessions ───────────────────────────────────────────────────────────────

def test_get_sessions_counts_without_naming_anyone(monkeypatch):
    server = _FakeServer(_config(max_concurrent_sessions=4))
    client = _client(monkeypatch, server)
    _generate(client, start_session="true", session_id="secret-id")

    body = client.get("/sessions").json()

    assert body == {
        "sessions_enabled": True, "live": 1, "closing": 0,
        "max_concurrent_sessions": 4,
    }
    assert "secret-id" not in client.get("/sessions").text


def test_get_session_describes_one_its_caller_names(monkeypatch):
    server = _FakeServer(_config())
    client = _client(monkeypatch, server)
    _generate(client, start_session="true", session_id="s")

    body = client.get("/sessions/s").json()

    assert body["session_id"] == "s"
    assert body["active_request_ids"]
    assert body["closing"] is False
    assert client.get("/sessions/other").status_code == 404


def test_delete_session_closes_it(monkeypatch):
    server = _FakeServer(_config())
    client = _client(monkeypatch, server)
    _generate(client, start_session="true", session_id="s")
    server.sessions.finish_request(server.submitted[0]["request_id"])

    response = client.delete("/sessions/s")

    assert response.status_code == 200
    assert response.json() == {"session_id": "s", "status": "closing"}
    assert server.torn_down == ["s"]


def test_delete_of_an_unknown_session_is_a_404(monkeypatch):
    server = _FakeServer(_config())

    response = _client(monkeypatch, server).delete("/sessions/nope")

    assert response.status_code == 404


def test_delete_with_a_request_in_flight_is_a_409(monkeypatch):
    server = _FakeServer(_config())
    client = _client(monkeypatch, server)
    _generate(client, start_session="true", session_id="s")

    response = client.delete("/sessions/s")

    assert response.status_code == 409
    assert server.torn_down == []


def test_the_non_streaming_body_reports_the_session_once(monkeypatch):
    server = _FakeServer(_config())

    async def _collect(request_id, raw_request=None):
        return [
            ResultChunk(
                request_id=request_id, modality="session", data=b"s",
                metadata={"session_id": "s"},
            ),
            ResultChunk(request_id=request_id, modality="text", data=b"ok"),
        ]

    server.collect_results = _collect
    response = _generate(
        _client(monkeypatch, server), start_session="true", session_id="s",
    )

    body = response.json()
    assert body["session_id"] == "s"
    assert set(body["outputs"]) == {"text"}


def test_a_second_delete_of_a_closing_session_is_a_409(monkeypatch):
    server = _FakeServer(_config())
    client = _client(monkeypatch, server)
    _generate(client, start_session="true", session_id="s")
    server.sessions.finish_request(server.submitted[0]["request_id"])
    assert client.delete("/sessions/s").status_code == 200

    response = client.delete("/sessions/s")

    assert response.status_code == 409
    assert server.torn_down == ["s"]


# ── /generate/ws ────────────────────────────────────────────────────────────

class _WsServer(_FakeServer):
    """Streams back what ``submit_request`` queues: the session frame it
    prepends, then the output."""

    upload_dir = Path("/tmp")

    async def iter_result_chunks(self, request_id):
        session = self.submitted[-1]["session"]
        if session.session_id is not None:
            yield ResultChunk(
                request_id=request_id, modality="session",
                data=session.session_id.encode(),
                metadata={"session_id": session.session_id},
            )
        yield ResultChunk(request_id=request_id, modality="text", data=b"ok")


def _ws_turn(client, **message) -> list[dict]:
    frames = []
    with client.websocket_connect("/generate/ws") as ws:
        ws.send_json({"text": "hi", "request_id": "r0", **message})
        while True:
            frame = ws.receive_json()
            frames.append(frame)
            if frame.get("finish") or frame.get("error"):
                return frames


def test_a_ws_session_reply_carries_the_session_frame_once(monkeypatch):
    client = _client(monkeypatch, _WsServer(_config()))

    frames = _ws_turn(client, start_session=True, session_id="s")

    assert [f.get("modality") for f in frames].count("session") == 1


def test_a_ws_session_flag_must_be_a_boolean(monkeypatch):
    server = _WsServer(_config())
    client = _client(monkeypatch, server)

    [frame] = _ws_turn(client, start_session="false")

    assert "must be a boolean" in frame["error"]
    assert server.submitted == []
    assert server.sessions.snapshot() == []
