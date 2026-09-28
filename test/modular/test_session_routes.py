"""The HTTP surface for sessions: ``/generate``'s flags and ``/sessions``.

Covers what a client can see — the status a refused session request gets, the
minted id coming back in the response, and the session flags reaching
``submit_request`` — with the engine stubbed out entirely.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
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

def test_list_sessions_reports_what_the_server_holds(monkeypatch):
    server = _FakeServer(_config())
    client = _client(monkeypatch, server)
    _generate(client, start_session="true", session_id="s")

    body = client.get("/sessions").json()

    assert body["sessions_enabled"] is True
    assert [s["session_id"] for s in body["sessions"]] == ["s"]
    assert body["sessions"][0]["active_request_ids"]


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
