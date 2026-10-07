"""Persistent sessions on ``/v1/chat/completions``.

The OpenAI route is the surface a chat client actually uses, so the session
fields have to work there as they do on ``POST /generate``: the registry
validates them, the resolved session reaches ``submit_request``, and the id the
server minted comes back to the client — in the response body, or on the
stream's opening chunk. Everything below stubs the engine out.
"""

from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, ".")

import pytest

from mstar.api_server.openai import serving_chat
from mstar.api_server.openai.adapters import SubmitArgs
from mstar.api_server.openai.protocol import ChatCompletionRequest
from mstar.api_server.request_types import ResultChunk
from mstar.api_server.sessions import SessionError, SessionRegistry
from mstar.model.sessions import SessionResourceConfig, SessionsConfig


class _Adapter:
    def chat_to_request(self, req, upload_dir):
        del upload_dir
        return SubmitArgs(
            text=req.messages[0].content, input_modalities=["text"],
        )


class _Api:
    """Just enough APIServer for the chat handler."""

    def __init__(self, sessions_config=None, chunks=()):
        self.submitted: list[dict] = []
        self.torn_down: list[str] = []
        self.upload_dir = "/tmp"
        self.model = None
        self.sessions = SessionRegistry(
            sessions_config, teardown=self.torn_down.append,
        )
        self._chunks = list(chunks)

    def submit_request(self, **kwargs):
        self.submitted.append(kwargs)
        return kwargs["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        del request_id, raw_request
        return self._chunks

    async def iter_result_chunks(self, request_id):
        del request_id
        for chunk in self._chunks:
            yield chunk


def _config(**kwargs):
    return SessionsConfig(
        resources={"kv_cache": SessionResourceConfig(max_state=32)}, **kwargs
    )


def _request(**fields):
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}], **fields,
    )


def _chat(api, **fields):
    return asyncio.run(serving_chat.create_chat_completion(
        api, "bagel", _Adapter(), _request(**fields),
    ))


def _text(data: str) -> ResultChunk:
    return ResultChunk(request_id="r", modality="text", data=data.encode())


def _sse_objects(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.split("\n\n")
        if line.startswith("data: ") and "[DONE]" not in line
    ]


# ── the session reaches the engine ──────────────────────────────────────────

def test_a_chat_request_can_open_a_session():
    api = _Api(_config(), chunks=[_text("ok")])

    body = _chat(api, start_session=True)

    [submitted] = api.submitted
    session = submitted["session"]
    assert session.session_id is not None
    assert session.created is True
    # and the client is told which session it got
    assert body["session_id"] == session.session_id


def test_a_chat_request_can_resume_a_session():
    api = _Api(_config(), chunks=[_text("ok")])
    opened = _chat(api, start_session=True, session_id="s")
    api.sessions.finish_request(api.submitted[0]["request_id"])

    _chat(api, resume_session=True, session_id="s")

    assert opened["session_id"] == "s"
    assert api.submitted[-1]["session"].resumed is True


def test_a_chat_request_can_end_its_session():
    api = _Api(_config(), chunks=[_text("ok")])
    _chat(api, start_session=True, session_id="s")
    api.sessions.finish_request(api.submitted[0]["request_id"])

    _chat(api, resume_session=True, session_id="s", end_session=True)

    # the tombstone that holds the id from here is submit_request's, which this
    # stub replaces; test_api_server_sessions covers it against the real one
    assert api.submitted[-1]["session"].end_session is True


def test_a_chat_request_with_no_session_fields_names_no_session():
    api = _Api(_config(), chunks=[_text("ok")])

    body = _chat(api)

    assert api.submitted[0]["session"].session_id is None
    assert "session_id" not in body


# ── what a refused session looks like ───────────────────────────────────────

@pytest.mark.parametrize(
    ("fields", "status"),
    [
        ({"resume_session": True, "session_id": "nope"}, 404),
        ({"start_session": True, "resume_session": True}, 400),
        ({"resume_session": True}, 400),
        ({"start_session": True, "session_timeout_s": 10_000_000}, 400),
    ],
)
def test_a_refused_session_raises_with_the_status_the_router_reports(fields, status):
    api = _Api(_config())

    with pytest.raises(SessionError) as e:
        _chat(api, **fields)

    # the router answers with getattr(exc, "status_code", 500)
    assert e.value.status_code == status
    assert api.submitted == []


def test_a_deployment_without_sessions_refuses_the_fields():
    api = _Api(None)

    with pytest.raises(SessionError) as e:
        _chat(api, start_session=True)

    assert e.value.status_code == 400


# ── streaming ───────────────────────────────────────────────────────────────

def test_the_stream_names_the_session_on_its_opening_chunk():
    api = _Api(_config(), chunks=[
        ResultChunk(request_id="r", modality="session", data=b"ignored"),
        _text("ok"),
    ])

    stream = _chat(api, start_session=True, stream=True)
    body = "".join(asyncio.run(_drain(stream)))

    objects = _sse_objects(body)
    session_id = api.submitted[0]["session"].session_id
    assert objects[0]["session_id"] == session_id
    # reported once, and the server's own session chunk is not sent as content
    assert [o for o in objects[1:] if "session_id" in o] == []
    assert "ignored" not in body


def test_a_sessionless_stream_names_no_session():
    api = _Api(_config(), chunks=[_text("ok")])

    body = "".join(asyncio.run(_drain(_chat(api, stream=True))))

    assert all("session_id" not in o for o in _sse_objects(body))


async def _drain(stream) -> list[str]:
    return [part async for part in stream]


def test_a_refused_submit_releases_the_session_it_claimed():
    api = _Api(_config())

    def _boom(**kwargs):
        raise RuntimeError("preprocess exploded")

    api.submit_request = _boom

    with pytest.raises(RuntimeError):
        _chat(api, start_session=True, session_id="s")

    # the failed request took the session with it, rather than leaving it
    # busy forever with a request that never ran
    assert api.torn_down == ["s"]
