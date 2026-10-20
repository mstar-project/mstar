"""Anthropic's /v1/messages on the GLM chat path: the request translates onto
the chat adapter, and the reply comes back as a message or its event stream."""

import json
import sys
import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class _Chunk:
    def __init__(self, modality, data, metadata=None):
        self.modality, self.data, self.metadata = modality, data, metadata or {}


def _usage_chunks(*parts, stop=True):
    """Text chunks as the data worker tags them: per-chunk token counts and the prompt's."""
    return [_Chunk("text", text, {"tokens": n, "prompt_tokens": 11, "stop_token": stop and i == len(parts) - 1})
            for i, (text, n) in enumerate(parts)]


class _StubAPI:
    """The APIServer as the agent routes use it: submit, then read the chunks
    in batches; a reader that stops early aborts the request."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = types.SimpleNamespace()
        self.upload_dir = "/nonexistent"
        self.last_submit = None
        self.next_chunks: list = []
        self.aborted: list = []
        self.ended: dict = {}  # how each submission stopped early ended

    def submit_request(self, **kw):
        self.last_submit = kw
        return kw["request_id"]

    def submission_key(self, request_id):
        return request_id

    def abort_submission(self, key, outcome="abort"):
        # once per submission, as the real one: after a handler ends a whole
        # reply (a success), the stream's close aborts nothing more
        if key not in self.ended:
            self.ended[key] = outcome
            self.aborted.append(key)

    async def iter_result_chunk_batches(self, request_id):
        finished = False
        try:
            if self.next_chunks:
                yield list(self.next_chunks)
            finished = True
        finally:
            if not finished:
                self.abort_submission(request_id)

WEATHER = {"name": "get_weather", "description": "Weather by city",
           "input_schema": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}
CALL = b"<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"


@pytest.fixture
def client_and_stub(monkeypatch):
    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    stub = _StubAPI("glm52")
    fake_ep.api_server = stub
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)

    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), stub


def _body(**kw):
    return {"model": "glm-5.2", "max_tokens": 256, "messages": [{"role": "user", "content": "weather in Paris?"}],
            **kw}


def _events(text):
    """(event name, data) per server-sent event; the name must match the data's type."""
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        data = json.loads(lines["data"])
        assert lines["event"] == data["type"]
        out.append(data)
    return out


def test_a_conversation_translates_onto_the_chat_template(client_and_stub):
    from mstar.api_server.openai import serving_messages

    req, stops = serving_messages.to_chat_request(_body(
        system=[{"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Use tools."}],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Need it.", "signature": "sig"},
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "no such city"}],
                 "is_error": True},
                {"type": "text", "text": "and Rome?"}]},
        ],
        tools=[WEATHER], tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        stop_sequences=["END"], temperature=0.2, top_k=20, metadata={"user_id": "u"},
        thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096,
    ))
    msgs = [m.model_dump(exclude_none=True) for m in req.messages]
    assert msgs == [
        {"role": "system", "content": "You are terse.\nUse tools."},
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "Checking.", "reasoning_content": "Need it.", "tool_calls": [{
            "id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "Error: no such city"},
        {"role": "user", "content": "and Rome?"},
    ]
    extra = req.model_extra
    assert extra["tools"] == [{"type": "function", "function": {
        "name": "get_weather", "parameters": WEATHER["input_schema"], "description": "Weather by city"}}]
    assert extra["tool_choice"] == "auto" and extra["parallel_tool_calls"] is False
    assert extra["enable_thinking"] is True and extra["top_k"] == 20 and "metadata" not in extra
    assert req.max_tokens == 4096 and req.temperature == 0.2 and stops == ["END"]


@pytest.mark.parametrize("choice, want", [
    ({"type": "any"}, "required"),
    ({"type": "tool", "name": "get_weather"}, {"type": "function", "function": {"name": "get_weather"}}),
    ({"type": "none"}, "none"),
])
def test_tool_choice_maps_onto_chat(choice, want):
    from mstar.api_server.openai import serving_messages

    req, _ = serving_messages.to_chat_request(_body(tools=[WEATHER], tool_choice=choice))
    assert req.model_extra["tool_choice"] == want


@pytest.mark.parametrize("body, message", [
    (_body(max_tokens=None), "max_tokens"),
    ({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, "max_tokens"),
    (_body(messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBO"}}]}]), "reads text"),
    (_body(tools=[{"type": "web_search_20250305", "name": "web_search"}]), "server tool"),
    (_body(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure:"}]), "prefill"),
    (_body(thinking={"type": "enabled", "budget_tokens": 512}), "budget_tokens"),
    (_body(thinking={"type": "enabled", "budget_tokens": 4096}, max_tokens=1024), "less than max_tokens"),
    (_body(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": [
        {"type": "redacted_thinking", "data": "x"}]}, {"role": "user", "content": "go"}]), "redacted_thinking"),
    (_body(container="c1"), "container"),
    (_body(stop_sequences=["", "x"]), "stop_sequences"),
    (_body(tool_choice={"type": "tool"}), "name"),
    (_body(messages=[{"role": "system", "content": "x"}]), "role"),
])
def test_what_cannot_be_served_is_an_anthropic_400(client_and_stub, body, message):
    client, stub = client_and_stub
    r = client.post("/v1/messages", json={k: v for k, v in body.items() if v is not None})
    assert r.status_code == 400 and stub.last_submit is None
    err = r.json()
    assert err["type"] == "error" and err["error"]["type"] == "invalid_request_error"
    assert message in err["error"]["message"]


def test_a_forced_tool_with_thinking_on_is_a_400_as_anthropic_makes_it(client_and_stub):
    client, _ = client_and_stub
    thinking = {"type": "enabled", "budget_tokens": 1024}
    r = client.post("/v1/messages", json=_body(tools=[WEATHER], tool_choice={"type": "any"}, thinking=thinking,
                                               max_tokens=4096))
    assert r.status_code == 400 and "enable_thinking" in r.json()["error"]["message"]


def test_a_model_without_the_api_is_an_anthropic_404(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    r = client.post("/v1/messages", json=_body())
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"


def test_a_reply_is_a_message_with_thinking_text_and_tool_use(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _usage_chunks((b"Need weather.</think>", 4), (b"Checking. ", 2), (CALL, 9), stop=True)
    r = client.post("/v1/messages", json=_body(tools=[WEATHER]))
    assert r.status_code == 200
    msg = r.json()
    assert msg["type"] == "message" and msg["role"] == "assistant" and msg["model"] == "glm52"
    assert msg["content"][:2] == [
        {"type": "thinking", "thinking": "Need weather.", "signature": ""},
        {"type": "text", "text": "Checking."},
    ]
    use = msg["content"][2]
    assert use["type"] == "tool_use" and use["name"] == "get_weather" and use["input"] == {"city": "Paris"}
    assert msg["stop_reason"] == "tool_use" and msg["stop_sequence"] is None
    assert msg["usage"] == {"input_tokens": 11, "output_tokens": 15}
    assert stub.last_submit["model_kwargs"]["max_output_tokens"] == 256


@pytest.mark.parametrize("stop, reason", [(True, "end_turn"), (False, "max_tokens")])
def test_why_a_reply_ended(client_and_stub, stop, reason):
    client, stub = client_and_stub
    stub.next_chunks = _usage_chunks((b"Hm.</think>", 2), (b"Hello", 1), stop=stop)
    assert client.post("/v1/messages", json=_body()).json()["stop_reason"] == reason


def test_a_stop_sequence_ends_the_reply_and_is_named(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _usage_chunks((b"Hm.</think>", 2), (b"one two", 2), (b" three", 1), stop=False)
    msg = client.post("/v1/messages", json=_body(stop_sequences=["two"])).json()
    assert msg["content"][-1] == {"type": "text", "text": "one "}
    assert (msg["stop_reason"], msg["stop_sequence"]) == ("stop_sequence", "two")
    assert stub.aborted == [stub.last_submit["request_id"]]
    assert stub.ended == {stub.last_submit["request_id"]: "success"}  # a whole reply, served


def test_a_call_cut_off_by_max_tokens_has_no_tool_use(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _usage_chunks((b"Hm.</think>", 2), (CALL[:60], 6), stop=False)
    msg = client.post("/v1/messages", json=_body(tools=[WEATHER])).json()
    assert [b["type"] for b in msg["content"]] == ["thinking"] and msg["stop_reason"] == "max_tokens"


def test_the_stream_is_anthropics_event_sequence(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _usage_chunks((b"Need ", 1), (b"weather.</think>", 3), (b"Checking.", 2),
                                     (CALL[:45], 5), (CALL[45:], 4), stop=True)
    r = client.post("/v1/messages", json=_body(tools=[WEATHER], stream=True))
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    start = events[0]["message"]
    assert start["content"] == [] and start["usage"]["input_tokens"] == 11 and start["stop_reason"] is None
    blocks = [e for e in events if e["type"] == "content_block_start"]
    assert [b["content_block"]["type"] for b in blocks] == ["thinking", "text", "tool_use"]
    assert [b["index"] for b in blocks] == [0, 1, 2]
    assert blocks[2]["content_block"]["name"] == "get_weather" and blocks[2]["content_block"]["input"] == {}
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    assert "".join(d.get("thinking", "") for d in deltas) == "Need weather."
    assert "".join(d.get("text", "") for d in deltas) == "Checking."
    assert json.loads("".join(d.get("partial_json", "") for d in deltas)) == {"city": "Paris"}
    # each block stops before the next starts
    assert [e["index"] for e in events if e["type"] == "content_block_stop"] == [0, 1, 2]
    order = [(e["type"], e.get("index")) for e in events if e["type"].startswith("content_block_")]
    assert order.index(("content_block_stop", 0)) < order.index(("content_block_start", 1))
    assert events[-2]["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}
    assert events[-2]["usage"] == {"output_tokens": 15}


def test_a_stream_reports_a_failure_in_band(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = [_Chunk("text", b"Hi</think>", {"tokens": 1, "prompt_tokens": 3}),
                        _Chunk("error", b"worker died", {"status": 500})]
    events = _events(client.post("/v1/messages", json=_body(stream=True)).text)
    assert events[-1] == {"type": "error", "error": {"type": "api_error", "message": "worker died"}}
    assert "message_stop" not in [e["type"] for e in events] and stub.aborted == []


def test_count_tokens_renders_the_prompt_the_model_would_get(client_and_stub):
    import torch

    client, stub = client_and_stub
    seen = {}

    def process_prompt(text, input_modalities, output_modalities, **kw):
        seen.update(kw)
        return {"text_inputs": [torch.zeros(37, dtype=torch.long)]}

    stub.model.process_prompt = process_prompt
    r = client.post("/v1/messages/count_tokens", json={
        "model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}], "system": "Be brief.", "tools": [WEATHER]})
    assert r.json() == {"input_tokens": 37} and stub.last_submit is None
    assert [m["role"] for m in seen["messages"]] == ["system", "user"] and seen["tools"][0]["function"]["name"]
    r = client.post("/v1/messages/count_tokens", json={"messages": [{"role": "user", "content": "hi"}],
                                                        "max_tokens": 5})
    assert r.status_code == 400 and "max_tokens" in r.json()["error"]["message"]
