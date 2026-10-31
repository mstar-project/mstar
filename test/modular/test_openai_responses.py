"""OpenAI's /v1/responses, stateless, on the GLM chat path: input items
translate onto the chat adapter, and the reply comes back as a response
object or its event stream."""

import json
import sys
import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from test_anthropic_messages import _Chunk, _events, _StubAPI, _usage_chunks  # noqa: E402


@pytest.fixture
def client_and_stub(monkeypatch):
    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    stub = _StubAPI("glm5_next")
    fake_ep.api_server = stub
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)

    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), stub


SHELL = {"type": "function", "name": "shell", "description": "Run a command", "strict": False,
         "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}},
                                                         "workdir": {"type": "string"}}}}
CALL = (b"<tool_call>shell<arg_key>command</arg_key><arg_value>[\"ls\", \"-a\"]</arg_value>"
        b"<arg_key>workdir</arg_key><arg_value>/repo</arg_value></tool_call>")


def _codex(**kw):
    """A request as Codex sends one."""
    return {"model": "glm-5.3", "instructions": "You are a coding agent.", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list the repo"}]}],
        "tools": [SHELL], "tool_choice": "auto", "parallel_tool_calls": False, "store": False, "stream": False,
        "reasoning": {"effort": "medium", "summary": "auto"}, "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "conv-1", "client_metadata": {"session_id": "s1"}, **kw}


def test_input_items_translate_onto_the_chat_template():
    from mstar.api_server.openai import adapters, serving_responses

    req = serving_responses.to_chat_request(_codex(input=[
        {"type": "message", "role": "developer", "content": "Prefer rg."},
        {"role": "user", "content": "list the repo"},
        {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "ls it"}]},
        {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": '{"command": ["ls"]}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "a.py\nb.py"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Two files."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "and hidden ones?"}]},
    ], max_output_tokens=900), ("low", "high", "max"))
    msgs = [m.model_dump(exclude_none=True) for m in req.messages]
    assert msgs == [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "developer", "content": "Prefer rg."},
        {"role": "user", "content": "list the repo"},
        {"role": "assistant", "reasoning_content": "ls it", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": '{"command": ["ls"]}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py\nb.py"},
        {"role": "assistant", "content": "Two files."},
        {"role": "user", "content": "and hidden ones?"},
    ]
    extra = req.model_extra
    assert extra["tools"] == [{"type": "function", "function": {
        "name": "shell", "parameters": SHELL["parameters"], "description": "Run a command"}}]
    assert extra["parallel_tool_calls"] is False and extra["reasoning_effort"] == "high"
    assert req.max_tokens == 900 and "prompt_cache_key" not in extra and "client_metadata" not in extra
    # the GLM template takes it as it is: tool results, calls, reasoning
    mk = adapters.get_adapter("glm5_next").chat_to_request(req, "/nonexistent").model_kwargs
    assert [m["role"] for m in mk["messages"]] == ["system", "system", "user", "assistant", "tool", "assistant", "user"]
    assert mk["messages"][3]["tool_calls"][0]["function"]["arguments"] == {"command": ["ls"]}


@pytest.mark.parametrize("effort, glm52, glm53", [
    ("minimal", "high", "low"), ("low", "high", "low"), ("medium", "high", "high"),
    ("high", "max", "max"), ("xhigh", "max", "max"),
])
def test_reasoning_effort_maps_onto_each_templates_levels(effort, glm52, glm53):
    from mstar.api_server.openai import adapters, serving_responses

    for model, want in (("glm52", glm52), ("glm5_next", glm53)):
        req = serving_responses.to_chat_request(_codex(reasoning={"effort": effort}),
                                                adapters.get_adapter(model).reasoning_efforts)
        assert req.model_extra["reasoning_effort"] == want


def test_effort_none_turns_thinking_off_where_the_model_can(client_and_stub):
    from mstar.api_server.openai import serving_responses

    req = serving_responses.to_chat_request(_codex(reasoning={"effort": "none"}), ("high", "max"))
    assert req.model_extra["enable_thinking"] is False
    client, _ = client_and_stub  # GLM-5.3 always reasons
    r = client.post("/v1/responses", json=_codex(reasoning={"effort": "none"}))
    assert r.status_code == 400 and "always reasons" in r.json()["error"]["message"]


@pytest.mark.parametrize("field, message", [
    ({"previous_response_id": "resp_1"}, "previous_response_id"),
    ({"store": True}, "store"),
    ({"background": True}, "background"),
    ({"conversation": "conv_1"}, "conversation"),
    ({"tools": [{"type": "web_search"}]}, "web_search"),
    ({"tools": [{"type": "custom", "name": "apply_patch"}]}, "custom"),
    ({"tools": [dict(SHELL, strict=True)]}, "strict"),
    ({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:"}]}]}, "reads text"),
    ({"text": {"format": {"type": "json_schema", "name": "x", "schema": {}}}}, "constrained decoding"),
    ({"text": "json"}, "constrained decoding"),
    ({"text": {"format": "json_object"}}, "constrained decoding"),
    ({"reasoning": {"effort": "high", "summary": "detailed"}}, "summary"),
    ({"include": ["message.output_text.logprobs"]}, "include"),
    ({"input": [{"type": "item_reference", "id": "msg_1"}]}, "item_reference"),
    ({"input": [{"type": "reasoning", "summary": [], "encrypted_content": "gAAA"},
                {"role": "user", "content": "go"}]}, "encrypted_content"),
    ({"input": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure"}]}, "assistant turn"),
    ({"truncation": "auto"}, "truncation"),
    ({"seed": 3}, "seed"),
    ({"tool_choice": {"type": "file_search"}}, "tool_choice"),
])
def test_what_needs_state_or_a_server_tool_is_a_400(client_and_stub, field, message):
    client, stub = client_and_stub
    r = client.post("/v1/responses", json=_codex(**field))
    assert r.status_code == 400 and stub.last_submit is None
    assert message in r.json()["error"]["message"] and r.json()["error"]["type"] == "invalid_request_error"


def test_a_reply_is_a_response_with_reasoning_message_and_call(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _usage_chunks((b"List ", 2), (b"it.</think>", 2), (b"Listing.", 2), (CALL, 20), stop=True)
    r = client.post("/v1/responses", json=_codex())
    assert r.status_code == 200
    resp = r.json()
    assert resp["object"] == "response" and resp["status"] == "completed" and resp["model"] == "glm5_next"
    assert resp["store"] is False and resp["error"] is None and resp["incomplete_details"] is None
    reasoning, message, call = resp["output"]
    assert reasoning["type"] == "reasoning" and reasoning["summary"] == []
    assert reasoning["content"] == [{"type": "reasoning_text", "text": "List it."}]
    assert message["type"] == "message" and message["role"] == "assistant" and message["status"] == "completed"
    assert message["content"] == [{"type": "output_text", "text": "Listing.", "annotations": [], "logprobs": []}]
    assert call["type"] == "function_call" and call["name"] == "shell" and call["status"] == "completed"
    assert call["call_id"].startswith("call_") and call["id"].startswith("fc")
    assert json.loads(call["arguments"]) == {"command": ["ls", "-a"], "workdir": "/repo"}
    assert resp["usage"] == {"input_tokens": 11, "output_tokens": 26, "output_tokens_details": {"reasoning_tokens": 4},
                             "total_tokens": 37}
    # one call (parallel_tool_calls false) and the effort reach the request
    mk = stub.last_submit["model_kwargs"]
    assert mk["chat_template_kwargs"] == {"reasoning_effort": "high"} and "parallel_tool_calls" not in mk


def test_a_reply_cut_off_by_max_output_tokens_is_incomplete(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _usage_chunks((b"Hm.</think>", 2), (CALL[:70], 7), stop=False)
    resp = client.post("/v1/responses", json=_codex(max_output_tokens=9)).json()
    assert resp["status"] == "incomplete" and resp["incomplete_details"] == {"reason": "max_output_tokens"}
    call = resp["output"][-1]
    assert call["status"] == "incomplete" and call["arguments"] == '{"command": ["ls", "-a"]'[:len(call["arguments"])]


def test_the_stream_is_the_responses_event_sequence(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _usage_chunks((b"List ", 2), (b"it.</think>", 2), (b"Listing.", 2), (CALL[:50], 8),
                                     (CALL[50:], 12), stop=True)
    r = client.post("/v1/responses", json=_codex(stream=True))
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
    assert events[0]["response"]["status"] == "in_progress" and events[0]["response"]["output"] == []
    added = [e for e in events if e["type"] == "response.output_item.added"]
    done = [e for e in events if e["type"] == "response.output_item.done"]
    assert [e["item"]["type"] for e in added] == ["reasoning", "message", "function_call"]
    assert [e["item"].get("content") for e in added] == [[], [], None]
    assert [e["output_index"] for e in done] == [0, 1, 2]
    # each item is done before the next is added
    assert kinds.index("response.output_item.done") < [i for i, k in enumerate(kinds)
                                                       if k == "response.output_item.added"][1]
    assert "".join(e["delta"] for e in events if e["type"] == "response.reasoning_text.delta") == "List it."
    assert "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta") == "Listing."
    args = "".join(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
    (args_done,) = [e for e in events if e["type"] == "response.function_call_arguments.done"]
    assert args == args_done["arguments"] and json.loads(args)["workdir"] == "/repo"
    assert [e["part"]["type"] for e in events if e["type"] == "response.content_part.added"] == [
        "reasoning_text", "output_text"]
    final = events[-1]["response"]
    assert final["status"] == "completed" and [i["status"] for i in final["output"]] == ["completed"] * 3
    assert final["output"] == [e["item"] for e in done] and final["usage"]["output_tokens"] == 26


def test_a_stream_reports_a_failure_as_response_failed(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = [_Chunk("text", b"Hm", {"tokens": 1, "prompt_tokens": 3}),
                        _Chunk("error", b"worker died", {"status": 500})]
    events = _events(client.post("/v1/responses", json=_codex(stream=True)).text)
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"] == {"code": "server_error", "message": "worker died"}
    assert stub.aborted == []


def test_a_stream_the_engine_refuses_answers_with_its_status(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = [_Chunk("error", b"prompt too long", {"status": 400})]
    r = client.post("/v1/responses", json=_codex(stream=True))
    assert r.status_code == 400 and r.json()["error"]["message"] == "prompt too long"
    assert stub.aborted == []


def test_a_model_without_the_api_is_a_404(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    assert client.post("/v1/responses", json=_codex()).status_code == 404
