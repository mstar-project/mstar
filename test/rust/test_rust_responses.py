"""OpenAI's ``/v1/responses`` on mstar-server (``rust/server/src/responses/``),
stateless, for Codex on the GLMs.

Each request goes to mstar-server over the real bridge. A request is refused
with the Python route's words, or submitted as the chat the Python route
translated it into (written out from ``serving_responses``) is submitted by
the live ``GlmChatAdapter``: the template's messages and tools in their order.
A reply is held to what the Python route answers (live, over a data plane
handing out the same chunks): the status and body or every event of the
stream, and the outcome the request was stopped with; cut into smaller chunks,
the model's bytes must come to the same reply. Skipped unless
the ``mstar_rust`` extension is installed and the server binary is built."""
import contextlib

import pytest

pytest.importorskip("mstar_rust")

from test_rust_frontend import BINARY
from test_rust_glm_chat import (
    CALL,
    WEATHER,
    _answer,
    _folded,
    _glm_stack,
    _GlmStub,
    _pieces,
    _python_side,
    _rust_answer,
    _translated_alike,
)

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

TOOL = {"type": "function", "name": "get_weather", "description": "Weather by city",
        "parameters": WEATHER["function"]["parameters"], "strict": False}


def _codex(model="glm52", **fields):
    return {"model": model, "input": [{"type": "message", "role": "user",
                                       "content": [{"type": "input_text", "text": "weather in Paris?"}]}], **fields}


def _chat(model="glm52", **fields):
    """The chat ``_codex(model)`` asks for, with ``fields``."""
    return {"model": model, "messages": [{"role": "user", "content": "weather in Paris?"}], "stream": False, **fields}


CHAT_TOOL = {"type": "function", "function": {
    "name": "get_weather", "parameters": WEATHER["function"]["parameters"], "description": "Weather by city"}}


CODEX_TURN = [
    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Be careful."}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
    {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "Need it."}],
     "encrypted_content": None},
    {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
     "arguments": '{"zip": "75001", "city": "Paris"}'},
    {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "It is sunny."}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "and Rome?"}]},
]

# name: (Responses body, the chat it asks for or its refusal)
REQUEST_CASES = {
    "codex-turn": (_codex(
        "glm5_next", instructions="You are Codex.", input=CODEX_TURN, tools=[TOOL], tool_choice="auto",
        parallel_tool_calls=False, reasoning={"effort": "medium", "summary": "auto"}, store=False,
        include=["reasoning.encrypted_content"], text={"format": {"type": "text"}, "verbosity": "medium"},
        prompt_cache_key="k", metadata={"a": "b"}, client_metadata={"c": "d"}, stream_options={"x": 1},
        service_tier="auto", truncation="disabled", user="u", safety_identifier="s", max_output_tokens=4096,
        temperature=0.7, top_p=0.95),
        _chat(
            "glm5_next",
            messages=[
                {"role": "system", "content": "You are Codex."},
                {"role": "developer", "content": "Be careful."},
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "reasoning_content": "Need it.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"zip\": \"75001\", \"city\": \"Paris\"}",
                            },
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                {"role": "assistant", "content": "It is sunny."},
                {"role": "user", "content": "and Rome?"},
            ],
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
            parallel_tool_calls=False,
            tools=[CHAT_TOOL],
            tool_choice="auto",
            reasoning_effort="high",
        )),
    "string-input": ({"model": "glm52", "input": "hello"},
        _chat(messages=[{"role": "user", "content": "hello"}])),
    "reasoning-then-message": (_codex(input=[
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "From the summary."}]},
        {"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "No."}]},
        {"type": "message", "role": "user", "content": "again"}]),
        _chat(
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "reasoning_content": "From the summary.", "content": "No."},
                {"role": "user", "content": "again"},
            ],
        )),
    "two-calls-one-turn": (_codex(input=[
        {"type": "message", "role": "user", "content": "q"},
        {"type": "function_call", "call_id": "a", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call", "call_id": "b", "name": "get_weather", "arguments": '{"city": "Rome"}'},
        {"type": "function_call_output", "call_id": "a", "output": [{"type": "input_text", "text": "rain"}]},
        {"type": "function_call_output", "call_id": "b", "output": "sun"}], tools=[TOOL]),
        _chat(
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        },
                        {
                            "id": "b",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "rain"},
                {"role": "tool", "tool_call_id": "b", "content": "sun"},
            ],
            tools=[CHAT_TOOL],
        )),
    "effort-none": (_codex(reasoning={"effort": "none"}),
        _chat(enable_thinking=False)),
    "effort-xhigh": (_codex(reasoning={"effort": "xhigh"}),
        _chat(reasoning_effort="max")),
    "effort-minimal-glm53": (_codex("glm5_next", reasoning={"effort": "minimal"}),
        _chat("glm5_next", reasoning_effort="low")),
    "effort-high-glm53": (_codex("glm5_next", reasoning={"effort": "high"}),
        _chat("glm5_next", reasoning_effort="high")),
    "prefix-cache-off": (_codex(prefix_cache=False),
        _chat(prefix_cache=False)),
    "tool-choice-function": (_codex(tools=[TOOL], tool_choice={"type": "function", "name": "get_weather"}),
        _chat(tools=[CHAT_TOOL], tool_choice={"type": "function", "function": {"name": "get_weather"}})),
    "tool-choice-required": (_codex(tools=[TOOL], tool_choice="required"),
        _chat(tools=[CHAT_TOOL], tool_choice="required")),
    "tool-without-parameters": (_codex(tools=[{"type": "function", "name": "now"}]),
        _chat(
            tools=[
                {
                    "type": "function",
                    "function": {"name": "now", "parameters": {"type": "object", "properties": {}}},
                },
            ],
        )),
    "store-zero": (_codex(store=0),
        _chat()),
    "top-logprobs-false": (_codex(top_logprobs=False),
        _chat()),
    "stream-off-values": (_codex(stream=None, background=False, conversation=None),
        _chat()),
    "glm53-effort-none": (_codex("glm5_next", reasoning={"effort": "none"}),
        _chat("glm5_next", enable_thinking=False)),
    "not-an-object": ([_codex()],
        "the request body must be a JSON object"),
    "unknown-field": (_codex(modalities=["text"]),
        "unsupported field(s) ['modalities']"),
    "previous-response-id": (_codex(previous_response_id="resp_1"),
        "previous_response_id='resp_1' is not served: responses are stateless and not stored"),
    "store-true": (_codex(store=True),
        "store=True is not served: responses are stateless and not stored"),
    "background": (_codex(background=True),
        "background=True is not served: responses are stateless and not stored"),
    "top-logprobs": (_codex(top_logprobs=5),
        "top_logprobs=5 is not served: responses are stateless and not stored"),
    "truncation-auto": (_codex(truncation="auto"),
        "truncation='auto' is not served: responses are stateless and not stored"),
    "service-tier-flex": (_codex(service_tier="flex"),
        "service_tier='flex' is not served: responses are stateless and not stored"),
    "include-other": (_codex(include=["message.output_text.logprobs"]),
        "include takes only 'reasoning.encrypted_content'; got ['message.output_text.logprobs']"),
    "text-json-schema": (_codex(text={"format": {"type": "json_schema", "schema": {}}}),
        "text takes the text format only, structured output needs constrained decoding; got {'format': "
        "{'type': 'json_schema', 'schema': {}}}"),
    "text-verbosity-low": (_codex(text={"verbosity": "low"}),
        "text takes the text format only, structured output needs constrained decoding; got {'verbosity': 'low'}"),
    "text-a-string": (_codex(text="plain"),
        "text takes the text format only, structured output needs constrained decoding; got 'plain'"),
    "input-empty": (_codex(input=[]),
        "input must be a string or a non-empty list of items"),
    "input-item-a-string": (_codex(input=["hi"]),
        "input items are objects; got 'hi'"),
    "item-web-search": (_codex(input=[{"type": "web_search_call", "id": "w"}]),
        "input item type 'web_search_call' is not served; send message, function_call, function_call_output "
        "or reasoning items"),
    "role-tool": (_codex(input=[{"type": "message", "role": "tool", "content": "x"}]),
        "a message's role must be user, assistant, system or developer, not 'tool'"),
    "input-image": (_codex(input=[{"type": "message", "role": "user",
                                  "content": [{"type": "input_image", "image_url": "data:"}]}]),
        "a user message's content takes text parts only; got 'input_image' (the model reads text)"),
    "content-a-number": (_codex(input=[{"type": "message", "role": "user", "content": 5}]),
        "a user message's content must be a string or a list of content parts"),
    "function-call-missing-name": (_codex(input=[{"type": "message", "role": "user", "content": "q"},
                                                {"type": "function_call", "call_id": "a", "arguments": "{}"}]),
        "a function_call item needs its call_id, name and arguments"),
    "output-missing-call-id": (_codex(input=[{"type": "function_call_output", "output": "x"}]),
        "a function_call_output item needs its call_id"),
    "encrypted-only-reasoning": (_codex(input=[{"type": "reasoning", "summary": [], "encrypted_content": "e"},
                                              {"type": "message", "role": "user", "content": "q"}]),
        "a reasoning item with only encrypted_content cannot be read; send its content"),
    "reasoning-not-text": (_codex(input=[{"type": "reasoning", "content": [{"type": "reasoning_text", "text": 5}]},
                                        {"type": "message", "role": "user", "content": "q"}]),
        "a reasoning item's content and summary are text parts; got {'type': 'reasoning', 'content': "
        "[{'type': 'reasoning_text', 'text': 5}]}"),
    "ends-with-assistant": (_codex(input=[{"type": "message", "role": "user", "content": "q"},
                                         {"type": "message", "role": "assistant", "content": "a"}]),
        "the input ends with an assistant turn; end with a user message or a function_call_output"),
    "tool-web-search": (_codex(tools=[{"type": "web_search"}]),
        "tools take function tools only; 'web_search' is not served"),
    "tool-no-name": (_codex(tools=[{"type": "function"}]),
        "a function tool needs a name; got {'type': 'function'}"),
    "tool-strict": (_codex(tools=[{**TOOL, "strict": True}]),
        "tool 'get_weather': strict schemas need constrained decoding; send strict: false"),
    "tools-not-a-list": (_codex(tools={"type": "function"}),
        "tools must be a list"),
    "tool-choice-bad": (_codex(tools=[TOOL], tool_choice={"type": "custom"}),
        "tool_choice takes 'auto', 'none', 'required' or a function by name; got {'type': 'custom'}"),
    "reasoning-a-string": (_codex(reasoning="high"),
        "reasoning must be an object; got 'high'"),
    "reasoning-summary-detailed": (_codex(reasoning={"summary": "detailed"}),
        "reasoning.summary='detailed': no summary is made; the reasoning comes whole"),
    "effort-bad": (_codex(reasoning={"effort": "max"}),
        "reasoning.effort takes none, minimal, low, medium, high or xhigh; got 'max'"),
    "prefix-cache-not-a-bool": (_codex(prefix_cache=1),
        "prefix_cache must be a boolean; got 1"),
    "lora-unknown": (_codex(model="no-such-adapter"),
        _chat("no-such-adapter")),
}


@pytest.fixture(scope="module")
def servers():
    """One mstar-server per GLM, for the whole module."""
    with contextlib.ExitStack() as stack:
        out = {}
        for model in ("glm52", "glm5_next"):
            stub = _GlmStub()
            port, _upload = stack.enter_context(_glm_stack(model, stub))
            out[model] = (port, stub)
        yield out


def _model(body):
    return body.get("model") if isinstance(body, dict) and body.get("model") in ("glm52", "glm5_next") else "glm52"


@pytest.mark.parametrize("case", sorted(REQUEST_CASES))
def test_a_response_request_is_translated_or_refused_as_python_did(case, servers, tmp_path):
    pytest.importorskip("pydantic")
    body, want = REQUEST_CASES[case]
    port, stub = servers[_model(body)]
    stub.reply, stub.finish, stub.ends_early, stub.fail_after = [b"ok"], "stop", False, None
    answer, _, submitted = _rust_answer(port, stub, "/v1/responses", body)
    if answer[0] != 200:
        assert (answer[1]["error"]["type"], answer[1]["error"]["code"]) == ("invalid_request_error", 400)
    _translated_alike(answer, submitted, want, _model(body), tmp_path)


TIME_CALL = "<tool_call>get_time</tool_call>"
# name: (model, request fields, the model's reply, its finish reason, whether the frontend ends it)
REPLY_CASES = {
    "reasoning-and-text": ("glm52", {}, "Let me think.</think>\n\nHello, Paris.", "stop", False),
    "function-call": ("glm52", {"tools": [TOOL]}, "Need the weather.</think>" + CALL, "stop", False),
    "text-then-calls": ("glm5_next", {"tools": [TOOL]}, "Plan</think>Checking." + CALL + CALL, "stop", False),
    "incomplete-call": ("glm52", {"tools": [TOOL], "max_output_tokens": 10}, "x</think>" + CALL[:60], "length",
                        False),
    "single-call": ("glm52", {"tools": [TOOL], "parallel_tool_calls": False}, "go</think>" + CALL + CALL, "stop",
                    True),
    "incomplete-reasoning": ("glm5_next", {"max_output_tokens": 4}, "still reasoning", "length", False),
    "thinking-off": ("glm52", {"reasoning": {"effort": "none"}}, "Hello </think> there", "stop", False),
}


@pytest.mark.parametrize("case", sorted(REPLY_CASES))
def test_a_response_comes_back_as_python_wrote_it(case, servers):
    """Held to Python with the reply in one chunk; in 3-byte chunks it must come to the same."""
    model, fields, reply, finish, ends_early = REPLY_CASES[case]
    body = _codex(model, **fields)
    port, stub = servers[model]
    whole = {}
    for size in (None, 3):
        for stream in (False, True):
            stub.reply, stub.finish, stub.ends_early, stub.fail_after = _pieces(reply, size), finish, ends_early, None
            answer, outcomes, _ = _rust_answer(port, stub, "/v1/responses", {**body, "stream": stream}, stream)
            if size is None:
                status, data, py_outcomes = _python_side("/v1/responses", {**body, "stream": stream}, stub.reply,
                                                         finish)
                assert (answer, outcomes) == (_answer(status, data, stream), py_outcomes), stream
                whole[stream] = (_folded(answer), outcomes)
            else:
                assert (_folded(answer), outcomes) == whole[stream], (size, stream)


def test_a_response_that_fails_is_failed(servers):
    """Whole: the error's status and OpenAI body. Streamed: ``response.failed``
    after what went out, the open item closed ``incomplete``."""
    body = _codex(tools=[TOOL])
    pieces = _pieces("plan</think>Checking " + CALL, 7)
    for stream in (False, True):
        port, stub = servers["glm52"]
        stub.reply, stub.finish, stub.ends_early, stub.fail_after = pieces, "stop", False, 4
        answer, outcomes, _ = _rust_answer(port, stub, "/v1/responses", {**body, "stream": stream}, stream)
        status, data, py_outcomes = _python_side("/v1/responses", {**body, "stream": stream}, pieces, "stop", 4)
        assert (answer, outcomes) == (_answer(status, data, stream), py_outcomes)


def test_a_body_that_is_not_json_is_a_400(servers):
    """The decoder's own words differ (serde's and Python's json share none)."""
    port, stub = servers["glm52"]
    for raw in (b"{not json", b'{"temperature": Infinity}'):
        (status, body), _, _ = _rust_answer(port, stub, "/v1/responses", raw)
        assert (status, body["error"]["type"], body["error"]["code"]) == (400, "invalid_request_error", 400)
        assert body["error"]["message"].startswith("the request body is not JSON: ")


def test_a_client_that_leaves_mid_response_aborts_it():
    import json

    from test_rust_release import _leave, _until

    stub = _GlmStub()
    stub.reply, stub.ends_early = [b"Plan</think>", b"Hello", b" there"], True
    with _glm_stack("glm52", stub) as (port, _upload):
        _leave(port, "/v1/responses", json.dumps(_codex(stream=True)).encode(), read=b"Hello")
        _until(lambda: stub.outcomes, what="the abort")
    assert list(stub.outcomes.values()) == ["abort"]
