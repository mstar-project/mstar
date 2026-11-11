"""Anthropic's ``/v1/messages`` and ``/v1/messages/count_tokens`` on mstar-server
(``rust/server/src/anthropic/``), for Claude Code on the GLMs.

Each request goes to mstar-server over the real bridge. A request is refused
with the Python route's words, or submitted as the chat the Python route
translated it into (written out from ``serving_messages``) is submitted by
the live ``GlmChatAdapter``: the template's messages and tools in their order.
A reply is held to what the Python route answers (live, over a data plane
handing out the same chunks): the status and body or every event of the
stream, and the outcome the request was stopped with; cut into smaller chunks,
the model's bytes must come to the same reply. Skipped unless
the ``mstar_rust`` extension is installed and the server binary is built."""
import contextlib
import json

import pytest

pytest.importorskip("mstar_rust")

from test_rust_frontend import BINARY
from test_rust_glm_chat import (
    CALL,
    WEATHER,
    _answer,
    _CountingModel,
    _folded,
    _glm_stack,
    _GlmStub,
    _pieces,
    _python_side,
    _rust_answer,
    _translated_alike,
)

from mstar.api_server.openai.adapters import ADAPTER_REGISTRY

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

TOOL = {"name": "get_weather", "description": "Weather by city", "input_schema": WEATHER["function"]["parameters"]}
KEY = "s3cret-key"


def _body(model="glm52", **fields):
    return {"model": model, "max_tokens": 256, "messages": [{"role": "user", "content": "weather in Paris?"}],
            **fields}


def _chat(model="glm52", **fields):
    """The chat ``_body(model)`` asks for, with ``fields``."""
    return {"model": model, "messages": [{"role": "user", "content": "weather in Paris?"}], "max_tokens": 256,
            "stream": False, **fields}


def _counted(chat):
    """The chat a ``count_tokens`` body asks for, which generates nothing."""
    return {k: v for k, v in chat.items() if k not in ("max_tokens", "stream")}


# TOOL, SEARCH and FETCH as chat tools
CHAT_TOOL = {"type": "function", "function": {
    "name": "get_weather", "parameters": WEATHER["function"]["parameters"], "description": "Weather by city"}}


# Claude Code 2.1.309's tool search, as recorded: a deferred tool, the search's call, and its result
FETCH = {"name": "WebFetch", "description": "Fetches a URL.", "defer_loading": True,
         "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}
SEARCH = {"name": "ToolSearch", "description": "Fetches deferred tools' schemas.",
          "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
CHAT_SEARCH = {"type": "function", "function": {
    "name": "ToolSearch", "parameters": SEARCH["input_schema"], "description": "Fetches deferred tools' schemas."}}
CHAT_FETCH = {"type": "function", "function": {
    "name": "WebFetch", "parameters": FETCH["input_schema"], "description": "Fetches a URL.", "defer_loading": True}}
LOADED = [
    {"role": "user", "content": "read https://example.com"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "call_6a3f", "name": "ToolSearch",
                                       "input": {"query": "select:WebFetch", "max_results": 1}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_6a3f",
                                  "content": [{"type": "tool_reference", "tool_name": "WebFetch"}]}]},
]
CONVERSATION = [
    {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "Need it.", "signature": "sig"},
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"zip": "75001", "city": "Paris"}}]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "no such city"}],
         "is_error": True},
        {"type": "text", "text": "and Rome?"}]},
]

# name: (Messages body, the chat it asks for or its refusal)
REQUEST_CASES = {
    "conversation": (_body(
        system=[{"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Use tools."}],
        messages=CONVERSATION, tools=[TOOL], tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        stop_sequences=["END"], temperature=0.2, top_k=20, metadata={"user_id": "u"},
        thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4096),
        _chat(
            messages=[
                {"role": "system", "content": "You are terse.\nUse tools."},
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "reasoning_content": "Need it.",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"zip": "75001", "city": "Paris"}},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_1", "content": "Error: no such city"},
                {"role": "user", "content": "and Rome?"},
            ],
            max_tokens=4096,
            temperature=0.2,
            top_k=20,
            tools=[CHAT_TOOL],
            tool_choice="auto",
            parallel_tool_calls=False,
            enable_thinking=True,
        )),
    "claude-code": (_body(
        "glm5_next",
        system=[{"type": "text", "text": "You are Claude Code."},
                {"type": "text", "text": "Be careful.", "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [{"type": "text", "text": "fix the test"}]},
                  {"role": "system", "content": [{"type": "text", "text": "# Environment\ncwd: /repo",
                                                  "cache_control": {"type": "ephemeral"}}]},
                  {"role": "user", "content": "go on"}],
        tools=[TOOL], thinking={"type": "adaptive", "display": "updates"},
        context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        output_config={"effort": "high"}, metadata={"user_id": "{}"}, max_tokens=32000),
        _chat(
            "glm5_next",
            messages=[
                {"role": "system", "content": "You are Claude Code.\nBe careful."},
                {"role": "user", "content": "fix the test"},
                {"role": "system", "content": "# Environment\ncwd: /repo"},
                {"role": "user", "content": "go on"},
            ],
            max_tokens=32000,
            tools=[CHAT_TOOL],
            enable_thinking=True,
            clear_thinking=False,
            reasoning_effort="high",
        )),
    "tool-search": (_body("glm5_next", messages=LOADED, tools=[SEARCH, FETCH]),
        _chat(
            "glm5_next",
            messages=[
                {"role": "user", "content": "read https://example.com"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_6a3f",
                            "type": "function",
                            "function": {
                                "name": "ToolSearch",
                                "arguments": {"query": "select:WebFetch", "max_results": 1},
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_6a3f",
                    "content": [{"type": "tool_reference", "name": "WebFetch"}],
                },
            ],
            tools=[CHAT_SEARCH, CHAT_FETCH],
        )),
    "tool-result-mixing-references": (_body(messages=[
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "tool_reference", "tool_name": "WebFetch"}, {"type": "text", "text": "and more"}]}]}],
        tools=[SEARCH, FETCH]),
        _chat(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "ToolSearch", "arguments": {}},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "t1", "content": "Tool loaded: WebFetch\nand more"},
            ],
            tools=[CHAT_SEARCH, CHAT_FETCH],
        )),
    "prefix-cache-off": (_body(prefix_cache=False),
        _chat(prefix_cache=False)),
    "effort-xhigh": (_body(output_config={"effort": "xhigh"}),
        _chat(reasoning_effort="max")),
    "effort-low-glm52": (_body(output_config={"effort": "low"}),
        _chat(reasoning_effort="high")),
    "effort-medium-glm53": (_body("glm5_next", output_config={"effort": "medium"}),
        _chat("glm5_next", reasoning_effort="high")),
    "effort-max-glm53": (_body("glm5_next", output_config={"effort": "max"}),
        _chat("glm5_next", reasoning_effort="max")),
    "tool-choice-any": (_body(tools=[TOOL], tool_choice={"type": "any"}),
        _chat(tools=[CHAT_TOOL], tool_choice="required")),
    "tool-choice-tool": (_body("glm5_next", tools=[TOOL], tool_choice={"type": "tool", "name": "get_weather"}),
        _chat("glm5_next", tools=[CHAT_TOOL], tool_choice={"type": "function", "function": {"name": "get_weather"}})),
    "tool-choice-none": (_body(tools=[TOOL], tool_choice={"type": "none"}),
        _chat(tools=[CHAT_TOOL], tool_choice="none")),
    "thinking-disabled": (_body(thinking={"type": "disabled"}),
        _chat(enable_thinking=False)),
    "string-system": (_body(system="Answer in French."),
        _chat(
            messages=[
                {"role": "system", "content": "Answer in French."},
                {"role": "user", "content": "weather in Paris?"},
            ],
        )),
    "empty-system": (_body(system=[]),
        _chat()),
    "stream-zero": (_body(stream=0),
        _chat()),
    "glm53-thinking-disabled": (_body("glm5_next", thinking={"type": "disabled"}),
        _chat("glm5_next", enable_thinking=False)),
    "not-an-object": ([_body()],
        "the request body must be a JSON object"),
    "unknown-field": (_body(service_tier="auto", container="x"),
        "unsupported field(s) ['container', 'service_tier']"),
    "no-max-tokens": ({"model": "glm52", "messages": [{"role": "user", "content": "hi"}]},
        "max_tokens is required, a positive integer; got None"),
    "max-tokens-zero": (_body(max_tokens=0),
        "max_tokens is required, a positive integer; got 0"),
    "max-tokens-bool": (_body(max_tokens=True),
        "max_tokens is required, a positive integer; got True"),
    "max-tokens-float": (_body(max_tokens=5.0),
        "max_tokens is required, a positive integer; got 5.0"),
    "empty-messages": (_body(messages=[]),
        "messages must be a non-empty list"),
    "role-tool": (_body(messages=[{"role": "tool", "content": "x"}]),
        "a message's role must be user, assistant or system, not 'tool'"),
    "image-block": (_body(messages=[{"role": "user", "content": [{"type": "image", "source": {}}]}]),
        "a user message takes text and tool_result blocks; got 'image' (the model reads text)"),
    "document-in-system": (_body(system=[{"type": "document", "source": {}}]),
        "system takes text blocks only; got a 'document' block (the model reads text)"),
    "system-not-text": (_body(system=5),
        "system must be a string or a list of content blocks"),
    "block-without-type": (_body(system=["plain"]),
        "system takes text blocks only; got a 'str' block"),
    "redacted-thinking": (_body(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": [
        {"type": "redacted_thinking", "data": "x"}]}, {"role": "user", "content": "go"}]),
        "an assistant message takes text, thinking and tool_use blocks; got 'redacted_thinking'"),
    "prefill": (_body(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure,"}]),
        "a final assistant message (prefill) is not supported; end with a user turn"),
    "content-a-number": (_body(messages=[{"role": "user", "content": 5}]),
        "a message's content must be a string or a list of blocks; got 5"),
    "server-tool": (_body(tools=[{"type": "web_search_20250305", "name": "web_search"}]),
        "tools take client tools with an input_schema; 'web_search_20250305' is a server tool"),
    "tool-no-schema": (_body(tools=[{"name": "f"}]),
        "a tool needs a name and an input_schema object; got {'name': 'f'}"),
    "tools-not-a-list": (_body(tools={"name": "f"}),
        "tools must be a list"),
    "tool-choice-bad": (_body(tools=[TOOL], tool_choice={"type": "required"}),
        "tool_choice takes type auto, any, tool or none; got {'type': 'required'}"),
    "tool-choice-tool-no-name": (_body(tools=[TOOL], tool_choice={"type": "tool"}),
        "tool_choice of type tool needs a name"),
    "budget-low": (_body(thinking={"type": "enabled", "budget_tokens": 100}),
        "thinking.budget_tokens must be an integer of at least 1024; got 100"),
    "budget-over-max": (_body(thinking={"type": "enabled", "budget_tokens": 4096}, max_tokens=2048),
        "thinking.budget_tokens (4096) must be less than max_tokens (2048)"),
    "thinking-bad-type": (_body(thinking={"type": "on"}),
        "thinking takes type enabled, adaptive or disabled; got {'type': 'on'}"),
    "context-clear-tool-uses": (_body(context_management={"edits": [{"type": "clear_tool_uses_20250919"}]}),
        "context_management edit {'type': 'clear_tool_uses_20250919'} is not served; only clear_thinking "
        "with keep 'all' is"),
    "context-not-a-list": (_body(context_management={"edits": "all"}),
        "context_management takes a list of edits; got {'edits': 'all'}"),
    "context-extra-key": (_body(context_management={"edits": [], "x": 1}),
        "context_management takes a list of edits; got {'edits': [], 'x': 1}"),
    "output-format": (_body(output_config={"format": {"type": "json_schema"}}),
        "output_config.format (structured output) needs constrained decoding, which is not served"),
    "output-extra-key": (_body(output_config={"effort": "high", "verbosity": "low", "a": 1}),
        "output_config takes effort only; got ['a', 'effort', 'verbosity']"),
    "effort-bad": (_body(output_config={"effort": "extreme"}),
        "output_config.effort takes low, medium, high, xhigh or max; got 'extreme'"),
    "stops-not-a-list": (_body(stop_sequences="END"),
        "stop_sequences must be a list of non-empty strings; got 'END'"),
    "stops-empty-string": (_body(stop_sequences=["END", ""]),
        "stop_sequences must be a list of non-empty strings; got ['END', '']"),
    "stops-17": (_body(stop_sequences=[f"s{i}" for i in range(17)]),
        "stop_sequences takes at most 16 strings of at most 256 characters"),
    "stop-too-long": (_body(stop_sequences=["x" * 257]),
        "stop_sequences takes at most 16 strings of at most 256 characters"),
    "prefix-cache-not-a-bool": (_body(prefix_cache="no"),
        "prefix_cache must be a boolean; got 'no'"),
    "tool-result-no-id": (_body(messages=[{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]),
        "a tool_result block needs its tool_use_id"),
    "tool-use-input-a-string": (_body(messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f", "input": "{}"}]},
        {"role": "user", "content": "go"}]),
        "a tool_use block's input must be an object; got '{}'"),
    "tool-reference-no-name": (_body(messages=[{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "tool_reference"}]}]}]),
        "a tool_reference block needs its tool_name; got {'type': 'tool_reference'}"),
    "lora-unknown": (_body(model="no-such-adapter"),
        _chat("no-such-adapter")),
}


@pytest.fixture(scope="module")
def servers():
    """One mstar-server per GLM, for the whole module."""
    with contextlib.ExitStack() as stack:
        out = {}
        for model in ("glm52", "glm5_next"):
            stub = _GlmStub()
            stub.model = _CountingModel()
            port, _upload = stack.enter_context(_glm_stack(model, stub))
            out[model] = (port, stub)
        yield out


def _model(body):
    return body.get("model") if isinstance(body, dict) and body.get("model") in ("glm52", "glm5_next") else "glm52"


@pytest.mark.parametrize("case", sorted(REQUEST_CASES))
def test_a_message_request_is_translated_or_refused_as_python_did(case, servers, tmp_path):
    pytest.importorskip("pydantic")
    body, want = REQUEST_CASES[case]
    port, stub = servers[_model(body)]
    stub.reply, stub.finish, stub.ends_early, stub.fail_after = [b"ok"], "stop", False, None
    answer, _, submitted = _rust_answer(port, stub, "/v1/messages", body)
    if answer[0] != 200:
        assert (answer[1]["type"], answer[1]["error"]["type"]) == ("error", "invalid_request_error")
    _translated_alike(answer, submitted, want, _model(body), tmp_path)


# name: (count_tokens body, the chat it asks for or its refusal)
COUNT_CASES = {
    "conversation": ({k: v for k, v in REQUEST_CASES["conversation"][0].items()
                     if k not in ("max_tokens", "stop_sequences", "temperature", "top_k", "metadata")},
        _counted(_chat(
            messages=[
                {"role": "system", "content": "You are terse.\nUse tools."},
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "reasoning_content": "Need it.",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"zip": "75001", "city": "Paris"}},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_1", "content": "Error: no such city"},
                {"role": "user", "content": "and Rome?"},
            ],
            tools=[CHAT_TOOL],
            tool_choice="auto",
            parallel_tool_calls=False,
            enable_thinking=True,
        ))),
    "tools-and-effort": ({"model": "glm5_next", "messages": [{"role": "user", "content": "hi"}], "tools": [TOOL],
                         "output_config": {"effort": "low"}},
        _counted(_chat(
            "glm5_next",
            messages=[{"role": "user", "content": "hi"}],
            tools=[CHAT_TOOL],
            reasoning_effort="low",
        ))),
    "max-tokens-is-unknown": (_body(),
        "unsupported field(s) ['max_tokens']"),
    "glm53-thinking-disabled": ({"model": "glm5_next", "messages": [{"role": "user", "content": "hi"}],
                                "thinking": {"type": "disabled"}},
        _counted(_chat("glm5_next", messages=[{"role": "user", "content": "hi"}], enable_thinking=False))),
    "not-an-object": ("hi",
        "the request body must be a JSON object"),
}

@pytest.mark.parametrize("case", sorted(COUNT_CASES))
def test_count_tokens_counts_the_prompt_python_rendered(case, servers, tmp_path):
    """Asks the model's ``process_prompt`` (here one that counts its arguments)
    for the prompt the chat renders, so the count matches only for the same
    rendered request."""
    pytest.importorskip("pydantic")
    from test_rust_openai_adapters import _python_request

    body, want = COUNT_CASES[case]
    model = _model(body)
    port, stub = servers[model]
    (status, answer), _, _ = _rust_answer(port, stub, "/v1/messages/count_tokens", body)
    if isinstance(want, dict):
        try:
            args = ADAPTER_REGISTRY[model].chat_to_request(_python_request(want), tmp_path)
        except ValueError as e:
            want = str(e)
        else:
            rendered = _CountingModel().process_prompt(args.text, args.input_modalities, args.output_modalities,
                                                       prompt_parts=args.prompt_parts, **args.model_kwargs)
            assert (status, answer) == (200, {"input_tokens": rendered["text_inputs"][0].numel()})
            return
    assert (status, answer["error"]["message"]) == (400, want)


# name: (model, request fields, the model's reply, its finish reason, whether the frontend ends it)
REPLY_CASES = {
    "reasoning-and-text": ("glm52", {}, "Let me think.</think>\n\nHello, Paris.", "stop", False),
    "tool-use": ("glm52", {"tools": [TOOL]}, "Need the weather.</think>" + CALL, "stop", False),
    "two-tool-uses": ("glm5_next", {"tools": [TOOL]}, "Plan</think>" + CALL + CALL, "stop", False),
    "stop-sequence": ("glm52", {"stop_sequences": ["END"], "thinking": {"type": "disabled"}},
                      "plain answer END and the rest", "stop", True),
    "stop-not-in-thinking": ("glm5_next", {"stop_sequences": ["END"]}, "an END here</think>no stop", "stop",
                             False),
    "cut-call": ("glm52", {"tools": [TOOL], "max_tokens": 10}, "x</think>" + CALL[:60], "length", False),
    "single-call": ("glm52", {"tools": [TOOL], "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}},
                    "go</think>" + CALL + CALL, "stop", True),
    "forced-tool": ("glm5_next", {"tools": [TOOL], "tool_choice": {"type": "tool", "name": "get_weather"}},
                    "<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>", "stop", False),
    "max-tokens": ("glm5_next", {"max_tokens": 4}, "still reasoning", "length", False),
}


@pytest.mark.parametrize("case", sorted(REPLY_CASES))
def test_a_message_reply_comes_back_as_python_wrote_it(case, servers):
    """Held to Python with the reply in one chunk; in 3-byte chunks it must come to the same."""
    model, fields, reply, finish, ends_early = REPLY_CASES[case]
    body = _body(model, **fields)
    port, stub = servers[model]
    whole = {}
    for size in (None, 3):
        for stream in (False, True):
            stub.reply, stub.finish, stub.ends_early, stub.fail_after = _pieces(reply, size), finish, ends_early, None
            answer, outcomes, _ = _rust_answer(port, stub, "/v1/messages", {**body, "stream": stream}, stream)
            if size is None:
                status, data, py_outcomes = _python_side("/v1/messages", {**body, "stream": stream}, stub.reply,
                                                         finish)
                assert (answer, outcomes) == (_answer(status, data, stream), py_outcomes), stream
                whole[stream] = (_folded(answer), outcomes)
            else:
                assert (_folded(answer), outcomes) == whole[stream], (size, stream)


def test_a_reply_that_fails_is_an_anthropic_error(servers):
    """Whole: the error's status and Anthropic body. Streamed: an in-band
    ``error`` event after what went out, and no ``message_stop``."""
    body = _body(tools=[TOOL])
    pieces = _pieces("plan</think>Checking " + CALL, 7)
    for stream in (False, True):
        port, stub = servers["glm52"]
        stub.reply, stub.finish, stub.ends_early, stub.fail_after = pieces, "stop", False, 4
        answer, outcomes, _ = _rust_answer(port, stub, "/v1/messages", {**body, "stream": stream}, stream)
        status, data, py_outcomes = _python_side("/v1/messages", {**body, "stream": stream}, pieces, "stop", 4)
        assert (answer, outcomes) == (_answer(status, data, stream), py_outcomes)


# the routes the key guards, how a client sends it, and the status (as the Python middleware answered)
AUTH_CASES = {
    "x-api-key": ("/v1/messages", {"x-api-key": KEY}, 200),
    "x-api-key-spaced": ("/v1/messages", {"x-api-key": f"  {KEY} "}, 200),
    "bearer": ("/v1/messages", {"Authorization": f"Bearer {KEY}"}, 200),
    "wrong-bearer-right-x-api-key": ("/v1/messages", {"Authorization": "Bearer no", "x-api-key": KEY}, 200),
    "wrong-x-api-key": ("/v1/messages", {"x-api-key": "no"}, 401),
    "no-key": ("/v1/messages", {}, 401),
    "count-tokens-x-api-key": ("/v1/messages/count_tokens", {"x-api-key": KEY}, 200),
    "count-tokens-no-key": ("/v1/messages/count_tokens", {}, 401),
    "chat-x-api-key": ("/v1/chat/completions", {"x-api-key": KEY}, 401),
}


@pytest.mark.parametrize("case", sorted(AUTH_CASES))
def test_the_api_key_is_taken_as_x_api_key_on_anthropics_routes_only(case):
    """A refusal on an Anthropic route is in Anthropic's shape; elsewhere
    ``x-api-key`` is no key."""
    path, headers, want = AUTH_CASES[case]
    body = {"model": "glm52", "messages": [{"role": "user", "content": "hi"}]}
    if path == "/v1/messages":
        body["max_tokens"] = 8
    stub = _GlmStub()
    stub.model = _CountingModel()
    with _glm_stack("glm52", stub, MSTAR_API_KEY=KEY) as (port, _upload):
        (status, answer), _, submitted = _rust_answer(port, stub, path, body, headers=headers)
    assert status == want
    assert submitted is None or status == 200, "a refused request reaches the backend"
    if status == 401 and path.startswith("/v1/messages"):
        message = "missing or invalid API key; send 'x-api-key: <key>' or 'Authorization: Bearer <key>'"
        assert answer == {"type": "error", "error": {"type": "authentication_error", "message": message}}
    elif status == 401:
        assert answer["error"]["message"] == "missing or invalid API key; send 'Authorization: Bearer <key>'"


def test_a_body_over_the_limit_is_an_anthropic_413():
    """The body limit answers Anthropic's routes in Anthropic's shape (the
    Python route answered this one in OpenAI's), and comes before the model is
    asked whether it serves the route."""
    stub = _GlmStub()
    big = json.dumps(_body(system="x" * (2 << 20))).encode()
    with _glm_stack("glm52", stub, MSTAR_MAX_BODY_MB="1") as (port, _upload):
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            (status, body), _, submitted = _rust_answer(port, stub, path, big)
            assert (status, body["type"], body["error"]["type"], submitted) == (
                413, "error", "request_too_large", None)
            assert "MSTAR_MAX_BODY_MB" in body["error"]["message"]


def test_a_body_that_is_not_json_is_an_anthropic_400(servers):
    """The decoder's own words differ (serde's and Python's json share none)."""
    port, stub = servers["glm52"]
    for raw in (b"{not json", b'{"max_tokens": NaN}', b'{"x": 1e400}'):
        (status, body), _, _ = _rust_answer(port, stub, "/v1/messages", raw)
        assert (status, body["type"], body["error"]["type"]) == (400, "error", "invalid_request_error")
        assert body["error"]["message"].startswith("the request body is not JSON: ")


def test_a_model_without_messages_gets_anthropics_404():
    from test_rust_frontend import _model_stack, _StubAPIServer

    with _model_stack("qwen3_5_0.8b", _StubAPIServer()) as (port, _upload):
        (status, body), _, _ = _rust_answer(port, _GlmStub(), "/v1/messages", _body())
    assert (status, body) == (404, {"type": "error", "error": {
        "type": "not_found_error", "message": "Model 'qwen3_5_0.8b' does not support this endpoint"}})


def test_a_client_that_leaves_mid_message_aborts_it():
    from test_rust_release import _leave, _until

    stub = _GlmStub()
    stub.reply, stub.ends_early = [b"Plan</think>", b"Hello", b" there"], True
    with _glm_stack("glm52", stub) as (port, _upload):
        _leave(port, "/v1/messages", json.dumps(_body(stream=True)).encode(), read=b"Hello")
        _until(lambda: stub.outcomes, what="the abort")
    assert list(stub.outcomes.values()) == ["abort"]
