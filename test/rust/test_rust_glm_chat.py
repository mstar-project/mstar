"""GLM-5.2 / GLM-5.3 chat on mstar-server (``GlmChat``, rust/server/src/adapters/glm.rs).

A chat reaches the backend as the Python ``GlmChatAdapter`` builds it, checked
live against the adapter: the messages and tools in their order, the template
switches, a forced call's prefix, and each refusal word for word. The reply
comes back split into reasoning, content and tool calls as the Python chat
route splits it (live, over a data plane handing out the same chunks), whole
and streamed, stop strings and ``parallel_tool_calls: false`` included; cut into
smaller chunks, the model's bytes must come to the same reply. Skipped unless
the ``mstar_rust`` extension is installed and the server binary is built."""
import json
from dataclasses import asdict

import pytest

pytest.importorskip("mstar_rust")

from test_rust_frontend import BINARY, _Chunk, _model_stack, _StubAPIServer
from test_rust_openai_adapters import PNG, _normalized, _post, _python_request

from mstar.api_server.entrypoint import APIServer
from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
from mstar.api_server.rust_frontend import CHAT_TEMPLATE_ENV, chat_template_facts
from mstar.api_server.served_models import ServedModels

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather by city", "parameters": {
    "type": "object",
    "properties": {"zip": {"type": "string"}, "city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city"],
}}}
TIME = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
CALL = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value><arg_key>zip</arg_key>"
        "<arg_value>75001</arg_value></tool_call>")
ARGS = "<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"


def _glm_stack(model, stub, **env):
    """mstar-server for ``model`` with the chat template facts its launcher passes."""
    facts = json.dumps(chat_template_facts(ADAPTER_REGISTRY[model]))
    return _model_stack(model, stub, env={CHAT_TEMPLATE_ENV: facts, **env})


def _chat(model, **fields):
    return {"model": model, "messages": [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather in Paris?"},
    ], **fields}


# ── a chat reaches the backend as GlmChatAdapter builds it ──────────────────

AGENT_TURN = [
    {"role": "developer", "content": "French."},
    {"role": "user", "content": "weather in Paris?", "name": "kirill"},
    {"role": "assistant", "content": None, "reasoning_content": "Look it up.", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"zeta": 1, "city": "Paris", "alpha": [2, {"b": 1, "a": 0}]}'}},
        {"type": "function", "function": {"name": "get_time", "arguments": {"tz": "CET", "at": "now"}}},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "12:00"}]},
]

GLM_CASES = {
    "plain": _chat("glm52", temperature=0.3, top_p="0.9", max_tokens=16, seed=2, top_k=5),
    "tools-and-switches": _chat("glm52", tools=[WEATHER, TIME], tool_choice="auto", enable_thinking=False,
                                chat_template_kwargs={"reasoning_effort": "high", "thinking": False,
                                                      "tokenize": False}),
    "nested-switches-win-over-flat": _chat("glm5_next", reasoning_effort="low",
                                           chat_template_kwargs={"reasoning_effort": "max", "clear_thinking": False}),
    "agent-turn": {"model": "glm5_next", "messages": AGENT_TURN, "tools": [WEATHER, TIME]},
    "agent-turn-glm52": {"model": "glm52", "messages": AGENT_TURN, "tools": [TIME, WEATHER],
                         "parallel_tool_calls": True},
    "empty-arguments": {"model": "glm52", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_time", "arguments": "  "}}]},
        {"role": "tool", "content": "noon"}]},
    "forced-one": _chat("glm52", tools=[WEATHER], tool_choice="required"),
    "forced-of-two": _chat("glm52", tools=[WEATHER, TIME], tool_choice="required"),
    "forced-named": _chat("glm52", tools=[WEATHER, TIME],
                          tool_choice={"type": "function", "function": {"name": "get_weather"}}),
    "glm53-forced-of-two": _chat("glm5_next", tools=[WEATHER, TIME], tool_choice="required"),
    "glm53-forced-named": _chat("glm5_next", tools=[WEATHER, TIME],
                                tool_choice={"type": "function", "function": {"name": "get_weather"}}),
    "tool-choice-none": _chat("glm52", tools=[WEATHER], tool_choice="none"),
    "parallel-off": _chat("glm52", tools=[WEATHER], parallel_tool_calls=False),
    "stop-strings": _chat("glm52", stop=["END", "\n\n"]),
    "stop-string": _chat("glm52", stop="END"),
    "falsy-switch-nested": _chat("glm52", chat_template_kwargs={"enable_thinking": 0}),
    "null-switch": _chat("glm52", enable_thinking=None),
    "glm53-thinking-on": _chat("glm5_next", enable_thinking=True),
    "envelope-and-off-values": _chat("glm52", stream_options={"include_usage": True}, user="u1", store=False,
                                     logprobs=False, response_format={"type": "text"}, tools=[], functions=[],
                                     prefix_cache=False),
    # refusals, in the adapter's order
    "glm53-thinking-off": _chat("glm5_next", chat_template_kwargs={"enable_thinking": False}),
    "glm53-thinking-off-flat": _chat("glm5_next", enable_thinking=False),
    "flat-switch-not-a-bool": _chat("glm52", enable_thinking=0),
    "required-without-tools": _chat("glm52", tool_choice="required"),
    "tool-choice-any": _chat("glm52", tools=[WEATHER], tool_choice="any"),
    "named-unknown": _chat("glm52", tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "nope"}}),
    "named-flat": _chat("glm52", tools=[WEATHER], tool_choice={"type": "function", "name": "get_weather"}),
    "forced-with-thinking": _chat("glm52", tools=[WEATHER], tool_choice="required", enable_thinking=True),
    "reply-prefix": _chat("glm52", reply_prefix="<tool_call>get_weather"),
    "parallel-not-a-bool": _chat("glm52", tools=[WEATHER], parallel_tool_calls="no"),
    "parallel-zero": _chat("glm52", tools=[WEATHER], parallel_tool_calls=0),
    "tool-without-function": _chat("glm52", tools=[{"type": "function"}]),
    "tools-a-string": _chat("glm52", tools="get_weather"),
    "tools-a-dict": _chat("glm52", tools={"name": "get_weather"}),
    "parameters-a-string": _chat("glm52", tools=[{"type": "function", "function": {"name": "f", "parameters": "{}"}}]),
    "properties-a-list": _chat("glm52", tools=[
        {"type": "function", "function": {"name": "f", "parameters": {"properties": ["a"]}}}]),
    "template-kwargs-a-string": _chat("glm52", chat_template_kwargs="thinking"),
    "five-stops": _chat("glm52", stop=["a", "b", "c", "d", "e"]),
    "empty-stop": _chat("glm52", stop=["END", ""]),
    "long-stop": _chat("glm52", stop="é" * 257),
    "legacy-functions": _chat("glm52", functions=[WEATHER["function"]]),
    "legacy-function-call": _chat("glm52", tools=[WEATHER], function_call={"name": "get_weather"}),
    "legacy-off-values": _chat("glm52", functions=[], function_call="auto"),
    "arguments-not-json": {"model": "glm52", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}]}]},
    "arguments-a-list": {"model": "glm52", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}]},
    "call-without-name": {"model": "glm52", "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "tool_calls": [{"function": {"arguments": "{}"}}]}]},
    "call-without-function": {"model": "glm52", "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "tool_calls": [{"id": "x"}]}]},
    "role-function": {"model": "glm52", "messages": [{"role": "function", "content": "x"},
                                                      {"role": "user", "content": "hi"}]},
    "image": {"model": "glm5_next", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}]}]},
    "logprobs": _chat("glm52", logprobs=True),
    "n": _chat("glm52", n=2),
}


@pytest.fixture(scope="module")
def glm_servers():
    """One mstar-server per GLM over a recording stub, for the whole module."""
    import contextlib

    with contextlib.ExitStack() as stack:
        servers = {}
        for model in ("glm52", "glm5_next"):
            stub = _GlmStub()
            port, upload = stack.enter_context(_glm_stack(model, stub))
            servers[model] = (port, upload, stub)
        yield servers


@pytest.mark.parametrize("case", sorted(GLM_CASES))
def test_a_glm_chat_reaches_the_backend_as_the_python_adapter_builds_it(case, glm_servers, tmp_path):
    """Refused alike, word for word, or submitted alike: the template's messages
    and tools also in the same key order, which the template renders."""
    pytest.importorskip("pydantic")
    body = GLM_CASES[case]
    try:
        want, refusal = ADAPTER_REGISTRY[body["model"]].chat_to_request(_python_request(body), tmp_path), None
    except ValueError as e:
        want, refusal = None, str(e)
    port, _upload, stub = glm_servers[body["model"]]
    before = len(stub.submitted)
    status, data = _post(port, "/v1/chat/completions", body, timeout=30)
    if refusal is not None:
        assert (status, json.loads(data)["error"]["message"]) == (400, refusal)
        assert len(stub.submitted) == before
        return
    assert status == 200, data
    sub = stub.submitted[-1]
    assert (sub["text"], sub["in"], sub["out"]) == (want.text, want.input_modalities, want.output_modalities)
    assert [asdict(p) for p in sub["parts"]] == [asdict(p) for p in want.prompt_parts]
    assert sub["mk"] == want.model_kwargs
    for key in ("messages", "tools", "chat_template_kwargs", "reply_prefix"):
        assert json.dumps(sub["mk"].get(key)) == json.dumps(want.model_kwargs.get(key)), key
    assert not sub["files"]


# ── the reply comes back as the Python chat route split it ──────────────────

class _GlmStub(_StubAPIServer):
    """The data plane answering each submission with ``reply``'s bytes, a
    token a chunk with the prompt's 11, then the ``usage`` chunk; or an
    ``error`` chunk in place of the rest when ``fail_after`` chunks are out.
    A reply the frontend should end itself (``ends_early``) waits for its
    abort instead of a usage chunk, so the abort finds the submission live."""

    def __init__(self):
        super().__init__()
        self.reply, self.finish, self.fail_after, self.ends_early = [b"ok"], "stop", None, False
        self.outcomes = {}
        self.finished = set()

    async def iter_result_chunks(self, request_id):
        try:
            async for chunk in self._chunks(request_id):
                yield chunk
        finally:
            self.finished.add(request_id)

    async def _chunks(self, request_id):
        import asyncio

        for i, data in enumerate(self.reply):
            if i == self.fail_after:
                yield _Chunk("error", b"engine failed", {"status": 500})
                return
            yield _Chunk("text", data, num_tokens=1, prompt_tokens=11)
        if self.ends_early:
            for _ in range(200):
                if request_id in self.outcomes:
                    return
                await asyncio.sleep(0.05)
            return
        yield _Chunk("usage", b"", {"prompt_tokens": 11, "completion_tokens": len(self.reply),
                                    "finish_reason": self.finish})

    def abort_submission(self, key, outcome="abort"):
        if key is not None:
            self.outcomes.setdefault(key, outcome)
        super().abort_submission(key)


TIME_CALL = "<tool_call>get_time</tool_call>"
UNOFFERED = "ok</think><tool_call>rm_rf<arg_key>path</arg_key><arg_value>/</arg_value></tool_call>"
# name: (model, request fields, the model's reply, its finish reason, whether the frontend ends it)
REPLY_CASES = {
    "reasoning": ("glm52", {}, "Two plus two.</think>\n\nIt is 4.", "stop", False),
    "glm53-reasoning": ("glm5_next", {}, "hmm, <b>bold</b>?</think>Hi there", "stop", False),
    "thinking-off": ("glm52", {"enable_thinking": False}, "Hello </think> there", "stop", False),
    "tool-call": ("glm52", {"tools": [WEATHER]}, "I need the weather.</think>\n" + CALL, "stop", False),
    "two-calls": ("glm5_next", {"tools": [WEATHER, TIME]}, "Plan</think>\n" + CALL + "\n" + TIME_CALL, "stop", False),
    "forced-one": ("glm52", {"tools": [WEATHER], "tool_choice": "required"}, ARGS, "stop", False),
    "forced-named": ("glm5_next", {"tools": [WEATHER, TIME], "tool_choice": {
        "type": "function", "function": {"name": "get_weather"}}}, ARGS, "stop", False),
    "forced-of-two": ("glm52", {"tools": [WEATHER, TIME], "tool_choice": "required"}, "get_weather" + ARGS, "stop",
                      False),
    "forced-made-up": ("glm52", {"tools": [WEATHER, TIME], "tool_choice": "required"},
                       "rm_rf<arg_key>path</arg_key><arg_value>/</arg_value></tool_call>", "stop", False),
    "unoffered-call": ("glm52", {"tools": [WEATHER]}, UNOFFERED, "stop", False),
    "cut-in-call": ("glm52", {"tools": [WEATHER], "max_tokens": 12}, "ok</think>" + CALL[:70], "length", False),
    "single-call": ("glm52", {"tools": [WEATHER], "parallel_tool_calls": False},
                    "Plan</think>" + CALL + "\nMore" + CALL, "stop", True),
    "stop-string": ("glm52", {"stop": ["END"], "enable_thinking": False}, "abcENDdef and more", "stop", True),
    "stop-not-in-reasoning": ("glm5_next", {"stop": ["END", "\n\n"]},
                              "the END of it</think>An answer, then END here", "stop", True),
    "stop-cuts-off-a-call": ("glm52", {"tools": [WEATHER], "stop": ["END"], "enable_thinking": False},
                             "doneEND" + CALL, "stop", True),
    "stop-never-matches": ("glm52", {"stop": ["\n\n"], "enable_thinking": False}, "one line, held \n", "length",
                           False),
    "length": ("glm5_next", {"max_tokens": 5}, "cut off mid reasoning", "length", False),
    "null-switch": ("glm52", {"enable_thinking": None}, "Two plus two.</think>\n\nIt is 4.", "stop", False),
    "include-usage": ("glm52", {"stream_options": {"include_usage": True}}, "r</think>Hi", "stop", False),
    "split-characters": ("glm52", {"enable_thinking": False}, "中文 😀 é", "stop", False),
}
# how the reply's bytes are cut into chunks; the first is compared with Python's
CHUNKINGS = {"whole": None, "bytes-1": 1, "bytes-5": 5}


def _pieces(text: str, size) -> list[bytes]:
    data = text.encode()
    return [data] if size is None else [data[i:i + size] for i in range(0, len(data), size)]


def _rust_reply(port, stub, body, stream):
    """The answer, and the outcomes the bridge stopped the request with once the
    data plane's side of it ended (an abort can trail the response)."""
    import time

    stub.outcomes = {}
    status, data = _post(port, "/v1/chat/completions", {**body, "stream": stream}, timeout=30)
    rid = stub.submitted[-1]["rid"]
    deadline = time.time() + 15
    while rid not in stub.finished and time.time() < deadline:
        time.sleep(0.01)
    return _normalized(status, data, stream), sorted(stub.outcomes.values())


def _folded(answer):
    """What a client makes of an answer in any chunking: a stream joined into
    its reply, and no usage, whose token counts follow the chunks."""
    status, body = answer
    if status != 200:
        return answer
    if isinstance(body, dict):
        return status, {k: v for k, v in body.items() if k != "usage"}
    reply = {"reasoning_content": "", "content": "", "tool_calls": [], "finish_reason": None, "tail": []}
    for event in body:
        if event == "[DONE]" or "error" in event:
            reply["tail"].append(event)
            continue
        for choice in event["choices"]:
            delta = choice["delta"]
            for key in ("reasoning_content", "content"):
                reply[key] += delta.get(key) or ""
            for call in delta.get("tool_calls", []):
                if "id" in call:
                    reply["tool_calls"].append({"name": call["function"]["name"], "arguments": ""})
                reply["tool_calls"][call["index"]]["arguments"] += call["function"]["arguments"]
            reply["finish_reason"] = choice["finish_reason"] or reply["finish_reason"]
    return status, reply


@pytest.mark.parametrize("case", sorted(REPLY_CASES))
def test_both_frontends_split_a_glm_reply_alike(case):
    model, fields, reply, finish, ends_early = REPLY_CASES[case]
    body = _chat(model, **fields)
    stub = _GlmStub()
    stub.finish, stub.ends_early = finish, ends_early
    with _glm_stack(model, stub) as (port, _upload):
        whole = {}
        for chunking, size in CHUNKINGS.items():
            stub.reply = _pieces(reply, size)
            for stream in (False, True):
                answer, outcomes = _rust_reply(port, stub, body, stream)
                if size is None:
                    status, data, py_outcomes = _python_side("/v1/chat/completions", {**body, "stream": stream},
                                                             stub.reply, finish)
                    assert (answer, outcomes) == (_normalized(status, data, stream), py_outcomes), stream
                    whole[stream] = (_folded(answer), outcomes)
                else:
                    assert (_folded(answer), outcomes) == whole[stream], (chunking, stream)


def test_a_reply_that_fails_mid_stream_is_an_in_band_error():
    """As on the Python route, but for its ``[DONE]`` after the error, which
    mstar-server's chat streams leave out (a failed request must not look
    complete)."""
    body = _chat("glm52", tools=[WEATHER])
    stub = _GlmStub()
    stub.reply, stub.fail_after = _pieces("plan</think>Checking " + CALL, 7), 4
    with _glm_stack("glm52", stub) as (port, _upload):
        for stream in (False, True):
            status, data, py_outcomes = _python_side("/v1/chat/completions", {**body, "stream": stream},
                                                     stub.reply, "stop", stub.fail_after)
            want = (_normalized(status, data, stream), py_outcomes)
            (status, events), outcomes = _rust_reply(port, stub, body, stream)
            if stream:
                assert want[0][1][-1] == "[DONE]" and want[0][1][-2]["error"]["code"] == 500
                want = ((want[0][0], want[0][1][:-1]), want[1])
            assert ((status, events), outcomes) == want


def test_a_client_that_leaves_mid_reply_aborts_it():
    """Leaving mid-stream stops the request in the engine as an abort, not as
    a reply the client had whole."""
    from test_rust_release import _leave, _until

    stub = _GlmStub()
    stub.reply, stub.ends_early = [b"Plan</think>", b"Hello", b" there"], True
    with _glm_stack("glm52", stub) as (port, _upload):
        _leave(port, "/v1/chat/completions", json.dumps(_chat("glm52", stream=True)).encode(), read=b"Hello")
        _until(lambda: stub.outcomes, what="the abort")
    assert list(stub.outcomes.values()) == ["abort"]


# ── the Python side ──────────────────────────────────────────────────────────

def _python_side(path, body, reply, finish, fail_after=None):
    """What the Python route answers ``body`` with, over a data plane that hands
    out ``reply``'s chunks one per batch as ``_GlmStub`` does: the status, the
    raw body, and the outcomes it stopped the request with."""
    import tempfile

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mstar.api_server import entrypoint

    model = body["model"] if body.get("model") in ADAPTER_REGISTRY else "glm52"

    class _PythonGlm:
        open_result_stream = APIServer.open_result_stream
        lora_kwargs = APIServer.lora_kwargs

        def __init__(self, upload_dir):
            self.model_name, self.upload_dir, self.model = model, upload_dir, None
            self.served_models = ServedModels.of(model, None)
            self.outcomes = {}

        def submit_request(self, **kw):
            return kw["request_id"]

        def submission_key(self, request_id):
            return request_id

        def abort_submission(self, key, outcome="abort"):
            if key is not None:
                self.outcomes.setdefault(key, outcome)

        def release_request(self, request_id):
            pass

        async def iter_result_chunk_batches(self, request_id):
            finished = False
            try:
                for i, data in enumerate(reply):
                    if i == fail_after:
                        yield [_Chunk("error", b"engine failed", {"status": 500})]
                        finished = True
                        return
                    yield [_Chunk("text", data, num_tokens=1, prompt_tokens=11)]
                yield [_Chunk("usage", b"", {"prompt_tokens": 11, "completion_tokens": len(reply),
                                             "finish_reason": finish})]
                finished = True
            finally:
                if not finished:  # the route stopped reading: the engine's request is aborted
                    self.abort_submission(request_id)

    saved = entrypoint.api_server
    with tempfile.TemporaryDirectory() as upload_dir:
        entrypoint.api_server = server = _PythonGlm(upload_dir)
        try:
            r = TestClient(entrypoint.app).post(path, json=body)
        finally:
            entrypoint.api_server = saved
    return r.status_code, r.content, sorted(server.outcomes.values())
