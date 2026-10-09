"""GLM chat: messages and tools reach the chat template, and the reply comes
back split into reasoning, content and tool calls, streamed or whole."""

import json

import pytest

pytest.importorskip("pydantic")

from mstar.api_server.openai import adapters, glm_chat  # noqa: E402
from mstar.api_server.openai.protocol import ChatCompletionRequest  # noqa: E402

WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "zip": {"type": "string"}},
}}}
TOOLS = glm_chat.tool_parameters([WEATHER])
CALL = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value><arg_key>zip</arg_key>"
        "<arg_value>75001</arg_value></tool_call>")


def _parser(thinking=True, tools=TOOLS):
    return glm_chat.GlmReplyParser(thinking=thinking, tools=tools)


def _streamed(text, pieces, **kw):
    """The deltas of ``text`` fed in ``pieces``-character chunks, merged back."""
    p = _parser(**kw)
    deltas = []
    for i in range(0, len(text), pieces):
        deltas += p.feed(text[i:i + pieces])
    deltas += p.finish()
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    calls = [c for d in deltas for c in d.get("tool_calls", [])]
    return reasoning, content, calls, p.finish_reason


def test_reasoning_then_answer():
    msg = _parser().message("Two plus two.</think>\n\nIt is 4.")
    assert msg == {"role": "assistant", "content": "It is 4.", "reasoning_content": "Two plus two."}


def test_tool_call_types_arguments_by_schema():
    p = _parser()
    msg = p.message("I need the weather.</think>\n" + CALL)
    assert msg["content"] is None and msg["reasoning_content"] == "I need the weather."
    (call,) = msg["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("call_") and "index" not in call
    assert call["function"]["name"] == "get_weather"
    # the string-typed zip stays a string though it parses as a number
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 3, "zip": "75001"}
    assert p.finish_reason == "tool_calls"


@pytest.mark.parametrize("text", [
    "Thinking it over.</think>\n\nHere: " + CALL + "\n" + CALL,
    "Plan</think>The answer is <b>bold</b> and a < b.",
    "No close tag, the reply was cut off mid reasoning <thi",
    "Call straight from reasoning " + CALL,
    "x</think>" + CALL[:40],
])
def test_streaming_in_any_chunks_matches_the_whole_reply(text):
    whole = _streamed(text, len(text))
    for pieces in (1, 2, 3, 7):
        got = _streamed(text, pieces)
        assert got[:2] == whole[:2] and got[3] == whole[3]
        assert [c["function"] for c in got[2]] == [c["function"] for c in whole[2]]


def test_two_calls_take_indices_and_drop_the_space_between():
    reasoning, content, calls, finish = _streamed("Plan</think>\n" + CALL + "\n" + CALL + "\n", 5)
    assert [c["index"] for c in calls] == [0, 1] and content == "" and finish == "tool_calls"


def test_a_call_ends_the_reasoning_without_a_close_tag():
    reasoning, content, calls, _ = _streamed("Look it up " + CALL, 4)
    assert reasoning == "Look it up " and content == "" and len(calls) == 1


def test_a_call_to_an_unoffered_function_stays_text():
    text = "ok</think><tool_call>rm_rf<arg_key>path</arg_key><arg_value>/</arg_value></tool_call>"
    _, content, calls, finish = _streamed(text, 3)
    assert calls == [] and content.startswith("<tool_call>rm_rf") and finish == "stop"


def test_without_tools_a_call_tag_is_text():
    _, content, calls, _ = _streamed("ok</think>" + CALL, 3, tools=None)
    assert calls == [] and content == CALL


def test_a_cut_off_call_is_returned_as_text():
    _, content, calls, finish = _streamed("ok</think>" + CALL[:50], 6)
    assert calls == [] and content == CALL[:50] and finish == "stop"


def test_thinking_off_reads_the_reply_as_the_answer():
    msg = _parser(thinking=False).message("Hello </think> there")
    assert msg == {"role": "assistant", "content": "Hello </think> there"}


def test_stray_think_tags_drop():
    msg = _parser().message("<think>hmm</think>A</think>B")
    assert msg == {"role": "assistant", "content": "AB", "reasoning_content": "hmm"}


def test_think_in_the_answer_reopens_the_reasoning():
    msg = _parser().message("r</think>A<think>more</think>B")
    assert msg == {"role": "assistant", "content": "AB", "reasoning_content": "rmore"}


def test_a_reply_cut_off_at_the_call_tag_keeps_it():
    assert _parser().message("ok</think>Let me check. <tool_call>")["content"] == "Let me check. <tool_call>"


@pytest.mark.parametrize("schema, raw, want", [
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "94110", "94110"),
    ({"type": ["string", "null"]}, "true", "true"),
    ({"enum": ["1", "2"]}, "1", "1"),
    ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "5", 5),
    ({"type": "number"}, "3", 3),
    ({"type": "integer"}, "3.5", "3.5"),  # off the schema: left as the model wrote it
    ({"type": "number"}, "NaN", "NaN"),  # not JSON: arguments must stay parseable
    ({}, "Infinity", "Infinity"),
    ({"type": "array"}, '["a", 1]', ["a", 1]),
])
def test_arguments_take_the_schema_type(schema, raw, want):
    tools = glm_chat.tool_parameters([{"type": "function", "function": {
        "name": "f", "parameters": {"type": "object", "properties": {"a": schema}}}}])
    call = f"<tool_call>f<arg_key>a</arg_key><arg_value>{raw}</arg_value></tool_call>"
    msg = _parser(tools=tools).message("x</think>" + call)
    args = json.loads(msg["tool_calls"][0]["function"]["arguments"])  # strict JSON
    assert args == {"a": want} and type(args["a"]) is type(want)


def test_a_partial_tag_is_held_until_it_resolves():
    p = _parser()
    assert p.feed("Plan</thi") == [{"reasoning_content": "Plan"}]
    assert p.feed("nk>Done <tool") == [{"content": "Done"}]
    assert p.feed("box> here") == [{"content": " <toolbox> here"}]
    assert p.finish() == []


def test_assistant_tool_calls_reach_the_template_as_objects():
    msgs = glm_chat.template_messages([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ])
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    assert "content" not in msgs[1]  # GLM-5.2's template prints a null as "None"
    with pytest.raises(ValueError, match="not JSON"):
        glm_chat.template_messages([{"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": "{oops"}}]}])
    with pytest.raises(ValueError, match="function name"):
        glm_chat.template_messages([{"role": "assistant", "tool_calls": [{"function": {"arguments": "{}"}}]}])


def test_developer_turns_are_system_turns():
    msgs = glm_chat.template_messages([{"role": "developer", "content": "French."}, {"role": "user", "content": "hi"}])
    assert [m["role"] for m in msgs] == ["system", "user"]


def _chat(**body):
    body.setdefault("messages", [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather in Paris?"},
    ])
    return ChatCompletionRequest(model="glm52", **body)


def test_adapter_sends_messages_tools_and_template_switches(tmp_path):
    adapter = adapters.get_adapter("glm52")
    req = _chat(tools=[WEATHER], tool_choice="auto", enable_thinking=False,
                chat_template_kwargs={"reasoning_effort": "high"})
    mk = adapter.chat_to_request(req, tmp_path).model_kwargs
    assert [m["role"] for m in mk["messages"]] == ["system", "user"]
    assert mk["tools"] == [WEATHER] and "tool_choice" not in mk
    assert mk["chat_template_kwargs"] == {"reasoning_effort": "high", "enable_thinking": False}
    assert "enable_thinking" not in mk
    p = adapter.make_output_parser(req)
    assert p.message("Hi") == {"role": "assistant", "content": "Hi"}  # thinking off


@pytest.mark.parametrize("value", [None, 0, ""])
def test_a_falsy_enable_thinking_turns_thinking_off(value, tmp_path):
    # the GLM-5.2 template reads truthiness: its prompt ends in <think></think>
    p = adapters.get_adapter("glm52").make_output_parser(_chat(chat_template_kwargs={"enable_thinking": value}))
    assert p.message("Hello there!") == {"role": "assistant", "content": "Hello there!"}


def test_template_keys_it_does_not_read_are_dropped(tmp_path):
    req = _chat(chat_template_kwargs={"enable_thinking": False, "thinking": False, "tokenize": False})
    mk = adapters.get_adapter("glm52").chat_to_request(req, tmp_path).model_kwargs
    assert mk["chat_template_kwargs"] == {"enable_thinking": False}


def test_glm53_always_reasons(tmp_path):
    p = adapters.get_adapter("glm5_next").make_output_parser(_chat(enable_thinking=False))
    assert p.message("hmm</think>Hi")["reasoning_content"] == "hmm"


def test_tool_choice_none_leaves_the_tools_out(tmp_path):
    adapter = adapters.get_adapter("glm52")
    req = _chat(tools=[WEATHER], tool_choice="none")
    assert "tools" not in adapter.chat_to_request(req, tmp_path).model_kwargs
    _, _, calls, _ = _streamed("x</think>" + CALL, 4, tools=None)
    assert adapter.make_output_parser(req).message("x</think>" + CALL)["content"] == CALL and not calls


@pytest.mark.parametrize("body", [
    {"tools": [WEATHER], "tool_choice": "required"},
    {"tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
    {"tools": [{"type": "function"}]},
    {"tools": "get_weather"},
    {"tools": [{"type": "function", "function": {"name": "f", "parameters": "{}"}}]},
    {"tools": [{"type": "function", "function": {"name": "f", "parameters": {"properties": ["a"]}}}]},
    {"chat_template_kwargs": "thinking"},
])
def test_adapter_refuses_what_it_cannot_honour(body, tmp_path):
    with pytest.raises(ValueError):
        adapters.get_adapter("glm52").chat_to_request(_chat(**body), tmp_path)


class _RecordingTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        import torch

        self.calls.append((messages, kw))
        return torch.tensor([[1, 2, 3]])


def _glm52():
    from mstar.model.glm52.config import Glm52ModelConfig
    from mstar.model.glm52.glm52_model import Glm52Model

    m = object.__new__(Glm52Model)
    m.config = Glm52ModelConfig.reduced()
    m._tokenizer_mode, m._tokenizer = "hf", _RecordingTokenizer()
    return m


def _glm53():
    from mstar.model.glm5_next.glm5_next_model import Glm5NextModel

    m = Glm5NextModel("x", tokenizer_mode="byte")
    m._tokenizer_mode, m._tokenizer = "hf", _RecordingTokenizer()
    return m


@pytest.mark.parametrize("make", [_glm52, _glm53])
def test_process_prompt_renders_the_chat(make, tmp_path):
    pytest.importorskip("torch")
    m = make()
    mk = adapters.get_adapter("glm52").chat_to_request(
        _chat(tools=[WEATHER], reasoning_effort="high"), tmp_path).model_kwargs
    out = m.process_prompt("ignored", ["text"], ["text"], **mk)
    assert out["text_inputs"][0].tolist() == [1, 2, 3]
    messages, kw = m._tokenizer.calls[-1]
    assert [x["role"] for x in messages] == ["system", "user"]
    assert kw["tools"] == [WEATHER] and kw["reasoning_effort"] == "high"
    assert kw["add_generation_prompt"] is True
    # a bare prompt (``/generate``, the SDK) is still one user turn, and only the
    # template's switches pass: tokenize=False made the template return a str
    m.process_prompt("hello", ["text"], ["text"], chat_template_kwargs={"tokenize": False, "enable_thinking": False})
    messages, kw = m._tokenizer.calls[-1]
    assert messages == [{"role": "user", "content": "hello"}] and kw["tools"] is None
    assert "tokenize" not in kw and kw["enable_thinking"] is False
    with pytest.raises(ValueError, match="messages"):
        m.process_prompt("hello", ["text"], ["text"], messages="hi")


@pytest.mark.parametrize("raw", ["1e400", "[1e309]", "-1e999", '{"x": 1e400}'])
def test_an_overflowing_number_stays_text(raw):
    # json.loads gives inf; dumped back it is "Infinity", which JSON.parse rejects
    tools = glm_chat.tool_parameters([{"type": "function", "function": {"name": "f", "parameters": {}}}])
    call = f"<tool_call>f<arg_key>a</arg_key><arg_value>{raw}</arg_value></tool_call>"
    args = json.loads(_parser(tools=tools).message("x</think>" + call)["tool_calls"][0]["function"]["arguments"])
    assert args == {"a": raw}


@pytest.mark.parametrize("schema", [{"anyOf": {"type": "string"}}, {"oneOf": "string"}, {"enum": 5}])
def test_a_malformed_property_schema_does_not_break_the_parse(schema):
    tools = glm_chat.tool_parameters([{"type": "function", "function": {
        "name": "f", "parameters": {"properties": {"a": schema}}}}])
    call = "<tool_call>f<arg_key>a</arg_key><arg_value>3</arg_value></tool_call>"
    assert _parser(tools=tools).message("x</think>" + call)["tool_calls"][0]["function"]["arguments"] == '{"a": 3}'
