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


def _calls(deltas):
    """The calls the deltas stream, put together: index, id, name and arguments."""
    calls = {}
    for d in deltas:
        for c in d.get("tool_calls", []):
            if "id" in c:
                calls[c["index"]] = {"index": c["index"], "id": c["id"], "function": dict(c["function"])}
            else:
                calls[c["index"]]["function"]["arguments"] += c["function"]["arguments"]
    return list(calls.values())


def _streamed(text, pieces, **kw):
    """The deltas of ``text`` fed in ``pieces``-character chunks, merged back."""
    p = _parser(**kw)
    deltas = []
    for i in range(0, len(text), pieces):
        deltas += p.feed(text[i:i + pieces])
    deltas += p.finish()
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    return reasoning, content, _calls(deltas), p.finish_reason


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


def _whole_call_arguments(body, tools):
    """A call's arguments as the parser read them whole, before they streamed."""
    import re

    props = tools[body.partition("<arg_key>")[0].strip()]
    pairs = re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", body, re.DOTALL)
    return json.dumps({k.strip(): glm_chat._arg_value(v, props.get(k.strip())) for k, v in pairs}, ensure_ascii=False)


EDIT = glm_chat.tool_parameters([{"type": "function", "function": {"name": "edit", "parameters": {
    "type": "object", "properties": {
        "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": ["string", "null"]},
        "line": {"type": "integer"}, "opts": {"type": "object"}, "tags": {"type": "array"}, "any": {}}}}}])


@pytest.mark.parametrize("body", [
    "edit<arg_key>path</arg_key><arg_value>src/a.py</arg_value>",
    'edit<arg_key>old</arg_key><arg_value>if a < b and "q" in s:\n\treturn \\x</arg_value>'
    "<arg_key>new</arg_key><arg_value>é√ 😀 </arg_valu </arg_value>",
    'edit<arg_key>line</arg_key><arg_value>3.0</arg_value><arg_key>opts</arg_key><arg_value>{"a": [1, "<"]}'
    '</arg_value><arg_key>tags</arg_key><arg_value>["x"]</arg_value><arg_key>any</arg_key><arg_value>nope'
    "</arg_value>",
    "edit\n<arg_key> path </arg_key>\n<arg_value>a</arg_value>\n<arg_key>line</arg_key> <arg_value>7</arg_value>\n",
    "edit",
    "edit<arg_key>unknown</arg_key><arg_value>NaN</arg_value><arg_key>old</arg_key><arg_value></arg_value>",
])
def test_streamed_arguments_add_up_to_the_whole_calls_json_in_any_chunking(body):
    text = "plan</think>Sure. <tool_call>" + body + "</tool_call>"
    want = _whole_call_arguments(body, EDIT)
    for pieces in (1, 2, 3, 5, 7, 11, len(text)):
        p, deltas = _parser(tools=EDIT), []
        for i in range(0, len(text), pieces):
            deltas += p.feed(text[i:i + pieces])
        deltas += p.finish()
        (call,) = _calls(deltas)
        assert call["function"] == {"name": "edit", "arguments": want}
        assert "".join(d.get("content", "") for d in deltas) == "Sure."
    # the id and name go out before the arguments are read
    p = _parser(tools=EDIT)
    first = p.feed("x</think><tool_call>edit<arg_key>pa")
    assert first[-1]["tool_calls"][0]["function"] == {"name": "edit", "arguments": ""}
    assert p.feed("th</arg_key><arg_value>src/") == [{"tool_calls": [{"index": 0, "function": {
        "arguments": '{"path": "src/'}}]}]


def test_a_string_value_streams_with_a_partial_tag_held():
    p = _parser(tools=EDIT)
    p.feed("x</think><tool_call>edit<arg_key>old</arg_key><arg_value>a <")
    assert p.feed("b> </arg_v") == [{"tool_calls": [{"index": 0, "function": {"arguments": "<b> "}}]}]
    assert p.feed("alue></tool_call>") == [{"tool_calls": [{"index": 0, "function": {"arguments": '"}'}}]}]


def test_a_single_call_parser_is_done_at_the_end_of_its_call():
    p = glm_chat.GlmReplyParser(thinking=True, tools=TOOLS, single_call=True)
    deltas = p.feed("Plan</think>" + CALL[:-3])
    assert not p.done
    deltas += p.feed(CALL[-3:] + "\nMore text" + CALL)
    assert p.done and p.feed("x") == [] and p.finish() == []
    assert [c["function"]["name"] for c in _calls(deltas)] == ["get_weather"] and p.finish_reason == "tool_calls"
    assert not any("content" in d for d in deltas)


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


def test_a_call_cut_off_before_its_name_ends_is_text():
    _, content, calls, finish = _streamed("ok</think>" + CALL[:20], 6)
    assert calls == [] and content == CALL[:20] and finish == "stop"


@pytest.mark.parametrize("cut, arguments", [
    (len("<tool_call>get_weather<arg_key>"), ""),
    (len("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par"), '{"city": "Par'),
    (len("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key>"
         "<arg_value>3"), '{"city": "Paris", "days": '),
])
def test_a_call_cut_off_after_its_name_stays_a_call_as_far_as_it_got(cut, arguments):
    for pieces in (1, 4, cut):
        _, content, calls, finish = _streamed("ok</think>" + CALL[:cut], pieces)
        assert content == "" and [c["function"] for c in calls] == [{"name": "get_weather", "arguments": arguments}]
        # not tool_calls: the arguments are not whole
        assert finish == "stop"


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
    ({"type": "integer"}, "3.0", 3),
    ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "2e1", 20),
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


@pytest.mark.parametrize("role", ["function", "observation", None])
def test_a_role_the_templates_skip_is_refused(role):
    with pytest.raises(ValueError, match="role"):
        glm_chat.template_messages([{"role": role, "content": "x"}, {"role": "user", "content": "hi"}])


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


@pytest.mark.parametrize("model", ["glm52", "glm5_next"])
def test_an_agent_turn_with_a_tool_result_reaches_the_template(model, tmp_path):
    req = _chat(tools=[WEATHER], messages=[
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ])
    mk = adapters.get_adapter(model).chat_to_request(req, tmp_path).model_kwargs
    assert [m["role"] for m in mk["messages"]] == ["user", "assistant", "tool"]
    assert mk["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}


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
    adapter = adapters.get_adapter("glm5_next")
    assert adapter.make_output_parser(_chat()).message("hmm</think>Hi")["reasoning_content"] == "hmm"
    assert "chat_template_kwargs" in adapter.chat_to_request(_chat(enable_thinking=True), tmp_path).model_kwargs


@pytest.mark.parametrize("value", [False, 0, ""])
def test_glm53_refuses_thinking_off(value, tmp_path):
    req = _chat(chat_template_kwargs={"enable_thinking": value})
    with pytest.raises(ValueError, match="enable_thinking"):
        adapters.get_adapter("glm5_next").chat_to_request(req, tmp_path)


def test_tool_choice_none_leaves_the_tools_out(tmp_path):
    adapter = adapters.get_adapter("glm52")
    req = _chat(tools=[WEATHER], tool_choice="none")
    assert "tools" not in adapter.chat_to_request(req, tmp_path).model_kwargs
    _, _, calls, _ = _streamed("x</think>" + CALL, 4, tools=None)
    assert adapter.make_output_parser(req).message("x</think>" + CALL)["content"] == CALL and not calls


@pytest.mark.parametrize("body", [
    {"tool_choice": "required"},
    {"tools": [WEATHER], "tool_choice": "any"},
    {"tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "get_time"}}},
    {"tools": [WEATHER], "tool_choice": {"type": "function", "name": "get_weather"}},
    {"tools": [WEATHER], "tool_choice": "required", "enable_thinking": True},
    {"reply_prefix": "<tool_call>get_weather"},
    {"tools": [WEATHER], "parallel_tool_calls": "no"},
    {"tools": [WEATHER], "parallel_tool_calls": 0},
    {"tools": [{"type": "function"}]},
    {"tools": "get_weather"},
    {"tools": [{"type": "function", "function": {"name": "f", "parameters": "{}"}}]},
    {"tools": [{"type": "function", "function": {"name": "f", "parameters": {"properties": ["a"]}}}]},
    {"chat_template_kwargs": "thinking"},
])
def test_adapter_refuses_what_it_cannot_honour(body, tmp_path):
    with pytest.raises(ValueError):
        adapters.get_adapter("glm52").chat_to_request(_chat(**body), tmp_path)


TIME = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}


ARGS = "<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"


@pytest.mark.parametrize("model, choice, tools, prefix, thinking, reply", [
    # one tool: it is named, and the reply starts at the arguments
    ("glm52", "required", [WEATHER], "<tool_call>get_weather", False, ARGS),
    ("glm52", "required", [WEATHER, TIME], "<tool_call>", False, "get_weather" + ARGS),
    ("glm52", {"type": "function", "function": {"name": "get_weather"}}, [WEATHER, TIME],
     "<tool_call>get_weather", False, ARGS),
    ("glm5_next", "required", [WEATHER, TIME], "</think><tool_call>", None, "get_weather" + ARGS),
    ("glm5_next", {"type": "function", "function": {"name": "get_weather"}}, [WEATHER, TIME],
     "</think><tool_call>get_weather", None, ARGS),
])
def test_a_forced_tool_choice_opens_the_call_in_the_prompt(model, choice, tools, prefix, thinking, reply, tmp_path):
    adapter = adapters.get_adapter(model)
    req = _chat(tools=tools, tool_choice=choice)
    mk = adapter.chat_to_request(req, tmp_path).model_kwargs
    assert mk["reply_prefix"] == prefix and mk["tools"] == tools
    # GLM-5.2's prompt ends <think></think>, as a turn that does not reason
    assert mk["chat_template_kwargs"].get("enable_thinking") is thinking
    p = adapter.make_output_parser(req)
    msg = p.message(reply)
    (call,) = msg["tool_calls"]
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert msg["content"] is None and "reasoning_content" not in msg and p.finish_reason == "tool_calls"


@pytest.mark.parametrize("model", ["glm52", "glm5_next"])
def test_a_forced_function_is_the_only_one_its_call_can_name(model):
    """Forced get_weather beside get_weather_hourly: the model writing on past the prompt's
    name parsed as a call to the other tool."""
    hourly = {"type": "function", "function": {
        "name": "get_weather_hourly", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
    req = _chat(tools=[WEATHER, hourly], tool_choice={"type": "function", "function": {"name": "get_weather"}})
    p = adapters.get_adapter(model).make_output_parser(req)
    msg = p.message("_hourly" + ARGS)
    assert not msg.get("tool_calls") and p.finish_reason == "stop"
    assert msg["content"].startswith("<tool_call>get_weather_hourly")


def test_a_forced_call_to_a_made_up_function_stays_text():
    p = glm_chat.GlmReplyParser(thinking=False, tools=TOOLS, forced_call="")
    msg = p.message("rm_rf<arg_key>path</arg_key><arg_value>/</arg_value></tool_call>")
    assert msg["content"].startswith("<tool_call>rm_rf<arg_key>") and p.finish_reason == "stop"


def test_auto_and_none_leave_the_prompt_alone(tmp_path):
    for choice in ("auto", "none", None):
        req = _chat(tools=[WEATHER], tool_choice=choice)
        mk = adapters.get_adapter("glm52").chat_to_request(req, tmp_path).model_kwargs
        assert "reply_prefix" not in mk and "enable_thinking" not in mk["chat_template_kwargs"]


class _RecordingTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        import torch

        self.calls.append((messages, kw))
        return torch.tensor([[1, 2, 3]])

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens  # no [gMASK]<sop> mid prompt
        return [100 + len(text)]


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
    assert out.new_input_tensors["text_inputs"][0].tolist() == [1, 2, 3]
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


@pytest.mark.parametrize("make", [_glm52, _glm53])
def test_process_prompt_ends_with_the_reply_prefix(make, tmp_path):
    pytest.importorskip("torch")
    m = make()
    mk = adapters.get_adapter("glm52").chat_to_request(
        _chat(tools=[WEATHER], tool_choice="required"), tmp_path).model_kwargs
    out = m.process_prompt("ignored", ["text"], ["text"], **mk)
    assert out["text_inputs"][0].tolist() == [1, 2, 3, 100 + len("<tool_call>get_weather")]
    m._tokenizer_mode = "byte"
    out = m.process_prompt("hi", ["text"], ["text"], reply_prefix="<t")
    assert out["text_inputs"][0].tolist() == list(b"hi<t")


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


def test_text_chunks_carry_their_usage():
    torch = pytest.importorskip("torch")
    from mstar.api_server.data_worker import _text_usage

    assert _text_usage(torch.tensor([5, 6, 9]), frozenset({9}), 12) == {
        "tokens": 3, "prompt_tokens": 12, "stop_token": True}
    assert _text_usage(torch.tensor([5, 6]), frozenset({9}), 12)["stop_token"] is False
    # a model that names no stop tokens leaves the reason alone
    assert _text_usage(torch.tensor([5]), None, None) == {"tokens": 1}


def test_the_data_worker_tags_text_chunks_with_usage():
    torch = pytest.importorskip("torch")
    import queue
    from types import SimpleNamespace

    from mstar.api_server.data_worker import PreprocessWorkerThread, RequestOutputState

    class _Model:
        def postprocess(self, tensor, modality, request_kwargs=None):
            return b"hi"

        def stop_token_ids(self):
            return frozenset({9})

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.out_queue, wt.model, wt.request_model_kwargs = queue.Queue(), _Model(), {}
    wt.tensor_uuid_to_metadata_per_request = {"r1": {"u1": {}}}
    wt.enable_prof = wt.enable_nvtx = False
    wt.request_output_state = {"r1": RequestOutputState(order={"u1": (0, None)}, next_sequence=1, prompt_tokens=7)}
    edge = SimpleNamespace(name="text_output", tensor_info=[SimpleNamespace(uuid="u1")])
    wt.tensor_manager = SimpleNamespace(get_ready_tensors=lambda: {"r1": [edge]}, take_failed_reads=lambda: [],
                                        get_tensor=lambda uuid: torch.tensor([4, 9]), dereference=lambda uuid: None)
    assert wt._process_read_tensors() is True
    assert wt.out_queue.get_nowait().metadata == {"tokens": 2, "prompt_tokens": 7, "stop_token": True}


def test_a_long_call_is_read_once_and_parses_as_whole():
    # quadratic otherwise: each feed copied and searched the whole call so far
    write = glm_chat.tool_parameters([{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}}])
    body = "x = 1  # filler line\n" * 5000
    text = ("plan</think><tool_call>write<arg_key>path</arg_key><arg_value>a.py</arg_value>"
            "<arg_key>content</arg_key><arg_value>" + body + "</arg_value></tool_call>")
    p, deltas, longest = _parser(tools=write), [], 0
    for i in range(0, len(text), 4):
        deltas += p.feed(text[i:i + 4])
        longest = max(longest, len(p._buf))
    deltas += p.finish()
    # what is still searched for the end tag stays a feed long, whatever the call's length
    assert longest < len(glm_chat.CALL_END) + 4
    [call] = _calls(deltas)
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py", "content": body}
    assert _parser(tools=write).message(text)["tool_calls"][0]["function"] == call["function"]


@pytest.mark.parametrize("make", [_glm52, _glm53])
def test_the_chat_stays_in_the_data_worker(make, tmp_path):
    from mstar.communication import wire
    from mstar.utils.ipc_format import ConductorMessage, ConductorMessageType, NewRequestConductor

    tool = {"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"n": {"type": "integer", "maximum": 2**64}}}}}
    mk = adapters.get_adapter("glm52").chat_to_request(_chat(tools=[tool]), tmp_path).model_kwargs
    # what the data worker sends the conductor: the wire carries no int past 64 bits
    sent = {k: v for k, v in mk.items() if k not in make().PREPROCESS_ONLY_KWARGS}
    mk["reply_prefix"] = "<tool_call>f"
    sent = {k: v for k, v in mk.items() if k not in make().PREPROCESS_ONLY_KWARGS}
    assert {"messages", "tools", "chat_template_kwargs"} <= set(mk)
    assert not {"messages", "tools", "reply_prefix"} & set(sent)
    body = NewRequestConductor("r", {}, ["text"], ["text"], {}, sent)
    wire.encode(ConductorMessage(message_type=ConductorMessageType.NEW_REQUEST, body=body))


@pytest.mark.parametrize("stops, text, want, stop", [
    (["END"], "abcENDdef", "abc", "END"),
    (["END"], "abcENdef", "abcENdef", None),
    (["xyz", "y"], "axyz", "ax", "y"),  # the stop that completes first wins
    (["bc", "abc"], "zabcd", "z", "abc"),  # of two ending together, the longer
    (["\n\n"], "a\nb\n\nc", "a\nb", "\n\n"),
    (["ab", "b"], "aab", "a", "ab"),
])
def test_a_stop_string_ends_the_text_in_any_chunking(stops, text, want, stop):
    from mstar.api_server.openai.reply import StopMatcher

    for pieces in (1, 2, 3, len(text)):
        m, out, got = StopMatcher(stops), "", None
        for i in range(0, len(text), pieces):
            emitted, got = m.feed(text[i:i + pieces])
            out += emitted
            if got is not None:
                break
        if got is None:
            out += m.flush()
        assert (out, got) == (want, stop)


def test_a_stop_string_holds_back_only_what_may_start_it():
    from mstar.api_server.openai.reply import StopMatcher

    m = StopMatcher(["</answer>"])
    assert m.feed("plain text") == ("plain text", None)
    assert m.feed(" and </ans") == (" and ", None)
    assert m.feed("wer") == ("", None) and m.feed(">!") == ("", "</answer>")


def test_any_reply_parses_alike_in_any_chunking_and_whole_calls_are_json():
    """Seeded fuzz over tags, partial tags and text: chunking never changes the
    reply, and a reply whose calls all closed has JSON arguments."""
    import random

    tools = glm_chat.tool_parameters([
        {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
            "s": {"type": "string"}, "n": {"type": "integer"}, "o": {}}}}},
        {"type": "function", "function": {"name": "g", "parameters": {}}}])
    frags = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
             "</arg_value>", "f", "g", "h", "s", "n", "o", " ", "\n", "<", "</", "x", "1", '"q"', "é", "[1]",
             "</arg_v", "<tool"]

    def run(text, size, **kw):
        p, deltas = glm_chat.GlmReplyParser(tools=tools, **kw), []
        for i in range(0, len(text), size):
            deltas += p.feed(text[i:i + size])
        deltas += p.finish()
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        return reasoning, content, [c["function"] for c in _calls(deltas)], p.finish_reason, p.done

    rng = random.Random(0)
    for _ in range(1500):
        text = "".join(rng.choice(frags) for _ in range(rng.randint(0, 40)))
        kw = {"thinking": rng.random() < 0.5, "single_call": rng.random() < 0.3,
              "forced_call": rng.choice([None, None, "", "f", "h"])}
        whole = run(text, max(1, len(text)), **kw)
        for size in (1, 2, 3, 7):
            assert run(text, size, **kw) == whole, (text, size, kw)
        if whole[3] == "tool_calls":
            for fn in whole[2]:
                json.loads(fn["arguments"])
