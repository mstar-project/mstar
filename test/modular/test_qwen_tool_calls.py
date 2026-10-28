"""Tool calls in Qwen's markup (``qwen3_coder``: Qwen3-Coder, Qwen3-Coder-Next, Qwen3.5 to
Qwen3.8): the reply's calls come back as OpenAI tool calls, streamed or whole.

``chat_templates/`` holds the checkpoints' own ``chat_template.jinja``:
Qwen/Qwen3-Coder-30B-A3B-Instruct @b2cff646e (Qwen3-Coder-480B's is the same),
Qwen/Qwen3-Coder-Next @a7fbcb5c0, Qwen/Qwen3.5-4B @851bf6e80 and Qwen/Qwen3.8-27B @1d4bf0f2f
(Qwen3.8-Flash-Next's is the same; Qwen3.6's differs from it in switches only). They write a
non-string argument back as Python (``True``, ``None``: Qwen3-Coder, Qwen3.5) or as JSON
(Coder-Next, Qwen3.8), and the Qwen3.5 family reasons.
"""

import json
import random
import re
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from mstar.api_server.openai.tool_calls import QWEN3_CODER, ReplyParser, tool_parameters  # noqa: E402

TEMPLATES = {p.stem: p.read_text() for p in (Path(__file__).parent / "chat_templates").glob("qwen3_*.jinja")}

EDIT = {"type": "function", "function": {"name": "edit", "description": "Edit a file", "parameters": {
    "type": "object", "properties": {
        "path": {"type": "string"}, "old": {"type": "string"}, "line": {"type": "integer"},
        "dry": {"type": "boolean"}, "opts": {"type": "object"}, "tags": {"type": "array"},
        "note": {"type": ["string", "null"]}, "ratio": {"type": "number"}, "maybe": {"type": ["integer", "null"]}}}}}
LS = {"type": "function", "function": {"name": "ls", "parameters": {"type": "object", "properties": {}}}}
WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}
TOOLS = [EDIT, LS, WEATHER]
PARAMS = tool_parameters(TOOLS)
# every type, and string values with markup in them: a value ends only at </parameter>
ARGS = {"path": "src/a.py", "old": "if a < b:\n    <div>x</div>\n</function> <parameter=x>\n", "line": 3,
        "dry": True, "opts": {"a": [1, "<"]}, "tags": ["x", "y"], "note": "", "ratio": 0.5, "maybe": None}
CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")


def _parser(thinking=False, tools=PARAMS, **kw):
    return ReplyParser(QWEN3_CODER, thinking=thinking, tools=tools, **kw)


def _calls(deltas):
    calls = {}
    for d in deltas:
        for c in d.get("tool_calls", []):
            if "id" in c:
                calls[c["index"]] = {"index": c["index"], "id": c["id"], "function": dict(c["function"])}
            else:
                calls[c["index"]]["function"]["arguments"] += c["function"]["arguments"]
    return list(calls.values())


def _run(text, size, **kw):
    """The reply fed ``size`` characters at a time: reasoning, content, calls, finish, done."""
    p, deltas = _parser(**kw), []
    for i in range(0, len(text), max(1, size)):
        deltas += p.feed(text[i:i + size])
    deltas += p.finish()
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    return reasoning, content, [c["function"] for c in _calls(deltas)], p.finish_reason, p.done


def _render(template, messages, tools=None, **kw):
    from transformers.utils.chat_template_utils import render_jinja_template

    out = render_jinja_template([messages], tools=tools, chat_template=template, **kw)
    return out[0][0] if isinstance(out, tuple) else out[0]


# --------------------------------------------------------------------------
# the reply, as the templates write a call
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_a_call_the_template_writes_parses_back_to_itself(name):
    """The assistant turn a template renders is what the model learnt to write: parsed, in
    any chunking, it gives back the call's name and typed arguments."""
    pytest.importorskip("transformers")
    messages = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "Let me edit.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "edit", "arguments": ARGS}},
            {"id": "c2", "type": "function", "function": {"name": "ls", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "tool", "tool_call_id": "c2", "content": "a.py"},
        {"role": "user", "content": "thanks"},
    ]
    text = _render(TEMPLATES[name], messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=False)
    turn = re.search(r"<\|im_start\|>assistant\n?(?:<think>\n\n</think>\n\n)?(.*?)<\|im_end\|>", text, re.S).group(1)
    want = [{"name": "edit", "arguments": json.dumps(ARGS, ensure_ascii=False)}, {"name": "ls", "arguments": "{}"}]
    for size in (1, 2, 3, 5, 7, 13, len(turn)):
        assert _run(turn, size) == ("", "Let me edit.", want, "tool_calls", False), (name, size)
    # and the tool results went back as the template's own turn
    assert "<tool_response>\nok\n</tool_response>" in text and "<tool_response>\na.py\n</tool_response>" in text


def test_the_id_and_name_go_out_before_the_arguments_and_a_string_streams_as_written():
    p = _parser()
    first = p.feed("<tool_call>\n<function=edit>\n<parameter=pa")
    assert first[-1]["tool_calls"][0]["function"] == {"name": "edit", "arguments": ""}
    assert first[-1]["tool_calls"][0]["id"].startswith("call_")
    # the template's newline after the key is not the value's; one before </parameter> may be
    assert p.feed("th>\nsrc/") == [{"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "src/'}}]}]
    assert p.feed("a.py\n") == [{"tool_calls": [{"index": 0, "function": {"arguments": "a.py"}}]}]
    assert p.feed("\n</param") == [{"tool_calls": [{"index": 0, "function": {"arguments": "\\n"}}]}]
    assert p.feed("eter>\n</function>") == [{"tool_calls": [{"index": 0, "function": {"arguments": '"}'}}]}]
    assert p.finish() == [] and p.finish_reason == "tool_calls"


@pytest.mark.parametrize("raw, schema, want", [
    ("True", {"type": "boolean"}, True), ("true", {"type": "boolean"}, True), ("False", {}, False),
    ("None", {"type": ["integer", "null"]}, None), ("null", {"type": ["integer", "null"]}, None),
    ("3", {"type": "integer"}, 3), ("3.0", {"type": "integer"}, 3), ("2.5", {"type": "number"}, 2.5),
    ("94110", {"type": "string"}, "94110"), ("None", {"type": "string"}, "None"),
    ('{"a": [1, 2]}', {"type": "object"}, {"a": [1, 2]}), ("[1, 2]", {}, [1, 2]),
    ("Paris", {}, "Paris"), ("True story", {"type": "boolean"}, "True story"),
])
def test_values_take_the_schema_type_in_python_or_json_spelling(raw, schema, want):
    tools = tool_parameters([{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"a": schema}}}}])
    text = f"<tool_call>\n<function=f>\n<parameter=a>\n{raw}\n</parameter>\n</function>\n</tool_call>"
    args = json.loads(_parser(tools=tools).message(text)["tool_calls"][0]["function"]["arguments"])
    assert args == {"a": want} and type(args["a"]) is type(want)


@pytest.mark.parametrize("value", [
    "", "\n", "x\n", "\nx", "<div>\n</div>", "</function>\n</tool_call",
    "<parameter=y>\nz\n", 'quote " and \\ and é😀', "<tool_call>\n<function=ls>\n",
])
def test_a_string_value_is_what_lies_between_the_template_s_newlines(value):
    text = f"<tool_call>\n<function=edit>\n<parameter=old>\n{value}\n</parameter>\n</function>\n</tool_call>"
    for size in (1, 2, 3, 5, len(text)):
        (fn,) = _run(text, size)[2]
        assert json.loads(fn["arguments"]) == {"old": value}, size


def test_content_before_calls_and_two_calls():
    text = "Checking two cities.\n\n" + CALL + "\n" + CALL.replace("Paris", "Rome")
    reasoning, content, calls, finish, _ = _run(text, 4)
    assert content == "Checking two cities." and finish == "tool_calls"
    assert [json.loads(c["arguments"])["city"] for c in calls] == ["Paris", "Rome"]


def test_reasoning_then_a_call_and_a_call_straight_from_reasoning():
    assert _run("Need weather.\n</think>\n\n" + CALL, 3, thinking=True)[:2] == ("Need weather.\n", "")
    reasoning, content, calls, finish, _ = _run("Look it up " + CALL, 5, thinking=True)
    assert reasoning == "Look it up " and content == "" and len(calls) == 1 and finish == "tool_calls"


def test_without_reasoning_think_tags_are_text():
    # Qwen3-Coder does not reason: its reply is all answer
    assert _run("a <think>b</think> c", 2)[1] == "a <think>b</think> c"


def test_a_call_to_an_unoffered_function_stays_text():
    text = "ok " + CALL.replace("get_weather", "rm_rf")
    _, content, calls, finish, _ = _run(text, 3)
    assert calls == [] and content == text and finish == "stop"


# what Qwen3-Coder-30B-A3B wrote on M* for the agent task "port" (benchmark/text_lm/tool_eval.py)
BARE = ("The user wants the port, which is in `config/server.yaml`. I will read it.\n<function=read_file>\n"
        "<parameter=path>\nconfig/server.yaml\n</parameter>\n</function>\n</tool_call>")


def test_a_call_written_without_its_tool_call_tag_is_still_a_call():
    tools = tool_parameters([{"type": "function", "function": {"name": "read_file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}}}}])
    for size in (1, 2, 3, 7, len(BARE)):
        _, content, calls, finish, _ = _run(BARE, size, tools=tools)
        assert content == "The user wants the port, which is in `config/server.yaml`. I will read it."
        assert calls == [{"name": "read_file", "arguments": '{"path": "config/server.yaml"}'}]
        assert finish == "tool_calls"
    # without the stray </tool_call>, what follows the call is answer again
    text = BARE[:-len("\n</tool_call>")] + "\nDone."
    assert _run(text, 3, tools=tools)[1:3] == (
        "The user wants the port, which is in `config/server.yaml`. I will read it.\nDone.",
        [{"name": "read_file", "arguments": '{"path": "config/server.yaml"}'}])
    # a function not offered, or no tools at all: text, and nothing after it is swallowed
    for kw in ({"tools": tool_parameters([LS])}, {"tools": None}):
        assert _run(BARE + "\nok", 3, **kw)[1:3] == (BARE + "\nok", [])


def test_a_tool_call_tag_without_a_function_stays_text():
    assert _run("x <tool_call>\nhello\n</tool_call>", 4)[1] == "x <tool_call>\nhello\n</tool_call>"


@pytest.mark.parametrize("cut, args, finish", [
    (len("<tool_call>\n<function=get_wea"), None, "stop"),  # before the name ends: text
    (len("<tool_call>\n<function=get_weather>\n<parameter=city>\nPar"), '{"city": "Par', "stop"),
    (len("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3"),
     '{"city": "Paris", "days": ', "stop"),
    # past </function> the arguments are whole, though </tool_call> never came
    (len(CALL) - len("\n</tool_call>"), '{"city": "Paris", "days": 3}', "tool_calls"),
])
def test_a_reply_cut_off_in_a_call(cut, args, finish):
    for size in (1, 4, cut):
        _, content, calls, got, _ = _run(CALL[:cut], size)
        if args is None:
            assert calls == [] and content == CALL[:cut]
        else:
            assert content == "" and calls == [{"name": "get_weather", "arguments": args}]
        assert got == finish


def test_a_single_call_parser_is_done_at_the_end_of_its_function():
    p = _parser(single_call=True)
    deltas = p.feed("Two. " + CALL[:-20])
    assert not p.done
    deltas += p.feed(CALL[-20:] + "\n" + CALL)
    assert p.done and p.feed("x") == [] and p.finish() == []
    assert [c["function"]["name"] for c in _calls(deltas)] == ["get_weather"] and p.finish_reason == "tool_calls"


def test_a_forced_call_starts_the_reply_inside_it():
    assert QWEN3_CODER.forced_prefix("get_weather") == "<tool_call>\n<function=get_weather>\n"
    assert QWEN3_CODER.forced_prefix("") == "<tool_call>\n"
    p = _parser(tools={"get_weather": PARAMS["get_weather"]}, forced_call="get_weather")
    msg = p.message("<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>")
    assert msg["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert msg["content"] is None and p.finish_reason == "tool_calls"
    # the model names the function; a made-up one stays text
    assert _parser(forced_call="").message(CALL[len("<tool_call>\n"):])["tool_calls"][0]["function"]["name"] \
        == "get_weather"
    msg = _parser(forced_call="").message("<function=rm_rf>\n</function>\n</tool_call>")
    assert msg["content"] == "<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>" and "tool_calls" not in msg


def test_a_long_value_is_read_once():
    body = "x = 1  # <filler> line\n" * 5000
    text = f"<tool_call>\n<function=edit>\n<parameter=old>\n{body}\n</parameter>\n</function>\n</tool_call>"
    p, deltas, longest = _parser(), [], 0
    for i in range(0, len(text), 4):
        deltas += p.feed(text[i:i + 4])
        longest = max(longest, len(p._buf))
    deltas += p.finish()
    # held back: at most a partial </parameter> or </tool_call>, and the newline before it
    assert longest < len("</tool_call>") + 5
    assert json.loads(_calls(deltas)[0]["function"]["arguments"]) == {"old": body}


def test_any_reply_parses_alike_in_any_chunking_and_whole_calls_are_json():
    """Seeded fuzz over tags, partial tags and text: chunking never changes the
    reply, and a reply whose calls all closed has JSON arguments."""
    frags = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=",
             "</parameter>", ">", "\n", "edit", "ls", "rm", "path", "line", "opts", " ", "<", "</", "x", "1",
             '"q"', "é", "[1]", "True", "None", "</param", "<tool", "<funct"]
    rng = random.Random(0)
    for _ in range(1500):
        text = "".join(rng.choice(frags) for _ in range(rng.randint(0, 40)))
        kw = {"thinking": rng.random() < 0.5, "single_call": rng.random() < 0.3,
              "forced_call": rng.choice([None, None, "", "edit"])}
        whole = _run(text, max(1, len(text)), **kw)
        for size in (1, 2, 3, 7):
            assert _run(text, size, **kw) == whole, (text, size, kw)
        if whole[3] == "tool_calls":
            for fn in whole[2]:
                json.loads(fn["arguments"])
