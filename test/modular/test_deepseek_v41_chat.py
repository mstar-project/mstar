"""DeepSeek-V4.1 chat: a request reaches the model as the prompt its reference encoder writes,
and the reply comes back split into reasoning, content and tool calls, streamed or whole.

The prompts are the encoder's (``chat_templates/deepseek_v41_encoding``, see
``test_deepseek_v41_chat_template``); the replies' grammar is in ``chat_formats/replies.json``.
"""

import json

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("transformers")

from test_deepseek_v41_chat_template import (  # noqa: E402
    CASES,
    ENCODING,
    _checkpoint_tokenizer,
    byte_tokenizer,
    rendered,
)

from mstar.api_server.openai import adapters, serving_chat  # noqa: E402
from mstar.api_server.openai.protocol import ChatCompletionRequest  # noqa: E402
from mstar.api_server.openai.reply import Finish, Usage  # noqa: E402

ADAPTER = adapters.DeepseekV41ChatAdapter()
EOS = "<｜end▁of▁sentence｜>"
WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}
TIME = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
ARGS = ('<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
CALL = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n' + ARGS


@pytest.fixture(scope="module")
def tokenizer():
    return byte_tokenizer()


def _chat(**body):
    body.setdefault("messages", [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather in Paris?"},
    ])
    return ChatCompletionRequest(model="deepseek_v41", **body)


def _prompt(tokenizer, req, tmp_path) -> str:
    """The prompt ``process_prompt`` renders for the request the adapter submits."""
    args = ADAPTER.chat_to_request(req, tmp_path)
    return rendered(tokenizer, args.model_kwargs, args.text, args.prompt_parts)


def _message(parser, text) -> dict:
    """A whole reply's chat ``message``, as the chat handler builds it from the parser's deltas."""
    events = [*parser.feed(text), *parser.finish(), Finish(parser.finish_reason, Usage())]
    return serving_chat._text_response("deepseek_v41", "chatcmpl-0", events)["choices"][0]["message"]


def _as_clients_send(messages):
    """Each call's arguments as the JSON string an OpenAI client sends."""
    messages = json.loads(json.dumps(messages))
    for m in messages:
        for call in m.get("tool_calls") or []:
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
    return messages


# ── the request ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["id"])
def test_a_chat_reaches_the_model_as_the_prompt_the_encoder_writes(tokenizer, case, tmp_path):
    req = _chat(messages=_as_clients_send(case["messages"]), tools=CASES["toolsets"].get(case["tools"]),
                enable_thinking=case["thinking"], **case["switches"])
    if case["prompt"] is None:
        with pytest.raises(ValueError):
            _prompt(tokenizer, req, tmp_path)
    else:
        assert _prompt(tokenizer, req, tmp_path) == case["prompt"]


@pytest.mark.parametrize("i", [1, 2])
def test_the_encoders_goldens_with_the_chat_apis_roles_render_through_the_adapter(tokenizer, i, tmp_path):
    """Its other three use roles and fields the chat API has no name for, or images."""
    raw = json.loads((ENCODING / f"test_input_{i}.json").read_text())
    raw = {"messages": raw} if isinstance(raw, list) else raw
    thinking = raw.get("thinking_mode") == "thinking"
    req = _chat(messages=raw["messages"], tools=raw.get("tools"), enable_thinking=thinking)
    assert _prompt(tokenizer, req, tmp_path) == (ENCODING / f"test_output_{i}.txt").read_text()


def test_an_image_part_is_refused_and_leaves_no_upload(tmp_path):
    """Vision is not served: the encoder's fifth golden, a chat with two images, is a 400."""
    png = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4"
           "DwABBAEAwS2OUAAAAABJRU5ErkJggg==")
    messages = json.loads((ENCODING / "test_input_5.json").read_text())["messages"]
    for part in messages[1]["content"]:
        if part["type"] == "image_url":
            part["image_url"]["url"] = png
    with pytest.raises(ValueError, match="text only"):
        ADAPTER.chat_to_request(_chat(messages=messages), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_the_adapter_sends_messages_tools_and_the_templates_switches(tmp_path):
    req = _chat(tools=[WEATHER], tool_choice="auto", reasoning_effort=40, messages=[
        {"role": "developer", "content": "French."},
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "reasoning_content": "Look it up.", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ], chat_template_kwargs={"drop_thinking": False, "tokenize": False, "add_generation_prompt": False})
    mk = ADAPTER.chat_to_request(req, tmp_path).model_kwargs
    assert [m["role"] for m in mk["messages"]] == ["system", "user", "assistant", "tool"]
    assert mk["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert mk["messages"][2]["reasoning_content"] == "Look it up."
    assert mk["tools"] == [WEATHER]
    # only the template's switches: tokenize=False would make it return a str
    assert mk["chat_template_kwargs"] == {"reasoning_effort": 40, "drop_thinking": False}
    assert not {"tool_choice", "reasoning_effort", "reply_prefix"} & set(mk)


SWITCHES = [
    ({}, {}),
    ({"enable_thinking": False}, {"enable_thinking": False}),
    ({"enable_thinking": None}, {}),
    ({"chat_template_kwargs": {"enable_thinking": None, "drop_thinking": None}}, {}),
    # the template's other names for its thinking mode, in its order of precedence
    ({"chat_template_kwargs": {"thinking": False}}, {"enable_thinking": False}),
    ({"thinking": False, "enable_thinking": True}, {"enable_thinking": False}),
    ({"thinking_mode": "chat", "thinking": True}, {"enable_thinking": False}),
    ({"chat_template_kwargs": {"thinking_mode": "thinking"}, "enable_thinking": False}, {"enable_thinking": True}),
    # chat_template_kwargs wins over a flat key
    ({"chat_template_kwargs": {"reasoning_effort": "max"}, "reasoning_effort": "low"}, {"reasoning_effort": "max"}),
    # OpenAI's levels, as DeepSeek's own template maps them onto the encoder's
    ({"reasoning_effort": "minimal"}, {"reasoning_effort": "low"}),
    ({"reasoning_effort": "medium"}, {"reasoning_effort": "high"}),
    ({"reasoning_effort": "xhigh"}, {"reasoning_effort": "high"}),
    ({"reasoning_effort": "none"}, {"enable_thinking": False}),
    ({"reasoning_effort": "none", "enable_thinking": True}, {"enable_thinking": True}),
    ({"reasoning_effort": 62}, {"reasoning_effort": 62}),
    # GLM's name, which /v1/messages sends for a chat that keeps every thinking block
    ({"clear_thinking": False}, {"drop_thinking": False}),
]


@pytest.mark.parametrize("fields, switches", SWITCHES, ids=[json.dumps(fields) for fields, _ in SWITCHES])
def test_every_spelling_of_a_switch_reaches_the_template_as_one(fields, switches, tmp_path):
    req = _chat(**fields)
    assert ADAPTER.chat_to_request(req, tmp_path).model_kwargs["chat_template_kwargs"] == switches
    # and the reply is split as the prompt was written: the reasoning first, unless thinking is off
    reply = _message(ADAPTER.make_output_parser(req), "Two plus two.</think>It is 4.")
    if switches.get("enable_thinking", True):
        assert (reply["reasoning_content"], reply["content"]) == ("Two plus two.", "It is 4.")
    else:
        assert reply == {"role": "assistant", "content": "Two plus two.</think>It is 4."}


@pytest.mark.parametrize("body", [
    {"tool_choice": "required"},
    {"tools": [WEATHER], "tool_choice": "any"},
    {"tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "get_time"}}},
    {"tools": [WEATHER], "tool_choice": {"type": "function", "name": "get_weather"}},
    {"tools": [WEATHER], "tool_choice": "required", "enable_thinking": True},
    {"tools": [WEATHER], "tool_choice": "required", "chat_template_kwargs": {"thinking_mode": "thinking"}},
    {"reply_prefix": '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'},
    {"tools": [WEATHER], "parallel_tool_calls": "no"},
    {"tools": [{"type": "function"}]},
    {"tools": "get_weather"},
    {"chat_template_kwargs": "thinking"},
    {"chat_template_kwargs": {"enable_thinking": "false"}},
    {"thinking": {"type": "disabled"}},
    {"thinking_mode": "off"},
    {"functions": [{"name": "get_weather"}]},
    {"function_call": {"name": "get_weather"}},
    {"messages": [{"role": "observation", "content": "x"}, {"role": "user", "content": "hi"}]},
    {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "tool_calls": [
        {"function": {"name": "f", "arguments": "{oops"}}]}]},
    {"response_format": {"type": "json_object"}},
], ids=",".join)
def test_the_adapter_refuses_what_it_cannot_honour(body, tmp_path):
    with pytest.raises(ValueError):
        ADAPTER.chat_to_request(_chat(**body), tmp_path)


def test_tool_choice_none_leaves_the_tools_out(tmp_path):
    req = _chat(tools=[WEATHER], tool_choice="none", enable_thinking=False)
    assert "tools" not in ADAPTER.chat_to_request(req, tmp_path).model_kwargs
    assert _message(ADAPTER.make_output_parser(req), "ok" + CALL)["content"] == "ok" + CALL


# ── a forced call ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", CASES["forced"], ids=lambda c: c["id"])
def test_a_forced_tool_choice_opens_the_turn_the_encoder_writes(tokenizer, case, tmp_path):
    """The prompt ends inside the call, in the template's chat mode: with its reply prefix it
    is the start of the encoder's own turn with that call, and what the model goes on to
    write parses as the call."""
    req = _chat(messages=case["messages"], tools=CASES["toolsets"][case["tools"]], tool_choice=case["tool_choice"])
    mk = ADAPTER.chat_to_request(req, tmp_path).model_kwargs
    assert mk["chat_template_kwargs"] == {"enable_thinking": False}
    prompt = _prompt(tokenizer, req, tmp_path)
    assert prompt.endswith(mk["reply_prefix"]) and case["turn"].startswith(prompt)
    parser = ADAPTER.make_output_parser(req)
    message = _message(parser, case["turn"][len(prompt):].removesuffix(EOS))
    (call,) = message["tool_calls"]
    name = case["id"].rsplit("-", 1)[1]  # the function the encoder's turn calls
    assert call["function"] == {"name": name, "arguments": '{"location": "Beijing"}' if name == "get_weather" else "{}"}
    assert message["content"] is None and "reasoning_content" not in message and parser.finish_reason == "tool_calls"


@pytest.mark.parametrize("case", CASES["forced"], ids=lambda c: c["id"])
def test_a_forced_calls_prefix_ends_on_a_token_the_model_writes(case, tmp_path):
    """With DeepSeek's tokenizer: the prompt and the prefix, tokenized apart as the renderer
    does, are the first ids of the turn, so the model goes on from a boundary it knows."""
    tok = _checkpoint_tokenizer()
    if tok is None:
        pytest.skip("DeepSeek-V4.1's tokenizer is not here (DSV41_REFERENCE_DIR or the Hugging Face cache)")
    req = _chat(messages=case["messages"], tools=CASES["toolsets"][case["tools"]], tool_choice=case["tool_choice"])
    prefix = ADAPTER.chat_to_request(req, tmp_path).model_kwargs["reply_prefix"]
    prompt = _prompt(byte_tokenizer(), req, tmp_path).removesuffix(prefix)
    ids = tok.encode(prompt, add_special_tokens=False) + tok.encode(prefix, add_special_tokens=False)
    assert tok.encode(case["turn"], add_special_tokens=False)[:len(ids)] == ids


@pytest.mark.parametrize("choice, tools, prefix, reply", [
    # one tool: it is named, and the reply starts at its arguments
    ("required", [WEATHER], '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n', ARGS),
    ("required", [WEATHER, TIME], '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="', 'get_weather">\n' + ARGS),
    ({"type": "function", "function": {"name": "get_weather"}}, [WEATHER, TIME],
     '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n', ARGS),
])
def test_a_forced_call_is_the_whole_reply(choice, tools, prefix, reply, tmp_path):
    req = _chat(tools=tools, tool_choice=choice)
    mk = ADAPTER.chat_to_request(req, tmp_path).model_kwargs
    assert mk["reply_prefix"] == prefix and mk["tools"] == tools
    parser = ADAPTER.make_output_parser(req)
    (call,) = _message(parser, reply)["tool_calls"]
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Paris", "days": 3}'}
    assert parser.finish_reason == "tool_calls"


def test_a_forced_call_to_a_made_up_function_stays_text():
    parser = ADAPTER.make_output_parser(_chat(tools=[WEATHER, TIME], tool_choice="required"))
    message = _message(parser, 'rm_rf">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    assert message["content"].startswith('<｜DSML｜ invoke name="rm_rf">') and parser.finish_reason == "stop"


# ── the reply ───────────────────────────────────────────────────────────────

def test_a_reply_splits_into_reasoning_content_and_calls():
    parser = ADAPTER.make_output_parser(_chat(tools=[WEATHER]))
    message = _message(parser, "Need the weather.</think>Checking." + CALL)
    assert message["reasoning_content"] == "Need the weather." and message["content"] == "Checking."
    (call,) = message["tool_calls"]
    assert call["id"].startswith("call_") and call["type"] == "function"
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Paris", "days": 3}'}
    assert parser.finish_reason == "tool_calls"


def test_parallel_tool_calls_false_ends_the_reply_at_its_first_call():
    parser = ADAPTER.make_output_parser(_chat(tools=[WEATHER], parallel_tool_calls=False, enable_thinking=False))
    two = CALL.removesuffix("</｜DSML｜ calls>") + '<｜DSML｜ invoke name="get_weather">\n' + ARGS
    assert len(_message(parser, two)["tool_calls"]) == 1 and parser.done


@pytest.mark.parametrize("case_id, thinking", [("content-then-call", True), ("typed-arguments", False)])
def test_a_parsed_turn_sent_back_renders_as_the_model_wrote_it(tokenizer, case_id, thinking, tmp_path):
    """An agent's next request extends this one's prompt: the reply the parser read, sent
    back as the assistant turn with its tool result, is the encoder's text again."""
    case = next(c for c in CASES["cases"] if c["id"] == case_id)
    system, user, _turn, result = case["messages"]
    tools = CASES["toolsets"][case["tools"]]
    opening = _chat(messages=[system, user], tools=tools, enable_thinking=thinking)
    start = _prompt(tokenizer, opening, tmp_path)
    assert case["prompt"].startswith(start)
    reply = case["prompt"][len(start):case["prompt"].index(EOS)]
    message = _message(ADAPTER.make_output_parser(opening), reply)
    (call,) = message["tool_calls"]
    req = _chat(messages=[system, user, message, {**result, "tool_call_id": call["id"]}], tools=tools,
                enable_thinking=thinking)
    assert _prompt(tokenizer, req, tmp_path) == case["prompt"]
