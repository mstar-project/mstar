"""Tool calls in Qwen's markup (``qwen3_coder``: Qwen3-Coder, Qwen3-Coder-Next, Qwen3.5 to
Qwen3.8): the request reaches the chat template with its tools, calls and results, and the
reply's calls come back as OpenAI tool calls, streamed or whole.

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

from mstar.api_server.openai import adapters  # noqa: E402
from mstar.api_server.openai.protocol import ChatCompletionRequest  # noqa: E402
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


# --------------------------------------------------------------------------
# the request: adapters
# --------------------------------------------------------------------------

def _chat(model="qwen3_coder_30b_a3b", **body):
    body.setdefault("messages", [{"role": "system", "content": "Be brief."},
                                 {"role": "user", "content": "weather in Paris?"}])
    return ChatCompletionRequest(model=model, **body)


AGENT_TURN = [
    {"role": "user", "content": "weather in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
     "function": {"name": "get_weather", "arguments": '{"city": "Paris", "days": 3}'}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "sunny"},
                                                           {"type": "text", "text": "22C"}]},
]


@pytest.mark.parametrize("model", ["qwen3_coder_30b_a3b", "qwen3_5_0.8b", "qwen3_8_27b"])
def test_the_adapters_send_the_agent_turn_to_the_template(model, tmp_path):
    adapter = adapters.get_adapter(model)
    assert adapter.supports_messages and adapter.supports_responses
    mk = adapter.chat_to_request(_chat(model, tools=[WEATHER], messages=AGENT_TURN), tmp_path).model_kwargs
    assert [m["role"] for m in mk["messages"]] == ["user", "assistant", "tool"]
    assert mk["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris", "days": 3}
    # the templates concatenate content: it goes as text
    assert mk["messages"][1]["content"] == "" and mk["messages"][2]["content"] == "sunny\n22C"
    assert mk["tools"] == [WEATHER] and "tool_choice" not in mk and "reply_prefix" not in mk


def test_qwen3_coder_does_not_reason_and_qwen3_5_reasons_unless_told_not_to(tmp_path):
    coder = adapters.get_adapter("qwen3_coder_30b_a3b").make_output_parser(_chat(tools=[WEATHER]))
    assert coder.message("<think>x</think>Hi") == {"role": "assistant", "content": "<think>x</think>Hi"}
    q35 = adapters.get_adapter("qwen3_5_0.8b")
    assert q35.make_output_parser(_chat("qwen3_5_0.8b")).message("hmm\n</think>\n\nHi") == {
        "role": "assistant", "content": "Hi", "reasoning_content": "hmm\n"}
    req = _chat("qwen3_5_0.8b", enable_thinking=False)
    assert q35.chat_to_request(req, tmp_path).model_kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    # nothing to split out of a reply with no tools and no reasoning: no parser, the text is the answer
    assert q35.make_output_parser(req) is None


@pytest.mark.parametrize("model, choice, tools, prefix, reply", [
    ("qwen3_coder_30b_a3b", "required", [WEATHER], "<tool_call>\n<function=get_weather>\n",
     "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
    ("qwen3_coder_30b_a3b", "required", [WEATHER, LS], "<tool_call>\n", CALL[len("<tool_call>\n"):]),
    ("qwen3_5_0.8b", {"type": "function", "function": {"name": "get_weather"}}, [WEATHER, LS],
     "<tool_call>\n<function=get_weather>\n", "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
])
def test_a_forced_tool_choice_opens_the_call_in_the_prompt(model, choice, tools, prefix, reply, tmp_path):
    adapter = adapters.get_adapter(model)
    req = _chat(model, tools=tools, tool_choice=choice)
    mk = adapter.chat_to_request(req, tmp_path).model_kwargs
    assert mk["reply_prefix"] == prefix and mk["tools"] == tools
    # Qwen3.5's prompt ends as a turn that does not reason
    assert mk["chat_template_kwargs"].get("enable_thinking") is (False if model == "qwen3_5_0.8b" else None)
    p = adapter.make_output_parser(req)
    msg = p.message(reply)
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather" and p.finish_reason == "tool_calls"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"])["city"] == "Paris"


@pytest.mark.parametrize("body", [
    {"tools": [WEATHER], "tool_choice": "required", "enable_thinking": True},
    {"tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "nope"}}},
    {"tool_choice": "required"},
    {"functions": [{"name": "get_weather"}]},
    {"function_call": {"name": "get_weather"}},
    {"reply_prefix": "<tool_call>"},
    {"messages": [{"role": "function", "content": "x"}, {"role": "user", "content": "hi"}]},
])
def test_qwen3_5_refuses_what_it_cannot_honour(body, tmp_path):
    with pytest.raises(ValueError):
        adapters.get_adapter("qwen3_5_0.8b").chat_to_request(_chat("qwen3_5_0.8b", **body), tmp_path)


def test_attachments_are_placeholders_where_the_message_has_them(tmp_path):
    from mstar.model.multimodal import messages_from_parts

    png = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhK"
           "mMIQAAAABJRU5ErkJggg==")
    req = _chat("qwen3_5_0.8b", tools=[WEATHER], messages=[{"role": "user", "content": [
        {"type": "text", "text": "what is"}, {"type": "image_url", "image_url": {"url": png}},
        {"type": "text", "text": "this?"}, {"type": "text", "text": "and here?"}]}])
    args = adapters.get_adapter("qwen3_5_0.8b").chat_to_request(req, tmp_path)
    # what the parts path would have rendered
    assert args.model_kwargs["messages"] == messages_from_parts(args.prompt_parts)
    assert len(args.file_paths["image"]) == 1


def test_qwen3_8_maps_reasoning_effort_onto_its_levels():
    from mstar.api_server.openai.serving_responses import _reasoning

    efforts = adapters.get_adapter("qwen3_8_27b").reasoning_efforts
    assert [_reasoning({"effort": e}, efforts) for e in ("low", "medium", "high")] == [
        {"reasoning_effort": "low"}, {"reasoning_effort": "medium"}, {"reasoning_effort": "xhigh"}]
    # no levels: a hint, dropped; none turns thinking off
    assert _reasoning({"effort": "high"}, ()) == {} and _reasoning({"effort": "none"}, ()) == {"enable_thinking": False}


# --------------------------------------------------------------------------
# the request: the models' prompts, through the vendored templates
# --------------------------------------------------------------------------

class _TemplateTokenizer:
    """A tokenizer whose chat template is a real one and whose ids are the text's bytes."""

    def __init__(self, template):
        self.chat_template, self.calls = template, []

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=True,
                            return_dict=False, **kw):
        self.calls.append(kw)
        text = _render(self.chat_template, messages, tools=tools, add_generation_prompt=add_generation_prompt, **kw)
        return list(text.encode()) if tokenize else text

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
        return list(text.encode())


# Qwen3-Coder-30B-A3B-Instruct's config.json, the fields the text LM reads, layers cut to 2
_CODER_CONFIG = {
    "model_type": "qwen3_moe", "attention_bias": False, "eos_token_id": 151645, "head_dim": 128,
    "hidden_size": 2048, "intermediate_size": 6144, "max_position_embeddings": 262144, "mlp_only_layers": [],
    "moe_intermediate_size": 768, "norm_topk_prob": True, "num_attention_heads": 32, "num_experts": 128,
    "num_experts_per_tok": 8, "num_hidden_layers": 2, "num_key_value_heads": 4, "decoder_sparse_step": 1,
    "rms_norm_eps": 1e-06, "rope_scaling": None, "rope_theta": 10000000, "tie_word_embeddings": False,
    "use_sliding_window": False, "vocab_size": 151936,
}


def _text_lm(tmp_path, template):
    from mstar.model.text_lm.text_lm_model import TextLMModel

    (tmp_path / "config.json").write_text(json.dumps(_CODER_CONFIG))
    model = TextLMModel(model_path_hf=str(tmp_path))
    model._tokenizer = _TemplateTokenizer(template)
    return model


def _prompt(model, mk) -> str:
    out = model.process_prompt("ignored", ["text"], ["text"], **mk)
    return bytes(out.new_input_tensors["text_inputs"][0].tolist()).decode()


def test_text_lm_renders_tools_calls_and_results_through_the_template(tmp_path):
    pytest.importorskip("transformers")
    model = _text_lm(tmp_path, TEMPLATES["qwen3_coder"])
    adapter = adapters.get_adapter("qwen3_coder_30b_a3b")
    mk = adapter.chat_to_request(_chat(tools=[WEATHER], messages=AGENT_TURN), tmp_path).model_kwargs
    prompt = _prompt(model, mk)
    assert "<tools>\n<function>\n<name>get_weather</name>" in prompt
    assert ("<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>") in prompt
    assert prompt.endswith("<tool_response>\nsunny\n22C\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n")
    # a forced call ends the prompt inside it
    mk = adapter.chat_to_request(_chat(tools=[WEATHER], tool_choice="required"), tmp_path).model_kwargs
    assert _prompt(model, mk).endswith("<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n")


def test_text_lm_renders_a_plain_chat_as_before(tmp_path):
    pytest.importorskip("transformers")
    model = _text_lm(tmp_path, TEMPLATES["qwen3_coder"])
    req = _chat(messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "bye"}])
    args = adapters.get_adapter("qwen3_coder_30b_a3b").chat_to_request(req, tmp_path)
    with_messages = _prompt(model, args.model_kwargs)
    out = model.process_prompt(args.text, ["text"], ["text"], prompt_parts=args.prompt_parts)
    assert bytes(out.new_input_tensors["text_inputs"][0].tolist()).decode() == with_messages


def test_the_chat_stays_in_the_data_worker(tmp_path):
    from mstar.communication import wire
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    from mstar.model.text_lm.text_lm_model import TextLMModel
    from mstar.utils.ipc_format import ConductorMessage, ConductorMessageType, NewRequestConductor

    tool = {"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"n": {"type": "integer", "maximum": 2**64}}}}}
    for model, cls in (("qwen3_coder_30b_a3b", TextLMModel), ("qwen3_5_0.8b", Qwen3_5DenseModel)):
        mk = adapters.get_adapter(model).chat_to_request(_chat(model, tools=[tool], tool_choice="required"),
                                                         tmp_path).model_kwargs
        # what the data worker sends the conductor: the wire carries no int past 64 bits
        sent = {k: v for k, v in mk.items() if k not in cls.PREPROCESS_ONLY_KWARGS}
        chat = {"messages", "tools", "reply_prefix"}
        assert chat <= set(mk) and not chat & set(sent)
        body = NewRequestConductor("r", {}, ["text"], ["text"], {}, sent)
        wire.encode(ConductorMessage(message_type=ConductorMessageType.NEW_REQUEST, body=body))


# --------------------------------------------------------------------------
# with the checkpoints' own tokenizers, where cached (laude; CI has none)
# --------------------------------------------------------------------------

def _cached(repo):
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo, allow_patterns=["*.json", "*.jinja", "*.txt"], local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached: skip
        return None


def test_qwen3_coder_tokenizer_renders_a_plain_chat_alike_both_ways(tmp_path):
    path = _cached("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    if path is None:
        pytest.skip("Qwen3-Coder-30B-A3B-Instruct's tokenizer is not cached")
    from mstar.model.text_lm.text_lm_model import TextLMModel

    model = TextLMModel(model_path_hf=path)
    adapter = adapters.get_adapter("qwen3_coder_30b_a3b")
    req = _chat(messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi é"},
                          {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "bye"}])
    args = adapter.chat_to_request(req, tmp_path)
    ids = model.process_prompt(args.text, ["text"], ["text"], prompt_parts=args.prompt_parts).new_input_tensors
    assert model.process_prompt(None, ["text"], ["text"], **args.model_kwargs).new_input_tensors[
        "text_inputs"][0].tolist() == ids["text_inputs"][0].tolist()
    # the forced call's prefix tokenizes as the model writes it
    mk = adapter.chat_to_request(_chat(tools=[WEATHER], tool_choice="required"), tmp_path).model_kwargs
    got = model.process_prompt(None, ["text"], ["text"], **mk).new_input_tensors["text_inputs"][0].tolist()
    whole = model.tokenizer.apply_chat_template(
        [*mk["messages"], {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "x"}}}]}],
        tools=[WEATHER], tokenize=True, return_dict=False)
    assert got == whole[:len(got)]


def test_qwen3_5_tokenizer_renders_a_chat_with_an_image_alike_both_ways(tmp_path):
    torch = pytest.importorskip("torch")
    if _cached("Qwen/Qwen3.5-0.8B") is None:
        pytest.skip("Qwen3.5-0.8B's tokenizer is not cached")
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    model = Qwen3_5DenseModel("Qwen/Qwen3.5-0.8B")
    png = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhK"
           "mMIQAAAABJRU5ErkJggg==")
    req = _chat("qwen3_5_0.8b", messages=[
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [
            {"type": "text", "text": "what is"}, {"type": "image_url", "image_url": {"url": png}},
            {"type": "text", "text": "it?"}]},
        {"role": "assistant", "content": "A dot."}, {"role": "user", "content": "sure?"}])
    args = adapters.get_adapter("qwen3_5_0.8b").chat_to_request(req, tmp_path)
    image = torch.zeros(3, 64, 64, dtype=torch.uint8)

    def ids(**kw):
        out = model.process_prompt(args.text, args.input_modalities, ["text"], tensors={"image_inputs": [image]},
                                   prompt_parts=args.prompt_parts, **kw)
        return [t.tolist() for t in out.new_input_tensors["text_inputs"]]

    plain = {k: v for k, v in args.model_kwargs.items() if k not in model.PREPROCESS_ONLY_KWARGS}
    assert ids(**args.model_kwargs) == ids(**plain)
