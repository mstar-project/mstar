"""The chat reply parser: however a reply's bytes arrive in chunks, its reasoning, its answer and
its tool calls come out the same, and as Qwen's chat template splits a reply when it renders it back."""

import json

import pytest

from mstar.api_server.reply_parser import TOOL_FORMATS, ReplyParser, ToolCallFormat

END = "</think>"
QWEN = TOOL_FORMATS["qwen_3_5"]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "zip": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
    "hours": {"type": "array"}, "where": {"type": "object"}}}}}]


def _template_split(reply: str) -> tuple[str, str]:
    """Qwen3.5's chat template, rendering an assistant turn back: the text before ``</think>`` is
    the reasoning (newlines around it stripped), the text after it the answer (opening newlines
    stripped)."""
    if END not in reply:
        return "", reply
    reasoning, answer = reply.split(END, 1)
    return reasoning.rstrip("\n").lstrip("\n"), answer.lstrip("\n")


def _feed(chunks: list[bytes], starts_in_reasoning: bool = True, tools=None):
    parser = ReplyParser(END, starts_in_reasoning, QWEN if tools else None, tools)
    reasoning, answer, calls = [], [], []
    for chunk in chunks:
        r, a, c = parser.feed(chunk)
        reasoning.append(r)
        answer.append(a)
        calls += c
    r, a, c = parser.finish()
    calls = [(c["index"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls + c]
    return "".join(reasoning) + r, "".join(answer) + a, calls


def _splits(data: bytes):
    yield [data]
    for cut in range(1, len(data)):
        yield [data[:cut], data[cut:]]
    yield [data[i:i + 1] for i in range(len(data))]


REPLIES = [
    "The user asks for France's capital.\nThat's Paris.\n</think>\n\nThe capital of France is Paris.",
    "short\n</think>\n\n{\"name\": \"Ada\", \"age\": 36}",
    "no newline</think>answer right away",
    "reasoning mentions </thin and <think> but goes on\n\n</think>\n\n\nanswer after three newlines\n",
    "unicode: café, 東京, 🚀\n</think>\n\nanswer: naïve ✓",
    "</think>\n\nempty reasoning",
    "\n\nreasoning that opens with newlines\n</think>\n\nanswer",
]


@pytest.mark.parametrize("reply", REPLIES)
def test_every_split_of_a_reply_parses_as_the_whole_reply(reply):
    want = (*_template_split(reply), [])
    for chunks in _splits(reply.encode()):
        assert _feed(chunks) == want, chunks
        assert _feed(chunks, tools=TOOLS) == want, chunks


def test_a_reply_cut_inside_its_reasoning_is_all_reasoning():
    """``max_tokens`` can end a reply before its reasoning closes: the client gets it as
    reasoning, held text included, and no answer."""
    reply = "still thinking about it\n</thi"
    assert _feed([reply.encode()]) == (reply, "", [])


def test_a_reply_that_starts_past_its_reasoning_passes_through():
    """Thinking off: the prompt already closed the reasoning, so the whole reply is the answer,
    even text that looks like the marker."""
    reply = "an answer that writes </think> in it"
    assert _feed([reply.encode()], starts_in_reasoning=False) == ("", reply, [])


def _call(**params) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function=get_weather>\n{body}</function>\n</tool_call>"


TOOL_REPLIES = [
    # (reply, answer, calls): a call after the reasoning, typed by its schema
    ("I'll check.\n</think>\n\n" + _call(city="Paris", zip="94110", days="3", metric="True",
                                          hours="[9, 12]", where='{"x": 1}'),
     "", [(0, "get_weather", {"city": "Paris", "zip": "94110", "days": 3, "metric": True, "hours": [9, 12],
                              "where": {"x": 1}})]),
    # the answer's text, the template's blank line, then two calls
    ("x</think>\n\nLet me look.\n\n" + _call(city="Paris") + "\n" + _call(city="Rome"),
     "Let me look.", [(0, "get_weather", {"city": "Paris"}), (1, "get_weather", {"city": "Rome"})]),
    # text after a call stays the answer
    ("x</think>\n\n" + _call(city="Paris") + "\nDone.", "Done.", [(0, "get_weather", {"city": "Paris"})]),
    # a value that isn't its type stays as written; a parameter with no schema parses if it can
    ("x</think>\n\n" + _call(days="three", extra="7"), "", [(0, "get_weather", {"days": "three", "extra": 7})]),
    # a call in the reasoning isn't one
    ("thinking " + _call(city="Paris") + "\n</think>\n\nNo call.", "No call.", []),
    # a call that never closes comes out as written (max_tokens cut it)
    ("x</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
     "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar", []),
    # a block whose name doesn't parse isn't a call
    ("x</think>\n\n<tool_call>\n<function=\n</function>\n</tool_call>",
     "<tool_call>\n<function=\n</function>\n</tool_call>", []),
]


@pytest.mark.parametrize(("reply", "answer", "calls"), TOOL_REPLIES)
def test_every_split_of_a_reply_with_calls_parses_as_the_whole_reply(reply, answer, calls):
    reasoning = reply.split(END, 1)[0].rstrip("\n")
    for chunks in _splits(reply.encode()):
        assert _feed(chunks, tools=TOOLS) == (reasoning, answer, calls), chunks


def test_with_no_tools_a_call_is_left_as_text():
    """``tool_choice`` none (or no tools): the reply is returned as written, a call included."""
    reply = "x</think>\n\n" + _call(city="Paris")
    assert _feed([reply.encode()]) == ("x", _call(city="Paris"), [])


def test_qwen3_5s_call_format_is_xgrammars():
    """The markers the parser reads are the ones a forced call's grammar holds a reply to."""
    xgr = pytest.importorskip("xgrammar")
    tag = json.loads(xgr.get_model_structural_tag("qwen_3_5", tools=TOOLS, tool_choice="required").model_dump_json())
    calls = next(e for e in tag["format"]["elements"] if e["type"] in ("triggered_tags", "tags_with_separator"))
    (call,) = calls["tags"] if "tags" in calls else calls["tag"]
    assert call["begin"] == QWEN.begin + "get_weather" + QWEN.name_end
    assert call["end"].lstrip("\n") == QWEN.end


@pytest.mark.parametrize("parameters", [
    {"type": "object", "properties": {"city": "string", "days": "integer"}},
    {"type": "object", "properties": ["city", "days"]},
    {"type": "object", "properties": None},
    None,
], ids=["property-not-an-object", "properties-a-list", "properties-null", "no-parameters"])
def test_a_tool_whose_schema_is_malformed_is_parsed_as_one_with_none(parameters):
    """The client wrote the schema, and the parser runs on the thread every request shares: a
    schema it can't read types nothing, and raises nothing."""
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": parameters}}]
    reply = "x</think>\n\n" + _call(city="Paris", days="3")
    assert _feed([reply.encode()], tools=tools) == ("x", "", [(0, "get_weather", {"city": "Paris", "days": 3})])


def test_nan_and_infinity_stay_text():
    """Python's JSON reads them, strict JSON parsers don't: the arguments stay strict JSON."""
    reply = "x</think>\n\n" + _call(days="NaN", extra="-Infinity")
    (call,) = ReplyParser(END, True, QWEN, TOOLS).feed(reply.encode())[2]
    assert call["function"]["arguments"] == '{"days": "NaN", "extra": "-Infinity"}'


class _JsonFormat(ToolCallFormat):
    """A call written as JSON between its markers, as Qwen2.5's template writes one."""

    begin, end = "<tool_call>\n", "\n</tool_call>"

    def parse(self, block, schemas):
        call = json.loads(block)
        return call["name"], call["arguments"]


def test_another_format_needs_only_its_markers_and_its_parse():
    """The streaming (markers held across chunks, newlines dropped, text around the call kept) is
    the parser's; a format says only where a call starts and ends and how to read it."""
    call = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    reply = f"x</think>\n\nChecking.\n\n<tool_call>\n{call}\n</tool_call>"
    for chunks in _splits(reply.encode()):
        parser = ReplyParser(END, True, _JsonFormat(), TOOLS)
        out = [parser.feed(c) for c in chunks] + [parser.finish()]
        answer = "".join(a for _, a, _ in out)
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for _, _, cs in out for c in cs]
        assert (answer, calls) == ("Checking.", [("get_weather", {"city": "Paris"})]), chunks
