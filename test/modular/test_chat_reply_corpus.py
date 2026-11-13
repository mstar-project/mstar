"""The reply corpus (``chat_formats/replies.json``): replies as models write them, per tool-call
format, with the reasoning, content, calls and finish the parser must make of them, malformed
and cut-off replies included. Each is fed whole and split at every byte, through the
``Utf8Stream`` the reply stream decodes with. The corpus is the formats' spec for any parser of
them, so a port in another language checks itself against the same file."""

import json
import random
from pathlib import Path

import pytest

from mstar.api_server.openai import glm_chat, tool_calls
from mstar.utils.utf8_stream import Utf8Stream

CORPUS = json.loads((Path(__file__).parent / "chat_formats" / "replies.json").read_text())


def _cases(fmt=None):
    cases = [c for c in CORPUS["cases"] if fmt in (None, c["format"])]
    return pytest.mark.parametrize("case", cases, ids=lambda c: f"{c['format']}/{c['id']}")


def _tools(case):
    tools = CORPUS["toolsets"][case["tools"]] if case["tools"] else None
    return glm_chat.tool_parameters(tools) if tools else None


def _parser(case):
    fmt, tools, forced = tool_calls.CALL_FORMATS[case["format"]], _tools(case), case["forced_call"]
    prefix = "" if forced is None else fmt.forced_prefix(forced)
    return tool_calls.ReplyParser(fmt, thinking=case["thinking"], tools=tools, reply_prefix=prefix,
                                  single_call=case["single_call"])


def _glm_parser(case):
    return glm_chat.GlmReplyParser(thinking=case["thinking"], tools=_tools(case), forced_call=case["forced_call"],
                                   single_call=case["single_call"])


def _feeds(parser, pieces) -> list[list[dict]]:
    """What each feed of the reply's bytes sends, then what the finish does; call ids,
    which are random, left out."""
    stream, feeds = Utf8Stream(), []
    for piece in pieces:
        # an empty feed between pieces (a chunk of tokens with no text yet) changes nothing
        feeds.append(parser.feed(stream.feed(piece)) + parser.feed(""))
    feeds.append(parser.feed(stream.flush()) + parser.finish())
    assert parser.finish() == []  # finishing again sends nothing
    for d in (d for feed in feeds for d in feed):
        for call in d.get("tool_calls", ()):
            if "id" in call:
                assert call.pop("id").startswith("call_")
                call["first"] = True
    return feeds


def _reply(parser, pieces) -> dict:
    """The reply the deltas add up to, checking the delta protocol on the way: one call per
    delta, its name on the first, then only arguments, indices in order."""
    deltas = [d for feed in _feeds(parser, pieces) for d in feed]
    calls: list[dict] = []
    for d in deltas:
        if "tool_calls" not in d:
            continue
        (call,) = d["tool_calls"]
        fn = call["function"]
        if "first" in call:
            assert call["index"] == len(calls) and fn["name"] and call["type"] == "function"
            calls.append({"name": fn["name"], "arguments": fn["arguments"]})
        else:
            assert call["index"] == len(calls) - 1 and set(fn) == {"arguments"}
            calls[-1]["arguments"] += fn["arguments"]
    return {"reasoning_content": "".join(d.get("reasoning_content", "") for d in deltas),
            "content": "".join(d.get("content", "") for d in deltas),
            "tool_calls": calls, "finish_reason": parser.finish_reason}


def _splits(data: bytes):
    yield [data]
    for i in range(1, len(data)):
        yield [data[:i], data[i:]]


@_cases()
def test_a_reply_parses_as_the_corpus_says_whole_and_split_at_every_byte(case):
    for pieces in _splits(case["text"].encode()):
        assert _reply(_parser(case), pieces) == case["expect"], len(pieces[0])


@_cases("glm47")
def test_glm_s_own_parser_sends_the_same_deltas_feed_by_feed(case):
    """GLM serves through ``glm_chat.GlmReplyParser``; the shared parser is the same state
    machine, so a change to one that the other lacks fails here."""
    for pieces in _splits(case["text"].encode()):
        assert _feeds(_parser(case), pieces) == _feeds(_glm_parser(case), pieces), len(pieces[0])


def test_the_corpus_covers_every_format():
    assert {c["format"] for c in CORPUS["cases"]} == set(tool_calls.CALL_FORMATS)


@pytest.mark.parametrize("fmt", tool_calls.CALL_FORMATS.values(), ids=lambda f: f.name)
def test_any_text_parses_alike_in_any_chunking_and_whole_calls_are_json(fmt):
    """Seeded fuzz over the format's tags, cut tags and text: chunking never changes the
    reply, and a reply whose calls all closed has JSON arguments."""
    tools = glm_chat.tool_parameters([
        {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
            "s": {"type": "string"}, "n": {"type": "integer"}, "o": {}}}}},
        {"type": "function", "function": {"name": "g", "parameters": {}}}])
    tags = [fmt.call_start, fmt.call_end, fmt.key_start, fmt.key_end, fmt.value_start, fmt.value_end]
    frags = ["<think>", "</think>", *tags, *(t[:len(t) // 2] for t in tags), *(t[len(t) // 2:] for t in tags),
             "f", "g", "h", "s", "n", "o", " ", "\n", "<", "</", ">", '"', "x", "1", '"q"', "é", "[1]"]

    def run(text, size, forced, **kw):
        prefix = "" if forced is None else fmt.forced_prefix(forced)
        return _reply(tool_calls.ReplyParser(fmt, tools=tools, reply_prefix=prefix, **kw),
                      [text[i:i + size].encode() for i in range(0, len(text), size)])

    rng = random.Random(0)
    for _ in range(1500):
        text = "".join(rng.choice(frags) for _ in range(rng.randint(0, 40)))
        kw = {"thinking": rng.random() < 0.5, "single_call": rng.random() < 0.3,
              "forced": rng.choice([None, None, "", "f", "g"])}
        whole = run(text, max(1, len(text)), **kw)
        for size in (1, 2, 3, 7):
            assert run(text, size, **kw) == whole, (text, size, kw)
        if whole["finish_reason"] == "tool_calls":
            for call in whole["tool_calls"]:
                json.loads(call["arguments"])
