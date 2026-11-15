"""The chat reply parser: however a reply's bytes arrive in chunks, its reasoning and its answer
come out the same, and as Qwen's chat template splits a reply when it renders it back."""

import pytest

from mstar.api_server.reply_parser import ReplyParser

END = "</think>"


def _template_split(reply: str) -> tuple[str, str]:
    """Qwen3.5's chat template, rendering an assistant turn back: the text before ``</think>`` is
    the reasoning (newlines around it stripped), the text after it the answer (opening newlines
    stripped)."""
    if END not in reply:
        return "", reply
    reasoning, answer = reply.split(END, 1)
    return reasoning.rstrip("\n").lstrip("\n"), answer.lstrip("\n")


def _feed(chunks: list[bytes], starts_in_reasoning: bool = True) -> tuple[str, str]:
    parser = ReplyParser(END, starts_in_reasoning)
    reasoning, answer = [], []
    for chunk in chunks:
        r, a = parser.feed(chunk)
        reasoning.append(r)
        answer.append(a)
    r, a = parser.finish()
    return "".join(reasoning) + r, "".join(answer) + a


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
    data = reply.encode()
    want = _template_split(reply)
    assert _feed([data]) == want
    for cut in range(1, len(data)):
        assert _feed([data[:cut], data[cut:]]) == want, cut
    assert _feed([data[i:i + 1] for i in range(len(data))]) == want


def test_a_reply_cut_inside_its_reasoning_is_all_reasoning():
    """``max_tokens`` can end a reply before its reasoning closes: the client gets it as
    reasoning, held text included, and no answer."""
    reply = "still thinking about it\n</thi"
    assert _feed([reply.encode()]) == (reply, "")


def test_a_reply_that_starts_past_its_reasoning_passes_through():
    """Thinking off: the prompt already closed the reasoning, so the whole reply is the answer,
    even text that looks like the marker."""
    reply = "an answer that writes </think> in it"
    assert _feed([reply.encode()], starts_in_reasoning=False) == ("", reply)
