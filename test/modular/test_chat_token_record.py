"""/v1/chat/completions: the sampled tokens' log-probs and ids, and the prompt's ids,
for a chat that asked for them (``serving_chat._TokenRecord``). The Rust frontend's
``TokenRecord`` follows the same rules; test/rust/test_rust_openai_adapters.py
compares the two."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from mstar.api_server.openai.serving_chat import _build_response, _stream_events, _TokenRecord

USAGE = {"prompt_tokens": 3, "completion_tokens": 3, "finish_reason": "stop"}


def _chunk(modality, data, metadata=None):
    return SimpleNamespace(modality=modality, data=data, metadata=metadata or {})


def _reply():
    """"café" split inside the "é", then the stop token, whose text is empty."""
    return [
        _chunk("text", b"caf\xc3", {"logprobs": [-0.5], "token_ids": [10], "prompt_token_ids": [1, 2, 3]}),
        _chunk("text", b"\xa9", {"logprobs": [-0.25], "token_ids": [11]}),
        _chunk("text", b"", {"logprobs": [-0.125], "token_ids": [12]}),
        _chunk("usage", b"", USAGE),
    ]


def _entry(token, logprob, data):
    return {"token": token, "logprob": logprob, "bytes": None if data is None else list(data), "top_logprobs": []}


ENTRIES = [_entry("caf�", -0.5, b"caf\xc3"), _entry("�", -0.25, b"\xa9"), _entry("", -0.125, b"")]
ASKED = {"return_logprobs": True, "return_token_ids": True}


def test_a_reply_carries_one_entry_per_sampled_token_stop_token_included():
    """As many entries and ids as ``usage.completion_tokens`` counts, which is
    what a trainer (ART) checks before it trusts them."""
    body = _build_response("m", "r", _reply(), 24000, record=_TokenRecord(ASKED))
    (choice,) = body["choices"]
    assert choice["message"]["content"] == "café"
    assert choice["logprobs"] == {"content": ENTRIES}
    assert choice["token_ids"] == [10, 11, 12]
    assert body["prompt_token_ids"] == [1, 2, 3]
    assert len(choice["logprobs"]["content"]) == body["usage"]["completion_tokens"]


def test_a_chat_that_asked_for_nothing_gets_the_same_body_as_before():
    body = _build_response("m", "r", _reply(), 24000, record=_TokenRecord({}))
    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}
    assert set(body["choices"][0]) == {"index", "message", "finish_reason"}


def test_each_ask_adds_only_its_own_fields():
    logprobs = _build_response("m", "r", _reply(), 24000, record=_TokenRecord({"return_logprobs": True}))
    assert "prompt_token_ids" not in logprobs and "token_ids" not in logprobs["choices"][0]
    ids = _build_response("m", "r", _reply(), 24000, record=_TokenRecord({"return_token_ids": True}))
    assert "logprobs" not in ids["choices"][0] and ids["prompt_token_ids"] == [1, 2, 3]


def _stream(chunks, record):
    async def batches():
        for c in chunks:
            yield [c]

    async def run():
        return [e async for e in _stream_events(batches(), "m", "r", 24000, False, record)]

    events = [e.removeprefix("data: ").strip() for e in asyncio.run(run())]
    return [e if e == "[DONE]" else json.loads(e) for e in events]


def test_a_stream_sends_a_held_back_token_s_entry_with_the_next_chunk():
    """The first chunk holds back half the "é", so its entry goes out with the
    "é"; the stop token's goes out with the finish chunk."""
    first, second, finish, done = _stream(_reply(), _TokenRecord(ASKED))
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "caf"}
    assert first["choices"][0]["logprobs"] == {"content": ENTRIES[:1]}
    assert first["choices"][0]["token_ids"] == [10] and first["prompt_token_ids"] == [1, 2, 3]
    assert second["choices"][0]["delta"] == {"content": "é"}
    assert second["choices"][0]["logprobs"] == {"content": ENTRIES[1:2]} and "prompt_token_ids" not in second
    assert finish["choices"][0]["finish_reason"] == "stop"
    assert finish["choices"][0]["logprobs"] == {"content": ENTRIES[2:]}
    assert finish["choices"][0]["token_ids"] == [12]
    assert done == "[DONE]"


def test_a_stream_that_asked_for_nothing_sends_no_token_fields():
    for event in _stream(_reply(), _TokenRecord(None))[:-1]:
        assert "prompt_token_ids" not in event
        assert not {"logprobs", "token_ids"} & set(event["choices"][0])


def test_a_stream_ending_inside_a_character_still_sends_every_entry():
    """A ``max_tokens`` cut can end a reply on half a character: the flushed tail
    carries the held token's entry (with the finish here; the Rust frontend sends
    the tail on its own chunk first, as on main), so none is lost."""
    record = _TokenRecord({"return_logprobs": True})
    chunks = [_chunk("text", b"a", {"logprobs": [-0.5]}), _chunk("text", b"\xe2\x88", {"logprobs": [-0.25]}),
              _chunk("usage", b"", {"prompt_tokens": 1, "completion_tokens": 2, "finish_reason": "length"})]
    first, last, done = _stream(chunks, record)
    assert first["choices"][0]["logprobs"] == {"content": [_entry("a", -0.5, b"a")]}
    assert last["choices"][0]["delta"] == {"content": "�"} and last["choices"][0]["finish_reason"] == "length"
    assert last["choices"][0]["logprobs"] == {"content": [_entry("�", -0.25, b"\xe2\x88")]}
    assert done == "[DONE]"
