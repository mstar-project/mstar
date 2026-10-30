"""A chat reply as protocol-neutral events, which each API writes in its own
shape: chat completions (``serving_chat``), Anthropic messages
(``serving_messages``) and responses (``serving_responses``).

``reply_events`` reads a request's result chunks and yields, per batch the
server hands out, a list of events:

* a delta, as the output parser writes them: ``{"reasoning_content": str}``,
  ``{"content": str}`` or ``{"tool_calls": [call]}``, where a call's first
  delta carries its ``id`` and ``function.name`` and later ones only
  ``function.arguments`` to append;
* :class:`Media` for an audio or image chunk, in a list of its own;
* :class:`Failure` when the request failed after it was submitted;
* :class:`Finish`, last, with why the reply ended and its usage.

A failed reply has no ``Finish``. A whole reply is ``collect``-ed, and
``runs`` puts its deltas together.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from fastapi import HTTPException

from mstar.utils.utf8_stream import Utf8Stream

# how often a non-streaming handler checks whether its client is still there
_DISCONNECT_POLL_S = 0.05


@dataclass
class Usage:
    """Token counts and why the reply ended, from the text chunks' metadata
    (see the data worker's ``_text_usage``)."""

    prompt: int = 0
    completion: int = 0
    stopped: bool = True
    # of ``completion``, the tokens of the chunks that began inside the
    # reasoning (the one that closes it too; under MTP a chunk can straddle)
    reasoning: int = 0

    def add(self, chunk) -> None:
        meta = chunk.metadata or {}
        self.prompt = meta.get("prompt_tokens", self.prompt)
        self.completion += meta.get("tokens", 0)
        if "stop_token" in meta:
            self.stopped = meta["stop_token"]


@dataclass
class Media:
    modality: str  # "audio" (16-bit PCM at the model rate) or "image" (PNG)
    data: bytes


@dataclass
class Failure:
    message: str
    status: int


@dataclass
class Finish:
    """``reason`` is ``stop`` (a stop token or ``stop_string``), ``length``
    (the token budget) or ``tool_calls`` (the reply made calls, all whole)."""

    reason: str
    usage: Usage
    stop_string: str | None = None


@dataclass
class Run:
    """A stretch of one kind in a whole reply: ``reasoning`` or ``content``
    text, or a ``call`` with its ``id``, ``name`` and ``arguments``."""

    kind: str
    text: str = ""
    call: dict = field(default_factory=dict)


class StopMatcher:
    """Ends a text at the first of ``stops`` to appear in it. ``feed`` holds
    back the end that may be the start of a stop (at most the longest stop
    less one character) and returns the text that may go out, with the stop
    once one has appeared; the text from the stop on never goes out."""

    def __init__(self, stops: list[str]):
        self._stops = stops
        self._held = ""

    def feed(self, text: str) -> tuple[str, str | None]:
        text = self._held + text
        # the stop that completes first ends the text; of two that end
        # together, the longer started first
        hits = [(i + len(s), -len(s), s) for s in self._stops if (i := text.find(s)) >= 0]
        if hits:
            end, neg_len, stop = min(hits)
            self._held = ""
            return text[:end + neg_len], stop
        keep = max(_started(text, s) for s in self._stops)
        self._held = text[len(text) - keep:]
        return text[:len(text) - keep], None

    def flush(self) -> str:
        held, self._held = self._held, ""
        return held


def _started(text: str, stop: str) -> int:
    """Length of the longest end of ``text`` that begins ``stop``, short of all of it."""
    for k in range(min(len(stop) - 1, len(text)), 0, -1):
        if text.endswith(stop[:k]):
            return k
    return 0


def _passed(deltas: list[dict], matcher: StopMatcher | None, out: list) -> str | None:
    """Appends to ``out`` the deltas that may go out; returns the stop string
    that ended the reply, if one did. Stops match the answer (``content``)
    only: reasoning and tool calls go out whole. Held content goes out before
    a delta of another kind, so a stop matches within one stretch of content."""
    if matcher is None:
        out.extend(deltas)
        return None
    for d in deltas:
        content = d.get("content")
        if content is None:
            if held := matcher.flush():
                out.append({"content": held})
            out.append(d)
            continue
        text, stop = matcher.feed(content)
        if text:
            out.append({"content": text})
        if stop is not None:
            return stop
    return None


def finish_reason(parser, usage: Usage, stop_string: str | None = None) -> str:
    if parser is not None and parser.finish_reason == "tool_calls":
        return "tool_calls"
    return "stop" if stop_string is not None or usage.stopped else "length"


def reply_events(api, request_id: str, parser=None, stops: list[str] | None = None,
                 usage: Usage | None = None, batches=None, key=None):
    """The events of a request's reply, one list per batch of result chunks.
    Bound to the submission when called: a stream's body is iterated later,
    by Starlette.

    A stop string ends the reply where it starts, and a parser that is
    ``done`` (one call, under ``parallel_tool_calls: false``) ends it too:
    the request stops in the engine, and counts as a success, as the client
    has its whole reply. Closing the chunk iterator early otherwise, as a
    client that goes away does, aborts it.
    Usage counts every token read until then, the stop's own included, as
    vLLM counts them. It can count more: the tokens that arrived in one chunk
    with the stop's last (up to k under MTP, k drafts a step), and those read
    while the parser held the stop's end, when it may start a tag, or is the
    answer's trailing whitespace (held until text follows it).
    ``usage``, when given, is filled as the chunks arrive (a stream that opens
    with the prompt's count reads it from there). ``batches`` and ``key``, when
    given, are the request's chunk batches already opened
    (``api.open_result_stream``) and its submission, named when it was submitted.
    """
    if batches is None:
        batches = api.iter_result_chunk_batches(request_id)
    if key is None:
        key = api.submission_key(request_id)
    return _events(batches, lambda: api.abort_submission(key, outcome="success"), parser, stops,
                   Usage() if usage is None else usage)


async def _events(batches, served, parser, stops, usage):
    matcher = StopMatcher(stops) if stops else None
    stop: str | None = None
    # a byte-level BPE token's bytes can end inside a character: hold them
    text = Utf8Stream()
    failed = False
    try:
        async for batch in batches:
            events: list = []
            for c in batch:
                if c.modality == "text":
                    usage.add(c)
                    if parser is not None and parser.reasoning:
                        usage.reasoning += (c.metadata or {}).get("tokens", 0)
                    piece = text.feed(c.data)
                    if not piece:
                        continue
                    deltas = [{"content": piece}] if parser is None else parser.feed(piece)
                    if (stop := _passed(deltas, matcher, events)) is not None:
                        break
                    if parser is not None and parser.done:
                        break
                elif c.modality == "error":
                    # The request failed after the stream opened (an engine error
                    # mid generation, a preprocess error). The error is the
                    # iterator's last chunk, so the loop ends on its own;
                    # breaking out instead would trip the iterator's abort on a
                    # request that is already gone.
                    failed = True
                    events.append(Failure(c.data.decode("utf-8", "replace"), int(c.metadata.get("status", 500))))
                else:
                    # its own list: a client's frame of text tokens stays one write
                    if events:
                        yield events
                        events = []
                    yield [Media(c.modality, c.data)]
            if stop is not None or parser is not None and parser.done:
                served()  # before the close below, which would count an abort
                yield [*events, Finish(finish_reason(parser, usage, stop), usage, stop)]
                return
            if events:
                yield events
    except HTTPException as exc:
        # the delivery timeout raises out of the iterator, which aborts the request on its way out
        yield [Failure(str(exc.detail), exc.status_code)]
        return
    finally:
        # a reply that ended early (a stop string, a client gone) stops the request
        await batches.aclose()
    if failed:
        return
    tail = text.flush()
    if parser is None:
        deltas = [{"content": tail}] if tail else []
    else:
        deltas = parser.feed(tail) + parser.finish()
    events = []
    stop = _passed(deltas, matcher, events)
    if stop is None and matcher is not None and (held := matcher.flush()):
        events.append({"content": held})
    events.append(Finish(finish_reason(parser, usage, stop), usage, stop))
    yield events


async def collect(events, raw_request=None) -> list:
    """Every event of a reply, for a non-streaming response. A failure raises
    as its HTTP error; a client that disconnects stops the request."""

    async def drain() -> list:
        out = []
        async for batch in events:
            out.extend(batch)
        return out

    task = asyncio.ensure_future(drain())
    try:
        while raw_request is not None and not task.done():
            await asyncio.wait({task}, timeout=_DISCONNECT_POLL_S)
            if not task.done() and await raw_request.is_disconnected():
                # unwinds the chunk iterator, which aborts the request
                task.cancel()
                await asyncio.wait({task})
                raise HTTPException(status_code=499, detail="client disconnected")
        out = await task
    finally:
        task.cancel()
    if out and isinstance(out[-1], Failure):
        raise HTTPException(status_code=out[-1].status, detail=out[-1].message)
    return out


def runs(deltas) -> list[Run]:
    """A whole reply's deltas as its stretches of reasoning, content and calls, in order."""
    out: list[Run] = []
    calls: dict[int, Run] = {}
    for d in deltas:
        if not isinstance(d, dict):
            continue
        if "tool_calls" in d:
            for c in d["tool_calls"]:
                fn = c.get("function") or {}
                if "id" in c:
                    calls[c["index"]] = run = Run("call", call={"id": c["id"], "name": fn["name"], "arguments": ""})
                    out.append(run)
                calls[c["index"]].call["arguments"] += fn.get("arguments", "")
            continue
        ((key, text),) = d.items()
        kind = "reasoning" if key == "reasoning_content" else "content"
        if out and out[-1].kind == kind:
            out[-1].text += text
        else:
            out.append(Run(kind, text))
    return out
