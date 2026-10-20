"""A chat reply as protocol-neutral events, which each API's encoder writes in
its own shape (chat completions here and in ``serving_chat``).

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
import codecs
from dataclasses import dataclass, field

from fastapi import HTTPException

# how often a non-streaming handler checks whether its client is still there
_DISCONNECT_POLL_S = 0.05


@dataclass
class Usage:
    """Token counts and why the reply ended, from the text chunks' metadata
    (see the data worker's ``_text_usage``)."""

    prompt: int = 0
    completion: int = 0
    stopped: bool = True

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
    """``reason`` is ``stop`` (a stop token), ``length`` (the token budget)
    or ``tool_calls`` (the reply made calls)."""

    reason: str
    usage: Usage


@dataclass
class Run:
    """A stretch of one kind in a whole reply: ``reasoning`` or ``content``
    text, or a ``call`` with its ``id``, ``name`` and ``arguments``."""

    kind: str
    text: str = ""
    call: dict = field(default_factory=dict)


def finish_reason(parser, usage: Usage) -> str:
    if parser is not None and parser.finish_reason == "tool_calls":
        return "tool_calls"
    return "stop" if usage.stopped else "length"


def reply_events(api, request_id: str, parser=None):
    """The events of a request's reply, one list per batch of result chunks.
    Bound to the submission when called: a stream's body is iterated later,
    by Starlette."""
    return _events(api.iter_result_chunk_batches(request_id), parser)


async def _events(batches, parser):
    usage = Usage()
    # a byte-level BPE token's bytes can end inside a character: hold them
    text = codecs.getincrementaldecoder("utf-8")("replace")
    failed = False
    try:
        async for batch in batches:
            events: list = []
            for c in batch:
                if c.modality == "text":
                    usage.add(c)
                    piece = text.decode(c.data)
                    if not piece:
                        continue
                    events.extend([{"content": piece}] if parser is None else parser.feed(piece))
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
            if events:
                yield events
    except HTTPException as exc:
        # the delivery timeout raises out of the iterator, which aborts the request on its way out
        yield [Failure(str(exc.detail), exc.status_code)]
        return
    finally:
        # a reply that ended early (a client gone) stops the request
        await batches.aclose()
    if failed:
        return
    tail = text.decode(b"", final=True)
    if parser is None:
        deltas = [{"content": tail}] if tail else []
    else:
        deltas = parser.feed(tail) + parser.finish()
    yield [*deltas, Finish(finish_reason(parser, usage), usage)]


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
