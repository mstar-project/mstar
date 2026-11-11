"""A chat reply as protocol-neutral events, which chat completions
(``serving_chat``) writes in its shape.

``reply_events`` yields a list of events per batch of result chunks:

* a delta, as the output parser writes them: ``{"reasoning_content": str}``,
  ``{"content": str}`` or ``{"tool_calls": [call]}``, where a call's first
  delta carries its ``id`` and ``function.name`` and later ones only
  ``function.arguments`` to append;
* :class:`Media` for an audio or image chunk, in a list of its own;
* :class:`Failure` when the request failed after it was submitted;
* :class:`Finish`, last, with why the reply ended and its usage.

A failed reply has no ``Finish``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from fastapi import HTTPException

from mstar.api_server.request_types import USAGE
from mstar.api_server.streaming_response import ReleasingStreamingResponse
from mstar.engine.resources.lora.config import LORA_ADAPTER_KWARG
from mstar.utils.utf8_stream import Utf8Stream

# how often a non-streaming handler checks whether its client is still there
_DISCONNECT_POLL_S = 0.05


@dataclass
class Usage:
    """A reply's token counts and why the engine ended it. The text chunks add
    up ``completion`` as they are read and carry ``prompt`` (``ResultChunk``'s
    ``num_tokens`` and ``prompt_tokens``), so a reply its handler ends early
    has both. The ``usage`` chunk that closes a reply which ran to its end
    (``APIServer.submit_request``'s ``report_usage``) restates them and gives
    its ``finish_reason``."""

    prompt: int = 0
    completion: int = 0
    # ``length`` once the closing chunk says the reply used its whole budget
    finish_reason: str = "stop"

    def close(self, reported: dict) -> None:
        """Takes the closing ``usage`` chunk's metadata."""
        self.prompt, self.completion = reported["prompt_tokens"], reported["completion_tokens"]
        self.finish_reason = reported["finish_reason"]


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
    (the token budget) or ``tool_calls`` (the reply made calls)."""

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


def submit(api, request_id: str, model_name: str, req, args) -> str:
    """Submits a chat request's ``args`` under the LoRA adapter its ``model``
    field or ``lora_adapter`` kwarg picks. Returns the model name the reply
    reports: the adapter that served it, as vLLM's does, else ``model_name``."""
    model_kwargs = api.lora_kwargs(req.model, args.model_kwargs).model_kwargs
    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=args.output_modalities,
        model_kwargs=model_kwargs,
        prompt_parts=args.prompt_parts,
        streaming=bool(req.stream),
        request_id=request_id,
        report_usage=True,
    )
    return (model_kwargs or {}).get(LORA_ADAPTER_KWARG) or model_name


async def stream_events(api, request_id: str, raw_request, parser=None, stops=None):
    """``reply_events`` of a streamed request once its results start: a
    request the engine refuses raises its status here rather than inside a 200."""
    batches = await api.open_result_stream(request_id, raw_request, batches=True)
    return reply_events(api, request_id, parser, stops, batches=batches)


def sse_response(api, request_id: str, body) -> ReleasingStreamingResponse:
    return ReleasingStreamingResponse(body, release=lambda: api.release_request(request_id),
                                      media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def finish_reason(parser, usage: Usage, stop_string: str | None = None) -> str:
    if parser is not None and parser.finish_reason == "tool_calls":
        return "tool_calls"
    return "stop" if stop_string is not None else usage.finish_reason


def reply_events(api, request_id: str, parser=None, stops: list[str] | None = None, batches=None):
    """The events of a request's reply, one list per batch of result chunks.
    Bound to the submission when called, as Starlette iterates a stream's body
    later.

    A stop string ends the reply where it starts: the request stops in the
    engine, as closing the iterator early does. Usage counts every
    token read until then, the stop's own included, as vLLM counts them, and
    can count a few past it: the rest of the stop's chunk (up to k under MTP),
    and what was read while the parser held the stop's end back.
    ``batches`` are the chunk batches already opened (``api.open_result_stream``).
    """
    if batches is None:
        batches = api.iter_result_chunk_batches(request_id)
    return _events(batches, parser, stops, Usage())


async def _events(batches, parser, stops, usage):
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
                    usage.completion += c.num_tokens
                    if c.prompt_tokens is not None:
                        usage.prompt = c.prompt_tokens
                    piece = text.feed(c.data)
                    if not piece:
                        continue
                    deltas = [{"content": piece}] if parser is None else parser.feed(piece)
                    if (stop := _passed(deltas, matcher, events)) is not None:
                        break
                elif c.modality == USAGE:
                    usage.close(c.metadata)
                elif c.modality == "error":
                    # the iterator's last chunk, so the loop ends on its own: breaking
                    # out would trip the iterator's abort on a request already gone
                    failed = True
                    events.append(Failure(c.data.decode("utf-8", "replace"), int(c.metadata.get("status", 500))))
                else:
                    # its own list: a client's frame of text tokens stays one write
                    if events:
                        yield events
                        events = []
                    yield [Media(c.modality, c.data)]
            if stop is not None:
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
