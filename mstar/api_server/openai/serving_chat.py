"""/v1/chat/completions handler (streaming + non-streaming).

Translates an OpenAI chat request into a submit_request via the model adapter,
then maps the resulting modality chunks back to OpenAI shapes: text into
``message.content``, audio into ``message.audio`` (base64, in ``audio.format``), images into
``image_url`` data-URL content parts, and the closing ``usage`` chunk into
``usage`` and ``finish_reason``. An adapter's output parser splits a text
reply into ``content``, ``reasoning_content`` and ``tool_calls``.

A streamed reply, and a whole one that is parsed or has stop strings, is read
as ``reply`` events.
"""

from __future__ import annotations

import asyncio
import base64

from mstar.api_server import media_io
from mstar.api_server.openai import reply
from mstar.api_server.openai._util import SSE_DONE, error_type, now, rid, sse
from mstar.api_server.request_types import USAGE


async def create_chat_completion(api, model_name, adapter, req, raw_request=None):
    """The response dict, or for ``req.stream`` the SSE response itself."""
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
    audio_format = _audio_format(req, sample_rate)
    # off the loop: media parts may be decoded, written or fetched (MSTAR_ALLOW_REMOTE)
    args = await asyncio.to_thread(adapter.chat_to_request, req, api.upload_dir)
    parser = adapter.make_output_parser(req)
    stops = adapter.stop_strings(req)
    request_id = rid("chatcmpl")

    try:
        model_name = reply.submit(api, request_id, model_name, req, args)
        if req.stream:
            events = await reply.stream_events(api, request_id, raw_request, parser, stops)
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return reply.sse_response(api, request_id, _stream_events(events, model_name, request_id, include_usage))
        if parser is not None or stops:
            # read chunk by chunk, so a stop string ends the request as it appears
            events = await reply.collect(reply.reply_events(api, request_id, parser, stops), raw_request)
            return _text_response(model_name, request_id, events)
        chunks = await api.collect_results(request_id, raw_request)
    finally:
        media_io.schedule_upload_cleanup(args.file_paths, api.upload_dir)
    return _build_response(model_name, request_id, chunks, sample_rate, audio_format)


def _usage(usage: reply.Usage) -> dict:
    out = {"prompt_tokens": usage.prompt, "completion_tokens": usage.completion,
           "total_tokens": usage.prompt + usage.completion}
    if usage.reasoning:
        # OpenAI's field for a reasoning model's share; a reply parser counts it
        out["completion_tokens_details"] = {"reasoning_tokens": usage.reasoning}
    return out


def _text_response(model_name, request_id, events) -> dict:
    finish = events[-1]
    content, reasoning, calls = "", "", []
    for run in reply.runs(events):
        if run.kind == "call":
            calls.append({"id": run.call["id"], "type": "function",
                          "function": {"name": run.call["name"], "arguments": run.call["arguments"]}})
        elif run.kind == "reasoning":
            reasoning += run.text
        else:
            content += run.text
    message: dict = {"role": "assistant", "content": content or (None if calls else "")}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": now(),
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish.reason}],
        "usage": _usage(finish.usage),
    }


def _audio_format(req, sample_rate: int) -> str:
    """The container ``message.audio`` is sent in, from OpenAI's ``audio.format``: a speech
    container, with ``pcm16`` naming raw PCM. Stream deltas are raw PCM16, so a stream takes only that."""
    fmt = str((req.audio or {}).get("format") or ("pcm16" if req.stream else "wav")).lower()
    allowed = ("pcm16",) if req.stream else tuple(
        "pcm16" if f == "pcm" else f for f in media_io.audio_formats(sample_rate=sample_rate))
    if fmt not in allowed:
        how = "streamed" if req.stream else "produced"
        raise ValueError(f"audio.format {fmt!r} cannot be {how} by this server; use one of: {', '.join(allowed)}")
    return "pcm" if fmt == "pcm16" else fmt


def _build_response(model_name, request_id, chunks, sample_rate, audio_format="wav") -> dict:
    """A whole reply with no output parser and no stop strings: text, audio and images."""
    text_parts: list[bytes] = []
    audio_pcm: list[bytes] = []
    images: list[bytes] = []
    usage = reply.Usage()
    for c in chunks:
        if c.modality == "text":
            text_parts.append(c.data)
        elif c.modality == "audio":
            audio_pcm.append(c.data)
        elif c.modality == "image":
            images.append(c.data)
        elif c.modality == USAGE:
            usage.close(c.metadata)

    # a byte-level BPE token's bytes can end inside a character
    text = b"".join(text_parts).decode("utf-8", "replace")
    message: dict = {"role": "assistant", "content": text}

    if audio_pcm:
        audio, _ = media_io.pcm16_to_container(b"".join(audio_pcm), sample_rate, audio_format)
        message["audio"] = {
            "id": rid("audio"),
            "data": base64.b64encode(audio).decode("ascii"),
            "expires_at": now() + 86400,
            "transcript": text,
        }
    if images:
        parts: list[dict] = []
        if text:
            parts.append({"type": "text", "text": text})
        for img in images:
            parts.append({"type": "image_url", "image_url": {"url": media_io.png_to_data_url(img)}})
        message["content"] = parts

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": now(),
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": usage.finish_reason}],
        "usage": _usage(usage),
    }


async def _stream_events(batches, model_name, request_id, include_usage=False):
    """The chat.completion.chunk encoder of a reply's events."""
    created = now()

    def chunk(delta, finish=None) -> str:
        return sse({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    # The role rides on the first delta: clients time the first token from the
    # first chunk, so nothing may go out before the model has produced one.
    role = {"role": "assistant"}
    usage = None
    # One yield per list of events: the tokens a request had waiting when its
    # handler ran go out as one body frame (one write), which is what keeps the
    # delivery loop ahead at high token rates. A list of one is the common
    # case when the loop keeps up.
    async for events in batches:
        out: list[str] = []
        for i, e in enumerate(events):
            if isinstance(e, dict):
                if i + 1 < len(events) and isinstance(events[i + 1], reply.Finish):
                    continue  # rides on the finish chunk
                out.append(chunk({**role, **e}))
            elif isinstance(e, reply.Finish):
                last = events[i - 1] if i and isinstance(events[i - 1], dict) else {}
                out.append(chunk({**role, **last}, finish=e.reason))
                usage = e.usage
            elif isinstance(e, reply.Failure):
                # The HTTP status is committed, so the failure travels in-band the
                # way the non-streaming path's error body does, not as a normal
                # ``stop`` a client would take for a complete answer.
                out.append(sse({"error": {"message": e.message, "type": error_type(e.status), "code": e.status}}))
            elif e.modality == "audio":
                # Streaming audio deltas are base64 16-bit PCM at the model rate.
                audio = {"id": rid("audio"), "data": base64.b64encode(e.data).decode("ascii")}
                out.append(chunk({**role, "audio": audio}))
            else:
                out.append(chunk({**role, "content": media_io.png_to_data_url(e.data)}))
            role = {}
        yield "".join(out)
    if include_usage and usage is not None:
        # OpenAI's stream_options.include_usage: one last chunk, no choices
        yield sse({"id": request_id, "object": "chat.completion.chunk", "created": created,
                   "model": model_name, "choices": [], "usage": _usage(usage)})
    yield SSE_DONE
