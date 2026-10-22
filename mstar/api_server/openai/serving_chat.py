"""/v1/chat/completions handler (streaming + non-streaming).

Translates an OpenAI chat request into a submit_request via the model adapter,
then maps the resulting modality chunks back to OpenAI shapes: text into
``message.content``, audio into ``message.audio`` (base64 WAV), images into
``image_url`` data-URL content parts. An adapter's output parser splits a text
reply into ``content``, ``reasoning_content`` and ``tool_calls``.

A streamed reply, and a whole one that is parsed or has stop strings, is read
as ``reply`` events, which this module writes as chat completion chunks or one
chat completion. Stop strings match the answer (``content``) only, not the
reasoning or a tool call's arguments (see ``reply``).
"""

from __future__ import annotations

import base64

from mstar.api_server import media_io
from mstar.api_server.openai import reply
from mstar.api_server.openai._util import SSE_DONE, error_type, now, rid, sse


async def create_chat_completion(api, model_name, adapter, req, raw_request=None):
    args = adapter.chat_to_request(req, api.upload_dir)
    parser = adapter.make_output_parser(req)
    stops = adapter.stop_strings(req)
    # checked before the request runs: a raise after submit leaves it running with no reader
    stream_options = getattr(req, "stream_options", None) or {}
    if not isinstance(stream_options, dict):
        raise ValueError("stream_options must be an object")
    include_usage = bool(stream_options.get("include_usage"))
    request_id = rid("chatcmpl")
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000

    try:
        api.submit_request(
            text=args.text,
            file_paths=args.file_paths,
            input_modalities=args.input_modalities,
            output_modalities=args.output_modalities,
            model_kwargs=args.model_kwargs,
            prompt_parts=args.prompt_parts,
            streaming=bool(req.stream),
            request_id=request_id,
        )
        if req.stream:
            return _stream(api, model_name, request_id, sample_rate, parser, include_usage, stops)
        if parser is not None or stops:
            # read chunk by chunk, so a stop string ends the request as it appears
            events = await reply.collect(reply.reply_events(api, request_id, parser, stops), raw_request)
            return _text_response(model_name, request_id, events)
        chunks = await api.collect_results(request_id, raw_request)
    finally:
        media_io.schedule_upload_cleanup(args.file_paths, api.upload_dir)
    return _build_response(model_name, request_id, chunks, sample_rate, parser)


def _usage(usage: reply.Usage) -> dict:
    return {"prompt_tokens": usage.prompt, "completion_tokens": usage.completion,
            "total_tokens": usage.prompt + usage.completion}


def _text_response(model_name, request_id, events) -> dict:
    """A text reply, from its events."""
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


def _build_response(model_name, request_id, chunks, sample_rate, parser=None) -> dict:
    text_parts: list[bytes] = []
    audio_pcm: list[bytes] = []
    images: list[bytes] = []
    usage = reply.Usage()
    for c in chunks:
        if c.modality == "text":
            text_parts.append(c.data)
            usage.add(c)
        elif c.modality == "audio":
            audio_pcm.append(c.data)
        elif c.modality == "image":
            images.append(c.data)

    # a byte-level BPE token's bytes can end inside a character
    text = b"".join(text_parts).decode("utf-8", "replace")
    message: dict = {"role": "assistant", "content": text}
    if parser is not None:
        message = parser.message(text)

    if audio_pcm:
        wav = media_io.pcm16_to_wav_bytes(b"".join(audio_pcm), sample_rate)
        message["audio"] = {
            "id": rid("audio"),
            "data": base64.b64encode(wav).decode("ascii"),
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
        "choices": [{"index": 0, "message": message, "finish_reason": reply.finish_reason(parser, usage)}],
        "usage": _usage(usage),
    }


def _stream(api, model_name, request_id, sample_rate, parser=None, include_usage=False, stops=None):
    # Bound to the submission now: the body is iterated later, by Starlette.
    batches = reply.reply_events(api, request_id, parser, stops)
    return _stream_events(batches, model_name, request_id, include_usage)


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
