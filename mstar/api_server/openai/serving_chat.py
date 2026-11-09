"""/v1/chat/completions handler (streaming + non-streaming).

Translates an OpenAI chat request into a submit_request via the model adapter,
then maps the resulting modality chunks back to OpenAI shapes: text into
``message.content``, audio into ``message.audio`` (base64, in ``audio.format``), images into
``image_url`` data-URL content parts.

Usage: ``completion_tokens`` is the reply's sampled token count, from the
``token_ids`` its text chunks carry (``return_token_ids``, which the handler
sets for a reply it reports usage on), stop tokens under ``ignore_eos``
included. The text can't give it back: a special token, or a split the
tokenizer would not make, re-encodes to fewer tokens (vllm bench serve counted
Kimi K3.1's 960-token replies as ~845 that way). A streamed reply reports it
when the request asks (``stream_options.include_usage``) in one last chunk with
no choices, as vLLM does; ``prompt_tokens`` (and ``total_tokens``) only once a
model reports its prompt length, so a client keeps its own count rather than
taking a zero.

finish_reason: ``length`` when the reply's sampled tokens reached the request's
own limit (``max_completion_tokens``, else ``max_tokens``), ``stop`` otherwise,
streamed or not, as vLLM reports it. A reply cut by a limit the request didn't
set (a model's output cap, its context) reads ``stop``: the engine reports no
reason.
"""

from __future__ import annotations

import asyncio
import base64

from fastapi import HTTPException

from mstar.api_server import media_io
from mstar.api_server.openai._util import SSE_DONE, error_type, now, rid, sse
from mstar.api_server.streaming_response import ReleasingStreamingResponse
from mstar.engine.resources.lora.config import LORA_ADAPTER_KWARG
from mstar.utils.utf8_stream import Utf8Stream


async def create_chat_completion(api, model_name, adapter, req, raw_request=None):
    """The response dict, or for ``req.stream`` the SSE response itself."""
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
    audio_format = _audio_format(req, sample_rate)
    # off the loop: media parts may be decoded, written or fetched (MSTAR_ALLOW_REMOTE)
    args = await asyncio.to_thread(adapter.chat_to_request, req, api.upload_dir)
    request_id = rid("chatcmpl")

    try:
        # resolved once, before it is submitted: the response names the adapter that
        # served it, as vLLM's does, whether the model field or the lora_adapter kwarg
        # picked it
        model_kwargs = api.lora_kwargs(req.model, args.model_kwargs)
        served = (model_kwargs or {}).get(LORA_ADAPTER_KWARG)
        if served is not None:
            model_name = served
        include_usage = _include_usage(req)
        # every reply counts its tokens: its usage, and its finish_reason against the request's limit
        model_kwargs = {**(model_kwargs or {}), "return_token_ids": True}
        limit = _token_limit(req)
        api.submit_request(
            text=args.text,
            file_paths=args.file_paths,
            input_modalities=args.input_modalities,
            output_modalities=args.output_modalities,
            model_kwargs=model_kwargs,
            prompt_parts=args.prompt_parts,
            streaming=bool(req.stream),
            request_id=request_id,
        )
        if req.stream:
            # a request the engine refuses gets its status, not a 200
            batches = await api.open_result_stream(request_id, raw_request, batches=True)
            return ReleasingStreamingResponse(
                _stream_events(batches, model_name, request_id, sample_rate, include_usage=include_usage,
                               limit=limit),
                release=lambda: api.release_request(request_id),
                media_type="text/event-stream", headers={"Cache-Control": "no-cache"},
            )
        chunks = await api.collect_results(request_id, raw_request)
    finally:
        media_io.schedule_upload_cleanup(args.file_paths, api.upload_dir)
    return _build_response(model_name, request_id, chunks, sample_rate, audio_format, limit=limit)


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


def _token_limit(req) -> int | None:
    """The reply's token limit the request set: ``max_completion_tokens``, else ``max_tokens``."""
    limit = getattr(req, "max_completion_tokens", None)
    return getattr(req, "max_tokens", None) if limit is None else limit


def _include_usage(req) -> bool:
    """A streamed request that asked for its usage (``stream_options.include_usage``)."""
    return bool(req.stream) and bool((getattr(req, "stream_options", None) or {}).get("include_usage"))


class _TokenCount:
    """A reply's sampled tokens, summed over its text chunks' ``token_ids``; None once a text chunk came without
    them (a model whose text is not one chunk per sampled token, or a request that didn't ask)."""

    def __init__(self) -> None:
        self.tokens: int | None = 0

    def add(self, chunk) -> None:
        if chunk.modality != "text" or self.tokens is None:
            return
        ids = (getattr(chunk, "metadata", None) or {}).get("token_ids")
        self.tokens = None if ids is None else self.tokens + len(ids)

    def finish_reason(self, limit: int | None) -> str:
        """``length`` once the counted tokens reach the request's limit, else ``stop``."""
        return "length" if self.tokens is not None and limit is not None and self.tokens >= limit else "stop"

    def usage(self) -> dict | None:
        """The usage a reply reports, None when its tokens weren't counted."""
        return None if self.tokens is None else {"completion_tokens": self.tokens}


def _build_response(model_name, request_id, chunks, sample_rate, audio_format="wav", limit=None) -> dict:
    text_parts: list[bytes] = []
    audio_pcm: list[bytes] = []
    images: list[bytes] = []
    count = _TokenCount()
    for c in chunks:
        count.add(c)
        if c.modality == "text":
            text_parts.append(c.data)
        elif c.modality == "audio":
            audio_pcm.append(c.data)
        elif c.modality == "image":
            images.append(c.data)

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
        "choices": [{"index": 0, "message": message, "finish_reason": count.finish_reason(limit)}],
        # the prompt's length is not known here: 0, as before, rather than a guess
        "usage": {"prompt_tokens": 0, "completion_tokens": count.tokens or 0, "total_tokens": count.tokens or 0},
    }


async def _stream_events(batches, model_name, request_id, sample_rate, include_usage=False, limit=None):
    created = now()
    count = _TokenCount()

    def chunk(delta, finish=None) -> str:
        return sse({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    def error(message: str, status: int) -> str:
        return sse({"error": {"message": message, "type": error_type(status), "code": status}})

    # The role rides on the first delta: clients time the first token from the
    # first chunk, so nothing may go out before the model has produced one.
    role = {"role": "assistant"}
    failed = False
    # a byte-level BPE token's bytes can end inside a character: hold them
    text = Utf8Stream()
    try:
        # One yield per batch of text chunks: the tokens a request had waiting
        # when its handler ran go out as one body frame (one write), which is
        # what keeps the delivery loop ahead at high token rates. A batch of
        # one is the common case when the loop keeps up.
        async for batch in batches:
            text_events: list[str] = []
            for c in batch:
                count.add(c)
                if c.modality == "text":
                    content = text.feed(c.data)
                    if content:
                        text_events.append(chunk({**role, "content": content}))
                        role = {}
                    continue
                if text_events:
                    yield "".join(text_events)
                    text_events = []
                if c.modality == "audio":
                    # Streaming audio deltas are base64 16-bit PCM at the model rate.
                    audio = {"id": rid("audio"), "data": base64.b64encode(c.data).decode("ascii")}
                    yield chunk({**role, "audio": audio})
                    role = {}
                elif c.modality == "image":
                    yield chunk({**role, "content": media_io.png_to_data_url(c.data)})
                    role = {}
                elif c.modality == "error":
                    # The request failed after the stream opened (an engine error
                    # mid generation, a preprocess error); the HTTP status is
                    # committed, so the failure travels in-band the way the
                    # non-streaming path's error body does, not as a normal
                    # ``stop`` a client would take for a complete answer. The error
                    # is the iterator's last chunk, so the loop ends on its own;
                    # returning here instead would trip the iterator's abort on a
                    # request that is already gone.
                    failed = True
                    yield error(c.data.decode("utf-8", "replace"), int(c.metadata.get("status", 500)))
            if text_events:
                yield "".join(text_events)
    except HTTPException as exc:
        # The delivery timeout raises out of the iterator (which aborts the
        # request on its way out); report it the same way.
        failed = True
        yield error(str(exc.detail), exc.status_code)
    if not failed:
        tail = text.flush()
        yield chunk({**role, "content": tail} if tail else role, finish=count.finish_reason(limit))
        usage = count.usage() if include_usage else None
        if usage is not None:
            yield sse({"id": request_id, "object": "chat.completion.chunk", "created": created,
                       "model": model_name, "choices": [], "usage": usage})
    yield SSE_DONE
