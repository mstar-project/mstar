"""/v1/chat/completions handler (streaming + non-streaming).

Translates an OpenAI chat request into a submit_request via the model adapter,
then maps the resulting modality chunks back to OpenAI shapes: text into
``message.content``, audio into ``message.audio`` (base64 WAV), images into
``image_url`` data-URL content parts.
"""

from __future__ import annotations

import base64
import codecs
from time import perf_counter

from fastapi import HTTPException

from mstar.api_server import media_io
from mstar.api_server.openai._util import SSE_DONE, error_type, now, rid, sse
from mstar.utils.profiler import PHASE_PERIOD, hop_stamp


async def create_chat_completion(api, model_name, adapter, req, raw_request=None):
    t_in = perf_counter()
    args = adapter.chat_to_request(req, api.upload_dir)
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
        key = api.submission_key(request_id) if PHASE_PERIOD else None
        if key is not None:
            hop_stamp(key, "api.http_in", t_in)
        if req.stream:
            return _stream(api, model_name, request_id, sample_rate, key)
        chunks = await api.collect_results(request_id, raw_request)
    finally:
        media_io.schedule_upload_cleanup(args.file_paths, api.upload_dir)
    return _build_response(model_name, request_id, chunks, sample_rate)


def _build_response(model_name, request_id, chunks, sample_rate) -> dict:
    text_parts: list[bytes] = []
    audio_pcm: list[bytes] = []
    images: list[bytes] = []
    for c in chunks:
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
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream(api, model_name, request_id, sample_rate, key=None):
    # Bound to the submission now: the body is iterated later, by Starlette.
    batches = api.iter_result_chunk_batches(request_id)
    return _stream_events(batches, model_name, request_id, sample_rate, key)


async def _stream_events(batches, model_name, request_id, sample_rate, key=None):
    """``key`` (the submission's, while MSTAR_PHASE_TIMING is on) stamps the
    first text delta's hand-off to the HTTP layer."""
    created = now()

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
    text = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        # One yield per batch of text chunks: the tokens a request had waiting
        # when its handler ran go out as one body frame (one write), which is
        # what keeps the delivery loop ahead at high token rates. A batch of
        # one is the common case when the loop keeps up.
        async for batch in batches:
            text_events: list[str] = []
            for c in batch:
                if c.modality == "text":
                    content = text.decode(c.data)
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
                if key is not None:
                    hop_stamp(key, "api.first_out")
                    key = None
                yield "".join(text_events)
    except HTTPException as exc:
        # The delivery timeout raises out of the iterator (which aborts the
        # request on its way out); report it the same way.
        failed = True
        yield error(str(exc.detail), exc.status_code)
    if not failed:
        tail = text.decode(b"", final=True)
        yield chunk({**role, "content": tail} if tail else role, finish="stop")
    yield SSE_DONE
