"""/v1/chat/completions handler (streaming + non-streaming).

Translates an OpenAI chat request into a submit_request via the model adapter,
then maps the resulting modality chunks back to OpenAI shapes: text into
``message.content``, audio into ``message.audio`` (base64 WAV), images into
``image_url`` data-URL content parts.
"""

from __future__ import annotations

import base64

from starlette.concurrency import run_in_threadpool

from mstar.api_server import media_io
from mstar.api_server.openai._util import SSE_DONE, now, rid, sse


async def create_chat_completion(api, model_name, adapter, req):
    args = adapter.chat_to_request(req, api.upload_dir)
    request_id = rid("chatcmpl")
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000

    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=args.output_modalities,
        model_kwargs=args.model_kwargs,
        streaming=bool(req.stream),
        request_id=request_id,
    )

    if req.stream:
        return _stream(api, model_name, request_id, sample_rate)
    chunks = await run_in_threadpool(api.collect_results, request_id)
    return _build_response(model_name, request_id, chunks, sample_rate)


def _build_response(model_name, request_id, chunks, sample_rate) -> dict:
    text_parts: list[str] = []
    audio_pcm: list[bytes] = []
    images: list[bytes] = []
    for c in chunks:
        if c.modality == "text":
            text_parts.append(c.data.decode("utf-8", "replace"))
        elif c.modality == "audio":
            audio_pcm.append(c.data)
        elif c.modality == "image":
            images.append(c.data)

    text = "".join(text_parts)
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


async def _stream(api, model_name, request_id, sample_rate):
    created = now()

    def chunk(delta, finish=None) -> str:
        return sse({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    yield chunk({"role": "assistant"})
    async for c in api.iter_result_chunks(request_id):
        if c.modality == "text":
            yield chunk({"content": c.data.decode("utf-8", "replace")})
        elif c.modality == "audio":
            # Streaming audio deltas are base64 16-bit PCM at the model rate.
            yield chunk({"audio": {"id": rid("audio"), "data": base64.b64encode(c.data).decode("ascii")}})
        elif c.modality == "image":
            yield chunk({"content": media_io.png_to_data_url(c.data)})
    yield chunk({}, finish="stop")
    yield SSE_DONE
