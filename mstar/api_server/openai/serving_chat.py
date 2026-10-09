"""/v1/chat/completions handler (streaming + non-streaming).

Translates an OpenAI chat request into a submit_request via the model adapter,
then maps the resulting modality chunks back to OpenAI shapes: text into
``message.content``, audio into ``message.audio`` (base64 WAV), images into
``image_url`` data-URL content parts. An adapter's output parser splits a text
reply into ``content``, ``reasoning_content`` and ``tool_calls``.
"""

from __future__ import annotations

import base64
import codecs

from fastapi import HTTPException

from mstar.api_server import media_io
from mstar.api_server.openai._util import SSE_DONE, error_type, now, rid, sse


async def create_chat_completion(api, model_name, adapter, req, raw_request=None):
    args = adapter.chat_to_request(req, api.upload_dir)
    parser = adapter.make_output_parser(req)
    # checked before the request runs: a raise after submit leaves it running with no reader
    stream_options = getattr(req, "stream_options", None) or {}
    if not isinstance(stream_options, dict):
        raise ValueError("stream_options must be an object")
    include_usage = bool(stream_options.get("include_usage"))
    request_id = rid("chatcmpl")
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000

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
        return _stream(api, model_name, request_id, sample_rate, parser, include_usage)
    chunks = await api.collect_results(request_id, raw_request)
    return _build_response(model_name, request_id, chunks, sample_rate, parser)


class _Usage:
    """Token counts and why the reply ended, from the text chunks' metadata
    (see the data worker's ``_text_usage``)."""

    def __init__(self):
        self.prompt = self.completion = 0
        self.stopped = True

    def add(self, chunk) -> None:
        meta = chunk.metadata or {}
        self.prompt = meta.get("prompt_tokens", self.prompt)
        self.completion += meta.get("tokens", 0)
        if "stop_token" in meta:
            self.stopped = meta["stop_token"]

    def finish_reason(self, parser) -> str:
        if parser is not None and parser.finish_reason == "tool_calls":
            return "tool_calls"
        return "stop" if self.stopped else "length"

    def as_dict(self) -> dict:
        return {"prompt_tokens": self.prompt, "completion_tokens": self.completion,
                "total_tokens": self.prompt + self.completion}


def _build_response(model_name, request_id, chunks, sample_rate, parser=None) -> dict:
    text_parts: list[bytes] = []
    audio_pcm: list[bytes] = []
    images: list[bytes] = []
    usage = _Usage()
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
        "choices": [{"index": 0, "message": message, "finish_reason": usage.finish_reason(parser)}],
        "usage": usage.as_dict(),
    }


async def _stream(api, model_name, request_id, sample_rate, parser=None, include_usage=False):
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
    usage = _Usage()
    try:
        async for c in api.iter_result_chunks(request_id):
            if c.modality == "text":
                usage.add(c)
                content = text.decode(c.data)
                if not content:
                    continue
                if parser is not None:
                    for delta in parser.feed(content):
                        yield chunk({**role, **delta})
                        role = {}
                    continue
                delta = {"content": content}
            elif c.modality == "audio":
                # Streaming audio deltas are base64 16-bit PCM at the model rate.
                delta = {"audio": {"id": rid("audio"), "data": base64.b64encode(c.data).decode("ascii")}}
            elif c.modality == "image":
                delta = {"content": media_io.png_to_data_url(c.data)}
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
                continue
            else:
                continue
            yield chunk({**role, **delta})
            role = {}
    except HTTPException as exc:
        # The delivery timeout raises out of the iterator (which aborts the
        # request on its way out); report it the same way.
        failed = True
        yield error(str(exc.detail), exc.status_code)
    if not failed:
        tail = text.decode(b"", final=True)
        if parser is None:
            yield chunk({**role, "content": tail} if tail else role, finish=usage.finish_reason(None))
        else:
            deltas = parser.feed(tail) + parser.finish()
            for delta in deltas[:-1]:
                yield chunk({**role, **delta})
                role = {}
            yield chunk({**role, **(deltas[-1] if deltas else {})}, finish=usage.finish_reason(parser))
        if include_usage:
            # OpenAI's stream_options.include_usage: one last chunk, no choices
            yield sse({"id": request_id, "object": "chat.completion.chunk", "created": created,
                       "model": model_name, "choices": [], "usage": usage.as_dict()})
    yield SSE_DONE
