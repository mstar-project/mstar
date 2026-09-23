"""/v1/audio/speech handler (text-to-speech).

Non-streaming returns the full audio as a container blob (WAV by default).
Streaming returns a single open-ended WAV response (header + PCM16 frames) as
the audio is produced.

Long inputs can be synthesized as ordered sentence chunks (one engine request
per chunk, ``speech_chunking.split_sentences``): the adapter's
``speech_chunk_min_chars`` turns this on for long texts, and a client can force
or suppress it per request with ``sentence_chunking: true|false``. The next
chunks are submitted while the current one streams, so the engine batches them
and playback never waits for a prefill.

A streaming request only commits to HTTP 200 once its first result chunk has
arrived and is not an error; an error chunk before that becomes the HTTP error
it carries (the non-streaming path gets the same from ``collect_results``).
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from mstar.api_server import media_io
from mstar.api_server.openai._util import rid
from mstar.api_server.openai.speech_chunking import split_sentences


def _plan_chunks(req, adapter, text: str) -> list[str]:
    """The texts to synthesize, in playback order (``[text]`` when unchunked)."""
    # Extra request fields live in pydantic's ``model_extra``; plain objects
    # (tests, other frontends) may carry the attribute directly.
    requested = (getattr(req, "model_extra", None) or {}).get("sentence_chunking")
    if requested is None:
        requested = getattr(req, "sentence_chunking", None)
    min_chars = getattr(adapter, "speech_chunk_min_chars", None)
    if requested is False or (requested is None and (min_chars is None or len(text) < min_chars)):
        return [text]
    chunks = split_sentences(text, max_chars=getattr(adapter, "speech_chunk_max_chars", 400))
    return chunks or [text]


def _chunk_kwargs(model_kwargs: dict, index: int) -> dict:
    """Per-chunk model kwargs: identical, except a client seed advances per chunk."""
    kwargs = dict(model_kwargs)
    kwargs.pop("sentence_chunking", None)
    seed = kwargs.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        kwargs["seed"] = seed + index
    return kwargs


async def create_speech(api, model_name, adapter, req, raw_request=None):  # noqa: ARG001
    args = adapter.speech_to_request(req, api.upload_dir)
    request_id = rid("speech")
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
    fmt = (req.response_format or "wav").lower()
    chunks = _plan_chunks(req, adapter, args.text or "")

    def submit(index: int) -> str:
        chunk_id = request_id if len(chunks) == 1 else f"{request_id}-{index}"
        return api.submit_request(
            text=chunks[index],
            file_paths=args.file_paths,
            input_modalities=args.input_modalities,
            output_modalities=args.output_modalities,
            model_kwargs=_chunk_kwargs(args.model_kwargs, index),
            streaming=bool(req.stream),
            request_id=chunk_id,
        )

    lookahead = max(1, int(getattr(adapter, "speech_chunk_lookahead", 2)))
    if req.stream:
        pending = [submit(i) for i in range(min(lookahead, len(chunks)))]
        # Look at the first result before committing to a 200: a request the
        # engine rejects (bad voice, dead worker, ...) must surface as an HTTP
        # error, not as an empty WAV.
        first_iter = api.iter_result_chunks(pending[0])
        first = await anext(first_iter, None)
        _raise_if_error(first)
        return StreamingResponse(
            _stream_wav(api, submit, len(chunks), pending, first_iter, first, sample_rate),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"},
        )

    pcm_parts: list[bytes] = []
    pending: list[str] = [submit(i) for i in range(min(lookahead, len(chunks)))]
    for index in range(len(chunks)):
        if len(pending) < len(chunks):
            pending.append(submit(len(pending)))
        results = await api.collect_results(pending[index], raw_request)
        pcm_parts.append(b"".join(c.data for c in results if c.modality == "audio"))
    audio_bytes, mime = media_io.pcm16_to_container(b"".join(pcm_parts), sample_rate, fmt)
    return Response(content=audio_bytes, media_type=mime)


def _raise_if_error(chunk) -> None:
    """A data-worker failure arrives as an ``error`` chunk; turn it into the HTTP error it carries."""
    if chunk is not None and chunk.modality == "error":
        raise HTTPException(
            status_code=int((chunk.metadata or {}).get("status", 500)),
            detail=chunk.data.decode("utf-8", "replace") if isinstance(chunk.data, bytes) else str(chunk.data),
        )


async def _stream_wav(api, submit: Callable[[int], str], num_chunks: int, pending: list[str],
                      first_iter, first, sample_rate: int):
    yield media_io.wav_stream_header(sample_rate)
    for index in range(num_chunks):
        if len(pending) < num_chunks:
            # Keep the next chunk generating while this one plays.
            pending.append(submit(len(pending)))
        if index == 0:
            iterator, head = first_iter, first
        else:
            iterator, head = api.iter_result_chunks(pending[index]), None
        if head is not None and head.modality == "audio" and head.data:
            yield head.data
        async for c in iterator:
            # Mid-stream the status is already sent; closing the stream is the
            # only honest signal left, so raise rather than end quietly.
            _raise_if_error(c)
            if c.modality == "audio" and c.data:
                yield c.data
