"""/v1/audio/speech and /v1/audio/voices handlers (text-to-speech).

Non-streaming speech returns the full audio as a container blob (WAV by
default). Streaming returns one open-ended response as the audio is produced:
a WAV header followed by PCM16 frames, or, with ``response_format="pcm"``,
the bare PCM16 frames that the OpenAI TTS clients (LiveKit, Pipecat) expect.
``/v1/audio/voices`` lists the ``voice`` values the served model accepts.

Long inputs can be synthesized as ordered sentence chunks (one engine request
per chunk, ``speech_chunking.split_sentences``): the adapter's
``speech_chunk_min_chars`` turns this on for long texts, and a client can force
or suppress it per request with ``sentence_chunking: true|false``. The next
chunks are submitted while the current one streams, so the engine batches them
and playback never waits for a prefill. An input needing more than the
adapter's ``speech_chunk_max_pieces`` chunks is rejected.

A streaming request only commits to HTTP 200 once its first result chunk has
arrived and is not an error; an error chunk before that becomes the HTTP error
it carries (the non-streaming path gets the same from ``collect_results``).

The adapter maps the request before anything is submitted, and that step can
decode an uploaded clip or, where the server allows it, fetch one over HTTP.
It runs in a worker thread so a slow reference cannot stall the streams the
event loop is serving.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from mstar.api_server import media_io
from mstar.api_server.openai._util import rid
from mstar.api_server.openai.protocol import VoiceCard, VoiceList
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
    max_pieces = getattr(adapter, "speech_chunk_max_pieces", 32)
    if len(chunks) > max_pieces:
        raise HTTPException(
            status_code=400,
            detail=f"input splits into {len(chunks)} sentence chunks; at most {max_pieces} are allowed",
        )
    return chunks or [text]


def _chunk_kwargs(model_kwargs: dict, index: int) -> dict:
    """Per-chunk model kwargs: identical, except a client seed advances per chunk."""
    kwargs = dict(model_kwargs)
    kwargs.pop("sentence_chunking", None)
    seed = kwargs.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool) and index:
        # the conductor's seed is an int64: a seed near the top wraps instead of overflowing
        kwargs["seed"] = (seed + index) % 2**63
    return kwargs


def list_voices(api) -> VoiceList | None:
    """The served model's voices, or ``None`` when it has no fixed list."""
    model = api.model
    voices = model.get_voices() if model is not None else None
    if voices is None:
        return None
    return VoiceList(
        voices=[VoiceCard(id=v, name=v) for v in voices],
        default_voice=model.get_default_voice(),
    )


async def create_speech(api, model_name, adapter, req, raw_request=None):  # noqa: ARG001
    # blocking work (base64 decode, file write, an allowed remote fetch) off the loop
    try:
        args = await asyncio.to_thread(adapter.speech_to_request, req, api.upload_dir)
    except ValueError as exc:
        # the adapter refused the request's fields (a bad data URL, a server path)
        raise HTTPException(status_code=400, detail=str(exc)) from None
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
    pending = [submit(i) for i in range(min(lookahead, len(chunks)))]
    if req.stream:
        raw_pcm = fmt == "pcm"
        # Look at the first result before committing to a 200: a request the
        # engine rejects (bad voice, dead worker, ...) must surface as an HTTP
        # error, not as an empty WAV.
        first_iter = api.iter_result_chunks(pending[0])
        try:
            first = await anext(first_iter, None)
            if _is_error(first):
                # Read it to the end so the finished request isn't aborted.
                await _drain(first_iter)
                raise _http_error(first)
        except BaseException:
            # The first piece released itself; release the ones submitted ahead.
            _release_unread(api, pending, 1)
            raise
        return StreamingResponse(
            _stream_pcm(api, submit, len(chunks), pending, first_iter, first, sample_rate,
                        with_wav_header=not raw_pcm),
            media_type="audio/pcm" if raw_pcm else "audio/wav",
            headers={"Cache-Control": "no-cache"},
        )

    pcm_parts: list[bytes] = []
    read = 0  # pieces collect_results has released; the rest are released on early exit
    try:
        for index in range(len(chunks)):
            if len(pending) < len(chunks):
                pending.append(submit(len(pending)))
            try:
                results = await api.collect_results(pending[index], raw_request)
            except HTTPException:
                read = index + 1
                raise
            read = index + 1
            if not results and raw_request is not None and await raw_request.is_disconnected():
                break
            pcm_parts.append(b"".join(c.data for c in results if c.modality == "audio"))
    finally:
        _release_unread(api, pending, read)
    audio_bytes, mime = media_io.pcm16_to_container(b"".join(pcm_parts), sample_rate, fmt)
    return Response(content=audio_bytes, media_type=mime)


def _is_error(chunk) -> bool:
    return chunk is not None and chunk.modality == "error"


def _http_error(chunk) -> HTTPException:
    """A data-worker failure arrives as an ``error`` chunk; the HTTP error it carries."""
    return HTTPException(
        status_code=int((chunk.metadata or {}).get("status", 500)),
        detail=chunk.data.decode("utf-8", "replace") if isinstance(chunk.data, bytes) else str(chunk.data),
    )


async def _drain(iterator) -> None:
    async for _ in iterator:
        pass


def _release_unread(api, pending: list[str], read: int) -> None:
    """Release every submitted piece from ``pending[read]`` on (not yet fully read)."""
    for chunk_id in pending[read:]:
        api.release_request(chunk_id)


async def _stream_pcm(api, submit: Callable[[int], str], num_chunks: int, pending: list[str],
                      first_iter, first, sample_rate: int, with_wav_header: bool = True):
    read = 0  # pieces read to the end; the rest are released if the stream stops early
    try:
        if with_wav_header:
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
            error = None
            async for c in iterator:
                # Read past an error to the end: leaving early would abort a
                # request that has already finished.
                if error is None and _is_error(c):
                    error = c
                elif error is None and c.modality == "audio" and c.data:
                    yield c.data
            read = index + 1
            if error is not None:
                # Mid-stream the status is already sent; closing the stream is
                # the only honest signal left, so raise rather than end quietly.
                raise _http_error(error)
    finally:
        _release_unread(api, pending, read)
