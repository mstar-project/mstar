"""/v1/audio/speech and /v1/audio/voices handlers (text-to-speech).

Non-streaming speech returns the full audio as a container blob (WAV by
default). Streaming returns one open-ended response as the audio is produced:
a WAV header followed by PCM16 frames, or, with ``response_format="pcm"``,
the bare PCM16 frames that the OpenAI TTS clients (LiveKit, Pipecat) expect.
A request streams when it sets ``stream`` or OpenAI's ``stream_format``, which
is what the OpenAI SDK sends: ``"audio"`` streams the bytes as above, ``"sse"``
wraps them in ``speech.audio.delta`` events closed by ``speech.audio.done``.
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
import base64
from collections.abc import Callable

import anyio
from fastapi import HTTPException
from fastapi.responses import Response

from mstar.api_server import media_io
from mstar.api_server.openai._util import error_type, release_unread, rid, sse
from mstar.api_server.openai.protocol import VoiceCard, VoiceList
from mstar.api_server.openai.speech_chunking import split_sentences
from mstar.api_server.streaming_response import ReleasingStreamingResponse
from mstar.utils.generation_kwargs import offset_seed


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
    split = getattr(adapter, "split_speech_text", None)
    chunks = split(text) if split else split_sentences(text, max_chars=getattr(adapter, "speech_chunk_max_chars", 400))
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
        kwargs["seed"] = offset_seed(seed, index)
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


async def create_speech(api, model_name, adapter, req, raw_request=None):
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
    # OpenAI's own streaming switch, which its SDK (and so LiveKit's TTS plugin)
    # sends instead of ``stream``. Popped so it does not reach the model as a kwarg.
    stream_format = (req.model_extra or {}).pop("stream_format", None)
    if stream_format not in (None, "audio", "sse"):
        raise HTTPException(status_code=400, detail=f"stream_format must be 'audio' or 'sse', got {stream_format!r}")
    # A compressed container (the SDK's default mp3) cannot be streamed here;
    # stream_format then falls back to one whole body, which those clients read too.
    stream = bool(req.stream) or (
        stream_format is not None and (req.response_format or "wav").lower() in media_io.audio_formats(stream=True)
    )
    if not stream:
        stream_format = None
    # before any upload or GPU work: a format we cannot produce is the client's error
    try:
        fmt = media_io.check_audio_format(req.response_format, stream, sample_rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # blocking work (base64 decode, file write, an allowed remote fetch) off the loop
    try:
        args = await asyncio.to_thread(adapter.speech_to_request, req, api.upload_dir)
    except ValueError as exc:
        # the adapter refused the request's fields (a bad data URL, a server path)
        raise HTTPException(status_code=400, detail=str(exc)) from None
    # Every chunk reads the same reference clip, and a stream submits the later
    # ones while it plays, so a returned stream schedules the cleanup itself.
    stream_owns_uploads = False
    try:
        request_id = rid("speech")
        chunks = _plan_chunks(req, adapter, args.text or "")

        def submit(index: int) -> str:
            chunk_id = request_id if len(chunks) == 1 else f"{request_id}-{index}"
            return api.submit_request(
                text=chunks[index],
                file_paths=args.file_paths,
                input_modalities=args.input_modalities,
                output_modalities=args.output_modalities,
                model_kwargs=_chunk_kwargs(args.model_kwargs, index),
                streaming=stream,
                request_id=chunk_id,
            )

        lookahead = max(1, int(getattr(adapter, "speech_chunk_lookahead", 2)))
        pending = [submit(i) for i in range(min(lookahead, len(chunks)))]
        if stream:
            raw_pcm = fmt == "pcm"
            # a request the engine rejects gets its status, not an empty WAV
            try:
                first_iter = await api.open_result_stream(pending[0], raw_request)
            except BaseException:
                # The first piece released itself; release the ones submitted ahead.
                release_unread(api, pending, 1)
                raise
            if raw_request is not None and await raw_request.is_disconnected():
                # The client left before the first audio and no response has
                # started, so nothing else would release the pieces submitted ahead.
                release_unread(api, pending, 0)
                return Response(status_code=499)

            def release_unstarted() -> None:
                # What _stream_pcm's finally does, for a body that never ran it.
                release_unread(api, pending, 0)
                media_io.schedule_upload_cleanup(args.file_paths, api.upload_dir)

            stream_owns_uploads = True
            body = _stream_pcm(api, submit, len(chunks), pending, first_iter, sample_rate,
                               with_wav_header=not raw_pcm, file_paths=args.file_paths)
            if stream_format == "sse":
                return _ClosingStreamingResponse(
                    _as_sse(body), release=release_unstarted,
                    media_type="text/event-stream", headers={"Cache-Control": "no-cache"},
                )
            return _ClosingStreamingResponse(
                body, release=release_unstarted,
                media_type="audio/pcm" if raw_pcm else "audio/wav", headers={"Cache-Control": "no-cache"},
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
            release_unread(api, pending, read)
    finally:
        if not stream_owns_uploads:
            media_io.schedule_upload_cleanup(args.file_paths, api.upload_dir)
    audio_bytes, mime = media_io.pcm16_to_container(b"".join(pcm_parts), sample_rate, fmt)
    return Response(content=audio_bytes, media_type=mime)


class _ClosingStreamingResponse(ReleasingStreamingResponse):
    """Closes its body iterator however the response ends. Starlette stops at
    the failed or cancelled ``send`` when a client leaves mid-stream and leaves
    the generator suspended at its ``yield``, so its release of the sentence
    pieces still in flight would wait for garbage collection. A body that never
    started is released by ``ReleasingStreamingResponse`` first, and closing it
    afterwards runs nothing."""

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()


async def _as_sse(audio):
    """OpenAI's ``stream_format="sse"``: one ``speech.audio.delta`` per piece of
    audio (base64, in ``response_format``), then ``speech.audio.done``. A failure
    after the stream opened ends it with an ``error`` event instead of
    ``speech.audio.done``: the 200 is already sent."""
    try:
        async for data in audio:
            yield sse({"type": "speech.audio.delta", "audio": base64.b64encode(data).decode("ascii")})
    except HTTPException as exc:
        yield sse({"type": "error", "error": {
            "message": str(exc.detail), "type": error_type(exc.status_code), "code": exc.status_code,
        }})
        return
    finally:
        await audio.aclose()
    yield sse({"type": "speech.audio.done"})


def _is_error(chunk) -> bool:
    return chunk is not None and chunk.modality == "error"


def _http_error(chunk) -> HTTPException:
    """A data-worker failure arrives as an ``error`` chunk; the HTTP error it carries."""
    return HTTPException(
        status_code=int((chunk.metadata or {}).get("status", 500)),
        detail=chunk.data.decode("utf-8", "replace") if isinstance(chunk.data, bytes) else str(chunk.data),
    )


async def _stream_pcm(api, submit: Callable[[int], str], num_chunks: int, pending: list[str],
                      first_iter, sample_rate: int, with_wav_header: bool = True,
                      file_paths: dict[str, list[str]] | None = None):
    read = 0  # pieces read to the end; the rest are released if the stream stops early
    try:
        if with_wav_header:
            yield media_io.wav_stream_header(sample_rate)
        for index in range(num_chunks):
            if len(pending) < num_chunks:
                # Keep the next chunk generating while this one plays.
                pending.append(submit(len(pending)))
            iterator = first_iter if index == 0 else api.iter_result_chunks(pending[index])
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
        release_unread(api, pending, read)
        if file_paths:
            media_io.schedule_upload_cleanup(file_paths, api.upload_dir)
