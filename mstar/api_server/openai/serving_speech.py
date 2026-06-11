"""/v1/audio/speech handler (text-to-speech).

Non-streaming returns the full audio as a container blob (WAV by default).
Streaming returns a single open-ended WAV response (header + PCM16 frames) as
the audio is produced.
"""

from __future__ import annotations

from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from mstar.api_server import media_io
from mstar.api_server.openai._util import rid


async def create_speech(api, model_name, adapter, req):  # noqa: ARG001
    args = adapter.speech_to_request(req, api.upload_dir)
    request_id = rid("speech")
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
    fmt = (req.response_format or "wav").lower()

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
        return StreamingResponse(
            _stream_wav(api, request_id, sample_rate),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"},
        )

    chunks = await run_in_threadpool(api.collect_results, request_id)
    pcm = b"".join(c.data for c in chunks if c.modality == "audio")
    audio_bytes, mime = media_io.pcm16_to_container(pcm, sample_rate, fmt)
    return Response(content=audio_bytes, media_type=mime)


async def _stream_wav(api, request_id, sample_rate):
    yield media_io.wav_stream_header(sample_rate)
    async for c in api.iter_result_chunks(request_id):
        if c.modality == "audio" and c.data:
            yield c.data
