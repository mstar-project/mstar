"""/v1/audio/speech and /v1/audio/voices handlers (text-to-speech).

Non-streaming speech returns the full audio as a container blob (WAV by
default). Streaming returns one open-ended response as the audio is produced:
a WAV header followed by PCM16 frames, or, with ``response_format="pcm"``,
the bare PCM16 frames that the OpenAI TTS clients (LiveKit, Pipecat) expect.
``/v1/audio/voices`` lists the ``voice`` values the served model accepts.
"""

from __future__ import annotations

from fastapi.responses import Response, StreamingResponse

from mstar.api_server import media_io
from mstar.api_server.openai._util import rid
from mstar.api_server.openai.protocol import VoiceCard, VoiceList


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
        raw_pcm = fmt == "pcm"
        return StreamingResponse(
            _stream_pcm(api, request_id, sample_rate, with_wav_header=not raw_pcm),
            media_type="audio/pcm" if raw_pcm else "audio/wav",
            headers={"Cache-Control": "no-cache"},
        )

    chunks = await api.collect_results(request_id, raw_request)
    pcm = b"".join(c.data for c in chunks if c.modality == "audio")
    audio_bytes, mime = media_io.pcm16_to_container(pcm, sample_rate, fmt)
    return Response(content=audio_bytes, media_type=mime)


async def _stream_pcm(api, request_id, sample_rate, with_wav_header=True):
    if with_wav_header:
        yield media_io.wav_stream_header(sample_rate)
    async for c in api.iter_result_chunks(request_id):
        if c.modality == "audio" and c.data:
            yield c.data
