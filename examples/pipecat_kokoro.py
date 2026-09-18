"""Kokoro on M* as the TTS of a Pipecat pipeline.

Pipecat's ``OpenAITTSService`` streams ``response_format="pcm"`` from any
OpenAI-compatible ``/v1/audio/speech``; point its ``base_url`` at an M* server
running ``mstar serve kokoro``. Kokoro's 24 kHz mono output matches the sample
rate the service assumes for OpenAI.

    pip install "pipecat-ai[openai]"
    mstar serve kokoro
    python examples/pipecat_kokoro.py            # synthesizes one line to kokoro_pipecat.wav

The script runs a two-stage pipeline (TTS -> WAV sink) driven by a
``TTSSpeakFrame``; in a real bot the same ``tts`` sits after the LLM
(``Pipeline([transport.input(), stt, llm, tts, transport.output()])``).
"""

import asyncio
import os
import wave

from pipecat.frames.frames import EndFrame, Frame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai import tts as openai_tts
from pipecat.services.openai.tts import OpenAITTSService

MSTAR_BASE_URL = os.environ.get("MSTAR_BASE_URL", "http://localhost:8000/v1")
SAMPLE_RATE = 24000  # Kokoro's output rate


def kokoro_tts(voice: str = "af_heart", speed: float = 1.0) -> OpenAITTSService:
    # The service only forwards voices from OpenAI's own list; register the
    # Kokoro voice (or blend such as "af_bella(2)+af_sky(1)") so it passes
    # through to M* unchanged.
    openai_tts.VALID_VOICES[voice] = voice
    return OpenAITTSService(
        base_url=MSTAR_BASE_URL,
        api_key="none",  # M* does not check it; the client insists on a value
        settings=OpenAITTSService.Settings(model="kokoro", voice=voice, speed=speed),
    )


class WavSink(FrameProcessor):
    """Collects the TTS audio frames and writes them as one WAV at the end."""

    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._pcm: list[bytes] = []
        self._sample_rate = SAMPLE_RATE

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self._pcm.append(frame.audio)
            self._sample_rate = frame.sample_rate
        elif isinstance(frame, EndFrame):
            with wave.open(self._path, "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(self._sample_rate)
                out.writeframes(b"".join(self._pcm))
            print(f"wrote {self._path}")
        await self.push_frame(frame, direction)


async def main() -> None:
    pipeline = Pipeline([kokoro_tts(), WavSink("kokoro_pipecat.wav")])
    task = PipelineTask(pipeline, params=PipelineParams(audio_out_sample_rate=SAMPLE_RATE))
    await task.queue_frames([TTSSpeakFrame("Hello from Pipecat, speaking through Kokoro on M star."), EndFrame()])
    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
