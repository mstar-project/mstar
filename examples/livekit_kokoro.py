"""Kokoro on M* as the TTS of a LiveKit Agents voice pipeline.

LiveKit's OpenAI TTS plugin talks to any OpenAI-compatible ``/v1/audio/speech``;
point its ``base_url`` at an M* server running ``mstar serve kokoro``.
``response_format="pcm"`` skips the client-side decode: Kokoro emits 24 kHz
mono PCM, which is exactly what the plugin expects from OpenAI.

    pip install "livekit-agents[openai]"
    mstar serve kokoro
    python examples/livekit_kokoro.py            # synthesizes one line to kokoro_livekit.wav

Drop the same ``tts`` object into an ``AgentSession`` (see ``voice_agent`` below)
to use it in a real room; ``voice`` accepts any bundled voice or a blend such as
``af_bella(2)+af_sky(1)`` (``GET /v1/audio/voices`` lists them).
"""

import asyncio
import os
import wave

from livekit.plugins import openai

MSTAR_BASE_URL = os.environ.get("MSTAR_BASE_URL", "http://localhost:8000/v1")


def kokoro_tts(voice: str = "af_heart", speed: float = 1.0) -> openai.TTS:
    return openai.TTS(
        base_url=MSTAR_BASE_URL,
        api_key="none",  # M* does not check it; the plugin insists on a value
        model="kokoro",
        voice=voice,
        speed=speed,
        response_format="pcm",
    )


def voice_agent(tts: openai.TTS):
    """How the same TTS plugs into a LiveKit voice agent (needs STT + LLM
    credentials of your choice; shown for wiring, not run by this script)."""
    from livekit.agents import Agent, AgentSession

    session = AgentSession(
        stt=openai.STT(),  # any LiveKit STT plugin
        llm=openai.LLM(),  # any LiveKit LLM plugin
        tts=tts,
    )
    agent = Agent(instructions="You are a helpful voice assistant.")
    return session, agent


async def main() -> None:
    tts = kokoro_tts()
    with wave.open("kokoro_livekit.wav", "wb") as out:
        out.setnchannels(tts.num_channels)
        out.setsampwidth(2)
        out.setframerate(tts.sample_rate)
        async with tts.synthesize("Hello from LiveKit Agents, speaking through Kokoro on M star.") as stream:
            async for event in stream:
                out.writeframes(bytes(event.frame.data))
    await tts.aclose()
    print("wrote kokoro_livekit.wav")


if __name__ == "__main__":
    asyncio.run(main())
