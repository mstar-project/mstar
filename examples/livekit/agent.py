"""A LiveKit Agents voice agent whose STT, LLM and TTS all run on M*.

Only LiveKit's stock OpenAI plugins are used, each pointed at an M* server
through ``base_url``. M* serves one model per process, so the pipeline is three
servers (see README.md):

    STT  Whisper large-v3   POST /v1/audio/transcriptions
    LLM  Qwen3.5            POST /v1/chat/completions (streamed)
    TTS  Kokoro / Qwen3-TTS POST /v1/audio/speech (streamed bare PCM)

Turn detection runs in the agent (Silero VAD), so Whisper gets one finished
utterance per turn and no server-side VAD is needed.

    python examples/livekit/agent.py start     # or `dev` for hot reload

Environment (defaults match README.md's ports):

    LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
    MSTAR_STT_URL  MSTAR_LLM_URL  MSTAR_TTS_URL       base URLs ending in /v1
    MSTAR_STT_MODEL  MSTAR_LLM_MODEL  MSTAR_TTS_MODEL  MSTAR_TTS_VOICE
    MSTAR_METRICS_LOG   append each turn's LiveKit metrics here as JSON lines
    AGENT_MAX_JOBS      take at most this many conversations, whatever the host CPU load
"""

from __future__ import annotations

import json
import logging
import os
import time

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    TurnHandlingOptions,
    cli,
)
from livekit.plugins import openai, silero

logger = logging.getLogger("mstar-livekit-agent")

STT_URL = os.environ.get("MSTAR_STT_URL", "http://localhost:8001/v1")
LLM_URL = os.environ.get("MSTAR_LLM_URL", "http://localhost:8002/v1")
TTS_URL = os.environ.get("MSTAR_TTS_URL", "http://localhost:8003/v1")
STT_MODEL = os.environ.get("MSTAR_STT_MODEL", "whisper_large")
LLM_MODEL = os.environ.get("MSTAR_LLM_MODEL", "qwen3_5_4b")
TTS_MODEL = os.environ.get("MSTAR_TTS_MODEL", "kokoro")
TTS_VOICE = os.environ.get("MSTAR_TTS_VOICE", "af_heart")
METRICS_LOG = os.environ.get("MSTAR_METRICS_LOG")

INSTRUCTIONS = (
    "You are a friendly voice assistant. Your replies are spoken aloud, so answer in one or two "
    "short sentences of plain text, with no lists, markdown or emoji."
)


def build_session(vad) -> AgentSession:
    return AgentSession(
        vad=vad,
        # The default turn detector is a LiveKit Cloud model; without Cloud
        # credentials every turn waits out max_delay (2.5 s). VAD silence alone
        # ends the turn after min_delay.
        turn_handling=TurnHandlingOptions(turn_detection="vad", endpointing={"min_delay": 0.5}),
        # use_realtime=False: Whisper answers the batch endpoint; the plugin's
        # realtime transcription speaks the GA session protocol, which M* does not serve.
        stt=openai.STT(base_url=STT_URL, api_key="none", model=STT_MODEL, language="en", use_realtime=False),
        llm=openai.LLM(
            base_url=LLM_URL,
            api_key="none",
            model=LLM_MODEL,
            temperature=0.7,
            max_completion_tokens=200,
            # Qwen3.5's template opens a <think> block by default, which the TTS would read out.
            extra_body={"enable_thinking": False},
        ),
        # pcm: bare 16-bit samples, no client decode. The plugin assumes 24 kHz,
        # which is Kokoro's and Qwen3-TTS's native rate.
        tts=openai.TTS(base_url=TTS_URL, api_key="none", model=TTS_MODEL, voice=TTS_VOICE, response_format="pcm"),
    )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


def _load_by_jobs(server: AgentServer) -> float:
    return len(server.active_jobs) / MAX_JOBS


# The worker reports host CPU as its load by default, and the LiveKit server
# stops dispatching to a worker near full load. On a shared or GPU-bound host
# that refuses rooms the agent could serve, so AGENT_MAX_JOBS caps by conversations.
MAX_JOBS = int(os.environ.get("AGENT_MAX_JOBS", "0"))
server = AgentServer(
    setup_fnc=prewarm,
    **({"load_fnc": _load_by_jobs, "load_threshold": 1.0} if MAX_JOBS else {}),
)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = build_session(ctx.proc.userdata["vad"])

    if METRICS_LOG:
        @session.on("metrics_collected")
        def _on_metrics(ev: MetricsCollectedEvent) -> None:
            record = {"room": ctx.room.name, "wall": time.time(), **ev.metrics.model_dump(mode="json")}
            with open(METRICS_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")

    await session.start(agent=Agent(instructions=INSTRUCTIONS), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
