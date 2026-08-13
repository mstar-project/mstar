"""OpenAI Realtime-API-compatible WebSocket server for the Nemotron VoiceChat
duplex model (Phase C).

This is a thin protocol layer over the model's *streaming* duplex inference
(Phase B / ``NemotronDuplexModel.create_stream``). It speaks a subset of the
OpenAI Realtime API events so existing Realtime clients can talk to it:

  client -> server:
    - session.update                      (config: voice, output/ input formats)
    - input_audio_buffer.append           (base64 PCM16 mono @ input_sample_rate)
    - input_audio_buffer.commit / clear
    - response.create / response.cancel

  server -> client:
    - session.created / session.updated
    - response.created
    - response.audio.delta                (base64 PCM16 @ 24k -> here 22.05k)
    - response.audio_transcript.delta     (agent text, streamed)
    - response.audio.done / response.done
    - error

The model is *full-duplex*: it consumes user audio continuously and emits agent
audio+text as it generates, so we run in a server-VAD-like continuous mode —
every appended audio chunk is fed to the stream and any generated agent
audio/transcript is pushed back as deltas. Turn boundaries (barge-in / end of
utterance) come from the model's own RNN-T turn signals when available.

Run: ``python -m mstar.model.nemotron_duplex.realtime_api --host 0.0.0.0 --port 8000``
(requires ``fastapi`` + ``uvicorn``). The heavy model streaming lives in the
model; this file only does event framing.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging

logger = logging.getLogger(__name__)

# 80 ms/frame; the model emits agent audio at the codec sample rate (22.05 kHz).
DEFAULT_INPUT_SR = 16000
DEFAULT_OUTPUT_SR = 22050


class RealtimeConnection:
    """Per-WebSocket session: bridges Realtime events to a model duplex stream.

    ``stream`` is a ``DuplexStream`` from ``NemotronDuplexModel.create_stream``
    (Phase B), exposing:
        - ``append_audio(pcm_int16_bytes) -> list[Output]`` where each ``Output``
          has ``.audio`` (int16 bytes @ output_sr) and/or ``.text`` (str delta),
        - ``reset()``.
    """

    def __init__(self, model, send_json, send_defaults: dict | None = None):
        self.model = model
        self.send_json = send_json                 # async callable(dict)
        self.cfg = {
            "voice": "Aria",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_sample_rate": DEFAULT_INPUT_SR,
            "output_sample_rate": DEFAULT_OUTPUT_SR,
            "temperature": 0.0,
            **(send_defaults or {}),
        }
        self.stream = None
        self._response_open = False

    async def start(self):
        self.stream = self.model.create_stream(
            voice=self.cfg["voice"], temperature=self.cfg["temperature"],
        )
        await self.send_json({"type": "session.created", "session": self.cfg})

    async def handle(self, event: dict):
        et = event.get("type")
        if et == "session.update":
            self.cfg.update(event.get("session", {}))
            await self.send_json({"type": "session.updated", "session": self.cfg})
        elif et == "input_audio_buffer.append":
            pcm = base64.b64decode(event["audio"])
            await self._consume(self.stream.append_audio(pcm))
        elif et == "input_audio_buffer.commit":
            await self._consume(self.stream.flush())
            await self._finish_response()
        elif et == "input_audio_buffer.clear":
            self.stream.reset()
        elif et == "response.cancel":
            self.stream.reset()
            await self._finish_response()
        elif et in ("response.create", "session.close"):
            pass  # continuous mode: responses stream as audio arrives
        else:
            await self.send_json({"type": "error", "error": {"message": f"unhandled event {et!r}"}})

    async def _consume(self, outputs):
        """Push generated agent audio/text deltas from a stream step."""
        for out in outputs or ():
            if not self._response_open:
                self._response_open = True
                await self.send_json({"type": "response.created"})
            if getattr(out, "text", None):
                await self.send_json({"type": "response.audio_transcript.delta", "delta": out.text})
            if getattr(out, "audio", None):
                await self.send_json({
                    "type": "response.audio.delta",
                    "delta": base64.b64encode(out.audio).decode("ascii"),
                })

    async def _finish_response(self):
        if self._response_open:
            await self.send_json({"type": "response.audio.done"})
            await self.send_json({"type": "response.done"})
            self._response_open = False


def build_app(model):
    """Build a FastAPI app exposing ``/v1/realtime`` for the given loaded model."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket):
        await ws.accept()
        conn = RealtimeConnection(model, ws.send_json)
        await conn.start()
        try:
            while True:
                conn_event = await ws.receive_text()
                await conn.handle(json.loads(conn_event))
        except WebSocketDisconnect:
            logger.info("realtime client disconnected")

    return app


def main():
    ap = argparse.ArgumentParser(description="Nemotron VoiceChat OpenAI-Realtime server")
    ap.add_argument("--model", default="nemotron_duplex")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import uvicorn

    from mstar.model.registry import HF_MODELS, get_model_class
    model = get_model_class(args.model)(**HF_MODELS[args.model])
    model.prepare_streaming(device=args.device)  # warm the submodules (Phase B)
    uvicorn.run(build_app(model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
