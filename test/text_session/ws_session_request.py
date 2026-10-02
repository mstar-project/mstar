#!/usr/bin/env python3
"""Drive a persistent session over ``/generate/ws``, the control-loop socket.

Same proof as ``session_request.py``, over the surface a control loop actually
uses: one socket, a turn per message, the second turn resuming the first with
nothing about it in its prompt. The session comes back as the first frame of
each reply, and a refused session is answered in-band without dropping the
socket.

    CUDA_VISIBLE_DEVICES=7 bash test/text_session/launch_server.sh
    python test/text_session/ws_session_request.py --url ws://localhost:8137
"""

import argparse
import asyncio
import json
import sys

import websockets

FIRST = "Remember this number: 8675309. Reply with just the word OK."
SECOND = "What was the number I asked you to remember? Answer with digits only."
NUMBER = "8675309"


async def _turn(ws, request_id: str, text: str, **session) -> list[dict]:
    """Send one message and collect its frames up to finish (or error)."""
    await ws.send(json.dumps({
        "text": text, "request_id": request_id,
        "output_modalities": "text",
        "model_kwargs": {"max_output_tokens": 32},
        **session,
    }))
    frames = []
    while True:
        frame = json.loads(await ws.recv())
        frames.append(frame)
        if frame.get("finish") or frame.get("error"):
            return frames


def _text(frames: list[dict]) -> str:
    import base64
    return "".join(
        base64.b64decode(f["data"]).decode("utf-8", "replace")
        for f in frames if f.get("modality") == "text"
    )


def _session_of(frames: list[dict]) -> str | None:
    for f in frames:
        if f.get("modality") == "session":
            return f["metadata"]["session_id"]
    return None


async def run(url: str) -> int:
    checks: dict[str, bool] = {}
    async with websockets.connect(f"{url}/generate/ws", max_size=None) as ws:
        first = await _turn(ws, "w1", FIRST, start_session=True)
        session_id = _session_of(first)
        print(f"[turn 1] session {session_id}: {_text(first)!r}", flush=True)
        checks["the socket reports the session it opened"] = bool(session_id)
        if not session_id:
            return 1

        second = await _turn(
            ws, "w2", SECOND, resume_session=True, session_id=session_id,
        )
        print(f"[turn 2] {_text(second)!r}", flush=True)
        checks["the resumed turn recalls the first"] = NUMBER in _text(second)
        checks["each reply names its session"] = (
            _session_of(second) == session_id
        )

        control = await _turn(ws, "w3", SECOND)
        print(f"[control (no session)] {_text(control)!r}", flush=True)
        checks["a sessionless turn does not"] = NUMBER not in _text(control)

        refused = await _turn(
            ws, "w4", SECOND, resume_session=True, session_id="no-such-session",
        )
        print(f"[refused] {refused[-1]}", flush=True)
        checks["an unknown session is refused in-band with a 404"] = (
            refused[-1].get("status") == 404
        )

        # the socket survived the refusal
        after = await _turn(ws, "w5", "Say OK.")
        checks["the socket survives a refusal"] = bool(
            after[-1].get("finish")
        )

        ended = await _turn(
            ws, "w6", "Say OK.", resume_session=True,
            session_id=session_id, end_session=True,
        )
        print(f"[ended] {ended[-1]}", flush=True)
        checks["a turn can end its session"] = bool(ended[-1].get("finish"))

    print()
    for what, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {what}", flush=True)
    return 0 if all(checks.values()) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="ws://localhost:8000")
    args = ap.parse_args(argv)
    return asyncio.run(run(args.url))


if __name__ == "__main__":
    sys.exit(main())
