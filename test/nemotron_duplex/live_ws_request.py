#!/usr/bin/env python3
"""Stream a user clip into a running Nemotron-Duplex call over ``/generate/ws``.

The clip goes in as 80 ms appends on an absolute schedule (or as fast as the
socket takes them with ``--no-pace``), then ``end_input``; the agent's text and
audio stream back while it is still being sent. Reports, per session:

- the agent text, token by token;
- unit lag: from the last sample of input frame ``k`` leaving the client to the
  agent's text token for frame ``k`` arriving (one token per frame);
- time to the first agent audio.

``--compare-one-shot`` also sends the whole clip to ``/generate``, and once
more over the socket as one append carrying ``end_input``: the same input as
the upload, cut into the same frame steps, so that call's text must match the
upload's token for token, save the last frame (the correctness gate for the
client stream). The last frame is encoded in a step of its own when the end of
input reaches the encoder after the audio, which can happen on either side.
The paced call is compared too, for information only: appends of a whole
number of frames reach the encoder as the upload's pieces, but other sizes cut
the clip differently, and under bf16 a different chunking can flip a
knife-edge frame.

    python test/nemotron_duplex/live_ws_request.py                       # demo clip, one call
    python test/nemotron_duplex/live_ws_request.py --chunk-ms 400 --compare-one-shot
    python test/nemotron_duplex/live_ws_request.py --sessions 8 --json out.json
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field

import msgpack
import numpy as np
from _audio_prep import prepare_user_clip
from _env import get_base_url, load_env

SR = 16000
FRAME_S = 0.08


@dataclass
class Session:
    request_id: str
    sent_at: list[float] = field(default_factory=list)       # per append, after it was sent
    tokens: list[tuple[float, str]] = field(default_factory=list)
    audio: bytearray = field(default_factory=bytearray)
    first_audio_s: float | None = None
    error: str | None = None
    finished: bool = False


def _wav_bytes(samples: np.ndarray) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _load_clip(path: str | None, trailing_s: float) -> np.ndarray:
    import soundfile as sf

    path = path or prepare_user_clip(trailing_s)
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio.functional as AF

        data = AF.resample(torch.from_numpy(data), sr, SR).numpy()
    return data


def run_session(url: str, clip: np.ndarray, args, session: Session, start_at: float,
                whole: bool = False) -> None:
    """One live call: the clip in paced appends then ``end_input``, or with
    ``whole`` as one append carrying ``end_input``."""
    import websockets.sync.client

    chunk = int(SR * args.chunk_ms / 1000)
    open_msg = {"request_id": session.request_id, "output_modalities": ["text", "audio"],
                "model_kwargs": {"temperature": args.temperature}}
    if args.prompt:
        open_msg["text"] = args.prompt
    with websockets.sync.client.connect(url, max_size=None) as ws:
        def receive():
            from websockets.exceptions import ConnectionClosed

            while True:
                try:
                    reply = msgpack.unpackb(ws.recv(), raw=False)
                except ConnectionClosed:
                    session.error = session.error or "connection closed before finish"
                    return
                now = time.perf_counter()
                if reply.get("error"):
                    session.error = reply["error"]
                    return
                if reply.get("finish"):
                    session.finished = True
                    return
                if reply.get("modality") == "text":
                    session.tokens.append((now, bytes(reply["data"]).decode("utf-8", "replace")))
                elif reply.get("modality") == "audio":
                    if session.first_audio_s is None:
                        session.first_audio_s = now - start_at
                    session.audio += bytes(reply["data"])

        receiver = threading.Thread(target=receive, daemon=True)
        while time.perf_counter() < start_at:
            time.sleep(0.001)
        ws.send(msgpack.packb(open_msg, use_bin_type=True))
        receiver.start()
        if whole:
            ws.send(msgpack.packb({
                "request_id": session.request_id, "end_input": True,
                "append": [{"name": "clip.wav", "data": _wav_bytes(clip)}],
            }, use_bin_type=True))
            session.sent_at.append(time.perf_counter())
            receiver.join(timeout=args.timeout)
            return
        for k, at in enumerate(range(0, len(clip), chunk)):
            if not args.no_pace:
                # absolute schedule: append k leaves when its audio has been "spoken"
                deadline = start_at + (k + 1) * args.chunk_ms / 1000
                while time.perf_counter() < deadline:
                    time.sleep(0.0005)
            ws.send(msgpack.packb({
                "request_id": session.request_id,
                "append": [{"name": "chunk.wav", "data": _wav_bytes(clip[at:at + chunk])}],
            }, use_bin_type=True))
            session.sent_at.append(time.perf_counter())
        ws.send(msgpack.packb({"request_id": session.request_id, "end_input": True}, use_bin_type=True))
        receiver.join(timeout=args.timeout)


def unit_lags(session: Session, chunk_ms: float) -> list[float]:
    """Seconds from the append that makes frame k final to frame k's token.

    Encoder frame k reads mel frames up to 8k, and mel frame m reads samples up
    to m * 160 + 255, so frame k is final once sample 1280 k + 255 has arrived;
    frame 0 once sample 256 has (the featurizer reflects the first 256)."""
    if not session.sent_at:
        return []
    chunk = int(SR * chunk_ms / 1000)
    lags = []
    for k, (arrived, _text) in enumerate(session.tokens):
        last = min(max(1280 * k + 255, 256) // chunk, len(session.sent_at) - 1)
        lags.append(arrived - session.sent_at[last])
    return lags


def one_shot_tokens(base_url: str, clip: np.ndarray, args) -> list[str]:
    from mstar.client.client import MStarClient
    from mstar.client.types import TextChunk

    events = MStarClient(base_url, timeout=args.timeout).stream(
        text=args.prompt or None, audio=("clip.wav", _wav_bytes(clip)),
        output_modalities=("text", "audio"), temperature=args.temperature,
    )
    return [e.text for e in events if isinstance(e, TextChunk)]


def _compare(name: str, live: list[str], reference: list[str]) -> dict:
    """Where ``live`` first departs from the one-shot text, frame by frame."""
    first = next((i for i, (a, b) in enumerate(zip(live, reference, strict=False)) if a != b),
                 None if len(live) == len(reference) else min(len(live), len(reference)))
    row = {"call": name, "one_shot_tokens": len(reference), "live_tokens": len(live),
           "identical": first is None, "first_differing_frame": first,
           "identical_but_last_frame": first is None or (
               len(live) == len(reference) and first == len(reference) - 1)}
    if first is not None:
        row["live_from_there"] = "".join(live[first:first + 8])
        row["one_shot_from_there"] = "".join(reference[first:first + 8])
    return row


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default=None, help="16 kHz mono user clip (default: the demo's first user turn)")
    ap.add_argument("--trailing-s", type=float, default=8.0, help="silence after the demo turn")
    ap.add_argument("--url", default=None, help="server base URL (default from .env)")
    ap.add_argument("--chunk-ms", type=float, default=80.0)
    ap.add_argument("--deadline-ms", type=float, default=80.0, help="a frame later than this counts as late")
    ap.add_argument("--no-pace", action="store_true", help="send appends as fast as possible")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--stagger-s", type=float, default=0.0, help="start offset between sessions")
    ap.add_argument("--prompt", default=None, help="system prompt")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--compare-one-shot", action="store_true")
    ap.add_argument("--output", default=None, help="write session 0's agent audio here (22.05 kHz wav)")
    ap.add_argument("--json", default=None, help="write the per-session summary here")
    args = ap.parse_args()
    if args.sessions < 1:
        ap.error("--sessions must be at least 1")

    base = (args.url or get_base_url()).rstrip("/")
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/generate/ws"
    clip = _load_clip(args.audio, args.trailing_s)
    sessions = [Session(request_id=f"live-{uuid.uuid4().hex[:8]}-{i}") for i in range(args.sessions)]
    t0 = time.perf_counter() + 0.5
    threads = [threading.Thread(target=run_session, args=(ws_url, clip, args, s, t0 + i * args.stagger_s))
               for i, s in enumerate(sessions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = []
    for s in sessions:
        lags = unit_lags(s, args.chunk_ms)
        frame_budget_misses = sum(lag > args.deadline_ms / 1000 for lag in lags)
        row = {
            "request_id": s.request_id, "finished": s.finished, "error": s.error,
            "tokens": len(s.tokens), "appends": len(s.sent_at),
            "text": "".join(t for _a, t in s.tokens),
            "first_audio_s": s.first_audio_s, "audio_bytes": len(s.audio),
            "lag_p50_ms": 1000 * statistics.median(lags) if lags else None,
            "lag_p95_ms": 1000 * float(np.percentile(lags, 95)) if lags else None,
            "lag_max_ms": 1000 * max(lags) if lags else None,
            f"frames_later_than_{args.deadline_ms:g}ms": frame_budget_misses,
        }
        summary.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}))
    print("agent text (session 0):", summary[0]["text"])

    ok = all(r["finished"] and not r["error"] for r in summary)
    if args.compare_one_shot:
        reference = one_shot_tokens(base, clip, args)
        whole = Session(request_id=f"live-{uuid.uuid4().hex[:8]}-whole")
        run_session(ws_url, clip, args, whole, time.perf_counter(), whole=True)
        for name, s in (("whole_clip_append", whole), ("paced", sessions[0])):
            row = _compare(name, [t for _a, t in s.tokens], reference)
            print(json.dumps(row))
            if name == "whole_clip_append":       # like for like: the gate
                ok = ok and whole.finished and not whole.error and row["identical_but_last_frame"]
    if args.output and sessions[0].audio:
        with wave.open(args.output, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(bytes(sessions[0].audio))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
