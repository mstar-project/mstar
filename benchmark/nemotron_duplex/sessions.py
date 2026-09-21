#!/usr/bin/env python3
"""Concurrent full-duplex sessions against a Nemotron-Duplex (VoiceChat-11B) server.

Replays the same user clip on N concurrent streamed ``/generate`` requests and
reports, per protocol (BENCHMARK_PROTOCOL.md, full-duplex S2S row):

* response latency: end of the user's speech -> the agent's first text token
  (the first non-special token; the talker voices it a few frames later), in
  interaction time (frames x 80 ms) and in wall time from request start. The
  first voiced audio frame is reported too, but the codec emits low-level
  sound during the user's turn, so text is the reliable turn signal;
* whether every session kept up with real time: a session of F frames must
  finish in <= F x 80 ms of wall time for its ticks to be "on time"
  (the server processes a replayed clip as fast as it can, so wall time per
  frame is the tick budget it would need live);
* per-frame wall time (ms) per session and aggregate frames/s.

    python -m benchmark.nemotron_duplex.sessions --url http://127.0.0.1:8019 --sessions 1 4 8 16 \
        --audio user_16k.wav --speech-end 2.7 --repeats 3 --output results/duplex_sessions.json

``--audio`` is a user-only 16 kHz mono clip with trailing silence for the reply
(``test/nemotron_duplex/_audio_prep.py`` makes one from the checkpoint's demo);
``--speech-end`` is where the user stops talking in that clip, in seconds.
"""
import argparse
import json
import statistics
import sys
import threading
import time

from mstar.client.client import MStarClient
from mstar.client.types import AudioChunk, TextChunk

FRAME_S = 0.08              # 12.5 Hz frames
SILENCE_RMS = 300           # int16 RMS below this = silent audio (the codec's idle output sits below)


def _rms_int16(pcm: bytes) -> float:
    import numpy as np

    if not pcm:
        return 0.0
    a = np.frombuffer(pcm, dtype="<i2").astype("float32")
    return float((a ** 2).mean() ** 0.5) if a.size else 0.0


def run_session(url: str, audio: str, temperature: float, out: dict) -> None:
    c = MStarClient(url, timeout=1800)
    t0 = time.time()
    frames = 0
    first_text_wall = first_audio_wall = None
    first_text_frame = first_audio_frames = None
    audio_secs = 0.0
    text = []
    try:
        for ev in c.generate(audio=audio, input_modalities=("audio",), output_modalities=("text", "audio"),
                             temperature=temperature, stream=True):
            now = time.time() - t0
            if isinstance(ev, TextChunk):
                frames += 1
                text.append(ev.text)
                if first_text_wall is None and ev.text.strip() and not ev.text.strip().startswith("<"):
                    first_text_wall = now
                    first_text_frame = frames - 1
            elif isinstance(ev, AudioChunk):
                dur = (len(ev.pcm) // 2) / (ev.sample_rate or 22050)
                if first_audio_wall is None and _rms_int16(ev.pcm) >= SILENCE_RMS:
                    first_audio_wall = now
                    first_audio_frames = audio_secs / FRAME_S
                audio_secs += dur
    except Exception as e:  # noqa: BLE001 - report the failure, keep the other sessions
        out["error"] = repr(e)
    out.update(
        wall_s=time.time() - t0, frames=frames, text="".join(text), audio_s=audio_secs,
        first_text_wall_s=first_text_wall, first_text_frame=first_text_frame,
        first_audio_wall_s=first_audio_wall, first_audio_frame=first_audio_frames,
    )


def run_batch(url: str, audio: str, n: int, temperature: float, speech_end_s: float) -> dict:
    results = [dict() for _ in range(n)]
    threads = [threading.Thread(target=run_session, args=(url, audio, temperature, results[i])) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ok = [r for r in results if "error" not in r and r["frames"]]
    frames = [r["frames"] for r in ok]
    per_frame_ms = [1000 * r["wall_s"] / r["frames"] for r in ok]
    on_time = [r["wall_s"] <= r["frames"] * FRAME_S for r in ok]
    # response latency: user speech end -> the agent's first text token, in
    # interaction time (which frame of the stream) and in wall time
    resp_frames = [r["first_text_frame"] * FRAME_S - speech_end_s for r in ok if r["first_text_frame"] is not None]
    resp_wall = [r["first_text_wall_s"] for r in ok if r["first_text_wall_s"] is not None]
    audio_frames = [r["first_audio_frame"] * FRAME_S for r in ok if r["first_audio_frame"] is not None]
    texts = {r["text"] for r in ok}
    return dict(
        sessions=n, completed=len(ok), failed=n - len(ok), wall_s=wall,
        frames_per_session=frames[0] if frames and len(set(frames)) == 1 else frames,
        aggregate_frames_per_s=sum(frames) / wall if wall else None,
        per_frame_ms_p50=statistics.median(per_frame_ms) if per_frame_ms else None,
        per_frame_ms_max=max(per_frame_ms) if per_frame_ms else None,
        ticks_on_time=all(on_time) if on_time else False,
        response_latency_interaction_s_p50=statistics.median(resp_frames) if resp_frames else None,
        response_latency_wall_s_p50=statistics.median(resp_wall) if resp_wall else None,
        response_latency_wall_s_p95=(sorted(resp_wall)[max(0, int(0.95 * len(resp_wall)) - 1)] if resp_wall else None),
        first_voiced_audio_s_p50=statistics.median(audio_frames) if audio_frames else None,
        identical_text=len(texts) == 1,
        errors=[r.get("error") for r in results if "error" in r],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8019")
    ap.add_argument("--audio", required=True, help="user-only 16 kHz mono clip with trailing silence")
    ap.add_argument("--speech-end", type=float, required=True, help="seconds into the clip where the user stops")
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="untimed single sessions first (kernel JIT, first touch)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", default=None, help="JSON file for the raw per-run results")
    args = ap.parse_args()

    for _ in range(args.warmup):
        run_batch(args.url, args.audio, 1, args.temperature, args.speech_end)

    runs = []
    print(f"{'sessions':>8} {'completed':>9} {'wall s':>7} {'ms/frame p50':>12} {'ms/frame max':>12} "
          f"{'on time':>7} {'resp s (interaction)':>21} {'resp s wall p50/p95':>20} {'same text':>9}")
    for n in args.sessions:
        for rep in range(args.repeats):
            r = run_batch(args.url, args.audio, n, args.temperature, args.speech_end)
            r["repeat"] = rep
            runs.append(r)
            fmt = lambda v, w=6: (f"{v:{w}.2f}" if isinstance(v, float) else f"{str(v):>{w}}")  # noqa: E731
            print(f"{n:>8} {r['completed']:>9} {fmt(r['wall_s'], 7)} {fmt(r['per_frame_ms_p50'], 12)} "
                  f"{fmt(r['per_frame_ms_max'], 12)} {str(r['ticks_on_time']):>7} "
                  f"{fmt(r['response_latency_interaction_s_p50'], 21)} "
                  f"{fmt(r['response_latency_wall_s_p50'], 9)}/{fmt(r['response_latency_wall_s_p95'], 9)} "
                  f"{str(r['identical_text']):>9}")
            if r["errors"]:
                print("   errors:", r["errors"][:3])
    if args.output:
        json.dump(dict(url=args.url, audio=args.audio, speech_end_s=args.speech_end, runs=runs),
                  open(args.output, "w"), indent=1)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
