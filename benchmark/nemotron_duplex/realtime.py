#!/usr/bin/env python3
"""Concurrent OpenAI Realtime conversations against a full-duplex server's ``/v1/realtime``.

Each session streams the same user clip in real time (``--chunk-ms`` appends,
paced to the clip's own clock) in the input format given, then silence until
``--seconds`` of audio have gone out, and closes once its replies are done.
Sessions connect ``--stagger-s`` apart, each led by silence so that every
session speaks at once; the timed window runs from the last session's start to
the first session's end, when all of them are live. Per concurrency it reports:

* reply start: the first reply's first audio, after the user stops speaking
  (``--speech-end``, seconds into the clip);
* underrun: within a reply, how far its audio arrives behind its own playback
  clock, ``max_k(arrival_k - arrival_0 - audio_before_k)``. A server that keeps
  up holds this near zero; one that grows with concurrency is falling behind
  real time;
* whole steps: whether every reply's audio is a whole number of 80 ms steps at
  the output rate, as the conversation path promises for every codec and rate;
* replies not completed (failed, incomplete, or still open at the end);
* CPU over the window, in cores: the API server's (``--server-pid``, the
  process listening on the port, which runs the resampling), and its event-loop
  thread's alone, which runs the short resamples inline; and this client's,
  with how late its sends ran (a row whose sends ran over 50 ms late measures
  the client, not the server).

The model's own rates are 16 kHz in and 22.05 kHz out, so a client at those
rates is resampled on neither side::

    PID=$(ss -ltnpH "sport = :8000" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    python -m benchmark.nemotron_duplex.realtime --url http://127.0.0.1:8000 --audio user_16k.wav \\
        --speech-end 2.7 --sessions 16 32 48 --rate 16000 --out-rate 22050 --server-pid $PID --output native.json
    python -m benchmark.nemotron_duplex.realtime --url http://127.0.0.1:8000 --audio user_16k.wav \\
        --speech-end 2.7 --sessions 16 32 48 --rate 24000 --out-rate 24000 --server-pid $PID --output resampled.json

``--audio`` is a user-only mono clip (``test/nemotron_duplex/_audio_prep.py``
makes one from the checkpoint's demo). ``--codec pcmu``/``pcma`` sends G.711 at
8 kHz; ``--out-codec``/``--out-rate`` pick the output format.
"""
import argparse
import base64
import json
import os
import statistics
import threading
import time

import numpy as np

STEP_S = 0.08
LATE_SEND_S = 0.05


def _encode(samples: np.ndarray, codec: str) -> bytes:
    """Float samples in [-1, 1] to the wire bytes of ``codec`` (``pcm``: PCM16)."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    if codec == "pcm":
        return pcm.tobytes()
    from mstar.api_server.media_io import g711_encode

    return g711_encode(pcm, f"audio/{codec}")


def _format(codec: str, rate: int) -> dict:
    return {"type": "audio/pcm", "rate": rate} if codec == "pcm" else {"type": f"audio/{codec}"}


def _rates(args) -> tuple[int, int, int]:
    """Input rate, output rate, output bytes per second."""
    in_rate = args.rate if args.codec == "pcm" else 8000
    out_rate = args.out_rate if args.out_codec == "pcm" else 8000
    return in_rate, out_rate, out_rate * (2 if args.out_codec == "pcm" else 1)


def _cpu_seconds(path: str) -> float:
    """utime + stime of a ``/proc/.../stat`` (a process: all its threads)."""
    with open(path) as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def run_session(args, clip: np.ndarray, start_at: float, lead_s: float, out: dict) -> None:
    import websockets.sync.client

    in_rate, out_rate, out_bps = _rates(args)
    lead = int(in_rate * lead_s)
    total = lead + int(in_rate * args.seconds)
    audio = np.zeros(total, np.float32)
    audio[lead:lead + min(clip.size, total - lead)] = clip[:total - lead]
    chunk = int(in_rate * args.chunk_ms / 1000)
    replies, errors, open_ = [], [], {}
    late, t_speech_end = 0.0, None
    url = args.url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/realtime"
    try:
        with websockets.sync.client.connect(url, max_size=None) as ws:
            def receive():
                while True:
                    try:
                        event = json.loads(ws.recv())
                    except Exception:  # noqa: BLE001 — the socket closed
                        return
                    try:
                        now, kind = time.perf_counter(), event.get("type")
                        if kind == "response.created":
                            # [created, first audio, audio bytes so far, worst underrun]
                            open_[event["response"]["id"]] = [now, None, 0, 0.0]
                        elif kind == "response.output_audio.delta":
                            r = open_[event["response_id"]]
                            if r[1] is None:
                                r[1] = now
                            r[3] = max(r[3], now - r[1] - r[2] / out_bps)
                            r[2] += len(base64.b64decode(event["delta"]))
                        elif kind == "response.done":
                            created, first, nbytes, underrun = open_.pop(event["response"]["id"])
                            replies.append({"status": event["response"]["status"], "created": created,
                                            "first_audio": first, "bytes": nbytes, "underrun_s": underrun})
                        elif kind == "error":
                            errors.append(event["error"].get("message", "")[:200])
                    except Exception as exc:  # noqa: BLE001 — reported, not a dead receiver
                        errors.append(f"receiver: {exc!r}"[:200])

            ws.recv()   # session.created
            ws.send(json.dumps({"type": "session.update", "session": {"audio": {
                "input": {"format": _format(args.codec, in_rate)},
                "output": {"format": _format(args.out_codec, out_rate)}}}}))
            receiver = threading.Thread(target=receive, daemon=True)
            receiver.start()
            time.sleep(max(0.0, start_at - time.perf_counter()))
            t0 = time.perf_counter()
            if args.speech_end is not None:
                t_speech_end = t0 + lead_s + args.speech_end
            for at in range(0, total, chunk):
                # absolute schedule: each append leaves when its audio has been "spoken"
                due = t0 + at / in_rate
                late = max(late, time.perf_counter() - due)
                time.sleep(max(0.0, due - time.perf_counter()))
                piece = _encode(audio[at:at + chunk], args.codec)
                ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(piece).decode()}))
            end = time.perf_counter() + args.tail_s
            # a reply still playing finishes first, and one that hasn't started gets its chance
            while (open_ or not replies) and time.perf_counter() < end:
                time.sleep(0.05)
        receiver.join(5)
    except Exception as exc:  # noqa: BLE001 — reported per session
        errors.append(repr(exc)[:200])
    step = round(out_bps * STEP_S)
    started = [r["first_audio"] for r in replies if r["first_audio"] is not None]
    out.update(
        replies=replies, errors=errors, unfinished=len(open_), send_late_s=late,
        whole_steps=all(r["bytes"] % step == 0 for r in replies) if replies else None,
        reply_start_s=(min(started) - t_speech_end) if started and t_speech_end is not None else None,
    )


def _pct(values: list[float], p: float) -> float | None:
    values = sorted(values)
    return round(values[min(len(values) - 1, int(p * len(values)))], 3) if values else None


def run_batch(args, clip: np.ndarray, n: int) -> dict:
    results = [{} for _ in range(n)]
    start = time.perf_counter() + 1.0
    last_start = start + (n - 1) * args.stagger_s
    # each session leads with silence to the last one's start, so all speak (and end) together
    threads = [threading.Thread(target=run_session, args=(args, clip, start + i * args.stagger_s,
                                                           (n - 1 - i) * args.stagger_s, results[i]))
               for i in range(n)]
    for t in threads:
        t.start()
    window = (last_start, last_start + args.seconds)
    procs = (f"/proc/{args.server_pid}/stat", f"/proc/{args.server_pid}/task/{args.server_pid}/stat") \
        if args.server_pid else ()
    time.sleep(max(0.0, window[0] - time.perf_counter()))
    cpu0 = [_cpu_seconds(p) for p in procs]
    client0 = sum(os.times()[:2])
    time.sleep(max(0.0, window[1] - time.perf_counter()))
    span = args.seconds
    cpu = [round((_cpu_seconds(p) - c) / span, 2) for p, c in zip(procs, cpu0, strict=True)]
    client_cpu = round((sum(os.times()[:2]) - client0) / span, 2)
    for t in threads:
        t.join()
    replies = [r for s in results for r in s["replies"]]
    late = max(s["send_late_s"] for s in results)
    starts = [s["reply_start_s"] for s in results if s["reply_start_s"] is not None]
    return {
        "sessions": n, "replies": len(replies),
        "not_completed": sum(r["status"] != "completed" for r in replies) + sum(s["unfinished"] for s in results),
        "sessions_without_a_reply": sum(not s["replies"] for s in results),
        "sessions_with_errors": sum(bool(s["errors"]) for s in results),
        "errors": sorted({e for s in results for e in s["errors"]})[:5],
        "whole_steps": None if not replies else all(s["whole_steps"] is not False for s in results),
        "reply_start_s_p50": _pct(starts, 0.5), "reply_start_s_max": _pct(starts, 1.0),
        "underrun_s_p50": _pct([r["underrun_s"] for r in replies], 0.5),
        "underrun_s_max": _pct([r["underrun_s"] for r in replies], 1.0),
        "api_server_cpu_cores": cpu[0] if cpu else None,
        "api_event_loop_cpu_cores": cpu[1] if cpu else None,
        "client_cpu_cores": client_cpu, "client_send_late_s_max": round(late, 3),
        "client_bound": late > LATE_SEND_S,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--audio", required=True, help="user-only mono clip")
    ap.add_argument("--speech-end", type=float, default=None, help="seconds into the clip where the user stops")
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1, help="untimed batches at the largest --sessions first")
    ap.add_argument("--codec", choices=("pcm", "pcmu", "pcma"), default="pcm")
    ap.add_argument("--rate", type=int, default=24000, help="input PCM16 rate (pcm only)")
    ap.add_argument("--out-codec", choices=("pcm", "pcmu", "pcma"), default="pcm")
    ap.add_argument("--out-rate", type=int, default=24000, help="output PCM16 rate (pcm only)")
    ap.add_argument("--chunk-ms", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="audio each session sends after its lead")
    ap.add_argument("--tail-s", type=float, default=10.0, help="longest wait for replies still open at the end")
    ap.add_argument("--stagger-s", type=float, default=0.25)
    ap.add_argument("--server-pid", type=int, default=None, help="the API server process, for its CPU")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import soundfile as sf
    from scipy.signal import resample_poly

    clip, sr = sf.read(args.audio, dtype="float32")
    clip = clip if clip.ndim == 1 else clip.mean(axis=1)
    in_rate = _rates(args)[0]
    g = np.gcd(sr, in_rate)
    clip = resample_poly(clip, in_rate // g, sr // g).astype(np.float32)

    for _ in range(args.warmup):
        run_batch(args, clip, max(args.sessions))
    rows = []
    for n in args.sessions:
        for rep in range(args.repeats):
            row = {"repeat": rep, **run_batch(args, clip, n)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=1)
    under = [r["underrun_s_p50"] for r in rows if r["underrun_s_p50"] is not None]
    if under:
        print("median of p50 underruns (s):", statistics.median(under))


if __name__ == "__main__":
    main()
