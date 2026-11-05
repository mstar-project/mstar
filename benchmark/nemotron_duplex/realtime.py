#!/usr/bin/env python3
"""Concurrent OpenAI Realtime conversations against a full-duplex server's ``/v1/realtime``.

Each session streams the same user clip in real time (``--chunk-ms`` appends,
paced to the clip's own clock) in the input format given, then keeps streaming
silence until ``--seconds`` of audio have gone out, and closes. It reports per
concurrency:

* reply lag: per response, its wall time (``response.created`` to
  ``response.done``) less the audio it carried. The model emits one 80 ms
  frame per 80 ms of input, so a server that keeps up holds this near zero; a
  lag that grows with concurrency means output is falling behind real time;
* whole steps: whether every response's audio is a whole number of 80 ms
  steps at the output rate, as the conversation path promises for every codec
  and rate;
* the API server's CPU (``--server-pid``, the ``mstar serve`` process, which
  runs the resampling), in cores, over the timed window.

    python -m benchmark.nemotron_duplex.realtime --url http://127.0.0.1:8000 --audio user_16k.wav \\
        --sessions 16 32 48 --rate 24000 --server-pid $(pgrep -of 'mstar serve') --output out.json

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


def _encode(samples: np.ndarray, codec: str) -> bytes:
    """Float samples in [-1, 1] to the wire bytes of ``codec`` (``pcm``: PCM16)."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    if codec == "pcm":
        return pcm.tobytes()
    from mstar.api_server.media_io import g711_encode

    return g711_encode(pcm, f"audio/{codec}")


def _format(codec: str, rate: int) -> dict:
    return {"type": "audio/pcm", "rate": rate} if codec == "pcm" else {"type": f"audio/{codec}"}


def _cpu_seconds(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def run_session(args, clip: np.ndarray, start_at: float, out: dict) -> None:
    import websockets.sync.client

    in_rate = args.rate if args.codec == "pcm" else 8000
    out_rate = args.out_rate if args.out_codec == "pcm" else 8000
    out_bytes_per_s = out_rate * (2 if args.out_codec == "pcm" else 1)
    chunk = int(in_rate * args.chunk_ms / 1000)
    total = int(in_rate * args.seconds)
    audio = np.zeros(total, np.float32)
    audio[:min(total, clip.size)] = clip[:total]
    responses, errors, open_ = [], [], {}
    url = args.url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/realtime"
    try:
        with websockets.sync.client.connect(url, max_size=None) as ws:
            def receive():
                while True:
                    try:
                        event = json.loads(ws.recv())
                    except Exception:  # noqa: BLE001 — the socket closed
                        return
                    now, kind = time.perf_counter(), event.get("type")
                    if kind == "response.created":
                        open_[event["response"]["id"]] = [now, 0]
                    elif kind == "response.output_audio.delta":
                        open_[event["response_id"]][1] += len(base64.b64decode(event["delta"]))
                    elif kind == "response.done":
                        started, nbytes = open_.pop(event["response"]["id"])
                        responses.append({"wall_s": now - started, "audio_s": nbytes / out_bytes_per_s,
                                          "status": event["response"]["status"]})
                    elif kind == "error":
                        errors.append(event["error"].get("message", "")[:200])
            ws.recv()   # session.created
            ws.send(json.dumps({"type": "session.update", "session": {"audio": {
                "input": {"format": _format(args.codec, in_rate)},
                "output": {"format": _format(args.out_codec, out_rate)}}}}))
            receiver = threading.Thread(target=receive, daemon=True)
            receiver.start()
            while time.perf_counter() < start_at:
                time.sleep(0.001)
            t0 = time.perf_counter()
            for k, at in enumerate(range(0, total, chunk)):
                # absolute schedule: append k leaves when its audio has been "spoken"
                time.sleep(max(0.0, t0 + k * args.chunk_ms / 1000 - time.perf_counter()))
                piece = _encode(audio[at:at + chunk], args.codec)
                ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(piece).decode()}))
            time.sleep(args.tail_s)   # the last reply's end
    except Exception as exc:  # noqa: BLE001 — reported per session
        errors.append(repr(exc)[:200])
    step_bytes = round(out_bytes_per_s * STEP_S)
    out.update(responses=responses, errors=errors,
               whole_steps=all(round(r["audio_s"] * out_bytes_per_s) % step_bytes == 0 for r in responses))


def run_batch(args, clip: np.ndarray, n: int) -> dict:
    results = [{} for _ in range(n)]
    start = time.perf_counter() + 1.0
    threads = [threading.Thread(target=run_session, args=(args, clip, start + i * args.stagger_s, results[i]))
               for i in range(n)]
    cpu0 = _cpu_seconds(args.server_pid) if args.server_pid else None
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    responses = [r for s in results for r in s["responses"]]
    lags = sorted(r["wall_s"] - r["audio_s"] for r in responses)

    def pct(p):
        return round(lags[min(len(lags) - 1, int(p * len(lags)))], 3) if lags else None
    return {
        "sessions": n, "responses": len(responses),
        "failed_responses": sum(r["status"] == "failed" for r in responses),
        "sessions_with_errors": sum(bool(s["errors"]) for s in results),
        "errors": sorted({e for s in results for e in s["errors"]})[:5],
        "whole_steps": all(s["whole_steps"] for s in results),
        "reply_lag_s_p50": pct(0.5), "reply_lag_s_p95": pct(0.95), "reply_lag_s_max": pct(1.0),
        "api_server_cpu_cores": None if cpu0 is None else round((_cpu_seconds(args.server_pid) - cpu0) / elapsed, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--audio", required=True, help="user-only mono clip")
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--codec", choices=("pcm", "pcmu", "pcma"), default="pcm")
    ap.add_argument("--rate", type=int, default=24000, help="input PCM16 rate (pcm only)")
    ap.add_argument("--out-codec", choices=("pcm", "pcmu", "pcma"), default="pcm")
    ap.add_argument("--out-rate", type=int, default=24000, help="output PCM16 rate (pcm only)")
    ap.add_argument("--chunk-ms", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="audio each session sends, clip then silence")
    ap.add_argument("--tail-s", type=float, default=3.0)
    ap.add_argument("--stagger-s", type=float, default=0.25)
    ap.add_argument("--server-pid", type=int, default=None, help="the API server process, for its CPU")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import soundfile as sf
    from scipy.signal import resample_poly

    clip, sr = sf.read(args.audio, dtype="float32")
    clip = clip if clip.ndim == 1 else clip.mean(axis=1)
    in_rate = args.rate if args.codec == "pcm" else 8000
    g = np.gcd(sr, in_rate)
    clip = resample_poly(clip, in_rate // g, sr // g).astype(np.float32)

    rows = []
    for n in args.sessions:
        row = run_batch(args, clip, n)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=1)
    lags = [r["reply_lag_s_p50"] for r in rows if r["reply_lag_s_p50"] is not None]
    if lags:
        print("median of p50 reply lags:", statistics.median(lags))


if __name__ == "__main__":
    main()
