"""Concurrent streamed PersonaPlex sessions: when audio arrives, when the agent speaks, and whether playback keeps up.

Replays one user clip (speech, then trailing silence for the reply) on N
concurrent streamed ``/generate`` requests, as fast as the server goes, and
reports per session, from timestamped chunk arrivals (monotonic clock):

* time to first audio: request start -> the first audio chunk arrives;
* response onset on the audio timeline: the start of the first agent speech
  segment beginning at or after ``speech_end - window``, as signed seconds
  from the user's speech end (negative = the agent began first). An agent
  segment already running at speech end is reported separately, not folded
  into a near-zero onset. Segments are runs of 20 ms windows whose RMS is at
  least ``--voice-threshold`` (default 0.02, -34 dBFS; PersonaPlex replies idle
  at or below about 0.009 and speak at 0.05-0.2), lasting at least
  ``--min-voiced`` seconds. Both are recorded with the results;
* delivery against playback: playback starts ``buffer`` after the first chunk
  and runs in real time; a chunk that arrives after its playback time pauses
  playback until it does (an underrun), and later deadlines shift by the stall;
* sustained speed: RTF (wall time / input duration), audio-s/s and req/s overall;
* goodput: sessions/s and audio-s/s counting only sessions that met the SLO,
  total stall under ``--stall-limit`` (10 ms) and time to first audio under
  ``--ttfa-limit`` (200 ms). The sessions meeting each limit on its own are
  reported too, as is the audio-s/s of those meeting the stall limit.

All N sessions of a cell start together, so it measures N simultaneous arrivals,
the hardest case for time to first audio. ``--client-procs`` splits the sessions
over client processes, so at high N a saturated Python client does not show up as
server stall.

Each cell first runs ``--warmup`` waves of N sessions whose results are
discarded, so a cold server's first requests do not count.

A cell is valid only if every one of its N sessions completes with the
input's length of audio; otherwise it is reported invalid and the script
exits non-zero. Headline throughput comes from ``benchmark.runner``; this
measures the streamed delivery that ``runner`` does not.

    python -m benchmark.personaplex.sessions --url http://127.0.0.1:8000 \\
        --audio user.wav --speech-end 3.1 --sessions 1 4 8 --output sessions.json
"""

import argparse
import json
import multiprocessing
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


VOICE_THRESHOLD = 0.02  # window RMS, full scale 1.0
MIN_VOICED_S = 0.1
STALL_LIMIT_S = 0.010
TTFA_LIMIT_S = 0.200


def speech_segments(pcm: np.ndarray, rate: int = SAMPLE_RATE, window_s: float = 0.02,
                    threshold: float = VOICE_THRESHOLD,
                    min_voiced_s: float = MIN_VOICED_S) -> list[tuple[float, float]]:
    """Voiced spans (seconds) of a float PCM track: runs of ``window_s`` windows with RMS >= ``threshold``.

    A fixed threshold, not one derived from the track: a reply that is mostly speech has no quiet
    percentile to calibrate on.
    """
    hop = int(rate * window_s)
    count = len(pcm) // hop
    if count == 0:
        return []
    rms = np.sqrt((pcm[: count * hop].reshape(count, hop).astype(np.float64) ** 2).mean(axis=1))
    voiced = rms >= threshold
    segments, start = [], None
    for i, on in enumerate([*voiced, False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if (i - start) * window_s >= min_voiced_s:
                segments.append((start * window_s, i * window_s))
            start = None
    return segments


def response_onset(segments: list[tuple[float, float]], speech_end: float, window: float = 1.0) -> dict:
    """The agent's response relative to the user's speech end, overlap-aware."""
    running = [seg for seg in segments if seg[0] < speech_end - window and seg[1] > speech_end]
    onset = next((seg for seg in segments if seg[0] >= speech_end - window), None)
    return {"already_speaking_at_speech_end": bool(running),
            "onset_s": None if onset is None else onset[0] - speech_end}


def playback(arrivals: list[tuple[float, int]], rate: int = SAMPLE_RATE, buffer_s: float = 0.32) -> dict:
    """Underruns and stall time for ``(arrival time, samples)`` chunks played from ``buffer_s`` after the first."""
    chunks = [(t, n) for t, n in arrivals if n > 0]  # an empty (final) chunk carries nothing to play
    if not chunks:
        return {"underruns": 0, "stall_s": 0.0}
    clock = chunks[0][0] + buffer_s
    underruns, stall = 0, 0.0
    for t, n in chunks:
        if t > clock:
            underruns += 1
            stall += t - clock
            clock = t
        clock += n / rate
    return {"underruns": underruns, "stall_s": stall}


def percentiles(values: list[float]) -> dict:
    """p50/p95/p99 and the maximum; ``None`` throughout for no values. Descriptive at small counts."""
    if not values:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    p50, p95, p99 = np.percentile(values, [50, 95, 99]).tolist()
    return {"p50": p50, "p95": p95, "p99": p99, "max": max(values)}


def slo_summary(rows: list[dict], wall: float, stall_limit_s: float = STALL_LIMIT_S,
                ttfa_limit_s: float = TTFA_LIMIT_S) -> dict:
    """Goodput over a cell's sessions. A session that did not complete meets neither limit."""
    done = [row for row in rows if row["complete"]]
    met_stall = [row for row in done if row["stall_s"] < stall_limit_s]
    met_ttfa = [row for row in done if row["time_to_first_audio_s"] < ttfa_limit_s]
    met_slo = [row for row in met_stall if row["time_to_first_audio_s"] < ttfa_limit_s]

    def audio_rate(subset):
        return sum(row["samples"] for row in subset) / SAMPLE_RATE / wall

    return {"stall_limit_s": stall_limit_s, "ttfa_limit_s": ttfa_limit_s,
            "ttfa_s": percentiles([row["time_to_first_audio_s"] for row in done]),
            "stall_s": percentiles([row["stall_s"] for row in done]),
            "met_stall": len(met_stall), "met_ttfa": len(met_ttfa), "met_slo": len(met_slo),
            "goodput_sessions_per_s": len(met_slo) / wall, "goodput_audio_s_per_s": audio_rate(met_slo),
            "stall_only_goodput_audio_s_per_s": audio_rate(met_stall)}


def run_session(url: str, audio: str, voice: str, timeout: float, out: dict) -> None:
    from mstar.client.client import MStarClient
    from mstar.client.types import AudioChunk

    start = time.monotonic()
    arrivals, pcm = [], []
    try:
        client = MStarClient(url, timeout=timeout)
        for event in client.generate(audio=audio, input_modalities=("audio",), output_modalities=("audio", "text"),
                                     voice=voice, greedy=True, stream=True):
            if isinstance(event, AudioChunk):
                samples = np.frombuffer(event.pcm, dtype="<i2")
                arrivals.append((time.monotonic() - start, len(samples)))
                pcm.append(samples)
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"session ran past {timeout} s")
    except Exception as exc:  # noqa: BLE001 - recorded; the cell is invalid
        out["error"] = repr(exc)
    out.update(wall_s=time.monotonic() - start, arrivals=arrivals,
               pcm=np.concatenate(pcm).astype(np.float32) / 32768 if pcm else np.zeros(0, np.float32))


def _run_threads(url: str, audio: str, voice: str, timeout: float, results: list[dict]) -> list[threading.Thread]:
    threads = [threading.Thread(target=run_session, args=(url, audio, voice, timeout, r)) for r in results]
    for thread in threads:
        thread.start()
    return threads


def _client_process(url: str, audio: str, voice: str, timeout: float, count: int, barrier, queue) -> None:
    results = [{} for _ in range(count)]
    barrier.wait()
    for thread in _run_threads(url, audio, voice, timeout, results):
        thread.join()
    queue.put(results)


def split_sessions(sessions: int, processes: int) -> list[int]:
    """Sessions per client process, as even as they go; no process without a session."""
    processes = max(1, min(processes, sessions))
    return [sessions // processes + (i < sessions % processes) for i in range(processes)]


def run_sessions(url: str, audio: str, voice: str, timeout: float, sessions: int,
                 client_procs: int = 1) -> tuple[list[dict], float]:
    """Every session's result and the wall time from their common start."""
    counts = split_sessions(sessions, client_procs)
    if len(counts) == 1:
        results = [{} for _ in range(sessions)]
        began = time.monotonic()
        for thread in _run_threads(url, audio, voice, timeout, results):
            thread.join()
        return results, time.monotonic() - began
    # Spawned before the clock starts: the barrier holds them until every client has imported.
    context = multiprocessing.get_context("spawn")
    barrier, queue = context.Barrier(len(counts) + 1), context.Queue()
    processes = [context.Process(target=_client_process, args=(url, audio, voice, timeout, count, barrier, queue))
                 for count in counts]
    for process in processes:
        process.start()
    barrier.wait(timeout=300)
    began = time.monotonic()
    # Read before joining: a child blocks on exit until its results are off the queue.
    results = [row for _ in processes for row in queue.get(timeout=timeout + 120)]
    wall = time.monotonic() - began
    for process in processes:
        process.join()
    return results, wall


def write_wav(path: Path, pcm: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((pcm * 32768).clip(-32768, 32767).astype("<i2").tobytes())


def run_cell(url: str, audio: str, sessions: int, speech_end: float, voice: str, buffer_s: float,
             timeout: float, voice_threshold: float = VOICE_THRESHOLD, min_voiced_s: float = MIN_VOICED_S,
             stall_limit_s: float = STALL_LIMIT_S, ttfa_limit_s: float = TTFA_LIMIT_S, client_procs: int = 1,
             save_audio: Path | None = None) -> dict:
    with wave.open(audio) as w:
        input_samples, rate = w.getnframes(), w.getframerate()
    if rate != SAMPLE_RATE:
        raise ValueError(f"{audio} is {rate} Hz; give a {SAMPLE_RATE} Hz clip")
    results, wall = run_sessions(url, audio, voice, timeout, sessions, client_procs)
    if save_audio is not None and len(results[0]["pcm"]) == input_samples:
        write_wav(save_audio, results[0]["pcm"])
    rows = []
    for r in results:
        samples = len(r["pcm"])
        complete = "error" not in r and samples == input_samples
        rows.append({
            "complete": complete, "error": r.get("error"), "samples": samples,
            "time_to_first_audio_s": r["arrivals"][0][0] if r["arrivals"] else None,
            "rtf": r["wall_s"] / (input_samples / SAMPLE_RATE),
            **playback(r["arrivals"], buffer_s=buffer_s),
            **response_onset(speech_segments(r["pcm"], threshold=voice_threshold, min_voiced_s=min_voiced_s),
                             speech_end),
            "arrivals": r["arrivals"],
        })
    return {"sessions": sessions, "valid": all(row["complete"] for row in rows),
            "completed": sum(row["complete"] for row in rows), "wall_s": wall,
            "audio_seconds_throughput": sum(row["samples"] for row in rows) / SAMPLE_RATE / wall,
            "requests_per_s": sum(row["complete"] for row in rows) / wall,
            **slo_summary(rows, wall, stall_limit_s, ttfa_limit_s),
            "buffer_s": buffer_s, "speech_end_s": speech_end, "voice_threshold": voice_threshold,
            "min_voiced_s": min_voiced_s, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--audio", required=True, help="24 kHz mono WAV: user speech, then trailing silence")
    parser.add_argument("--speech-end", type=float, required=True, help="where the user stops talking, in seconds")
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--voice", default="NATF2")
    parser.add_argument("--buffer", type=float, default=0.32, help="playback buffer after the first chunk, seconds")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--warmup", type=int, default=1, help="waves of N sessions run and discarded before each cell")
    parser.add_argument("--voice-threshold", type=float, default=VOICE_THRESHOLD,
                        help="speech window RMS, full scale 1")
    parser.add_argument("--min-voiced", type=float, default=MIN_VOICED_S, help="shortest speech segment, seconds")
    parser.add_argument("--stall-limit", type=float, default=STALL_LIMIT_S, help="SLO: total stall, seconds")
    parser.add_argument("--ttfa-limit", type=float, default=TTFA_LIMIT_S, help="SLO: time to first audio, seconds")
    parser.add_argument("--client-procs", type=int, default=1, help="client processes the sessions are split over")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    for n in args.sessions:
        detector = {"voice_threshold": args.voice_threshold, "min_voiced_s": args.min_voiced,
                    "stall_limit_s": args.stall_limit, "ttfa_limit_s": args.ttfa_limit,
                    "client_procs": args.client_procs}
        for _ in range(args.warmup):
            run_cell(args.url, args.audio, n, args.speech_end, args.voice, args.buffer, args.timeout, **detector)
        cell = run_cell(args.url, args.audio, n, args.speech_end, args.voice, args.buffer, args.timeout, **detector)
        cells.append(cell)
        print(json.dumps({k: v for k, v in cell.items() if k != "rows"}), flush=True)
    args.output.write_text(json.dumps({"audio": args.audio, "cells": cells}, indent=2) + "\n")
    invalid = [c["sessions"] for c in cells if not c["valid"]]
    if invalid:
        print(f"INVALID: cells with failed or incomplete sessions: {invalid}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
