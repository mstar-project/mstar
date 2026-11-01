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
* sustained speed: RTF (wall time / input duration) and audio-s/s overall.

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
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


VOICE_THRESHOLD = 0.02  # window RMS, full scale 1.0
MIN_VOICED_S = 0.1


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


def run_cell(url: str, audio: str, sessions: int, speech_end: float, voice: str, buffer_s: float,
             timeout: float, voice_threshold: float = VOICE_THRESHOLD, min_voiced_s: float = MIN_VOICED_S) -> dict:
    with wave.open(audio) as w:
        input_samples, rate = w.getnframes(), w.getframerate()
    if rate != SAMPLE_RATE:
        raise ValueError(f"{audio} is {rate} Hz; give a {SAMPLE_RATE} Hz clip")
    results = [{} for _ in range(sessions)]
    threads = [threading.Thread(target=run_session, args=(url, audio, voice, timeout, r)) for r in results]
    began = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - began
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    for n in args.sessions:
        detector = {"voice_threshold": args.voice_threshold, "min_voiced_s": args.min_voiced}
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
