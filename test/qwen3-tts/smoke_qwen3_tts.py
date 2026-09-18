#!/usr/bin/env python3
"""End-to-end smoke test of a running Qwen3-TTS server through ``/v1/audio/speech``.

Exercises the served path (API -> conductor -> Talker/RefEncoder -> codec
stream -> WAV) for whichever variant the server hosts, saving every WAV and
checking that audio is present, non-silent, of plausible length, and that a
greedy request is repeatable. Run inside the GPU allocation against
``mstar serve <key>``::

    python test/qwen3-tts/smoke_qwen3_tts.py --url http://127.0.0.1:8000 --variant custom_voice --out results/smoke
    python test/qwen3-tts/smoke_qwen3_tts.py --url http://127.0.0.1:8000 --variant base \\
        --ref-audio $BENCH/tts/ref/clone_2.wav --ref-text "Okay. Yeah. I resent you. ..."
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests

SAMPLE_RATE = 24000
TEXT = "The train to the coast leaves at seven tomorrow morning, so please pack your bag tonight."
LONG_TEXT = " ".join([
    "The train to the coast leaves at seven tomorrow morning, so please pack your bag tonight.",
    "She opened the window and let the cool evening air drift into the kitchen.",
    "Our meeting has been moved to Thursday afternoon because the conference room is being repainted.",
    "A gentle rain fell over the harbor while the fishing boats returned one by one.",
    "Remember to water the tomatoes twice a week during the hottest part of the summer.",
    "The museum's new exhibit traces the history of printing from wooden blocks to modern digital presses.",
    "He laughed so hard at the joke that he spilled coffee all over his notes.",
])


def speech(url: str, payload: dict, stream: bool, timeout: float) -> tuple[bytes, float, float]:
    """Return (pcm16 bytes, time to first audio, total time)."""
    start = time.perf_counter()
    first = None
    pcm = bytearray()
    with requests.post(f"{url}/v1/audio/speech", json={**payload, "stream": stream, "response_format": "wav"},
                       stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        skip = 44
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            if skip:
                drop = min(skip, len(chunk))
                chunk = chunk[drop:]
                skip -= drop
                if not chunk:
                    continue
            if first is None:
                first = time.perf_counter() - start
            pcm.extend(chunk)
    if not stream:
        # non-streaming: the WAV may carry a full header with the data length; PCM after byte 44
        pass
    return bytes(pcm), first if first is not None else float("nan"), time.perf_counter() - start


def write_wav(path: Path, pcm: bytes) -> None:
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1,
                         SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b"data", len(pcm))
    path.write_bytes(header + pcm)


def stats(pcm: bytes) -> dict:
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
    return {
        "seconds": len(audio) / SAMPLE_RATE,
        "peak": float(np.abs(audio).max()) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--variant", choices=("custom_voice", "voice_design", "base"), required=True)
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--out", default="results/smoke")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-instruct", action="store_true", help="skip the instruction case (0.6B CustomVoice)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base: dict = {"model": args.variant, "language": "English"}
    if args.variant == "custom_voice":
        base["voice"] = args.voice
    elif args.variant == "voice_design":
        base["instructions"] = "A clear, friendly adult female voice with a neutral accent."
    else:
        if not args.ref_audio:
            sys.exit("--ref-audio is required for the base variant")
        base["ref_audio"] = "data:audio/wav;base64," + base64.b64encode(Path(args.ref_audio).read_bytes()).decode()
        if args.ref_text:
            base["ref_text"] = args.ref_text
        else:
            base["x_vector_only_mode"] = True

    report: dict = {"variant": args.variant, "cases": {}}
    failures = []

    def run(name: str, payload: dict, stream: bool, min_seconds: float, max_seconds: float) -> bytes:
        try:
            pcm, ttfa, total = speech(args.url, payload, stream, args.timeout)
        except requests.RequestException as exc:
            failures.append(f"{name}: {exc}")
            report["cases"][name] = {"error": str(exc)}
            print(f"{name:28s} FAILED: {exc}")
            return b""
        info = {**stats(pcm), "ttfa_s": ttfa, "total_s": total, "stream": stream}
        report["cases"][name] = info
        write_wav(out / f"{args.variant}_{name}.wav", pcm)
        problems = []
        if not (min_seconds <= info["seconds"] <= max_seconds):
            problems.append(f"length {info['seconds']:.2f}s outside [{min_seconds}, {max_seconds}]")
        if info["peak"] < 0.05 or info["rms"] < 0.005:
            problems.append(f"near-silent audio (peak {info['peak']:.3f}, rms {info['rms']:.4f})")
        if problems:
            failures.append(f"{name}: " + "; ".join(problems))
        print(
            f"{name:28s} {info['seconds']:6.2f}s audio  peak {info['peak']:.2f}  "
            f"ttfa {ttfa * 1000:6.0f} ms  total {total:.2f}s"
        )
        return pcm

    # 1. streamed sentence: 5-10 s of speech, first audio quickly.
    run("stream", {**base, "input": TEXT}, stream=True, min_seconds=3.0, max_seconds=12.0)
    # 2. non-streaming container response.
    run("blob", {**base, "input": TEXT}, stream=False, min_seconds=3.0, max_seconds=12.0)
    # 3. greedy is repeatable byte for byte (same seed).
    greedy = {**base, "input": TEXT, "do_sample": False, "subtalker_dosample": False, "seed": 7}
    first = run("greedy_a", greedy, stream=True, min_seconds=3.0, max_seconds=12.0)
    second = run("greedy_b", greedy, stream=True, min_seconds=3.0, max_seconds=12.0)
    report["greedy_repeatable"] = first == second
    if first != second:
        failures.append("greedy runs with the same seed differ")
    # 4. long input goes through sentence chunking (server side) and stays continuous.
    run("long_chunked", {**base, "input": LONG_TEXT}, stream=True, min_seconds=25.0, max_seconds=90.0)
    # 5. instruction control (1.7B CustomVoice style, VoiceDesign voice description).
    if args.variant in ("custom_voice", "voice_design") and not args.no_instruct:
        run("instruct", {**base, "input": TEXT, "instructions": "Whisper, very quietly."}, stream=True,
            min_seconds=2.0, max_seconds=15.0)
    # 6. error paths surface as 4xx, not hangs.
    for name, bad in (("bad_voice", {**base, "input": TEXT, "voice": "nobody"}),
                      ("empty_input", {**base, "input": ""})):
        try:
            r = requests.post(f"{args.url}/v1/audio/speech", json=bad, timeout=60)
            report["cases"][name] = {"status": r.status_code}
            print(f"{name:28s} HTTP {r.status_code}")
            if args.variant == "custom_voice" and name == "bad_voice" and r.status_code // 100 != 4:
                failures.append(f"{name}: expected 4xx, got {r.status_code}")
            if name == "empty_input" and r.status_code // 100 != 4:
                failures.append(f"{name}: expected 4xx, got {r.status_code}")
        except requests.RequestException as exc:
            failures.append(f"{name}: {exc}")

    report["failures"] = failures
    (out / f"{args.variant}_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SMOKE OK" if not failures else "SMOKE FAILED:\n  " + "\n  ".join(failures))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
