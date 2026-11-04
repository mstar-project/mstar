"""Goodput sweep: TTFA, stall, audio-s/s, req/s, WER and boot time across concurrency, each on a fresh server.

For each concurrency N (default up to 128) this launches a server with
``max_batch_size`` N, runs a discarded warm-up wave and ``--rounds`` measured waves
of N simultaneous streamed sessions (``benchmark.personaplex.sessions``, which
defines the metrics), and stops the server. Goodput counts the sessions meeting
the SLO: total stall under 10 ms and time to first audio under 200 ms.

The default YAML admits 5 requests of up to 2,999 forwards. Here every request is
one known clip, so the sweep's YAML sizes the KV pool for N of that clip and sets
no ``max_concurrent_requests``: startup's full-length check does not apply, and the
client is what bounds concurrency. Results therefore say nothing about N full-length
requests, which take 1.5 GiB of KV each.

Boot time is launch -> ``/health`` healthy. Cold starts from empty Triton,
TorchInductor and FlashInfer caches, warm from the same caches filled by the cold
launch. The OS page cache is not dropped (that needs root), so a cold boot may
still read the weights from memory. Each cell also records its own boot, at its
own capture sizes.

WER is reported beside listening, not as a quality score: Whisper transcripts of one
session per N against M*'s reply at N=1, and against NVIDIA's reply if given. A
reply that diverges after a near-tied token says different words and still counts
as errors here.

A cell that fails stops the sweep, since higher concurrency needs more of what ran
out; the rest are reported NOT RUN. Results are written after every cell, and
``--resume`` keeps the cells already finished.

    python -m benchmark.personaplex.sweep --settings settings.json --audio test/qwen3-omni/audio.wav \\
        --asr-model whisper-large-v3-turbo --reference-wav nvidia.wav --output runs/sweep
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

from benchmark.personaplex.judge import read_wav, word_error_rate
from benchmark.personaplex.serving import Server, write_json
from benchmark.personaplex.sessions import (
    STALL_LIMIT_S,
    TTFA_LIMIT_S,
    percentiles,
    run_cell,
    split_values,
)

DEFAULT_PAGE_SIZE = 128  # PagedKVConfig's, read in a test: importing it here would pull in the engine
DEFAULT_SESSIONS = [1, 2, 4, 8, 16, 32, 64, 96, 128]
# More than a default request's conditioning (voice, silence and persona frames).
CONDITIONING_FORWARDS = 128
FRAME_SAMPLES = 1920
SESSIONS_PER_CLIENT = 16
CACHE_VARIABLES = {"TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton",
                   "FLASHINFER_WORKSPACE_BASE": "flashinfer"}


def sweep_config(base_path: str, concurrency: int, clip_frames: int, output: Path) -> str:
    """The base YAML with ``max_batch_size`` N, and a KV pool for N requests of this clip plus the sink page."""
    import yaml

    config = yaml.safe_load(Path(base_path).read_text())
    config.setdefault("model_kwargs", {})["max_batch_size"] = concurrency
    config.pop("max_concurrent_requests", None)
    kv = config.setdefault("resources", {}).setdefault("lm_kv", {})
    # None derives a default page, as PagedKVConfig does; PersonaPlex's window-free KV gets DEFAULT_PAGE_SIZE.
    page_size = kv.get("page_size") or DEFAULT_PAGE_SIZE
    kv["max_num_pages"] = 1 + concurrency * math.ceil((clip_frames + CONDITIONING_FORWARDS) / page_size)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(output)


def fingerprint(args, settings: dict) -> dict:
    """What a cell's numbers depend on. ``--resume`` keeps earlier cells only if all of it still holds."""
    def digest(path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": head, "audio": digest(args.audio), "server_config": digest(settings["server_config"]),
            "speech_end": args.speech_end, "rounds": args.rounds, "warmup": args.warmup, "voice": args.voice,
            "buffer": args.buffer, "stall_limit": args.stall_limit, "ttfa_limit": args.ttfa_limit}


def aggregate(concurrency: int, boot_s: float, waves: list[dict]) -> dict:
    """One row per concurrency: rates are the median wave, latencies pooled over every session of every wave."""
    rows = [row for wave_result in waves for row in wave_result["rows"] if row["complete"]]

    def median(key):
        return statistics.median(wave_result[key] for wave_result in waves)

    return {"concurrency": concurrency, "status": "PASS" if all(w["valid"] for w in waves) else "FAIL",
            "boot_s": boot_s, "rounds": len(waves), "sessions": sum(w["sessions"] for w in waves),
            "completed": sum(w["completed"] for w in waves),
            "audio_s_per_s": median("audio_seconds_throughput"), "requests_per_s": median("requests_per_s"),
            "goodput_sessions_per_s": median("goodput_sessions_per_s"),
            "goodput_audio_s_per_s": median("goodput_audio_s_per_s"),
            "stall_only_goodput_audio_s_per_s": median("stall_only_goodput_audio_s_per_s"),
            "met_slo": sum(w["met_slo"] for w in waves), "met_stall": sum(w["met_stall"] for w in waves),
            "met_ttfa": sum(w["met_ttfa"] for w in waves),
            "ttfa_s": percentiles([row["time_to_first_audio_s"] for row in rows]),
            "setup_s": percentiles(split_values(rows, "setup_s")),
            "first_audio_after_setup_s": percentiles(split_values(rows, "first_audio_after_setup_s")),
            "stall_s": percentiles([row["stall_s"] for row in rows]), "waves": waves}


def summarize(cells: list[dict]) -> dict:
    """The highest concurrency where every session met each limit, and the best goodput seen."""
    passed = [cell for cell in cells if cell["status"] == "PASS"]

    def highest(key):
        return max((c["concurrency"] for c in passed if c[key] == c["sessions"]), default=None)

    return {"highest_concurrency_meeting_slo": highest("met_slo"),
            "highest_concurrency_meeting_stall": highest("met_stall"),
            "highest_concurrency_meeting_ttfa": highest("met_ttfa"),
            "best_goodput_audio_s_per_s": max((c["goodput_audio_s_per_s"] for c in passed), default=0.0),
            "best_stall_only_goodput_audio_s_per_s": max(
                (c["stall_only_goodput_audio_s_per_s"] for c in passed), default=0.0),
            "best_audio_s_per_s": max((c["audio_s_per_s"] for c in passed), default=0.0)}


def write_results(output: Path, results: dict) -> None:
    write_json(output / "sweep.json", results)
    columns = ["concurrency", "status", "boot_s", "audio_s_per_s", "requests_per_s", "goodput_audio_s_per_s",
               "stall_only_goodput_audio_s_per_s", "met_slo", "met_stall", "met_ttfa", "sessions", "ttfa_p50_s",
               "ttfa_p95_s", "ttfa_p99_s", "setup_p50_s", "first_audio_after_setup_p50_s",
               "first_audio_after_setup_p95_s", "stall_p50_s", "stall_p95_s", "stall_max_s",
               "wer_vs_c1", "wer_vs_nvidia"]
    with (output / "sweep.csv").open("w", newline="") as table:
        writer = csv.DictWriter(table, columns, extrasaction="ignore")
        writer.writeheader()
        for cell in results["cells"]:
            ttfa, stall = cell.get("ttfa_s") or {}, cell.get("stall_s") or {}
            setup, after = cell.get("setup_s") or {}, cell.get("first_audio_after_setup_s") or {}
            writer.writerow({**cell, "ttfa_p50_s": ttfa.get("p50"), "ttfa_p95_s": ttfa.get("p95"),
                             "ttfa_p99_s": ttfa.get("p99"), "setup_p50_s": setup.get("p50"),
                             "first_audio_after_setup_p50_s": after.get("p50"),
                             "first_audio_after_setup_p95_s": after.get("p95"), "stall_p50_s": stall.get("p50"),
                             "stall_p95_s": stall.get("p95"), "stall_max_s": stall.get("max")})


def timed_boot(settings: dict, root: Path) -> tuple[Server, float]:
    """A healthy server and the seconds it took from launch."""
    began = time.monotonic()
    server = Server(settings, root, False).__enter__()
    return server, time.monotonic() - began


def measure_boot(settings: dict, config: str, output: Path, rounds: int) -> list[dict]:
    """Cold then warm launches, each pair from newly emptied caches."""
    caches = output / "boot-caches"
    saved = {name: os.environ.get(name) for name in CACHE_VARIABLES}
    os.environ.update({name: str(caches / leaf) for name, leaf in CACHE_VARIABLES.items()})
    measured = []
    try:
        for attempt in range(rounds):
            shutil.rmtree(caches, ignore_errors=True)
            for kind in ("cold", "warm"):
                server, boot_s = timed_boot({**settings, "server_config": config},
                                            output / "boot" / f"{kind}-{attempt}")
                server.close()
                measured.append({"kind": kind, "round": attempt, "boot_s": boot_s})
    finally:
        for name, value in saved.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value
        shutil.rmtree(caches, ignore_errors=True)
    return measured


def run_concurrency(settings: dict, args, concurrency: int, clip_frames: int) -> dict:
    root = args.output / f"c{concurrency}"
    root.mkdir(parents=True, exist_ok=True)
    config = sweep_config(settings["server_config"], concurrency, clip_frames, root / "config.yaml")
    server, boot_s = timed_boot({**settings, "server_config": config}, root)
    client = {"client_procs": math.ceil(concurrency / SESSIONS_PER_CLIENT), "stall_limit_s": args.stall_limit,
              "ttfa_limit_s": args.ttfa_limit}
    wave_arguments = (server.url, args.audio, concurrency, args.speech_end, args.voice, args.buffer, args.timeout)
    try:
        for _ in range(args.warmup):
            run_cell(*wave_arguments, **client)
        waves = [run_cell(*wave_arguments, **client, save_audio=root / "session.wav" if not i else None)
                 for i in range(args.rounds)]
    finally:
        server.close()
    return aggregate(concurrency, boot_s, waves)


def transcribe_cells(args, results: dict) -> None:
    """Add each cell's transcript and WER, once the servers have released the GPU."""
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=args.asr_model, device=0, chunk_length_s=30)

    def transcript(path):
        audio, rate = read_wav(path)
        return asr({"raw": audio, "sampling_rate": rate}, generate_kwargs={"language": "en"})["text"].strip()

    nvidia = transcript(args.reference_wav) if args.reference_wav else None
    results["reference_transcript"] = nvidia
    for cell in results["cells"]:
        path = args.output / f"c{cell['concurrency']}" / "session.wav"
        if path.is_file():
            cell["transcript"] = transcript(path)
    alone = next((c["transcript"] for c in results["cells"] if c["concurrency"] == 1 and "transcript" in c), None)
    for cell in results["cells"]:
        if "transcript" not in cell:
            continue
        # Only against a real N=1 reply: without one the comparison does not exist.
        if alone is not None:
            cell["wer_vs_c1"] = word_error_rate(alone, cell["transcript"])
        if nvidia is not None:
            cell["wer_vs_nvidia"] = word_error_rate(nvidia, cell["transcript"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", type=Path, required=True, help="the suite's settings JSON")
    parser.add_argument("--audio", required=True, help="24 kHz mono WAV every session replays")
    parser.add_argument("--speech-end", type=float, help="where the user stops talking; default the clip's end")
    parser.add_argument("--sessions", type=int, nargs="+", default=DEFAULT_SESSIONS)
    parser.add_argument("--rounds", type=int, default=3, help="measured waves per concurrency")
    parser.add_argument("--warmup", type=int, default=1, help="discarded waves per concurrency")
    parser.add_argument("--voice", default="NATF2")
    parser.add_argument("--buffer", type=float, default=0.32, help="playback buffer after the first chunk, seconds")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--stall-limit", type=float, default=STALL_LIMIT_S)
    parser.add_argument("--ttfa-limit", type=float, default=TTFA_LIMIT_S)
    parser.add_argument("--boot-rounds", type=int, default=2, help="cold + warm launch pairs; 0 skips them")
    parser.add_argument("--boot-concurrency", type=int, default=8, help="max_batch_size of those launches")
    parser.add_argument("--asr-model", help="local Whisper checkpoint; without it WER is not computed")
    parser.add_argument("--reference-wav", type=Path, help="NVIDIA's reply to the same clip, for WER")
    parser.add_argument("--deadline-seconds", type=float, help="launch no new cell after this many seconds")
    parser.add_argument("--resume", action="store_true", help="keep finished cells from an earlier sweep.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    began = time.monotonic()
    settings = json.loads(args.settings.read_text())
    with wave.open(args.audio) as clip:
        samples, rate = clip.getnframes(), clip.getframerate()
    args.speech_end = samples / rate if args.speech_end is None else args.speech_end
    clip_frames = math.ceil(samples / FRAME_SAMPLES)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    path = args.output / "sweep.json"
    measured = fingerprint(args, settings)
    results = {"status": "RUNNING", "audio": args.audio, "clip_frames": clip_frames, "boot": [], "cells": [],
               "stall_limit_s": args.stall_limit, "ttfa_limit_s": args.ttfa_limit, "fingerprint": measured}
    if args.resume and path.exists():
        earlier = json.loads(path.read_text())
        if earlier.get("fingerprint") != measured:
            saved = earlier.get("fingerprint") or {}
            changed = sorted(k for k in measured if saved.get(k) != measured[k])
            print(f"Refusing to resume {path}: its cells were measured under a different {changed}", file=sys.stderr)
            return 2
        results = earlier
        results["cells"] = [cell for cell in results["cells"] if cell["status"] == "PASS"]
    if args.boot_rounds and not results["boot"]:
        config = sweep_config(settings["server_config"], args.boot_concurrency, clip_frames,
                              args.output / "boot-config.yaml")
        results["boot"] = measure_boot(settings, config, args.output, args.boot_rounds)
        write_results(args.output, results)
    finished = {cell["concurrency"] for cell in results["cells"]}
    stopped = None
    for concurrency in args.sessions:
        if concurrency in finished:
            continue
        if stopped is None and args.deadline_seconds and time.monotonic() - began > args.deadline_seconds:
            stopped = "Deadline reached"
        if stopped is not None:
            results["cells"].append({"concurrency": concurrency, "status": "NOT RUN", "reason": stopped})
            continue
        try:
            cell = run_concurrency(settings, args, concurrency, clip_frames)
        except Exception as exc:  # noqa: BLE001 - recorded; a server that cannot hold N ends the sweep here
            cell = {"concurrency": concurrency, "status": "FAIL", "error": repr(exc)}
        results["cells"].append(cell)
        if cell["status"] != "PASS":
            stopped = f"Stopped after c={concurrency} failed"
        results["summary"] = summarize(results["cells"])
        write_results(args.output, results)
        print(json.dumps({k: v for k, v in cell.items() if k != "waves"}), flush=True)
    results["cells"].sort(key=lambda cell: cell["concurrency"])
    if args.asr_model:
        transcribe_cells(args, results)
    results["summary"] = summarize(results["cells"])
    results["status"] = "PASS" if all(cell["status"] == "PASS" for cell in results["cells"]) else "FAIL"
    write_results(args.output, results)
    print(json.dumps(results["summary"], indent=2))
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
