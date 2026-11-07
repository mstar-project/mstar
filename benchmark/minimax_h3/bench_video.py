"""Single-request latency of MiniMax-H3 text-to-audio+video, M* or SGLang, same workload.

One request at a time (concurrency 1), after ``--warmup`` untimed requests; the client
times submit to the last byte of the mp4. Both engines answer with one H.264 mp4 carrying
the AAC stereo track: M* through its OpenAI ``/v1/videos/generations`` route (which muxes
the audio in), SGLang through its asynchronous ``/v1/videos``, whose arm polls every
``--poll`` seconds (the published SGLang numbers include up to 1 s of polling too). Each
output is saved, and its streams recorded, so the arms' clips can be compared side by side.

    python -m benchmark.minimax_h3.bench_video --engine mstar --url http://127.0.0.1:8100 --out runs/m
    python -m benchmark.minimax_h3.bench_video --engine sglang --url http://127.0.0.1:30010 --out runs/s
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
from pathlib import Path

import requests

# MiniMax-H3's README T2VA case, as H3-Context-IR wrote it (a realistic structured prompt;
# README "Full 2K-Workflow", case-T2VA).
PROMPT = (
    "integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in slowly. In the "
    "cavernous, dimly lit bridge of a starship, sleek metallic consoles with glowing amber displays flank a "
    "massive, curved observation window. A female captain, in her late 40s with an athletic build and short "
    "silver-streaked black hair, stands in the center midground. She wears a structured, high-collared dark navy "
    "military tunic with silver chest insignias. Her back is to the camera, silhouetted against the cool, ambient "
    "starlight pouring through the thick glass. She stands perfectly still with her hands clasped tightly behind "
    "her back. Outside the window, a massive armada of jagged, dark grey dreadnoughts hovers in tight formation "
    "against a deep purple space nebula. The fleet's massive rear thrusters begin to glow with an intense, "
    "escalating bright blue light.\n"
    "overall_soundscape: A low, resonant hum of the ship's ambient life support systems serves as the baseline, "
    "soon drowned out by an audible, escalating, high-pitched electronic whine as the fleet outside charges its "
    "hyperdrives.\n"
    "non_diegetic_music: Cinematic space-opera orchestral score, slow tempo, featuring a solitary, mournful French "
    "horn melody over deep, sustained string dissonances."
)


def run_mstar(url: str, args, seed: int) -> bytes:
    body = {
        "model": "minimax_h3", "prompt": args.prompt, "size": f"{args.width}x{args.height}",
        "num_frames": args.num_frames, "seed": seed, "num_inference_steps": args.steps,
    }
    res = requests.post(f"{url}/v1/videos/generations", json=body, timeout=3600)
    res.raise_for_status()
    return base64.b64decode(res.json()["data"][0]["b64_json"])


def streams(mp4: bytes) -> list[str]:
    """``codec`` per stream of an mp4, e.g. ``["h264", "aac"]``."""
    import av

    with av.open(io.BytesIO(mp4)) as container:
        return [s.codec_context.name for s in container.streams]


def run_sglang(url: str, args, seed: int) -> bytes:
    short_edge = min(args.width, args.height)
    aspect = "16:9" if args.width >= args.height else "9:16"
    body = {
        "model": "MiniMaxAI/MiniMax-H3", "prompt": args.prompt, "seconds": round(args.num_frames / 24),
        "task": "t2va", "conditions": [],
        "target": {"short_edge": short_edge, "aspect_ratio": aspect, "duration_seconds": args.num_frames / 24},
        "num_outputs_per_prompt": 1, "num_inference_steps": args.steps, "flow_shift": 12.0, "audio_flow_shift": 3.0,
        "seed": seed,
    }
    job = requests.post(f"{url}/v1/videos", json=body, timeout=600)
    job.raise_for_status()
    vid = job.json()["id"]
    while True:
        status = requests.get(f"{url}/v1/videos/{vid}", timeout=60).json()
        if status.get("status") == "completed":
            break
        if status.get("status") == "failed":
            raise RuntimeError(f"SGLang job failed: {status}")
        time.sleep(args.poll)
    content = requests.get(f"{url}/v1/videos/{vid}/content", timeout=600)
    content.raise_for_status()
    return content.content


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("mstar", "sglang"), required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--num-frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=1101)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--poll", type=float, default=0.25)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run = run_mstar if args.engine == "mstar" else run_sglang
    for i in range(args.warmup):
        t0 = time.perf_counter()
        run(args.url, args, args.seed)
        print(f"warmup {i}: {time.perf_counter() - t0:.2f} s", flush=True)
    times, video = [], b""
    for i in range(args.repeat):
        t0 = time.perf_counter()
        video = run(args.url, args, args.seed)
        times.append(time.perf_counter() - t0)
        (args.out / f"{args.engine}_{i}.mp4").write_bytes(video)
        print(f"request {i}: {times[-1]:.2f} s ({len(video)} bytes)", flush=True)
    summary = {
        "engine": args.engine, "height": args.height, "width": args.width, "num_frames": args.num_frames,
        "steps": args.steps, "seconds": times, "median_s": statistics.median(times), "min_s": min(times),
        "prompt_chars": len(args.prompt), "streams": streams(video) if video else [],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
