#!/usr/bin/env python3
"""Robotics model benchmark: pi0.5 (VLA) and V-JEPA 2-AC (world model).

Uses lerobot/droid_100 — a 100-episode subset of the DROID robot manipulation
dataset with three RGB cameras, 7-DOF actions, proprioceptive state, and per-
episode language instructions.

Usage
-----
# Test pi0.5 VLA inference (5 episodes from DROID)
python benchmark/run_robotics.py \
    --model pi05 --host localhost --port 20002 \
    --num-requests 5 --output-dir /tmp/robotics_benchmark

# Test V-JEPA 2-AC world model rollout (5 episodes, H=4)
python benchmark/run_robotics.py \
    --model vjepa2_ac --host localhost --port 20003 \
    --num-requests 5 --rollout-horizon 4 \
    --output-dir /tmp/robotics_benchmark

# Use a specific HuggingFace cache dir to avoid re-downloading
python benchmark/run_robotics.py \
    --model pi05 --host localhost --port 20002 \
    --num-requests 5 --hf-cache /data/hf_cache \
    --output-dir /tmp/robotics_benchmark

Output
------
Each request prints:
  - Status (success/failed)
  - E2E latency
  - Output shape / statistics (decoded from raw float32 bytes)

Decoded outputs are also saved as .npy files in --output-dir.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import numpy as np
import requests as http_requests

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Robotics model benchmark")
    p.add_argument("--model", choices=["pi05", "vjepa2_ac"], required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=20002)
    p.add_argument("--num-requests", type=int, default=5)
    p.add_argument("--rollout-horizon", type=int, default=4,
                   help="Rollout horizon H for vjepa2_ac (ignored for pi05)")
    p.add_argument("--output-dir", default="/tmp/robotics_benchmark",
                   help="Directory for decoded outputs (.npy files)")
    p.add_argument("--hf-cache", default=None,
                   help="HuggingFace cache directory (avoids re-downloading DROID)")
    p.add_argument("--local-file-dir", default=None,
                   help="Directory for extracted images/videos (default: output-dir/media)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------

def _send_pi05(url: str, req_input) -> tuple[bytes | None, float]:
    """Send a pi0.5 VLA request and return (raw_action_bytes, latency_s)."""
    model_kwargs = json.dumps(req_input.model_kwargs)

    files = [("files", (os.path.basename(req_input.image_path),
                         open(req_input.image_path, "rb"), "image/png"))]
    for path in req_input.extra_image_paths:
        files.append(("files", (os.path.basename(path),
                                 open(path, "rb"), "image/png")))

    data = {
        "text":             req_input.prompt,
        "input_modalities": "image,text",
        "output_modalities": "action",
        "model_kwargs":     model_kwargs,
    }

    t0 = time.monotonic()
    resp = http_requests.post(url, data=data, files=files, timeout=120)
    latency = time.monotonic() - t0

    resp.raise_for_status()

    # Server returns either streaming NDJSON or a blocking JSON blob
    raw_bytes = None
    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = resp.json()
        chunks = payload.get("outputs", {}).get("action", [])
        if chunks:
            raw_bytes = base64.b64decode(chunks[0]["data"])
    else:
        # Streaming: parse NDJSON lines
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("modality") == "action" and msg.get("data"):
                raw_bytes = base64.b64decode(msg["data"])
                break

    # Close file handles
    for _, (_, fh, _) in files:
        fh.close()

    return raw_bytes, latency


def _send_vjepa2_ac(url: str, req_input) -> tuple[list[bytes], float]:
    """Send a V-JEPA 2-AC rollout request and return (latent_chunks, latency_s)."""
    model_kwargs = json.dumps(req_input.model_kwargs)

    with open(req_input.video_path, "rb") as vf:
        video_bytes = vf.read()

    files = [("files", (os.path.basename(req_input.video_path),
                         video_bytes, "video/mp4"))]
    data = {
        "input_modalities":  "video",
        "output_modalities": "video",
        "model_kwargs":      model_kwargs,
    }

    t0 = time.monotonic()
    resp = http_requests.post(url, data=data, files=files, stream=True, timeout=300)
    latency_first = None
    chunks = []

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("modality") == "video" and msg.get("data"):
            if latency_first is None:
                latency_first = time.monotonic() - t0
            chunks.append(base64.b64decode(msg["data"]))

    resp.raise_for_status()
    latency = time.monotonic() - t0
    return chunks, latency


# ---------------------------------------------------------------------------
# Output decoding + reporting
# ---------------------------------------------------------------------------

def _report_pi05(raw: bytes | None, latency: float, req_id: int, output_dir: str):
    print(f"  req {req_id:02d}: latency={latency:.2f}s", end="")
    if raw is None:
        print(" — NO ACTION OUTPUT")
        return

    ACTION_DIM = 32
    arr = np.frombuffer(raw, dtype=np.float32)
    n_steps = arr.size // ACTION_DIM
    actions = arr[: n_steps * ACTION_DIM].reshape(n_steps, ACTION_DIM)
    print(
        f"  action shape={actions.shape}"
        f"  abs_max={float(np.abs(actions).max()):.4f}"
        f"  mean_abs={float(np.abs(actions).mean()):.4f}"
    )
    print(f"    first step (8D): {actions[0, :8].tolist()}")

    path = os.path.join(output_dir, f"req_{req_id:02d}_actions.npy")
    np.save(path, actions)
    print(f"    saved → {path}")


def _report_vjepa2_ac(chunks: list[bytes], latency: float, req_id: int,
                       output_dir: str, hidden_size: int = 1408):
    print(f"  req {req_id:02d}: latency={latency:.2f}s", end="")
    if not chunks:
        print(" — NO LATENT OUTPUT")
        return

    all_arr = []
    for i, raw in enumerate(chunks):
        arr = np.frombuffer(raw, dtype=np.float32)
        n_tokens = arr.size // hidden_size
        latents = arr[: n_tokens * hidden_size].reshape(n_tokens, hidden_size)
        all_arr.append(latents)
        print(
            f"\n    iter {i}: latent shape={latents.shape}"
            f"  norm={float(np.linalg.norm(latents, axis=-1).mean()):.4f}"
            f"  finite={np.isfinite(latents).all()}"
        )

    full = np.stack(all_arr)  # [H, N, D]
    path = os.path.join(output_dir, f"req_{req_id:02d}_latents.npy")
    np.save(path, full)
    print(f"    saved → {path}  (shape={full.shape})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    local_file_dir = args.local_file_dir or os.path.join(args.output_dir, "media")
    os.makedirs(local_file_dir, exist_ok=True)

    url = f"http://{args.host}:{args.port}/generate"
    print(f"Target : {url}")
    print(f"Model  : {args.model}")
    print(f"Requests: {args.num_requests}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    from benchmark.dataset import DROIDDataset

    task = "pi05" if args.model == "pi05" else "vjepa2_ac"
    dataset = DROIDDataset(
        local_file_dir=local_file_dir,
        num_requests=args.num_requests,
        task=task,
        rollout_horizon=args.rollout_horizon,
        cache_dir=args.hf_cache,
    )
    print(f"\nLoaded {len(dataset)} episodes\n")

    # ------------------------------------------------------------------
    # Run requests
    # ------------------------------------------------------------------
    n_ok = 0
    latencies = []

    for i, req_input in enumerate(dataset):
        print(f"[{i + 1}/{len(dataset)}] prompt={req_input.prompt!r}")
        try:
            if args.model == "pi05":
                raw, lat = _send_pi05(url, req_input)
                _report_pi05(raw, lat, i, args.output_dir)
                success = raw is not None
            else:
                chunks, lat = _send_vjepa2_ac(url, req_input)
                _report_vjepa2_ac(chunks, lat, i, args.output_dir)
                success = len(chunks) > 0

            if success:
                n_ok += 1
                latencies.append(lat)
        except Exception as exc:
            print(f"  req {i:02d}: FAILED — {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Results: {n_ok}/{len(dataset)} succeeded")
    if latencies:
        lats = sorted(latencies)
        print(f"Latency : mean={sum(lats)/len(lats):.2f}s  "
              f"p50={lats[len(lats)//2]:.2f}s  "
              f"max={lats[-1]:.2f}s")
    print(f"Outputs : {args.output_dir}")


if __name__ == "__main__":
    main()
