"""TP=2 vs TP=1 equivalence smoke for Orpheus.

Compares output from a TP=1 server against a TP=2 server on the same
fixed prompt with seed pinned. Used to catch token-level regressions
that perceptual audio checks (the current colleague workflow) miss —
e.g. a MoE reduce-then-top-k vs top-k-then-reduce ordering bug, or a
vocab-parallel LM head all-gather producing a one-off shape.

Protocol (manual, two-step — production would automate via subprocess):

  Step 1 (TP=1 baseline):
    # configs/orpheus.yaml has LLM TP=1
    bash test/orpheus/launch_server_orpheus.sh \\
        # but edit the script to use orpheus.yaml instead of orpheus_tp2.yaml
    python test/distributed/test_orpheus_tp2_vs_tp1.py \\
        --mode save --output baseline_tp1.json
    # stop the server

  Step 2 (TP=2 comparison):
    bash test/orpheus/launch_server_orpheus.sh  # uses orpheus_tp2.yaml
    python test/distributed/test_orpheus_tp2_vs_tp1.py \\
        --mode compare --baseline baseline_tp1.json

Equivalence guarantees:
  * Audio byte-stream is NOT expected to match exactly. FP non-
    determinism in NCCL all-reduces means the LM head's logits differ
    by ~ULP, the sampler picks the same token in 99%+ of cases but
    occasionally a tie breaks differently.
  * Total chunk count and total audio duration should match to within
    ±2 chunks / ±5% bytes — that's what this script checks. A larger
    delta means TP introduced a real token-level divergence.
  * For full token-level equivalence we'd need a server-side flag that
    emits each sampled token alongside the audio (no such flag today).

Requirements:
  * Server running at the URL in $HOST:$PORT (see test/orpheus/_env.py).
  * Voice + text below are fixed so seed-determinism is the only
    remaining source of variation.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import requests

# Importing _env from sibling orpheus/ test dir picks up HOST/PORT.
sys.path.insert(0, str(Path(__file__).parent.parent / "orpheus"))
from _env import get_server_url  # noqa: E402

# Fix everything so the only source of cross-run variation is FP
# non-determinism inside the NCCL collectives.
PROMPT_TEXT = "The capital of France is Paris."
VOICE = "tara"
REQUEST_ID_HINT = "tp_equiv_orpheus_v1"


def run_one(url: str) -> dict:
    """Send the fixed prompt, collect audio chunks, return summary stats."""
    chunk_count = 0
    total_bytes = 0
    pcm_buffer = io.BytesIO()

    with requests.post(
        url,
        data={
            "text": PROMPT_TEXT,
            "output_modalities": "audio",
            "model_kwargs": json.dumps({"voice": VOICE}),
            # request_id propagates into the conductor seed (md5-based),
            # so pinning it gives the sampler a deterministic seed across
            # runs. See conductor.py:_req_id_to_seed.
            "request_id": REQUEST_ID_HINT,
        },
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("modality") != "audio":
                continue
            data_b64 = msg.get("data", "")
            if not data_b64:
                continue
            decoded = base64.b64decode(data_b64)
            if not decoded:
                continue
            chunk_count += 1
            total_bytes += len(decoded)
            pcm_buffer.write(decoded)

    sample_rate = 24000
    sample_width = 2  # int16
    duration_s = total_bytes / (sample_rate * sample_width)
    return {
        "chunk_count": chunk_count,
        "total_bytes": total_bytes,
        "duration_s": duration_s,
    }


def cmd_save(args):
    url = args.url or get_server_url()
    stats = run_one(url)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved baseline to {args.output}:")
    print(json.dumps(stats, indent=2))


def cmd_compare(args):
    url = args.url or get_server_url()
    current = run_one(url)
    with open(args.baseline) as f:
        baseline = json.load(f)

    chunk_delta = current["chunk_count"] - baseline["chunk_count"]
    byte_delta_pct = (
        100.0 * (current["total_bytes"] - baseline["total_bytes"])
        / max(baseline["total_bytes"], 1)
    )

    print("Baseline:", json.dumps(baseline, indent=2))
    print("Current: ", json.dumps(current, indent=2))
    print(f"chunk_count delta: {chunk_delta:+d}")
    print(f"total_bytes delta: {byte_delta_pct:+.2f}%")

    chunk_tol = 2
    byte_tol_pct = 5.0
    if abs(chunk_delta) > chunk_tol:
        print(
            f"FAIL: chunk_count drift {chunk_delta:+d} exceeds ±{chunk_tol}",
            file=sys.stderr,
        )
        sys.exit(1)
    if abs(byte_delta_pct) > byte_tol_pct:
        print(
            f"FAIL: total_bytes drift {byte_delta_pct:+.2f}% exceeds ±{byte_tol_pct}%",
            file=sys.stderr,
        )
        sys.exit(1)
    print("PASS")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=["save", "compare"], required=True)
    parser.add_argument("--url", default=None, help="Override server URL.")
    parser.add_argument("--output", help="Where to save baseline (mode=save).")
    parser.add_argument("--baseline", help="Baseline file (mode=compare).")
    args = parser.parse_args()

    if args.mode == "save":
        if not args.output:
            parser.error("--output is required for --mode save")
        cmd_save(args)
    else:
        if not args.baseline:
            parser.error("--baseline is required for --mode compare")
        cmd_compare(args)


if __name__ == "__main__":
    main()
