"""TP=2 vs TP=1 equivalence smoke for Qwen3-Omni Thinker.

Compares the Thinker's text output between a TP=1 server and a TP=2
server on the same fixed prompt with seed pinned. Thinker text is more
amenable to exact equivalence than Orpheus audio: there is no
downstream codec to absorb logit noise, so a one-token divergence shows
up immediately as a character-level mismatch in the response.

Protocol (manual, two-step):

  Step 1 (TP=1 baseline):
    # configs/qwen3omni.yaml has Thinker on a single rank
    bash test/qwen3-omni/launch_server.sh \\
        # but edit to point at qwen3omni.yaml
    python test/distributed/test_qwen3omni_thinker_tp2_vs_tp1.py \\
        --mode save --output baseline_thinker_tp1.json
    # stop the server

  Step 2 (TP=2 comparison):
    bash test/qwen3-omni/launch_server.sh   # uses qwen3omni_thinker_tp2.yaml
    python test/distributed/test_qwen3omni_thinker_tp2_vs_tp1.py \\
        --mode compare --baseline baseline_thinker_tp1.json

Equivalence levels (the script checks all three, in order of strictness):

  1. Strict: identical text. Holds when greedy decoding is used + the
     NCCL all-reduces produce bit-equal results (they often do for
     small TP groups on the same GPU model; not guaranteed).
  2. Loose: first N=20 characters identical. Catches reordering /
     wrong-token bugs but tolerates a late-sequence divergence.
  3. Smoke: response length within ±10% and contains the substring
     "Paris" (the prompt fixes the answer). Catches catastrophic
     breakage but not subtle drift.

Requirements:
  * Server running at $HOST:$PORT.
  * Output modality "text" so the Talker / Code2Wav don't participate
    (their state isn't relevant for Thinker TP equivalence).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

# Pick up HOST/PORT from the qwen3-omni env helper if present.
sys.path.insert(0, str(Path(__file__).parent.parent / "qwen3-omni"))
try:
    from _env import get_server_url  # noqa: E402
except ImportError:
    # Fall back to the orpheus helper — same env format.
    sys.path.insert(0, str(Path(__file__).parent.parent / "orpheus"))
    from _env import get_server_url  # type: ignore  # noqa: E402


# Deterministic prompt + greedy decoding so the only cross-TP source
# of variation is FP non-determinism in the NCCL collectives.
PROMPT_TEXT = "What is the capital of France? Answer in one sentence."
REQUEST_ID_HINT = "tp_equiv_thinker_v1"


def run_one(url: str) -> dict:
    """Send the fixed prompt, accumulate streamed text deltas, return the
    full response + length stats."""
    text_buf: list[str] = []
    chunk_count = 0
    with requests.post(
        url,
        data={
            "text": PROMPT_TEXT,
            "output_modalities": "text",
            # Greedy decoding (temperature=0) keeps the sampler
            # deterministic; the only remaining source of TP divergence
            # is FP non-determinism inside the LLM forward.
            "model_kwargs": json.dumps({"temperature": 0.0, "top_k": 1}),
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
            if msg.get("modality") != "text":
                continue
            data = msg.get("data", "")
            if not data:
                continue
            text_buf.append(data)
            chunk_count += 1
    full = "".join(text_buf)
    return {
        "text": full,
        "char_count": len(full),
        "chunk_count": chunk_count,
    }


def _level_strict(baseline: dict, current: dict) -> bool:
    return baseline["text"] == current["text"]


def _level_loose(baseline: dict, current: dict, n_chars: int = 20) -> bool:
    return baseline["text"][:n_chars] == current["text"][:n_chars]


def _level_smoke(baseline: dict, current: dict) -> bool:
    len_ratio = (
        abs(current["char_count"] - baseline["char_count"])
        / max(baseline["char_count"], 1)
    )
    has_paris = "Paris" in current["text"] and "Paris" in baseline["text"]
    return len_ratio < 0.10 and has_paris


def cmd_save(args):
    url = args.url or get_server_url()
    out = run_one(url)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved baseline to {args.output} ({out['char_count']} chars).")


def cmd_compare(args):
    url = args.url or get_server_url()
    current = run_one(url)
    with open(args.baseline) as f:
        baseline = json.load(f)

    print(f"Baseline: {baseline['char_count']} chars, {baseline['chunk_count']} chunks")
    print(f"Current:  {current['char_count']} chars, {current['chunk_count']} chunks")

    strict = _level_strict(baseline, current)
    loose = _level_loose(baseline, current)
    smoke = _level_smoke(baseline, current)

    print(f"Strict (identical text): {'PASS' if strict else 'FAIL'}")
    print(f"Loose (first 20 chars):  {'PASS' if loose else 'FAIL'}")
    print(f"Smoke (length + Paris):  {'PASS' if smoke else 'FAIL'}")

    if not strict:
        print("\n--- baseline ---")
        print(baseline["text"])
        print("--- current ---")
        print(current["text"])

    if not smoke:
        sys.exit(1)
    # Strict / loose failures are non-fatal — they're diagnostic. v1 only
    # requires the smoke level to hold; the strict / loose deltas tell
    # us whether TP introduced FP non-determinism (acceptable) or a real
    # token-level bug (not acceptable).


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
