"""Audio cosine equivalence smoke for full Qwen3-Omni TP=2 (Thinker + Talker).

Compares the end-to-end TTS waveform of a TP=1 (or Thinker-TP=2-only)
baseline against a full Thinker-TP=2 + Talker-TP=2 server. Audio bytes
won't match exactly under any TP because the NCCL all-reduces in the
parallel MLP / MoE / attention introduce ~ULP-level FP drift that
cascades through the residual codebook layers. Cosine similarity of
the int16 PCM waveform is the right v1 acceptance criterion: ≥ 0.95
implies perceptually identical audio with no token-level garble.

Protocol (manual, two-step):

  Step 1 (baseline):
    # Either configs/qwen3omni.yaml (pure TP=1) or
    # configs/qwen3omni_thinker_tp2.yaml (Thinker TP=2, Talker TP=1).
    # The Thinker-only baseline is closer to the full-TP setup and
    # gives the cleanest signal for the Talker-TP migration.
    bash test/qwen3-omni/launch_server.sh \\
        # edit to point at the baseline yaml
    python test/distributed/test_qwen3omni_full_tp_audio.py \\
        --mode save --output baseline_audio.json
    # stop the server

  Step 2 (full TP=2):
    bash test/qwen3-omni/launch_server.sh \\
        # edit to point at configs/qwen3omni_full_tp2.yaml
    python test/distributed/test_qwen3omni_full_tp_audio.py \\
        --mode compare --baseline baseline_audio.json

Acceptance:
  * Cosine similarity ≥ 0.95 (plan v1 target).
  * Total bytes within ±10% (catches truncated streams).
  * Both runs must produce non-empty audio.

Stricter checks (informational, non-fatal):
  * cosine ≥ 0.99 → near-bit-equal modulo NCCL FP drift.
  * cosine ≥ 0.999 → essentially bit-equal (rare; would mean NCCL is
    fully deterministic for these shapes).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import requests

# Pick up HOST/PORT from the qwen3-omni env helper if present.
sys.path.insert(0, str(Path(__file__).parent.parent / "qwen3-omni"))
try:
    from _env import get_server_url  # noqa: E402
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "orpheus"))
    from _env import get_server_url  # type: ignore  # noqa: E402


SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # int16

# Fixed prompt + seed-pinned request id so the only source of run-to-run
# variation is FP non-determinism inside the NCCL collectives.
PROMPT_TEXT = "The capital of France is Paris. It is known for the Eiffel Tower."
VOICE = "Ethan"
REQUEST_ID_HINT = "tp_equiv_qwen3omni_audio_v1"


def run_one(url: str) -> dict:
    """Send the fixed prompt, accumulate the PCM byte stream, return base64-encoded
    audio + stats."""
    pcm_buffer = io.BytesIO()
    chunk_count = 0
    with requests.post(
        url,
        data={
            "text": PROMPT_TEXT,
            "output_modalities": "audio",
            "model_kwargs": json.dumps({"voice": VOICE}),
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
            pcm_buffer.write(decoded)

    pcm = pcm_buffer.getvalue()
    return {
        "chunk_count": chunk_count,
        "total_bytes": len(pcm),
        "duration_s": len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH),
        # Keep audio inline so a single baseline file is self-contained.
        # int16 PCM at 24 kHz is ~50 KB/s — short prompts stay under a
        # few hundred KB, fine for a JSON payload.
        "pcm_b64": base64.b64encode(pcm).decode("ascii"),
    }


def _waveform_cosine(a_bytes: bytes, b_bytes: bytes) -> tuple[float, int]:
    """Cosine similarity over the overlap of two int16 PCM streams.

    Returns (cosine, overlap_samples). Cosine is between -1 and 1; 1 is
    identical, 0 is orthogonal. We compute in float64 to avoid overflow
    in the dot product (int16 values can sum past int32 range over
    long sequences).
    """
    import array
    import math

    a = array.array("h")
    a.frombytes(a_bytes)
    b = array.array("h")
    b.frombytes(b_bytes)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0, n
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b)), n


def cmd_save(args):
    url = args.url or get_server_url()
    stats = run_one(url)
    with open(args.output, "w") as f:
        json.dump(stats, f)
    summary = {k: v for k, v in stats.items() if k != "pcm_b64"}
    print(f"Saved baseline to {args.output}:")
    print(json.dumps(summary, indent=2))


def cmd_compare(args):
    url = args.url or get_server_url()
    current = run_one(url)
    with open(args.baseline) as f:
        baseline = json.load(f)

    base_pcm = base64.b64decode(baseline["pcm_b64"])
    cur_pcm = base64.b64decode(current["pcm_b64"])
    cosine, n = _waveform_cosine(base_pcm, cur_pcm)

    base_summary = {k: v for k, v in baseline.items() if k != "pcm_b64"}
    cur_summary = {k: v for k, v in current.items() if k != "pcm_b64"}

    byte_delta_pct = (
        100.0 * (current["total_bytes"] - baseline["total_bytes"])
        / max(baseline["total_bytes"], 1)
    )

    print("Baseline:", json.dumps(base_summary, indent=2))
    print("Current: ", json.dumps(cur_summary, indent=2))
    print(f"Overlap samples:  {n}")
    print(f"PCM byte delta:   {byte_delta_pct:+.2f}%")
    print(f"Waveform cosine:  {cosine:.6f}")

    cosine_floor = 0.95
    byte_tol_pct = 10.0

    if baseline["total_bytes"] == 0 or current["total_bytes"] == 0:
        print("FAIL: at least one run produced empty audio", file=sys.stderr)
        sys.exit(1)
    if abs(byte_delta_pct) > byte_tol_pct:
        print(
            f"FAIL: PCM byte delta {byte_delta_pct:+.2f}% exceeds "
            f"±{byte_tol_pct}% — likely truncation or extra padding.",
            file=sys.stderr,
        )
        sys.exit(1)
    if cosine < cosine_floor:
        print(
            f"FAIL: waveform cosine {cosine:.4f} < {cosine_floor} — "
            "indicates token-level divergence between TP configurations.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Informational tiers (non-fatal).
    if cosine >= 0.999:
        print("Note: cosine ≥ 0.999 — essentially bit-equal.")
    elif cosine >= 0.99:
        print("Note: cosine ≥ 0.99 — near-bit-equal modulo NCCL FP drift.")
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
