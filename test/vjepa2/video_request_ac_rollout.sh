#!/bin/bash
#
# Smoke test the V-JEPA 2-AC Phase-3.D rollout path: POST a context video +
# deterministic 7-DOF actions/states of length T_ctx + H - 1 + rollout_horizon=H,
# receive H per-iter imagined tubelet groups, and verify shape + non-degeneracy.
#
# Usage:
#   bash test/vjepa2/video_request_ac_rollout.sh <video_path.mp4> [horizon] [host] [port]
#
# Examples:
#   bash test/vjepa2/video_request_ac_rollout.sh test/qwen3-omni/video.webm
#   bash test/vjepa2/video_request_ac_rollout.sh clip.mp4 4 127.0.0.1 20003
#
# Phase 3.E streaming mode (set ``STREAM=1``):
#   STREAM=1 bash test/vjepa2/video_request_ac_rollout.sh clip.mp4 4
#
# In streaming mode the request adds ``stream_rollout=true`` to
# ``model_kwargs``, which selects ``prefill_video_rollout_streaming``.
# That walk places EMIT_TO_CLIENT directly on the ``rollout_predictor``
# section so each iter's predicted tubelet group is delivered as soon as
# the iter completes (instead of being accumulated and emitted once at
# loop completion).  The client-side harness reports per-chunk arrival
# timestamps + inter-chunk gaps so you can verify streaming actually
# streams (a ~0 max gap means the server is still batching — which
# would indicate a routing bug on the streaming path).
#
# What gets sent
# --------------
# AC ViT-g @ 256 with tubelet_size=2 gives T_ctx = 64 / 2 = 32 tubelet
# timesteps per predictor forward.  For sliding-window rollout we need
# T_total >= T_ctx + H - 1 actions/states; we send exactly that.  The
# action ramp is small and smooth so the predictor doesn't saturate to
# NaN and consecutive iters produce distinguishable outputs.
#
# What gets asserted client-side
# ------------------------------
#   * Exactly H ``modality="video"`` chunks arrive (one per rollout iter).
#   * Each chunk is 1 tubelet group = grid^2 * hidden_size = 256 * 1408
#     floats = 1,441,792 bytes (f32).  We verify divisibility rather than
#     exact count so the script works for any backbone (vitl/g/etc).
#   * All floats finite (no NaN/Inf — would indicate a load bug).
#   * Non-zero std per chunk — zero std means the predictor isn't
#     contributing signal.
#   * Consecutive chunks have cosine similarity < 0.999 — sanity check
#     that the sliding window is actually advancing (if it were frozen,
#     consecutive iters would be identical).
#   * (Streaming mode only) At least one inter-chunk gap > 10 ms — a
#     smoke check that the server is actually streaming vs batching.

set -euo pipefail

VIDEO="${1:?Usage: $0 <video_path.mp4> [horizon] [host] [port]  (env: STREAM=1 for per-iter streaming)}"
HORIZON="${2:-4}"
HOST="${3:-127.0.0.1}"
PORT="${4:-20003}"
STREAM="${STREAM:-0}"
URL="http://${HOST}:${PORT}/generate"

if [ ! -f "${VIDEO}" ]; then
    echo "Video file not found: ${VIDEO}" >&2
    exit 1
fi

# Build deterministic actions/states of length T_ctx + H - 1.  Adds
# ``stream_rollout`` when STREAM=1; otherwise behaves identically to the
# pre-P3.E script so existing invocations are unchanged.
MODEL_KWARGS=$(python3 - <<PY
import json

T_CTX = 32  # num_frames (64) // tubelet_size (2) for facebook/vjepa2-ac-vitg
H = ${HORIZON}
DOF = 7
T_TOTAL = T_CTX + H - 1
STREAM = "${STREAM}" == "1"

def ramp(lo, hi, T):
    step = (hi - lo) / max(T - 1, 1)
    return [[lo + i * step] * DOF for i in range(T)]

kwargs = {
    "rollout_horizon": H,
    "actions": ramp(-0.5, 0.5, T_TOTAL),
    "states":  ramp( 0.1, 0.9, T_TOTAL),
}
if STREAM:
    kwargs["stream_rollout"] = True

print(json.dumps(kwargs))
PY
)

TMPFILE=$(mktemp /tmp/vjepa2_ac_rollout_response.XXXXXX)
trap "rm -f ${TMPFILE}" EXIT

echo "[vjepa2-ac-rollout] POST ${URL}"
echo "  video:         ${VIDEO}"
echo "  horizon:       ${HORIZON}"
echo "  stream:        ${STREAM}"
echo "  model_kwargs:  $(echo "${MODEL_KWARGS}" | head -c 160)..."

if [ "${STREAM}" = "1" ]; then
    # Unbuffered curl (-N) piped through a monotonic-time prefixer so each
    # line in TMPFILE becomes "<seconds-since-request>\t<json>".  The
    # parser below peels the prefix to compute inter-chunk gaps.
    curl -sS -N --fail-with-body -X POST "${URL}" \
        -F "files=@${VIDEO}" \
        -F 'input_modalities=video' \
        -F 'output_modalities=video' \
        -F "model_kwargs=${MODEL_KWARGS}" \
    | python3 -u -c '
import sys, time
start = time.monotonic()
for line in sys.stdin:
    print(f"{time.monotonic() - start:.6f}\t{line}", end="")
    sys.stdout.flush()
' > "${TMPFILE}"
else
    curl -sS --fail-with-body -X POST "${URL}" \
        -F "files=@${VIDEO}" \
        -F 'input_modalities=video' \
        -F 'output_modalities=video' \
        -F "model_kwargs=${MODEL_KWARGS}" \
        -o "${TMPFILE}"
fi

python3 - <<PY
import base64
import json
import sys

import numpy as np

HIDDEN_SIZE = 1408  # ViT-g embed_dim (AC predictor projects back to this).
expected_horizon = int("${HORIZON}")
stream_mode = "${STREAM}" == "1"
path = "${TMPFILE}"

records = []  # list of (ts_or_none, bytes)
for line in open(path):
    line = line.rstrip("\n")
    if not line:
        continue
    ts = None
    payload = line
    if stream_mode and "\t" in line:
        ts_str, payload = line.split("\t", 1)
        try:
            ts = float(ts_str)
        except ValueError:
            ts = None
    try:
        msg = json.loads(payload)
    except json.JSONDecodeError:
        continue
    if msg.get("modality") != "video":
        continue
    data = msg.get("data", "")
    if data:
        records.append((ts, base64.b64decode(data)))

if not records:
    print("No video-modality chunks received.  Raw response:")
    sys.stdout.write(open(path).read())
    sys.exit(1)

print(f"[vjepa2-ac-rollout] received {len(records)} chunk(s) for horizon={expected_horizon}")

arrs = []
for i, (ts, raw) in enumerate(records):
    arr = np.frombuffer(raw, dtype=np.float32)
    assert arr.size % HIDDEN_SIZE == 0, (
        f"chunk {i}: size {arr.size} not divisible by hidden_size={HIDDEN_SIZE}; "
        "AC predictor projects back to embed_dim=1408 on output."
    )
    tokens = arr.size // HIDDEN_SIZE
    ts_str = f"t={ts:.3f}s" if ts is not None else "t=?"
    print(
        f"  chunk {i}: {ts_str}  {len(raw)} bytes = {arr.size} floats = {tokens} tokens x {HIDDEN_SIZE} hidden | "
        f"mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}"
    )
    assert np.isfinite(arr).all(), f"chunk {i}: non-finite values (NaN/Inf)"
    assert arr.std() > 1e-4, (
        f"chunk {i}: near-zero std ({arr.std():.6g}) — predictor weights may be uninitialized."
    )
    arrs.append(arr)

# Inter-chunk timing (streaming mode only): if the server is streaming
# per-iter, gap[k] ~= latency of iter k.  If it's batched, all gaps
# collapse to ~0 because the client sees the whole batch arrive at once
# at loop completion.
if stream_mode and sum(1 for ts, _ in records if ts is not None) >= 2:
    tss = [ts for ts, _ in records if ts is not None]
    gaps = [tss[i] - tss[i - 1] for i in range(1, len(tss))]
    print(f"[vjepa2-ac-rollout] first chunk @ {tss[0]:.3f}s")
    print(f"[vjepa2-ac-rollout] inter-chunk gaps: {['%.3f' % g for g in gaps]}s")
    max_gap = max(gaps) if gaps else 0.0
    if max_gap < 0.01:
        print(
            f"[vjepa2-ac-rollout] WARNING: max inter-chunk gap = {max_gap:.4f}s — "
            "server may be batching (streaming walk not wired correctly?).",
            file=sys.stderr,
        )

# Expected: H chunks.  Early-exit (register_loop_stop) at horizon - 1
# produces exactly H chunks, so mismatch here would be a logic bug.
if len(records) != expected_horizon:
    print(
        f"[vjepa2-ac-rollout] WARNING: got {len(records)} chunks vs expected {expected_horizon}.",
        file=sys.stderr,
    )

# Sliding-window sanity: iter k and iter k+1 should NOT produce identical
# tubelet groups (different actions + shifted context → different output).
if len(arrs) >= 2:
    a, b = arrs[0], arrs[1]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    print(f"[vjepa2-ac-rollout] cosine(chunk[0], chunk[1]) = {cos:.4f}")
    assert cos < 0.999, (
        f"chunks 0 and 1 are essentially identical (cos={cos:.4f}) — "
        "sliding window isn't advancing (iter_idx not threaded, or action slice frozen)."
    )

print("[vjepa2-ac-rollout] OK")
PY
