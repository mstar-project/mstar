#!/bin/bash
#
# Smoke test the V-JEPA 2-AC MPC walk: POST a context video + K candidate
# action sequences + a synthetic pre-encoded goal latent, then decode the
# response and check ``best_index``, per-candidate ``costs``, and
# ``predicted_hidden`` shape.
#
# Usage:
#   bash test/vjepa2/video_request_mpc.sh <video_path.mp4> [host] [port] [K]
#
# Examples:
#   bash test/vjepa2/video_request_mpc.sh test/qwen3-omni/video.webm
#   bash test/vjepa2/video_request_mpc.sh clip.mp4 127.0.0.1 20003 8
#
# What gets sent
# --------------
# Phase 3.B MPC walk (prefill_video_mpc):
#
#   video_encoder  ──► ac_predictor_mpc ──► mpc_scorer ──► EMIT_TO_CLIENT
#                           ▲      ▲                        (best_index,
#                           │      └── actions, states      costs,
#                           │          [K, 32, 7] each      predicted_hidden)
#                           └── goal_hidden [1, 8192, 1408]
#                               (pre-encoded by client)
#
# For this smoke test we DON'T pre-encode a real goal frame — we send a
# synthetic ramp-latent so the scorer's argmin has something non-trivial
# to pick from.  In production, the client would run
# ``prefill_video_encoder_only`` on a goal image once and cache the result
# locally, then attach it as ``goal_hidden`` on every MPC request.
#
# Goal-hidden is large (8192 × 1408 × 4 bytes ≈ 46 MB).  JSON-list transport
# works for a smoke test but is inefficient — real deployments should use
# a binary / shared-mem side channel.  Handled downstream as a Phase 4 polish.
#
# What gets asserted client-side
# ------------------------------
#   * ``best_index`` in [0, K)
#   * ``costs`` has length K and all values finite
#   * ``predicted_hidden`` bytes are K × (tokens × 1408 × 4)

set -euo pipefail

VIDEO="${1:?Usage: $0 <video_path.mp4> [host] [port] [K]}"
HOST="${2:-127.0.0.1}"
PORT="${3:-20003}"
K="${4:-4}"
URL="http://${HOST}:${PORT}/generate"

if [ ! -f "${VIDEO}" ]; then
    echo "Video file not found: ${VIDEO}" >&2
    exit 1
fi

# Build model_kwargs: K candidates of (actions, states) + scalar goal_hidden_fill.
#
# We use the server-side ``goal_hidden_fill`` convenience (server expands
# a scalar to the full [1, N, D] tensor) because the full goal_hidden at
# ViT-g is ~46 MB raw / ~100 MB as JSON list-of-lists, which blows past
# Starlette's default ``max_part_size`` of 1 MB per form field — requests
# carrying it return 400 before even reaching the FastAPI handler.
#
# The ``goal_hidden_fill`` path serializes to <20 bytes and still exercises
# the full MPC graph (encoder -> ac_predictor_mpc -> mpc_scorer) end-to-end,
# so it's a viable smoke test.  Real production MPC clients that need a
# meaningful goal latent should either (a) bump Starlette's max_part_size
# in the server config, or (b) use a separate binary-upload field once
# that plumbing lands.
MODEL_KWARGS=$(python3 - <<PY
import json

K = ${K}
T_ACTION = 32  # num_frames (64) // tubelet_size (2)
DOF = 7

def ramp(lo, hi, T):
    step = (hi - lo) / max(T - 1, 1)
    return [[lo + i * step] * DOF for i in range(T)]

actions = [ramp(-0.5 + 0.1 * k, 0.5 + 0.1 * k, T_ACTION) for k in range(K)]
states  = [ramp( 0.1 + 0.1 * k, 0.9 + 0.1 * k, T_ACTION) for k in range(K)]

print(json.dumps({
    "mpc": True,
    "actions": actions,
    "states":  states,
    # Scalar — server broadcasts to torch.full((1, N, D), fill) in process_prompt.
    "goal_hidden_fill": 0.05,
}))
PY
)

TMPFILE=$(mktemp /tmp/vjepa2_mpc_response.XXXXXX)
trap "rm -f ${TMPFILE}" EXIT

echo "[vjepa2-mpc] POST ${URL}"
echo "  video:  ${VIDEO}"
echo "  K:      ${K}"
echo "  kwargs: ${MODEL_KWARGS}"

curl -sS --fail-with-body -X POST "${URL}" \
    -F "files=@${VIDEO}" \
    -F 'input_modalities=video' \
    -F 'output_modalities=scalar,tensor,video' \
    -F "model_kwargs=${MODEL_KWARGS}" \
    -o "${TMPFILE}"

python3 - <<PY
import base64
import json
import sys

import numpy as np

HIDDEN = 1408
K = ${K}
path = "${TMPFILE}"

# Collect chunks grouped by modality.  MPC emits 3 separate EMIT_TO_CLIENT
# edges (best_index / costs / predicted_hidden) — each is its own chunk.
per_modality: dict[str, list[bytes]] = {}
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    mod = msg.get("modality")
    data = msg.get("data", "")
    if not mod or not data:
        continue
    per_modality.setdefault(mod, []).append(base64.b64decode(data))

print(f"[vjepa2-mpc] modalities received: {sorted(per_modality.keys())}")

# best_index: scalar int64 → 8 bytes.
if "scalar" not in per_modality:
    print("No scalar chunk (best_index missing).  Raw response:")
    sys.stdout.write(open(path).read())
    sys.exit(1)
best_raw = b"".join(per_modality["scalar"])
best = int(np.frombuffer(best_raw, dtype=np.int64)[0])
assert 0 <= best < K, f"best_index {best} out of range [0, {K})"
print(f"[vjepa2-mpc] best_index = {best}")

# costs: [K] float32.
if "tensor" not in per_modality:
    print("No tensor chunk (costs missing).")
    sys.exit(1)
costs_raw = b"".join(per_modality["tensor"])
costs = np.frombuffer(costs_raw, dtype=np.float32)
assert costs.size == K, f"costs has {costs.size} entries, expected K={K}"
assert np.isfinite(costs).all(), f"costs has non-finite values: {costs}"
print(f"[vjepa2-mpc] costs (K={K}): {costs.tolist()}")
# Sanity: argmin of costs matches best_index.
argmin = int(np.argmin(costs))
assert argmin == best, f"argmin(costs)={argmin} != best_index={best}"

# predicted_hidden: [K, tokens, HIDDEN] float32.
if "video" not in per_modality:
    print("No video chunk (predicted_hidden missing).")
    sys.exit(1)
pred_raw = b"".join(per_modality["video"])
pred = np.frombuffer(pred_raw, dtype=np.float32)
assert pred.size % (K * HIDDEN) == 0, (
    f"predicted_hidden size {pred.size} not divisible by K*HIDDEN={K*HIDDEN}"
)
tokens = pred.size // (K * HIDDEN)
print(f"[vjepa2-mpc] predicted_hidden shape: [{K}, {tokens}, {HIDDEN}]  "
      f"mean={pred.mean():.4f}  std={pred.std():.4f}")
assert np.isfinite(pred).all(), "predicted_hidden has NaN/Inf"

print("[vjepa2-mpc] OK")
PY
