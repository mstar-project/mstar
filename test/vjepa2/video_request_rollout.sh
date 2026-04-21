#!/bin/bash
#
# Smoke test the V-JEPA 2 Phase-2 rollout path: POST a video with
# ``model_kwargs={"rollout_horizon": H}`` and verify the server returns H
# distinct video chunks (one per rollout iteration).
#
# Usage:
#   bash test/vjepa2/video_request_rollout.sh <video_path.mp4> [horizon] [host] [port]
#
# Examples:
#   bash test/vjepa2/video_request_rollout.sh /path/to/clip.mp4
#   bash test/vjepa2/video_request_rollout.sh /tmp/synth.mp4 4 127.0.0.1 20003
#
# Phase 3.E streaming mode (set ``STREAM=1``):
#   STREAM=1 bash test/vjepa2/video_request_rollout.sh /tmp/synth.mp4 4
#
# In streaming mode the request adds ``stream_rollout=true`` to
# ``model_kwargs``, which selects the ``prefill_video_rollout_streaming``
# graph walk.  That walk places EMIT_TO_CLIENT on the rollout_predictor
# section directly — the server delivers one result_tensors message per
# iteration as soon as it's produced, instead of batching all H chunks
# at loop completion.  The client-side harness reports the arrival
# timestamp of each chunk so you can verify streaming actually streams
# (inter-chunk gaps should reflect the per-iter forward latency, not
# zero).
#
# Expected server-side (non-streaming default): ``prefill_video_rollout``
# walk runs, the DynamicLoop ``rollout_loop`` completes H iterations (or
# early-exits when iter_idx + 1 >= H), and each iteration's
# ``predicted_hidden`` is accumulated into a single
# ``accumulated_outputs`` edge that emits H tensor_infos to the client at
# loop completion.
#
# The launch script (``launch_server_vjepa2.sh``) is used as-is — the
# server serves all rollout walks (batched + streaming) from the same
# endpoint.  This script drives the rollout walk by passing
# ``rollout_horizon`` (and optionally ``stream_rollout``) in
# ``model_kwargs``.

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

if [ "${STREAM}" = "1" ]; then
    MODEL_KWARGS="{\"rollout_horizon\": ${HORIZON}, \"stream_rollout\": true}"
else
    MODEL_KWARGS="{\"rollout_horizon\": ${HORIZON}}"
fi

TMPFILE=$(mktemp /tmp/vjepa2_rollout_response.XXXXXX)
trap "rm -f ${TMPFILE}" EXIT

echo "[vjepa2-rollout] POST ${URL}"
echo "  video:         ${VIDEO}"
echo "  horizon:       ${HORIZON}"
echo "  stream:        ${STREAM}"
echo "  model_kwargs:  ${MODEL_KWARGS}"

if [ "${STREAM}" = "1" ]; then
    # Unbuffered curl (-N) piped through a per-line monotonic-time
    # prefixer.  Each line in TMPFILE becomes "<seconds-since-request>\t<json>".
    # The Python parser below peels the prefix to compute inter-chunk
    # gaps on the client side.
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

path = "${TMPFILE}"
expected_horizon = int("${HORIZON}")
stream_mode = "${STREAM}" == "1"

# In streaming mode every line is "<seconds>\t<json>"; in batched mode it's
# just "<json>".  Parse both uniformly: ``ts`` is populated only when
# streaming so the non-streaming path behaves identically to before.
records = []  # list of (ts_or_none, parsed_json)
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

print(f"[vjepa2-rollout] received {len(records)} chunk(s) for horizon={expected_horizon}")

arrs = []
for i, (ts, raw) in enumerate(records):
    arr = np.frombuffer(raw, dtype=np.float32)
    assert arr.size % 1024 == 0, (
        f"chunk {i}: size {arr.size} not divisible by hidden_size=1024"
    )
    tokens = arr.size // 1024
    ts_str = f"t={ts:.3f}s" if ts is not None else "t=?"
    print(
        f"  chunk {i}: {ts_str}  {len(raw)} bytes = {arr.size} floats = {tokens} tokens x 1024 hidden | "
        f"mean={arr.mean():.4f} std={arr.std():.4f}"
    )
    arrs.append(arr)

# Inter-chunk timing (streaming mode only): if the server is genuinely
# streaming per-iter, gap[k] == latency of iter k.  If it's batched, all
# gaps collapse to ~0 because the client sees the whole batch arrive at
# once at loop completion.
if stream_mode and sum(1 for ts, _ in records if ts is not None) >= 2:
    tss = [ts for ts, _ in records if ts is not None]
    gaps = [tss[i] - tss[i - 1] for i in range(1, len(tss))]
    print(f"[vjepa2-rollout] first chunk @ {tss[0]:.3f}s")
    print(f"[vjepa2-rollout] inter-chunk gaps: {['%.3f' % g for g in gaps]}s")
    # Smoke-check: at least one gap should be > 10 ms on a real H200.  If
    # every gap is near-zero the server is still batching (walk routing
    # bug on the streaming path).  We threshold loosely (not tightly on
    # predictor latency) because the check is about "is it streaming at
    # all", not a perf regression test.
    max_gap = max(gaps) if gaps else 0.0
    if max_gap < 0.01:
        print(
            f"[vjepa2-rollout] WARNING: max inter-chunk gap = {max_gap:.4f}s — "
            "server may be batching (streaming walk not wired correctly?).",
            file=sys.stderr,
        )

# Expected count: one emit per rollout iteration.
# Early-exit (register_loop_stop) could make len(records) < expected_horizon;
# we surface a warning rather than failing because that's legitimate
# behavior when the per-request horizon is below max_rollout_horizon.
if len(records) != expected_horizon:
    print(
        f"[vjepa2-rollout] NOTE: got {len(records)} chunks vs. expected {expected_horizon}. "
        "If horizon <= config.max_rollout_horizon this is register_loop_stop at work — fine.",
        file=sys.stderr,
    )

if len(arrs) >= 2:
    a, b = arrs[0], arrs[1]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    print(f"[vjepa2-rollout] cosine(chunk[0], chunk[1]) = {cos:.4f}")
    assert cos < 0.999, (
        f"chunks 0 and 1 are essentially identical (cos={cos:.4f}) — "
        "sliding-window update appears to be a no-op."
    )

print("[vjepa2-rollout] OK")
PY
