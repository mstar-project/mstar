#!/bin/bash
#
# Smoke test the V-JEPA 2 server: POST a video, decode the response, and
# verify the shape of the returned embedding tensor.
#
# Usage:
#   bash test/vjepa2/video_request.sh <video_path.mp4> [host] [port]
#
# Examples:
#   bash test/vjepa2/video_request.sh /path/to/clip.mp4
#   bash test/vjepa2/video_request.sh clip.mp4 127.0.0.1 20003
#   SKIP_PREDICTOR=1 bash test/vjepa2/video_request.sh clip.mp4
#     -> encoder-only mode (returns encoder_hidden instead of predicted_hidden)
#
# No test video handy?  Make a synthetic one with:
#   python -c "
#   import torch, torchvision
#   frames = (torch.rand(32, 3, 256, 256) * 255).byte().permute(0, 2, 3, 1)
#   torchvision.io.write_video('/tmp/synth.mp4', frames, fps=8)
#   "

set -euo pipefail

VIDEO="${1:?Usage: $0 <video_path.mp4> [host] [port]}"
HOST="${2:-127.0.0.1}"
PORT="${3:-20003}"
URL="http://${HOST}:${PORT}/generate"

if [ ! -f "${VIDEO}" ]; then
    echo "Video file not found: ${VIDEO}" >&2
    exit 1
fi

MODEL_KWARGS='{}'
if [ "${SKIP_PREDICTOR:-0}" = "1" ]; then
    MODEL_KWARGS='{"skip_predictor": true}'
fi

TMPFILE=$(mktemp /tmp/vjepa2_response.XXXXXX)
trap "rm -f ${TMPFILE}" EXIT

echo "[vjepa2] POST ${URL}"
echo "  video:         ${VIDEO}"
echo "  model_kwargs:  ${MODEL_KWARGS}"

curl -sS --fail-with-body -X POST "${URL}" \
    -F "files=@${VIDEO}" \
    -F 'input_modalities=video' \
    -F 'output_modalities=video' \
    -F "model_kwargs=${MODEL_KWARGS}" \
    -o "${TMPFILE}"

python3 - <<PY
import base64, json, sys
import numpy as np

path = "${TMPFILE}"
hidden_size = None  # Filled from response
chunks = []
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    if msg.get("modality") != "video":
        continue
    data = msg.get("data", "")
    if not data:
        continue
    chunks.append(base64.b64decode(data))

if not chunks:
    print("No video-modality chunks received.  Raw response:")
    sys.stdout.write(open(path).read())
    sys.exit(1)

raw = b"".join(chunks)
arr = np.frombuffer(raw, dtype=np.float32)
print(f"[vjepa2] received {len(chunks)} chunk(s), {len(raw)} bytes, {arr.size} floats")
print(f"[vjepa2] first 8 values: {arr[:8]}")
print(f"[vjepa2] stats: mean={arr.mean():.4f}  std={arr.std():.4f}  "
      f"min={arr.min():.4f}  max={arr.max():.4f}")
# For vitl @ 64-frame clip the predictor returns [1, 2048, 1024] = 2,097,152 floats.
# For encoder-only mode it returns [1, N, 1024] where N depends on input length.
# We don't hard-assert here — just print the divisibility check vs hidden_size=1024.
if arr.size % 1024 == 0:
    tokens = arr.size // 1024
    print(f"[vjepa2] shape divisible by 1024: {tokens} tokens * 1024 hidden_size")
else:
    print(f"[vjepa2] WARNING: size not divisible by 1024 (expected hidden_size)")
PY
