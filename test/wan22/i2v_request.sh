#!/bin/bash
#
# Single image-to-video request against the M* Wan2.2-TI2V-5B server. The
# conditioning frame is sent inline as a base64 data URI in the JSON body's
# "image" field (Wan22Adapter routes it to the VAE-encode / first-frame-injection
# I2V path). Saves the returned mp4.
#
# Usage:
#   IMAGE=/path/to/frame.jpg bash test/wan22/i2v_request.sh
#   PORT=8100 IMAGE=cond.jpg PROMPT="the cat turns its head" bash test/wan22/i2v_request.sh
#
# Env: HOST, PORT, IMAGE (required), PROMPT, SIZE (WxH), FRAMES, STEPS, GUIDANCE, FPS, SEED, OUT.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8100}"
IMAGE="${IMAGE:?set IMAGE to the conditioning frame path (jpg/png)}"
PROMPT="${PROMPT:-the scene comes to life with gentle motion}"
NEG="${NEG:-}"
SIZE="${SIZE:-832x480}"          # WxH == the 480x832 (HxW) grid cell
FRAMES="${FRAMES:-33}"
STEPS="${STEPS:-20}"
GUIDANCE="${GUIDANCE:-5.0}"
FPS="${FPS:-24}"
SEED="${SEED:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/wan22_i2v.mp4}"

[ -f "$IMAGE" ] || { echo "image not found: $IMAGE" >&2; exit 1; }
echo "[wan22 i2v] POST http://${HOST}:${PORT}/v1/videos/generations  image=${IMAGE}  '${PROMPT}'  ${SIZE}x${FRAMES}f steps=${STEPS}"

# Build the JSON body in python (embeds the image as a data URI) and POST it.
python - "$IMAGE" "$HOST" "$PORT" "$PROMPT" "$NEG" "$SIZE" "$FRAMES" "$STEPS" "$GUIDANCE" "$FPS" "$SEED" "$OUT" <<'PY'
import base64, json, mimetypes, sys, urllib.request
img, host, port, prompt, neg, size, frames, steps, gs, fps, seed, out = sys.argv[1:13]
mime = mimetypes.guess_type(img)[0] or "image/jpeg"
data_uri = f"data:{mime};base64," + base64.b64encode(open(img, "rb").read()).decode()
body = json.dumps({
    "prompt": prompt, "negative_prompt": neg, "size": size, "seed": int(seed),
    "guidance_scale": float(gs), "num_inference_steps": int(steps),
    "num_frames": int(frames), "fps": int(fps), "flow_shift": 5.0, "image": data_uri,
}).encode()
req = urllib.request.Request(f"http://{host}:{port}/v1/videos/generations",
                             data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=3600) as r:
    d = json.load(r)
mp4 = base64.b64decode(d["data"][0]["b64_json"])
open(out, "wb").write(mp4)
print(f"wrote {out} {len(mp4)} bytes")
PY
