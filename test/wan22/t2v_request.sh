#!/bin/bash
#
# Single text-to-video request against the M* Wan2.2-TI2V-5B server, saving the
# returned mp4 (POST /v1/videos/generations; the response mp4 is data[0].b64_json).
#
# Usage:
#   bash test/wan22/t2v_request.sh                                  # defaults below
#   PORT=8100 PROMPT="a red fox running" bash test/wan22/t2v_request.sh
#
# Env: HOST, PORT, PROMPT, SIZE (WxH), FRAMES, STEPS, GUIDANCE, FPS, SEED, OUT.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8100}"
PROMPT="${PROMPT:-a person playing guitar}"
NEG="${NEG:-}"
SIZE="${SIZE:-832x480}"          # WxH == the 480x832 (HxW) grid cell
FRAMES="${FRAMES:-33}"
STEPS="${STEPS:-20}"
GUIDANCE="${GUIDANCE:-5.0}"
FPS="${FPS:-24}"
SEED="${SEED:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/wan22_t2v.mp4}"

echo "[wan22 t2v] POST http://${HOST}:${PORT}/v1/videos/generations  '${PROMPT}'  ${SIZE}x${FRAMES}f steps=${STEPS}"

# Capture the JSON, extract b64_json, decode to an mp4. Python does the base64
# decode (portable; no base64 CLI flag differences).
curl -sS -X POST "http://${HOST}:${PORT}/v1/videos/generations" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"${PROMPT}\", \"negative_prompt\": \"${NEG}\", \"size\": \"${SIZE}\", \
       \"seed\": ${SEED}, \"guidance_scale\": ${GUIDANCE}, \"num_inference_steps\": ${STEPS}, \
       \"num_frames\": ${FRAMES}, \"fps\": ${FPS}, \"flow_shift\": 5.0}" \
  | python -c "import sys,json,base64; d=json.load(sys.stdin); open('${OUT}','wb').write(base64.b64decode(d['data'][0]['b64_json'])); print('wrote ${OUT}', len(base64.b64decode(d['data'][0]['b64_json'])), 'bytes')"
