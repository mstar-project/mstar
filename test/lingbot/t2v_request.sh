#!/bin/bash
#
# Single text-to-video request against the M* LingBot dense-1.3B server, saving the
# returned mp4 (POST /v1/videos/generations; the response mp4 is data[0].b64_json).
#
# Usage:
#   bash test/lingbot/t2v_request.sh                                  # defaults below
#   PORT=8200 PROMPT="a red fox running" bash test/lingbot/t2v_request.sh
#
# Env: HOST, PORT, PROMPT, NEG, SIZE (WxH), FRAMES, STEPS, GUIDANCE, SHIFT, FPS, SEED, OUT.
# Defaults mirror LingBotConfig (480x480, 81 frames, 40 steps, guidance 6.0, shift 3.0).
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8200}"
PROMPT="${PROMPT:-a red fox trotting across fresh snow at dawn, cinematic}"
NEG="${NEG:-}"
SIZE="${SIZE:-480x480}"          # WxH
FRAMES="${FRAMES:-81}"           # 1 or 4n+1
STEPS="${STEPS:-40}"
GUIDANCE="${GUIDANCE:-6.0}"
SHIFT="${SHIFT:-3.0}"
FPS="${FPS:-24}"
SEED="${SEED:-42}"
OUT="${OUT:-${TMPDIR:-/tmp}/lingbot_t2v.mp4}"

echo "[lingbot t2v] POST http://${HOST}:${PORT}/v1/videos/generations  '${PROMPT}'  ${SIZE}x${FRAMES}f steps=${STEPS}"

# size/seed/num_frames/fps are first-class VideoGenerationRequest fields; the rest
# (negative_prompt, guidance_scale, num_inference_steps, shift) pass through as
# model_kwargs via the request's extra fields (LingBotAdapter._passthrough).
# Python does the base64 decode (portable; no base64 CLI flag differences).
curl -sS -X POST "http://${HOST}:${PORT}/v1/videos/generations" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"${PROMPT}\", \"negative_prompt\": \"${NEG}\", \"size\": \"${SIZE}\", \
       \"seed\": ${SEED}, \"guidance_scale\": ${GUIDANCE}, \"num_inference_steps\": ${STEPS}, \
       \"num_frames\": ${FRAMES}, \"fps\": ${FPS}, \"shift\": ${SHIFT}}" \
  | python -c "import sys,json,base64; d=json.load(sys.stdin); open('${OUT}','wb').write(base64.b64decode(d['data'][0]['b64_json'])); print('wrote ${OUT}', len(base64.b64decode(d['data'][0]['b64_json'])), 'bytes')"
