#!/bin/bash
#
# Single text-to-video request against the M* LingBot dense-1.3B server, saving the
# returned mp4 (POST /v1/videos/generations; the response mp4 is data[0].b64_json).
#
# LingBot-Video consumes STRUCTURED-JSON prompts (rewriter output), NOT raw text —
# a raw prompt garbles the video on the reference model too. Prompt precedence:
#   PROMPT_JSON=/path/to/prompt.json   structured caption; its "caption" field is
#                                      serialized and sent as the prompt (recommended)
#   PROMPT="...raw text..."            raw string (smoke only; expect poor output)
#   (neither set)                      a built-in structured default caption is used,
#                                      so the script produces a good clip out of the box
#
# Usage:
#   PORT=8200 bash test/lingbot/t2v_request.sh                                # default caption
#   PORT=8200 PROMPT_JSON=assets/cases/t2v/example_1/prompt.json bash test/lingbot/t2v_request.sh
#   PORT=8200 PROMPT="a red fox running" bash test/lingbot/t2v_request.sh    # raw smoke
#
# Env: HOST, PORT, PROMPT_JSON | PROMPT, NEG, SIZE (WxH), FRAMES, STEPS, GUIDANCE,
#      SHIFT, FPS, SEED, OUT. Omit NEG to use the model's built-in default negative.
# Defaults mirror the dense-t2v reference (832x480, 81 frames, 40 steps, guidance 3).
set -euo pipefail

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8200}"
export PROMPT_JSON="${PROMPT_JSON:-}"
export PROMPT="${PROMPT:-}"
# Bundled default structured caption, used when neither PROMPT_JSON nor PROMPT is set.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_PROMPT_JSON="${DEFAULT_PROMPT_JSON:-$_SCRIPT_DIR/default_prompt.json}"
export NEG="${NEG:-}"
export SIZE="${SIZE:-832x480}"          # WxH
export FRAMES="${FRAMES:-81}"           # 1 or 4n+1
export STEPS="${STEPS:-40}"
export GUIDANCE="${GUIDANCE:-3.0}"
export SHIFT="${SHIFT:-3.0}"
export FPS="${FPS:-24}"
export SEED="${SEED:-42}"
export OUT="${OUT:-${TMPDIR:-/tmp}/lingbot_t2v.mp4}"

# The body is assembled in Python (json.dumps) so a structured caption — which is
# itself JSON with quotes/braces — is escaped correctly. size/seed/num_frames/fps
# are first-class VideoGenerationRequest fields; guidance_scale/num_inference_steps/
# shift/negative_prompt pass through as model_kwargs (LingBotAdapter._passthrough).
python3 - <<'PY'
import base64, json, os, urllib.request

host, port = os.environ["HOST"], os.environ["PORT"]
size = os.environ["SIZE"]

# Prompt precedence: PROMPT_JSON (file) > PROMPT (raw text) > bundled default file.
# LingBot needs a structured caption, so with no input we load the bundled default
# prompt.json rather than sending raw text (which garbles).
prompt_json = os.environ.get("PROMPT_JSON", "")
raw_prompt = os.environ.get("PROMPT", "")
using_default = not prompt_json and not raw_prompt
if using_default:
    prompt_json = os.environ["DEFAULT_PROMPT_JSON"]

if prompt_json:
    doc = json.load(open(prompt_json))
    caption = doc.get("caption", doc)  # accept a full prompt.json or a bare caption
    prompt = json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    src = ("default " if using_default else "") + f"PROMPT_JSON={prompt_json}"
else:
    prompt = raw_prompt
    src = "raw PROMPT (smoke; structured JSON recommended)"

body = {
    "prompt": prompt,
    "size": size,
    "num_frames": int(os.environ["FRAMES"]),
    "num_inference_steps": int(os.environ["STEPS"]),
    "guidance_scale": float(os.environ["GUIDANCE"]),
    "shift": float(os.environ["SHIFT"]),
    "fps": int(os.environ["FPS"]),
    "seed": int(os.environ["SEED"]),
    "response_format": "b64_json",
}
neg = os.environ.get("NEG", "")
if neg:  # omit -> server applies the model's default structured negative prompt
    body["negative_prompt"] = neg

out = os.environ["OUT"]
print(f"[lingbot t2v] POST http://{host}:{port}/v1/videos/generations  {src}  "
      f"{size} x{body['num_frames']}f steps={body['num_inference_steps']}", flush=True)
req = urllib.request.Request(
    f"http://{host}:{port}/v1/videos/generations",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req, timeout=1800))
blob = base64.b64decode(resp["data"][0]["b64_json"])
with open(out, "wb") as f:
    f.write(blob)
print(f"wrote {out} ({len(blob)} bytes)")
PY
