#!/bin/bash
# Untimed single-request preflights (absorb first-use JIT before any measured
# run): one T2S, one long-audio A2T. Usage: preflight_e6e.sh <port>
set -uo pipefail
PORT=$1
source /m-coriander/coriander/atindra/mstar_rebuttal/bench/e6e_env.sh

T2S=$(curl -s --max-time 900 -X POST "http://localhost:$PORT/generate" \
  -F "text=Preflight check sentence for the speech pipeline." \
  -F "output_modalities=audio" \
  -F 'model_kwargs={"max_tokens":64,"max_output_tokens":64,"thinker_temperature":0.0}' \
  -o /dev/null -w "%{http_code} %{size_download}")
echo "preflight t2s: http=$T2S"

A2T=$(curl -s --max-time 900 -X POST "http://localhost:$PORT/generate" \
  -F "text=Transcribe the speech in this audio clip." \
  -F "output_modalities=text" -F "input_modalities=audio,text" \
  -F 'model_kwargs={"max_tokens":32,"max_output_tokens":32,"thinker_temperature":0.0}' \
  -F "files=@$LIBRI_LONG/long_000.wav" \
  -o /dev/null -w "%{http_code} %{size_download}")
echo "preflight a2t: http=$A2T"

[[ "$T2S" == 200* && "$A2T" == 200* ]] || { echo "PREFLIGHT_FAIL"; exit 1; }
echo "PREFLIGHT_OK"
