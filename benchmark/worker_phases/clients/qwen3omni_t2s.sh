#!/bin/bash
# qwen3-omni text-to-speech.
#   mstar serve qwen3_omni --config configs/qwen3omni_thinker_tp2.yaml --gpus 0,1,2 --port 8100
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_bench --model qwen3omni --profiling-type closed_loop --request-type text_to_speech \
    --num-requests "${N:-12}" --max-concurrency "${CONC:-4}" --num-warmup "$WARMUP" \
    --inference-system ours --dataset seed_tts "$@"
