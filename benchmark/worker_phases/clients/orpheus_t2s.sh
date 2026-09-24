#!/bin/bash
# Orpheus text-to-speech. Pairs with:
#   mstar serve orpheus --config configs/orpheus_colocated.yaml --gpus 0 --port 8100
#   mstar serve orpheus --config configs/orpheus_tp2.yaml --gpus 0,1 --port 8100
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_bench --model orpheus --profiling-type closed_loop --request-type text_to_speech \
    --num-requests "${N:-200}" --max-concurrency "$CONC" --num-warmup "$WARMUP" \
    --inference-system ours --dataset seed_tts "$@"
