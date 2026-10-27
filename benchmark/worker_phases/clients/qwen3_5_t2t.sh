#!/bin/bash
# Qwen3.5 text-to-text, output length drawn from [OUTMIN, OUTMAX] with EOS
# ignored, so decode length is not a free variable.
#   MSTAR_RUST_GRAPH=1 MSTAR_RUST_ZMQ=1 MSTAR_SHM_ARENA=1 \
#       mstar serve qwen3_5_0.8b --gpus 0 --port 8100
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_bench --model qwen3.5 --model-id "${MODEL_ID:-Qwen/Qwen3.5-0.8B}" \
    --profiling-type closed_loop --request-type text_to_text \
    --num-requests "${N:-$((CONC * 8))}" --max-concurrency "$CONC" --num-warmup "$WARMUP" \
    --inference-system ours --ignore-eos \
    --output-len-min "${OUTMIN:-256}" --output-len-max "${OUTMAX:-512}" "$@"
