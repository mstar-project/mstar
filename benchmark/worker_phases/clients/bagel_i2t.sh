#!/bin/bash
# Bagel image-to-text. Fixed output length so every request is identical work
# and the steady-state batch size is flat -- the cleanest signal to segment on.
#   mstar serve bagel --gpus 0 --port 8100
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_bench --model bagel --profiling-type closed_loop --request-type image_to_text \
    --num-requests "${N:-64}" --max-concurrency "$CONC" --num-warmup "$WARMUP" \
    --inference-system ours --ignore-eos --output-dir .bench_outs/ \
    --output-len-min "${OUTLEN:-128}" --output-len-max "${OUTLEN:-128}" "$@"
