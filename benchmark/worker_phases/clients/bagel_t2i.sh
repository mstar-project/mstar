#!/bin/bash
# Bagel image generation (CFG parallel). One image is ~7s of GPU work, so keep
# the counts small; concurrency 1 means batch size is flat at 1 throughout.
#   mstar serve bagel_cfg_parallel --config configs/bagel_cfg_parallel.yaml --gpus 0,1,2 --port 8100
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_bench --model bagel --profiling-type closed_loop --request-type text_to_image \
    --num-requests "${N:-6}" --max-concurrency 1 --num-warmup "${WARMUP:-1}" \
    --inference-system ours --dataset vbench \
    --vbench-cache-dir "${VBENCH_CACHE_DIR:-./vbench_cache}" "$@"
