#!/bin/bash
set -euo pipefail

WHO=naomi
CACHE_DIR=/mnt/storage/$WHO/vbench

python -m benchmark.runner \
    --url "${URL:-http://localhost:8000}" \
    --model "${MODEL:-bagel}" \
    --dataset ${DATASET:-vbench} \
    --profiling-type "${PROF_TYPE:-offline}" \
    --request-type "${TASK:-text_to_image}" \
    --vbench-cache-dir "$CACHE_DIR" \
    --num-requests "${NUM_REQUESTS:-10}" \
    --inference-system "${INF_SYS:-ours}" \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"}
    ${RATE:+--rate "$RATE"}