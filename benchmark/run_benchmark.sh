#!/bin/bash
set -euo pipefail

WHO=naomi
CACHE_DIR=/mnt/storage/$WHO/vbench

python -m benchmark.runner \
    --url "${URL:-http://localhost:8000}" \
    --model "${MODEL:-bagel}" \
    --dataset vbench \
    --request-type "${TASK:-image_to_text}" \
    --vbench-cache-dir "$CACHE_DIR" \
    --num-requests "${NUM_REQUESTS:-10}" \
    --inference-system "${INF_SYS:-ours}" \
    ${RATE:+--rate "$RATE"}