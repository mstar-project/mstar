#!/bin/bash
set -euo pipefail

# Source .env without overriding env vars set on the command line
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        key="${key%%[[:space:]]*}"
        [[ -z "$key" || "$key" == \#* ]] && continue
        value="${value#"${value%%[![:space:]]*}"}"
        if [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR:-./vbench_cache}

python -m benchmark.runner \
    --url "${URL:-http://${HOST}:${PORT}}" \
    --model "${MODEL:-bagel}" \
    --profiling-type "${PROF_TYPE:-offline}" \
    --request-type "${TASK:-text_to_image}" \
    --vbench-cache-dir "$VBENCH_CACHE_DIR" \
    --local-cache "$BENCHMARK_LOCAL_DIR" \
    --num-requests "${NUM_REQUESTS:-10}" \
    --inference-system "${INF_SYS:-ours}" \
    --num-warmup "${WARMUP:-3}" \
    ${DATASET:+--dataset "$dataset"} \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
    ${RATE:+--rate "$RATE"} \
    ${EXTRA_ARGS:+"$EXTRA_ARGS"}
