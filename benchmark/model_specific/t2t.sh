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
    --model "qwen3.5" \
    --model-id "Qwen/Qwen3.5-4B" \
    --profiling-type "${PROF_TYPE:-closed_loop}" \
    --request-type "${TASK:-text_to_text}" \
    --num-requests "${NUM_REQUESTS:-80}" \
    --inference-system "${INF_SYS:-ours}" \
    --num-warmup "${WARMUP:-3}" \
    --max-concurrency 16 \
    --ignore-eos \
    --output-len-min 256 \
    --output-len-max 4096 \
    --local-cache /mnt/storage/naomi/mstar-bench-cache \
    --output-dir .bench_outs
