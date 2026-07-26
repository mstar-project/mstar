#!/bin/bash
set -euo pipefail

# Sweep bagel image-generation benchmarks: T2I and I2I (both vbench dataset),
# closed-loop, max-concurrency 1. Runs against a single inference system
# (INF_SYS, one of {ours, vllm_omni}); run twice to compare.
#
# For each task it writes both the raw benchmark output (.txt) and a parsed
# one-row TSV to benchmark_results/bagel_img_gen/<tag>_<inf_sys>.{txt,tsv}.
#
# cf. benchmark/run_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Source .env without overriding env vars set on the command line.
ENV_FILE="$REPO_ROOT/.env"
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
URL=${URL:-http://${HOST}:${PORT}}
INF_SYS=${INF_SYS:-ours}              # ours | vllm_omni
NUM_WARMUP=${WARMUP:-10}
NUM_REQUESTS=${NUM_REQUESTS:-10}
MAX_CONCURRENCY=1
VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR:-./vbench_cache}
OUTPUT_DIR=.bench_outs

ABLATION_NAME=${ABLATION_NAME:-default}
RESULTS_DIR="benchmark_results/bagel_img_gen/ablation/${ABLATION_NAME}"

mkdir -p "$RESULTS_DIR"

run_one() {
    local tag=$1 req_type=$2
    local raw="$RESULTS_DIR/${tag}_${INF_SYS}.txt"
    local tsv="$RESULTS_DIR/${tag}_${INF_SYS}.tsv"
    # Per (inference system, modality) so runs don't clobber each other's
    # dumped artifacts. Trials within a run may overwrite.
    local out_dir="$OUTPUT_DIR/${INF_SYS}_${tag}"

    echo ">>> ${tag} (${req_type}) [${INF_SYS}] warmup=${NUM_WARMUP} requests=${NUM_REQUESTS}"
    python -m benchmark.runner \
        --url "$URL" \
        --model bagel \
        --profiling-type closed_loop \
        --request-type "$req_type" \
        --dataset vbench \
        --num-warmup "$NUM_WARMUP" \
        --num-requests "$NUM_REQUESTS" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --inference-system "$INF_SYS" \
        --vbench-cache-dir "$VBENCH_CACHE_DIR" \
        --output-dir "$out_dir" 2>&1 | tee "$raw"

    python "$SCRIPT_DIR/parse_img_gen_table.py" "$raw" \
        --max-con "$MAX_CONCURRENCY" \
        --num-warmup "$NUM_WARMUP" \
        --num-requests "$NUM_REQUESTS" \
        --run 1 > "$tsv"

    echo "Wrote $raw and $tsv"
}

run_one t2i text_to_image
run_one i2i image_to_image
