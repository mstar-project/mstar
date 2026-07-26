#!/bin/bash
set -euo pipefail

# Sweep text-to-speech benchmarks (qwen3omni | orpheus) on the Seed-TTS test
# set, closed-loop continuous.
#
# One sweep: (max concurrency / num requests) = 1/12, 2/20, 4/24, 8/40,
# 16/80, 32/160. Each config runs once (no repeated trials / average row);
# every run is warmed up.
#
# Runs against a single inference system (INF_SYS); run twice to compare.
# Writes the accumulated raw output (.txt) and a parsed TSV (one row per
# config) to
# benchmark_results/tts/ablation/<ablation_name>/tts_<model>_<inf_sys>.{txt,tsv}.
#
# cf. benchmark/sweeps/run_bagel_i2t_ablation.sh

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
MODEL=${MODEL:-qwen3omni}             # qwen3omni | orpheus
INF_SYS=${INF_SYS:-ours}              # ours | vllm_omni | vox_serve | sglang_omni
NUM_WARMUP=${NUM_WARMUP:-2}
SEED_TTS_LOCALE=${SEED_TTS_LOCALE:-en}
# Per (model, inference system) so runs don't clobber each other's dumped
# artifacts. Configs within a run may overwrite each other.
ABLATION_NAME=${ABLATION_NAME:-default}
OUTPUT_DIR=".bench_outs/${MODEL}_${INF_SYS}_tts"
RESULTS_DIR="benchmark_results/tts/ablation/${ABLATION_NAME}"

# (max concurrency / num requests) pairs.
CON_REQ_PAIRS=("1/12" "2/20" "4/24" "8/40" "16/80" "32/160")

mkdir -p "$RESULTS_DIR"
RAW="$RESULTS_DIR/tts_${MODEL}_${INF_SYS}.txt"
TSV="$RESULTS_DIR/tts_${MODEL}_${INF_SYS}.tsv"
TRIAL_DIR="$(mktemp -d)"
trap 'rm -rf "$TRIAL_DIR"' EXIT

: > "$RAW"
: > "$TSV"

first_block=1
for pair in "${CON_REQ_PAIRS[@]}"; do
    con=${pair%/*}
    reqs=${pair#*/}

    trial_raw="$TRIAL_DIR/con${con}.txt"
    banner="### con/reqs=${pair} warmup=${NUM_WARMUP} [${MODEL} / ${INF_SYS}]"
    echo "$banner"
    {
        echo ""
        echo "$banner"
    } >> "$RAW"

    python -m benchmark.runner \
        --url "$URL" \
        --model "$MODEL" \
        --profiling-type closed_loop \
        --request-type text_to_speech \
        --dataset seed_tts \
        --seed-tts-locale "$SEED_TTS_LOCALE" \
        --max-concurrency "$con" \
        --num-requests "$reqs" \
        --num-warmup "$NUM_WARMUP" \
        --inference-system "$INF_SYS" \
        --output-dir "$OUTPUT_DIR" 2>&1 | tee "$trial_raw"
    cat "$trial_raw" >> "$RAW"

    # One row per config. Header only on the first.
    header_flag="--no-header"
    [ "$first_block" -eq 1 ] && header_flag=""
    python "$SCRIPT_DIR/parse_tts_sweep.py" "$trial_raw" \
        --max-con "$con" \
        --num-requests "$reqs" \
        --num-warmup "$NUM_WARMUP" \
        $header_flag >> "$TSV"
    first_block=0
done

echo "Wrote $RAW and $TSV"
