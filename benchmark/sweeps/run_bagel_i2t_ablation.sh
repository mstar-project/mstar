#!/bin/bash
set -euo pipefail

# Sweep bagel image-to-text (image understanding) benchmarks, closed-loop
# continuous, with --ignore-eos so every request decodes a fixed length.
#
# Two nested sweeps:
#   * max-token range (output-len min/max): 64/256, 16/128, 128/512
#   * (max concurrency / num requests):     1/20, 2/24, 4/30, 8/40, 16/80
# Each config is run 5 times (Run 1..5); only Run 1 is warmed up.
#
# Runs against a single inference system (INF_SYS, one of {ours, vllm_omni});
# run twice to compare. Writes the accumulated raw output (.txt) and a parsed
# TSV (5 rows + an average row per config) to
# benchmark_results/bagel_i2t/i2t_<inf_sys>.{txt,tsv}.
#
# cf. benchmark/sweeps/run_bagel_img_gen.sh

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
WARMUP_FIRST=${WARMUP_FIRST:-2}       # warmup on Run 1 only (0 on Runs 2-5)
NUM_TRIALS=${NUM_TRIALS:-5}
# Per (inference system, modality) so runs don't clobber each other's dumped
# artifacts. Trials/configs within a run may overwrite each other.
ABLATION_NAME=${ABLATION_NAME:-default}
OUTPUT_DIR=".bench_outs/${INF_SYS}_i2t"
RESULTS_DIR="benchmark_results/bagel_i2t/ablation/${ABLATION_NAME}"

# (min/max output len) pairs and (max concurrency/num requests) pairs.
SEQLEN_PAIRS=("64/256")
CON_REQ_PAIRS=("1/20" "2/24" "4/30" "8/40" "16/80")

mkdir -p "$RESULTS_DIR"
RAW="$RESULTS_DIR/i2t_${INF_SYS}.txt"
TSV="$RESULTS_DIR/i2t_${INF_SYS}.tsv"
TRIAL_DIR="$(mktemp -d)"
trap 'rm -rf "$TRIAL_DIR"' EXIT

: > "$RAW"
: > "$TSV"

first_block=1
for seqlen in "${SEQLEN_PAIRS[@]}"; do
    min_tok=${seqlen%/*}
    max_tok=${seqlen#*/}
    for pair in "${CON_REQ_PAIRS[@]}"; do
        con=${pair%/*}
        reqs=${pair#*/}

        trial_files=()
        for run in $(seq 1 "$NUM_TRIALS"); do
            warmup=0
            [ "$run" -eq 1 ] && warmup=$WARMUP_FIRST

            trial_raw="$TRIAL_DIR/seq${min_tok}-${max_tok}_con${con}_run${run}.txt"
            banner="### seqlen=${seqlen} con/reqs=${pair} run=${run} warmup=${warmup} [${INF_SYS}]"
            echo "$banner"
            {
                echo ""
                echo "$banner"
            } >> "$RAW"

            python -m benchmark.runner \
                --url "$URL" \
                --model bagel \
                --profiling-type closed_loop \
                --request-type image_to_text \
                --dataset food101 \
                --ignore-eos \
                --output-len-min "$min_tok" \
                --output-len-max "$max_tok" \
                --max-concurrency "$con" \
                --num-requests "$reqs" \
                --num-warmup "$warmup" \
                --inference-system "$INF_SYS" \
                --output-dir "$OUTPUT_DIR" 2>&1 | tee "$trial_raw"
            cat "$trial_raw" >> "$RAW"
            trial_files+=("$trial_raw")
        done

        # One block: NUM_TRIALS rows + an average row. Header only on the first.
        header_flag="--no-header"
        [ "$first_block" -eq 1 ] && header_flag=""
        python "$SCRIPT_DIR/parse_i2t_sweep.py" "${trial_files[@]}" \
            --max-con "$con" \
            --max-tok-range "$seqlen" \
            --num-requests "$reqs" \
            --num-warmup-first "$WARMUP_FIRST" \
            $header_flag >> "$TSV"
        first_block=0
    done
done

echo "Wrote $RAW and $TSV"
