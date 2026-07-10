#!/bin/bash
# Native M* Ming-flash-omni-2.0 T2T scaling sweep.
#
# Mirrors the vllm-omni sweep in results/ming_t2t_sweep/SUMMARY.md exactly so
# the native numbers drop into the same table:
#   OFFLINE B=1 (50 reqs) + CLOSED_LOOP c=2,4,8,16,32 (80 reqs each).
#
# Drives the NATIVE server (--inference-system ours) launched by
# serve_ming_tp4.sh. Run that first and wait for /health, OR let this script
# poll /health itself (it does).
#
# Each point writes results.json under OUT/<point>/ ; summarize.py rolls them
# up into a SUMMARY.md table.
set -euo pipefail

REPO=/sgl-workspace/stu/multimodal_inference
PY=/root/venvs/mminf/bin/python
URL=${URL:-http://0.0.0.0:8000}
OUT=${OUT:-$REPO/results/ming_t2t_sweep_native}
INF_SYS=${INF_SYS:-ours}

cd "$REPO"
mkdir -p "$OUT"

echo "[sweep] waiting for native server at $URL/health ..."
until curl -sf "$URL/health" >/dev/null 2>&1; do sleep 3; done
echo "[sweep] server healthy."

run_point () {
    local name=$1 prof=$2 reqs=$3 conc=$4
    echo "[sweep] === $name (prof=$prof reqs=$reqs conc=$conc) ==="
    "$PY" -m benchmark.runner \
        --url "$URL" \
        --model ming_flash_omni \
        --inference-system "$INF_SYS" \
        --request-type text_to_text \
        --dataset text \
        --profiling-type "$prof" \
        --num-requests "$reqs" \
        --num-warmup 2 \
        --max-concurrency "$conc" \
        --batch-size 1 \
        --local-cache "$REPO/data_cache" \
        --output-dir "$OUT/$name"
}

# Match the vllm-omni sweep grid 1:1.
run_point offline_b1     offline      50  1
run_point closed_loop_c2  closed_loop  80  2
run_point closed_loop_c4  closed_loop  80  4
run_point closed_loop_c8  closed_loop  80  8
run_point closed_loop_c16 closed_loop  80 16
run_point closed_loop_c32 closed_loop  80 32

echo "[sweep] done. Rolling up summary ..."
"$PY" "$REPO/scripts/ming_native/summarize.py" "$OUT"
echo "[sweep] SUMMARY -> $OUT/SUMMARY.md"
