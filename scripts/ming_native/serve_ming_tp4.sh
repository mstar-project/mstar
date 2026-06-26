#!/bin/bash
# Launch the NATIVE M* (mstar) Ming-flash-omni-2.0 thinker-only server, TP=4.
#
# Native Walk-graph path — NOT proxied through vllm-omni. Uses
# MingFlashOmniModel from mstar/model/ming_omni_flash/ (text-only thinker
# path; steps 1-5 of PORTING_NOTES.md are DONE).
#
# Pinned to physical GPUs 4-7 (logical ranks 0-3) per
# configs/ming_flash_omni_thinker_only_tp4.yaml. This sidesteps the orphan
# process on GPU 0 and gives a true 4-GPU layout comparable to the
# vllm-omni 4xH100 column in results/ming_t2t_sweep/SUMMARY.md.
#
# WARNING (from the config + PORTING_NOTES): TP=4 is documented to OOM at
# ~78.5/80 GB per rank (re-verified 2026-06-08). If load OOMs, that IS the
# finding for the 4-GPU native column; re-run with the TP=8 config
# (configs/ming_flash_omni_thinker_only.yaml, CUDA_VISIBLE_DEVICES=0-7) for
# a known-good throughput number.
set -euo pipefail

REPO=/sgl-workspace/stu/multimodal_inference
PY=/root/venvs/mminf/bin/python
CONFIG=${CONFIG:-$REPO/configs/ming_flash_omni_thinker_only_tp4.yaml}
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
GPUS=${GPUS:-4,5,6,7}
LOG=${LOG:-$REPO/scripts/ming_native/server_tp4.log}

cd "$REPO"

# Ming tokenizer/processor source repo (auto-discovered, but set explicitly).
export MING_CODE_DIR=${MING_CODE_DIR:-/tmp/ming_repo}
# TP=4 lives on the OOM edge; let the allocator give back fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES="$GPUS"

WHO=$(whoami 2>/dev/null || echo mstar)

echo "[serve] native mstar Ming thinker-only TP=4"
echo "[serve] config=$CONFIG  gpus=$GPUS  port=$PORT"
echo "[serve] MING_CODE_DIR=$MING_CODE_DIR"
echo "[serve] log -> $LOG"

exec "$PY" -m mstar.api_server.entrypoint \
    --config "$CONFIG" \
    --host "$HOST" \
    --port "$PORT" \
    --socket-path-prefix "/tmp/mstar_${WHO}/" \
    --upload-dir "/tmp/mstar_uploads_${WHO}/" \
    --tensor-comm-protocol SHM \
    --log-level INFO \
    > "$LOG" 2>&1
