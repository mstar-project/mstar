#!/bin/bash
# Launch one E6E server. Usage: serve_e6e.sh <colo_3g|pd_3g> [gpus] [port]
# Both arms are SHIPPED configs that differ only in where the Thinker
# prefill_* walks run (rank1 with decode vs rank0 with the encoders).
set -euo pipefail
ARM=$1
source /m-coriander/coriander/atindra/mstar_rebuttal/bench/e6e_env.sh
GPUS=${2:-$E6E_GPUS}; PORT=${3:-$E6E_PORT}
LOG=$EXP/server_${ARM}.log

case $ARM in
  colo_3g) CFG=configs/qwen3omni.yaml;;
  pd_3g)   CFG=configs/qwen3omni_pd_disaggregated.yaml;;
  # Shipped pd yaml + one placement edit: Code2Wav moves to rank2 beside the
  # Talker (the co-placement qwen3omni_thinker_tp2.yaml ships), so the T2S
  # chunk path never shares a GPU with prefills.
  pdv_3g)  CFG=$BENCH/configs_local/qwen3omni_pd_c2w_rank2.yaml;;
  *) echo "unknown arm: $ARM" | tee -a "$LOG"; exit 1;;
esac

# Rotate the log so a stale "running" line can never fake readiness.
if [ -f "$LOG" ]; then mv "$LOG" "$LOG.prev.$(date +%s)"; fi

SOCK=/tmp/mstar_atindra_e6e_${ARM}
UPLOADS=/tmp/mstar_uploads_e6e_${ARM}
rm -rf "$SOCK" "$UPLOADS"
mkdir -p "$SOCK" "$UPLOADS"

echo "[serve_e6e] arm=$ARM cfg=$CFG gpus=$GPUS port=$PORT log=$LOG"
{
  echo "=== launched $(date -Is) arm=$ARM cfg=$CFG gpus=$GPUS port=$PORT ==="
  echo "server_sha: $(git -C $BENCH/mstar rev-parse HEAD)"
} >> "$LOG"

cd $BENCH/mstar
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENVS/rebuttal-mstar/bin:$CUDA_HOME/bin:$PATH
# setsid: own process group so `kill -- -PGID` reaps every spawn_main child.
setsid $ENVS/rebuttal-mstar/bin/mstar serve qwen3_omni \
  --config $CFG --gpus "$GPUS" --port $PORT \
  --socket-path-prefix ${SOCK}/ \
  --upload-dir ${UPLOADS}/ \
  --cache-dir $HF_HUB_CACHE --tensor-comm-protocol SHM >> "$LOG" 2>&1 &
PID=$!
echo "[serve_e6e] pid=$PID pgid=$(ps -o pgid= -p $PID | tr -d ' ')"
echo "$PID" > $EXP/server_${ARM}.pid
