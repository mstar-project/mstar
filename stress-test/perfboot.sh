#!/usr/bin/env bash
# Boot one mstar server for a reproduction cell, with a pre-boot cleanliness gate.
#
# Usage: perfboot.sh <tag> <cfg> <gpus> <port> <celldir> <cold|warm> [maxwait]
#   tag      base | head          (selects checkout + venv)
#   cfg      config name, e.g. bagel_single_gpu / bagel_cfg_parallel
#   gpus     CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1,2"
#   port     HTTP port
#   celldir  output directory for this cell
#
# Env (defaults shown):
#   STRESS_WORK=$PWD/stress-run             venvs, checkouts, sockets, pid files
#   STRESS_VENVS=$STRESS_WORK/venvs         CO_BASE/CO_HEAD=$STRESS_WORK/co-{base,head}
#   HF_HOME=$HOME/.cache/huggingface        model cache; also passed as --cache-dir
#   MSTAR_LOG_LEVEL=INFO                    DEBUG for the scheduler lines
#   MSTAR_EXTRA_ARGS=""                     e.g. --enable-nvtx
#   MSTAR_WRAP=""                           e.g. an "nsys launch ..." prefix
#
# Writes <celldir>/{pre.txt,clocks_pre.txt,boot.txt,server.log}.
set -u
TAG="$1"; CFG="$2"; GPUS="$3"; PORT="$4"; CELL="$5"; CW="$6"; MAXWAIT="${7:-1800}"

STRESS_WORK="${STRESS_WORK:-$PWD/stress-run}"
STRESS_VENVS="${STRESS_VENVS:-$STRESS_WORK/venvs}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
LOGLEVEL="${MSTAR_LOG_LEVEL:-INFO}"
EXTRA_ARGS="${MSTAR_EXTRA_ARGS:-}"
WRAP="${MSTAR_WRAP:-}"
export HF_HOME

case "$TAG" in
  base) CO="${CO_BASE:-$STRESS_WORK/co-base}"; VENV="$STRESS_VENVS/mstar-base" ;;
  head) CO="${CO_HEAD:-$STRESS_WORK/co-head}"; VENV="$STRESS_VENVS/mstar-head" ;;
  *) echo "bad tag '$TAG' (want base|head)"; exit 2 ;;
esac
[ -x "$VENV/bin/mstar-serve" ] || { echo "no mstar-serve in $VENV, run stress-test/setup_venvs.sh first"; exit 2; }
[ -f "$CO/configs/${CFG}.yaml" ] || { echo "no config $CO/configs/${CFG}.yaml"; exit 2; }

mkdir -p "$CELL" "$STRESS_WORK/sock" "$STRESS_WORK/logs"
# absolute, because we cd into the checkout before launching
CELL=$(cd "$CELL" && pwd)
STRESS_WORK=$(cd "$STRESS_WORK" && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="${TAG}_${CFG}_${STAMP}"
LOG="$CELL/server.log"
N=$(cat "$STRESS_WORK/sock/.counter" 2>/dev/null || echo 0); N=$((N+1)); echo "$N" > "$STRESS_WORK/sock/.counter"
SOCK="$STRESS_WORK/sock/$(printf '%s%02d' "${TAG:0:1}" "$N")"
mkdir -p "$SOCK"

# Per-commit compile caches: a cache shared between the two commits freezes the
# Inductor autotune choices of whichever commit ran first.
CO_SHA=$(git -C "$CO" rev-parse --short=8 HEAD)
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$STRESS_WORK/cache/inductor-$CO_SHA}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STRESS_WORK/cache/triton-$CO_SHA}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Server is pinned to the NUMA node the GPUs hang off (node 0 on the reference box).
SRV_NODE="${STRESS_SERVER_NUMA:-0}"
CORES=$(lscpu | awk '/NUMA node0 CPU/{print $4}')
{
  echo "=== pre-boot $(date -u +%FT%TZ) run=$RUN cold_warm=$CW"
  echo "--- target GPUs: $GPUS  server numa=$SRV_NODE  node0_cores=$CORES"
  echo "--- cpuset: $(grep Cpus_allowed_list /proc/self/status | cut -f2)"
  echo "--- inductor cache: $TORCHINDUCTOR_CACHE_DIR"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo "--- compute processes ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true
  echo "--- uptime ---"; uptime
} > "$CELL/pre.txt" 2>&1
nvidia-smi -q -d CLOCK,TEMPERATURE -i "$GPUS" > "$CELL/clocks_pre.txt" 2>&1

# Gate: every target GPU idle and no compute process anywhere.
DIRTY=0
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  [ "${used:-1}" != "0" ] && { echo "GPU $g not clean: ${used} MiB"; DIRTY=1; }
done
nap=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$nap" != "0" ] && { echo "compute processes present: $nap"; DIRTY=1; }
[ "$DIRTY" = "1" ] && { echo "PRE-BOOT GATE FAILED for $RUN" | tee -a "$CELL/boot.txt"; exit 3; }

LOAD=$(cut -d' ' -f1 /proc/loadavg)
T0=$(date +%s.%N)
cd "$CO" || exit 2
# shellcheck disable=SC2086
setsid env CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME="$HF_HOME" PATH="$VENV/bin:$PATH" \
  TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  numactl --cpunodebind="$SRV_NODE" --membind="$SRV_NODE" \
  $WRAP \
  "$VENV/bin/mstar-serve" --config "$CO/configs/${CFG}.yaml" \
    --port "$PORT" --host 127.0.0.1 \
    --socket-path-prefix "$SOCK" \
    --cache-dir "$HF_HOME" \
    --log-level "$LOGLEVEL" $EXTRA_ARGS > "$LOG" 2>&1 &
SRV=$!
echo "$SRV" > "$STRESS_WORK/logs/${RUN}.pid"

OK=0
while :; do
  el=$(awk -v a="$(date +%s.%N)" -v b="$T0" 'BEGIN{print a-b}')
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "$RUN SERVER DIED after ${el}s" | tee -a "$CELL/boot.txt"; tail -40 "$LOG" >> "$CELL/boot.txt"; exit 4
  fi
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)
  [ "$code" = "200" ] && { OK=1; break; }
  awk -v e="$el" -v m="$MAXWAIT" 'BEGIN{exit !(e>m)}' && { echo "$RUN TIMEOUT after ${el}s" | tee -a "$CELL/boot.txt"; break; }
  sleep 1
done
TTH=$(awk -v a="$(date +%s.%N)" -v b="$T0" 'BEGIN{printf "%.1f", a-b}')
printf "run=%s cold_warm=%s time_to_health=%s load1=%s gpus=%s server_numa=%s port=%s pid=%s\n" \
  "$RUN" "$CW" "$TTH" "$LOAD" "$GPUS" "$SRV_NODE" "$PORT" "$SRV" | tee -a "$CELL/boot.txt"
[ "$OK" = "1" ] || exit 5
