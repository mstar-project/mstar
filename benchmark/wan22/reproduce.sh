#!/bin/bash
# Reproduce the Wan2.2-TI2V-5B serving benchmark (mstar vs vllm-omni vs SGLang).
# Mirrors benchmark/cosmos3/reproduce.sh.
#
# Cross-system latency is only valid ONE SYSTEM AT A TIME on one card. Run:
#   ./reproduce.sh preflight 0
#   DEVICES=0 PORT=8100 bash test/wan22/launch_server_wan22.sh   # then, elsewhere:
#   ./reproduce.sh bench ours 8100 mstar
#   # tear mstar down, then per baseline (see {vllm_omni,sglang_omni}_instructions.md):
#   ./reproduce.sh preflight 0
#   SERVER_PYTHON=<venv>/bin/python ./reproduce.sh serve-vllm 0 8091   # and, elsewhere:
#   SERVER_PYTHON=<venv>/bin/python ./reproduce.sh bench vllm 8091 vllm-omni
#
# Running systems concurrently on separate GPUs is valid for THROUGHPUT only:
# under power cap the cards clock differently (a 6.7% spread was measured), more
# than the latency effect being measured.
#
# Env: MSTAR, SCRATCH, SERVER_PYTHON, CLEAN_MIB.
set -eu

MSTAR="${MSTAR:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}}"
CLEAN_MIB="${CLEAN_MIB:-4096}"   # a swept card idles well under this

# Kill every process on the GPU and verify it is empty. Not optional: peak VRAM
# is whole-GPU, so an orphaned worker from the last run (killing a server by port
# leaves its children alive) is counted as this run's peak.
# usage: reproduce.sh preflight <gpu>
preflight() {
  local gpu="$1"
  local pids; pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" | tr -d ' ')"
  [ -n "$pids" ] && { echo "$pids" | xargs -r kill -9 2>/dev/null || true; sleep 5; }
  local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
  if [ "$used" -ge "$CLEAN_MIB" ]; then
    echo "[preflight] gpu $gpu still holds ${used} MiB — refusing to measure a dirty card" >&2
    return 1
  fi
  echo "[preflight] gpu $gpu clean (${used} MiB)"
}

# mstar. usage: reproduce.sh serve-mstar <gpu> <port>
serve_mstar() {
  preflight "$1"
  DEVICES="$1" PORT="$2" bash "$MSTAR/test/wan22/launch_server_wan22.sh"
}

# vllm-omni baseline, accelerations off. Install vllm 0.18.0 (0.20.0+ are cu13-only
# wheels; 0.24.0 will not load on a cu12 driver). --enforce-eager because its Wan2.2
# DiT compiles by default. NOT `vllm serve --omni`, which serves no video routes.
# usage: reproduce.sh serve-vllm <gpu> <port>
serve_vllm() {
  preflight "$1"
  CUDA_VISIBLE_DEVICES="$1" \
    python -m vllm_omni.entrypoints.cli.main serve Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --dtype bfloat16 --enforce-eager --host 0.0.0.0 --port "$2"
}

# SGLang baseline. Only these two documented flags exist in the shipped build;
# the rest kill the server on unrecognized args (see the instructions doc).
# usage: reproduce.sh serve-sglang <gpu> <port>
serve_sglang() {
  preflight "$1"
  CUDA_VISIBLE_DEVICES="$1" \
    sglang serve --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --enable-torch-compile false --enable-breakable-cuda-graph false \
    --host 0.0.0.0 --port "$2"
}

# Run the grid against a live server. SERVER_PYTHON (a baseline's own venv) stamps
# the server's torch into the row instead of this client's.
# usage: reproduce.sh bench <engine> <port> [label]
bench() {
  local engine="$1" port="$2" label="${3:-wan22}" extra=""
  [ "$engine" = "ours" ] && extra="--log-stats-file $SCRATCH/wan22_stats.txt"
  [ -n "${SERVER_PYTHON:-}" ] && extra="$extra --server-python $SERVER_PYTHON"
  PYTHONPATH="$MSTAR" python "$MSTAR/test/wan22/benchmark_wan22.py" \
    --engine "$engine" --port "$port" $extra \
    --out-json "$SCRATCH/wan22_${engine}.json" --out-csv "$SCRATCH/wan22_${engine}.csv" \
    --label "$label"
}

case "${1:-}" in
  preflight)   shift; preflight "$@";;
  serve-mstar) shift; serve_mstar "$@";;
  serve-vllm)  shift; serve_vllm "$@";;
  serve-sglang) shift; serve_sglang "$@";;
  bench)       shift; bench "$@";;
  *) echo "usage: $0 {preflight <gpu> | serve-{mstar,vllm,sglang} <gpu> <port> | bench <engine> <port> [label]}";;
esac
