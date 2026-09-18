#!/bin/bash
# GLM-5.3-Flash (glm5_next) TP8 serve on the Laude team1 node — the v1
# (resource-pool) ENGINE PORT of the lane's serve script. Same preflight,
# same greedy multipart /generate smoke, same 30-min hold; what changed:
#   - configs/glm53_flash_tp8.yaml carries a `resources:` block (kv_cache /
#     kda_state) instead of the lane's top-level `kv_cache:` (raises on main);
#   - MSTAR_DIST_TIMEOUT_S is gone (no reader on main); the NCCL heartbeat
#     timeout below is what keeps a slow 306 GB load from killing rank 0;
#   - WT points at the checkout of users/kirill/glm53-flash-rp, not the lane.
#
#   sbatch env/serve_glm53_laude.sbatch
#
# STRICT PREFLIGHT: refuses to launch unless all 8 GPUs are physically clean
# (< 2 GiB used). node1 has repeatedly carried an ORPHANED vLLM process (no
# SLURM cgroup, e.g. ~70 GB on GPU 0, 2026-08-31) that a fresh 8-GPU
# allocation cannot evict — launching into it OOMs rank 0 ~40 min in, after
# the full 306 GB load. Better to exit 75 immediately and free the node than
# burn the launch. Clearing the orphan is an admin/owner action.
#SBATCH -p team1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH -J glm53-serve
#SBATCH -o /data/kvasilev-cd4cf3/logs/glm53-serve-%j.out
set -uo pipefail
source /data/kvasilev-cd4cf3/env.sh          # sets $D
# The box checkout of branch users/kirill/glm53-flash-rp (the laptop worktree
# is mstar-glm53-rp-wt). Override WT= if it lives elsewhere.
WT=${WT:-$D/mstar-glm53-rp}
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"   # glm5_next lives here, not $D/mstar
export HF_HOME=$D/hf HF_HUB_OFFLINE=1
# A short NCCL heartbeat kills rank 0 mid-load (the lane's lesson); the
# cold-NFS 306 GB read is tens of minutes. (MSTAR_DIST_TIMEOUT_S dropped.)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
# Capture the EAGER decode forward as cuda graphs, NOT the torch.compile'd
# one (already the v1 default; explicit so a flip is one edit). Compiling the
# fused forward (KDA + mHC + fp8 MoE) under Inductor crashed on the lane and
# silently degraded to eager; eager-graph capture gets the launch-overhead
# win without the compile fragility. MSTAR_GLM53_GRAPH_COMPILE=1 opts in.
export MSTAR_GLM53_GRAPH_COMPILE=0
cd "$WT"
say() { echo "$(date -u +%H:%M:%S) $*"; }

# --- preflight: wait out a preemption teardown, then require 8 clean --------
# When an Owner job preempts a guest, SLURM frees the GPUs but process teardown
# can lag a few seconds; retry before giving up. A persistent non-clean GPU
# (an out-of-band squatter SLURM can't see) still aborts — launching into it
# OOMs rank 0 after the full 306 GB load.
NCLEAN=0
for try in $(seq 1 8); do
  CLEAN=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | awk -F", " '$2<2000{print $1}' | paste -sd,)
  NCLEAN=$(echo "$CLEAN" | tr ',' '\n' | grep -c .)
  say "preflight try $try: $NCLEAN of 8 clean ($CLEAN)"
  [ "$NCLEAN" -ge 8 ] && break
  sleep 15
done
if [ "$NCLEAN" -lt 8 ]; then
  say "ABORT after ~2 min: only $NCLEAN/8 clean — a process SLURM can't see is squatting a GPU."
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  exit 75
fi

PORT=8100; OUT=$D/tmp/glm53-serve-$SLURM_JOB_ID; mkdir -p "$OUT" \
  "$D/tmp/mstar-sockets" "$D/tmp/mstar-uploads"
say "commit $(git log --oneline -1 | cut -c1-60)"
say "config configs/glm53_flash_tp8.yaml, weights $HF_HOME (offline)"
LOG=$OUT/serve.log
# The worktree shares $D/mstar's venv; PYTHONPATH above makes it import
# mstar (and glm5_next) from this worktree, not $D/mstar.
VENV=${VENV:-$D/mstar/.venv}
# Build the fused-MoE align CUDA op ONCE before eight workers race for the
# same JIT lock (a loser falls back to the uncapturable torch path and drops
# every captured decode bucket for the whole TP group).
say "pre-building the fused MoE align op"
"$VENV/bin/python" -c "from mstar.utils.fused_moe.align import _cuda_op_available as f; print('moe_align cuda op:', f())" 2>&1 | tail -1
setsid "$VENV/bin/mstar-serve" --config configs/glm53_flash_tp8.yaml \
    --tensor-comm-protocol SHM --socket-path-prefix "$D/tmp/mstar-sockets/" \
    --upload-dir "$D/tmp/mstar-uploads/" --timeout 7200 --host 0.0.0.0 \
    --port $PORT > "$LOG" 2>&1 &
PG=$!
teardown() {
  kill -TERM -"$PG" 2>/dev/null; sleep 12; kill -9 -"$PG" 2>/dev/null
  pkill -9 -u "$USER" -f "[m]star-serve.*--port $PORT" 2>/dev/null; sleep 5
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
}
trap teardown EXIT

# 306 GB cold load can take tens of minutes; poll up to ~50 min.
ready=""
for i in $(seq 1 300); do
  sleep 10
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $PG 2>/dev/null || { say "server exited during load"; tail -40 "$LOG"; exit 1; }
done
[ -n "$ready" ] || { say "readiness timeout (~50 min)"; tail -30 "$LOG"; exit 1; }
say "READY after ~$((i*10)) s"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo

# --- greedy first-tok/s smoke via the NATIVE /generate endpoint -------------
# mstar-serve speaks multipart-form /generate with an NDJSON reply (no OpenAI
# adapter for glm5_next -> /v1/* 404s). Save each raw reply for inspection,
# then best-effort parse the text + token count.
say "greedy /generate smoke (3 prompts, 128 tok, temperature 0)"
i=0; ELAPSED=()
for prompt in \
  "The capital of France is" \
  "Write a Python function that returns the nth Fibonacci number." \
  "Explain in one sentence why the sky is blue."; do
  i=$((i+1))
  t0=$(date +%s.%N)
  curl -s -X POST "http://localhost:$PORT/generate" \
    -F "text=$prompt" -F "output_modalities=text" -F "streaming=true" \
    -F 'model_kwargs={"temperature":0.0,"max_tokens":128}' \
    -o "$OUT/gen-$i.ndjson"
  t1=$(date +%s.%N)
  dt=$(echo "$t1 - $t0" | bc); ELAPSED+=("$dt")
  say "prompt $i done in ${dt}s -> $OUT/gen-$i.ndjson ($(wc -c <"$OUT/gen-$i.ndjson") bytes)"
done
"$VENV/bin/python" env/parse_generate_ndjson.py "$OUT" "${ELAPSED[@]}"
say "smoke done; RAW replies in $OUT/gen-*.ndjson — compare against the lane's Paris/Fibonacci/sky outputs"

# --- HOLD the server up (Laude has no reaper) so follow-up prompts / tok/s
# runs go against a warm server without a 10-min reload. Release early:
# touch $OUT/release.
say "HOLDING server up 30 min for follow-ups (touch $OUT/release to drop early)"
for h in $(seq 1 180); do
  [ -f "$OUT/release" ] && { say "released early"; break; }
  kill -0 $PG 2>/dev/null || { say "server died during hold"; break; }
  sleep 10
done
