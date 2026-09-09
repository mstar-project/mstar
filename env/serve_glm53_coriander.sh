#!/usr/bin/env bash
# GLM-5.3-Flash (glm5_next) TP8 serve on CORIANDER (8×H200 141 GB) — the v1
# (resource-pool) ENGINE PORT of the lane's serve script. What changed vs the
# lane: MSTAR_DIST_TIMEOUT_S is gone (no reader on main; the NCCL heartbeat
# timeout stays), WT points at the users/kirill/glm53-flash-rp checkout, and
# the greedy smoke goes through the NATIVE multipart /generate endpoint (the
# lane's /v1/completions smoke never had an adapter behind it — glm5_next has
# no OpenAI adapter, so /v1/* 404s; the Laude sbatch already used /generate).
#
# Coriander is NOT SLURM — direct launch + the GPU-management-daemon keeper
# protocol (wiki/coriander.md §"The GPU-management daemon": a process under
# ~10% own-GPU SM util is idle-killed at ~30 min; a same-user GPU-0 keeper
# bridges the low-util 306 GB load window until READY, then exits so the
# bench is uncontended). H200 fits GLM-5.3 with huge headroom (TP8 ≈ 38 GB of
# 141/rank; TP4 also fits) — this is the alternative venue when Laude's node1
# GPU-0 is orphan-blocked.
#
#   bash env/serve_glm53_coriander.sh          # weights must be in $P/hf first
#
# PRECONDITION: the 306 GB checkpoint must already be on coriander's hub
# ($P/hf/hub/models--zai-org--GLM-5.3-Flash). Staging it is a disk decision
# (pool is ~99% full) — NOT done by this script.
set -uo pipefail
P=${P:-/m-coriander/coriander/kirill}
# The box checkout of branch users/kirill/glm53-flash-rp (the laptop worktree
# is mstar-glm53-rp-wt). Override WT= if it lives elsewhere.
WT=${WT:-$P/mstar-glm53-rp}
VENV=${VENV:-$P/mstar/.venv}
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$P/hf} HF_HUB_OFFLINE=1
# A short NCCL heartbeat kills rank 0 mid-load (the lane's lesson).
# (MSTAR_DIST_TIMEOUT_S dropped: nothing on main reads it.)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
# Eager-forward decode capture (the v1 default; explicit so a flip is one
# edit) — see serve_glm53_laude.sbatch.
export MSTAR_GLM53_GRAPH_COMPILE=0
cd "$WT"
say() { echo "$(date -u +%H:%M:%S) $*"; }

SNAP=$(ls -d "$HF_HOME"/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ] || [ ! -f "$SNAP/model.safetensors.index.json" ]; then
  say "ABORT: GLM-5.3-Flash not in $HF_HOME/hub — stage the 306 GB checkpoint first (disk decision)."
  exit 75
fi
say "checkpoint: $SNAP"
say "commit $(git log --oneline -1 | cut -c1-60)"

PORT=8100; OUT=$P/glm-m3/glm53-serve-$(date -u +%Y%m%dT%H%M%S); mkdir -p "$OUT" \
  "$P/tmp/mstar-sockets" "$P/tmp/mstar-uploads"
LOG=$OUT/serve.log

# Keeper: hold GPU 0 above the daemon's idle line through the load window.
KEEP_UNTIL=$(date -u -d '+14 min' +%H:%M:%S)
CUDA_VISIBLE_DEVICES=0 nohup "$VENV/bin/python" env/gpu0_keeper.py \
  --until "$KEEP_UNTIL" --gpu 0 --target 15 --log "$OUT/keeper.log" >/dev/null 2>&1 &
KEEPER=$!
say "gpu0 keeper up (pid $KEEPER, hard stop $KEEP_UNTIL UTC)"

setsid "$VENV/bin/mstar-serve" --config configs/glm53_flash_tp8.yaml \
  --tensor-comm-protocol SHM --socket-path-prefix "$P/tmp/mstar-sockets/" \
  --upload-dir "$P/tmp/mstar-uploads/" --timeout 7200 --host 0.0.0.0 \
  --port $PORT > "$LOG" 2>&1 &
PG=$!
teardown() {
  kill "$KEEPER" 2>/dev/null
  kill -TERM -"$PG" 2>/dev/null; sleep 12; kill -9 -"$PG" 2>/dev/null
  pkill -9 -u "$USER" -f "[m]star-serve.*--port $PORT" 2>/dev/null; sleep 5
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
}
trap teardown EXIT

ready=""
for i in $(seq 1 120); do
  sleep 10
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $PG 2>/dev/null || { say "server exited during load"; tail -40 "$LOG"; exit 1; }
done
[ -n "$ready" ] || { say "readiness timeout (~20 min)"; tail -30 "$LOG"; exit 1; }
kill "$KEEPER" 2>/dev/null   # READY — drop the keeper so the bench is uncontended
say "READY after ~$((i*10)) s (keeper stopped)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo

# --- greedy first-tok/s smoke via the NATIVE /generate endpoint -------------
# multipart-form /generate with an NDJSON streaming reply (one chunk per
# token). Raw replies are kept for inspection; the parser is best-effort.
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
"$VENV/bin/python" env/parse_generate_ndjson.py "$OUT" "${ELAPSED[@]}" | tee "$OUT/greedy.txt"
say "smoke done; artifacts in $OUT (raw gen-*.ndjson + greedy.txt)"
