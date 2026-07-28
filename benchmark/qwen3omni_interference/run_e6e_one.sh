#!/bin/bash
# One E6E benchmark run against a live server.
# Usage: run_e6e_one.sh <arm> <port> <kind>
#   kind: mix_rate<R>   online Poisson mix at rate R (240 s window)
#         t2s_c1        closed-loop c1 T2S-only control (16 reqs)
#         a2t_c1        closed-loop c1 A2T-only long-audio control (10 reqs)
#         t2s_rate<R>   online T2S-only at rate R (no-interference tails)
set -euo pipefail
ARM=$1; PORT=$2; KIND=$3
source /m-coriander/coriander/atindra/mstar_rebuttal/bench/e6e_env.sh
PY=$ENVS/rebuttal-mstar/bin/python

RUN=${ARM}_${KIND}
OUT=$EXP/$RUN
mkdir -p "$OUT/outputs"

BASE="$PY -m benchmark.runner --url http://localhost:$PORT --model qwen3omni --inference-system ours --local-cache $BENCH/data/runner_cache --output-dir $OUT/outputs --seed-tts-dir $SEEDTTS --seed-tts-locale en"
case $KIND in
  mix_rate*)
    R=${KIND#mix_rate}
    N=$($PY -c "import math; print(max(30, math.ceil($R*$E6E_WINDOW_S)))")
    CMD="$BASE --request-mix '$E6E_MIX' --mix-seed $E6E_MIX_SEED --profiling-type online --rate $R --num-requests $N --num-warmup 4"
    ;;
  t2s_c1)
    CMD="$BASE --request-type text_to_speech --dataset seed_tts --profiling-type closed_loop --max-concurrency 1 --num-requests 16 --num-warmup 3"
    ;;
  a2t_c1)
    CMD="$BASE --request-type audio_to_text --dataset libri --profiling-type closed_loop --max-concurrency 1 --num-requests 10 --num-warmup 2"
    ;;
  t2s_rate*)
    R=${KIND#t2s_rate}
    N=$($PY -c "import math; print(max(30, math.ceil($R*$E6E_WINDOW_S)))")
    CMD="$BASE --request-type text_to_speech --dataset seed_tts --mix-seed $E6E_MIX_SEED --profiling-type online --rate $R --num-requests $N --num-warmup 4"
    ;;
  smoke)
    CMD="$BASE --request-mix '$E6E_MIX' --mix-seed $E6E_MIX_SEED --profiling-type online --rate 0.5 --num-requests 8 --num-warmup 2"
    ;;
  mix18s)
    # 120 s window at rate 1.8: n=216 sits below the ~270-request horizon at
    # which the worker KeyError wedge has bitten pd_3g twice at this rate.
    CMD="$BASE --request-mix '$E6E_MIX' --mix-seed $E6E_MIX_SEED --profiling-type online --rate 1.8 --num-requests 216 --num-warmup 4"
    ;;
  probe_rate*)
    R=${KIND#probe_rate}
    N=$($PY -c "import math; print(max(12, math.ceil($R*90)))")
    CMD="$BASE --request-mix '$E6E_MIX' --mix-seed $E6E_MIX_SEED --profiling-type online --rate $R --num-requests $N --num-warmup 2"
    ;;
  *) echo "unknown kind: $KIND" ; exit 1;;
esac

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)  cluster: coriander"
  echo "arm:  $ARM   kind: $KIND"
  echo "cwd:  $BENCH/mstar-bagel2 (client)"
  echo "cmd:  $CMD"
  echo "env:  BENCH_LIBRI_WAV_DIR=$BENCH_LIBRI_WAV_DIR HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
  echo "client_sha(mstar-bagel2): $(git -C $BENCH/mstar-bagel2 rev-parse HEAD)"
  echo "server_sha(mstar):        $(git -C $BENCH/mstar rev-parse HEAD)"
  echo "server_log: $EXP/server_${ARM}.log"
  echo "gpus: $E6E_GPUS  port: $PORT"
} > "$OUT/cmd.txt"

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$OUT/audit_before.csv" || true
echo "=== [$RUN] ==="
(cd $BENCH/mstar-bagel2 && eval "$CMD") 2>&1 | tee "$OUT/runner_stdout.log"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$OUT/audit_after.csv" || true

# Results-judged gate: the runner exits 0 even when every request fails.
$PY - "$OUT/outputs/results.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
n, done, failed = r["num_requests"], r["completed"], r["failed"]
assert done == n and failed == 0, f"run not clean: completed={done}/{n} failed={failed}"
print(f"RUN_OK completed={done}/{n}")
EOF
