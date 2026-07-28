#!/bin/bash
# One E6EF rep (120 s seeded mix window against a live server).
# Usage: e6ef_one.sh <arm> <port> <bucket> <rate> <n> <seed> <out_dir>
set -euo pipefail
ARM=$1; PORT=$2; BUCKET=$3; RATE=$4; N=$5; SEED=$6; OUT=$7
source /m-coriander/coriander/atindra/mstar_rebuttal/bench/e6e_env.sh
export BENCH_LIBRI_WAV_DIR=$BENCH/data/$BUCKET
PY=$ENVS/rebuttal-mstar/bin/python
mkdir -p "$OUT/outputs"

CMD="$PY -m benchmark.runner --url http://localhost:$PORT --model qwen3omni --inference-system ours --local-cache $BENCH/data/runner_cache --output-dir $OUT/outputs --seed-tts-dir $SEEDTTS --seed-tts-locale en --request-mix '$E6E_MIX' --mix-seed $SEED --profiling-type online --rate $RATE --num-requests $N --num-warmup 2"
{
  echo "date: $(date -Is)"
  echo "arm: $ARM  bucket: $BUCKET  rate: $RATE  n: $N  seed: $SEED"
  echo "wav_dir: $BENCH_LIBRI_WAV_DIR"
  echo "cmd: $CMD"
  echo "client_sha(mstar-bagel2): $(git -C $BENCH/mstar-bagel2 rev-parse HEAD)"
  echo "server_sha(mstar):        $(git -C $BENCH/mstar rev-parse HEAD)"
  echo "gpus: $E6E_GPUS  port: $PORT"
} > "$OUT/cmd.txt"

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$OUT/audit_before.csv" || true
(cd $BENCH/mstar-bagel2 && eval "$CMD") > "$OUT/runner_stdout.log" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$OUT/audit_after.csv" || true

$PY - "$OUT/outputs/results.json" "$N" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
n = int(sys.argv[2])
ok = r["completed"] == n and r["failed"] == 0
print(("REP_OK" if ok else "REP_BAD"), f"{r['completed']}/{n}")
sys.exit(0 if ok else 1)
EOF
