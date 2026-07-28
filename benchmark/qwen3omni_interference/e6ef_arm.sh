#!/bin/bash
# E6e-frontier driver, v3 (= v2 + censored-rung descend). Changes vs v1: (a) after EVERY server recovery a
# short mixed mini-ramp re-warms the instance before any measured rep (cold
# instances wedge at ~50%+ on mixed load — v1 retried cold and lost whole
# cells); (b) FRONTIER_BUCKETS env selects a bucket subset for redo passes
# (start rungs + seeds keyed by bucket NAME, so redo seeds match v1's);
# (c) reps that already exist clean on disk are reused, making redo passes
# resume-safe.
# Usage: [FRONTIER_BUCKETS="libri_len120 libri_len240"] e6ef_arm.sh <arm>
set -uo pipefail
ARM=$1
B=/m-coriander/coriander/atindra/mstar_rebuttal/bench
source $B/e6e_env.sh
PORT=$E6E_PORT
EXPF=/m-coriander/coriander/atindra/mstar_rebuttal/experiments/E6E/frontier
PY=$ENVS/rebuttal-mstar/bin/python
mkdir -p $EXPF/$ARM
DEC=$EXPF/$ARM/decisions.jsonl

BUCKETS=(${FRONTIER_BUCKETS:-libri_len15 libri_len60 libri_len120 libri_len240})
RUNGS=(0.6 0.9 1.2 1.5 1.8 2.1 2.4)
REWARM_BUCKET=libri_len15
REWARM_N=0

bucket_index() {  # canonical index → seed base identical across passes
  case $1 in
    libri_len15) echo 0;; libri_len60) echo 1;;
    libri_len120) echo 2;; libri_len240) echo 3;;
    *) echo 9;;
  esac
}
start_ri_for() {
  case $1 in
    libri_len15|libri_len60) echo 3;;   # 1.5
    libri_len120) echo 2;;              # 1.2
    libri_len240) echo 1;;              # 0.9
    *) echo 2;;
  esac
}

gate_only() {
  local PID; PID=$(cat /m-coriander/coriander/atindra/mstar_rebuttal/experiments/E6E/server_${ARM}.pid)
  local i
  for i in $(seq 1 300); do
    if curl -sf -o /dev/null http://localhost:$PORT/health; then return 0; fi
    kill -0 "$PID" 2>/dev/null || { echo "[driver] SERVER_DIED"; return 1; }
    [ "$i" -eq 300 ] && { echo "[driver] GATE_TIMEOUT"; return 1; }
    sleep 3
  done
}

serve_and_gate() {
  bash $B/kill_e6e.sh $ARM || true
  bash $B/serve_e6e.sh $ARM || return 1
  gate_only || return 1
  bash $B/preflight_e6e.sh $PORT || return 1
  # Re-warm with a short mixed window before anything measured; a wedge here
  # gets ONE fresh-serve retry, then we proceed regardless (warmth partly
  # persists in on-disk compile caches across restarts).
  REWARM_N=$((REWARM_N+1))
  local RW=$EXPF/$ARM/rewarm_${REWARM_N}
  if ! bash $B/e6ef_one.sh $ARM $PORT $REWARM_BUCKET 0.6 36 9996 $RW; then
    echo "[driver] rewarm wedged — one fresh-serve retry"
    bash $B/kill_e6e.sh $ARM || true
    bash $B/serve_e6e.sh $ARM || return 1
    gate_only || return 1
    bash $B/preflight_e6e.sh $PORT || return 1
    bash $B/e6ef_one.sh $ARM $PORT $REWARM_BUCKET 0.6 36 9996 ${RW}_retry || \
      echo "[driver] rewarm retry also wedged — proceeding on preflighted instance"
  fi
  return 0
}

rep_clean() {  # $1=rep dir  $2=expected n
  [ -f "$1/outputs/results.json" ] || return 1
  $PY - "$1/outputs/results.json" "$2" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if (r["completed"] == int(sys.argv[2]) and r["failed"] == 0) else 1)
EOF
}

echo "[driver] === arm $ARM start (v2) $(date -Is) buckets: ${BUCKETS[*]} ==="
serve_and_gate || { echo "[driver] ABORT: cannot start $ARM"; exit 1; }

for BUCKET in "${BUCKETS[@]}"; do
  bi=$(bucket_index $BUCKET)
  ri=$(start_ri_for $BUCKET)
  REWARM_BUCKET=$BUCKET
  descended=0
  evaluated=0
  echo "[driver] --- $ARM bucket $BUCKET ladder start (rung index $ri) ---"
  while [ $ri -ge 0 ] && [ $ri -le 6 ] && [ $evaluated -lt 6 ]; do
    RATE=${RUNGS[$ri]}
    N=$($PY -c "import math; print(math.ceil($RATE*120))")
    rung_bad=0
    REPDIRS=()
    for rep in 0 1 2; do
      SEED=$((7000 + bi*100 + ri*10 + rep))
      OUT=$EXPF/$ARM/$BUCKET/r${RATE}_rep${rep}
      if rep_clean "$OUT" "$N"; then
        echo "[driver] reuse existing clean rep $OUT"
        REPDIRS+=("$OUT")
        continue
      fi
      rm -rf "$OUT"
      if ! bash $B/e6ef_one.sh $ARM $PORT $BUCKET $RATE $N $SEED $OUT; then
        echo "[driver] REP_FAIL $ARM $BUCKET r$RATE rep$rep — recover + retry once"
        mv $OUT ${OUT}_failed_try1 2>/dev/null || true
        serve_and_gate || { echo "[driver] ABORT: server unrecoverable"; exit 1; }
        if ! bash $B/e6ef_one.sh $ARM $PORT $BUCKET $RATE $N $SEED $OUT; then
          echo "[driver] rung CENSORED (two failures) $ARM $BUCKET r$RATE"
          mv $OUT ${OUT}_failed_try2 2>/dev/null || true
          rung_bad=1
          serve_and_gate || { echo "[driver] ABORT: server unrecoverable"; exit 1; }
          break
        fi
      fi
      REPDIRS+=("$OUT")
      sleep 15
    done
    evaluated=$((evaluated+1))
    if [ $rung_bad -eq 1 ]; then
      echo "{\"arm\":\"$ARM\",\"bucket\":\"$BUCKET\",\"rate\":$RATE,\"censored\":true}" >> $DEC
      # A censored rung moves the ladder like a both-fail rung: descending
      # keeps lower rungs measurable instead of abandoning the cell.
      if [ $ri -le 0 ]; then break; fi
      if [ $descended -eq 0 ] && [ $evaluated -eq 1 ]; then descended=1; fi
      if [ $descended -eq 1 ]; then ri=$((ri-1)); continue; fi
      break
    fi
    V=$($PY $B/e6ef_eval.py "${REPDIRS[@]}")
    echo "[driver] RUNG $ARM $BUCKET r$RATE: $V"
    echo "{\"arm\":\"$ARM\",\"bucket\":\"$BUCKET\",\"rate\":$RATE,\"verdict\":\"$V\"}" >> $DEC
    TP=$(echo "$V" | grep -o 'ttfa_pass=[01]' | cut -d= -f2)
    IP=$(echo "$V" | grep -o 'itl_pass=[01]' | cut -d= -f2)
    ST=$(echo "$V" | grep -o 'stationary=[01]' | cut -d= -f2)
    if [ $(( (TP == 1 || IP == 1) && ST == 1 )) -eq 1 ]; then
      if [ $descended -eq 1 ]; then break; fi
      ri=$((ri+1))
    else
      if [ $ri -le 0 ]; then break; fi
      if [ $descended -eq 0 ] && [ $evaluated -eq 1 ]; then descended=1; fi
      if [ $descended -eq 1 ]; then ri=$((ri-1)); else break; fi
    fi
  done
  echo "[driver] --- $ARM bucket $BUCKET ladder done ---"
done

bash $B/kill_e6e.sh $ARM || true
echo "[driver] === arm $ARM done (v2) $(date -Is) ==="
