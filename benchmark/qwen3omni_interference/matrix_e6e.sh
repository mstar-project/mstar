#!/bin/bash
# Full E6E measurement set against a live, healthy server.
# Usage: matrix_e6e.sh <arm> <port> "<mix rates>" <t2s_solo_rate>
# Assumes serve_e6e.sh already ran and /health returns 200.
set -euo pipefail
ARM=$1; PORT=$2; RATES=$3; T2S_SOLO=$4
B=/m-coriander/coriander/atindra/mstar_rebuttal/bench
source $B/e6e_env.sh
trap 'echo "MATRIX_ERR arm='"$ARM"' at line $LINENO"' ERR

echo "=== matrix_e6e $ARM port=$PORT rates=[$RATES] t2s_solo=$T2S_SOLO $(date -Is) ==="

bash $B/run_e6e_one.sh $ARM $PORT smoke

for R in $RATES; do
  bash $B/run_e6e_one.sh $ARM $PORT mix_rate$R
  sleep 10
done

bash $B/run_e6e_one.sh $ARM $PORT t2s_rate$T2S_SOLO
bash $B/run_e6e_one.sh $ARM $PORT t2s_c1
bash $B/run_e6e_one.sh $ARM $PORT a2t_c1

echo "MATRIX_DONE $ARM $(date -Is)"
