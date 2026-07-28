#!/bin/bash
# Kill an E6E server process group and verify nothing survives on our GPUs.
# Usage: kill_e6e.sh <colo_3g|pd_3g>
set -uo pipefail
ARM=$1
EXP=/m-coriander/coriander/atindra/mstar_rebuttal/experiments/E6E
PID=$(cat $EXP/server_${ARM}.pid 2>/dev/null || true)
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  echo "killing pid=$PID pgid=$PGID"
  kill -- "-$PGID" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$PID" 2>/dev/null; then echo "escalating to -9"; kill -9 -- "-$PGID" 2>/dev/null || true; sleep 2; fi
else
  echo "no live pid recorded for $ARM"
fi
sleep 3
LEFT=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep -E "a4c23cef|9c505405|9032fd52" | grep -i atindra || true)
if [ -n "$LEFT" ]; then echo "LEFTOVERS on GPUs 2/3/7:"; echo "$LEFT"; exit 1; fi
echo "KILL_CLEAN $ARM"
