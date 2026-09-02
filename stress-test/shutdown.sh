#!/usr/bin/env bash
# Tear down every mstar server process this user owns and wait for the GPUs to
# return to 0 MiB. Reaps orphaned multiprocessing ranks, which otherwise keep
# holding GPU memory and fail the next cell's pre-boot gate.
set -u
STRESS_WORK="${STRESS_WORK:-$PWD/stress-run}"
pats='mstar-serve|mstar\.api_server|multiprocessing.spawn|from multiprocessing.spawn'
echo "=== shutdown $(date -u +%FT%TZ)"
for sig in TERM TERM KILL; do
  pids=$(pgrep -u "$(id -u)" -f "$pats" | tr '\n' ' ')
  [ -z "$pids" ] && break
  echo "  sending SIG$sig to: $pids"
  # shellcheck disable=SC2086
  kill -$sig $pids 2>/dev/null
  sleep 3
done
for i in $(seq 1 40); do
  live=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$live" = "0" ] && break
  sleep 2
done
echo "  remaining compute processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
rm -f "$STRESS_WORK"/logs/*.pid 2>/dev/null
echo "=== shutdown done $(date -u +%FT%TZ)"
