#!/usr/bin/env bash
# One reproduction cell: boot a server, run every spec line in a spec file under
# a single dmon trace, tear down.
#
# Usage: perfcell.sh <tag> <cfg> <gpus> <port> <celldir> <cold|warm> <specfile>
#   spec file lines are  <name>|<benchmark/runner.py args>
#
# Env: see perfboot.sh. Additionally
#   KEEP_SERVER_LOG=0   1 keeps the (large) raw server log
set -u
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG="$1"; CFG="$2"; GPUS="$3"; PORT="$4"; CELL="$5"; CW="$6"; SPEC="$7"

STRESS_WORK="${STRESS_WORK:-$PWD/stress-run}"
STRESS_VENVS="${STRESS_VENVS:-$STRESS_WORK/venvs}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
STRESS_CACHE="${STRESS_CACHE:-$STRESS_WORK/cache/bench}"
STRESS_VBENCH_CACHE="${STRESS_VBENCH_CACHE:-$STRESS_WORK/cache/vbench}"
mkdir -p "$STRESS_WORK" "$STRESS_CACHE" "$STRESS_VBENCH_CACHE"
# absolute and exported: the run loop cd's into the checkout, so a $PWD-relative
# default would resolve differently in perfboot.sh / shutdown.sh than here.
STRESS_WORK=$(cd "$STRESS_WORK" && pwd)
STRESS_CACHE=$(cd "$STRESS_CACHE" && pwd)
STRESS_VBENCH_CACHE=$(cd "$STRESS_VBENCH_CACHE" && pwd)
export HF_HOME STRESS_WORK STRESS_CACHE STRESS_VBENCH_CACHE
case "$TAG" in
  base) CO="${CO_BASE:-$STRESS_WORK/co-base}"; VENV="$STRESS_VENVS/mstar-base" ;;
  head) CO="${CO_HEAD:-$STRESS_WORK/co-head}"; VENV="$STRESS_VENVS/mstar-head" ;;
  *) echo "bad tag '$TAG' (want base|head)"; exit 2 ;;
esac
[ -f "$SPEC" ] || { echo "no spec file $SPEC"; exit 2; }
# absolute, because the loop below cd's into the checkout before each run
SPEC=$(cd "$(dirname "$SPEC")" && pwd)/$(basename "$SPEC")
mkdir -p "$CELL"
CELL=$(cd "$CELL" && pwd)
NAME=$(basename "$CELL")

# Client NUMA pinning. Keep the benchmark client off the server's NUMA node when
# the cpuset spans both; when it covers a single node --cpunodebind on the other
# fails (and fails *silently* under `env`, leaving the cell unrun), so bind memory
# only. Whichever is chosen is recorded in the cell.
SRV_NODE="${STRESS_SERVER_NUMA:-0}"
CLIENT_NODE="${STRESS_CLIENT_NUMA:-1}"
if numactl --cpunodebind="$CLIENT_NODE" --membind="$CLIENT_NODE" true 2>/dev/null; then
  CLIENT_PIN="numactl --cpunodebind=$CLIENT_NODE --membind=$CLIENT_NODE"
else
  CLIENT_PIN="numactl --membind=$SRV_NODE"
fi
echo "client_pin=$CLIENT_PIN cpuset=$(grep Cpus_allowed_list /proc/self/status | cut -f2)" \
  | tee "$CELL/pinning.txt"

if ! "$SCRIPT_DIR/perfboot.sh" "$TAG" "$CFG" "$GPUS" "$PORT" "$CELL" "$CW" 1800 >> "$CELL/boot.txt" 2>&1; then
  echo "$NAME BOOT FAILED - $(tail -3 "$CELL/boot.txt" | tr '\n' ' ')"
  "$SCRIPT_DIR/shutdown.sh" >/dev/null 2>&1
  exit 1
fi
TTH=$(grep -o 'time_to_health=[0-9.]*' "$CELL/boot.txt" | tail -1 | cut -d= -f2)
echo "$NAME boot $CW time_to_health=${TTH}s gpus=$GPUS"

setsid nvidia-smi dmon -s u -d 1 -i "$GPUS" -o DT > "$CELL/dmon.txt" 2>&1 &
DMON=$!
trap 'kill $DMON 2>/dev/null' EXIT

: > "$CELL/spans.txt"
while IFS='|' read -r rname rargs; do
  [ -z "${rname:-}" ] && continue
  case "$rname" in \#*) continue ;; esac
  # spec files carry ${STRESS_CACHE} / ${STRESS_VBENCH_CACHE} placeholders so no
  # absolute path is baked in; expand them here rather than eval'ing the line.
  rargs="${rargs//\$\{STRESS_CACHE\}/$STRESS_CACHE}"
  rargs="${rargs//\$\{STRESS_VBENCH_CACHE\}/$STRESS_VBENCH_CACHE}"
  S=$(date -u +%FT%TZ); S_EPOCH=$(date +%s)
  cd "$CO"
  # shellcheck disable=SC2086
  $CLIENT_PIN env PATH="$VENV/bin:$PATH" HF_HOME="$HF_HOME" PYTHONPATH="$CO" \
    "$VENV/bin/python" "$SCRIPT_DIR/run_bench.py" \
      --harness-out "$CELL/${rname}.json" --harness-tag "${NAME}/${rname}" \
      --url "http://127.0.0.1:$PORT" $rargs > "$CELL/${rname}.log" 2>&1
  RC=$?
  E=$(date -u +%FT%TZ); ELA=$(( $(date +%s) - S_EPOCH ))
  echo "$rname $S $E rc=$RC wall=${ELA}s" >> "$CELL/spans.txt"
  echo "  run $rname rc=$RC wall=${ELA}s $(grep -o '\[harness\] wrote .*' "$CELL/${rname}.log" | tail -1)"
  [ $RC -ne 0 ] && tail -6 "$CELL/${rname}.log" | sed 's/^/      /'
done < "$SPEC"

kill $DMON 2>/dev/null; trap - EXIT
nvidia-smi -q -d CLOCK,TEMPERATURE -i "$GPUS" > "$CELL/clocks_post.txt" 2>&1
"$SCRIPT_DIR/shutdown.sh" > "$CELL/shutdown.txt" 2>&1
if [ -f "$CELL/server.log" ]; then
  SZ=$(du -h "$CELL/server.log" | cut -f1); LN=$(wc -l < "$CELL/server.log")
  { echo "### $NAME server log (original $SZ, $LN lines)"; echo "### counts"
    for p in "cuda-graph miss" "Mooncake read failed" "error in main loop" "Client cancelled request"; do
      echo "  $p: $(grep -c "$p" "$CELL/server.log")"; done
    # the capture log line has two shapes: 6666c623 prints
    # CudaGraphKey(graph_walk=..., bs=N), 485beeb7 prints walk[bs=N,...]
    echo; echo "### captured CUDA graphs"
    grep -oE "Captured CUDA graph for [A-Za-z_]+: ([a-z_]+\[bs=[0-9]+|CudaGraphKey\(graph_walk='[a-z_]+'.*bs=[0-9]+, )" "$CELL/server.log" \
      | sed -E "s/CudaGraphKey\(graph_walk='([a-z_]+)'.*bs=([0-9]+), /\1[bs=\2/" | sort -u
    echo; echo "### head 400"; head -400 "$CELL/server.log"
    echo; echo "### tail 400"; tail -400 "$CELL/server.log"; } > "$CELL/server.excerpt.txt" 2>&1
  echo "  server log $SZ / $LN lines; cuda-graph miss=$(grep -c 'cuda-graph miss' "$CELL/server.log")"
  [ "${KEEP_SERVER_LOG:-0}" = "1" ] || rm -f "$CELL/server.log"
fi
echo "$NAME done $(date -u +%FT%TZ)"
