#!/bin/bash
# Live session tests for the `test_text_session` deployment, one array task per
# session configuration in configs/text_session/. Each task brings a server up
# on its own GPU, port and ZMQ socket prefix, runs the scenarios that config can
# show, tears the server down, and fails if any scenario failed.
#
#   sbatch test/text_session/session_matrix.sbatch
#   squeue -u $USER
#   tail -f $LOG_ROOT/<jobid>_<task>/task.log
#
# LOG_ROOT (sbatch --export, or edit below) is where the per-task logs land; it
# must exist at submit time because #SBATCH --output cannot create it.
#
#SBATCH --job-name=mstar-sessions
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --array=0-3%2
#SBATCH --output=/m-coriander/coriander/naomi/mstar_session_tests/slurm-%A_%a.out
#SBATCH --requeue

set -uo pipefail  # not -e: a failing scenario is a result, not a reason to stop

REPO=${REPO:-/m-coriander/coriander/naomi/mstar}
LOG_ROOT=${LOG_ROOT:-/m-coriander/coriander/naomi/mstar_session_tests}
JOB=${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
TASK=${SLURM_ARRAY_TASK_ID:-0}

# One config per array task; add a config here and widen --array to match.
CONFIGS=(capacity_keep capacity_evict ttl_idle ttl_absolute)
NAME=${CONFIGS[$TASK]}
CONFIG=configs/text_session/$NAME.yaml

OUT=$LOG_ROOT/$JOB-$NAME
mkdir -p "$OUT"
SERVER_LOG=$OUT/server.log
# A port and a socket prefix of its own per task: --port alone does not isolate
# the ZMQ IPC sockets, so two servers on one host steal each other's handshake.
PORT=$((8200 + TASK))
PREFIX=/tmp/mstar_${USER}_sessmatrix_${JOB}/
URL=http://localhost:$PORT

cd "$REPO" || exit 1
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
export PATH="$(dirname "$PYTHON"):$PATH"

echo "=== $NAME on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, port $PORT"
echo "=== config $CONFIG"
echo "=== logs   $OUT"
sed -n '/^sessions:/,/^node_groups:/p' "$CONFIG" | sed '$d'

# ── the server ──────────────────────────────────────────────────────────────
CONFIG=$CONFIG PORT=$PORT SOCKET_PREFIX=$PREFIX \
    bash test/text_session/launch_server.sh > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

stop_server() {
    echo "=== stopping the server"
    # SIGINT, not SIGTERM: the entrypoint's handler is what shuts the conductor
    # and the workers down and frees the GPU.
    kill -INT "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 2
    done
    # the conductor outlives its parent often enough to need its own signal; it
    # is the one holding the weights, and this job's prefix identifies it
    for pid in $(pgrep -u "$USER" -f "$PREFIX" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        echo "=== signalling leftover $pid"
        kill -INT "$pid" 2>/dev/null
    done
    sleep 5
    pgrep -u "$USER" -f "$PREFIX" >/dev/null 2>&1 \
        && echo "=== WARNING: processes still up for $PREFIX"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}
trap stop_server EXIT

echo "=== waiting for the model to load (up to 25 min)"
READY=0
for _ in $(seq 1 150); do
    if curl -sf -m 5 "$URL/health" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "=== the server exited while loading; its last lines:"
        tail -40 "$SERVER_LOG"
        exit 1
    fi
    sleep 10
done
if [ "$READY" != 1 ]; then
    echo "=== the server never became healthy; its last lines:"
    tail -40 "$SERVER_LOG"
    exit 1
fi
echo "=== healthy; session state: $(grep -h 'Session state held by' "$SERVER_LOG" | tail -1)"

# ── the scenarios ───────────────────────────────────────────────────────────
FAILED=()
run() {  # run <label> <command...>
    local label=$1; shift
    echo
    echo "--- $label"
    if "$@" 2>&1 | sed 's/^/    /'; then
        echo "--- $label: PASS"
    else
        echo "--- $label: FAIL"
        FAILED+=("$label")
    fi
}

scenario() {  # scenario <name>
    run "$1" "$PYTHON" test/text_session/scenarios.py "$1" \
        --url "$URL" --config "$CONFIG"
}

case $NAME in
  capacity_keep)
    run happy "$PYTHON" test/text_session/session_request.py --url "$URL"
    run happy-ws "$PYTHON" test/text_session/ws_session_request.py \
        --url "ws://localhost:$PORT"
    scenario limits
    scenario capacity_refuse
    run budget "$PYTHON" test/text_session/budget_probe.py --url "$URL" \
        --config "$CONFIG"
    ;;
  capacity_evict)
    run happy "$PYTHON" test/text_session/session_request.py --url "$URL"
    scenario capacity_evict
    run budget "$PYTHON" test/text_session/budget_probe.py --url "$URL" \
        --config "$CONFIG"
    ;;
  ttl_idle)
    scenario limits
    scenario ttl_idle
    ;;
  ttl_absolute)
    scenario ttl_absolute
    ;;
  *)
    echo "no scenarios for $NAME"; exit 1 ;;
esac

# ── the verdict ─────────────────────────────────────────────────────────────
echo
echo "=== tracebacks in the server log: $(grep -c 'Traceback' "$SERVER_LOG")"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "=== $NAME: every scenario passed"
    exit 0
fi
echo "=== $NAME: ${#FAILED[@]} scenario(s) failed: ${FAILED[*]}"
exit 1
