#!/bin/bash
# Does a resumed windowed rollout still come out the same as on `main`?
#
# Runs the same seeded two-turn rollout twice on one GPU -- once against a
# checkout of main (whose session knobs ride on model_kwargs) and once against
# this branch (where they are request fields) -- and diffs the decoded frames.
# Sequential on purpose: cosmos3 is deterministic only at concurrency 1.
#
# Submit from the repo root; everything is resolved relative to it. Override
# REPO or LOG_ROOT only to point somewhere else on purpose.
#
#   sbatch test/cosmos3/parity_session_resume.sbatch
#   cat .parity-logs/parity-<jobid>/*.log
#
# One venv serves both sides: this branch's rust/ and mstar/graph/runtime/ are
# byte-identical to main's, so the installed mstar_rust extension matches both.
# If main moves and that stops holding, the main side fails at request time with
# a signature mismatch -- the check below catches it before burning the GPU.
#
#SBATCH --job-name=cosmos3-session-parity
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --requeue

set -uo pipefail  # not -e: a difference is a result, not a crash

# the submission directory under sbatch, the shell's cwd when run by hand
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
LOG_ROOT=${LOG_ROOT:-$REPO/.parity-logs}
JOB=${SLURM_JOB_ID:-local}
OUT=$LOG_ROOT/parity-$JOB
mkdir -p "$OUT"
cd "$REPO" || exit 1

PYTHON=${PYTHON:-$REPO/.venv/bin/python}
export PATH="$(dirname "$PYTHON"):$PATH"   # flashinfer JITs with ninja from here
CONFIG=configs/cosmos3_nano_ar.yaml
PORT=8160
PROBE=test/cosmos3/parity_session_resume.py
# The branch commit under test; main is whatever origin/main points at now.
BRANCH_SHA=$(git rev-parse HEAD)
MAIN_WT=$OUT/main-checkout

echo "=== cosmos3 session parity on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}"
echo "=== branch $BRANCH_SHA"
echo "=== logs   $OUT"

git fetch origin main --quiet
echo "=== main   $(git rev-parse --short origin/main)"
# The whole reason one venv works. If this prints anything, stop: the main side
# would run this branch's extension against main's python.
SKEW=$(git diff --name-only HEAD origin/main -- rust/ mstar/graph/runtime/)
if [ -n "$SKEW" ]; then
    echo "=== ABORT: main and this branch differ in the extension's surface:"
    echo "$SKEW"
    echo "=== main's side needs its own mstar_rust build; not guessing here."
    exit 2
fi
# a requeue reuses this job id, so clear a worktree left by the previous try
git worktree remove --force "$MAIN_WT" 2>/dev/null
git worktree add --detach "$MAIN_WT" origin/main >/dev/null 2>&1 \
    || { echo "=== could not create the main worktree"; exit 1; }
trap 'cd "$REPO" && git worktree remove --force "$MAIN_WT" 2>/dev/null' EXIT

# ── one side ────────────────────────────────────────────────────────────────
run_side() {  # run_side <label> <workdir> <api> <npz>
    local label=$1 workdir=$2 api=$3 npz=$4
    local log=$OUT/$label-server.log
    local prefix=/tmp/mstar_${USER}_parity_${JOB}_${label}/

    echo
    echo "--- $label: starting the server from $workdir"
    ( cd "$workdir" && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        "$PYTHON" -m mstar.api_server.entrypoint \
        --config "$CONFIG" --port $PORT --socket-path-prefix "$prefix" \
        --upload-dir "/tmp/mstar_uploads_${USER}/" \
        --tensor-comm-protocol SHM > "$log" 2>&1 ) &
    local shell_pid=$!

    local ready=0
    for _ in $(seq 1 180); do   # cosmos3 loads in ~2-4 min cold
        if curl -sf -m 5 "http://localhost:$PORT/health" >/dev/null 2>&1; then
            ready=1; break
        fi
        kill -0 "$shell_pid" 2>/dev/null || break
        sleep 10
    done
    if [ "$ready" != 1 ]; then
        echo "--- $label: never became healthy; its last lines:"
        tail -30 "$log"
        stop_side "$prefix"
        return 1
    fi
    echo "--- $label: healthy; driving the two turns"
    "$PYTHON" "$PROBE" --url "http://localhost:$PORT" --api "$api" \
        --out "$npz" 2>&1 | sed 's/^/    /'
    local rc=${PIPESTATUS[0]}
    echo "--- $label: probe exit $rc, tracebacks in its server log: $(grep -c Traceback "$log")"
    stop_side "$prefix"
    return $rc
}

stop_side() {  # the conductor outlives its parent, so signal by socket prefix
    local prefix=$1
    for pid in $(pgrep -u "$USER" -f "$prefix" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        kill -INT "$pid" 2>/dev/null
    done
    for _ in $(seq 1 30); do
        pgrep -u "$USER" -f "$prefix" >/dev/null 2>&1 || break
        sleep 2
    done
    pgrep -u "$USER" -f "$prefix" >/dev/null 2>&1 \
        && echo "--- WARNING: processes still up for $prefix"
    sleep 5
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1
}

# main first: if its side cannot run, there is nothing to compare against
run_side main "$MAIN_WT" kwargs "$OUT/main.npz" || { echo "=== main's side failed"; exit 1; }
run_side branch "$REPO" fields "$OUT/branch.npz" || { echo "=== this branch's side failed"; exit 1; }

# ── the verdict ─────────────────────────────────────────────────────────────
echo
echo "=== comparing"
"$PYTHON" "$PROBE" --compare "$OUT/main.npz" "$OUT/branch.npz" 2>&1 | sed 's/^/    /'
rc=${PIPESTATUS[0]}
echo
if [ $rc -eq 0 ]; then
    echo "=== PARITY: the resume is numerically unchanged from main"
else
    echo "=== DIFFERENCE: the port changed the rollout; frames are in $OUT"
fi
exit $rc
