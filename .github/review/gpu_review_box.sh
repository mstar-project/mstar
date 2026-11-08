#!/usr/bin/env bash
# The live-test session on a GPU box: one Claude Code run of the live-pr-test skill against the
# reviewed commit. gpu_review.py launches it with `gpu.py launch --run`, so the box (a Slurm job or
# an Eras Pod) is this script, and its GPUs go when it ends.
#
# Its inputs are box_common.sh, run.json (repo, PR, commits, model), brief.md (what to do), findings.json (the
# review's notes for the run: model text, which the brief calls untrusted data), pr-body.md (the
# PR's title and description, untrusted data too), and on Slurm
# anthropic.key (mode 0600, removed once read). On Slurm they are in the job's directory, where
# the job runs (--chdir; $0 is Slurm's spool copy of this script), and gpu.py removes that
# directory when the job ends. On Eras they are in /run/live-pr, and the key is an Eras secret at
# /run/live-pr-key/value. Nothing here holds a GitHub token or a cloud key.
set -uo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then W=$PWD IN=$PWD; else W=/workspace/live-pr IN=/run/live-pr; fi
OUT=$W/out
mkdir -p "$OUT" "$W/home" "$W/tmux"
# shellcheck disable=SC1091
source "$IN/box_common.sh"
MAIN_SHA=$(field main_sha)

# A fresh home and Claude config: on a shared node /root/.claude holds other people's settings,
# hooks and credentials. tmux gets its own socket directory, so nothing joins a shared server.
export HOME=$W/home CLAUDE_CONFIG_DIR=$W/home/.claude TMUX_TMPDIR=$W/tmux XDG_CACHE_HOME=$W/cache
export LIVE_PR_DIR=$W/live-pr LIVE_PR_IN=$IN LIVE_PR_OUT=$OUT HF_HOME=${HF_HOME:-$W/hf}
# The bot's cache volume (Eras, read-only at /cache, when a free one was bound): the compile caches
# where bootstrap.sh looks for them (beside the HF home), and any models on it in this run's HF cache
# as local directories of links (the Hub client writes refs/ inside a model's directory).
[ ! -d /cache/live-pr-compile-cache ] || ln -sfn /cache/live-pr-compile-cache "$(dirname "$HF_HOME")/live-pr-compile-cache"
if [ -d /cache/hf/hub ]; then
  mkdir -p "$HF_HOME/hub"
  for m in /cache/hf/hub/models--*; do [ -e "$HF_HOME/hub/${m##*/}" ] || cp -rs "$m" "$HF_HOME/hub/"; done
fi
ANTHROPIC_API_KEY=$(cat /run/live-pr-key/value 2> /dev/null || cat "$IN/anthropic.key")
export ANTHROPIC_API_KEY
rm -f "$IN/anthropic.key"

# bootstrap.sh keeps the compile-cache tarball beside the HF home, which on a Slurm node is one file every
# live test restores, main arms included. So HF_HOME is a link in the job's directory: the session's
# CACHE_TAR is the job's, seeded with links to the shared tarballs. They restore from there; save_caches
# replaces the link in the job's directory and never reaches the shared file.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ln -sfn "$HF_HOME" "$W/hf" && mkdir -p "$W/live-pr-compile-cache"
  for t in "$(dirname "$HF_HOME")"/live-pr-compile-cache/*.tar; do [ ! -e "$t" ] || ln -sf "$t" "$W/live-pr-compile-cache/"; done
  HF_HOME=$W/hf
fi

# Two runs can share a Slurm node (no job container, network namespace or private /tmp), so each takes a
# port block from its job id (a server's default socket prefix comes from its port), a TMPDIR of its own,
# and its own server.sh state; on Eras the box is its own.
export LIVE_PR_PORT=$((10000 + ${SLURM_JOB_ID:-0} % 1100 * 20))
export TMPDIR=$W/tmp SERVER_STATE_DIR=$W/servers
mkdir -p "$TMPDIR"

finish() {
  [ -s "$OUT/result.json" ] || printf '{"verdict": "unavailable", "summary": "%s"}\n' "$1" > "$OUT/result.json"
  touch "$OUT/done"
  [ -n "${SLURM_JOB_ID:-}" ] || await_collection
  exit 0
}
# Slurm preempts the job with a TERM: say so, rather than leaving no result at all. `wait`, below, lets
# the trap run at once instead of after the session has exited.
trap 'kill "${claude_pid:-}" 2> /dev/null; finish "preempted before the session finished"' TERM

# Trusted: main's skills, AGENTS.md and scripts, at the commit the workflow read its rules from.
clone_main "$MAIN_SHA"
curl -fsSL https://claude.ai/install.sh | bash -s "$(field claude)" > "$OUT/install.log" 2>&1 \
  || finish "couldn't install Claude Code"

cd "$W" || finish "no work directory"
"$HOME/.local/bin/claude" -p "$(cat "$IN/brief.md")" \
  --model "$(field model)" --effort high --max-turns "$(field turns)" --output-format json \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep" --disallowedTools "Skill,WebSearch" \
  > "$OUT/claude.json" 2> "$OUT/claude.err" &
claude_pid=$!
wait "$claude_pid"
finish "the session ended without a result (exit $?)"
