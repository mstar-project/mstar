#!/usr/bin/env bash
# The live-test session on a GPU box: one Claude Code run of the live-pr-test skill against the
# reviewed commit. gpu_review.py launches it with `gpu.py launch --run`: on Eras the box (a Pod) is
# this script, and its GPUs go when it ends; on DigitalOcean cloud-init starts it at boot, and the
# runner deletes the droplet once it has the results.
#
# Its inputs, in /run/live-pr, are box_common.sh, run.json (repo, PR, commits, model), brief.md (what
# to do), findings.json (the review's notes for the run: model text, which the brief calls untrusted
# data) and pr-body.md (the PR's title and description, untrusted data too). The Anthropic key is at
# /run/live-pr-key/value: an Eras secret, or on DigitalOcean the review's own key. Nothing here holds a
# GitHub token or a cloud key.
set -uo pipefail

W=/workspace/live-pr IN=/run/live-pr
OUT=$W/out
mkdir -p "$OUT" "$W/home"
# shellcheck disable=SC1091
source "$IN/box_common.sh"
MAIN_SHA=$(field main_sha)

# A fresh home and Claude config, so the session loads no settings or hooks the image came with.
export HOME=$W/home CLAUDE_CONFIG_DIR=$W/home/.claude XDG_CACHE_HOME=$W/cache
export LIVE_PR_DIR=$W/live-pr LIVE_PR_IN=$IN LIVE_PR_OUT=$OUT HF_HOME=${HF_HOME:-$W/hf}
# The bot's cache volume (Eras, read-only at /cache, when a free one was bound): the compile caches
# where bootstrap.sh looks for them (beside the HF home), and any models on it in this run's HF cache
# as local directories of links (the Hub client writes refs/ inside a model's directory).
[ ! -d /cache/live-pr-compile-cache ] || ln -sfn /cache/live-pr-compile-cache "$(dirname "$HF_HOME")/live-pr-compile-cache"
if [ -d /cache/hf/hub ]; then
  mkdir -p "$HF_HOME/hub"
  for m in /cache/hf/hub/models--*; do [ -e "$HF_HOME/hub/${m##*/}" ] || cp -rs "$m" "$HF_HOME/hub/"; done
fi
ANTHROPIC_API_KEY=$(cat /run/live-pr-key/value)
export ANTHROPIC_API_KEY

# What finish writes when the session left no result.json: `unavailable`, which says nothing about the PR, until
# the session starts; then a verdict outside the enum, which the bot reads as no valid verdict (`none`) and counts
# as a failure: the PR's code may have run, so a session that dies, or stops at its turn cap, doesn't count as
# nothing.
VERDICT=unavailable
finish() {
  trap - EXIT
  [ -s "$OUT/result.json" ] || printf '{"verdict": "%s", "summary": "%s"}\n' "$VERDICT" "$1" > "$OUT/result.json"
  touch "$OUT/done"
  await_collection
  exit 0
}
# However bash exits (an unset variable, say), the runner sees `done`: a DigitalOcean droplet has no deadline
# of its own, so without it the runner would wait out its timeout on a box billing for nothing.
trap 'finish "the box script stopped early (exit $?)"' EXIT
# Trusted: main's skills, AGENTS.md and scripts, at the commit the workflow read its rules from.
clone_main "$MAIN_SHA"
curl -fsSL https://claude.ai/install.sh | bash -s "$(field claude)" > "$OUT/install.log" 2>&1 \
  || finish "couldn't install Claude Code"

cd "$W" || finish "no work directory"
VERDICT=no-result
# The session stops at run.json's end_by, a margin before the box's deadline, so what it wrote (the report it
# writes as it goes, progress.log) is collected rather than lost with the box; one that ran out of time is partial.
END_BY=$(field end_by) NOW=$(date +%s)
timeout --kill-after=60 "$(( END_BY > NOW ? END_BY - NOW : 1 ))" "$HOME/.local/bin/claude" -p "$(cat "$IN/brief.md")" \
  --model "$(field model)" --effort high --max-turns "$(field turns)" --output-format json \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep" --disallowedTools "Skill,WebSearch" \
  > "$OUT/claude.json" 2> "$OUT/claude.err"
CODE=$?
[ "$(date +%s)" -lt "$END_BY" ] || { VERDICT=partial; finish "the session ran out of time"; }
finish "the session ended without a result (exit $CODE)"
