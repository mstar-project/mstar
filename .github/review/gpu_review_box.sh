#!/usr/bin/env bash
# The live-test session on a GPU box: one Claude Code run of the live-pr-test skill against the
# reviewed commit. gpu_review.py launches it with `gpu.py launch --run`, so the box (an Eras Pod)
# is this script, and its GPUs go when it ends.
#
# Its inputs, in /run/live-pr, are box_common.sh, run.json (repo, PR, commits, model), brief.md (what
# to do), findings.json (the review's notes for the run: model text, which the brief calls untrusted
# data) and pr-body.md (the PR's title and description, untrusted data too). The Anthropic key is an
# Eras secret at /run/live-pr-key/value. Nothing here holds a GitHub token or a cloud key.
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

finish() {
  [ -s "$OUT/result.json" ] || printf '{"verdict": "unavailable", "summary": "%s"}\n' "$1" > "$OUT/result.json"
  touch "$OUT/done"
  await_collection
  exit 0
}
# Trusted: main's skills, AGENTS.md and scripts, at the commit the workflow read its rules from.
clone_main "$MAIN_SHA"
curl -fsSL https://claude.ai/install.sh | bash -s "$(field claude)" > "$OUT/install.log" 2>&1 \
  || finish "couldn't install Claude Code"

cd "$W" || finish "no work directory"
"$HOME/.local/bin/claude" -p "$(cat "$IN/brief.md")" \
  --model "$(field model)" --effort high --max-turns "$(field turns)" --output-format json \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep" --disallowedTools "Skill,WebSearch" \
  > "$OUT/claude.json" 2> "$OUT/claude.err"
finish "the session ended without a result (exit $?)"
