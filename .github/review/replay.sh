#!/usr/bin/env bash
# Reproduce the CI reviewer locally against one past PR.
#
# Builds the same layout the workflow gives the model -- base tree at the root,
# the PR's tree in pr-head/, the diff in pr.diff -- then overlays your *working
# copy* of AGENTS.md and .github/review/ on top, so you can tune rules that
# aren't committed yet and see the effect on a real PR.
#
# Use it on PRs where a human reviewer caught a real bug: if the reviewer misses
# those, tighten AGENTS.md. If it reports things nobody cared about, tighten the
# bar in review-prompt.md. Do this before making it a required check.
#
#   usage: .github/review/replay.sh <pr-number> [outfile]
#   needs: gh (authenticated), the `claude` CLI, ANTHROPIC_API_KEY

set -euo pipefail

PR="${1:?usage: replay.sh <pr-number> [outfile]}"
OUT="${2:-/tmp/mstar-review-$PR.md}"
REPO="${REPO:-mstar-project/mstar}"
MODEL="${MODEL:-claude-sonnet-5-5}"

ROOT="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d)"
cleanup() {
  git -C "$ROOT" worktree remove --force "$WORK/base" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

read -r base head title < <(
  gh pr view "$PR" --repo "$REPO" \
    --json baseRefOid,headRefOid,title \
    -q '[.baseRefOid, .headRefOid, .title] | @tsv' | tr '\t' ' '
)
echo "PR #$PR: $title" >&2
echo "base $base -> head $head" >&2

# refs/pull/N/head works for fork PRs, where the head sha is not on origin.
git -C "$ROOT" fetch -q origin "refs/pull/$PR/head" "$base"
git -C "$ROOT" worktree add -q --detach "$WORK/base" "$base"

# The PR tree as a plain directory: no git metadata, nothing runnable wired up.
mkdir -p "$WORK/base/pr-head"
git -C "$ROOT" archive "$head" | tar -x -C "$WORK/base/pr-head"

gh pr diff "$PR" --repo "$REPO" > "$WORK/base/pr.diff"

# Overlay the rules and skills you're editing, rather than the versions at the base sha.
cp "$ROOT/AGENTS.md" "$WORK/base/AGENTS.md"
mkdir -p "$WORK/base/.github/review"
cp "$ROOT/.github/review/"*.md "$WORK/base/.github/review/"
rm -rf "$WORK/base/.claude/skills"
mkdir -p "$WORK/base/.claude"
cp -r "$ROOT/.claude/skills" "$WORK/base/.claude/skills"

cd "$WORK/base"
claude -p \
  --model "$MODEL" \
  --disallowedTools Edit,Write,MultiEdit,NotebookEdit,WebFetch,WebSearch \
  "Read .github/review/review-prompt.md from the repository root and follow it exactly.
   It defines your trust boundary, your method, your reporting bar and your output format.

   PR #$PR: $title" | tee "$OUT"

echo >&2
echo "saved to $OUT" >&2
