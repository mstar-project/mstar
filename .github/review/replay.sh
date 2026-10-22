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
# Nothing is posted. The model prints the inline comments and the summary it
# would have posted, so you can check them (and their line anchors) by hand.
#
#   usage: .github/review/replay.sh <pr-number> [outfile]
#   MODE=thorough for the on-demand mode (default: light)
#   HEAD=<full sha> to review the PR as it was at an earlier commit, e.g. the one a
#   human reviewer flagged a bug on; existing comments are then left out so the
#   reviewer can't read the answer
#   REMOTE=<git remote> that has mstar-project/mstar's PR refs (default: origin; in a fork clone, the
#   remote pointing at mstar-project/mstar, e.g. REMOTE=upstream)
#   needs: gh (authenticated), the `claude` CLI, ANTHROPIC_API_KEY

set -euo pipefail

PR="${1:?usage: replay.sh <pr-number> [outfile]}"
OUT="${2:-/tmp/mstar-review-$PR.md}"
REPO="${REPO:-mstar-project/mstar}"
MODEL="${MODEL:-claude-sonnet-5-5}"
MODE="${MODE:-light}"
REMOTE="${REMOTE:-origin}"

ROOT="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d)"
cleanup() {
  git -C "$ROOT" worktree remove --force "$WORK/base" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

# gh api rather than gh pr view, whose baseRefOid field older gh releases lack.
read -r base head title < <(
  gh api "repos/$REPO/pulls/$PR" \
    -q '[.base.sha, .head.sha, .title] | @tsv' | tr '\t' ' '
)
if [ -n "${HEAD:-}" ]; then head="$HEAD"; fi
echo "PR #$PR: $title" >&2
echo "base $base -> head $head" >&2

# refs/pull/N/head works for fork PRs, where the head sha is not on origin.
git -C "$ROOT" fetch -q "$REMOTE" "refs/pull/$PR/head" "$base" ${HEAD:+"$HEAD"}
git -C "$ROOT" worktree add -q --detach "$WORK/base" "$base"

# The PR tree as a plain directory: no git metadata, nothing runnable wired up.
mkdir -p "$WORK/base/pr-head"
git -C "$ROOT" archive "$head" | tar -x -C "$WORK/base/pr-head"

gh pr view "$PR" --repo "$REPO" --json body --jq .body > "$WORK/base/pr-body.md"
if [ -n "${HEAD:-}" ]; then
  git -C "$ROOT" diff "$(git -C "$ROOT" merge-base "$base" "$head")" "$head" > "$WORK/base/pr.diff"
  echo '[]' > "$WORK/base/pr-comments.json"
else
  gh pr diff "$PR" --repo "$REPO" > "$WORK/base/pr.diff"
  gh api --paginate "repos/$REPO/pulls/$PR/comments" \
    --jq '[.[] | {id, user: .user.login, path, line, side, in_reply_to_id, body}]' > "$WORK/base/pr-comments.json"
fi
lines=$(wc -l < "$WORK/base/pr.diff")

# Overlay the rules and skills you're editing, rather than the versions at the base sha.
cp "$ROOT/AGENTS.md" "$WORK/base/AGENTS.md"
mkdir -p "$WORK/base/.github/review"
cp "$ROOT/.github/review/"*.md "$WORK/base/.github/review/"
rm -rf "$WORK/base/.claude/skills"
mkdir -p "$WORK/base/.claude"
cp -r "$ROOT/.claude/skills" "$WORK/base/.claude/skills"

cd "$WORK/base"
# The prompt goes before the flags: --disallowedTools takes a variadic list and
# would otherwise swallow it as one more tool name.
claude -p "Read .github/review/review-prompt.md from the repository root and follow it exactly.
   It defines your trust boundary, your method, your reporting bar and your output format.

   Mode: $MODE
   PR #$PR: $title
   pr.diff is $lines lines.

   DRY RUN: nothing will be posted, and you have no posting tools. Where the prompt
   says to call create_inline_comment, print the call instead as a fenced json block
   with keys path, line, startLine, side and body. Where it says to return the summary
   fields, print them as one fenced json block instead. Print nothing else." \
  --model "$MODEL" \
  --disallowedTools Edit,Write,MultiEdit,NotebookEdit,WebFetch,WebSearch | tee "$OUT"

echo >&2
echo "saved to $OUT" >&2
