# Automated PR review

An advisory LLM reviewer runs on contributors' non-draft PRs. It posts each finding as an inline comment, tagged P0-P3, plus one summary comment that covers the PR description and any open questions, and it updates that summary on each run. It is not a merge gate and does not submit GitHub reviews.

"Contributor" means GitHub's author association `OWNER`, `MEMBER`, `COLLABORATOR` or `CONTRIBUTOR`: the team, plus anyone who has had a PR merged here. First-time contributors get no automatic review; a contributor can comment `/review` on their PR to run one.

| Trigger | Mode |
| --- | --- |
| A contributor opens, reopens or marks ready a non-draft PR | `light` (P0-P2), once |
| A contributor comments exactly `/review` on a PR | `light`, e.g. to re-review after pushes |
| A contributor comments exactly `/review thorough`, or someone with triage access applies the `ai-review-thorough` label | `thorough` (also P3 improvements and design questions) |

Pushes don't trigger a review on their own. The `skip-ai-review` label turns off the automatic run; an explicit `/review` still runs.

| File | Role |
| --- | --- |
| [`../../AGENTS.md`](../../AGENTS.md) | **The substance.** Numbered invariants the reviewer checks and cites. Also read by contributors' own coding agents. |
| [`review-prompt.md`](review-prompt.md) | How the reviewer behaves: trust boundary, path routing, reporting bar, output format. |
| [`../workflows/claude-review.yml`](../workflows/claude-review.yml) | The workflow, and the safety rules that come with `pull_request_target`. |
| [`replay.sh`](replay.sh) | Run the reviewer locally against a past PR, to tune before trusting it. |
| [`../../.claude/skills/benchmarking/SKILL.md`](../../.claude/skills/benchmarking/SKILL.md) | Benchmarking methodology. The reviewer reads it to judge performance claims; contributors follow it to produce them. |
| [`../../.claude/skills/async-worker/SKILL.md`](../../.claude/skills/async-worker/SKILL.md) | The worker's async step loop, stage by stage. The reviewer reads it before judging any worker-loop change. |
| [`../../.claude/skills/add-mstar-model/SKILL.md`](../../.claude/skills/add-mstar-model/SKILL.md) | The model-porting workflow end to end, plus a registration checklist and a topology map. |

Most tuning belongs in `AGENTS.md`, not in the prompt. The prompt says *how to review*; the invariants say *what is true about this codebase*, which is the part that makes the review worth more than a generic bot.

Procedural knowledge — how to run a benchmark, how to debug a hang — belongs in `.claude/skills/`, not in `AGENTS.md`. `AGENTS.md` states invariants the reviewer cites by number; a skill is a runbook an agent loads when it has that task in hand. Mirror a skill to `.agents/skills/` if you want non-Claude tooling to pick it up too.

## One-time setup

1. Add `ANTHROPIC_API_KEY` as an **organization** secret on `mstar-project` (Settings → Secrets and variables → Actions), scoped to this repository. Bill it to a project account rather than a personal one so usage is attributable.
2. Create a `skip-ai-review` label (applying it to a PR skips the automatic review) and an `ai-review-thorough` label (applying it runs one thorough review).
3. Install the Claude GitHub App on the repository (https://github.com/apps/claude, or `/install-github-app` in Claude Code). The action exchanges the job's OIDC token for the app's token and posts as the app.
4. Nothing else. The workflow is advisory, so it needs no branch-protection change.

Until the secret exists the job logs a warning and exits green, so the workflow can land first.

## Tuning loop

Do this before anyone comes to rely on it. The failure mode for a review bot is not missing a bug — it is being noisy enough that contributors stop reading it, after which it catches nothing at all.

```bash
.github/review/replay.sh 295        # a PR where review caught something real
.github/review/replay.sh 328        # a large, mostly-mechanical PR
```

Pick ten or so merged PRs: a few where a human reviewer caught a genuine defect, a few that were clean, and a couple of big refactors. Then:

- **Missed a real defect** → the invariant it should have cited is missing or too vague in `AGENTS.md`. Add it with the concrete failure mode, not as a principle.
- **Reported something nobody would ask for a change on** → tighten the "Don't report" list in `review-prompt.md`. Being specific about the *category* works better than lowering a severity dial.
- **Rambled or padded to look useful** → the Findings criteria in `review-prompt.md` are the lever. Zero findings must read as a success.

Track hit rate across the sample as you change things; it is easy to fix one case and regress two.

## Promoting it to a required check

Reasonable criteria, once you have replay data:

- Over your last ~20 PRs, at least 80% of its findings are ones a maintainer agrees with.
- It has caught at least one defect that human review missed.
- Median runtime under ten minutes, and it does not flake.

Even then, prefer keeping it advisory: a non-deterministic check shouldn't have veto power over a merge.

## Cost control

Current levers, roughly in order of effect: only contributors' PRs are reviewed automatically, once per PR rather than per push (re-reviews are on request with `/review`); draft PRs are skipped; `concurrency` cancels a run superseded by a newer one on the same PR; in light mode, diffs over about 4,000 lines are triaged rather than covered in full; `--max-turns 40` caps a runaway session. The model is pinned to Opus in the workflow; drop to Sonnet if cost matters more than recall and replay shows it catching the same defects.

## Threat model

`pull_request_target` and `issue_comment` run with secrets on PRs from forks, so the PR's contents and comments are untrusted input. A comment only selects the mode, by exact match in the job's `if`; its text never reaches a shell or the prompt. The workflow never executes PR code, checks the PR out only into `pr-head/`, reads its rules from the base tree, disables mutation and network tools, and allows only two write tools (inline comments and `gh pr comment`). It posts as the Claude GitHub App, whose installation token can carry broader repository permissions than the job's own `pull-requests: write`; the tool allowlist is what keeps a prompt injection to a bad comment, so keep it narrow. The residual risk is prompt injection producing a misleading comment — annoying, not dangerous. Preserve all four properties listed in the workflow header when editing it.
