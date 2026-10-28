# Automated PR review

An advisory LLM reviewer runs on contributors' non-draft PRs, including PRs from forks. It posts each finding as an inline comment, tagged P0-P3, plus a summary comment that covers the PR description and any open questions. The summary opens with a status line: 🔴 for any open P0 or P1, 🟡 for anything else to report, 🟢 for nothing, then the head commit and the mode. An inline finding stays open until its thread is resolved or a commit changes its lines. Each run, including each `/review`, posts a new summary comment at the bottom of the conversation and hides the earlier ones as outdated. It is not a merge gate and does not submit GitHub reviews. Each review also sets the PR's `type/*` labels, one to three of `new-model`, `bug`, `perf`, `feature`, `refactor`, `test`, `docs` and `ci`, replacing any earlier ones; the `area/*`, `model/*` and `size/*` labels come from [`../workflows/pr-labels.yml`](../workflows/pr-labels.yml).

"Contributor" means the PR author's GitHub author association is `OWNER`, `MEMBER`, `COLLABORATOR` or `CONTRIBUTOR`: the team, plus anyone who has had a PR merged here. Their PRs are reviewed automatically, and they can comment `/review` on their own PR. Any other request (`/review` on someone else's PR, `/review thorough`, or the label) takes write access to this repo, which the workflow checks against the user's actual permission. Someone with write access can `/review` any PR, including a first-timer's.

| Trigger | Mode |
| --- | --- |
| A contributor's non-draft PR is opened, reopened or marked ready | `light` (P0-P2, effort `high`), once |
| Someone with write access comments exactly `/review` on a PR | `light`, e.g. to re-review after pushes |
| A contributor comments exactly `/review` on their own PR, without the `skip-ai-review` label | `light` |
| Someone with write access comments exactly `/review thorough`, or applies the `ai-review-thorough` label | `thorough` (also P3 improvements and design questions; effort `xhigh`) |
| Anyone who may `/review` comments exactly `/review continue` after a review stopped part-way | continues that review in its mode, in a fresh session; a thorough one needs write access |

Pushes don't trigger a review on their own. The `skip-ai-review` label turns off the automatic run and the author's own `/review`; a `/review` from someone with write access still runs.

The action refuses actors without write access, which is most contributors on a fork. For those runs alone the workflow passes the PR author's login, and nobody else's, as the action's `allowed_non_write_users`.

A stacked PR (one whose base is another feature branch) gets the automatic review only if its base branch has a current copy of the workflow, because `pull_request_target` runs the base branch's workflow file. Comment `/review` on stacked PRs; that always runs `main`'s. Either way the rules are read from `main`.

The bot reacts 👀 to a `/review` comment when it picks it up. A review that doesn't complete (an API error, running out of turns, a timeout, a fork that no longer exists) leaves a PR comment linking the log, never a red check; a later failure updates that comment rather than adding another, and a later run that finishes rewrites it to point at its summary. A request the workflow declines (no write access and not the author, or no API key) exits green without commenting, and so does an automatic review the API refuses because the account is out of credit (with a warning in the log); a `/review` refused that way gets a one-line reply saying so.

Running out of turns is handled in two ways. A review that finished but went past `--max-turns` (the action then reports a failure) still has its summary in the action's execution file, and the workflow posts it with a note. A review that stopped part-way gets a "didn't complete" comment listing the inline comments it made (the action posts them even when the run fails), and a handoff is saved as a workflow artifact for 7 days. The handoff ([`session.py`](session.py) `handoff`) holds the model's latest progress note (the prompt has it keep one, with its candidate findings and their anchors, what it covered and what is left; it lists unverified leads, so it stays out of the public comment), the inline comments it made and whether each posted, the files it read, and the changed files it never opened; it is built from the action's execution file, or after a timeout from the session transcript. `/review continue` starts a fresh session with a fresh turn budget and gives it the handoff as `handoff.md`, rather than resuming the old session, which sends the whole old transcript again. Once that transcript's prompt cache has expired, all of it is written to cache again on the first call and reread on every later turn (a hypothesis from one resumed run on #568, whose token counts weren't recorded: its 8 turns cost $3.28 by the action's `total_cost_usd`, against $2.13 for the 41 turns of the run it resumed). The fresh session knows only what the handoff says, so it rereads `AGENTS.md` and the code its findings anchor on; inline comments already on the PR are in `pr-comments.json` and in the handoff, and it posts only findings that have none. A continued run that stops too merges its handoff onto the one it started from. `/review continue` refuses, in a comment of its own that leaves the stopped review's comment as it was, if there is no saved handoff, if a later run already finished (any run that finishes marks the stopped review's comment resolved), if the PR has new commits since (comment `/review` then), or if a thorough review is continued without write access. Only artifacts made by this workflow on `issue_comment` or `pull_request_target` are used: a fork's own `pull_request` workflow can upload an artifact with any name.

| File | Role |
| --- | --- |
| [`../../AGENTS.md`](../../AGENTS.md) | **The substance.** Numbered invariants the reviewer checks and cites. Also read by contributors' own coding agents. |
| [`review-prompt.md`](review-prompt.md) | How the reviewer behaves: trust boundary, path routing, reporting bar, output format. |
| [`../workflows/claude-review.yml`](../workflows/claude-review.yml) | The workflow, and the safety rules that come with `pull_request_target`. |
| [`render_summary.py`](render_summary.py) | The summary's fields (its `--schema`), and the comment they render to. |
| [`session.py`](session.py) | Reads a run's execution file and transcripts: the usage record, and the handoff for `/review continue`. |
| [`replay.sh`](replay.sh) | Run the reviewer locally against a past PR, to tune before trusting it. |
| [`../../.claude/skills/benchmarking/SKILL.md`](../../.claude/skills/benchmarking/SKILL.md) | Benchmarking methodology. The reviewer reads it to judge performance claims; contributors follow it to produce them. |
| [`../../.claude/skills/async-worker/SKILL.md`](../../.claude/skills/async-worker/SKILL.md) | The worker's async step loop, stage by stage. The reviewer reads it before judging any worker-loop change. |
| [`../../.claude/skills/add-mstar-model/SKILL.md`](../../.claude/skills/add-mstar-model/SKILL.md) | The model-porting workflow end to end, plus a registration checklist and a topology map. |

Most tuning belongs in `AGENTS.md`, not in the prompt. The prompt says *how to review*; the invariants say *what is true about this codebase*, which is the part that makes the review worth more than a generic bot.

Procedural knowledge — how to run a benchmark, how to debug a hang — belongs in `.claude/skills/`, not in `AGENTS.md`. `AGENTS.md` states invariants the reviewer cites by number; a skill is a runbook an agent loads when it has that task in hand. Mirror a skill to `.agents/skills/` if you want non-Claude tooling to pick it up too.

## One-time setup

1. Add `ANTHROPIC_API_KEY` as an **organization** secret on `mstar-project` (Settings → Secrets and variables → Actions), scoped to this repository. Bill it to a project account rather than a personal one so usage is attributable.
2. Create a `skip-ai-review` label (applying it to a PR skips the automatic review) and an `ai-review-thorough` label (applying it runs one thorough review).
3. Give the bot its own name and logo with the `mstar-review` GitHub App, owned by `mstar-project` and installed on this repository only, with contents read and issues and pull requests write. Add its client ID as the repository variable `REVIEW_APP_CLIENT_ID` and a private key as the secret `REVIEW_APP_PRIVATE_KEY`. This workflow and [`../workflows/pr-labels.yml`](../workflows/pr-labels.yml) post with the app's token, and with the job's own token as `github-actions[bot]` until both are set.
4. Nothing else. The workflow is advisory, so it needs no branch-protection change.

Until the API key exists the job logs a warning and exits green, so the workflow can land first.

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

Current levers, roughly in order of effect: only contributors' PRs are reviewed automatically, once per PR rather than per push; re-reviews are on request with `/review`, which takes write access or authorship of the PR (light mode only for authors, and `skip-ai-review` stops them); draft PRs are skipped; `concurrency` cancels a run superseded by a newer one on the same PR; in light mode, diffs over about 4,000 lines are triaged rather than covered in full; `--max-turns 40` caps a runaway session (`/review continue` grants another 40; the `CLAUDE_REVIEW_MAX_TURNS` repository variable overrides it). The model is pinned to Opus in the workflow, at effort `high` for light reviews and `xhigh` for thorough ones (Claude Code's default without `--effort` is medium); lowering light to `medium` is the next cost lever. Drop to Sonnet if cost matters more than recall and replay shows it catching the same defects.

Every run that reaches Claude records what it used: API calls (one per API response, however many transcript lines it spans), uncached input, cache-write, cache-read and output tokens (per model from the run's result, or summed from the transcript when a timeout leaves no result), turns and cost, with the mode, whether it was a `/review continue`, the PR and the head commit. It goes in the run's job summary as a table, in the log as one `review-usage {...}` JSON line, and in a `review-usage-<pr>-<run id>-<attempt>` artifact kept for 30 days, never on the PR. Compare runs with `gh run download <run id> -n review-usage-<pr>-<run id>-<attempt>`.

## Threat model

`pull_request_target` and `issue_comment` run with secrets on PRs from forks, so the PR's contents and comments are untrusted input. A comment only selects the mode, by exact match in the job's `if`; its text never reaches a shell or the prompt. The workflow never executes PR code, checks the PR out only into `pr-head/`, reads its rules from the base tree, gives the model no shell and no mutation or network tools, redacts credential-shaped strings from everything it posts, and holds write permission on pull requests alone. The residual risk is prompt injection producing a misleading comment: annoying, not dangerous. A contributor's run on their own PR reads exactly the content a maintainer's `/review` of that PR would, with the same tools and token, so it adds cost exposure (bounded by the rules above and one run at a time per PR), not new capability. A saved handoff (for `/review continue`) is a public artifact holding the model's notes on the PR's content, redacted like everything else the bot posts; a continued run trusts only artifacts from this workflow's own `issue_comment` and `pull_request_target` runs, checks the handoff's PR, head commit and mode, and gives the model its notes as untrusted input. Preserve the rules listed in the workflow header when editing it.
