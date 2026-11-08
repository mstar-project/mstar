# Automated PR review

An advisory LLM reviewer runs on a PR when someone asks for it with `/review`, including on PRs from forks. Opening a PR or pushing to it never starts a review. It posts each finding as an inline comment, tagged P0-P3, plus a summary comment that covers the PR description and any open questions. The summary opens with a status line: 🔴 for any open P0 or P1, 🟡 for anything else to report, 🟢 for nothing, then the head commit and the mode. Every summary also carries a design-risk rating (`low`, `medium` or `high`: whether a human must make a design or direction decision before merging), with a "Design" section (the design verdict, in thorough runs) and, when the risk isn't low, a "Where to look" list of those decisions. An inline finding stays open until a later review (every run re-checks the bot's earlier inline findings) judges it fixed or moot, or a commit changes its lines; resolving its thread doesn't close it. The status line and the approval rule count open findings the same way (`render_summary.open_findings`). Each run, including each `/review`, posts a new summary comment at the bottom of the conversation and hides the earlier ones as outdated. It is not a merge gate. It submits a GitHub review only in `gpu` mode, to approve a PR that meets the rule (or, for an `area/engine` or `area/resources` change, to say it would) under [GPU review and approval](#gpu-review-and-approval). Each review also sets the PR's `type/*` labels, one to three of `new-model`, `bug`, `perf`, `feature`, `refactor`, `test`, `docs` and `ci`, replacing any earlier ones; the `area/*`, `model/*` and `size/*` labels come from [`../workflows/pr-labels.yml`](../workflows/pr-labels.yml).

"Contributor" means the PR author's GitHub author association is `OWNER`, `MEMBER`, `COLLABORATOR` or `CONTRIBUTOR`: the team, plus anyone who has had a PR merged here. They can comment `/review` on their own PR. Any other request (`/review` on someone else's PR, or `/review thorough`) takes write access to this repo, which the workflow checks against the user's actual permission. Someone with write access can `/review` any PR, including a first-timer's.

| Trigger | Mode |
| --- | --- |
| Someone with write access comments exactly `/review` on a PR | `light` (P0-P2, effort `high`) |
| A contributor comments exactly `/review` on their own PR, without the `skip-ai-review` label | `light` |
| Someone with write access comments exactly `/review thorough` | `thorough` (also P3 improvements and design questions; effort `xhigh`) |
| Someone with write access comments exactly `/review gpu` | `gpu`: `thorough`, then a live test on GPUs, then approval if the rule holds (below) |
| Anyone who may `/review` comments exactly `/review continue` after a review stopped part-way | continues that review in its mode, in a fresh session; a thorough one needs write access, and a stopped `gpu` review starts over |

The method in every mode is the [`pr-review`](../../.claude/skills/pr-review/SKILL.md) skill; [`review-prompt.md`](review-prompt.md) adds the bot's inputs, re-review scope and output. A run someone with write access starts has a turn cap meant only to stop a loop (`CLAUDE_REVIEW_MAX_TURNS_TEAM`, 2000, and a 5-hour timeout); a contributor's own light review keeps `CLAUDE_REVIEW_MAX_TURNS` (40).

**Two passes in thorough and gpu modes.** After its own pass, the reviewer gives the same inputs to a subagent with a fresh context that reviews with [`review-standard`](../../.claude/skills/review-standard/SKILL.md) (Gaurav Shah's method: his hard rules, review bar and four passes). The reviewer verifies what it returns like its own findings and posts one merged review; the summary's checked list says what the second pass added.

**Re-reviews.** When the bot's last summary is on a commit that is still an ancestor of the head, the next run re-checks each earlier finding at the head (fixed, still open, or moot), reviews only the commits since, plus whatever they reach elsewhere, and checks what merges of `main` changed under the PR. After a force-push it reviews the whole PR again, and a `thorough` or `gpu` run does too unless the last review was `thorough` or `gpu` and skipped no file. An earlier finding counts as open until a later review judges it fixed or moot or a commit outdates its thread, whether or not someone resolved the thread.

To re-review after pushes, comment `/review` again. The `skip-ai-review` label turns off the author's own `/review`; a `/review` or `/review thorough` from someone with write access still runs.

The action refuses actors without write access, which is most contributors on a fork. For those runs alone the workflow passes the PR author's login, and nobody else's, as the action's `allowed_non_write_users`.

A comment always runs `main`'s copy of the workflow and reads the rules from `main`, whatever the PR's base, so stacked PRs are reviewed the same way.

The bot reacts 👀 to a `/review` comment when it picks it up. A review that doesn't complete (an API error, running out of turns, a timeout, a fork that no longer exists) leaves a PR comment linking the log, never a red check; a later failure updates that comment rather than adding another, and a later run that finishes rewrites it to point at its summary. A request the workflow declines (no write access and not the author, or no API key) exits green without commenting. A review the API refuses because the account is out of credit gets a one-line reply saying so (and a warning in the log) instead of a "didn't complete" comment.

Running out of turns is handled in two ways. A review that finished but went past `--max-turns` (the action then reports a failure) still has its summary in the action's execution file, and the workflow posts it with a note. A review that stopped part-way gets a "didn't complete" comment listing the inline comments it made (the action posts them even when the run fails), and a handoff is saved as a workflow artifact for 7 days. The handoff ([`session.py`](session.py) `handoff`) is built from the action's execution file, or after a timeout from the session transcript. Its reliable part is what the workflow recovers itself: the inline comments the run made and whether each posted, the files it read, the changed files it never opened, and its latest messages, which usually say what it was doing. The prompt also asks the model to keep a progress note (candidate findings with their anchors, what it covered, what is left); that is best-effort, and adds the candidate findings when the model kept one. The notes list unverified leads, so they stay out of the PR conversation, though the artifact can be downloaded.

`/review continue` starts a fresh session with a fresh turn budget and gives it the handoff as `handoff.md`, rather than resuming the old session, which sends the whole old transcript again, probably after its prompt cache has expired: in a measured run, resuming after the cache expired cost about a quarter more than a fresh session, and while it is warm the two cost about the same. The fresh session knows only what the handoff says, so it rereads `AGENTS.md`, any skill its findings cite, and the code they anchor on; inline comments already on the PR are in `pr-comments.json` and in the handoff, and it posts only findings that have none. A continued run that stops too merges its handoff onto the one it started from. `/review continue` refuses, in a comment of its own that leaves the stopped review's comment as it was, if there is no saved handoff, if a later run already finished (any run that finishes marks the stopped review's comment resolved), if the PR has new commits since (comment `/review` then), or if a thorough review is continued without write access. Only artifacts made by this workflow on `issue_comment` are used: a fork's own `pull_request` workflow can upload an artifact with any name.

| File | Role |
| --- | --- |
| [`../../AGENTS.md`](../../AGENTS.md) | **The substance.** Numbered invariants the reviewer checks and cites. Also read by contributors' own coding agents. |
| [`review-prompt.md`](review-prompt.md) | How the reviewer behaves: trust boundary, path routing, reporting bar, output format. |
| [`../workflows/claude-review.yml`](../workflows/claude-review.yml) | The workflow, and the safety rules that come with running on `issue_comment`. |
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
2. Create a `skip-ai-review` label (applying it to a PR stops its author's own `/review`).
3. Give the bot its own name and logo with the `mstar-review` GitHub App, owned by `mstar-project` and installed on this repository only, with contents read and issues and pull requests write. Add its client ID as the repository variable `REVIEW_APP_CLIENT_ID` and a private key as the secret `REVIEW_APP_PRIVATE_KEY`. This workflow and [`../workflows/pr-labels.yml`](../workflows/pr-labels.yml) post with the app's token, and with the job's own token as `github-actions[bot]` until both are set. An app token lasts an hour, so the workflow mints a fresh one before each group of posts, and a run someone with write access starts (up to 5 hours) posts its inline comments with the job's own token.
4. Nothing else. The workflow is advisory, so it needs no branch-protection change.

Until the API key exists the job logs a warning and exits green, so the workflow can land first.

For `/review gpu`, also:

5. **The Slurm node:** a deploy key for the bot (`REVIEW_SLURM_SSH_KEY` secret; add its public key to the node's `extra_authorized_keys`), `REVIEW_SLURM_HOST` (`root@<host>`), `REVIEW_SLURM_PORT` and `REVIEW_SLURM_HOST_KEY` (the node's `ssh-ed25519 …` line, so a reused ngrok port can't impersonate it). Optional: `REVIEW_SLURM_PARTITION` (default `background`).
6. **Eras Cloud:** `ERAS_API_KEY`, from an organization or wallet used only by CI with automatic top-up off, since its balance is the real spending cap. Save the box's Anthropic key as an Eras secret and set `ERAS_RUN_SECRET_ID`; without it the bot skips Eras.
7. **The box's Anthropic key:** `LIVE_TEST_ANTHROPIC_KEY`, a key of its own in a workspace with a spend limit, since the box runs PR code with a shell. It never falls back to `ANTHROPIC_API_KEY`; without it the bot skips the Slurm node.
8. **Limits:** `REVIEW_GPU_MAX_GPUS` (4), `REVIEW_GPU_HOURS` (4), `REVIEW_GPU_MAX_USD_PER_HOUR` (30), and `REVIEW_GPU_MAX_RUNS` (2), the most live tests the bot runs at once across all PRs: `gpu.py launch` checks it (`budget.max_running`) against the bot's Slurm jobs and the live `live-pr-*` instances in the Eras organization (another reason that organization is the bot's alone).
9. **Media:** a public repository under the same owner with GitHub Pages on, named in `REVIEW_MEDIA_REPO` (the name only), with the `mstar-review` app installed and given contents write there.

Without the Slurm and Eras settings a `/review gpu` reviews and then reports that no GPUs are configured.

## GPU review and approval

`/review gpu` runs in two jobs, split by what each may hold:

- **`review`** (as for `/review thorough`, no shell) also says what the live test should run (`full`, `targeted`, or `none` when nothing since the last passing live test reaches code it exercised), and how many GPUs its largest config needs. It re-reviews only from an earlier `thorough` or `gpu` review that skipped no file, as `/review thorough` does, so an approval never rests on a light review; otherwise it reviews the whole PR. `targeted` and `none` hold only when the bot's last live test passed on an ancestor of the head; otherwise the live test runs in full. A `targeted` run is told that commit and gets that test's report, as data.
- **`live-test`** runs no model and no PR code. [`gpu_review.py`](gpu_review.py) gets GPUs through `live-pr-test`'s [`gpu.py`](../../.claude/skills/live-pr-test/scripts/gpu.py): the team's Slurm node when it has them free (in its preemptible `background` partition, so people's jobs come first), else the cheapest open Eras listing under the hourly cap. It starts [`gpu_review_box.sh`](gpu_review_box.sh) there as the job itself: one Claude Code session that follows `live-pr-test` against the reviewed commit and writes a report, media and a verdict. The job waits, copies them back, posts the report (images inline, audio and video in a gallery), and decides approval.
- **When there are no GPUs**, it posts what it found (free GPUs on the node, Eras listings and prices) and stops; the review stands without approval.
- **The box** gets the PR's public code, a fixed brief, the review's notes as data (`findings.json`) and its own spend-capped Anthropic key, never a GitHub token, the Slurm key or the Eras key. It ends when its session does, and Slurm's `--time` or the Eras Pod's deadline end it anyway, whatever happens to the workflow.
- **Media** go to a separate repository served by GitHub Pages (`REVIEW_MEDIA_REPO`), not into this one, whose every clone would carry them: audio, images and video only (never a page or script), up to 100 MB a run, and the newest runs that fit in 900 MB (Pages serves at most 1 GB), kept as a single commit. The comment links the gallery only once the push succeeded. The report and result kept as the run's artifact are the redacted copies.

**The approval rule** (`gpu_review.py approve`), all of: the review finished on its own; the PR's head is still the reviewed commit; the author has write access; the live test passed on that commit (or `none`, and the last passing one was on an ancestor); the review rated the design risk `low` (the PR follows the existing design and decides nothing the team hasn't: no new abstraction, shared-layer restructuring or public surface, and no change to security-relevant code: credentials, auth, API keys and secrets handling, redaction, and the bot's own workflows and sandbox, not every path a request takes) with no open question and no skipped file; no open P0–P2 finding, however its thread was closed; and the change touches none of the bot's own rules, CI, or the dependencies and toolchains (`.github/`, `.claude/`, `scripts/ci/`, `.cargo/`, `AGENTS.md`, `CLAUDE.md`, `pyproject.toml`, `setup.py`, any `Cargo.toml`, `rust-toolchain.toml` or `requirements*.txt`, any lockfile). Other areas are left to the findings, the live test and the design-risk rating. The bot then approves that commit; [`../workflows/pr-labels.yml`](../workflows/pr-labels.yml) dismisses the approval on the next push or base change. A change with a file in `area/engine` or `area/resources` (as [`label_prs.py`](../labels/label_prs.py) maps them) gets a review comment instead, saying the bot would approve it but that this doesn't count toward required reviews. Otherwise the live-test comment says which conditions failed.

Whether the bot's approvals count toward required reviews is a branch-protection setting, an admin's call; the engine rule above keeps the engine's approvals with people either way.

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

Current levers, roughly in order of effect: nothing runs automatically, only on request with `/review`, which takes write access or authorship of the PR (light mode only for authors, and `skip-ai-review` stops them); `concurrency` cancels a run superseded by a newer one on the same PR; a contributor's own light review triages a diff over about 4,000 lines rather than covering it in full, and `--max-turns 40` caps it (`/review continue` grants another 40; the `CLAUDE_REVIEW_MAX_TURNS` repository variable overrides it); a run someone with write access starts reviews a big diff in slices with subagents, with a cap meant only to stop a loop (`CLAUDE_REVIEW_MAX_TURNS_TEAM`, 2000) and a 5-hour timeout. `/review gpu` adds GPU time, bounded by the limits under [One-time setup](#one-time-setup). The model is pinned to Opus in the workflow, at effort `high` for light reviews and `xhigh` for thorough and gpu ones (Claude Code's default without `--effort` is medium); lowering light to `medium` is the next cost lever. Drop to Sonnet if cost matters more than recall and replay shows it catching the same defects.

Every run that reaches Claude records what it used: API calls (one per API response in the transcript, however many lines it spans), uncached input, cache-write, cache-read and output tokens (per model from the run's result, which also counts calls the transcript doesn't hold, or summed from the transcript when a timeout leaves no result), turns and cost, with the mode, whether it was a `/review continue`, the PR and the head commit. It goes in the run's job summary as a table and in a `review-usage-<pr>-<run id>-<attempt>` artifact kept for 30 days, never on the PR. Compare runs with `gh run download <run id> -n review-usage-<pr>-<run id>-<attempt>`.

## Threat model

`issue_comment` runs with secrets on PRs from forks, so the PR's contents and comments are untrusted input. A comment only selects the mode, by exact match in the job's `if`; its text never reaches a shell or the prompt. The workflow never executes PR code, checks the PR out only into `pr-head/`, reads its rules from the base tree, gives the model no shell and no mutation or network tools, redacts credential-shaped strings from everything it posts, and holds write permission on pull requests alone. The residual risk is prompt injection producing a misleading comment: annoying, not dangerous.

`/review gpu` adds one place PR code runs: the GPU box, and only for a PR whose author has write access (checked against their permission, not their association). The privilege split keeps the rest as above: the `live-test` job holds the Slurm and Eras keys and the app token but runs no model and no PR code; the box runs the model with a shell and the PR's code but holds none of those. The box's session reads skills from `main`'s tree and has the Skill tool off. Its brief is fixed text plus the commits and the review's `full`/`targeted` call; the review's model-written notes go beside it as `findings.json`, which the brief calls untrusted data, so text an attacker gets into the review reaches the session only as data, never as its instructions. What the box can reach is still wide, and that is the residual risk. Its Anthropic key is in the environment of every process there, the PR's code included, so it is a key of its own (`LIVE_TEST_ANTHROPIC_KEY`, spend-capped, never the review's key). On the Slurm node the session and the PR's code run as root on a shared machine: they can read and change anything on it, including other people's files, jobs and credentials there, and reach whatever the node's network reaches. That is why the live test runs only for authors with write access, who can already push code the team runs. The session gets a fresh home and Claude config so it loads nobody's settings or hooks, the key file is mode 0600 and removed once read, and the job's directory is removed when it ends. What the box writes is posted only after redaction of the credential patterns and the exact values of every secret the job holds. The approval is a fixed rule over the review's fields, the live test's verdict and the file list, so a misleading review can't approve a change to the review rules or dependencies, and what it says of an engine change doesn't count as an approval. A contributor's run on their own PR reads exactly the content a maintainer's `/review` of that PR would, with the same tools and token, so it adds cost exposure (bounded by the rules above and one run at a time per PR), not new capability. A saved handoff (for `/review continue`) is a public artifact holding the model's notes on the PR's content, redacted like everything else the bot posts; a continued run trusts only artifacts from this workflow's own `issue_comment` runs, checks the handoff's PR, head commit and mode, and gives the model its notes as untrusted input. Preserve the rules listed in the workflow header when editing it.
