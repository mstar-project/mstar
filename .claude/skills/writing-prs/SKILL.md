---
name: writing-prs
description: Preparing an M* pull request for review — splitting the work into a stack, what the description must contain (tests run and their results, evidence for performance and correctness claims), and how to write it so a reviewer can read it. Use when opening or updating a PR, writing or revising a PR description, or splitting a branch into stacked PRs.
---

# Writing a PR for review

A PR description is for a reviewer who has not seen the branch's history, the plan, or the debugging that produced it. It describes the final code and why it is that way. Read "Splitting work across PRs" and the style rules in [AGENTS.md](../../../AGENTS.md) alongside this; the automated reviewer in `.github/review/` cites AGENTS.md by number.

## Before writing

- **Split first.** System changes the model needed (a new resource kind, anything under `mstar/engine/`, `mstar/worker/`, `mstar/conductor/`) go in their own PR underneath; a capture/optimization layer can stack on an eager MVP when that makes each diff easier to review. Each PR in a stack should build, pass its tests and make sense on its own, and its description says what it is stacked on.
- **Rebase onto current `main`** and rerun the tests on each layer, not only the top. A conflict resolution that kept both sides reads as a plausible merge; the tests are what catch it.
- **Run what the description will claim.** The PR must state which tests were run and what they returned, so run them on the commit being pushed.

## What goes in the description

- **What changed and why, organized by concern**, not by commit. For a model: the graph and nodes, resources, correctness, performance, API, other changes, tests, follow-ups.
- **Tests: the commands and their results** (`ruff check .`, `python -m pytest @test/cpu-core.txt` with the pass/skip counts, the new test files and what they cover, GPU and live checks).
- **Evidence for every correctness or performance claim**: the harness, the hardware, and the number (invariant 10). Label an unverified mechanism as a hypothesis. For parity, give the yardstick (the reference against itself) next to the result.
- **What the PR does not do**, as follow-ups, so a reviewer doesn't ask.
- **New dependencies and environment variables**, with the reason (AGENTS.md "Also", invariant 9).

## What stays out

- **Fixes to the PR's own intermediate states.** A bug introduced and fixed while developing the branch isn't news to a reviewer of the final diff. It belongs in the description only if it changes something that exists on `main`, or if the lesson generalizes beyond this PR (a buffer-zeroing rule every capture author will hit, say).
- **The development environment.** Slurm node names, scheduler details, local toolchain workarounds, a box's load average. If a condition affects how a number should be read, state it in a reviewer's terms ("the node's CPU was shared with other jobs") and only then. Though, **do** state the hardware the PR was run on (e.g., 2 x H100, 4 x B200, 6 x XPU, etc.)
- **Debugging narratives.** State the rationale, not the story: "request facts live under `kwargs["request"]` because the conductor merges step metadata into kwargs", not the sequence of failed runs that revealed it (AGENTS.md S2 applies to descriptions as much as to comments).

## Formatting

- **One line per paragraph or list item; no hard wrapping.** GitHub renders a single newline in a PR body as a line break, so text wrapped at a fixed width looks ragged on any narrower screen.
- Tables for results; numbers with units; a code-path reference (`path:line`) where a mechanism is claimed.
- Keep it as short as the change allows. A reviewer reads the description before the diff, and one that restates the diff line by line is skipped.

## Updating a PR

When the code changes under review, update the description to match the final state rather than appending a changelog of what moved; review threads already record the history. Recheck every number and test count the description quotes.
