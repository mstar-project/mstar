# Hard rules

The hard-rules pass (SKILL.md §4.2) checks every PR against R1–R12, the premise checks and the
tests-and-claims checks below. Cite a broken rule as "hard rule R<n>". Rank it by the bar in
SKILL.md §2; the tier given here is the floor.

## Rules that reject the approach

Breaking one means the approach has to be replaced, or the PR closed. Usually P1.

- **R1. The bug exists on current main**, and nothing merged or open already fixes it.
- **R2. Decisions on record stand.** Reversing a call made in an issue, RFC, review thread or code
  comment needs new evidence that answers the reasons given there. Cite where it was made (#449
  reopened #348's trade-off without answering why #348 closed).
- **R3. One mechanism per concern.** Extend the mechanism `main` already has; a second one beside it
  needs the PR to show why the first can't take the change (#465's resend queue beside the drain
  record's TTL; #729's `unadmittable_buckets` beside `_blocked_buckets`).
- **R4. Structural, not tuned.** Correctness, liveness and scheduling never rest on a wall-clock
  window, timeout, retry count or tuned constant when a structural signal exists: a step count, the
  pool's state, a reservation, message order, an explicit marker (#461's one-second windows became
  a step count). A model's quality default is exempt, but its comment states the rule that picks
  the value.
- **R5. Fix the cause.** No stopgap and nothing that hides a failure: a broad `except` over
  half-updated state, a leak bounded by a budget, a retry for a failure the system never produces, a
  WARNING where `main` logged an ERROR, an exemption from a check.
- **R6. Lockstep.** No control flow from per-rank state unless one rank decides and the others
  replay it (AGENTS.md invariant 11).

## Rules fixed before merge

Usually P2; P1 when the description claims the opposite.

- **R7. No regression.** No model gets slower, host time included. New per-step or per-message work
  in a core path needs a measurement against `main`. Boot stays within 5% of `main`. A step that now
  runs eagerly, or with a memory peak the capture plan never measured, is a regression.
- **R8. One runtime, two frontends.** A change to graph-runtime behaviour, the wire format or edge
  semantics lands in `rust/` with a `test/rust` case (invariant 5). The Python API server and the
  Rust frontend (`mstar-server`, `--rust-frontend`) answer the same request the same way: status,
  error text, output format.
- **R9. Handle-keyed state is purged on every release path** and listed in
  `mstar/worker/rid_table.py` (invariant 14).
- **R10. One concern per PR.** Separable changes bundled into core files could be split. This is a
  note, never above P3 and never blocking (AGENTS.md "Splitting work across PRs"); raise it only
  when the split changes what lands.
- **R11. The description matches the code**, including comments and docstrings that state a premise
  ("no step of this shape can be served"). Every sentence the author wrote is a claim: description,
  commit messages, comments, "N tests fail on main" counts.
- **R12. Nothing beyond what is needed.** Every added line changes behaviour or explains a hazard,
  and the change deletes what it replaces. No guard for a case no caller reaches, no new env var
  where the code around uses config or CLI, comments at the file's density.

## Premise checks

- **"Nothing can reach this."** Check padding (a step pads into the smallest bucket that holds it,
  so a bucket the pool can't admit at full size still serves smaller steps, #729), every model on
  `main` rather than one (#561), and the transport (the control mesh is PUSH sockets that block at
  the high-water mark and never raise for a dead peer, #465).
- **Main moved.** Re-derive each finding against today's `main`. A finding that comes only from a
  rebase is a post-rebase ask, not a design flaw (#471).
- **Pre-existing or made reachable.** A gap is pre-existing only if `main` reaches it today. If
  `main` fails earlier and the PR removes that failure, the gap is the PR's (#729).
- **The decision history.** Read the linked issues, the other reviewers' asks and the author's
  replies before writing anything.
- **One mechanism across PRs.** Two open PRs that count the same thing become one count, folded by
  whichever lands second (#459 and #551).
- **The structural direction.** A finding that asks for a change names the smallest structural fix,
  starting from a mechanism `main` already has.

## Tests and claims

- **The oracle.** What would each new test accept that is wrong? A test whose expected value comes
  from the code under test guards nothing.
- **Would it fail on main?** Trace each new test against `main`'s code. Count apart the ones that
  fail for the bug and the ones that fail only on a name the PR adds.
- **Proportion.** Say which tests model a failure the system never produces (#465: about half the
  tests modelled a dead socket as `send` raising, which neither transport does). Don't report how
  many tests a PR adds.
- **Performance as a derivation.** Derive the work and bytes the change must cost per step and per
  request, and set that beside what the PR measured.
