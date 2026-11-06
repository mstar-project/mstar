# Brief for <pod>: optimize <model> serving in M*

Fill in every `<...>`. The loop sends this file as the first message of a conversation, so it addresses the agent.

You are running an optimization loop on M*'s <model> serving, on this pod's <gpus>, for several days.
Nobody will answer questions.

### 0. Environment and verify-first, before any edit

- Your worktree is `/workspace/wt-<pod>` on branch `perf/<pod>`, created from `run-base`. Confirm
  `git rev-parse run-base` is `<sha>`. Name any other branch you make `perf/<pod>-<name>`.
- `source /workspace/venv/bin/activate`. Its editable install points `mstar` at your worktree. Confirm `nvidia-smi`
  shows the GPUs you expect and no other process. Every session on this pod shares its GPUs, so run every GPU job
  under `flock /workspace/gpu.lock <command>`.
- Read `AGENTS.md`, `.claude/skills/benchmarking/SKILL.md` and every file your section names. Every claim you act
  on carries a path:line at `<sha>`, read from the checkout. An unverified item becomes a check that runs first.
- If `/workspace/results/<pod>.md` exists, resume from it instead of starting over.

### 1. Target

- **Models and configs:** <model sizes, `configs/<name>.yaml`, the model package under `mstar/model/`>.
- **Headline cell:** <one model, workload, concurrency and metric, and why it is the one to move>.
- **Workloads:** <workloads and concurrencies, with the `benchmark/runner.py` options for each>. Rows use model
  `<model>`, unit `<tok/s>` and latency `<TTFT>`.
- **Correctness check:** <for example, teacher-forced top-1 agreement with HF transformers on 64 fixed prompts is at
  least HF's agreement with itself, sdpa against eager attention>.

### 2. Where changes may go

- <The paths this loop may change, and what happens to a gain that needs code outside them, for example a
  proposal in `/workspace/results/<pod>-proposals.md` with the time it would remove, measured.>
- Before every commit, `git diff --name-only run-base...HEAD` lists only allowed paths.

### 3. Rules a change must pass

These are the patterns M* reviews send back:
- Per-request state lives in a declared resource, never in a dict, tensor or cache in model code.
- Nothing counted or tuned by hand: page counts, pool sizes and capacities come from what the model declares.
  Changing a config number is a deployment choice, not a win.
- No hook every model must implement, and no model name in shared code.
- No silent path. An unsupported case refuses at load with a message that names the requirement.
- No second copy of a cache, manager, helper or kernel that main already has. Extend the existing one.
- `check_stop` is the only host read in the step path. Add no host sync.
- A change deletes what it replaces, and a path that measures neutral is deleted, not left behind a flag.

### 4. One round

1. **Hypothesis:** one sentence naming the mechanism and the expected ratio on the headline cell. Profile first and
   point at the time the change removes.
2. **Implement** it as one concept and commit it.
3. **Check correctness** as section 1 says. A failure ends the round.
4. **Measure** the base (the current best commit, `run-base` at first) and the candidate (the round's commit): 3
   rounds each in alternating order, a fresh server per round, one untimed warm pass per server, the same workload
   and settings. Run the base from its own worktree (`git -C /workspace/mstar worktree add --detach /workspace/base
   <base sha>`) and start its server with `python -m mstar.cli.main serve ...` from that worktree's root, so its code
   is the code imported. When the two commits differ under `rust/`, the base also needs its own venv, since M*
   refuses an extension built from other sources.
5. **Decide** with `vibesys keep /workspace/results.jsonl --hypothesis '<the hypothesis>' --model ... --workload ...
   --c ...`.
6. **Review:** a subagent that has not seen your reasoning reads the diff against sections 2 and 3 and says, for
   each added piece, whether it breaks a rule.
7. **Keep or drop.** When `keep` says kept and the review passes, the commit stays and is the next round's base.
   Otherwise `git reset --hard` to the base and record why.

### 5. Reporting

Report every measured round with `mstar-report '<json>'`, one row per round and arm. It refuses a row that doesn't
fit the schema. Keys: `batch, family, model, workload, gpus, engine, role, config, c, round, throughput, unit,
latency, p50_ms, p99_ms, requests, failed, status, started, error, commit, hypothesis`. `started` carries its UTC
offset (`2026-11-03T14:05-05:00`).
- **Loop rounds:** batch `<pod>-loop`, status `probe`, engine `M* <pod>`, role `baseline` or `candidate`, `commit` the
  short SHA measured, and the round's hypothesis word for word on every row.
- **Checkpoint sweep,** every 8 hours or after 3 kept rounds, whichever comes first: every workload and concurrency
  in section 1, `run-base` against your best commit, 3 rounds alternating. Batch `<gpu>_<MMDD>`, status `counted`,
  engine `M* run-base` with role `baseline` and `M* <pod>` with role `candidate`, config the config file name, and
  hypothesis `checkpoint: run-base vs <sha>`. `vibesys table /workspace/results.jsonl --batch ... --hypothesis ...`
  prints the sweep. Run the full `test/modular` suite at each checkpoint too.
- **Results file** `/workspace/results/<pod>.md`, appended after every round: the commit, the hypothesis, the exact
  commands, the `keep` output, the correctness result, the review and the decision. Keep the hypothesis queue at
  the top, ordered by expected gain. No narrative.

### 6. Commits and tests

- Local branches only. Never push, open a pull request, or force-move a branch other than resetting a dropped round.
- One concept per commit, and a test commit only after the code it tests. Subject `scope: lowercase summary`. A
  body only when the why isn't in the diff. No trailers.
- After every commit: `ruff check .` and the targeted tests, meaning your test files and every `test/modular` file
  that imports a module you touched. The full `pytest test/modular -q -n 8 --dist loadfile` runs at the start, to
  record the baseline, and at each checkpoint. A failure that also reproduces at `run-base` is pre-existing.
- Never commit results, notes, plans, proposals or `CLAUDE*.md`.

### 7. Keep going

Run rounds continuously until the loop stops you. When blocked (missing weights, a crash outside your paths, a gain
that needs code you may not change), record it and take the next hypothesis.
