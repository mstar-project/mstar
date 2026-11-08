# The mstar philosophy

mstar's design principles, each with the questions to ask of a diff. AGENTS.md states the rules
(invariants 1–14, style rules S1–S4) and [patterns.md](patterns.md) the concrete review checks
(A1–A10, B1–B13, C1–C16). This file is why those exist, plus the values neither writes down. The
philosophy pass (SKILL.md §4.10) applies it. Cite a miss as "mstar philosophy §N".

## 1. Models declare, the engine executes, and shared layers know nothing about any one model

Model code says what it needs; the engine, worker, conductor, data worker and API server provide it
the same way for every model. Shared code must not name a model, compare graph-walk names
(`== "decode"`), guess a dimension from a tensor's size, or check a resource's concrete class. When
shared code needs model-specific behaviour, it calls a base-class hook with a default. The engine
is where optimizations are written once for every model: each model-specific branch in shared code
is where the next model breaks, and each engine mechanism a model reimplements opts that model out
of batching, capture and backpressure.

Maps to: invariants 1, 2, 3, 8; S3; A6, A7, A10, B1.

- Does a file under `mstar/engine/`, `mstar/worker/`, `mstar/conductor/` or `mstar/api_server/`
  (including the shared OpenAI adapters) gain a model name, one model's tensor name, a graph-walk
  string comparison, or an `isinstance` check on a concrete resource?
- Does model code do engine work: capture graphs, allocate or evict state, choose batches, create
  streams and syncs, or run a data-dependent loop and its stop decision inside `forward`?
- When shared code needs a per-model answer, is there a base-class method with a default, or
  `hasattr`/`getattr` probing?
- Is a hook used to wire a cross-cutting feature into every model, when the engine could find the
  same answer from the module tree or a declaration? Hooks are for answers that really differ per
  model.

## 2. When a model has to work around the system, it has found a system gap

The fix belongs in the engine or a resource so every model gets it, as its own PR stacked under
this one.
Signs: comments saying "workaround" or "the engine doesn't…", per-request dicts in a submodule,
locks in model code, hand-written sampling or capture. A per-model copy becomes dead weight once the
system feature lands, and it hides the gap from the next model. Things "only this model needs"
(sessions, cross-attention, recurrent state, samplers) have repeatedly turned out to be needed by
others.

Maps to: invariant 1; AGENTS.md "Splitting work across PRs"; A1, A3, A4.

- Does model code reimplement something under `mstar/engine/resources/`,
  `mstar/engine/accelerator_graph_runner.py` or `mstar/model/components/`? (#249: a model's own
  sampling; #150: an audio encoder with a hand-made CUDA-graph warm-up.)
- Is a class that doesn't depend on the model (an MLP, a kernel, a scheduler, a quantization
  helper, speculative-decoding pieces) filed under one model's package? (#257: DSpark kernels in
  `kimi_k3/`.)
- If the system fix is deferred to an issue or a later PR, that is the finding: ask for it in this PR
  or in a PR stacked under it.

## 3. Add an abstraction once the second case is real, and make it a type, not a branch

Add a base class or capability object once a second variant exists or is clearly on the way (a
second hardware backend, a third all-reduce, a second quantization format). Avoid unions, flags and
`if device == ...` checks that grow with each PR. Before the trigger, direct code is fine.

Maps to: S3, S4; invariant 8; A5, B6.

- Does the diff add a second or third arm to an `if`/`elif` on device type, transport protocol or
  quantization type, or grow a `Union[...]` of wrapper types?
- Does a new optional argument change what an existing class means? S3 says make a sibling class.
- Does a new abstraction have a real second user?

## 4. Fewer moving parts: one path, one source of truth, and delete the rest

Two code paths or two counters for the same fact drift apart. Keep one path with a branch only for
the real difference, one owner per counter, and validation where an object is built so a `None`
doesn't flow downstream. Backward compatibility inside the repo is not a value: all callers are
in-tree, so old paths, unused functions, enum members, files and always-true checks are removed.

Maps to: S4; invariant 9; A2, A10, B2, B10.

- Are there two mostly identical branches (captured and eager, TP and non-TP, single and batched)?
- Does the PR add a second field, counter or cache for something already tracked?
- Grep each removed or renamed symbol for leftover callers, and each new enum member, env var or
  function for a reader. Are stale comments or docs left? (#243: `MSTAR_NUM_SLOTS` left as dead
  code.)

## 5. The correctness bar: silent wrong output is the worst failure

A crash or a 400 is acceptable; garbage audio, a wrong image or a hang is not. Most mistakes here
don't raise: a misaligned buffer inside a captured graph or a padding row writing into another
request's KV pages silently corrupts someone else's output, and a rank on a different branch hangs
the others. Assert hidden assumptions, refuse unsupported configs at boot, make configs do exactly
what they say. A bug in shared code is a bug even when no current model triggers it (#198).

Maps to: invariant 1 heuristic 3; invariant 11; B5, B8, B11, C11. (AGENTS.md's "enforce at
startup" design pattern explains this but is not citable.)

- What happens on an unsupported input or config? "It runs and returns something" is a finding.
  Look for silent fallbacks, bare `except`, `.get(key, default)` on required keys, clamping without
  a log line, `in str(e)` matching.
- Does the code work only by a coincidence in today's configs (two sizes equal, one caller, an even
  dimension)? Ask for an assert or a startup check.
- In engine and resource code, is the bad case only "unreachable with current models"? Still a
  finding, at lower priority.

## 6. A change for one model must not cost any other model

Global defaults, process-wide settings and shared contracts don't move to suit one model. If one has
to move, the PR shows the other models still work and run as fast (BAGEL, Orpheus, Qwen3-Omni,
Whisper are representative).

Maps to: invariants 4, 10; C14.

- Does the diff change a default, a process-wide setting (dynamo or inductor config, the allocator,
  an env-var default), a timeout or a base-class contract? Who else reads it?
- Is the PR framed as being about one model while it edits shared files? (#103: a shared-contract
  change framed as model-only.)
- Does the PR say which other models and workloads were checked?

## 7. Complexity must pay for itself in measured performance

Every optimization, cache, flag or second path stays only if an A/B shows it helps. Remove
mechanisms that measure neutral, even working ones. Host cost counts as much as GPU cost: a decode
step can be about 2 ms of GPU work, so per-step Python, object creation and host syncs show up in
throughput, measured with `benchmark/worker_phases/`.

Maps to: invariants 6, 7, 9, 10; B7, C4, C15.

- For each new mechanism in a performance PR, is there a number for it alone? If the body says it is
  neutral, it should be removed, not left behind a switch.
- Does the step path gain per-request Python, `.item()`/`.tolist()`/`.cpu()`, a device tensor built
  from host data, a `deepcopy`, or a per-step allocation?
- Does a small win come with hard-to-reason-about concurrency (#368)?

## 8. Evidence over plausibility

A description's numbers and explanations are hypotheses until reproduced. Benchmarks compare like
with like; unfair comparisons are bugs (one camera against two, `n=4` images counted as one, EOS
tokens undercounted, default configs that don't match the concurrency behind the numbers). An
agent-written explanation is distrusted until checked against the code.

Maps to: AGENTS.md preamble; invariant 10; AGENTS.md "Testing"; C2.

- For each claim (a speedup, parity, "fixes X"), is there a command and hardware? Is the mechanism
  shown (a profile, a test that fails without the fix), or labelled a hypothesis?
- Do the compared systems do the same work: inputs, `n`, output length, EOS handling, defaults?
- Do the shipped configs match the configuration behind the numbers (#251)?

## 9. Correct means "matches the reference", and every output change is deliberate and stated

Output is judged against the model's reference implementation (transformers, diffusers or the
official repo) and against `main`. Preprocessing must match training (language ids, audio
normalization, chat templates): getting it wrong gives fluent but wrong output. A PR that moves
outputs on purpose says so, with a parity test; a deliberate deviation from the reference has a
comment saying why.

Maps to: AGENTS.md "Testing"; add-mstar-model; C9.

- Does the PR change default sampling, padding, normalization, RNG use or prompt templates? Is that
  in the body, with before and after?
- For a port, are the reference's preprocessing steps ported? Diff against the upstream source.
- Is there a parity test against an independent reference, not the PR's own earlier code?

## 10. Own the model code, and depend on upstream deliberately

Port natively onto mstar's resources and capture instead of wrapping upstream packages; a wrapped
model can't use the engine's attention, batching or capture and breaks when upstream refactors.
Dependencies are pinned, declared, shipped and tested across the allowed range.

Maps to: add-mstar-model "Port natively by default"; AGENTS.md "Don't add a dependency without
saying why"; invariant 9; C3, C16.

- Does a model call transformers, diffusers or an upstream package at runtime for its main compute,
  or import private upstream functions?
- For a new dependency: right `pyproject.toml` extra and in `all`; a floor or pin; in
  `docs/installation.rst`; the body says why.
- Are new non-Python files listed in `package-data` (#377)? Is any API used that only exists in a
  torch newer than the floor?

## 11. Deployment facts go in config, and flags are temporary

Anything that depends on hardware or deployment (kernel backend, device, paths, memory sizes,
timeouts, per-step budgets) belongs in the YAML or CLI, not model code. Knobs are global and few
(one `MSTAR_DECODE_BUCKETS`, not one env var per model). A/B and debug flags go once decided, bug
fixes are never behind a flag, and a flag that stays is documented as a general tool.

Maps to: invariants 2, 8, 9; B7, B9, B12.

- For each new `MSTAR_*` env var: a row in `docs/environment_variables.rst`? Per-model (generalize)
  or an A/B leftover (delete)? Does it gate a bug fix (ungate it)?
- A hard-coded backend, device, path or size in model code (`.to("cuda")`, `"fa2"`, `/mnt/<user>`,
  a checkpoint path in YAML)?
- Does code silently override a value the config set explicitly (#220)?

## 12. One bad request fails alone, with the right error, and leaves nothing behind

The client gets a 400 for bad input and a 500 for a server fault; other requests in the batch keep
running; every way a request ends (done, failed, aborted, disconnected) releases what it held. An
out-of-range index or over-long clip that reaches the GPU becomes a device-side assert that kills
the worker.

Maps to: invariants 11, 14; add-mstar-model Phase 5; C1, C13.

- For each new `except` or early return in worker, engine or API code: does it fail just that
  request, release its state, and keep TP ranks symmetric?
- For each new request field: is its type and range validated at the API boundary with a 400?
- Is every new per-request dict or set cleared on the completed, failed, aborted and disconnected
  paths?

## 13. One reviewable change per PR, and nothing deferred

A reviewer has to be able to run every change. System changes a model needs land underneath it,
stacked; unrelated sweeps and optional features go in their own PRs. Work a finding needs is done
here, not filed as an issue for later. Split requests are notes (AGENTS.md "Splitting work across PRs").

Maps to: AGENTS.md "Splitting work across PRs", "Keep PRs focused"; C10, C12.

- Does a model PR also change `mstar/engine/`, `mstar/worker/` or `mstar/conductor/`?
- Is an unrelated sweep or an optional feature riding along (#185's FP8 experts)?
- Does the PR defer anything (a TODO, a "follow-up", an issue)? Each deferral that a finding needs
  is itself the finding.

## 14. A test counts only if something runs it and it can fail

New tests sit where something runs them, exercise the real path, use an independent oracle, and fail
if the code is wrong. CPU tests go under `CPU_TEST_PATHS`, or in `NOT_CPU_CORE` with where they run;
GPU-only tests go in a path the GPU live test runs and skip cleanly on CPU. A test nothing runs is the
finding. Tests whose setup makes the assertion vacuous, tests comparing code with a copy of itself,
and tests pinning incidental details (literal YAML, log text) give false confidence.

Maps to: AGENTS.md "Testing" (`CPU_TEST_PATHS` and `NOT_CPU_CORE` in
`scripts/ci/select_cpu_tests.py`); C7.

- Do new CPU-runnable modules sit under a `CPU_TEST_PATHS` directory, or in `NOT_CPU_CORE` with
  where they run? Are GPU, weight and optional-package tests guarded to skip cleanly on CPU?
- Would each new test fail if the fix were reverted? Is the oracle independent?
- Does a test assert incidental details instead of the behaviour that matters?

## 15. Write for the next model author

A hook a model must implement isn't finished until `docs/adding_models.rst` says what it returns
and when. Comments are short, explain the non-obvious, and are written for someone who didn't see
the debugging session. Names describe the general concept (`custom_pos_advance`, not
`mrope_pos_advance`). Aggregates are typed.

Maps to: S1, S2; invariant 9; B1, B3, B6, B8, B10, B13, C5.

- Is every new or changed hook on `Model`, `NodeSubmodule` or a resource documented in
  `docs/adding_models.rst`, with its return contract?
- Comment blocks over about three lines that tell a debugging story or cite plan steps or PR
  numbers? Surprising code with no "why"?
- Shared code using a name that mentions a model, protocol or parallelism mode, or an old name whose
  meaning changed?

## Tensions

When a PR sits on one side of a tension, say which side it took, whether that matches the
resolution below, and ask why if the PR doesn't say.

- **Generality vs. shipping the model now.** Push hard on generality, especially in early
  integrations of a kind of model, which set the pattern. A per-model version of a system
  feature is accepted only under invariant 1's caveat (a long-running system effort already
  underway, named in the description); otherwise the system fix belongs in this PR or one stacked
  under it. It is a
  finding when the PR also adds model-specific code to shared layers. When flexibility across model
  architectures competes with flexibility across backends, architectures win.
- **A fast path for one model vs. a model-agnostic design.** Welcome the optimization, but as an
  opt-in helper or declared policy the engine provides, not in one model's code (invariant 1's
  `keep_final_chunk_samples` example).
- **Determinism and parity vs. speed.** Parity with the reference is required; bit-exact
  determinism across batch compositions is not bought at a large cost (20% slower decode is not
  worth it). An output change is a finding if `main` doesn't show it, not if it is within the noise
  `main` already has.
- **Fewer knobs vs. A/B and debugging.** A global, documented, general-purpose switch (one "disable
  torch.compile" flag for all models) is acceptable; a per-model or per-PR switch is not.
