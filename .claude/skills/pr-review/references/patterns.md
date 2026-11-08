# Factoring and style patterns

What mstar reviewers ask for again and again, mined from human review comments on past PRs. Much of
it is written down nowhere else. Section A is factoring and placement, B is style, C is everything
else. The factoring pass (SKILL.md §4.7) and style pass (§4.10) list every check; open a pattern here
for its examples when a finding needs one.

Cite a miss as "factoring pattern A3", "style pattern B1" or "pattern C1". Where AGENTS.md states
the rule, cite the invariant instead (it outranks a pattern). The recurring moves behind all of
them: model-specific code in a shared layer goes back to the model behind a hook with a default; a
copy of something that exists is replaced by the existing thing; two paths doing one job merge;
long comments get cut; flags no longer needed get deleted; a model working around the system is a
system gap.

## A. Factoring and placement

### A1. Reuse the abstraction that already exists
Partly AGENTS.md invariant 1 (resources, capture).
- A hand-made CUDA-graph warm-up in an audio encoder, where extending the existing piecewise runner
  was the ask (#150). Own Gumbel sampling where the `Sampler` resource gives the fused prep kernel
  and FlashInfer sampler (#249). Memoized `functools.partial` handles where `AttentionCallable` in
  `mstar/engine/resources/convenience.py` exists (#286).
- Also: follow conventions other models use (config overrides from the HF `config.json`); reuse a
  trick another model already uses; compose with an open PR solving the same problem (#257 and
  #258; #185 and #171); point out two open PRs fixing the same thing (#286 and #251/#291, #330 and
  #296).
- Look for: new code doing capture, sampling, attention planning or host/device buffering; new
  factories, wrappers or memoized callables; names that look like an existing helper. Grep
  `mstar/engine/`, `mstar/engine/resources/`, `mstar/model/components/`, `mstar/utils/`.

### A2. One path, not two
Partly AGENTS.md S4.
- Separate preprocess functions for captured and eager paths where only one part differs (#121).
  Two RNG offset counters for the eager and graph samplers where one row could serve both (#312).
  A near-copy of `agree_across_ranks` differing only in `amax` vs `amin` (#341).
- Also: drop a legacy "no partitions" path for one default partition (#26); one lockstep-instance
  interface instead of combining TP and SP on the fly (#143); loop over specs instead of repeated
  if/else (#216).
- Look for: `if captured: ... else: ...` with mostly identical branches; a function differing from
  an existing one by one operator or argument; two fields tracking one fact; an old path kept "for
  compatibility".

### A3. Reusable code goes in shared locations, not one model's package
add-mstar-model Phase 4 covers model ports.
- A copy of `ParallelGatedMLP` (#121); generalizable DSpark components under `kimi_k3/` (#257); a
  kernel not specific to fused MoE (#270); a whole Gemma text model under one diffusion model's
  `components/` when `mstar/model/components/diffusion/` already holds shared encoders (#375).
- Also: shared attention primitives (#150), a cross-attention module (#159), a conv3d dtype helper
  (#165), merging two models' code predictors (#184), a single-file safetensors case inside
  `load_weights_from_hf_shards` (#159), a generic EP config (#360).
- Look for: classes under `mstar/model/<name>/` that don't depend on that model (MLPs, attention,
  kernels, loaders, schedulers, quantization, speculative-decoding pieces). Ask "would another model
  want this?"

### A4. When a model works around the system, fix or extend the system
AGENTS.md invariant 1.
- #121 led to attention backends, per-request state, compile config and postprocess moving into
  the system. #297's persistent sessions made the case for session primitives. A model's
  single-request mixin was a system bug fixed at engine level (#249).
- Look for: comments like "workaround", "the engine doesn't…", "because X doesn't support";
  per-request dicts in a submodule; model-side locks; sampling or capture special cases.

### A5. A class hierarchy instead of growing if/else, unions, flags or globals
AGENTS.md S3, partly invariant 8.
- A growing union of FlashInfer wrapper types (#192); three all-reduce implementations in one class
  where an abstract communicator with subclasses fits (#257); `{"cuda", "xpu"}` checks spread
  through the code (#262).
- Also: a global override setter that should be a runner class (#150); transport-specific fields
  split into subclasses (#168); an explicit message type instead of an "empty" sentinel (#224); an
  enum method instead of `== "SHM"` (#156).
- Look for: unions growing per PR; `if device.type ==` / `if protocol ==` across files;
  module-level mutable state; a sentinel value of an existing message reused to mean something else.

### A6. Let the model or submodule declare it; don't string-match names or guess from shapes
- Graph-walk string matching that belongs in a submodule hook with a default (#43); hard-coded walk
  names where the submodule could declare chunkable walks (#81); `per_request_states is None` used
  to mean "synthetic" instead of an explicit flag (#360).
- Look for, in engine, worker and scheduler code: `graph_walk == "decode"`, `.startswith("prefill")`,
  picking a dimension by matching a size, `x is None` meaning warm-up, relying on padding rows
  having length zero. Suggest a submodule or config hook with a default.

### A7. Model-specific logic stays out of shared layers
Partly AGENTS.md invariants 1 and 4. Covers the worker, conductor, engine, data worker, API server
and shared adapters.
- A model-specific offload flag in the data worker that belongs on the model base class with a
  default (#185); mixed batches added as long model-specific code in `worker.py` and
  `micro_scheduler.py` (#185); every unknown chat role mapped to "user" in the shared adapter to
  suit one model (#340).
- Also: docstrings and names in shared code shouldn't name one model (#82, #185).
- Look for: model names, config names or one model's tensor names in shared code; a capture-order
  or scheduling heuristic justified by one model's OOM (#330).

### A8. Pull logic into a named helper, or a method on the class that owns the data
- Shared logic between speculative and non-speculative paths (#224); a block better as `__eq__` on
  its dataclass (#185); hoisted `columns.*.append` calls better as `columns.add_row(...)` (#368).
- Also: an abort that also does the cleanup instead of callers calling both (#136); a setter
  instead of reaching into `cache_manager._plan_states` (#82).
- Look for: long inline blocks in big loops (`worker.py`); callers that must call two things in
  order; code touching another class's private fields; multi-field comparisons inline.

### A9. Files go where the repo expects them
- Test-only classes in the tests, not under `mstar/` (#121); test modules where CI selects them
  (AGENTS.md "Testing"); benchmarks in top-level `benchmark/`, with a client-only script (#121,
  #251); experiment configs apart from production configs (#192); repeated string constants in a
  `constants.py` (#121); kernel families in their own subdirectory (#257); constants a subclass may
  override on the class (#286).
- Look for: new files under `mstar/` only tests or benchmarks import; experiment configs mixed with
  production ones; string constants repeated across files.

### A10. Shared code in the base class; optional hooks declared there, not probed
- Logic common to all tensor communication managers on the base class (#48); a base-class typo
  fixed in the base, not overridden in a subclass (#121); an empty base method instead of `getattr`
  (#336).
- Look for: `hasattr(submodule, ...)` or `getattr(x, "...", None)` on project classes; a method
  copied into two siblings; an override that only fixes a base bug; a base method returning a dummy
  value where it should be abstract.

## B. Style

Usually P3 with "Nit:", unless the issue hides a bug.

### B1. Names are accurate and general
Ruff does not check this. Report a name only when it is wrong, not for taste: shared code named
after one model, protocol or parallelism mode (`mrope_pos_advance` → `custom_pos_advance`, #82;
`local_uuids` rather than a protocol-specific name, #156); a clash with an existing component (a
"conductor" that isn't the conductor, #169; `stats` vs `_arena.stats`, #168); a name left after its
meaning changed (`tp_groups` once SP joined, #143); a hook named for less than it does (#286).

### B2. Delete dead code
Partly AGENTS.md invariant 9.
- References to a removed engine (#89); unused injection machinery (#52); a check that is always
  true in production and false only in tests (#262); a defensive XPU check for a model that can't
  run on XPU (#262).
- For each removed symbol and new enum member, grep for leftover or missing users. A new or touched
  function with no callers is a delete finding, not a note: on #816 an uncalled `slot_bytes` was
  noted, then broken by a later fix.

### B3. Cut comments and docstrings down
AGENTS.md S2. Comment blocks over about three lines mid-code, module docstrings longer than a
paragraph, comments that tell a debugging story or cite PR or plan-step numbers (#86, #159).

### B4. Simplify
Configs built from `fields()` instead of listing fields one by one (#159, #251, #257); merged
conditions (#179); values built once at load instead of per request (#161, #165); initialization in
`__init__` instead of `__dict__.setdefault` (#286); no arguments that only switch a function off
(#224).

### B5. Make assumptions explicit; fail loudly and early
`in str(e)` matching that breaks when a message is reworded (#38); a fallback that isn't
capture-compatible, which should refuse when capture is on (#257); failing at launch rather than per
request (#169); asserting hard-coded token counts and alignment sizes (#184, #377); logging a silent
clamp (#314); code that works only because no current config uses disaggregation (#380).

### B6. Name aggregates
AGENTS.md S1. NamedTuple, dataclass or Enum instead of three-plus-part tuples, fixed-key dicts and
`COLUMN = "column"` constants (#161, #216, #257); new data gets its own field rather than riding in a
generic `metadata` dict (#185); YAML specs parsed into a dataclass (#335).

### B7. Delete A/B and debug flags once decided; no bug fix behind a flag
AGENTS.md invariant 9. A debug flag for a compile that always helps (#121); a fix behind an env var
(#150); a knob the PR's own A/B found neutral (#368); `debug_*: true` in a production YAML (#360);
make a behaviour the default rather than adding an env var for it (#380).

### B8. Explain why, and write down fragile contracts
AGENTS.md S2. Surprising code with no comment (padding to a maximum length, #165); a contract that
could foreseeably break, with a defensive check (#224).

### B9. Config and CLI over hard-coded values and per-model env vars
Partly AGENTS.md invariants 2 and 9. A hard-coded `flashinfer_backend="fa2"` that forfeits FA3 on
Hopper for everyone (#184); a checkpoint path in YAML instead of the serve command (#192); an env
var per model where one global knob covers them (#368); `.to("cuda")` in preprocessing (#150); a
binary env var standing in for a choice among several (#156); personal paths as defaults (#251).

### B10. Update comments, docs and TODOs when the code changes
AGENTS.md S2. Stale comments after async scheduling or a transport change (#121, #168, #270); a
leftover "add TP support" TODO (#250). A comment, docstring or AGENTS.md line that gives a reason or
states an invariant is checked against the code (#822's frames commented as serving vision adapters
that no leased row can run; #705's docstrings describing the old one-pass `new_group`).

### B11. Justify, derive or parametrize magic numbers
Can a fixed limit be too small and let the bug return (#73)? Make a constant an argument with a
default (#165); derive a recompile limit instead of fixing it to today's configs (#178).

### B12. Document every new env var
AGENTS.md invariant 9: a row in `docs/environment_variables.rst` (#164, #185).

### B13. Type-annotate parameters model authors implement against (#95). Minor.

## C. Other recurring asks

- **C1. Error paths.** The right 400 vs 500, cleanup in `finally`, the error reaching the client. A
  loop that exits before `synchronize` and `commit_done.set()` (#275); a hard-coded "server_error"
  where `_fail_request` maps `ValueError` to 400 (#296); swallowed errors (`.ok()`,
  `unwrap_or_default`) and a failed stream ending with `finish_reason: stop` (#169); missing
  `abort_step` on error paths (#275). Also security: path traversal through upload filenames and
  arbitrary delete in a frontend (#169). Look at new `except` blocks, early returns inside loops
  holding events or locks, hard-coded error types, and each route's exception-to-status mapping.
- **C2. Run it.** Reviewers ran PRs on GPUs, sent malformed requests at both frontends, ran soak
  tests and A/Bs with fresh restarts, and found import errors (#150), concurrency-2 failures (#249)
  and version breaks (#341). Without a GPU, write the check that would show it.
- **C3. Dependencies.** Port or pin upstream internals (#150); test each torch major in the allowed
  range (#341); new data files in `package-data` (#377); a fallback when an optional kernel is
  missing.
- **C4. Performance changes need A/B numbers per mechanism** (AGENTS.md invariants 9, 10; #185,
  #368).
- **C5. Register and document new models:** `docs/models.rst`, `mstar/cli/main.py`, a model class in
  `benchmark/base.py`, the `all` extra (#184, #249).
- **C6. Keep `forward` capturable** (AGENTS.md invariant 3): data-dependent control flow captures one
  branch (#31); keep `preprocess` a pure function of its inputs (#184).
- **C7. Tests that run and test something.** A default that makes the test vacuous (#169); a
  `skipif` that makes it a no-op in CI (#347); a module outside CI's selection (#358).
- **C8. Frontend parity.** The Python API server and the Rust frontend answer alike (#169: a client
  expecting MP3 could get WAV back).
- **C9. Match the reference's pre- and post-processing** (#249: a raw string where training put a
  language id).
- **C10. Deferred work is done in the PR, not filed for later.** A TODO or a follow-up issue doesn't
  resolve a finding (#165).
- **C11. TP/lockstep symmetry** (AGENTS.md invariant 11; #358: per-rank expert selection that needs
  a broadcast).
- **C12. Split big PRs and unrelated sweeps** (#185, #262). A note, per AGENTS.md.
- **C13. Thread safety.** A check-then-create race fixed by a cheap lock (#168); GIL-holding calls on
  the hot path (AGENTS.md invariant 7).
- **C14. A change for one model must not change defaults for others** (AGENTS.md invariant 4; #262).
- **C15. No host syncs in the step path** (AGENTS.md invariant 7; #249: a dynamic loop index on the
  host instead of `.item()`).
- **C16. Port natively instead of wrapping upstream** (#150).
- **Benchmark honesty.** Dropping EOS undercounts `ignore_eos` throughput (#340); default configs
  should match the concurrency behind the numbers (#251).
