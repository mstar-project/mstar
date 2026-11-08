# Behaviour lenses

The behaviour pass (SKILL.md §4.1) applies these eight lenses to the whole diff. Their one job
is concrete defects: what the PR changes that someone could observe, and the cases nobody walked.
Design, factoring and style belong to the other passes.

Most defects that reviews of past PRs missed, and that a reader of the code could have seen, fell
into one of these lenses. Give them real time; don't skim them to get to the design passes.

Rate each finding by the bar in SKILL.md §2. Try to disprove it in the code first, and label it
**verified** (traced end to end) or **suspected** (name the inferred step). For each lens, write
the coverage line shown, so the review shows the lens ran.

## 1. Split every input into its cases, and walk each case

For each function, check, flag or hook the PR adds or changes, list its input classes and trace
every one through the new code:

- **Sets:** empty, one, *some*, all. A check that handles "all dropped" and "none dropped" but not
  "some dropped" is the usual miss (a stale pre-plan reused for a batch that lost some rows).
- **Sizes and budgets:** 0, 1, the minimum, the maximum, one past it. For sizes that must relate
  (a span and T, a count and a capacity, an index and a table's length), check what happens when
  the relation breaks.
- **Request shapes:** text only, media only, empty, both; a single-item and a multi-item form of
  the same argument.
- **Modes:** every TP degree and bs=1 (raw pointers reach kernels there that a `.contiguous()` copy
  hides at bs>1); eager, capture, replay and padding rows; CPU and GPU; colocated, data-parallel
  and disaggregated; async speculation on and off; TP async scheduling on and off; prefix cache on
  and off; preemption and rebuild; CPU offload; streaming; and the feature's own off or null config,
  which must leave `main`'s path and cost unchanged (#365's null config still measured every row and
  barred speculation).
- **Every path that builds or changes a checked object.** For each check, validation or invariant
  the PR adds or touches, list every path that builds or changes the object (constructor,
  YAML/config override, `build`, setters) and every mode the check runs in, and confirm each one
  enforces it or doesn't need to. (#816: a `max_slots >= 1` check in `__post_init__` that the YAML
  override skipped; a band check that saw 0 retained keys under capture.)
- **Rows the runtime makes for a request.** A feature that adds per-request or per-row state (an
  adapter slot, sampling params, a position, a cache key) has to follow every mechanism that makes,
  copies or rewrites rows: CFG/guidance's unconditional rows (a model's `cfg_weight`, a
  CFG-parallel rank), speculative and draft rows, `n>1`, padding rows, a packed replay's tail rows,
  rows rebuilt after preemption. For each one the PR's models reach, check the copy carries the
  state. (#822: with guidance on, the LoRA output diverged from the merged-weights output within
  the first tokens and matched exactly at `cfg_weight: 0`; the unconditional rows ran the base
  model.)
- **Failure kinds.** For each endpoint, refusal or load path the PR adds, list every way it can
  fail: missing, unreadable, wrong type, readable but the wrong content, a conflict, a timeout.
  Give the status and text each one gets. Two kinds the description or docs say answer alike but
  don't is a finding (P1 when the description claims it), and so is a server path or internal name
  that reaches the client. (#823: a missing file said "could not be read", but a model checkpoint
  passed as an adapter answered with one of its tensor names.)
- **Configs:** every value the validation *accepts*, not only the shipped ones. A crash on an
  accepted flag is a finding even when no YAML sets it.

Coverage line: `inputs walked: <function> → <classes>; …`

## 2. Follow each behaviour change out to everyone who relies on it

List what the PR changes that a user, client or other component could observe: defaults, limits,
validation (what used to be clamped or ignored and is now refused, or the reverse), precedence
between two keys that set one value, which keys reach which stage, removed or renamed request
fields, wire formats, error codes.

Then find every consumer in the repo and ask what each one now sees: `benchmark/`, the SDK client,
examples, configs, tests (including GPU-gated ones that skip on CPU), docs. A repo client that
sends a value the PR now refuses, or a harness that sends a key the PR now ignores, is a finding.
So is a default that moved without a line in the description.

A new message type is a change for every receiver: `InputIngest._dispatch`
(`mstar/worker/input_ingest.py`) drops an unknown type silently, where the conductor raises. A
per-request message can reach a worker before its NEW_REQUEST or after its REMOVE: it joins
`_NEEDS_ACTIVE_REQUEST` so it is parked or dropped, and a dropped one that carries tensors is acked so its producer frees them.

Coverage line: `behaviour changes → consumers checked: <change> → <files>; …`

## 3. Follow each new assumption back to whatever guarantees it

For every new reliance on something outside the diff, find where it is pinned and check it holds:

- **A library symbol or behaviour** (especially a private `_name` or a recent API) → the version
  that introduced it → the floor in `pyproject.toml`.
- **Every version in the supported range,** not only the floor. A torch or library behaviour that
  changed inside the range (a call that raises on 2.11 and works on 2.12) breaks someone the pins
  allow.
- **A kernel's contract:** argument defaults (a "null block" index that defaults to a real slot),
  alignment and dtype rules → every call site, at every TP degree and batch size.
- **The CI environment:** a new test in a CPU path that skips without `mstar_rust`, CUDA or an
  optional package isn't tested there. Say where it actually runs.
- **The host:** cpusets, NUMA layout, Slurm or cgroup allocations, file and socket paths, a small
  `/dev/shm`, ROCm and XPU when the code is gated on CUDA only.
- **The merge target as it is now.** The PR lands on current `main` (or, stacked, on its base PR's
  current head), not on the commit it branched from. The repository root you review against is
  current `main`: check that every symbol the PR calls still exists there with the signature the PR
  uses, that defaults it relies on (`mstar/utils/knobs.py`, env-var defaults) haven't flipped, that
  nothing new on `main` calls what the PR changes, and that no lint or ban rule (ruff's TID251
  banned imports) now rejects it. With a shell, list the commits since the branch point that touch
  the PR's files (`git log --oneline $(git merge-base <head> origin/main)..origin/main -- <files>`,
  plus `-S<symbol>` for each symbol the PR depends on).

  Do this for the PR's GPU-only files too: `benchmark/`, `test/integration/` and any harness the
  description cites. CPU CI never imports them, so they rot unnoticed. Trace each assertion such a
  harness makes through the code it hits on the merged tree. Check each test fake or stub
  against the real class it stands in for, since a fake that still has a method `main` removed hides
  the crash. (#822's harnesses called `enter_compute` and `prefix_key_streams` after `main` changed
  both; #948 called `get_sp_config_for_node` after `main` removed it, and a test stub hid that;
  #870, #871 and #873 measured before #938 and #1075 flipped defaults they depend on.)

  Check too that every base-class hook the PR overrides, and every registry it adds to, still
  exists under that name: an override of a hook `main` renamed is never called, and nothing errors
  (#379 implemented `get_cuda_graph_configs` after `main` renamed the hook, so decode would have run
  eager). A finding that only `main`'s later commits cause is a post-rebase ask: report it at its
  priority and say the PR needs it once rebased. Don't report what the merge itself shows: a
  conflict, an import error, a failing CI job (except as SKILL.md §2's reachability rule says).

Coverage line: `assumptions → where pinned: <assumption> → <pin, holds | doesn't>; …; merge target: <removed symbols, flipped defaults, new callers, rules: none | list>`

## 4. For each piece of shared mutable state, list every writer and reader in every concurrency domain

For each buffer, dict, socket, file, client, stream or resource state the PR adds or newly
touches, list who writes and who reads it, across:

- threads: the API server's request handlers and result-drain thread; the worker's main, GPU and
  plan threads; executor callbacks; sweepers;
- CUDA streams: the plan stream's copies against default-stream kernels;
- steps under async scheduling: step N+1's plan or H2D overwriting what step N still reads
  (AGENTS.md invariant 7, `force_double_buffer`);
- processes: the API server (where `load_*`, `process_prompt` and output `postprocess` run on its own
  `Model` instance), the conductor and each worker. State set up in one (a config refreshed when
  submodules are built, a cache, a registration) doesn't exist in the others (#249: the checkpoint
  config was refreshed only where submodules are built, so preprocessing never saw it);
- TP ranks: invariant 11, *when* as well as whether. A per-rank report the conductor aggregates
  reads replicated values (new-token counts) from rank 0's report only (`is_first_tp_rank`) and sums
  sharded ones over ranks. A new `WorkerGraphsDone` field or worker metric read the other way double
  counts or drops data;
- separate deployments on one host: file names, socket paths, SHM prefixes.

Then say what orders them. Two or more threads on a handle that isn't thread-safe by contract
(libzmq sockets are single-thread; most clients and in-place-mutated Python containers aren't safe
either), with no lock, event, single owning thread or double buffer between them, is a finding.
Name both sides and the call sites. (#346: a session teardown sent on one ZMQ socket from the HTTP
thread and the result-drain thread with no lock.)

Then walk **failure, then retry**. For every timeout, rollback, cancel or undo the PR adds, step
through the client sending the same operation again right after the error, while the slow side is
still finishing or undoing the first one:

- Can a late ack or a late rollback from the first attempt undo the retry?
- Does the retry see half-undone state (a name still marked loaded on one worker and gone on
  another)?
- What tells the two attempts apart: an op id, a generation counter, a fence?

A retry that ends with the two sides disagreeing about what is served is a finding (P1 when it
splits TP ranks). (#823: a load retried right after its 504 either got a 400 "already loaded" with
nothing serving the name, or a 200 that the slow worker's rollback then undid.)

Coverage line: `shared state → writers/readers → ordering: …; retries: <timeout/undo site> → <what fences the retry> → <holds?>`

## 5. Treat every silent fallback as suspect

List every place the PR, or code it relies on, turns an error into a warning, a default, a clamp,
a catch-all `except` or a degraded path: a capture failure that drops a bucket, an offline start
that falls back to built-in defaults, a value the config set explicitly that the code overrides
with only a log line. Include a feature that is accepted and then has no effect on some path: a
module bound for LoRA whose parent reads its `.weight` directly, so the delta never runs (#821), or
a flag a branch never reads. For each, ask whether the fallback is correct and whether anyone would
notice it firing. A fallback that hides a defect (a misaligned capture served through a larger
bucket) or makes behaviour depend on the host is a finding.

Coverage line: `fallbacks: <site> → <correct?, visible?>; …`

## 6. For every new op on a compiled or captured path, check it runs the same compiled as eager

List each new or changed function that runs inside a `torch.compile`d region or a captured graph:
a forward, a resource op the forward calls, a kernel wrapper, a helper doing view, slice, `expand`,
`as_strided`, in-place or index arithmetic. Find how its siblings are made safe there
(`torch.library.custom_op` like the `mstar::lora_*` ops, `torch.compiler.disable` launchers, an
opaque Triton call) and check the new op does the same. If Inductor traces into it instead, that
is a finding: Inductor can mis-lower strided views and in-place writes, so the op is right eager
and wrong or crashing compiled, and CPU tests, which run eager, never see it. Also flag a new op
whose only tests run eager. The check that settles it is a compiled-vs-eager parity test of that op
on CUDA at a few shapes. (#946: `apply_experts_` was traced while its dense siblings are custom
ops; Inductor mis-lowered its sliced expand and Qwen3-Omni crashed at startup under compile. Only
the GPU test found it.)

On a captured path, also check: a Python value read inside the captured callable is frozen at
capture; a static input reassigned instead of written in place (`copy_`) detaches from the address
the graph holds; and a `cg_key_info()` that disagrees with the step's declaration misses the capture
and runs eager with no error.

Coverage line: `compiled path: <op> → <opaque like siblings? | traced> → <compiled test? y/n>; …`

## 7. For every test the PR adds or changes, find what can make it fail under CI and not alone

CI runs CPU Core as `-n 4 --dist worksteal` on a loaded runner, in an order no one picked. For each
test the PR adds or touches, and each existing test whose code path the PR changes, name the class
below it falls in, or "none":

- **Timing:** a `sleep` used as synchronization, an assertion on elapsed time, a timeout sized for
  an idle machine (an 8 s `init_process_group` timeout a loaded runner exceeds; #723), polling with
  no condition to wait for, or a race the test wins only because one side is usually faster.
- **State shared across tests or xdist workers:** a fixed port, socket path, `/tmp` or `/dev/shm`
  name or SHM prefix; an env var, `torch.backends` flag (`allow_tf32`), `torch.set_default_dtype`,
  seed, cwd, logging level, module global or `lru_cache` set by direct assignment and never
  restored (use `monkeypatch` or a fixture); a singleton or registry a test mutates. (#1146's GPU
  test left `allow_tf32 = False` for the rest of the session.)
- **Order and randomness:** a test that passes only after another one ran (an import that registers
  ops, a warmed cache), unseeded randomness, set or dict order across processes, `PYTHONHASHSEED`.
- **Load and limits:** CPU count, threads (`OMP_NUM_THREADS`), memory or file descriptors that are
  fine alone and short at `-n 4` beside the whole suite.
- **Background work at teardown:** for a test that spawns processes, threads or sockets, or
  exercises a changed collective, process group, socket or async path, walk the lifecycle: spawn or
  connect, import order, init (`init_process_group`, stores, servers), the operation, a failure on
  one side only, teardown (`destroy_process_group`, socket close, event-loop shutdown), interpreter
  exit. At each step ask what still runs in the background (gloo/NCCL threads, threadpool hops,
  pending futures) and whether the test waits for it. For each object the test builds around a
  group, ask whether it is freed before destroy. Known triggers: an import of a module that
  compiles or registers ops (`torch._dynamo`) moved to after a process group exists keeps gloo's
  threads alive past `destroy_process_group`, and the exit unwind hits `std::terminate` (#881:
  SIGABRT in about 2% of spawns under suite load, never alone); anything that still references the
  group after destroy, such as a reference cycle through a bound method or closure, or
  `torch.distributed.nn`'s `group=WORLD` defaults; a client-stream test that closes its socket while
  a call is still in a threadpool hop.

A test in any class is a finding at P2 at least: a flaky test on `main` fails everyone's CI. Give
its fix (the event to wait on, the fixture, the free port, the `monkeypatch`) and, as how to
confirm, "repeat the test N times inside the full suite at CI's `-n`", since such flakes often never
show alone.

Coverage line: `tests: <test> → <flake class, or none> (→ <background work at teardown, waited for? y/n> for lifecycle tests); …`

## 8. Trace each claimed benefit through the code that decides whether it happens

For every gain the PR or its description claims (capacity, memory freed, requests admitted,
throughput, latency), find the component that decides whether the gain is realised (usually an
admission check, a budget, the scheduler or a cost model) and walk the PR's own example through it
with real numbers. Freeing or saving something counts only if that decider sees it. A resource
returned to a pool but still charged to its owner, a cheaper path the dispatcher never picks, and a
faster kernel behind a cache that never misses are all no-ops, however good the local change looks.

Do this on **every configuration branch the claim covers**: single rank and leader and follower
ranks, replicated or not, one GPU or several, each backend or mode. A gain that holds on one branch
and vanishes on another is a finding, and so is a description that states the number without naming
the branch.

Then check **how the number was produced**. A benchmark, harness or test that reaches the mechanism
through a side door (building state directly, skipping the admit or reserve step, calling a helper
production never calls) measures the side door. It has to go through the entry points a served
request uses. For shared code the PR changes on configurations it never runs, say which
configurations reach the change and which of them anything exercises.

Coverage line: `claims: <claim> → <deciding component> → <branches traced> → <holds where?>; measured via <entry point>; …`
