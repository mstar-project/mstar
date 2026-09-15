# mstar-vibesys

mstar-side glue that drives **VibeSys** (the agentic build/optimize loop;
`pip install vibesys`) against the mstar repo. VibeSys is
generic; this package is the thin mstar adapter that, for a given *task*, renders
a VibeSys input bundle, prepares an isolated seed of mstar plus a GPU eval image,
hands off to the `vibesys` CLI, and keeps all generated state under one gitignored
directory.

It is a **registry of task types** over shared machinery. Adding a task type is a
new module under `tasks/`; `core/` does not change. The first (and currently
only) task type is **`add-model`**: implement a new model natively in mstar
(submodule nodes, walk graph, engine types, weight loading — not a wrapper around
the upstream pipeline), verified against the reference implementation, at
reasonable performance.

## Quickstart

Run from the mstar repo root, in the env that has `vibesys` (e.g.
`conda activate mstar`):

```bash
python -m tools.vibesys add-model lingbot show          # resolved spec + paths
python -m tools.vibesys add-model lingbot build-image   # build the GPU eval image (once)
python -m tools.vibesys add-model lingbot run --rounds 5 --cli-provider codex
python -m tools.vibesys add-model lingbot clean         # remove this instance's generated state
```

`run` **defaults to docker** (the candidate serves on a real GPU). Add
`--no-docker` to run locally instead (faster setup, but no GPU in the agent
sandbox — see [Docker vs local](#docker-vs-local)). `run` also does the `build`
step (clone seed + render bundle) itself. Add `--dry-run` to print the exact
`vibesys …` invocation without executing.

## Source layout (`tools/vibesys/` — tracked)

```
tools/vibesys/
├── cli.py                 # `python -m tools.vibesys <task> <instance> <action>`
├── __main__.py            # enables `python -m tools.vibesys`
├── agent.toml             # VibeSys agent config (codex/gpt-5.5, backend cuda); passed via --config
├── Dockerfile             # minimal GPU eval image: CUDA base + uv + ffmpeg (no mstar baked)
├── core/                  # task-agnostic machinery (never changes per task)
│   ├── layout.py          #   .vibesys/ paths (per-instance) + image tags; repo_root()
│   ├── render.py          #   string.Template rendering (${var}, brace-safe)
│   ├── seed.py            #   standalone `git clone` of mstar @ pinned commit
│   ├── image.py           #   docker build / image-exists
│   └── runner.py          #   builds & execs the `vibesys` CLI (synthesis flags)
└── tasks/
    ├── base.py            # TaskType ABC + REGISTRY + register()
    ├── __init__.py        # imports each task module → populates REGISTRY
    └── add_model/
        ├── task.py        # AddModelTask: spec parsing, bundle render, seed files, vibesys wiring
        ├── templates/     # rendered per instance:
        │   ├── OBJECTIVE.md.tmpl    #   implement-it-yourself contract (→ bundle)
        │   ├── reference.py.tmpl    #   generic oracle stub (→ bundle vibeval/)
        │   ├── config.yaml.tmpl     #   starter mstar config (→ workspace-seed)
        │   ├── run.sh.tmpl          #   launch contract (uv run mstar-serve … SHM)
        │   ├── serve_and_eval.sh.tmpl  # brings the server up for the official gate
        │   └── modalities/<modality>/  # MODALITY-specific eval (selected by spec.modality)
        │       ├── checker.py.tmpl  #     e.g. video_generation: SSIM vs the reference
        │       └── benchmark.py.tmpl
        └── instances/          # per-model inputs; real <model>.* are gitignored (local)
            ├── example.toml           # tracked template: the per-model spec
            └── example.reference.py    # tracked template: the optional oracle override
```

## Generated runtime layout (`.vibesys/` — gitignored)

Everything for one port lives under `.vibesys/<task>/<instance>/`; nothing is
committed (`/.vibesys/` is in the root `.gitignore`). The tool authors `bundle/`
and `workspace-seed/`; VibeSys creates `runs/` at run time.

```
.vibesys/
├── add-model/lingbot/                 # one instance = one place
│   ├── bundle/                        # the evaluation bundle (what the agent must satisfy)
│   │   ├── OBJECTIVE.md
│   │   ├── reference/meta.json        #   HF id+revision → weight download
│   │   └── evaluator/vibeval/{checker.py, benchmark.py, reference.py}
│   ├── workspace-seed/                # standalone mstar clone @ pinned commit (immutable input)
│   │   └── … + configs/<model>.yaml + run.sh
│   └── runs/                          # VibeSys runtime store (--runs-dir)
│       ├── _inputs/<exp>/             #   synthesized bundle (OBJECTIVE, vibesys.input.toml, reference/, vibeval/, _seed/)
│       └── <timestamp>-…-<exp>/       #   ONE dir per run:
│           ├── workspace/             #     the live candidate the agent edits (seed ⊕ evaluator, a throwaway copy)
│           └── logs/                  #     rounds.json, per-round logs, usage.jsonl, effective-objective.md
├── hf_cache/                          # shared weight cache across instances (HF_HOME)
└── logs/                              # tool stdout mirror of runs + operator notes
```

Naming note: `workspace-seed/` matches VibeSys's `--input-workspace-seed` flag —
it is the **immutable per-run input** that VibeSys copies into the mutable
`runs/<run>/workspace/`. The tool regenerates it on each `build`; a run never
writes back to it.

## How it fits together

```
tools/vibesys/{templates,instances}       .vibesys/<task>/<inst>/        vibesys runtime (runs/)
──────────────────────────────────  build  ────────────────────  run  ──────────────────────────
 instance spec  ─render─►  bundle  ───────►  bundle/         ──┐
                 ─clone──►  seed   ───────►  workspace-seed/ ──┼─► runs/_inputs/<exp>  (synthesized)
                                                               └►                     ─► runs/<run>/workspace/
                                                                                          (agent edits this)
```

- **bundle** = the *truth*: objective + correctness gate + oracle.
- **workspace-seed** = the *candidate start*: a real `git clone` of mstar (own
  `.git`, so the agent can `git diff` its work) + a starter config and `run.sh`.
- VibeSys fuses them into **`runs/<run>/workspace/`**, `cd`s the coding agent
  (codex) into it, and that workspace is the only thing the agent edits.

### What the agent sees, and how output is stored

Three roles run per round (`--rounds N`):

- **Orchestrator** — reads the objective + memory (`progress.md`, `roadmap.md`,
  `pareto-frontier.md`) + git history + profiler hints → emits **one** hypothesis
  (`workspace/progress-artifacts/plans/round-NNNN.json`).
- **Implementer** — edits the candidate (the mstar port), runs local checks,
  writes `progress-artifacts/evidence/…`, and nominates.
- **Judge** — fresh, read-only; verdict pass/fail; checks for reward-hacking
  (e.g. weakening `vibeval/checker.py`). **Only after a PASS does the framework
  itself** run `vibeval/checker.py` (accuracy) and `vibeval/benchmark.py` (perf).

Per-round artifacts, the agent's memory files, and logs accumulate under
`runs/<run>/` (`workspace/…` and `logs/…`). This is per-run and **ephemeral**:
each `run` synthesizes a fresh workspace from the seed, so memory/profiling do
not carry across separate runs (only within a run, or via `--resume`).

> **Two known interleaving issues (deferred).** (1) VibeSys copies the evaluator
> (`vibeval/`) into the workspace via `--input-evaluator-dir`, so it is *visible*
> to the agent — the guard against tampering is the judge's reward-hack check;
> `--input-evaluator-source` would make it agent-invisible. (2) VibeSys's per-round
> git commits track the whole workspace (mstar code + scaffolding interleaved), so
> the mstar contribution must be extracted with a path filter
> (`git diff <base> -- mstar/ configs/ pyproject.toml`). A clean split (mstar in its
> own repo) needs a VibeSys-side change and is tracked separately.

## The `add-model` task

The agent must **implement the model natively** in mstar (submodule nodes, walk
graph, engine types, weight loading) — *not* wrap the upstream pipeline. The task
is described by client-facing **modalities** (input/output), never by walks
(those are the agent's internal design). You provide two gitignored files under
`instances/` (copy the tracked `example.*`):

- **`<model>.toml`** — the declarative spec (below). Required.
- **`<model>.reference.py`** — the oracle + client I/O hooks (`sample_input`,
  `to_request`, `from_response`, `load`, `oracle`). Needed for the default
  `strict` gate; overrides `templates/reference.py.tmpl`. The oracle is the
  independent *upstream* model — it must not reuse the candidate's mstar code.

`checker.py` / `benchmark.py` are **modality-specific** (from
`templates/modalities/<modality>/`, selected by `spec.modality`) and reused
across models of that modality; the model specifics live in `reference.py`.

### Instance spec (`tasks/add_model/instances/<model>.toml`)

See `example.toml` for the tracked template.

```toml
[model]
name = "lingbot"
hf_id = "robbyant/lingbot-video-dense-1.3b"
reference_model = "wan22"           # mstar model to copy STRUCTURE from
served_model_name = "lingbot"
endpoint = "/v1/videos/generations"
modality = "video_generation"       # selects the vibeval templates + vibesys --modality
input_modalities = ["text"]         # client-facing — what the walks are is up to the agent
output_modalities = ["video"]
accuracy_mode = "strict"            # strict (fidelity vs oracle) | smoke (validity only)

[accuracy]
compare = "ssim"                    # output-level metric: ssim | audio_mse | cosine | ...
tol = 0.2

[headline]
metric = "latency_p50_ms"           # reasonable-latency target (not an optimization goal)
result_arg = "--output-json"
```

- **Goal = correct implementation at reasonable perf, then stop** — the objective
  tells the agent this is not a latency-optimization task.
- **`accuracy_mode`**: `strict` (default) compares the served output to the
  reference oracle via `compare`/`tol`; `smoke` only asserts a valid non-empty
  response (fast bring-up). The strict checker runs in the candidate env
  (`uv run --extra <model>`) so it can execute the oracle.
- **Weights** load through mstar's `HF_MODELS` mapping. Under `--docker` they are
  bind-mounted at `/model`; the shared `HF_HOME` (`.vibesys/hf_cache`) means the
  checkpoint downloads once and is reused.
- **New modality?** Add `templates/modalities/<modality>/{checker,benchmark}.py.tmpl`.

### Adding a new task type

1. Create `tasks/<type>/task.py` with a `TaskType` subclass implementing
   `load_spec`, `render_bundle`, `seed_files`, `synthesis_inputs`, `run_options`
   (and optional `image`). See `tasks/base.py` for the contract.
2. Register it in `tasks/__init__.py`: `register(<Type>())`.
3. Add `tasks/<type>/instances/<name>.toml`.

The oracle concept generalizes: `add-model` uses an **external** reference (HF
model); a future `optimize`/`kernel` task would use mstar **at a baseline commit**
as the oracle. Only `render_bundle` changes; `core/` is untouched.

## Docker vs local

- **`--docker`** (default, recommended for GPU serving): VibeSys runs the
  candidate + codex in a GPU-passthrough container (`backends/cuda` →
  `--gpus device=N`) built from `Dockerfile`, with weights at `/model`. The image
  is a **minimal CUDA base + uv** — mstar and its deps resolve at run time from
  the candidate's own `pyproject`/`uv.lock` via `uv run` (self-contained Python,
  so no sandbox stdlib breakage; the candidate's torch/CUDA pins are honored).
- **`--no-docker`**: runs locally. Lower setup cost, but the coding agent's
  execution sandbox (codex `bwrap`) has **no usable CUDA** (`cudaGetDeviceCount`
  Error 304), so the candidate server can start and route but cannot run the
  model on GPU — good for wiring/graph/port correctness, not a live GPU smoke.

## Notes / gotchas the runner handles

- The official accuracy/benchmark run through `vibeval/serve_and_eval.sh`, which
  brings `run.sh` up on the fixed port and waits for `/health` before running the
  gate, then tears it down. The framework-owned eval commands are pure HTTP
  clients — nothing else starts the candidate server for them.
- Candidate launch uses `uv run mstar-serve … --tensor-comm-protocol SHM`:
  `uv run` gives a self-contained Python, and SHM avoids `mstar-serve`'s RDMA
  default (which needs the optional `mooncake-transfer-engine`; `mstar serve`
  already defaults to SHM).
- Eval commands run via `uv run --no-project --with httpx --with numpy` so they
  work regardless of the container's system Python.
- Strips exported HPC bash functions (`BASH_FUNC_module/scl/ml/…`) that otherwise
  spew "error importing function definition" on every agent command.
- Passes `--config tools/vibesys/agent.toml` (installed VibeSys resolves a
  default `agent.toml` from the launch dir).
- `--input-benchmark-result-arg=<opt>` is joined with `=` (argparse reads a bare
  `--output-json` as a flag otherwise).
- Eval harness is namespaced under a single `vibeval/` dir so it never collides
  with mstar's own top-level `benchmark/`.
- Seeds via standalone `git clone` (not `git worktree`) so the copied `.git` is
  not a dangling gitlink.
- `--runs-dir .vibesys/<task>/<inst>/runs` keeps VibeSys's store under the
  instance; the stale `_inputs/<exp>` from a prior run is auto-cleared before
  each run.
- Docker build context is `tools/vibesys/` (small), not the repo root.

## Requirements

VibeSys must be installed in the active environment (`pip install vibesys`) so
the `vibesys` CLI is on `PATH`; the tool shells out to it. A coding-agent CLI
(e.g. Codex or Claude Code) must also be installed and authenticated. Upgrade
VibeSys the usual way — `pip install --upgrade vibesys`.
