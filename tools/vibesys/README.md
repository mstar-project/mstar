# mstar-vibesys

mstar-side glue that drives **VibeSys** (the agentic build/optimize loop under
`3rd_party/vibesys`, `pip install vibesys`) against the mstar repo. VibeSys is
generic; this package is the thin mstar adapter that, for a given *task*, renders
a VibeSys input bundle, prepares an isolated seed of mstar (+ optional Docker
eval image), hands off to the `vibesys` CLI, and keeps all generated state under
one gitignored directory.

It is a **registry of task types** over shared machinery. Adding a task type is a
new module under `tasks/`; `core/` does not change. The first (and currently
only) task type is **`add-model`**: port a new model into mstar under a
correctness gate, then optimize its serving.

## Quickstart

Run from the mstar repo root, in the env that has `vibesys` (e.g.
`conda activate mstar`):

```bash
python -m tools.vibesys add-model lingbot show          # print resolved spec + paths
python -m tools.vibesys add-model lingbot build-image   # build the Docker eval image (docker path)
python -m tools.vibesys add-model lingbot build         # render bundle + seed clone
python -m tools.vibesys add-model lingbot run --rounds 5 --cli-provider codex
python -m tools.vibesys add-model lingbot clean          # remove this instance's bundle + seed
```

Add `--dry-run` to `run` to print the exact `vibesys …` invocation without
executing. `--no-docker` runs locally instead of in a container (see
[Docker vs local](#docker-vs-local)).

## Source layout (`tools/vibesys/` — tracked)

```
tools/vibesys/
├── cli.py                 # `python -m tools.vibesys <task> <instance> <action>`
├── __main__.py            # enables `python -m tools.vibesys`
├── agent.toml             # VibeSys agent config (codex/gpt-5.5, backend cuda); passed via --config
├── Dockerfile             # reproducible eval image: mstar deps + model extra (+ flash-attn slot)
├── core/                  # task-agnostic machinery (never changes per task)
│   ├── layout.py          #   .vibesys/ paths + image tags; repo_root()
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
        │   ├── OBJECTIVE.md.tmpl    #   the goal + correctness contract (→ bundle)
        │   ├── checker.py.tmpl      #   accuracy gate (→ bundle vibeval/)
        │   ├── benchmark.py.tmpl    #   perf driver (→ bundle vibeval/)
        │   ├── reference.py.tmpl    #   generic oracle stub (→ bundle vibeval/)
        │   ├── config.yaml.tmpl     #   starter mstar config (→ seed)
        │   └── run.sh.tmpl          #   launch contract (→ seed)
        └── instances/
            ├── lingbot.toml          # the per-model spec (the only file you write per model)
            └── lingbot.reference.py   # optional concrete oracle, overrides reference.py.tmpl
```

## Generated runtime layout (`.vibesys/` — gitignored)

Nothing here is committed (`/.vibesys/` is in the root `.gitignore`). The tool
authors the first two; VibeSys creates the rest at run time.

```
.vibesys/
├── bundles/<task>/<inst>/        # the evaluation bundle (what the agent must satisfy)
│   ├── OBJECTIVE.md
│   ├── reference/meta.json       #   HF id+revision → weight download
│   └── evaluator/vibeval/{checker.py, benchmark.py, reference.py}
├── seed/<task>/<inst>/           # standalone mstar clone @ pinned commit (the mutable seed)
│   └── … + configs/<model>.yaml + run.sh
├── exp_env/                      # VibeSys runtime store (via --runs-dir .vibesys/exp_env)
│   ├── _inputs/<exp>/            #   synthesized bundle (OBJECTIVE, vibesys.input.toml, reference/, vibeval/, _seed/)
│   └── <timestamp>-…-<exp>/      #   ONE dir per run:
│       ├── workspace/            #     the live candidate the agent edits (seed ⊕ evaluator, a throwaway copy)
│       └── logs/                 #     rounds.json, per-round logs, usage.jsonl, effective-objective.md
├── hf_cache/                     # persistent HF_HOME so the checkpoint downloads once
└── logs/                         # stdout mirror of the run + FRICTION notes (operator-side)
```

## How it fits together

```
tools/vibesys/{templates,instances}          .vibesys/                       vibesys runtime
────────────────────────────────────   build   ──────────────────   run   ─────────────────────────
 instance spec  ─render─►  bundle  ─────────►  .vibesys/bundles/…  ──┐
                           seed    ─git clone►  .vibesys/seed/…    ──┼─► exp_env/_inputs/<exp>  (synthesized)
                                                                     └─►                       ─► exp_env/<run>/workspace/
                                                                                                    (agent edits this)
```

- **bundle** = the *truth*: objective + correctness gate + oracle. Read-only intent.
- **seed** = the *candidate start*: a real `git clone` of mstar (own `.git`, so
  the agent can `git diff` its work) plus a starter config and `run.sh`.
- VibeSys fuses them into **`exp_env/<run>/workspace/`**, `cd`s the coding agent
  (codex/claude) into it, and that workspace is the only thing the agent edits.

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

Each round is committed in the workspace's git repo; per-run artifacts,
memory files, and logs accumulate under `exp_env/<run>/`.

> **Evaluator visibility.** The runner passes the gate via `--input-evaluator-dir`,
> so `vibeval/` (checker + reference + benchmark) is copied into the workspace and
> is *visible* to the agent — the only guard against tampering is the judge's
> reward-hack detection. For a hard, agent-invisible gate, switch to
> `--input-evaluator-source` / `hidden_evaluator` in `core/runner.py`.

## The `add-model` task

### Instance spec (`tasks/add_model/instances/<model>.toml`)

The only file you write per model. See `lingbot.toml` for a worked example.

```toml
[model]
name = "lingbot"
hf_id = "robbyant/lingbot-video-dense-1.3b"
revision = ""
extra = "all"                      # mstar pip extra baked into the eval image
reference_model = "wan22"          # closest mstar model to copy from
served_model_name = "lingbot"
endpoint = "/v1/videos/generations"
port = 8000
modality = "video_generation"      # vibesys --modality
accuracy_mode = "smoke"            # smoke (valid output) | strict (numeric vs oracle)

[headline]
metric = "latency_p50_ms"
walk = "t2v_generate"
result_arg = "--output-json"

[[walk]]                           # one per observable Walk; compare ∈
name = "t2v_generate"              # {exact_tokens, logit_kl, cosine, audio_mse, ssim}
kind = "generative"
compare = "ssim"
tol = 0.25
```

- **`<model>.reference.py`** (optional) replaces `reference.py.tmpl` with a
  concrete oracle. It holds *all* model-specific logic — `sample_input`,
  `to_request` (how to ask the mstar server), `from_response` (parse candidate
  output), and `oracle` (the reference ground truth for strict mode). The
  generic `checker.py`/`benchmark.py` drive it, so they never change per model.
- **`accuracy_mode`**: `smoke` asserts a valid, non-empty response (fast bring-up);
  `strict` additionally compares to the oracle under the per-walk tolerance.
- **Weights** load through mstar's `HF_MODELS` mapping. Under `--docker` they are
  bind-mounted at `/model`; locally they come from the HF cache (`HF_HOME` →
  `.vibesys/hf_cache`). `run.sh` prepends the workspace to `PYTHONPATH` so the
  candidate's edited `mstar/` wins over any editable install.

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

- **`--docker`** (recommended for GPU serving): VibeSys runs the candidate in a
  GPU-passthrough container (`backends/cuda` → `--gpus device=N`) built from
  `Dockerfile`, with weights at `/model`. This is the path to a real live smoke.
- **`--no-docker`**: runs locally. Lower setup cost, but the coding agent's
  execution sandbox (codex `bwrap`) has **no usable CUDA** (`cudaGetDeviceCount`
  Error 304), so the candidate server can start and route but cannot run the
  model on GPU — good for wiring/graph/port correctness, not a live GPU smoke.

## Notes / gotchas handled by the runner

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
- `--runs-dir .vibesys/exp_env` keeps VibeSys's store under `.vibesys/` instead of
  the launch dir.

## Keeping VibeSys in sync

`3rd_party/vibesys` is a clone of `uw-syfi/vibesys`, installed editable into the
env, so upstream sync is:

```bash
git -C 3rd_party/vibesys pull        # + `uv pip install -e 3rd_party/vibesys` only if entry points/deps changed
```
