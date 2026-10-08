---
name: benchmarking
description: How to get a trustworthy performance number out of M* — starting and stopping a server, choosing concurrency and warmup, reading benchmark.runner output, deciding whether you are CPU- or GPU-bound, and A/B testing a branch against main. Use this before running any benchmark, profiling a slowdown, or claiming a speedup or regression in a PR.
---

# Benchmarking M*

Most wrong performance conclusions in this repo come from measurement error, not from the code under test. The traps at the bottom of this page — compile costs landing on one request, kernel selection changing across server processes, sampling variance that only appears after a restart — have each produced a confident, wrong answer before. Read those before reporting a regression.

Invariant 10 in [AGENTS.md](../../../AGENTS.md) is the rule this skill serves: a performance claim needs a number from one of these harnesses.

## Before you start

**Activate the venv, don't call `.venv/bin/python` directly.**

```bash
source .venv/bin/activate
```

FlashInfer JIT-compiles kernels on first use and shells out to `ninja`, which lives in `.venv/bin/`. Off `PATH`, the server starts fine and then every request fails with `FileNotFoundError: 'ninja'`. The JIT cache is shared across branches, so the first run on a cold cache pays a one-time compile cost — let warmup absorb it before trusting numbers.

On a shared box: pick explicit GPUs and check `nvidia-smi` first, then again during the run to catch someone else double-booking. Check the port is free before claiming it (`--port`).

## Start a server

```bash
mstar serve <model> --gpus 0,1          # mstar serve -h lists the models
mstar serve <model> --gpus 0,1 --config configs/<name>.yaml
```

The default config is the standard one; the others in `configs/` cover stress testing and the TP configurations. `--gpus` must supply at least as many devices as the config's rank list uses — check the `ranks:` lists in the YAML, which is the authority (a `tp2_sp2` config wants four).

**Poll `/health` for readiness. Do not grep the log for "Application startup complete".**

```bash
curl -sf http://127.0.0.1:$PORT/health
```

If another server still holds the port, uvicorn logs startup-complete and only *afterwards* logs `[Errno 98] address already in use`. A log-grep readiness check sails straight past that and every request then fails with connection-refused.

**Transport:** benchmark over SHM, which is the default `--tensor-comm-protocol` and the safe single-node choice — **and set `MSTAR_SHM_ARENA=1` explicitly, because it is off by default.** That variable picks the SHM *implementation*, not the protocol: `0` (the default) is per-uuid files, `1` is the Rust shared-memory arena, `AUTO` is the arena when the `rust/` extension imports. It must match across the deployment, since arena locations ride in the tensor descriptors. Benchmarking the default without setting it measures the file path, which is not what production should be running. **TCP is very high variance — never use it for numbers.** Do run an untimed functional smoke test with `--tensor-comm-protocol TCP` occasionally to confirm the path works.

## Stop a server

`SIGINT` (Ctrl-C) on the `mstar` process shuts the whole deployment down gracefully. Give it up to a minute. You normally do not need to clean up processes individually.

**But killing it non-gracefully does not kill its workers.** The conductor and workers are `multiprocessing` spawns; orphaned, they hold both their GPU memory and their IPC handles, and the next server fails to start. After any hard kill, reap the tree:

```bash
pgrep -a -u "$USER" -f 'mstar.cli.main serve'
pgrep -a -u "$USER" -f 'multiprocessing'
pgrep -a -u "$USER" -f 'torch/_inductor/compile_worker'
```

Then confirm with `nvidia-smi` that the memory actually came back. Tens of GB in use with no server of yours running means orphans. Leftover IPC socket files under `/tmp/mstar_$USER/` (or `--socket-path-prefix`) are harmless and persist between runs — what breaks a new server is a *live* process still holding one.

## Run a benchmark

The server model name and the client `--model` value **are not the same string**. Check both (`mstar serve -h` and the `--model` choices in `benchmark/runner.py`); `qwen3_omni` on the server is `qwen3omni` on the client, `vjepa2_ac` is `vjepa2ac`.

Use `--profiling-type closed_loop` (semaphore-bounded continuous) over `online` for throughput work.

**Concurrency.** For bandwidth-focused modalities — text out, speech out — sweep it: 1, 2, 4, 8, 16, 32. For long single generations (image, video) one request at a time is fine.

**Request count.** At least 5× the maximum concurrency, unless that is prohibitively slow.

**Warmup counts waves, not requests.** The runner computes `warmup_total = wave_size * num_warmup`, where `wave_size` is `--max-concurrency` for `closed_loop`, `--batch-size` for `offline`, and 1 for `online` (`_warmup` in `benchmark/runner.py`). So `--num-warmup 10` against `--max-concurrency 16` fires 160 warmup requests. Warmup deliberately uses the measurement cadence so the first measured wave doesn't hit cold concurrency paths.

Repeating against the *same* running server carries warm state over, so only the first repeat needs heavy warmup: ~5 waves first, 1 for the rest. Under-warming shows up as repeats getting monotonically faster (22.7s → 17.7s → 9.0s); that trend means the early numbers are warmup artifacts.

**Text output:** prefer `--ignore-eos` with `--output-len-min` / `--output-len-max` set, so output length stops being a free variable.

Repeat closed-loop runs several times — they vary. Batch-size-1 generation is much less run-to-run sensitive and usually needs no repeats unless the box is loaded.

## Read the results

`benchmark.runner` prints its full metrics table (TTFT, ITL, audio SV, RTF, throughput in req/s, text tok/s, audio sec/s) **to stdout only**. The `results.json` from `--output-dir` has a smaller set: JCT mean/median/p90/p95/p99, `request_throughput`, `completed`/`failed`, and per-request records. So **always tee stdout** if you care about TTFT or RTF:

```bash
python -m benchmark.runner ... --output-dir "$RUN" 2>&1 | tee "$RUN/bench.log"
```

Give every run its own `--output-dir` — the example scripts hardcode `.bench_outs`, so `results.json` is overwritten each invocation.

**Always check `failed` in `results.json` before believing a number.** A run where every request returned HTTP 500 still produces a plausible wall time and throughput.

**Prefer output-length-agnostic metrics.** How much text or audio gets produced varies run to run, because it depends on how requests happened to batch. TTFT, RTF for audio, and throughput in tok/s or audio-sec/s are robust; req/s and total benchmark time are not.

## CPU-bound or GPU-bound

Answer this before optimizing anything. It is what `benchmark/worker_phases/` is for (see its [README](../../../benchmark/worker_phases/README.md)):

```bash
python -m benchmark.worker_phases.server \
    --command "mstar serve orpheus --config configs/orpheus_tp2.yaml --gpus 0,1 --port 8100" \
    --period 100 --server-log /tmp/run.log
```

Reading the table:

- **`event_sync` is the only real GPU wait.** Near zero → you are **CPU-bound**. A sizeable value → there is a **GPU-bound** component, and the next step is profiling kernels, not shaving Python. For a firm answer, measure GPU busy time from an nsys capture (the README says how); it cannot be derived from the table.
- **`await_gpu` is not that signal.** It is waiting for the CPU part of the GPU thread.
- **`check_stop` is a side-stream D2H**, not pure Python/Rust CPU work.
- **A host sync hides in whatever phase encloses it** and makes `event_sync` look smaller. An unexpectedly expensive `prepare_inputs` is the usual tell.

The README's table of which phases are waits and which are work is worth reading before you draw any conclusion. Summing a wait into a work total is the most common way to misread this output.

Prefer this over `mstar/profile/` for worker-level timing — the profile output is per-request and much harder to read. If you suspect the slowdown is *above* the worker, at the API server, then `--log-stats` (which is `mstar/profile/`) is the only thing that sees it.

## A/B a branch against main

`configs/` differs between branches even when `benchmark/` doesn't, so **each side must run with its own checked-out configs.** Never copy a config across branches.

The editable install maps the `mstar` package to this checkout's absolute path via a `sys.meta_path` finder, so the `mstar` console script always resolves to *this* directory whatever your CWD is. The simplest A/B is therefore to `git checkout` in place — stash, checkout `main`, run, checkout back — with both sides sharing one venv, which is what you want for a fair comparison. If you do use a second clone, invoke it as `python -m mstar.cli.main` from that clone's root so `sys.path[0]` beats the editable finder.

**Let the checkout settle before starting a server.** Branches don't share a package layout, so a server launched against a half-switched tree dies with `ImportError: cannot import name ... (unknown location)`. After every checkout:

```bash
find mstar -name __pycache__ -type d -prune -exec rm -rf {} +
python -c "import mstar, mstar.conductor.conductor, mstar.worker.worker"
```

Never run two things that touch the repo at once — a `git checkout` racing a server start produces exactly this failure.

Keep the two sides **adjacent in time**: run branch and main for one model back to back before moving to the next model, so drift in machine load doesn't get attributed to the code change.

## Three traps that produce fake regressions

**1. Kernel selection changes across server processes.** Compilation uses `mode="max-autotune-no-cudagraphs"` (`mstar/engine/cuda_graph_runner.py`). Inductor benchmarks candidate kernels at compile time and keeps the fastest by measured wall time — and those measurements are noisy, so two processes running *identical* code can select different kernels. That changes float accumulation order and perturbs logits in the low bits. Greedy stages are unaffected (argmax rarely flips), but any sampled stage can flip a token, and one flipped token changes the whole downstream generation. Observed on qwen3-omni: identical byte totals across repeats within one process, but mean audio duration 55.6s vs 59.0s across two processes of the same code. It moves throughput too, not only sampled output: the same Whisper code over four restarts gave 368–455 RTFx at c=32, while c=1 stayed within ~1%. So don't attribute a cross-process difference to the branch; sample several processes per side, and treat a high-concurrency delta under ~10% as noise until it survives several restarts. When the change can be switched at runtime, run both arms on one build rather than two. Setting `TORCHINDUCTOR_CACHE_DIR` to a shared persistent path should pin selection after the first compile (worth doing, unverified).

**2. The first request of a new shape pays compilation, not inference.** That cost lands entirely on one request and reads like a tail regression. Concrete case: in the VBench I2V set, `req7` is the first landscape image (1024x672; req0-6 are portrait) and takes ~23s where its neighbours take ~7s. `req9` is also landscape and is *not* slow, because req7 already paid. Re-running confirms it — req7 drops from 23.8s to 6.7s. When a p95 or p99 moves on a generation benchmark, check whether one request is carrying a compile, and compare compile cost and steady-state cost separately: a branch can be slower to compile and faster to run.

**3. Repeats must span server restarts.** Repeating against one long-lived server does not measure run-to-run variance for models with stochastic output. qwen3-omni is the clear case: the harness sends `"seed": <request_id>` and `get_model_kwargs` forces `thinker_temperature: 0.0` but deliberately leaves the **talker** at the model default of 0.9 (see the comment in `benchmark/base.py`). The seed pins talker sampling *within* a process, so repeats come back byte-identical and look deterministic. Restart the server and the talker samples differently, changing per-request audio length by up to 20x.

- Byte-identical repeats are **not** evidence of determinism. They may only mean "same process".
- Length-sensitive metrics (`audio sec/s`, `text tok/s`, RTF mean/p95/p99) can shift several hundred percent between server instances with no code change. One short generation moved RTF p99 from 0.11 to 0.60.
- For any model with a sampling stage, **an A/B of one server per side proves nothing.** Run several restarts per branch and compare distributions.
- `RTF p50` and TTFT stayed stable across instances in practice; the tails did not. Prefer p50 and TTFT when you only have a few restarts, and treat tail movement as suspect.
- Check total output volume (`Output bytes`) between the two sides first. If it differs, the length-sensitive metrics aren't comparable and the "regression" may just be a shorter generation.

How many repeats, and how large a delta counts as real, depend on the model and how noisy the box is that day. Look at the spread across repeats before deciding.
