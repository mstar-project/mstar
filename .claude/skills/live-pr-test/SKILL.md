---
name: live-pr-test
description: Live-test an M* pull request on real GPUs — read the PR, size the cheapest Hopper-or-newer box for it, provision it on RunPod or Lambda (or use the persistent cluster), test the PR's claims plus regressions against main, copy media back, tear the box down, and report in markdown. Use when asked to live-test, GPU-test, smoke-test or benchmark a PR, or to "run the PR on a real box".
---

# Live-testing a PR

The job: say whether a PR does what it claims on real hardware, and whether it breaks or slows anything else, with evidence the requester can check (numbers, logs, and media to look at and listen to).

Read alongside this:

- **[benchmarking](../benchmarking/SKILL.md): load it at the start of every live test, before planning.** Don't wait for a perf row: it covers functional and correctness checks too, and every server start and stop. A subagent that runs the PR loads it as well. Its traps apply unchanged. These are the pointers that would have changed earlier live-test verdicts:
  - "Kernel selection changes across server processes" (trap 1). On #642 we blamed the torch version for an exactness failure that came from which kernel autotune picked.
  - "Output-equivalence checks need a same-code control", and "Byte-identical repeats are not evidence of determinism".
  - "Repeats must span server restarts", and "Report a delta only from arms run back to back in one allocation".
  - "When the expected effect is smaller than the noise, time the mechanism directly". Several of our "not separable" and "didn't fire" rows needed this.
  - "Verify what the client actually sent" (probe reach). Also "Check `failed` in `results.json`" and "Prefer output-length-agnostic metrics".
  - "The first request of a new shape pays compilation" (trap 2), and "Slow eager steps are often recompiles" (`TORCH_LOGS=recompiles`), before calling a tail or a stall a regression.
  - "CPU-bound or GPU-bound" via `benchmark.worker_phases`, whenever a throughput row moves.
  - "Match the configuration before comparing" for any reference or competitor arm: sampling defaults, state dtype, prefix caching.
- [AGENTS.md](../../../AGENTS.md) for the invariants a PR can break: invariant 4 (one model's change must not degrade others) and invariant 5 (Python/Rust drift) drive most regression choices.
- [references/test-matrix.md](references/test-matrix.md) for touched paths → suites, the worker A/B set, concurrency and sizing.
- [references/report.md](references/report.md) for the report and the PR-comment rules.

## Modes

- **cloud** (default): provision with `scripts/gpu.py`, work over ssh, copy results back with scp, terminate.
- **cluster** (`cluster` in the request, or `mode: cluster` in the config): no provisioning, ssh or scp. Run on the cluster directly and write results to `cluster.results_dir/pr-<n>-<timestamp>/`, which the requester opens in VS Code. Everything else is the same.

Settings live in `config.local.json` next to this file (gitignored; copy [config.example.json](config.example.json)). API keys come from `RUNPOD_API_KEY` and `LAMBDA_API_KEY` in the environment. Never print them, write them to the box, or put them in a command line that gets logged.

## Several PRs at once

When asked to test more than one PR, the session that was asked (the parent) keeps every decision that needs the requester, and one subagent per PR does the running.

- **The parent** does step 1 for every PR: trust checks, plans, boxes and estimated cost. It shows all the plans together, and gets one answer on the batch budget. It sets `budget.max_concurrent_instances` and `budget.max_total_usd` for the batch; `launch` enforces both under a lock on the shared ledger, so parallel launches can't overshoot them.
- **Each subagent** (Agent tool; a Workflow only if the requester opted into one) gets one approved plan, and runs steps 2–5 for it: provision, run, collect, tear down, write `report.md`. It doesn't ask the requester anything, pass `--allow-over-budget`, or extend past the plan. When the plan can't fit, or a decision would change it, the subagent cuts suites as in step 3, or stops and returns the question to the parent.
- **The parent** reads each report, relays the verdicts, posts PR comments when asked, and runs `gpu.py list` at the end, after every subagent has finished.
- **Cluster mode:** the PRs share GPUs, so only functional checks run in parallel, on disjoint GPUs and with distinct ports, `SERVER_STATE_DIR` and `--socket-path-prefix`. Perf phases run one PR at a time.

## 1. Read the PR and plan

```bash
gh pr view <n> --json number,title,body,author,headRefName,headRefOid,baseRefName,isCrossRepository,files,additions,deletions
gh api repos/mstar-project/mstar/pulls/<n> --jq .author_association
gh pr diff <n>
```

- **Trust check first.** The box runs the PR's code with the team's persistent storage mounted (HF cache, possibly an HF token). If `author_association` is not `OWNER`, `MEMBER` or `COLLABORATOR`, stop and ask before provisioning; mention anything in the diff that reaches the network, the filesystem outside the repo, or environment variables.
- A PR based on another feature branch (stacked) is compared against its base branch, not `main`. Say so in the plan.
- Extract the claims: every behavior or number the description asserts. Each becomes a check. For each claimed number, note the GPU, harness and metric it came from, so the report can say when yours isn't comparable.
- A case the description hands to another open PR is that PR's: test it there (or on both merged) if it matters, otherwise mark it "owned by #N" in the report.
- **Map what could break, and test each risk directly.** The claims table checks what the author thought of. The risk map checks what they didn't, and it is where human reviewers found most of the bugs our runs missed. Read the whole diff, not the description.
  - For every changed function, find its callers and the paths that reach it: eager and captured, TP leader and follower, the Python and Rust runtimes and frontends, streaming and not, the first request of a shape and steady state, and failure, abort and teardown.
  - For each path, write down what would go wrong if the change were subtly wrong, and plan a test that fails if it is.
  - Run that test on a model, config and layout that actually take that path, with the input or fault that provokes it.
  - Put the risk map in the plan, one line per risk: path → what could break → the test. [test-matrix.md](references/test-matrix.md) lists, by layer, the probes that have found bugs.
- **Reach before results.** Before trusting a pass, show that the test reached the changed code: a log line, a counter, a probe hit, or an injected fault firing there. A pass on a setup that never takes the changed branch is ➖, not ✅. On #440 the bug was on the default Rust runtime and we ran only Python. On #557 the race needs the two stages on different workers.
- **Never CPU tests alone.** Unless the PR changes only tests, docs or comments, it gets a served run with a real checkpoint that goes through the changed code. CPU suites are a gate, not a verdict. Code that only non-CUDA workers or the API server run still gets a served run that reaches it.
- **Both runtimes when they apply, Rust otherwise.** When the change touches code that has both a Python and a Rust implementation (graph runtime, control transport, wire codec, API server frontend), run every functional and risk check on both. That means `MSTAR_RUST_GRAPH`/`MSTAR_RUST_ZMQ` at their default and at `0`, and `mstar-serve` with and without `--rust-frontend` (AGENTS.md invariant 5). Otherwise run the default Rust runtime and confirm in the server log that it ran. `MSTAR_RUST_GRAPH=0` is never the only arm.
- **Real checkpoints only.** Use the real released weights for every served check: no dummy-mode models, random weights, or edited and truncated copies of a checkpoint. If the PR's case only exists in a checkpoint no one ships, say so in the plan and ask. On #658, the real checkpoint already showed the bug that we had built an artificial copy to show.
- **Test on the merged tree too.** Run the CPU gate and the risk checks once more on the PR merged into the current base, and on top of its stacked neighbours when it has them. Bugs on #346 and #519 only showed up there. Label each row with the tree it came from.
- Map touched files to regression suites with the test matrix.
- Size the box: GPU count from the largest config in the plan, 80 GB-class Hopper or newer. Prefer the GPU type the claimed numbers came from when it fits the budget. When the code branches on architecture (`has_cuda_capability`, sm80 paths, ROCm), list the architectures you couldn't run under "Not tested".
- If the PR changes only docs, tests or comments and claims no number, say so and stop.
- **Correctness is checked against the reference, never only against main.** A check that compares PR with main shows the outputs moved or didn't; it can't show that either side is right. #452 shipped noise because every check compared Orpheus with itself.
  - The reference is the official implementation (transformers, or the model's own package or `trust_remote_code` files), pinned, in its own venv.
  - Compare at the lowest level that both sides expose: logits or hidden states for the same inputs, then greedy tokens, then decoded output.
  - For speech, also run an objective check against the intended content, such as ASR word error rate on the transcript, and speaker similarity when the PR touches voice.
  - A new checkpoint on an existing architecture still gets the reference row.
  - **When the output is media and the metrics can't decide** (garble, artifacts, timbre or prosody, an image that looks off), don't give a verdict. Save reference, PR and main for the same prompt and seed side by side, mark the row "needs human review", and say what to listen or look for.
- Fetching the reference implementation or a competitor engine is allowed, so a model-add PR gets a parity row and a perf claim gets its competitor arm. Use the official repository, pinned to a commit or release (the one the PR's description used, if it names one); a checkpoint's own `trust_remote_code` files count as its reference. Read the code that will run for network access, file writes outside its directory and environment reads before running it. Install it in a venv of its own, never the shared environment, and record the repository and commit in `harness-notes.md`. Fetch it once into the work directory, outside the timed runs.

Show the plan before provisioning: claims → checks, suites, concurrency, the box, the estimated hours and cost. Then proceed without waiting unless the run would exceed the budget (below) or the trust check failed.

## 2. Provision (cloud mode)

```bash
S=.claude/skills/live-pr-test/scripts
python3 $S/gpu.py check                         # keys, storage regions, budget
python3 $S/gpu.py offers --gpus 4               # cheapest first, only regions with team storage
python3 $S/gpu.py offers --gpus 4 --all-regions # what exists anywhere, for the no-availability report
python3 $S/gpu.py launch <offer-id> --pr <n>    # last line is the instance key; arms the deadline watchdog
python3 $S/gpu.py wait <instance>               # blocks until ssh works; prints GPUs and hf_home
python3 $S/gpu.py ssh <instance> '<command>'    # also: push <instance> <local> <remote>, pull <instance> <remote> <local>
python3 $S/gpu.py list [--all]                  # what this tool launched that is still up (--all: whole accounts)
python3 $S/gpu.py terminate <instance>
```

- Take the cheapest offer that has the GPU count, in a region with the team's persistent storage (`offers` filters for that). An offer with more GPUs than needed is fine when it's cheaper.
- **No availability:** if `offers` returns nothing, do not provision anything. Report "no availability" with the `--all-regions` output: what GPU counts and types are available where, at what price, and whether a smaller plan (e.g. the 2-GPU substitutes in the test matrix) would fit what is available. Stop there.
- **Budget:** `budget.max_usd_per_hour` and `budget.max_hours` from the config. `launch` refuses an offer above the hourly cap, and arms a watchdog that terminates the instance at the deadline even if this session dies. The watchdog runs on the machine that launched it, so a laptop that sleeps or shuts down stops it: when this session ends early, `gpu.py list --all` and terminate by hand. To go over either cap, ask first; `gpu.py extend <instance> --hours H` moves the deadline. `budget.max_concurrent_instances` (default 1) and `budget.max_total_usd` (what the live instances cost if they all run to their deadlines) bound parallel runs. If `terminate` fails for lack of the API key (e.g. after a terminal restart), `gpu.py extend <instance> --hours 0`: the watchdog was started with the key and terminates the instance within a minute.
- **Check the GPUs before building anything.** `gpu.py push <instance> $S/gpu_health.py '~/'`, then `gpu.py ssh <instance> 'python3 ~/gpu_health.py'` (the image's python has torch). It loads every GPU for a minute and exits 1 on a thermal throttle or a GPU running far hotter or slower than the rest. A TP instance runs at its slowest rank's pace, so one badly cooled GPU invalidates every number on the box, competitors' included. On failure, terminate and launch another offer.
- **Always terminate**, on success and on failure, after the copy-back. Before ending the session, run `gpu.py list` and confirm nothing launched for this PR is still up.

Then set up the box:

```bash
python3 $S/gpu.py push <instance> $S/bootstrap.sh '~/'
python3 $S/gpu.py push <instance> .claude/skills/benchmarking/scripts/server.sh '~/'
python3 $S/gpu.py ssh <instance> 'bash ~/bootstrap.sh <n> --hf-home <hf_home from wait> > ~/bootstrap.log 2>&1; tail -20 ~/bootstrap.log'
```

`bootstrap.sh` builds two arms, `~/live-pr/pr` and `~/live-pr/main`, each with its own venv and its own `mstar_rust` build, and writes `~/live-pr/env.sh` (HF home, local caches, `MSTAR_SHM_ARENA=1`, and the `arm pr|main`, `stage_model` and `save_caches` helpers). It takes a few minutes, and restores the Inductor, Triton and FlashInfer compile caches from a tarball on persistent storage when one exists for the box's compute capability. Pass `--base <branch>` for a stacked PR, and `--no-main` when nothing in the plan compares against the base. Read `results/env.txt`: a non-empty dependency diff between the arms has to be explained before any A/B number means anything. Also check each arm's installed versions against the repo's floors (`pyproject.toml`, and the docs' install notes for packages installed separately, such as Mooncake): a dependency below its floor can fail in ways the PR never touches. Upgrade it, or flag it in the report's header line.

In cluster mode, run the same script on the cluster with `--workdir` and `--hf-home` from the config.

Installing packages and making new venvs is fine, and expected when a check needs them: a PR that changes dependencies, an extra the shared environment lacks (a G2P package, an older `setuptools`), a pinned torch build, or a library version the code requires (e.g. FlashInfer). Build them in the work directory, never in the shared environment, and name them in `harness-notes.md`. A `--no-deps` overlay put first on `PYTHONPATH` is often enough for one package.

A venv with a different torch (or FlashInfer/Triton) version gets its own JIT caches: point `FLASHINFER_WORKSPACE_BASE`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` at a directory of its own whenever that arm runs. FlashInfer keys its cache by its own version, not torch's, so two torch versions share one directory, rebuild each other's kernels, and hold the build lock while they do. One torch-2.10 arm's first boot (a `gdn_prefill` rebuild) blocked every other Qwen3.5 server on the node until it was stopped. Expect that arm's first boot to recompile everything (20–30 minutes); size its readiness timeout for it.

## 3. Run

Never serve weights straight from FUSE-mounted persistent storage (RunPod's `/workspace` is geesefs): a server mmaps the checkpoint, and random page-ins through FUSE ran at ~4 MB/s per worker, so a 27B TP4 load had not finished after 20 minutes. `bootstrap.sh` detects a FUSE mount and points `HF_HOME` at local disk; run `stage_model <hf repo>` for every model in the plan before its first server start. It downloads from the Hub (52 GB in ~1 min, where a sequential copy off geesefs took ~9) and falls back to the persistent cache for a repo the Hub won't serve. On NFS (the cluster, Lambda) it is a no-op.

Start every server through the benchmarking skill's `server.sh` (copied to `~/server.sh`, so both arms use the same one even when a PR predates or edits it), from the arm's checkout, with the arm's venv active:

```bash
source ~/live-pr/env.sh && arm pr
MSTAR_TP_ASYNC_SCHED=1 ~/server.sh start pr 8100 --timeout 2400 \
  --log $LIVE_RESULTS/logs/<suite>.pr.r1.log -- \
  mstar serve <model> --config configs/<cfg>.yaml --gpus 0,1,2 --port 8100 --socket-path-prefix /tmp/mstar_pr
```

- Start servers with `mstar serve` (or `python -m mstar.cli.main serve`), whose tensor transport defaults to SHM. `python -m mstar.api_server.entrypoint` defaults `--tensor-comm-protocol` to RDMA, so a run through it exercises Mooncake, and can fail there, even when the PR doesn't touch transport. Pick another transport only when the PR is about transport, and record the transport in the report's header line either way.
- Run anything longer than a couple of minutes detached on the box (`nohup ... > log 2>&1 &`, or tmux if present), then poll its log. An ssh drop must not kill a sweep.
- Record the PR head you test. Before a follow-up run, and again before posting, compare it with `gh pr view <n> --json headRefOid`: if the head moved, rerun what the new commits touch or label those rows "old head `<sha>`".
- Timing-dependent bugs (stalls, races, hangs) depend on the host: when the base arm fails for you and not for the author, or the reverse, put host load on the row, and try reproducing under deliberate CPU load (a busy-loop per core) or with SM clocks locked before calling either side.
- Order: functional checks of the claims and the risk map first, cheapest before expensive, so an early failure is found before an hour of benchmarks.
- **Explain inconclusive results.** When a rate doesn't separate, or a fix "didn't fire", don't stop at the counts.
  - Make the bug deterministic: inject a delay or fault at the code the PR changes, shrink the pool, or force the setting that reaches the rare path.
  - Or trace the mechanism in the logs. On #461 a reviewer traced the refusals to a circular wait and reached p=0.014, where our counts had stopped at p=0.19.
  - Only then call it ➖ or not separable.
- **Watch resources as well as status codes.** For input handling, uploads, decoding or anything that buffers, record the API server's RSS and event-loop latency during the check. A 200 that stalls the loop for seconds, or a request that allocates gigabytes, is a finding (#631, #639).
- **Close your own "Not tested" list.** Before writing the report, go through every "not tested" line: anything the box can reach (another model, the other runtime, a fault you could inject) gets run, not listed.
- **Idle GPUs get killed on the cluster.** A process that holds GPUs at 0% SM for about 30 minutes is killed. mstar workers turn the SIGTERM into exit code 0, so all ranks exit together with no traceback. A checkpoint that takes longer than that to load, such as GLM-5.2-FP8 at 756 GB over NFS, dies before it is ready. Ask how to exempt the job before spending an 8-GPU slot on it.
- Each A/B model: main and PR back to back, restart the server between repeats, then the next model. When nothing is being timed, the two arms may run in parallel on disjoint GPUs.
- `MSTAR_TP_ASYNC_SCHED=1` on any TP/SP config; `MSTAR_SHM_ARENA=1` always (from `env.sh`). Check each server log for `control mesh transport: rust` and `graph runtime: rust`, and on TP configs for `all-reduce fusion workspace:`. On cloud containers without NVSwitch multicast, FlashInfer's MNNVL backend fails, and before #563 mstar fell back to NCCL all-reduce, 1.4–1.6x slower on a 27B TP4 model. The same hosts need `NCCL_NVLS_ENABLE=0` for competitors whose NCCL tries NVLS. Note either in the report when it applies.
- One `--output-dir` per cell under `$LIVE_RESULTS/bench/<suite>/<arm>/c<conc>.r<rep>/`, stdout tee'd to `bench.log` inside it. Check `failed` in every `results.json` before using a number.
- Log host load with every perf cell: the 1-minute load average and `nvidia-smi pmon` (with each pid's owner, since other users' processes can land on your GPUs). `gpu_health.py` catches a throttled GPU, not a busy host. A cell run with more than half the cores busy, or with the GPU mostly idle because the host is the bottleneck, is rerun or carries a caveat, and a delta between cells at very different load is not evidence.
- On a shared machine: run Python from a directory you own (a stray module in a shared `/tmp` gets imported), give each extra server in a job a port from its own range rather than base+1 (the next job's base port is base+1), and give a start that is expected to hang (a bug repro on the base arm) a short `--timeout`.
- Save media as the test matrix describes. Keep it small: a few representative outputs per suite and arm.
- If a server fails to start or every request errors on **both** arms, it is an environment problem, not a regression. Fix it or record it under "Not tested", and keep going with the rest of the plan. When the box itself makes the PR's code slower (a kernel backend that fails here and silently falls back, a missing NVLink/NCCL feature), find the setting that works, then rerun the claim checks and the base and competitor arms with it, so the headline numbers reflect the code and not the box. Log every setup problem and its fix in `$LIVE_RESULTS/harness-notes.md` as you go; none of it belongs in the report (see [references/report.md](references/report.md)).
- Watch the clock against the deadline. If the plan won't fit, cut the lowest-priority suites (test matrix order) and record them as not tested.

## 4. Collect and tear down

```bash
OUT=$(python3 -c 'import json,os;print(os.path.expanduser(json.load(open(".claude/skills/live-pr-test/config.local.json"))["results_dir"]))')/pr-<n>-$(date +%Y%m%d-%H%M)
python3 $S/gpu.py ssh <instance> 'source ~/live-pr/env.sh && save_caches'   # compile caches for the next box
mkdir -p "$OUT" && python3 $S/gpu.py pull <instance> '~/live-pr/results/.' "$OUT"/
python3 $S/gpu.py terminate <instance>
python3 $S/gpu.py list
```

Copy back before terminating; the box's local disk goes with it. `save_caches` overwrites the tarball with this box's caches, which are a superset of what it restored; skip it when the run was cut short before any server finished compiling. A restored Inductor cache also reuses autotune choices, so cross-restart repeats no longer sample kernel-selection variance (benchmarking skill, trap 1): that narrows the spread between arms, and both arms share it. Server logs over a few MB: copy the tail and grep hits (errors, tracebacks, recompiles), not the whole file.

## 5. Report

Write `report.md` per [references/report.md](references/report.md) into the results directory, then give the requester the verdict, the claims table and regressions table inline, and the path to the directory with the media. The report covers the PR only: setup problems and their fixes go in `harness-notes.md`, and the requester gets one line pointing at it when it is non-empty. Only post to the PR when asked, with the AI-generated header from the report reference.
