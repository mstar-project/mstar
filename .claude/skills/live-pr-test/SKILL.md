---
name: live-pr-test
description: Live-test an M* pull request on real GPUs — read the PR, size the cheapest Hopper-or-newer box for it, provision it on RunPod or Lambda (or use the persistent cluster), test the PR's claims plus regressions against main, copy media back, tear the box down, and report in markdown. Use when asked to live-test, GPU-test, smoke-test or benchmark a PR, or to "run the PR on a real box".
---

# Live-testing a PR

The job: say whether a PR does what it claims on real hardware, and whether it breaks or slows anything else, with evidence the requester can check (numbers, logs, and media to look at and listen to).

Read alongside this:

- [benchmarking](../benchmarking/SKILL.md) for every perf number and every server start/stop. Its traps apply unchanged, especially "Repeats must span server restarts" and "Report a delta only from arms run back to back in one allocation".
- [AGENTS.md](../../../AGENTS.md) for the invariants a PR can break: invariant 4 (one model's change must not degrade others) and invariant 5 (Python/Rust drift) drive most regression choices.
- [references/test-matrix.md](references/test-matrix.md) for touched paths → suites, the worker A/B set, concurrency and sizing.
- [references/report.md](references/report.md) for the report and the PR-comment rules.

## Modes

- **cloud** (default): provision with `scripts/gpu.py`, work over ssh, copy results back with scp, terminate.
- **cluster** (`cluster` in the request, or `mode: cluster` in the config): no provisioning, ssh or scp. Run on the cluster directly and write results to `cluster.results_dir/pr-<n>-<timestamp>/`, which the requester opens in VS Code. Everything else is the same.

Settings live in `config.local.json` next to this file (gitignored; copy [config.example.json](config.example.json)). API keys come from `RUNPOD_API_KEY` and `LAMBDA_API_KEY` in the environment. Never print them, write them to the box, or put them in a command line that gets logged.

## 1. Read the PR and plan

```bash
gh pr view <n> --json number,title,body,author,headRefName,headRefOid,baseRefName,isCrossRepository,files,additions,deletions
gh api repos/mstar-project/mstar/pulls/<n> --jq .author_association
gh pr diff <n>
```

- **Trust check first.** The box runs the PR's code with the team's persistent storage mounted (HF cache, possibly an HF token). If `author_association` is not `OWNER`, `MEMBER` or `COLLABORATOR`, stop and ask before provisioning; mention anything in the diff that reaches the network, the filesystem outside the repo, or environment variables.
- A PR based on another feature branch (stacked) is compared against its base branch, not `main`. Say so in the plan.
- Extract the claims: every behavior or number the description asserts. Each becomes a check.
- Map touched files to suites with the test matrix. Add the edge cases the diff branches on.
- Size the box: GPU count from the largest config in the plan, 80 GB-class Hopper or newer.
- If nothing needs a GPU (docs, CPU-only tests, benchmark scripts without a claimed number), say so and stop.

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
- **Budget:** `budget.max_usd_per_hour` and `budget.max_hours` from the config. `launch` refuses an offer above the hourly cap, and arms a watchdog that terminates the instance at the deadline even if this session dies. The watchdog runs on the machine that launched it, so a laptop that sleeps or shuts down stops it: when this session ends early, `gpu.py list --all` and terminate by hand. To go over either cap, ask first; `gpu.py extend <instance> --hours H` moves the deadline.
- **Always terminate**, on success and on failure, after the copy-back. Before ending the session, run `gpu.py list` and confirm nothing launched for this PR is still up.

Then set up the box:

```bash
python3 $S/gpu.py push <instance> $S/bootstrap.sh '~/'
python3 $S/gpu.py push <instance> .claude/skills/benchmarking/scripts/server.sh '~/'
python3 $S/gpu.py ssh <instance> 'bash ~/bootstrap.sh <n> --hf-home <hf_home from wait> > ~/bootstrap.log 2>&1; tail -20 ~/bootstrap.log'
```

`bootstrap.sh` builds two arms, `~/live-pr/pr` and `~/live-pr/main`, each with its own venv and its own `mstar_rust` build, and writes `~/live-pr/env.sh` (HF home, local caches, `MSTAR_SHM_ARENA=1`, and an `arm pr|main` helper). It takes a few minutes. Pass `--base <branch>` for a stacked PR, and `--no-main` when nothing in the plan compares against the base. Read `results/env.txt`: a non-empty dependency diff between the arms has to be explained before any A/B number means anything.

In cluster mode, run the same script on the cluster with `--workdir` and `--hf-home` from the config.

## 3. Run

Start every server through the benchmarking skill's `server.sh` (copied to `~/server.sh`, so both arms use the same one even when a PR predates or edits it), from the arm's checkout, with the arm's venv active:

```bash
source ~/live-pr/env.sh && arm pr
MSTAR_TP_ASYNC_SCHED=1 ~/server.sh start pr 8100 --timeout 2400 \
  --log $LIVE_RESULTS/logs/<suite>.pr.r1.log -- \
  mstar serve <model> --config configs/<cfg>.yaml --gpus 0,1,2 --port 8100 --socket-path-prefix /tmp/mstar_pr
```

- Run anything longer than a couple of minutes detached on the box (`nohup ... > log 2>&1 &`, or tmux if present), then poll its log. An ssh drop must not kill a sweep.
- Order: functional checks of the claims first, cheapest suites before expensive ones, so an early failure is found before an hour of benchmarks.
- Each A/B model: main and PR back to back, restart the server between repeats, then the next model. When nothing is being timed, the two arms may run in parallel on disjoint GPUs.
- `MSTAR_TP_ASYNC_SCHED=1` on any TP/SP config; `MSTAR_SHM_ARENA=1` always (from `env.sh`). Check each server log for `control mesh transport: rust` and `graph runtime: rust`.
- One `--output-dir` per cell under `$LIVE_RESULTS/bench/<suite>/<arm>/c<conc>.r<rep>/`, stdout tee'd to `bench.log` inside it. Check `failed` in every `results.json` before using a number.
- Save media as the test matrix describes. Keep it small: a few representative outputs per suite and arm.
- If a server fails to start or every request errors on **both** arms, it is an environment problem, not a regression. Fix it or record it under "Not tested", and keep going with the rest of the plan.
- Watch the clock against the deadline. If the plan won't fit, cut the lowest-priority suites (test matrix order) and record them as not tested.

## 4. Collect and tear down

```bash
OUT=$(python3 -c 'import json,os;print(os.path.expanduser(json.load(open(".claude/skills/live-pr-test/config.local.json"))["results_dir"]))')/pr-<n>-$(date +%Y%m%d-%H%M)
mkdir -p "$OUT" && python3 $S/gpu.py pull <instance> '~/live-pr/results/.' "$OUT"/
python3 $S/gpu.py terminate <instance>
python3 $S/gpu.py list
```

Copy back before terminating; the box's local disk goes with it. Server logs over a few MB: copy the tail and grep hits (errors, tracebacks, recompiles), not the whole file.

## 5. Report

Write `report.md` per [references/report.md](references/report.md) into the results directory, then give the requester the verdict, the claims table and regressions table inline, and the path to the directory with the media. Only post to the PR when asked, with the AI-generated header from the report reference.
