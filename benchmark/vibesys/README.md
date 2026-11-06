# An optimization loop for M*, after VibeSys

This runs Claude Code, headless, through a brief for days on one GPU pod. The brief (`brief.md` is the template)
has the agent follow the VibeSys round: one falsifiable hypothesis, one change, a correctness check, an A/B of the
base and the candidate, and a keep decision by a fixed rule, which `python -m benchmark.vibesys keep` applies to the
round's rows. Kept changes stack on a local branch, and every few hours the agent runs a checkpoint sweep of the
starting commit against the best one on every workload of the brief.

It differs from VibeSys in what runs it. One agent works through the brief and has a subagent review each diff, in
place of separate orchestrator, implementer, judge and profiler roles. It runs on the pod directly, without
VibeSys's Docker sandbox. The agent's Edit and Write tools reach `/workspace` but not the harness, the loop's
state or the results file, and `git push`, `gh` and the web are denied. Bash is allowed, so this guards against
mistakes and is not a sandbox. A separate job pushes the agent's branches to a fork.

| File | What it does |
|---|---|
| `pod_boot.sh` | Prepares a pod (tools, the repo, a venv with M*), then starts the loop and the push job. |
| `loop.sh` | Sends the brief to Claude Code and resumes its conversation, call after call, until a deadline or a STOP file. |
| `brief.md` | The template of a brief: target, allowed paths, review rules, the round, reporting and commits. |
| `rows.py` | The result row schema, the keep rule and the checkpoint table. |
| `progress.py` | The best-so-far chart of a loop's headline cell. |
| `cli.py` | `python -m benchmark.vibesys`, with `report`, `keep`, `table` and `chart`. On a pod it is `vibesys`. |

## Start a pod

On RunPod, one H200 with the `runpod/pytorch` image, a volume at `/workspace` and port 22. Set the variables
`pod_boot.sh` lists at its top, with the boot script and the brief encoded as

```bash
BOOT_B64=$(base64 -w0 benchmark/vibesys/pod_boot.sh)
BRIEF_GZB64=$(gzip -9nc my_brief.md | base64 -w0)
```

and the container start command

```bash
bash -c 'echo "$BOOT_B64" | base64 -d >/root/boot.sh; (bash /root/boot.sh 2>&1 | tee -a /workspace/boot.log) &
  if [ -x /start.sh ]; then exec /start.sh; else sleep infinity; fi'
```

The first boot ends with `READY, waiting for a Claude login`. Run `claude auth login` over SSH and restart the
pod. The login is kept on the volume.

## Watch and steer

The pod log carries a `[boot]` line per step, a `[loop]` line per Claude Code call, a `[result <pod>]` line per
recorded row and a `[push]` line when a push fails. `tmux attach -t loop` shows the loop live.

- **Amend the brief:** add text at the end of the brief and set the new `BRIEF_GZB64`. The pod restarts, and the
  next call resumes the conversation with the addition in its prompt.
- **Replace the brief:** any other edit starts a fresh conversation, and the old brief is kept in `/workspace/loop`
  as `brief.prev-<date>.md`.
- **Stop:** `touch /workspace/loop/STOP` ends the loop after the current call. Restarting with a changed brief clears it.
- **Run longer:** `LOOP_DEADLINE` (epoch seconds) replaces the default of 72 hours after the first call.

The harness is copied into `/workspace/vibesys` on the first boot only, so the row schema stays the one the results
file was written with. Delete that folder to take a newer `HARNESS_REF`.

## Read the results

Every round of every arm is a row in `/workspace/results.jsonl`. The fork's `perf/<pod>` branches carry the code,
and the agent's results file under `/workspace/results/` carries the commands and decisions.

```bash
python -m benchmark.vibesys keep results.jsonl --hypothesis '<hypothesis>' --model Qwen3.5-0.8B --workload ShareGPT --c 64
python -m benchmark.vibesys table results.jsonl --batch h200_1106 --hypothesis 'checkpoint: run-base vs 61b95ae05'
python -m benchmark.vibesys chart results.jsonl --model Qwen3.5-0.8B --workload ShareGPT --c 64 --out progress.png
```

The chart counts a round as kept when it passes the keep rule on the charted cell, so it is drawn for the loop's
headline cell. A round dropped at review still steps the line up, until the next checkpoint sweep scales the
segment to what it measured.
