# Choosing what to run

The plan has three parts, in this order of priority. Size the hardware for the largest config the plan needs, so every arm of every A/B runs in one allocation.

1. **The PR's claims.** Every behavior or number the description asserts gets a live check that would fail if the claim were false. A "fixes X" PR reproduces X on `main` first, then shows it gone on the PR.
2. **Regression suites** for what the touched paths can break, from the table below.
3. **Edge cases specific to the diff**: the inputs and request mixes the changed code branches on (empty/long prompts, mixed modalities in one batch, cancellation mid-stream, eviction under KV pressure, TP rank disagreement, the first request of a new shape). Read the diff for these; they vary per PR.

## Touched paths → regression suites

| Touched | Suites |
| --- | --- |
| `mstar/model/<family>/` only | that family's functional suite and its perf suite at its usual concurrency, PR vs main |
| new model (new `mstar/model/<family>/`, new `DEFAULT_CONFIGS` entry) | functional suite with media for every request type it serves; competitor benchmark per the benchmarking skill ("Profiling a gap against another engine") |
| `mstar/worker/`, `mstar/engine/engine.py`, `mstar/engine/cuda_graph_runner.py`, `mstar/graph/`, `mstar/conductor/` | **worker A/B set** below |
| `mstar/engine/resources/<kind>` | the worker A/B rows whose models declare that resource (grep `get_node_resources` for the class), plus the PR's own model |
| `mstar/distributed/`, `mstar/model/components/distributed/`, anything TP/SP | the TP rows of the worker A/B set |
| `mstar/communication/`, `rust/` | worker A/B set, plus a functional smoke over `--tensor-comm-protocol TCP` (untimed) and one with `MSTAR_RUST_ZMQ=0 MSTAR_RUST_GRAPH=0` (Python runtime) so both implementations are exercised (AGENTS.md invariant 5) |
| `mstar/api_server/` | functional suite for each modality family it routes (text, image, speech, video), including streaming and the OpenAI routes the change touches; the CPU-overhead row if it sits on the request path |
| `benchmark/`, `docs/`, `test/` only | no GPU run needed unless the PR claims a number; say so and stop before provisioning |

## Worker A/B set

PR vs `main`, back to back per model, two or more server restarts per arm (benchmarking skill: "Repeats must span server restarts"). Pick the rows the change can plausibly reach; a scheduler change reaches all of them.

| Row | Server | Client | Concurrency | GPUs |
| --- | --- | --- | --- | --- |
| BAGEL CFG parallel | `mstar serve bagel_cfg_parallel` (`configs/bagel_cfg_parallel.yaml`) | `--model bagel --request-type text_to_image` | 1, 4 | 3 |
| Cosmos3 TP/SP | `mstar serve cosmos3 --config configs/cosmos3_nano_tp2_sp2.yaml` (also `_tp2` / `_sp2` when 4 GPUs are not needed elsewhere) | `benchmark/cosmos3/` scripts | 1 | 4 |
| Qwen3-Omni a2s + t2s, thinker TP | `mstar serve qwen3_omni --config configs/qwen3omni_thinker_tp2.yaml` | `--model qwen3omni --request-type audio_to_speech`, then `text_to_speech` | 1, 8, 32 | 3 |
| Qwen3.5 0.8B CPU overhead | `mstar serve qwen3_5_0.8b` | `--model qwen3.5 --model-id Qwen/Qwen3.5-0.8B --request-type text_to_text --ignore-eos --output-len-min/max` | 8, 32, **128** | 1 |
| Orpheus TP | `mstar serve orpheus --config configs/orpheus_tp2.yaml` | `--model orpheus --request-type text_to_speech` | 1, 8, 32 | 2 |

Use the Qwen3.5 0.8B row for anything that could add host work on the critical path (per-step Python, extra syncs, serialization, scheduler bookkeeping). It is the most host-bound model in the repo, so it shows CPU overhead first. Run `benchmark.worker_phases` on both arms when it moves, to say whether the change is in `event_sync` (GPU) or host phases.

- **Run it for every PR that touches the worker, engine, resources or graph runtime, correctness fixes included.** A fix that adds bookkeeping to admit, plan, commit or the scheduling pass can cost host time that no functional check shows. Run it at c=8/32/128 by default (add 1 or 64 if a number there is in question); c=32 and c=128 are where host overhead appears, so they are never dropped.
- **Skip it only for a change that is compute-neutral on the per-step path**: per-request bookkeeping in the API server or data worker (a counter per request, a queue drained once per submit), startup-only code, or error paths a normal request never reaches. Read the diff to decide, and say in one clause of the report why the row was skipped.

Prefer Orpheus TP over Qwen3-Omni full TP when the question is TP itself and GPUs are scarce: it is two GPUs and quicker to start.

Verify the exact config, client name and flags against the PR's checkout before running: configs move between branches (benchmarking skill: "A/B a branch against main"), and server and client model names differ.

## Server settings for every run

- `MSTAR_SHM_ARENA=1` (set in `env.sh`; off by default otherwise).
- `MSTAR_TP_ASYNC_SCHED=1` whenever the config has `tp_size` or `sp_size` > 1. It must match on every rank, which it does when set on the `mstar serve` process.
- The Rust extension built and active: check for `control mesh transport: rust` and `graph runtime: rust` in each server log, and record them in the report.
- Distinct `--port` and `--socket-path-prefix` per concurrently running server; start and stop everything through the benchmarking skill's `scripts/server.sh`.
- Readiness timeout sized for the slowest cold compile in the plan (qwen3-omni's first Inductor compile has exceeded 15 minutes): `--timeout 2400`.

## Concurrency

- Text and speech output: 1, 8, 32 (`closed_loop`, at least 5× the concurrency in requests, warmup counted in waves).
- Qwen3.5 0.8B CPU-overhead row: 8, 32, 128.
- Image and video: 1 (and 4 for BAGEL), unless the PR is about batching or concurrency for those modalities, in which case sweep what it claims.
- `--ignore-eos` with fixed output lengths for text, except Qwen3-Omni (its talker misbehaves past EOS).

## Media

Save outputs for every functional check that produces them, so they can be reviewed by ear and eye:

- `benchmark.runner --output-dir` writes `req_<id>.{txt,png,wav}`; copy a handful per cell, not all of them, into `results/media/<suite>/<arm>/`.
- `MStarClient`: `tts(...).to_wav(path)`, `generate_image(...)` returns PNG bytes, `GenerateResult.save_image` / `save_audio`. Video has no helper: write the mp4 bytes from the `result.raw` entries whose modality is `video`.
- Name files so the PR/main pair sits side by side: `<suite>/<prompt-id>.<arm>.<ext>`, with the prompt text in `<suite>/prompts.txt`.
- For a PR that changes a numerical path, put PR and main outputs for the same prompt and seed next to each other, and include a main-vs-main pair from two server processes as the noise floor (benchmarking skill: "Output-equivalence checks need a same-code control").

## Hardware sizing

- GPU count = the largest `ranks` + 1 among the configs in the plan (respecting `rank_devices`). Most models fit on one 80 GB GPU, so 80 GB-class Hopper is the floor; ask for more VRAM only when a config needs it (e.g. `cosmos3_super` at TP2).
- Hopper or newer only: H100 (PCIe/SXM/NVL), H200, GH200, B200, B300.
- When GPU count is driven by one optional row (say the 4-GPU Cosmos TP2+SP2), check whether the 2-GPU `_tp2` and `_sp2` configs cover the PR's risk. If they do, take the cheaper box and say in the report which row you substituted.
