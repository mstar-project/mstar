# LoRA checks on a GPU

Nothing here runs in CI. Sections 1-3 fit on one GPU box with Qwen3.5-0.8B
(an H100 is plenty; any CUDA GPU with Triton works) and the HF cache warm; section
4 needs two GPUs.

`peft` must be installed for the HF reference in section 2.

## 1. Kernels, eagerly and inside one captured graph (minutes)

```
pytest test/integration/test_lora_triton_gpu.py -v
```

* `test_bf16_deltas_against_fp32`: every Qwen3.5 target's delta against an fp32
  per-token reference, decode to a 513-token prefill, mixed adapters and base rows
  (left bitwise untouched).
* `test_delta_net_in_proj_rank_shards[2|4]`: one rank's delta-net in-projection
  delta under TP (six fused slices, `a` padded per rank, the padding left zero).
* `test_lm_head_rows_per_request_against_fp32[1|4]`: the LM head's delta, one row
  per request, over the full 248320-wide vocabulary (one rank's columns at TP4).
* `test_one_capture_serves_every_adapter_mix[layers|layers-compiled|lm_head]`:
  captures the deltas once (every row on the zero adapter, as the graph runner
  captures), then replays for four mixes that move adapters between slots (an
  eviction, a reload and a base-only batch included). Each replay must equal the
  same rows run eagerly bit for bit, leave the bucket's padding untouched and match
  fp32 to bf16 accuracy. `layers-compiled` captures the Inductor-compiled function
  (`max-autotune-no-cudagraphs`), as the runner does; `lm_head` has one row per
  request.
* `test_an_fp32_input_under_autocast_takes_the_weights_dtype[False|True]`: an fp32
  `LayerNorm` output under autocast is cast to the slot dtype, eagerly and captured.
* `test_kernels_match_the_reference[...]`: `lora_apply_` (split-K, ranks 16 to 512
  including tiled ranks past 64, token tails from 1 to 1500) and the embedding's
  `lora_gather_shrink` + `lora_expand_` (ids off the vocabulary) against the fp32
  reference ops, with a rank-0 slot and base rows left untouched.
* `test_a_tiled_rank_inside_a_captured_graph`: `lora_apply_` at max rank 128
  captured once and replayed for changing slot mixes and ranks, bit for bit against
  eager.

The CPU tests the GPU box should also run (they need no GPU, but a laptop may be too
slow for the 27B meta-device build):

```
pytest test/modular/test_lora_models.py test/modular/test_lora_targets.py \
    test/modular/test_lora_pool.py test/modular/test_lora_tp_lockstep.py \
    test/modular/test_lora_api.py test/modular/test_lora_graphs.py -v
```

## 2. Served output: captured vs eager vs HF + PEFT (about 20 minutes)

```
python -m benchmark.lora.qwen3_5_parity --model qwen3_5_0.8b --out ~/lora_parity
```

The seven default adapters include three on the gated delta-net projections (two
PEFT `all-linear`) and one (`agnews`, unsloth) on the vision tower's blocks and
merger; see the script's docstring. Every adapter runs six text prompts and two
image prompts (`img0`, `img1`; `--no-images` drops them), so a vision adapter's delta
reaches the comparison. The rest of the model, one size at a time (the same phases;
each needs the size's weights, and `--out` per size):

```
# the conv patch embed (comma), and a 2B in the released layout
python -m benchmark.lora.qwen3_5_parity --model qwen3_5_2b --out ~/lora_parity_2b
# the LM head and embedding (pirate: PEFT lm_head + embed_tokens; clausetagger:
# lm_head), Tinker's split delta-net q/k/v and `unembed_tokens` (oneinbox), tied weights
python -m benchmark.lora.qwen3_5_parity --model qwen3_5_4b --out ~/lora_parity_4b
# 9B (untied head): nimble, Tinker (rlvr), a vision adapter (mkd2world); one H100
python -m benchmark.lora.qwen3_5_parity --model qwen3_5_9b --out ~/lora_parity_9b
```

PEFT cannot load Tinker's keys into transformers' Qwen3.5 (it has `in_proj_qkv`, no
`unembed_tokens`), nor a text-only-layout adapter into the multimodal class an image
prompt needs; for those the reference merges the adapter into the HF weights by the
script's own name map (`_merge`, head untied first) and `reference.json`'s `_how`
says which reference each adapter got per prompt kind. A merged reference is not
PEFT's code, so a failure there is first a question about `_merge`.

Phases (each rerunnable alone with `--phase`): `reference` (HF + PEFT greedy, top-5
per step), `serve` (mstar as configured, so with CUDA graphs and `torch.compile`),
`serve --eager` (the same server with `benchmark/lora/eager_site` on `PYTHONPATH`,
which turns both off for the LLM node; the log is checked for it), `compare` (each
served run against the reference by vLLM's greedy rule) and `graphs` (captured
against eager, row by row: identical, or a first difference where both tokens are in
the reference's top-5).

Pass: `compare` and `graphs` exit 0. `graphs.json` lists every row with where it
diverged; a run where most rows are not token-identical deserves a look even if it
passes. Copy `~/lora_parity` off the box after each phase.

## 3. Worker phases: what LoRA costs per step (about 1.5 hours)

```
python -m benchmark.lora.phase_plan --out ~/lora_phases
```

Twelve servers, each driven closed-loop at concurrency 1, 32 and 64 with 128-token
outputs (`benchmark/lora/load.py`, which can put requests on adapters;
`benchmark.runner` cannot):

| LoRA mode | config | requests |
| --- | --- | --- |
| `off` | stock `configs/qwen3_5_0.8b.yaml` | base model |
| `base` | stock + `resources: lora:` (3 adapters, 3 slots) | base model |
| `adapters` | same | round-robin over the 3 adapters |

each `captured` (as served) and `eager` (`eager_site`), and each run twice: a
`throughput` pass with no instrumentation and a `profile` pass with
`MSTAR_PHASE_TIMING=100`, whose `phases.txt` holds the worker_phases tables
(`benchmark/worker_phases/README.md` says how to read them). Finished runs are
skipped on a rerun; `--only adapters-captured` runs one combination; `--summary`
prints tok/s and the key phase means per run and concurrency. Copy `~/lora_phases`
off the box after each run.

What to read:

* `base-captured` against `off-captured` at c=32/64: the price of turning LoRA on
  for traffic that does not use it (the zero-adapter rows still run the kernels,
  which exit at once). Expect `res.plan.lora` in tens of microseconds and
  `worker.gpu_thread.exec` close to `off`.
* `adapters-captured` against `base-captured`: the kernels' GPU time
  (`worker.postprocess.event_sync` grows if the step is GPU-bound) and tok/s.
* `*-eager` against `*-captured`: the graphs' win with adapters, which was the
  point of capturing them. `eager` LoRA should not be faster than `captured`.

Any `failed` request in a `load_c*.json` invalidates that run.

A step none of whose rows names an adapter replays a capture with no LoRA kernel in it
(`LoraBaseOnly`, see "Capture and compile" in `docs/adding_models.rst`). On an H200 with
`compare_engines` (3 rounds, per-arm `e_off`), Qwen3.5-9B base-only traffic (GPU-bound)
kept 96.2/96.2/97.2% of its LoRA-off throughput at c=1/32/64 without it and
100.0/99.8/100.4% with it; one-adapter and mixed traffic did not move, and 0.8B
(CPU-bound) stayed within run-to-run noise. It doubles a LoRA node's captures (9B: 109
-> 218 buckets, about 3.7 GiB more, a minute more startup).

## 4. Tensor parallelism: TP2 against one GPU, and slot churn (about 30 minutes, 2 GPUs)

Needs at least two GPUs and the `reference` and captured `serve` output of
section 2 in the same `--out`. With `MSTAR_TP_ASYNC_SCHED=0` and `=1` in turn, the
same server with the LLM node at TP2 (vision encoder on rank 0):

```
python -m benchmark.lora.qwen3_5_parity --model qwen3_5_0.8b --out ~/lora_parity --phase tp --tp 2
```

(`--model qwen3_5_9b` for the 9B, whose stock config is single-GPU too; `--tp 2`
rewrites the node groups either way. `--async-sched 1` runs one setting.)

* Parity: every `serve` request against the single-GPU captured run, by the
  `graphs` rule (identical, or a first difference where both tokens are in the
  reference's top-5). Bitwise identity is not expected: the all-reduce sums the
  rank partials in a different order from one GPU's GEMM.
* Churn: on the same server, 512 requests at concurrency 64 over the base model and
  all three adapters with two slots, so slots are evicted while requests leave.
  Every rank decides its own slots; this is where a rank deciding differently would
  hang the instance or fail a request. Pass: `failed == 0`, the server log shows
  evictions, and the run finishes (each request times out at 300 s, so a hang
  surfaces as failures).

Pass: the phase exits 0 for both settings. Results: `mstar_tp2_async{0,1}.json`,
`tp_tp2_async{0,1}.json` and `server_tp2_async{0,1}.log`; copy them off after each
setting. `MSTAR_TP_ASYNC_SCHED=1` is the setting that exercises the leader's
teardowns landing behind a speculated step's admit; `=0` is today's default.

## 5. Prefix caching keyed by adapter, on Orpheus (about 15 minutes)

```
python -m benchmark.lora.prefix_cache --model orpheus --config configs/orpheus_colocated.yaml \
    --adapter a=<Tachyeon/orpheus-3b-sft-lora>/adapter_model.safetensors \
    --adapter b=<derekurban2001/orpheus-jarvis-voice-lora>/adapter_model.safetensors \
    --max-rank 64 --out ~/lora_prefix
```

Orpheus is the first model that keys a prefix-cache stream *and* serves LoRA adapters
(`get_lora_specs` through `module_lora_spec`). Requests go through `/generate` with the
adapter in `model_kwargs` (`--endpoint speech`, the default for Orpheus), and a
request's output is its audio; a text model takes `--endpoint chat`. Qwen3.5 serves
adapters but keys no stream, because its gated DeltaNet state cannot skip a cached
prefix. The CPU check that requests on an adapter key under its own root (and base
keys stay as they were) is `test/modular/test_lora_prefix_keys.py`.

On one server, one long prompt is sent greedily, one request at a time: first with
`prefix_cache: false` once per namespace as the reference, then cached in the order
base, a, b, a, base, b. The `KV ... <id>/<label> matched N of its M pages` log line
gives each cached request's hit. Pass, with exit code 0, when:

* the first request in each namespace matched 0 pages (no namespace was served another
  one's KV),
* each repeat matched more than 0 (the cache was on, so those misses mean something),
* every output equals its namespace's uncached output.

Results go to `prefix_cache.json` and `server.log` under `--out`.

The CPU tests behind it are in `test/modular/test_lora_prefix_keys.py` (a real
`KVManager` fed the preprocess worker's kwargs as the conductor hands them over) and
the Orpheus test above. The prefix cache is off under tensor
parallelism (`KVManager.enable_prefix_cache`), so there is no TP version of this phase.

## 6. Loading and unloading adapters at runtime, under base traffic (about 15 minutes)

Needs the `reference` phase of section 2 in the same `--out` (same `--max-tokens`).

```
python -m benchmark.lora.runtime_load --model qwen3_5_0.8b --out ~/lora_parity
```

One server whose `resources: lora:` block names no adapter (with `runtime_updates: true`), with 32 closed-loop
base-model requests (`--base-concurrency`) running from start to end. While they run:

* **load**: `proofreader` and `router` loaded with `POST /v1/load_lora_adapter` (their
  PEFT directories); both 200, both listed by `/v1/models` with `parent`, a second load
  of one a 400, and a request on one before its load a 400.
* **parity**: every parity prompt on each adapter against the HF + PEFT reference by
  vLLM's greedy rule, and the control that each adapter changes some output against the
  base model.
* **unload**: 8 requests of 4096 tokens (`ignore_eos`) on `proofreader`, then its unload
  while they run: 200; a new request on it 400; every held request runs to its length
  (`finish_reason` length, its text re-tokenized to at least 90% of 4096: the
  non-streaming `usage` is not filled in); a second unload 404; `/v1/models` no longer lists it.
* **reload**: `proofreader` loaded again from the same directory (retried for up to a
  minute while the workers still hold it for the requests that just ended), then the
  parity prompts again, which must equal the first load's outputs exactly.

Pass: exit 0, which includes `failed == 0` for the base traffic. Results:
`runtime_load.json` (every check, the outputs, the held requests, the base traffic's
latency overall and while each load or unload was in flight) and
`server_runtime_load.log`. Copy both off the box when it finishes.

Worth reading even on a pass: `base_traffic.during_load_*`. A worker reads the adapter
on its main loop as the load arrives, so a large adapter could stall that worker's steps
for the read's duration; for these rank-16 0.8B adapters it should be small. A bump
there is the number that would justify moving the read off the main loop.

Qwen3.5 keys no prefix-cache stream, so no adapter has a prefix-cache root in this run.
That a reload of the same weights keeps its root, and of other weights does not, is the
CPU test `test_a_runtime_load_and_unload_end_to_end` in `test/modular/test_lora_api.py`.

Under TP (two GPUs; after section 4 passes), once per `MSTAR_TP_ASYNC_SCHED` setting:

```
python -m benchmark.lora.runtime_load --model qwen3_5_0.8b --out ~/lora_parity --tp 2 --async-sched 0
python -m benchmark.lora.runtime_load --model qwen3_5_0.8b --out ~/lora_parity --tp 2 --async-sched 1
```

Loads and unloads reach each rank of the instance on its own, at whatever step it is on;
the pool is built so that this cannot change a slot decision (`LoraAdapterPool`'s
docstring). A hang (requests time out at 300 s and count as failed), a failed base
request or a parity failure here is the sign it can. Results are suffixed
`_tp2_async<0|1>`.

## 7. Against vLLM and SGLang: per-request multi-LoRA serving (about 3.5 hours, GPU 0 only)

Each engine in its own venv; record nothing by hand, the script writes every version to
`versions.json`:

```
uv venv ~/venvs/vllm --python 3.12 && uv pip install --python ~/venvs/vllm/bin/python vllm
uv venv ~/venvs/sglang --python 3.12 && uv pip install --python ~/venvs/sglang/bin/python sglang
uv pip install peft          # in mstar's own venv, for the HF + PEFT reference
```

Then, from mstar's venv (activated, so `ninja` is on `PATH`), with the Rust extension built:

```
python -m benchmark.lora.compare_engines --out ~/lora_compare --dry-run      # the plan
python -m benchmark.lora.compare_engines --out ~/lora_compare \
    --vllm-python ~/venvs/vllm/bin/python --sglang-python ~/venvs/sglang/bin/python
python -m benchmark.lora.compare_engines --out ~/lora_compare --summary      # any time
```

Stages (`--stage correctness|perf|profile`, default all, in that order; everything is
resumable, a finished cell is never rerun, `--first-round N` adds rounds):

* **correctness** (about 35 min): section 2's HF + PEFT reference, then one server per
  engine answering the parity prompts greedily on the base model and each adapter, one at
  a time and all at once, scored by the same top-k rule (`qwen3_5_parity._agree`), with
  the control that each adapter changes some answer. The table puts the engines side by
  side, and `correctness/scores.json` lists the rows they judge differently. The same
  again for the all-linear adapters below, on the `alllinear` server, against their own
  reference (the same `qwen3_5_parity --phase reference`, given those adapters) under
  `correctness/alllinear/`. An adapter an engine rejects, or would apply only part of, is
  skipped for that engine with the reason.
* **perf** (about 2.5 h): 3 rounds, every server restarted per round, engine order rotated.
  Per engine, four servers: `off` (no LoRA), `fit` (4 slots, all three adapters resident),
  `churn` (2 slots) and `alllinear` (4 slots, the all-linear adapters). Scenarios `a_base`
  (LoRA on, base requests), `b_one` (one adapter), `c_mixed` (three adapters round-robin),
  `d_churn` (the same over 2 slots), `e_off` (LoRA off, the anchor), `f_all_linear_one`
  (every request on `transaction`) and `g_all_linear_mixed` (`transaction`, `atemoko` and
  the base round-robin), each at concurrency 1, 32 and 64: closed loop, streamed, 5
  warmup waves before a server's first cell and 1 before the others, `ignore_eos` with a
  seeded output length (256-512) that is the same per request on every engine. Cells under
  `cells/<scenario>/<engine>/c<c>_round<r>/result.json`. A cell counts only with
  `failed == 0`. Every engine's requests are credited alike, with the tokens their
  streamed text re-tokenizes to (mstar streams no usage); a request fails if that is
  under 90% of its `max_tokens`, or if the engine reports a `completion_tokens` other
  than its `max_tokens`.
* **profile** (about 10 min): mstar only, untimed, `MSTAR_PHASE_TIMING=100`, each scenario
  at concurrency 1 and 64, rendered with `benchmark/worker_phases` into
  `profile/<scenario>_c<c>.txt`.

Matched settings: bf16, max rank the loaded adapters' largest `r` (16 for the defaults), the same slot count (mstar `max_slots: S`, vLLM
`--max-loras S`, SGLang `--max-loras-per-batch S+1`, since SGLang counts base-model rows as
one of a batch's adapters), every adapter in host memory (vLLM `--max-cpu-loras`), prefix
caching off (mstar's Qwen3.5 has none; vLLM `--no-enable-prefix-caching`, SGLang
`--disable-radix-cache`), context length and running requests from mstar's config
(32768, 128), fp32 GDN state, CUDA graphs on (mstar with `MSTAR_REQUIRE_CUDA_GRAPHS=1`),
thinking off, greedy. SGLang's `--lora-target-modules` is the adapters' own union, which
must be attention and MLP only. Adapters are selected by the OpenAI `model` field: the
adapter's name on mstar and vLLM, `<base>:<adapter>` on SGLang (`--sglang-adapter-model`
changes it). Extra competitor flags for tuning: `--vllm-arg`, `--sglang-arg`.

An engine that fails to start, or rejects LoRA or one adapter, is recorded in
`support/<engine>_<kind>.json` and shown as not supported with an excerpt of its error; the
run goes on without it (`--retry-unsupported` tries it again).

**All-linear adapters** (`f_all_linear_one`, `g_all_linear_mixed`): `transaction`
(indrapurnayasa/transaction-qwen3.5-0.8b-lora, PEFT all-linear, rank 16, text-only naming),
`atemoko` (BurnyCoder/qwen3.5-0.8b-atemokoloporos-positive-expanded, all-linear, rank 16,
released naming) and, for correctness only, `medvqa`
(thealper2/qwen3.5-0.8b-slake-medvqa-lora, rank 8, `in_proj_qkv`, `in_proj_z`, `out_proj`).
Besides attention and MLP they touch the gated delta-net's `in_proj_qkv`, `in_proj_z`,
`in_proj_a`, `in_proj_b` and `out_proj`. `--all-linear-adapter NAME=REPO` (repeatable) and
`--all-linear-perf-adapters a,b` change them.

* mstar serves them only with LoRA on the delta-net (`lora/gdn` or later). On an earlier
  branch it refuses each adapter (`LoraSpec.check_adapter`), and (f), (g) and its
  all-linear correctness rows read "mstar: not supported on this branch".
* Before anything starts, the driver reads each adapter's `target_modules` and the
  modules its file holds, and logs what each engine will apply
  (`support/adapter_modules.json`). mstar and vLLM refuse an adapter with a module they
  have no target for, so they apply all of it or none. SGLang gets
  `--lora-target-modules` as the adapters' modules in its names (`in_proj_qkv`/`in_proj_z`
  -> `in_proj_qkvz`, `in_proj_a`/`in_proj_b` -> `in_proj_ba`, ...), but its Qwen3.5
  `supported_lora_modules` has no `in_proj_ba` (`--sglang-lora-modules` if a newer SGLang
  adds it). An adapter with a module SGLang has no target for would be refused or
  silently cut down, so SGLang is **excluded** from every cell and correctness row on that
  adapter, shown as `excl.` with the reason, rather than timed doing less work. With the
  defaults that excludes SGLang from (f) and (g), and leaves it `medvqa`'s correctness
  rows, if it starts at all with `in_proj_ba` among its targets (a refusal is recorded as
  not supported; `--sglang-alllinear-target-modules` overrides the list).

`summary.md`: per scenario and concurrency, each engine's tok/s (mean ± sd over rounds),
TTFT p50, TPOT p50 and its overhead against its own `e_off`; which modules of each
all-linear adapter each engine applied; then every metric (req/s, p99s, failed), the
correctness tables, what was not supported or excluded, mstar's worker phases
and the versions. Copy `~/lora_compare` off the box after each stage.

## Behaviour changes for non-LoRA deployments

The LoRA stack changes three things a deployment without a `resources: lora:` block can
see. Each is deliberate:

* **`autocast_dtype` is parsed.** A string in the serving config (`float16`,
  `torch.bfloat16`) becomes the `torch.dtype` it names before any node is built; a value
  that names no dtype fails at startup instead of reaching the engine as a string.
  `autocast_dtype: null` is passed through as None, as before.
* **The conductor validates requests against the deployment's specs.** It resolves
  resource specs as the workers do (`serving_resources`: the model's declaration with the
  YAML's `resources:` overrides applied), where it used to validate against the model's
  defaults. A request is now checked against the limits the workers actually built.
* **`lora_adapter` on a deployment without LoRA is a 400.** A request naming an adapter
  (`model_kwargs: {lora_adapter: ...}` on `/generate`) is refused rather than run on the
  base model, which would answer it with weights it did not ask for. The OpenAI `model`
  field is unaffected: without LoRA it is not read, as before.

## LoRA from a model's modules (other models than Qwen3.5)

The CPU tests first (no GPU; minutes):

```
pytest test/modular/test_lora_models.py test/modular/test_lora_targets.py -v
```

Real weights, one GPU, per model (`module_parity`; every enabled model is CPU-validated
and GPU-pending until these pass):

* `merged`: the node's own stack and `declare_step`, the Triton kernels in bf16, every
  adapter and the base model in one mixed step, against each adapter merged into the
  weights;
* `hf`: the same against transformers + PEFT where transformers has the node (Orpheus,
  Higgs Audio's Qwen3 LLM, the Qwen3-Omni Thinker, Chatterbox T3's Llama backbone), RoPE
  as the reference computes it;
* `serve`: the stock config plus a `lora:` block, captured and compiled as shipped, a
  mixed batch included, against a server on a checkpoint with the adapter merged in.

```
python -m benchmark.lora.module_parity --model orpheus --out ~/lora_module_parity/orpheus
python -m benchmark.lora.module_parity --model chatterbox --out ~/lora_module_parity/chatterbox
python -m benchmark.lora.module_parity --model qwen3_tts --out ~/lora_module_parity/qwen3_tts
python -m benchmark.lora.module_parity --model higgs_audio --audio clip.wav --out ~/lora_module_parity/higgs
python -m benchmark.lora.module_parity --model qwen3_omni --out ~/lora_module_parity/omni  # ~70 GB
```

Orpheus's default adapters are four Hub PEFT adapters (rank 16 to 64, one rsLoRA); the
others have none known, so random adapters of the real shapes are written
(`--synthetic N`).

No cost without a `lora:` block (`non_lora_ab`): Qwen3.5-0.8B, Orpheus and Kokoro
served from this branch and from a base, alternating per round, each run with the
worker phase tables:

```
python -m benchmark.lora.non_lora_ab --phase setup --work ~/ab --base 44ab46571 --branch lora/generic
python -m benchmark.lora.non_lora_ab --phase run --work ~/ab --out ~/ab/results --rounds 3
python -m benchmark.lora.non_lora_ab --phase compare --out ~/ab/results
```
