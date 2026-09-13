# Multi-stage / streaming models, the serving stack, and debugging hangs

Read this once your model has more than one autoregressive stage, streams output as it
generates, or you're bringing up `mstar-serve`. It layers on top of `model-contract.md`.
The canonical patterns to copy live in `mstar/model/qwen3_omni/` (encoder → LLM → talker →
vocoder, streaming) and `mstar/model/orpheus/` (LLM → codec, streaming). Read their
`get_graph_walk_graphs`, `get_partitions`, `get_partition_topology`, and the conductor
state machines (`_get_*_forward`) — do not invent these from scratch.

## How the engine drives your submodule (mental model)

- The **conductor** owns request scheduling and the walk state machine; **workers** own GPUs
  and run **engines** (one `KV_CACHE` engine, one `STATELESS` engine per flavor). Each engine
  drives your submodule as `prepare_inputs → preprocess → forward (→ postprocess, check_stop)`.
- **Prefill vs decode is just the `graph_walk` string** passed to every method — branch on it;
  it is not a flag the engine interprets.
- The **cache handle is `engine_inputs.cache_manager`** — there is no separate argument. On a
  `STATELESS` engine it is `None` (and so is the sampler). `preprocess` plans attention
  (`cache_manager.plan_attention(seq_lens, ...)`, `plan_rope(...)`); `forward` computes
  (`cache_manager.run_attention(q, k, v, layer_idx)`).
- **Sampling**: return `{"logits": [...]}` and let the engine sample, OR sample inside `forward`
  via `engine_inputs.sampler.sample(...)`. A second token channel (residual codebooks, a function
  channel) is declared by the model's `get_aux_sampling_configs` and sampled with
  `sampler.sample_aux("<label>", request_ids, logits)`. The aux label *set* must be static.
- **Custom per-request state** (recurrent state, a talker's own KV, a diffusion scheduler) lives
  in `engine_inputs.per_request_states[rid]` (a `PerRequestState`; `.add`/`.get`). The engine
  injects the batch's states and drops them on request removal. It is `None` during CUDA-graph
  capture — which is why nodes holding such state set `disable_torch_compile = True` (eager).
- **Batching**: `can_batch(batch, inputs) -> len(inputs) > 1` + `forward_batched` returning
  `{request_id: NameToTensorList}`. Read each request's last-token output at its packed offset
  (`itertools.accumulate(seq_lens)`).

## Async partitions (multi-stage streaming)

When stage B must start consuming stage A's output *before* A finishes (LLM streaming tokens to
a talker/vocoder), the stages run as **partitions** scheduled independently:

- **`get_partitions()`** → one `PartitionDefinition` per stage: `name`, `graph_walks`,
  `initial_walk`, and `producer_partitions` (who it consumes from). The producer has
  `producer_partitions=[]`; consumers name their upstream.
- **Cross-partition edges are `StreamingGraphEdge(next_node=…, name=…, target_partition=…)`.**
  The producer emits them unaware; on the consumer they land in a **StreamBuffer** gated by a
  **chunk policy**.
- **`get_partition_topology()`** → `PartitionTopology(partitions=[…], connections=[Connection(
  from_partition, to_partition, edge_name, chunk_policy_factory)])`. `FixedChunkPolicy(chunk_size=1)`
  delivers one token/frame per consumer step; `LeftContextChunkPolicy(chunk, left_context)` gives a
  causal decoder its overlap so it can decode a chunk with enough left context and emit only the new tail.
- **The per-partition state machine** (`get_initial_forward_pass_args` +
  `get_partition_forward_pass_args`) is called by the conductor per partition step. The producer
  transitions prefill→decode and finishes on EOS; a **consumer partition gates on its stream** and
  finishes when the upstream is done. Mirror `qwen3_omni._get_talker_forward` / `_get_code2wav_forward`.

**The streamed tensor arrives via the StreamBuffer, NOT via the forward-pass-args `inputs`.** The
conductor's forward-pass-args supply only the *non-streamed* / self-fed inputs (the fed-back token,
a trigger). A decode loop that also needs a streamed input lists it in the node's `input_names` and
gets it from the buffer; **all of `input_names` must be satisfied for the node to run** — which is
the classic hang: on iteration 0 a fed-back input has no prior value, so seed it.

**Loop termination** is the worker-side `Loop`'s `check_stop`/`max_iters`, not the conductor. The
consumer partition's `get_partition_forward_pass_args` should return `request_done=True` when the
loop returns control — don't invent a separate stream-exhaustion condition unless the reference does.

**Multiple output modalities**: give each producing edge an `output_modality` (`"text"`/`"audio"`)
pointing at `EMIT_TO_CLIENT`, and branch on `modality` in `Model.postprocess`.

## Verify the graph without a GPU

Before any live run, these resolve every edge route and cross-partition connection on CPU:
- `model.get_graph_walk_graphs()` / `get_partitions()` / `get_partition_topology()` construct cleanly.
- `model.get_worker_graphs(config_path)` derives a worker graph for **every** walk — a dangling edge
  name or unresolved partition raises here.
- Constructing a `Conductor(model, model_config_file, socket_path_prefix=…)` derives the full
  per-worker topology (worker ids, per-worker graphs) — one level deeper than `get_worker_graphs`.
- Wrap the above in a `test/modular/test_<name>_model.py` (build the model with
  `object.__new__(<Model>)` + a config, no weights) so it's CI-checked.

## The live serving stack

Request flow: HTTP `POST /generate` → **data worker** loads media (`model.load_<modality>`) and
calls `model.process_prompt(...)` → tensors are registered and handed to the **conductor** →
partitions execute on **workers** → outputs stream back tagged by modality → `model.postprocess`.

Launch:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  mstar-serve --config configs/<name>.yaml --host 127.0.0.1 --port 8000 \
  --tensor-comm-protocol SHM --socket-path-prefix /tmp/mstar_<name>/ --upload-dir /tmp/mstar_up_<name>/
```
Client: `from mstar.client.client import MStarClient; MStarClient("http://127.0.0.1:8000").generate(
audio="x.wav", input_modalities=("audio",), output_modalities=("text","audio"), temperature=0.0)`.

### First-run gotchas (each of these cost real debugging time)
- **`import mstar` must resolve your checkout.** The installed `mstar-serve` console script imports
  the *main* checkout; a model added in a worktree/branch won't be in its registry → `Unknown model`.
  Set `PYTHONPATH=$PWD` (your checkout) so your code wins.
- **`mstar serve <name>`** (the quickstart wrapper) validates against a hardcoded allow-list and will
  reject a new model. Use the low-level **`mstar-serve --config <yaml>`**.
- **Tensor transport defaults to RDMA/Mooncake**, which fails to register memory without active
  InfiniBand (`mlx5 … not active` in the log). On a single node use `--tensor-comm-protocol SHM`.
- **`buffered `conda run`/`nohup` hide startup logs.** Launch the env binary directly with
  `PYTHONUNBUFFERED=1` so you can watch weight-loading + warmup and catch errors live.
- **Media ingestion** calls `model.load_audio/load_image/load_video`; the base `load_audio` uses
  `torchcodec` (fragile native libs). Override it to decode via `soundfile`/PIL and return the same
  `TensorAndMetadata` shape.
- **Media key naming.** The data worker stores loaded media under `f"{modality}_inputs"`
  (`audio_inputs`, `image_inputs`). Your `process_prompt` receives that dict and must expose it under
  the *node input name* your encoder consumes (e.g. `out["audio_features"] = tensors["audio_inputs"]`).
- After teardown, a worker may hold the GPU briefly; free it with
  `kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)` before relaunching.

## Debugging a hung request (no crash, no error)

A hang means a node never became *ready* (an `input_names` entry never arrived) or a loop never
terminated. It won't show a traceback, so instrument and localize:

1. Add trace logging (`logger.warning`) at the top of each node's `forward`/`prepare_inputs`, and in
   `get_initial_forward_pass_args` / `get_partition_forward_pass_args`. Restart, send one *short*
   input (fast iteration), and read the trace order.
2. Localize:
   - **First-stage forward never logs** → the producer partition isn't scheduled, or its input didn't
     route (check `process_prompt` produced the node's input key; check the initial forward-pass-args).
   - **Producer logs, consumer's `prepare_inputs` never logs** → the stream isn't delivering: the
     producer's output key ≠ the edge name, or the `Connection`/chunk policy is missing/misnamed, or
     the consumer's fed-back `input_names` aren't seeded on iteration 0.
   - **Consumer logs N times then stops** → the loop isn't terminating: fix `check_stop` / the
     partition's `request_done` transition.
3. A request that hangs for exactly the client timeout, then "client cancelled" in the server log, is
   this class of bug — not a slow model.
