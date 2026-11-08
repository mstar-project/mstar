# Architecture map of main

Pinned to main at `fa381afd6` (Oct 25 2026). It lists the 76 mechanisms main already has, in five areas. Each entry says what the mechanism does, where it lives, what it keeps true, who uses it, how a change should extend it and the duplicate to watch for, and what breaks it. Every path:line was checked against that commit.

## How a review uses it

1. Read this file, then the area files for the directories the PR touches.
2. For each new class, field, flag, collective, queue, cache or helper in the diff, find the mechanism that owns its concern. Its **Extend via** says how a change should extend it and names the tempting duplicate. A PR that builds that duplicate is an R3 finding, and the fix is the extension, not a patch beside it.
3. For each mechanism the diff touches, read **Keeps** and **Breaks if**. Those are the assumptions on main the PR can break. Check each consumer under **Used by** against the head.
4. Before citing main from the map, check the line in the base tree (the repository root). The map can lag main.

## Keeping it right

The map is a snapshot of main at `fa381afd6` (Oct 25 2026), so it lags the base tree. Check every line in the repository root before citing it. Where the map is wrong, or lacks a mechanism the PR touches, work from the code and say so in the finding; this pass doesn't edit the map. "Landing soon" and "Gaps on main" below are as of that commit: check each in the base tree before calling a gap pre-existing.

## Areas

| File | Covers |
|---|---|
| [`worker-scheduling.md`](worker-scheduling.md) | Worker scheduling: `mstar/worker/` (scheduler, batch builder, speculation, pre-plan, teardown) |
| [`engine-resources.md`](engine-resources.md) | Engine and resources: `mstar/engine/`, `mstar/model/submodule_base.py` (step declaration, admit to commit, capture, KV, prefix cache, sampler, eviction) |
| [`graph-runtime.md`](graph-runtime.md) | Graph runtime and streaming: `mstar/graph/`, `rust/src/graph/`, `mstar/streaming/` (sections, walks, loops, routing, ready sets, streams) |
| [`lockstep.md`](lockstep.md) | TP and SP lockstep: `mstar/distributed/` and the TP paths of `mstar/worker/` (leader role, groups, schedule broadcast, follower replay, journals, TP async) |
| [`lifecycle.md`](lifecycle.md) | Request lifecycle and transport: `mstar/conductor/`, `mstar/communication/`, `mstar/api_server/` (admission, wire codec, teardown barrier, tensor transport, delivery) |

## Mechanisms

### Worker scheduling ([`worker-scheduling.md`](worker-scheduling.md))

| Mechanism | Does |
|---|---|
| Request handles (RidTable) | Maps each wire request id to a small recycled integer that keys all worker-local request state |
| Ready scan and engine readiness | Finds runnable rows by taking graph-ready nodes from the runtime and then asking the engine whether each row's resources are ready |
| Do-not-start set (failed, pending remove, held) | Stops the scheduler starting new work for rows that failed, have a deferred remove, or are in OOM backoff |
| Step selection order and round robin | Picks the next (node, walk) to run: the TP-follow FIFO head first, then backlogged keys oldest first, then the least recently scheduled ready key |
| Backlog | Parks rows that were popped off their ready queues but did not fit in the step, and serves them before anything new |
| Batch builder | Decides, for one key's backlog and fresh rows, which rows run this step, which stay parked and which go back to their ready queues |
| Batch caps | Bounds the rows in one step for each (node, walk), and per real walk inside a mixed step |
| Capture groups | Keeps rows that would replay different capture buckets out of one step when the submodule opts in |
| Combined walks | Lets rows of several real walks of one node share a scheduling key and run in one mixed step |
| In-flight flag | Marks a (rid, node) as claimed by a running, speculated or backlogged batch so that an arriving input does not queue it a second time |
| Admit refusal and OOM eviction | Re-queues a step whose admit refused it, and on allocation failure offloads one least-recently-used request or holds rows so the retry is smaller |
| Teardown ordering: admit fence, deferred removes, drains | Orders every teardown and page move against the in-flight step, so nothing is freed under the GPU thread and followers repeat rank 0's order |
| TP lockstep follow | Makes followers run exactly the steps rank 0 broadcasts, either serially from a FIFO or by rebuilding rank 0's speculative head during their own step N |
| Speculation chain | Builds step N+1 while N runs, as N's continuing rows plus fresh rows of the same target, or as a yield-away batch from the scheduler |
| Pre-plan on the plan thread | Reserves a replay slot and plans step N+1's resources on a separate thread while N's kernels run |

### Engine and resources ([`engine-resources.md`](engine-resources.md))

| Mechanism | Does |
|---|---|
| Resource declaration and build | A model returns `NodeResourceSpec`s, the worker applies YAML overrides and startup refusals, and the engine builds one `Resource` per spec for the nodes it hosts and binds each node's subset into the submodule and its layers |
| Step declaration | Each step, the submodule's `declare_step` returns a frozen `SubmoduleStep` that maps resource keys to `ResourceStep`s made of `Segment(request_id, label, span)`, and the engine attaches the `StepContext` |
| Admit, plan, forward, commit sweep | `StepRunner` drives a declared step through admit, plan and commit in dependency order, and `Engine.exec` wraps that around `preprocess` and the forward, as one batched or leased forward or as a per-request fallback |
| Readiness, retrieve and publish | `check_ready` reloads an offloaded request and then sweeps `admit_retrieve`, so each resource pulls in state another instance published, and after a step `finalize_batch` records what each resource publishes for the next node or worker |
| Pre-planning and slot double-buffering | The worker's plan thread leases step N+1's replay slot and runs admit and plan for the pre-planning resources on a side stream while step N replays, and the full sweep for N+1 later promotes what was staged |
| Host-to-device staging | `utils/h2d.py`: one non-blocking H2D from pinned memory (`to_device_nonblocking`), a pinned ring per resource (`PinnedStager`), and skipping a copy the destination already holds (`H2DMirror`) |
| Capture configs and bucket selection | A submodule declares `AcceleratorGraphConfig`s, and the runner expands them into buckets keyed by walk, batch size, token count and `cg_key_info`, sizes and plans the captures against free memory, captures the largest first, and leases the tightest bucket for each step |
| Padding rows and sink storage | A replay pads the batch to its bucket with per-slot dummy rows that carry negative handles, run against a reserved sink page or slot, reserve and commit nothing, and have their outputs dropped |
| Piecewise region capture | A submodule captures one inner callable, such as a block loop, as its own graph per (batch size, tokens, slot), and on each replay the region declares, admits, plans and commits its own resource step while the rest of the forward stays eager |
| prepare_inputs and the uniform-row path | Before a forward the engine calls `prepare_inputs` per row, probes the prefix cache on keyed walks, drops rows that return None or raise, and asks `can_batch`, or for a walk whose inputs do not depend on the row it calls `uniform_row_inputs` once and shares that object across the rows |
| Per-request submodule state | `PerRequestState` is an engine-owned store where a submodule keeps per-request objects across steps, reached through `request_state(rid)` in `prepare_inputs` and through `engine_inputs.per_request_states` in `preprocess` and the forward |
| Device loopback of sampled tokens | On a walk whose output signals stay on the device, the sampler keeps each request's last token in slot masters, the engine records which rows looped back, and the next step reads the token on the device instead of routing a tensor |
| Output collection and the stop-check host rows | The engine turns a forward's `BatchedModelOutput` into per-rid outputs for the real rows, and the GPU thread stages the row-addressed stop buffers into a pinned host ring behind the step for the main thread's stop check |
| Per-node float32 precision | A submodule declares `float32_matmul_precision` and `cudnn_allow_tf32`, and the engine applies them around that node's forward, compile and capture instead of the model flipping process-wide flags |
| Sampler resource | `SamplerResource` samples one token per row inside the forward, through a captured sampler under a lease and an eager one otherwise, with per-request configs, penalties, generation history and RNG offsets |
| Paged KV manager | `KVManager` owns a paged arena and one `CacheStream` per (request, label), reserves pages at admit, builds paged index arrays and per-token write slots at plan, and at commit advances stored lengths, applies forks and retention, and indexes filled pages |
| Prefix cache | A keyed walk's leading tokens are matched against a page index rooted in the weights, the preprocessing and the resources' fingerprints, every resource on the node agrees on the shortest held prefix, and the engine cuts that prefix off the inputs or skips the walk when all of it is held |
| Eviction, offload and the TP resident journal | When an admit runs out of pages the worker picks a cold victim, the engine offloads that request's state on the node's evicting resources to a pinned host pool, the next readiness check reloads it, and TP followers replay rank 0's ordered moves |
| Positions | `RopeManager` keeps a position counter per (request, label), builds each step's position ids in the KV plan's row order, advances the counters at commit, and applies RoPE through `apply_qk` |
| Recurrent state pool and linear attention | `RecurrentStatePool` gives each (request, label) a fixed-size slot in per-layer state tensors and builds per-step slot addressing, and the `LinearAttnManager` backends (GDN, Mamba-2) plan their kernels against that addressing |
| Attention managers | `AttentionManager.build` picks a backend class from the spec, and the backend keeps one persistent wrapper per (bucket, slot, label) for captured steps and per (label, slot, kind) for eager ones, planned from the KV plan's index arrays |

### Graph runtime and streaming ([`graph-runtime.md`](graph-runtime.md))

| Mechanism | Does |
|---|---|
| Graph sections and I/O derivation | `GraphNode`, `Sequential`, `Parallel` and `Loop` compose a walk's graph, and `get_inputs_outputs` derives each section's external inputs, external outputs and loop-back pairs from `(edge name, next_node)` pairs |
| Walks and worker graphs | Each walk from `get_graph_walk_graphs` is split by `node_groups` into `WorkerGraph`s, merging adjacent sections on the same group, and each worker keeps a per-request instance of every local worker graph |
| Combined walks | Lets the scheduler batch rows of several real walks of one node under one combined walk name while the runtime keeps each row in its own real walk |
| Input slots and ready sets | Each node keeps three input slots (current, next iteration, speculative), and each per-request worker graph keeps the set of nodes whose current inputs are complete and the set whose only missing inputs are streamed |
| Loops | A `Loop` runs its section up to `max_iters`, re-injects its external inputs each iteration, caches its declared and accumulated outputs, and finishes early once its finish signal is set |
| Loop stops | Turns `check_stop`'s loop names into finish signals on every local worker graph that carries the loop, records each stop as pending for one postprocess, sends `STOP_LOOPS` to peer workers, and applies a peer's stop only when its observation is newer |
| Completion and done reporting | Marks a node complete, counts top-level entity completions per worker graph, resets every worker graph of the walk that is now done, and reports those in `WORKER_GRAPHS_DONE` with the persist signals, token counts, emitted chunk names and stream item counts buffered since the last report |
| Output routing and fanout | Sorts each completed node's output edges into local ingest, peer worker, client emit, conductor persist, local stream and remote stream, and slices each edge per destination rank with the sharding fanout |
| Ready scan, pop and push back | Reports the `(node, walk, rids)` whose graph inputs are satisfied, pops a chosen set out of the ready set, and puts a popped set back when its batch cannot run |
| In-flight flag | Marks a node as executing for given rids, normally or speculatively, so arriving inputs fill its slots without re-adding it to the ready sets |
| Speculation targets and staged prep | From a running node's structural outputs, finds the nodes that will be ready next, then per rid ingests buffered stream chunks, checks readiness against the current, next-iteration, held and speculative slots, gathers inputs, and parks what it ingested under a `spec_id` that `commit_speculation` settles |
| Columnar edge blocks | Carries batches of edges across the runtime boundary as parallel columns (signal-name table, per-edge rid, tensor counts, flat uuids, final-chunk flags, optional destinations) rather than one object per edge |
| StreamBuffer and chunk policies | Per request and streaming edge on the consumer worker, holds arriving items in producer order and releases a chunk when the connection's `ChunkPolicy` says enough items are buffered, ending with one final chunk once the producer is done |
| Streaming edges and consumer triggering | Moves data between async partitions: the producer routes a streaming edge like any other but it bypasses the graph queues, the consumer worker buffers it, and the poll loop ingests ready chunks as synthetic edges that make the consumer node ready |

### TP and SP lockstep ([`lockstep.md`](lockstep.md))

| Mechanism | Does |
|---|---|
| Lockstep instance and leader role | Defines a node's TP×SP block as one lockstep instance and makes instance rank 0 its only scheduler |
| Process-group creation schedule | Creates every TP and SP subgroup on every rank in one globally sorted order before serving |
| Setup-time backend votes | Has every rank of a comm group vote on whether it could build an optional collective backend, and keeps the backend only if all could |
| Capture agreement and per-spec barriers | Reduces every capture decision across the joint group and barriers once per spec, so all ranks keep, drop, order and size the same accelerator graphs |
| Startup fences and symmetry checks | Brackets warmup with global barriers and refuses to serve when per-rank settings the protocol depends on disagree |
| ScheduleTPNode broadcast | Sends each step the leader runs on a parallel node to every follower as one `ScheduleTPNode`, stamped with a monotonic seq and carrying the leader-only decisions the step depends on |
| Follower FIFO and serial follow path | Queues the leader's steps on each follower and runs them in order, all rows or none, ahead of the follower's own work |
| Follower settle seam | Applies everything a head carries for the follower's resource state before the follower builds that step |
| Resident-set journal | Records every offload and reload the leader executes on a parallel node, in order, ships them on the next `ScheduleTPNode`, and has followers replay them verbatim instead of choosing victims |
| Prefix-cache lease replication | Has the leader lease each row's cached prefix pages as it sends the step and ship the page ids in `prefix_hits`, so followers adopt exactly those pages |
| Page-trail divergence check | Under `MSTAR_KV_DEBUG_ASSERTS=1` with `prefix_cache_tp`, hashes each paged KV arena's allocation sequence so a follower compares it with the leader's before every step |
| Admit fence | Blocks the worker's main thread at the top of every loop iteration until the in-flight step has admitted |
| Stamped teardown ordering | Has the leader forward removals and drains to its followers, stamping each removal with the last seq it broadcast, and has followers tear down only at that point in the step stream and in the leader's order |
| TP async scheduling protocol | With `MSTAR_TP_ASYNC_SCHED` on, has the leader speculate step N+1 during N and broadcast it at once as a head with `spec_from_seq` N, and send exactly one of a head or a `TPNoSpeculation` marker for every leader step, so followers rebuild N+1 during their own N |

### Request lifecycle and transport ([`lifecycle.md`](lifecycle.md))

| Mechanism | Does |
|---|---|
| ZMQ control mesh and wire codec | Carries every control message between `api_server`, `api_server_preprocess_worker`, `conductor` and `worker_N`, one PULL inbox per entity with PUSH sockets to each peer, encoded as typed msgpack |
| Conductor admission and dispatch | Takes a NEW_REQUEST, queues it at `max_concurrent_requests`, picks workers for each worker graph, resolves the request's sharding and resource configs, and sends one NewRequest per worker and partition |
| Graph-walk transitions and partition completion | Counts WORKER_GRAPHS_DONE per worker graph and rank, and when a partition's walk is complete asks the model for the next walk, then sends its inputs, sends producer_done downstream, or ends the partition |
| Persisted signals across walks | Keeps tensors a later walk needs (the request's inputs and outputs routed as persist) alive on their producer, with the conductor tracking how many readers each was sent to, and releases them with one UNPERSIST_TENSORS per producer |
| Two-phase teardown barrier | Tears a request down in two phases: DRAIN_REQUEST makes each participant stop starting reads and send READS_DONE once its in-flight reads finish, and only then does the conductor send REMOVE_REQUEST to every participant and, on the fail path, `request_failed` to the client |
| Request failure reporting and fatal shutdown | Routes every per-request failure to the client as one `request_failed` or error chunk and frees the request through the barrier, and turns a dead worker or conductor into a 503 for every waiting client followed by shutdown |
| Worker removal sequencing | Applies REMOVE_REQUEST on a worker in a safe order: deferred while an in-flight step or speculation holds the rid, replayed on TP followers in rank 0's order and step position, then purging engine, scheduler, tensor and graph-runtime state with the handle released last |
| Late messages for unknown and torn-down ids | Parks INPUT_SIGNALS, STOP_LOOPS and forwarded REMOVE messages for a request this worker has no handle for until its NEW_REQUEST arrives, and drops them instead, acking any tensors unread, when the id is one of the last 4096 torn down here |
| Worker-local request handles | Interns each wire request id to a recycled small integer on NEW_REQUEST and turns it back into the string at every send |
| Tensor transport backends and reads | Moves tensor bytes out of band of the control mesh: the producer stores and registers a tensor and ships a `TensorPointerInfo`, and the consumer reads it over Mooncake RDMA or TCP, a per-uuid file under `/dev/shm`, or the Rust SHM arena |
| Tensor ref counts and TENSOR_RECEIVED acks | Counts every stake in a stored tensor per uuid (node use, transit, each remote reader), frees the tensor and its transport resources when the count reaches zero and it is not persisted, and turns each consumer's finished read into a TENSOR_RECEIVED ack to the producer |
| Output routing out of a worker | Is the one place a worker's step outputs leave the process: it marks nodes complete, decides each output's destinations, and sends INPUT_SIGNALS to peers, RESULT_TENSORS (or RESULT_TOKENS) to the API server and WORKER_GRAPHS_DONE to the conductor |
| API server submissions, result delivery and disconnects | Gives each submission a never-reused key `<request_id>#<n>`, hands it to the data worker, collects result chunks on a message thread, holds a finished request open until every counted chunk has arrived, and aborts the submission when the client disconnects or times out |
| Media decode and model preprocessing in the data worker | Calls the model's `load_image`, `load_audio` or `load_video` per input file and one `process_prompt` on the data worker thread, merges the result into the request's tensors and stores them with the request |
| Process CPU budget | Gives the number of CPUs a process can keep busy (affinity capped by the cgroup quota) and sizes the API server's torch, encode and PNG pools from it |
| Speech sentence chunking on `/v1/audio/speech` | Cuts a long speech input into sentence groups on the API server's event loop (`_plan_chunks`, `split_sentences`) and submits each as its own engine request, a few ahead of the one being streamed |
| Worker metrics | Records step-time and batch-size histograms and samples KV, prefix-cache, pending-row and per-edge transfer metrics on each worker, sends them to the conductor as WORKER_STATUS once per interval, and the conductor merges them into the ConductorStatus it sends the API server |

## Gaps on main, checked

Each was read at `fa381afd6`. The consequence is stated only where it was checked.

- **Speculating into a parallel node is not filtered.** Nothing calls `GraphRuntime.set_node_metadata` since #333 removed it from `Worker.__init__`, so both runtimes' filter on a parallel target runs on empty sets (`mstar/graph/runtime/python.py:707-716`, `rust/src/graph/runtime.rs:123`). Shipped configs can reach it: in `cosmos3_nano_tp2` the `vae_encoder` on rank 0 feeds the TP `dit`. Whether that misbehaves was not run. See `lockstep.md` › Findings.
- **`can_use_accelerator_graphs` has no caller.** The engine's call went with #228. Six overrides in orpheus, qwen3_omni, qwen3_tts, chatterbox and fish_s2_pro refuse captures on batch shape and are called only by tests. Whether bucket selection covers each refusal was not checked.
- **A refused send-side registration fails the whole batch.** `_register_outputs` lets the refusal reach `_handle_main_loop_error` (`mstar/worker/worker.py:2030` at `11c910bcf`), and a failed Mooncake unregister raises out of `REMOVE_REQUEST`'s cleanup (`mstar/communication/tensors.py:2195`). #629 fixes both. The failed-read stall listed here before is fixed by #419.
- **Python and Rust disagree on a peer's older loop stop.** Rust stores it (`rust/src/graph/runtime.rs:3155`, asserted by `test/rust/test_rust_graph_runtime.py:873`) and Python keeps the newer one (`mstar/graph/runtime/python.py:1283`). This breaks Invariant 5. The consequence was not run.
- **The do-not-start check is written out at eight places, and one differs.** `has_ready_excluding` leaves out deferred removes. That costs one overlapped step per deferred remove, nothing worse. See `worker-scheduling.md` › Do-not-start set.
- **Three copies of the cross-rank vote.** `_all_ranks_agree` (`mstar/distributed/all_reduce.py:37`), `_all_ranks` (`mstar/distributed/ar_fusion.py:28`, the same body) and `agree_across_ranks` (`mstar/engine/accelerator_graph_runner.py:99`). A new vote reuses one of them.
- **A worker drops an unknown `WorkerMessageType`.** The handler chain in `Worker._process_message_list` has no `else`, while the conductor raises.
- **Stale docs.** `NodeSubmodule.get_autocast_dtype` says the engine reads it every forward, but it is read once at load to cast parameters. Invariant 6's "falls back to pickle" no longer holds (pickle frames are refused). Smaller ones are listed at the end of each area file.

## Landing soon

- #365 (about to merge): `get_chunking_policy`, `keep_final_chunk_samples` and `disable_spec_node_walks`, which `AGENTS.md` already describes. Update `engine-resources.md`, `graph-runtime.md` and `worker-scheduling.md` when it merges.
- #419 (merged): per-request read failures and the TP read check. `lifecycle.md` › Tensor transport backends and reads is updated. The read check still has to be added to `lockstep.md`.
- #383 (`RecurrentStatePool.gather` and `scatter_`) and #385 (`Connection.consumer_walk`) are open.
