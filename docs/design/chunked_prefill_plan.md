# Chunked prefill

Split a request's prefill for one `(node, graph_walk)` into several forward passes of bounded token count, so that (a) a long prompt cannot monopolise a step, (b) prefill and decode rows can share a batch, and (c) KV admission happens a chunk at a time instead of all-or-nothing.

Opt-in per `(node, graph_walk)`: for example, BAGEL's image prefill declares `causal=False` ([bagel/submodules.py](mstar/model/bagel/submodules.py)) and is never eligible.

---

## 1. Model-facing API

### 1.1 Eligibility

`NodeSubmodule` gains, next to `max_batch_size`
([submodule_base.py](mstar/model/submodule_base.py)):

```python
def supports_chunked_prefill(self, graph_walk: str) -> bool:
    """Whether this walk's inputs may be split across forward passes.
    """
    return False

def max_batch_tokens(self, graph_walk: str) -> int | None:
    """Token cap for one step of this walk; None for no cap."""
    return None
```

For a first implementation, `Engine.get_max_batch_tokens(node_name, graph_walk)` directly calls the submodule-level function instead of also capping on the largest captured `num_tokens` bucket for the walk.

`prepare_inputs_is_async_safe` (§2) is a hard prerequisite: assert at load that every chunking-eligible `(node, walk)` has it set.

A token cap on a walk that is *not* chunking-eligible warns instead of erroring; any prompt over the cap will just ignore the cap for that one step; warning instead of erroring allows us to later derive the cap from the cuda graph configs.

### 1.2 `split_inputs`

```python
def split_inputs(
    self,
    graph_walk: str,
    fwd_info: CurrentForwardPassInfo,
    inputs: NodeInputs,
    start: int,
    end: int,
) -> NodeInputs:
    """This walk's inputs restricted to tokens [start, end).

    `inputs` is the full-prompt result of `prepare_inputs`, prepared once and
    sliced per chunk — never re-derived, so anything read out of a resource
    at prepare time (a position counter, a KV length) is read once, before
    any chunk commits.
    """
```

Default on `NodeSubmodule`: raise. Default on `ARNodeSubmodule`:

- `input_ids`, `input_embeds`: slice `[start:end]` on dim 0.
- `input_seq_len = end - start`.
- `custom_pos_ids`: slice on the **trailing** dim when that dim equals the full `input_seq_len` (covers qwen's `(3, seq)` mrope layout and the plain `(seq,)` layout); raise otherwise. Also handles the `dict[str, Tensor]` multi-label form entry-by-entry.
- `tensor_inputs`, `kwargs`, `resource_step_info` non-empty: **raise**. These are opaque to the framework; a submodule that has them must override.

> Note: slicing the trailing dim is unambiguous whenever it matches the sequence length, and is likely a common override, e.g. Qwen's 3 x seq mrope.

### 1.3 Chunk metadata on `NodeInputs`

Set by the engine after `split_inputs` returns, not by the submodule:

```python
@dataclass
class NodeInputs:
    ...
    chunk_start: int = 0
    # None => not chunked. Total tokens in the walk's full input.
    chunk_total: int | None = None

    @property
    def is_final_chunk(self) -> bool:
        return self.chunk_total is None or \
            self.chunk_start + self.input_seq_len >= self.chunk_total
```

`is_final_chunk` is how a submodule knows, e.g., whether it should sample a token.


A batch can hold several non-final chunks and one final chunk at once; in the cuda graphed case, this is typically handled by always sampling a token furing `forward_batched` and filtering the sampled tokens in `postprocess`. `postprocess` currently does not take in `NodeInputs`, this metadata will also be added to `CurrentForwardPassInfo`.

 `forward` must select the rows it samples
(the samplers already take explicit `request_ids`, so this is natural).

### 1.4 Combined graph walks

```python
class Model:
    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        """{node: {combined_walk: {constituent walks}}}.

        The scheduler batches the constituents together under `combined_walk`;
        the engine declares, preprocesses and runs the batch under it. Each
        request keeps its real walk for input preparation and output routing.
        """
        return {}
```

Validated at load: constituent sets disjoint per node, every constituent a real
walk of that node, no constituent name colliding with a combined name. The
worker builds `(node, real_walk) -> combined_walk` once and hands it to the
`MicroScheduler` and the `Engine`.

The canonical use is `{"LLM": {"mixed_ar": {"prefill_text", "decode"}}}`, i.e., a chunk of prefill riding along with the decode rows. Without it, chunking on its own only *adds* per-step overhead; mixed batching is where the latency win actually comes from.

Where the combined walk is used vs. the real walk:

| Site | Walk |
|---|---|
| `MicroScheduler` grouping, RR bookkeeping, backlog key, batch/token caps | combined |
| `ScheduledBatch.graph_walk`, `StepContext.graph_walk` | combined |
| CUDA-graph capture/replay keys, `cg_key_info`, `can_batch`, `can_use_cuda_graphs` | combined |
| `submodule.preprocess`, `declare_step`, `forward`/`forward_batched` | combined |
| `submodule.prepare_inputs`, `split_inputs` | **real** (per rid) |
| `submodule.postprocess`, `check_stop` | already real, via `request_info.graph_walk` |
| `process_node_outputs`, `store_and_populate_graph_edges`, `_send_outputs` | **real** (per rid) |
| `ScheduleTPNode` follower lookup | **real** (per rid) |

Note `CurrentForwardPassInfo.graph_walk`
([request_info.py](mstar/conductor/request_info.py)) already carries the real
walk per request and is already threaded everywhere the engine and worker need
it, so most of the "real walk" column needs no new plumbing, only the call
sites that currently pass `batch.graph_walk` need to switch to the per-rid
value.

---

## 2. `prepare_inputs` on the main thread

Today `prepare_inputs` runs on the GPU thread inside `_execute_on_gpu_thread`
([worker.py](mstar/worker/worker.py)), after the plan future
resolves. The scheduler cannot see token counts there, so chunking needs it
earlier.

- New class attribute on `NodeSubmodule`, alongside `disable_torch_compile`:
  `prepare_inputs_is_async_safe: bool = False`. It asserts: no `.item()`,
  `.cpu()`, or any other read of a value the in-flight step may still be
  producing.
- `Engine.prepare_inputs` ([engine.py](mstar/engine/engine.py)) is
  unchanged; it becomes idempotent-guarded: `exec` skips it when
  `batch.inputs is not None`.
- The worker calls it from the main thread when the flag is set (§4 for the
  chunking path; the non-chunked path can simply call it right after
  `_build_executing_batch` in the main loop).
- Speculation is disallowed for chunked prefill (node, walk) pairs in this version: a speculative batch's inputs come from N's outputs via `_thread_outputs_to_speculative`, which runs after `pending.future.result()`, and after speculation.

As a note, the main-thread `prepare_inputs` enqueues kernels (`embed_tokens`) onto the same stream the GPU thread is filling. This is correct, but it interleaves ahead of some of step N's launches and slightly delays N (amortized, it should be fine, unless any affected `prepare_inputs` is particularly heavy).

---

## 3. Chunk progress state

Per `(request_id, node_name)`, owned by the `MicroScheduler`:

```python
@dataclass
class ChunkProgress:
    node_name: str
    graph_walk: str          # the real walk
    inputs: NodeInputs       # full-prompt prepare_inputs result, prepared once
    total: int               # inputs.input_seq_len at prepare time
    consumed: int = 0        # tokens committed by chunks that have LANDED
```

`consumed` advances in postprocess, never at scheduling time, and v1 does not
schedule chunk k+1 until chunk k lands (`_can_speculate` returns False for a
batch with incomplete rows, so the main loop takes the await → postprocess →
schedule path). Those two facts together mean `consumed` is always accurate
when the scheduler reads it, with no in-flight delta to track. Pipelining
chunks (§11) is what would need one.

A chunk that cannot be admitted is skipped for that scheduling round.
Admit is decided before the forward, so a refusal commits nothing.

**The progress record must not live on the backlog entry**, because that
push-back returns the `GraphNode` to its ready queue and bypasses the backlog.
Keyed by `(rid, node)` it survives automatically; folded into the
`ScheduledBatch` it would be lost, and the prompt would re-prefill from token 0.

Lifecycle:

- Created when a rid is first scheduled under an eligible walk.
- `consumed += chunk_len` on successful postprocess of that chunk.
- Untouched by admit failure, hold/backoff and offload/reload: none of those
  commit a span, so `consumed` still describes the stream.
- Dropped in `_drop_backlogged_rid`
  ([micro_scheduler.py] and
  `clear_rid` ([micro_scheduler.py](mstar/worker/micro_scheduler.py)),
  and when the final chunk completes.

Debug invariant (behind a flag; it costs a resource query per chunk): the KV
stream length for the walk's label equals `progress.consumed`. Offload/reload
preserves both, so this holds across eviction.

---

## 4. Scheduling

`ScheduledBatch` ([micro_scheduler.py](mstar/worker/micro_scheduler.py))
gains:

```python
graph_walk: str                            # now the COMBINED walk
request_walks: dict[str, str]              # rid -> real walk
prepared_inputs: dict[str, NodeInputs]     # rid -> this step's (possibly split) inputs
incomplete_node_rids: set[str]             # rids that need at least one more chunk
```

`merge` and `split_off_first`
([micro_scheduler.py](mstar/worker/micro_scheduler.py)) carry all four
across; the existing `keep_rids`/`exclude_rids` logic is untouched.

`get_next_batch` ([micro_scheduler.py](mstar/worker/micro_scheduler.py))
picks up a token-cap stage after the existing batch-size cap:

```
1. expire holds; TP-follow                           [unchanged]
2. pick (node, combined_walk): a key with a backlog entry wins,
   else round-robin over the ready scan               [see "seed, don't return"]
3. seed the batch with that key's backlog entry, if any
4. top up from the ready scan for the SAME key, under max_batch_size
5. token cap, if get_max_batch_tokens(...) is not None:
     for rid in the batch, backlog rids first, then FIFO:
       inputs = progress[rid].inputs or engine.prepare_inputs_for(rid)
       remaining = progress[rid].total - progress[rid].consumed
       unit = smallest schedulable slice of `remaining`
              (= remaining for an ineligible walk; the split
               granularity for an eligible one, usually 1 token)
       if remaining <= budget:
           take whole remainder; budget -= remaining
       elif unit <= budget and walk is chunking-eligible:
           take [consumed, consumed + budget); budget = 0
           mark incomplete; remainder to the backlog with progress
       elif batch is empty:
           take `unit`, over budget; stop            [forward progress]
       else:
           backlog rid whole
6. _cap_batch_and_schedule -> RR bookkeeping         [unchanged]
```

Some notes:
- Seed, don't return: `_schedule_from_backlogged` currently early-returns its entry, preventing it from being batched with other requests.
- A row that cannot be reduced to fit the budget is backlogged only while the batch is non-empty; in the next round, it gets a full budget. If it is alone and *still* does not fit, it runs over budget, otherwise it will never be able to be scheduled. We must ensure that a batch is never empty while a schedulable row exists.
- FIFO, not longest-first: this prevents head-of-line blocking in which long prefills would consistently take precedence over latency-sensitive decode rows.
- At most one rid is split per step (the one that straddles the budget), but a batch may contain several rids that are mid-prefill.
- A rid backlogged at step 5 keeps its cached `inputs`, so `prepare_inputs` runs exactly once per request per walk regardless of how many steps it takes to drain.

`_max_batch_size` ([micro_scheduler.py](mstar/worker/micro_scheduler.py))
gains a `_max_batch_tokens` sibling; both look up by the combined walk.

---

## 5. Completion semantics in the worker

From the engine's perspective, a chunk is a full step (all it has to do is thread `is_final_chunk` through `NodeInputs` (§1.3)). However, the worker's postprocess must differentiate between final and non-final chunks, so that it can properly route outputs, update graph node readiness, and send messages to the Conductor. We add two functions to the `ExecutingBatch`:
1. `batch.completes_node(rid)`: whether the node has finished its work, true unless this is a non-final chunk of a chunked node.
2. `batch.completes_walk(rid)`: whether the step also finished the graph walk. If the node is not a streaming consumer of a chunked prefill, this is equivalent to `completes_node`. For a streaming consumer, we may want to make each upstream prefill chunk its own forward pass, but we can't send a walk completion message to the Conductor unless the upstream has also complete its walk.

Specifically, the behavior is:

| | `completes_node` False | `completes_walk` False, `completes_node` True |
|---|---|---|
| Who | the node being chunked | a downstream / streaming consumer |
| Meaning | more chunks of this node's work follow | the step is done; the upstream walk is not |
| `mark_node_complete` / routing | **suppressed** | normal |
| `_cleanup_consumed_inputs` | **suppressed** | normal |
| Non-streaming outputs | accumulated (§5.3) | routed |
| Streaming outputs | emitted, `finished_graph_walk=False` | emitted, `finished_graph_walk=False` |
| worker graph done (WGD) ack to conductor | suppressed, accumulated (§5.4) | suppressed, accumulated (§5.4) |

### 5.1 Carrying the flags

```python
class ExecutingBatch:
    # rids for which this step does not complete the node's own work
    incomplete_node_rids: set[str] = field(default_factory=set)
    # superset: also rids whose streaming consumed input carried  finished_graph_walk=False
    incomplete_walk_rids: set[str] = field(default_factory=set)

    def completes_node(self, rid: str) -> bool:
        return rid not in self.incomplete_node_rids

    def completes_walk(self, rid: str) -> bool:
        return rid not in self.incomplete_walk_rids
```

`_build_executing_batch` fills
`incomplete_node_rids` from the `ScheduledBatch` and adds to
`incomplete_walk_rids` any rid with an ingested edge carrying
`finished_graph_walk=False`.

### 5.2 `_postprocess_batch`

In `_postprocess_batch` ([worker.py](mstar/worker/worker.py)), when
`completes_node(rid)` is False:

- **`_cleanup_consumed_inputs`** ([worker.py](mstar/worker/worker.py)): skipped; otherwise, if a new input arrives, it would go directly into `ready_signals` instead of getting buffered into `ready_next_iter`.
- **`mark_node_complete` / `process_node_outputs` / `_register_outputs`** are skipped. The node stays incomplete.
- **The node is marked in flight**, or the microscheduler will try to re-queue it. We can use the existing `_speculatively_scheduled` flag, renaming it "`_in_flight`".
- **Outputs** go to a per-rid accumulator instead of `store_and_populate_graph_edges`, except streaming outputs (§6), which are emitted every chunk.
- **`progress.consumed += chunk_len`**; on the final chunk the rid is no longer
  in `incomplete_node_rids` and takes the normal path.

When `completes_walk(rid)` is False but `completes_node(rid)` is True,
everything above runs normally and only the worker graph done ack is gated (§5.4).

### 5.3 Non-streaming output accumulation

Per `(rid, node)`, `{edge_name: list[Tensor]}`, appended in chunk order and emitted as one `GraphEdge` on the final chunk, with one `TensorPointerInfo` per tensor. Accumulated on the producer.

`tensor_info` is already a list and `prepare_inputs` already receives a `NameToTensorList`, so the consumer concatenates arbitrarily.

### 5.4 WGD gating and accumulation

`_send_outputs` ([worker.py](mstar/worker/worker.py)) builds the
`WORKER_GRAPHS_DONE` message. When `completes_walk(rid)` is False, suppress the send and accumulate instead:

- `persist_signals`, `new_token_counts`, `output_signal_names` are already buffered by `WorkerGraphsManager` (`buffer_persist_signals` etc.); just don't flush.
- `resource_publish_info` is read from live state, so the final message
  naturally carries the latest.
- `completed_worker_graph_ids` is unchanged. A walk that could complete different nodes in the worker graph on different chunks would strand the conductor, and must set `allow_partial_input=False` (§6.2) and consume only whole walks.
- `stream_tokens_consumed` and `output_loop_indices` are last-value-wins; fine to send only the final value.

---

## 6. `finished_graph_walk` and streaming

### 6.1 The edge flag

Add a `finished_graph_walk: bool = True` flag to `GraphEdge`, set by the producing step's `completes_walk(rid)` (§5.0). 

### 6.2 StreamBuffer

The flag rides `pre_read_register`, which is called on both paths that feed a
buffer — the local path in `_send_outputs` and the remote one in
`_process_new_inputs` — and both have the `GraphEdge` in
hand:

```python
def pre_read_register(self, tensor_id: str, finished_graph_walk: bool = True):
```

The buffer keeps track of the length of the buffer, up to the latest chunk that finishes a graph walk. By default, it uses that length in lieu of the buffer's length. 

To override this behavior (e.g., the Qwen3Omni Talker, for optimal performance, needs to ingest chunks as they arrive), add the following to `ChunkPolicy`:

```python
def allow_partial_input(self) -> bool:
    """Whether chunks may be popped before the producer's graph walk finishes.
    """
    return False
```

Add a `finished_graph_walk: bool` flag to `StreamChunk`, computed as
`any(flag for flag in items)`.

`_pop_streaming_edge` ([worker.py](mstar/worker/worker.py)) stamps
`finished_graph_walk` onto the synthetic edge it builds, which is how it
reaches `incomplete_walk_rids` from §5.1.

---

## 7. TP / SP followers

The leader must broadcast the relevant information for chunked prefill. Specifically, add the following to `ScheduleTPNode` ([ipc_format.py](mstar/utils/ipc_format.py)):

```python
@dataclass
class ScheduleTPNode(MessageBody):
    node_name: str
    graph_walk: str                            # combined walk
    request_ids: list[str]
    request_walks: dict[str, str] = field(default_factory=dict)      # rid -> real walk
    chunk_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)  # rid -> [start, end)
    incomplete_node_rids: list[str] = field(default_factory=list)
```

- `request_walks` is needed by `_try_schedule_tp_follow`
  ([micro_scheduler.py](mstar/worker/micro_scheduler.py)), which calls
  `get_worker_graph_id_for_node(..., graph_walk=...)`.
- `chunk_ranges` lets the follower call `split_inputs` over the same ranges with a single source of truth, avoiding potential asymmetry between ranks.
- `incomplete_node_rids` drives the follower's postprocess gating identically to the leader's (another protection against asymmetry). 

