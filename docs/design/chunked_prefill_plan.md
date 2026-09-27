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

A token cap on a walk that is *not* chunking-eligible warns instead of erroring and any prompt over the cap will just ignore the cap for that one step; warning instead of erroring allows us to later derive the cap from the cuda graph configs.

### 1.2 `split_inputs`

The required behavior was already implemented for prefix caching; I'm just including the spec below for reference:
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
    # the request's REAL walk (§1.4). Differs from the batch's walk only
    # under a combined walk; stamped by the engine, never by the submodule.
    graph_walk: str | None = None

    @property
    def is_final_chunk(self) -> bool:
        return self.chunk_total is None or \
            self.chunk_start + self.input_seq_len >= self.chunk_total
```

`is_final_chunk` is how a submodule knows, e.g., whether it should sample a
token. `postprocess` already receives the step's `NodeInputs`
(`inputs: NodeInputs | None`,
[submodule_base.py](mstar/model/submodule_base.py)), so this needs no new
plumbing there and does **not** go on `CurrentForwardPassInfo`.

A batch can hold several non-final chunks and one final chunk at once; see
§1.5 for how sampling is restricted to the final chunk.

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
| `ScheduledBatch.graph_walk`, `StepContext.graph_walk` | combined, with `StepContext.request_walks` giving the resource layer the per-rid real walk |
| CUDA-graph capture/replay keys, `cg_key_info`, `can_batch`, `can_use_cuda_graphs` | combined |
| `submodule.preprocess`, `declare_step`, `forward`/`forward_batched` | combined, with the per-request real walk stamped onto `NodeInputs` |
| `submodule.prepare_inputs`, `split_inputs` | **real** (per rid) |
| `submodule.postprocess`, `check_stop` | already real, via `request_info.graph_walk` |
| `process_node_outputs`, `store_and_populate_graph_edges`, `_send_outputs` | **real** (per rid) |
| `ScheduleTPNode` follower lookup | **real** (per rid) |

`declare_step` and the resource layer are the reason the combined rows still
need the real walk alongside: they decide per row which KV labels get written,
whether attention is causal, whether prompt tokens are tracked for the
repetition penalty, and how `_index_filled_pages` keys a stream. Handing them
only the combined name would silently change all of those (BAGEL would drop
`cfg_img` and declare non-causal attention; prefix caching would un-key every
stream). `declare_step` reads it off each row's `NodeInputs.graph_walk`, and
the resource layer off `StepContext.request_walks`.

Note `CurrentForwardPassInfo.graph_walk`
([request_info.py](mstar/conductor/request_info.py)) already carries the real
walk per request and is already threaded everywhere the engine and worker need
it, so most of the "real walk" column needs no new plumbing, only the call
sites that currently pass `batch.graph_walk` need to switch to the per-rid
value.

### 1.5 Sampling on non-final chunks

On the CUDA-graph path every row in the batch samples, and the tokens belonging
to non-final chunks are dropped in `postprocess`. Dropping the token does not
undo the sampler's side effects, though: the RNG offset has already advanced,
and with a repetition penalty the token is already in the seen-token mask.
Because the number of chunks depends on load, seeded output would depend on
load too.

This is a **pre-existing bug** rather than one chunked prefill introduces; a
model with several prefill graph walks (BAGEL) already samples once per walk
and keeps only the last — but chunking amplifies it, so it gets fixed here.

Fix: `SamplerStep`
([sampler/config.py](mstar/engine/resources/sampler/config.py)) names the rows
that actually sample, and `SamplerResource.commit` persists RNG offset and
seen-token state only for those. The forward still draws for every row, so the
captured shape is unchanged; only the commit is selective. The engine fills
that set from `NodeInputs.is_final_chunk`.

---

## 2. Declaration of input sequence length

Instead of moving `prepare_inputs` to the main thread, which forces it to be async-safe and also partially blocks speculation and preplanning, a `get_input_sequence_len` function will be added to submodules that opt into chunked prefill:

```
def get_input_sequence_len(
    self, graph_walk: str,
    fwd_info: CurrentForwardPassInfo,
    inputs: NameToTensorList,
    **kwargs
) -> InputSeqLenInfo | None:
    return

class InputSeqLenInfo(NamedTuple):
    seq_len: int
    # Sequence length that each resource will reserve, passed into admit_retrieve.
    resource_segment_lengths: dict[str, int]

    def get_admit_segment_len(self, resource: str):
        return self.resource_segment_lengths.get(resource, self.seq_len)
```

This will be called in the micro-scheduler during batch-building, before
`admit_retrieve`. It is expected to read **shapes only** — the host already
knows `input_ids.shape` for everything the tensor manager hands it — so it
never syncs and never has to be async-safe. Anything derived from, e.g.,
(`prefill_vision`'s `.item()`, say) stays in `prepare_inputs` on the GPU
thread, where it is today.

The default implementation returns `None`, which means "not declared"; a walk
that opts into chunked prefill must override it, asserted at load next to the
`supports_chunked_prefill` check.


---

## 3. Chunk progress state

Per `(request_id, node_name)`, owned by the `MicroScheduler`:

```python
@dataclass
class ChunkProgress:
    node_name: str
    graph_walk: str          # the real walk
    inputs: NodeInputs       # full-prompt prepare_inputs result, prepared once on the first chunk
    total: int               # inputs.input_seq_len at prepare time
    consumed: int = 0        # tokens committed by chunks that have LANDED
```

`consumed` advances in postprocess, never at scheduling time, and v1 does not
schedule chunk k+1 until chunk k lands (`_can_speculate` returns False for a
batch with incomplete rows — see §8, where chunked rows are filtered out of a
speculative batch's *continuing* set, so the main loop takes the await →
postprocess → schedule path for them). Those two facts together mean
`consumed` is always accurate when the scheduler reads it, with no in-flight
delta to track.

**Note (interaction with prefix caching)**: on a prefix-cache hit, the hit sets
the initial value of `consumed` to the number of tokens already in the cache,
instead of the prefix cache cutting the prompt itself with `split_inputs`.
`ChunkProgress.inputs` then holds the untrimmed inputs, the prompt is cut in
exactly one place, and the debug invariant below still holds.

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

### 3.1 Admission and the partial-prompt deadlock

Today a prompt that doesn't fit takes no pages at all; `try_allocate` is
all-or-nothing. Chunk-at-a-time admission changes that: a mid-prefill request
holds its earlier chunks' pages while it waits for the next one. If its next
chunk can't be admitted and that puts the whole batch on hold, the decode rows
sharing the batch stop too, and with offload off by default nothing frees until
requests time out.

Reserving the whole prompt at the first chunk would avoid this, but it gives up
the main benefit of chunked admission: that a long prompt no longer has to fit
all at once, and decode rows can drain while its chunks run.

Instead, capacity is checked **per row at batch-build time**:
`admit_retrieve` takes an optional sequence length (per resource, from
`InputSeqLenInfo.resource_segment_lengths` via `get_admit_segment_len`) and
fails when the resource doesn't have room for it. A row that fails is simply
**excluded from the batch** for that round rather than putting the full batch on
hold.

Residual exposure: a pool held entirely by mid-prefill rows that each need more
space still can't make progress on its own. The existing hold/backoff and
offload paths remain the backstop for that; chunking doesn't make it worse than
the unchunked case, where the same pool would be held by unadmitted whole
prompts.

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
5. token cap, if get_max_batch_tokens(...) is not None: see below
6. _cap_batch_and_schedule -> RR bookkeeping         [unchanged]
```

### 4.1 Dividing the token budget

The batch's rows are picked FIFO (steps 3–4), and only then is the budget
divided over the rows it contains. The division is max-min fair share: rows that fit under an
equal share go in whole, and whatever they leave behind is re-divided among the
rows that are still too big.

```
budget  = get_max_batch_tokens(node, combined_walk)
pending = rids in the batch, FIFO order
remaining(rid) = progress[rid].total - progress[rid].consumed
unit(rid)      = smallest schedulable slice
                 (= remaining(rid) for an ineligible walk; the split
                  granularity for an eligible one, usually 1 token)

# (a) water-filling: take every row that fits under an equal share, whole
while pending:
    share = budget // len(pending)
    fits  = [rid for rid in pending if remaining(rid) <= share]
    if not fits:
        break
    for rid in fits:
        take remaining(rid) whole; budget -= remaining(rid)
    pending -= fits

# (b) everything still pending is over its share: chunk it
share = budget // len(pending) if pending else 0
for rid in pending:
    if walk is chunking-eligible and share >= unit(rid):
        take [consumed, consumed + share); mark incomplete; budget -= share
    else:
        drop rid from the batch  # returns to its ready queue, uncommitted

# (c) forward progress: never emit an empty batch while a row is schedulable
if the batch is empty:
    take the FIFO-oldest row at unit(rid) granularity, over budget
```

Why this shape:

- **Decode rows are free.** `remaining == 1` for a decode row, so it always
  lands in the first water-filling pass. That reproduces TensorRT-LLM's
  "generation requests first, prefill chunks split what's left" without needing
  a separate rule for it.
- **Long prefills are not starved.** A long prefill is never *skipped* in
  favour of shorter work; it is in `pending` and gets `budget / len(pending)`
  every round. It goes slower when the batch is busy, which is the point, but
  it always advances. Picking the batch FIFO is what guarantees it gets into a
  batch in the first place.
- **Head-of-line blocking is bounded.** Without this, a single long prompt
  under `mixed_ar` would take the whole budget every round until its last
  chunk, and the decode rows sharing the batch would be pushed back every time.

This is still a heuristic and will want tuning against real traces.

### 4.2 Push back to the ready queue, not the backlog

A non-final chunk's remainder is pushed back onto the node's **ready queue**,
and so is a row dropped at step (b). Neither goes to the backlog. Three reasons:

- The backlog is served before anything else in step 2, so a chunked prompt's
  chunks would run back to back and monopolise the budget until the prompt
  finished.
- TP followers have no backlog path: their scan skips parallel nodes and they
  reach a node only through `pop_ready_rids`, which needs it in
  `ready_node_names`. With the backlog, nothing puts a mid-prefill node back
  there and the follower's `_try_schedule_tp_follow` would return `None` on
  every attempt while the leader waits in the collective (§7).
- `_handle_admit_failure` / `_handle_allocation_failure` push every node in a
  failed batch back onto the ready queue. If the split remainder were in the
  backlog, the same `(rid, node)` would end up holding two scheduling handles.

Some more notes:
- Seed, don't return: `_schedule_from_backlogged` currently early-returns its entry, preventing it from being batched with other requests.
- A row dropped at step (b) is refused only while the batch is non-empty; in the next round it can be the FIFO-oldest and get a full budget. If it is alone and *still* does not fit, it runs over budget — otherwise it could never be scheduled. We must ensure a batch is never empty while a schedulable row exists.
- More than one rid may be split in a step, and a batch may hold several rids that are mid-prefill.
- A chunked rid pushed back to the ready queue keeps its cached `inputs` from the first forward pass of the walk, so `prepare_inputs` runs exactly once per request per walk regardless of how many steps it takes to drain.

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
| `mark_node_complete` / routing | **suppressed** | **suppressed** |
| `_cleanup_consumed_inputs` | **suppressed** | **suppressed** for non-streaming inputs, normal for streaming inputs |
| Non-streaming outputs | accumulated (§5.3) | routed |
| Streaming outputs | accumulated (§5.3) by default; under `allow_partial_input` (§6) emitted with `finished_graph_walk=False` | emitted, `finished_graph_walk=False` |
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
- **Outputs** go to a per-rid accumulator instead of `store_and_populate_graph_edges`, streaming outputs included — unless the consuming edge opted into `allow_partial_input` (§6), in which case they are emitted every chunk.
- **`progress.consumed += chunk_len`**; on the final chunk the rid is no longer
  in `incomplete_node_rids` and takes the normal path.

When `completes_walk(rid)` is False but `completes_node(rid)` is True,
everything above runs normally and only the worker graph done ack is gated (§5.4).

### 5.3 Output accumulation

Per `(rid, node)`, `{edge_name: list[Tensor]}`, appended in chunk order and
emitted as one `GraphEdge` on the final chunk. Accumulated on the producer.

What that one edge looks like is declared per output edge:

```python
class ChunkedPrefillOutputMode(Enum):
    # torch.cat the per-chunk tensors along `dim` at the final chunk; the
    # consumer sees a single tensor, exactly as in the unchunked case.
    CONCAT = "concat"
    # keep one TensorPointerInfo per chunk in `tensor_info` and let the
    # consumer decide (`prepare_inputs` already takes a `NameToTensorList`).
    LIST = "list"

@dataclass
class ChunkedPrefillOutputPolicy:
    mode: ChunkedPrefillOutputMode = ChunkedPrefillOutputMode.CONCAT
    dim: int = 0

class NodeSubmodule:
    def get_chunked_prefill_output_policies(
        self, graph_walk: str,
    ) -> dict[str, ChunkedPrefillOutputPolicy]:
        """edge name -> policy; unlisted edges get the CONCAT default."""
        return {}
```

`CONCAT` is the default because essentially every consumer today reads
`inputs[name][0]` — the Talker's `thinker_states` included — and none of them
concatenates the list. Defaulting to `LIST` would silently drop every chunk
after the first.

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

### 6.0 Default: the producer accumulates

By default, a chunked walk's **streaming** outputs are accumulated exactly like
its non-streaming ones (§5.3) and emitted as a single item at the final chunk.
A streaming consumer therefore sees precisely the item sequence it would have
seen without chunking: no consumer changes, no `ChunkPolicy` changes, and no
way for a `FixedChunkPolicy(1)` to fire on a partial prompt. vLLM-Omni and
SGLang-Omni don't stream prompt chunks to the talker either.

Emitting per chunk is opt-in, for a consumer that wants to pipeline
against the producer. Everything in §6.1–§6.3 describes that opt-in path; under
the default none of it is reachable.

### 6.1 The edge flag

Add a `finished_graph_walk: bool = True` flag to `GraphEdge`, set by the producing step's `completes_walk(rid)` (§5).

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

### 6.3 Worked example: the Qwen3-Omni Talker

`talker_prefill` consumes `thinker_states` (streaming) and a `talker_trigger`
that the conductor sends once, after the Talker's WGD. Under
`allow_partial_input`, the Thinker emits a `thinker_states` item per chunk and
the Talker's WGD is held back (§5.4) — so there is no second trigger, and the
naive version stalls after the first chunk.

The fix is the `_cleanup_consumed_inputs` row of the §5 table: on a step where
`completes_walk(rid)` is False, the worker clears only the **streaming** inputs
and leaves the non-streaming ones in `ready_signals`. The `talker_trigger`
therefore stays satisfied, and the node becomes ready again as soon as the next
`thinker_states` chunk lands, with nothing re-sent by the conductor.

This is admittedly narrow behavior in service of one model's pipelining, which
is why it lives behind `allow_partial_input` rather than in the default path.
The alternative — accumulating on the producer (§6.0) and giving up the
Thinker/Talker overlap — stays available and is what every other consumer gets.

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
- `chunk_ranges` lets the follower call `split_inputs` over the same ranges with a single source of truth, avoiding potential asymmetry between ranks. The leader is authoritative for chunk size; a follower never decides one.
- `incomplete_node_rids` drives the follower's postprocess gating identically to the leader's (another protection against asymmetry).

### 7.1 Followers keep their own `ChunkProgress`

A follower maintains its own `ChunkProgress` per `(rid, node)`, with the same
lifecycle as the leader's (§3). It is used for one thing only: caching
`inputs`, the full-prompt `prepare_inputs` result, so that `prepare_inputs`
runs **once** per request per walk on the follower too. A follower that
re-prepared per chunk would read the position counter *after* the previous
chunk's commit and produce different positions from the leader.

`start`/`end` and `consumed` on the follower come from `chunk_ranges`, not from
its own accounting, so the two ranks cannot drift even if their views of
progress diverge.

### 7.2 `pop_ready_rids` under a combined walk

`MicroScheduler.pop_ready_rids`
([micro_scheduler.py](mstar/worker/micro_scheduler.py)) currently resolves one
worker graph id from `request_ids[0]` and the batch's walk, then pops every rid
out of that queue. A combined walk has no worker graph, so this has to go **per
rid**, resolving `get_worker_graph_id_for_node(rid, node, graph_walk=...)` from
`request_walks[rid]`. The all-or-none check stays all-or-none across the whole
set; only the lookup moves inside the loop.

### 7.3 Reaching a mid-prefill node at all

Followers have no backlog path — their scan skips parallel nodes, and they
reach a node only through `pop_ready_rids`, which requires it in
`ready_node_names`. §4.2's push-back onto the ready queue is what makes chunk
*k+1* reachable on a follower; with the remainder in the backlog instead,
`_try_schedule_tp_follow` would return `None` on every attempt while the leader
sat in the collective.

---

## 8. Speculation

Speculation is disabled for **chunked-prefill walks**, not for combined walks.
The distinction matters: under `mixed_ar`, disabling per *walk* would turn
speculation off for every decode step that happens to share a batch with a
prefill chunk, which is most of them, and those steps would lose async overlap
for their decode rows too.

The exemption is **per row**, at the point where `_try_speculate_next` ([worker.py](mstar/worker/worker.py)) builds its batch:

1. **Continuing rids**: chunked-prefill rows are filtered out of the
   `continuing` set before anything else. Non-chunked rows in the same
   in-flight batch — decode under `mixed_ar` — still speculate normally. This
   is also what preserves §3's invariant that chunk *k+1* is never scheduled
   before chunk *k* lands: a chunked row can never continue into a speculative
   batch, so it always takes the await → postprocess → schedule path.
2. **Fresh rids**: a chunked-prefill row *may* join a speculative batch as a
   fresh rid. Fresh rids come through the normal tensor-manager path and have
   not run `prepare_inputs` yet, so there is no in-flight `consumed` delta to
   reason about; it is the first chunk of the walk, same as any other first
   scheduling of that node.

Two consequences to get right in the implementation:

- The token cap (§4.1) must count the **continuing** rows when
  `_try_speculate_next` merges fresh ones in; the budget is for the merged
  batch, not the fresh half.
- `_is_tp_follow_pending` must mirror the new condition exactly. It currently
  mirrors `_can_speculate`; if the two drift, a TP-async follower waits on a
  head or marker the leader never sends.

`_can_speculate` itself stays batch-level and structural (async-scheduling
flags, leader/follower role) — the chunk-awareness is entirely in the row
filter.

