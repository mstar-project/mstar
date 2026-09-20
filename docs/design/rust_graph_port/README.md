# Porting the graph layer to Rust — scoping + first integration

Scoping for a Rust port of `mstar/worker/node_manager_utils.py` (+ the graph
state it drives) with **batched** APIs, so the per-rid Python loops in
`worker.py` disappear rather than shrink — plus a first cut of the integration.

## What has landed (§9 for status and what's left)

| where | what |
|---|---|
| `rust/src/graph/` | the graph core, in the `mstar-rust` crate |
| `mstar/graph/rust_core.py` | `GraphSection` → Rust spec compile seam |
| `mstar/graph/runtime/` | the harness: ABC + Python + Rust + Shadow + factory |
| `mstar/worker/node_manager_utils.py` | `WorkerGraphQueues` delegates to the harness |
| `test/rust/test_graph_runtime_parity.py` | 68 parity tests incl. real model graphs |

`MSTAR_RUST_GRAPH=0|shadow|1` (default `0` — no behavior change).

Artifacts here:

| file | what |
|---|---|
| `bench_python.py` | baseline: current per-rid Python cost |
| `proto/` | Rust prototype crate (`GraphRuntime`, batched) |
| `bench_rust.py` | prototype vs baseline, same graph/batch |
| `bench_sends.py` | per-rid message cost in `_send_outputs` |
| `test_parity.py` | flat-loop parity: loop iters, emit/persist, ready set |
| `test_nested_parity.py` | nested-loop parity + the Python divergence it found |
| `graph_fixture.py` | synthetic decode-loop and nested worker graphs |
| `rust_spec.py` | `GraphSection` → Rust spec (superseded by `mstar/graph/rust_core.py`) |
| `bench_harness.py` | head-to-head through the shipped harness |

Repro:

```bash
cd docs/design/rust_graph_port/proto && cargo build --release \
  && cp target/release/libmstar_graph_proto.so mstar_graph_proto.so && cd -
D=docs/design/rust_graph_port
python $D/bench_python.py
PYTHONPATH=$D/proto python $D/bench_rust.py
PYTHONPATH=$D/proto python $D/test_parity.py
PYTHONPATH=$D/proto python $D/test_nested_parity.py
python $D/bench_sends.py
```

---

## 1. Measured baseline

Synthetic decode loop `Loop[llm -> sampler]`, `width` extra fan-out nodes in the
loop body. Single worker, TP=1. Numbers are ns per request per forward pass.

```
    B  width  py postproc  rs postproc  speedup    py sched  rs sched  speedup   (ns/rid)
    1      0        22322          765    29.2x         981       235     4.2x
    8      0        20792          390    53.3x         708        47    15.0x
   32      0        20350          281    72.4x         609        18    34.4x
  128      0        21666          276    78.5x         612        12    52.1x
    1      4        42263          993    42.5x        2178       413     5.3x
    8      4        48228          402   120.0x        1847        53    34.6x
   32      4        48781          325   149.9x        1647        30    54.7x
  128      4        58983          324   181.9x        1733        13   138.3x
```

- `postproc` = `mark_node_complete` + edge clone + `process_node_outputs`
  (`worker.py:2315-2368`), the loop that routes a step's outputs.
- `sched` = the ready-scan in `MicroScheduler.get_next_batch`
  (`micro_scheduler.py:343-373`), minus the engine `check_ready` call.

Request admission (`WorkerGraphQueues.add_request`, which `deepcopy`s the whole
graph section **per request per worker graph**), µs per request:

```
 width   py add_request   rs add_requests   speedup
     0          160.5 us           0.86 us      186x
     4          781.5 us           1.41 us      553x
    16         1662.6 us           3.33 us      499x
```

Per-rid message cost in `_send_outputs` (`worker.py:2395-2402`):

```
                 message     repr    codec   before    after  (us/send)
    INPUT_SIGNALS (peer)     9.98    10.94    20.93    10.94
 result_tensors (client)     4.18     6.22    10.39     6.22
```

Both communicators passed an eager `str(msg)` to a disabled `logger.debug`,
**doubling** every send. Fixed (see §8.1); the `codec` column is what remains
and only goes away when Rust owns the send.

**Budget for a B=128 decode step, width=0, single worker:** ~2.8 ms routing +
~1.3 ms `emit_to_client` sends ≈ **4 ms of pure Python per step**, growing
linearly in batch size and graph width. The Rust path is ~40 µs.

The cost is not concentrated in one hot spot — it is ~150 interpreter-level
calls per rid spread over `GraphEdge.clone`, `NodeAndGraphWalk` hashing, set
ops, and `compute_fanout`. Confirmed by removing the eager `logger.debug`
argument formatting in `process_node_outputs`: <2% change. There is no
Python-level fix worth making here; the loop has to leave Python.

---

## 2. Where the abstraction barrier should go

PR #170 ports the *state machine* and keeps Python driving it once per request.
That caps the win at the per-call speedup and keeps B boundary crossings per
step. The barrier should instead be drawn so that **one Python call serves the
whole batch**, and per-rid results that Python doesn't need never become Python
objects.

```
                       Python                    |            Rust
  ------------------------------------------------|----------------------------------
  torch tensors, engines, submodules,            |  graph topology (interned)
  CUDA graphs, StreamBuffer payloads,            |  per-request walk state
  conductor/API-server policy                    |  ready sets, loop counters
                                                 |  sharding fanout
                                                 |  routing -> wire-ready blobs
                                                 |  scheduler pick (rr / backlog)
```

Concretely, three rules:

1. **Requests are dense `u32` handles**, minted once at admission. No `rid`
   string hashing on the hot path. Python holds `list[int]` parallel to its own
   batch order.
2. **Tensors are `u64` handles**, not `str(uuid4())`. The tensor manager owns
   the handle→storage map; the graph layer only moves handles.
3. **Edges bound for peer workers never enter Python at all.** See §3.1 — the
   Rust transport is already there, so the send can stay entirely Rust-side.

---

## 3. Proposed batched API

Replacing `WorkerGraphsManager`. Names match the Python they subsume.

```python
class GraphRuntime:                     # one per worker, owns all worker graphs
    def add_requests(rids, partition, walk, wg_ids) -> list[Handle]
    def remove_requests(handles)
    def set_graph_walk(handles, partition, walk)     # was update_request_info

    # --- postprocess: replaces worker.py:2315-2402 in one call -------------
    def complete_and_route_batch(
        node, walk, partition, handles,
        uuids: list[u64], tlens: list[u32],          # flat, per (rid, out-edge)
    ) -> BatchRouting

    # --- scheduling: replaces the O(B x nodes) scan ------------------------
    def ready_scan(exclude=None, target=None, max_bs=None) -> (node, walk, handles)
    def pop_ready(node, walk, handles) -> bool       # all-or-none
    def push_back(node, handles)                     # OOM hold

    # --- loops -------------------------------------------------------------
    def stop_loops_batch(partition, handles, loop_names) -> StopRouting
    def loop_iters(handles, partition) -> list[dict[str, int]]
    def nested_loop_idxs(node, handles) -> ...

    # --- ingest ------------------------------------------------------------
    def ingest_batch(blob) -> list[Handle]           # a peer's InputSignals blob
```

```python
class BatchRouting:
    sent:        list[worker_id]            # already on the wire (see 3.1)
    emit:        list[(rid_idx, name, modality, uuid)]
    persist:     list[(rid_idx, uuid)]
    new_tokens:  list[(rid_idx, name, uuid)]
    completed:   list[rid_idx]              # worker graph finished this pass
    local_ready: list[rid_idx]              # re-ingested here; already in ready set
```

Everything except `emit` is a flat array consumed by a batched Python call. The
`emit` list stays per-rid because the API server protocol is per-request today;
batching that message is a separate (worthwhile) change.

### 3.1 Peer sends: skip Python entirely

`RawZmqCommunicator::send(&self, peer_id, payload)` (`rust/src/communicator.rs:216`)
takes `&self` and holds its peer table behind a `Mutex`, and `PyZmqCommunicator`
is a thin wrapper over it. So `GraphRuntime` can hold an
`Arc<RawZmqCommunicator>` — the *same* instance Python's communicator wraps —
and put peer edges on the wire from inside `complete_and_route_batch`:

```rust
// attach once at worker startup
fn attach_communicator(&mut self, comm: &PyZmqCommunicator)   // clones the Arc
```

That drops `BatchRouting.to_workers` to a list of peers notified, and buys three
things beyond removing the loop: no Python encode (~11 µs/send), no GIL
round-trip per peer, and the whole fan-out can run under one `py.allow_threads`
instead of one per send.

Two things this needs:

- **A frame discriminator.** The receiving end has to route graph-edge frames to
  its `GraphRuntime` and everything else to Python. Natural end state: peer
  `INPUT_SIGNALS` never enters Python on *either* side — Rust sends, Rust
  receives and ingests (`ingest_batch`).
- **A split frame.** `InputSignals` also carries `request_info:
  CurrentForwardPassInfo` and `partition_name`, which are conductor policy, not
  graph state. Either `GraphRuntime` takes ownership of the per-partition fwd
  info (it already needs `graph_walk` for routing), or the frame carries an
  opaque Python-owned tail that the receiver hands back up. Worth deciding
  early — it sets whether `GraphRuntime` subsumes `PerPartitionInfo`.

Note this win is zero on single-worker configs (no peer sends); it lands on
TP, disaggregated, and colocated-partition deployments.

### The three worker.py sites this collapses

| site | today | after |
|---|---|---|
| `_postprocess_batch` 2315-2368 | B x (`mark_node_complete` + clone + `process_node_outputs`) | 1 x `complete_and_route_batch` |
| `_postprocess_batch` 2395-2402 | B x `_send_outputs`, each B x #peers `communicator.send` | `for w, blob in r.to_workers: send(w, blob)` |
| `_postprocess_batch` 2265-2307 | B x `check_dyn_loop` + `stop_loops` + peer fan-out | 1 x `stop_loops_batch` |
| `micro_scheduler` 343-373 | B x N `get_partition_for_node`/`get_graph_walk`/`get_fwd_info` | 1 x `ready_scan` |
| `_build_executing_batch` 959-971 | B x per-input `tensor_manager.get_tensor` | needs a batched tensor-manager API (below) |

---

## 4. Prerequisites outside the graph layer

These gate the win; none is large, but the port is only worth its full value
with them.

1. **Batched tensor manager.** `store_and_populate_graph_edges` (per rid),
   `set_persist` (per uuid), `set_output_ref_counts` (per rid),
   `get_tensor` (per input per rid) all need `*_batch` forms taking flat arrays.
   Without this, `_postprocess_batch` keeps a B-loop for tensor bookkeeping and
   the graph win is halved.
2. **`u64` tensor handles.** `str(uuid4())` per output tensor is both an
   allocation and a hash-per-lookup. A monotonically-issued `u64` is the natural
   shape once the registry is Rust-side (the SHM arena already is).
3. **A Rust-owned wire format for peer edges.** Per §3.1 the frame is produced
   and consumed in Rust, so the Python `Codec` choice (pickle by default,
   `rust_communicator.py:54`) stops applying to that edge — but the frame needs
   a discriminator so a mixed Python/Rust receiver routes it correctly, and
   `MSTAR_RUST_ZMQ=0` needs a fallback path.
4. **Batched engine readiness.** `MicroScheduler._check_ready` calls
   `engine.check_ready` per (rid, node) and it has side effects
   (`reload_request`, `admit_retrieve`) — it cannot move to Rust. Give the
   engine `check_ready_batch(node, rids, fwd_infos) -> mask` and have Rust
   consult a readiness bitmap that Python refreshes once per scheduling step.

---

## 5. Relation to PR #170

Keep from #170:
- `core/graph.rs` compile step and the `GraphSection` → spec translation in
  `mstar/graph/rust_core.py` — that seam is the right shape and is the tedious
  part. (`rust_spec.py` here is a smaller stand-in for it.)
- `sched.rs`'s decide-vs-mutate split and injected time.
- The shadow/strict/authority rollout ladder (`MSTAR_RUST_WALK=0|shadow|1`).

Change:
- **Granularity.** #170's `WalkState` is per-request and driven per-request from
  Python. Make the owner a per-worker `GraphRuntime` holding a dense
  `Vec<RequestState>`; all entry points take handle arrays.
- **Representation.** `BTreeMap<String, Vec<TensorRef>>` per node per request →
  fixed-size slots indexed by a compile-time input slot, readiness as a `u64`
  mask compare. `ready_nodes()` rescans all nodes on every call; maintain a
  ready bitset incrementally instead (this is what makes `ready_scan` 12 ns/rid).
- **Scope.** #170 stops at the walk machine. `process_node_outputs`'s other half
  — `ShardingConfig.fanout_graph_edges`, ~55% of `process_node_outputs`'s
  cumulative time — has to come too, or edges cross the boundary twice.
- **Rebase surface.** #170 predates `walk_node_to_worker_graph_id`,
  `get_worker_graph_id_for_node(graph_walk=)`, streaming edges/`StreamBuffer`,
  and partitions. The walk-machine core rebases cleanly; the seam files do not.

## 6. Prototype: what it covers

`proto/` builds and runs (`cargo build --release`). It is a full port of the
**request-state machine** — everything in `mstar/graph/base.py` +
`graph_io.py`, minus the dead code in §7.6:

- interned compiled spec; `u64` bitmask readiness; incremental ready /
  ready-for-streaming bitsets, seeded from `only_streaming_inputs`
- **arbitrarily nested loops**: per-level iterate / terminate, recursive
  reset-for-iter and clear of the body subtree, termination cascading to the
  parent, `_cached_outputs` (extends within an iteration, cleared on advance)
  vs `_accumulated_cache` (extends across iterations)
- external-input recording bubbled through *every* enclosing loop and
  re-injected per level on advance
- `NestedLoopIndices`: parent-chain order, all loop indices, `num_times_run`
- speculation: `speculative_signals` slot, `ingest_for_speculation`,
  `is_ready_for_speculation`, `clear_speculative_inputs`
- `ReadySignals.remove` (spec rollback), `pop_ready`, loop stop signals
- replicated/sharded fanout, `complete_and_route_batch`, `ready_scan`

Parity is driven against the real Python objects, and the Rust spec is
compiled from the same `GraphSection` (`rust_spec.py`), so both sides describe
one graph by construction:

- `test_parity.py` — flat decode loop, 6 iterations at B=4, through the real
  `WorkerGraphsManager`.
- `test_nested_parity.py` — `Loop_outer[pre -> Loop_inner[step] -> post]`,
  3x2 iterations through the real `WorkerGraphIO`, comparing ready set, every
  loop's `curr_iter`, emit count, completion count and `num_times_run` after
  **every** node completion.

Nesting costs nothing measurable: 308-410 ns/rid/node on the nested graph vs
~350 ns on the flat one, across inner-loop depths 2 and 8.

Still out of scope — these are `WorkerGraphsManager`-level, not request-state:
- multiple worker graphs / partitions / graph walks per request
- `StreamBuffer` payload semantics (the readiness half *is* modelled)
- TP-follow FIFO, backlog, OOM hold, admit errors
- tensor refcount side effects (see §7.7)

## 7. Semantics — resolved

1. **Completed loop members re-entering `ready_names`** —
   `WorkerGraphStateRegistry.register_ingested_input` (`graph/base.py:735`) reads
   the *still-full* `ready_signals` of a node that already completed this
   iteration, so a loop-back arrival re-adds a popped node to `ready_names`.
   **Resolved: a bug; take the stricter reading (`!completed`).**

   Mechanism, corrected — re-entry after a loop advance does *not* depend on
   this. The advance runs inside `mark_entity_complete`, before outputs are
   routed, so loop-back edges land in freshly-swapped empty slots and mark the
   node ready normally. The re-add is purely premature-ready, and only fires
   when a member completes before the loop's *last* entity. Repro (width=2):
   after `sampler` completes, `llm` sits in `ready_names` holding its
   already-consumed iteration-N tensors while the fresh ones wait in
   `ready_next_iter`; popping it there runs it on stale inputs.
2. **`process_node_outputs`'s done-sweep** — it scans *all* of the request's
   worker graphs for this walk on every node completion. **Open, leaning
   O(1):** a worker graph should only be able to become done off a completion
   *inside* that worker graph, which would make a per-(request, wg) completion
   counter sufficient and the comment above the loop a misconception. Needs a
   pass over the cases that comment cites (Orpheus prefill, BAGEL vae_decoder,
   Code2Wav) before committing.
3. **Tie-breaking** — **Resolved: don't preserve it.** Scheduling is heuristic
   and unmotivated today; the Rust ready-bitset's node-index order is as good a
   tie-break as dict order. If it shows up as a deficit in benchmarking, fix it
   then, deliberately.
4. **`_speculatively_scheduled`** — **Resolved:** a per-request flag, not a field
   on the shared `GraphNode`. Lives in `RequestState` in Rust (it already is in
   the prototype); a `(request, node) -> bool` map is the equivalent if any of
   this stays Python during the transition.

5. **Nested loops with declared `outputs` are broken in Python.** Found while
   porting `Loop.complete_iter`, reproduced in `test_nested_parity.py`
   (`check_known_divergence`). Two ordering faults, both only reachable when an
   inner loop's termination also terminates its parent:

   - `complete_iter` calls `self._managing_registry.mark_entity_complete(self.name)`
     **before** populating its own outputs' `tensor_info` (`base.py:535-539`),
     so the parent's `maybe_cache_output` snapshots the child's still-empty
     edges — and keeps whatever stale value an earlier entity left under that
     name.
   - the return value of that call is **discarded**, so a parent that finishes
     in the same cascade never routes its declared outputs at all.

   Demonstrated on `Loop_outer[pre -> Loop_inner[step]]` with `acc` declared by
   both: the outer loop's `acc -> sink` ends up carrying `pre`'s intermediate
   value, and never appears in the routed edges — while the worker graph
   reports done regardless. Strongly suggests no shipped model uses a nested
   loop with declared outputs.

   The prototype implements the intended behavior: populate, then cascade, and
   append the parent's outputs to the same routed list. Worth fixing in Python
   independently, or at least asserting against at graph-construction time.

6. **Dead code not to port.** `WorkerGraphStateRegistry.reset_for_iter`
   (`base.py:769-780`) never runs — measured 0 calls. `reset_for_iter` is only
   ever reached as `self.inner_registry.reset_for_iter()` (`base.py:507, 635`)
   and `inner_registry` is always a `LoopStateRegistry`. So the root registry's
   ready-set swap is unreachable and `ready_next_iter` /
   `ready_streaming_next_iter` on the root are write-only (written at
   `base.py:750-756`, read nowhere).

7. **Refcount side effects move to the boundary.** `ReadySignals.clear` →
   `dereference`, `maybe_cache_output` → `increment_ref`, `_uncache_outputs` →
   `dereference` are threaded through the state today. Rust must still *emit*
   them, but as flat arrays returned to a batched tensor-manager call —
   inline per-tensor calls would reintroduce the crossing the port removes.
   `_persist_for_loop` is carried on `RoutedEdge` for the same reason.

8. **`is_ready_for_streaming` is trivially true.** `ReadySignals.update`
   (`graph/base.py:183`) guards with
   `input_names.issuperset(ready_names | streaming_inputs)`. Both operands are
   always subsets of `input_names`, so the test never fails: **every** node is
   "ready for streaming" after its first ingested input, streaming or not. The
   comment directly above states the intent — `issubset`, i.e. ready once the
   only missing inputs are streaming.

   Found by the shadow harness on its first run. Consequence: the gate in
   `process_new_streaming_inputs` is far more permissive than intended. The
   Rust core implements the intent, so the two disagree by construction and
   `ready_for_streaming` is excluded from `COMPARED_FIELDS`. Fixing Python is a
   behavior change the streaming models (Orpheus, Code2Wav, Qwen3-TTS) have to
   be re-validated against — worth doing, separately.

9. **A promoted next-iter slot never re-enters `ready_names`** — the latent
   half of §7.1. When a loop member's next-iteration inputs have *all* arrived
   before the loop advances, the advance promotes them (`reset_for_outer_iter`
   swaps the slots) and the node genuinely holds every input — but nothing
   re-advertises it, because the promotion that would (`WorkerGraphStateRegistry.reset_for_iter`)
   is the dead code in §7.6. The node is runnable and invisible to the
   scheduler until some later ingest happens to call `register_ingested_input`.

   Pinned by `test_buffered_next_iter_is_invisible_to_python_scheduler`.

   **The two bugs are coupled**: the premature re-add in §7.1 is what normally
   papers over this. Taking the stricter reading *without* also making the
   advance recompute readiness would turn a latent stall into a real one. The
   Rust core recomputes on advance, so it is correct on both counts — but this
   is the reason §7.1 cannot simply be "fixed" in Python on its own.

## 8. Staging

### 8.1 Python-side fixes — done

Eager arguments to disabled `logger.debug` calls on per-request paths:

| file | what |
|---|---|
| `communication/communicator.py:135` | `str(msg)` per send → lazy (~10 µs/send) |
| `communication/rust_communicator.py:156` | same |
| `graph/graph_io.py:5` | `format_graph_edge_list` returns a lazy proxy; covers all 3 call sites (`node_manager_utils.py:520`, `worker.py:623`) |
| `communication/tensors.py:489` | two eager list comps per output per rid → `isEnabledFor` guard |
| `worker/worker.py:1883,2868,3118` | f-strings in the schedule/speculate/yield paths → lazy `%s` |
| `model/bagel/submodules.py:1265,1625` | eager tensor repr per forward |

Effect: halves every `_send_outputs` message (table in §1). It does **not**
move `process_node_outputs` — re-measured at 21.7 µs/rid, unchanged — which is
the point: there is no Python-level fix for that loop.

### 8.2 The port

1. `GraphRuntime` + compile seam + `add_requests`/`remove_requests`, shadow-mode
   only. Kills the `deepcopy` cost first and is independently shippable.
2. `complete_and_route_batch` behind a flag, peer edges still decoded to Python
   `GraphEdge`s (parity-checkable against `process_node_outputs`).
3. Batched tensor manager + `u64` handles.
4. Attach the Rust communicator (§3.1); peer edges leave the Python path on the
   send side, then on the receive side.
5. `ready_scan` + batched engine readiness; retire the Python scan.
6. Speculation surface last — it is the most entangled with torch-side state.

---

## 9. Integration status

### 9.1 The harness

`mstar/graph/runtime/` is the seam Naomi asked for: one ABC, swappable
backend, so shadowing and head-to-head both fall out of the same interface.

```
GraphRuntimeBase          one worker graph's walk state, for every request
 ├─ PythonGraphRuntime    today's WorkerGraphIO behind the interface
 ├─ RustGraphRuntime      mstar_rust.GraphRuntime; interns rids and uuids
 └─ ShadowGraphRuntime    primary answers, shadow mirrors, diffs reported
```

`make_graph_runtime()` picks from `MSTAR_RUST_GRAPH`; a worker graph the Rust
core cannot compile falls back to Python *for that graph alone*, logged.

Two details worth reviewing:

* **`PythonGraphRuntime` writes into the caller's dict.** `WorkerGraphQueues`
  passes its own `per_request_queues` as `io_store`, so the call sites that
  still reach into `WorkerGraphIO` directly (`micro_scheduler`, the worker's
  speculation path) keep seeing the same objects. That is what made this
  integration small; it is also why `MSTAR_RUST_GRAPH=1` is not yet usable
  end-to-end (§9.3).
* **`snapshot()` is the parity primitive** and `COMPARED_FIELDS` says which
  fields are expected to agree. `ready_for_streaming` is excluded on purpose
  (§7.8), and an asymmetry in `ready` is classified rather than failed (§7.1),
  because the two backends disagree there *by design*.

### 9.2 Measured, through the harness

Request admission — the `deepcopy` goes away, and this is available now:

```
       graph     python       rust   speedup
  decode w=0      271.8 us    0.75 us     361x
  decode w=4      628.1 us    1.22 us     516x
      nested      382.0 us    0.76 us     502x
```

Per-node completion, **per-event API** (`complete` once per rid):

```
       graph     B     python       rust   speedup
  decode w=0     8       4040       5701      0.7x
  decode w=4    64       3414       6846      0.5x
      nested    64       4548       5114      0.9x
```

**The per-event path is a regression, and that is the expected result.** One
PyO3 crossing per rid per event, plus uuid interning, plus rebuilding
`GraphEdge` objects, costs more than the Python it replaces. The 59-126x in §1
came from `complete_and_route_batch` — one crossing per *batch*. So the
integration as it stands buys the admission win and the parity harness; the
throughput win needs §9.3.

This is the whole thesis of the port restated as a measurement: the win is
batching, not Rust.

### 9.3 What is not wired yet

`MSTAR_RUST_GRAPH=1` works at the graph layer (`WorkerGraphQueues` routes
everything through the harness) but **not end to end**, because two call sites
still reach past it into `WorkerGraphIO`:

* `micro_scheduler.pop_ready_rids` / `has_ready_excluding` —
  `queue.per_request_queues.get(rid)`
* `worker._get_wgio_for_rid` and the speculation path, which manipulate
  `GraphNode.ready_signals` / `_speculatively_scheduled` directly

Under the Rust backend `per_request_queues` is empty, so those see nothing.
Shadow mode is unaffected (Python is authoritative and its dict is populated).

Order of work from here:

1. Migrate those two call sites onto the harness — `pop_ready`,
   `ready_nodes`, and a speculation surface for the worker's rollback.
2. Switch `_postprocess_batch` to `complete_and_route_batch`. This is where
   the throughput win is; it needs the batched tensor-manager APIs in §4.
3. Then `u64` tensor handles, and the Rust-owned peer send in §3.1.

---

## 10. Batched tensor manager + batched postprocess

### 10.1 Batched tensor-manager APIs

Four batched forms on `TensorCommunicationManager`, all **default-implemented
on the base class** as loops — every transport gets the surface for free, and
only the arena needs to override:

| method | default | who overrides |
|---|---|---|
| `store_and_populate_graph_edges_batch` | loop, one CUDA sync hoisted | — |
| `set_persist_batch` | loop over flat `(rid, uuid)` | — |
| `set_output_ref_counts_batch` | loop | — |
| `register_for_send_batch` | loop, one CUDA sync hoisted | `ArenaShmCommunicationManager` |

The arena override is the one that matters. `register_for_send` ends in a
host-blocking `self._d2h_stream.synchronize()`, so calling it once per request
costs **B serialized stalls per forward pass**. The batched form stages the
whole batch inside one `_d2h_ctx()` and syncs once. The per-tensor body is
factored into `_stage_one`, so both forms share it and cannot drift.

### 10.2 `WorkerGraphsManager.complete_and_route_batch`

Completes a node and routes its outputs for the whole batch, hoisting
everything that is keyed by (node, walk) rather than by request:

- the destination worker-graph lookup (`walk_node_to_worker_graph_id`)
- the sharding group and `is_first_tp_rank`
- the walk-filtered worker-graph sweep set
- the `NodeAndGraphWalk` keys for external destinations
- **the fanout plan for every replicated signal** —
  `ShardingConfig.plan_replicated_fanout` splits "which workers" (dims-
  independent, so computed once per signal) from "materialize the edge"
  (per request). A sharded signal falls back to `fanout_graph_edges`, whose
  slicing genuinely depends on each request's tensor dims.

What stays per request: the completion, the edge clones, and any sharded slice.

`process_node_outputs` and the batched form share both halves —
`_routing_context` (the hoisted lookups) and `_route_one` (the per-request
work) — so the single-request and batched paths cannot drift.

Hoisting the sharding config assumes every request in the batch shares one.
That holds whenever the conductor handed them the same
`worker_graph_to_workers` map — always, today — but rather than rely on it,
a batch spanning more than one config falls back to the per-request path.

Also removed: `process_node_outputs` built a `my_node_names` set on every call
(union of `all_worker_graph_ids_to_nodes` over the request's worker graphs) and
never read it.

`test_complete_and_route_batch_matches_the_per_request_path` asserts the
batched result is identical to routing one request at a time, field by field
including `_shard_dim`, `_total_fanin` and tensor uuids.

### 10.3 Measured

`python docs/design/rust_graph_port/bench_postprocess.py` — mark-complete +
route, ns per request per node completion. Min of 5 runs with GC disabled
during timing; both paths allocate an edge clone per output per request, so a
collection landing in one run and not the other swamps the difference (an
earlier version of this benchmark reported 0.49x-2.61x noise for exactly that
reason, and also never advanced the loop, letting `_cached_outputs` grow
without bound).

```
 width  TP     B  per-request    batched   speedup
     0   1     8        17675      12486     1.42x
     0   1   128        18944      13144     1.44x
     0   2   128        18798      12762     1.47x
     4   1    32        17833      11795     1.51x
     4   1   128        18641      12344     1.51x
     4   2   128        18401      12207     1.51x
```

**~1.4-1.5x, and this is the Python backend** — pure hoisting, no Rust
involved. At B=128, width=0 that is ~5.8 µs/rid/completion saved, so with two
completions per decode step, ~1.5 ms per step.

This is the first actual throughput win in the port. It is also the ceiling of
what restructuring Python can give: the remaining ~12 µs is the per-request
completion and edge cloning, which is what §9.2 showed the Rust backend can
only beat once it owns the routing too (one boundary crossing per batch rather
than per request).

---

## 11. The routing wrapper, and why its Rust backend is still empty

### 11.1 The seam

Routing now goes through a wrapper, same shape as the graph runtime one level
below it:

```
mstar/graph/runtime/    per worker GRAPH  — walk state      (python | rust | shadow)
mstar/graph/routing/    per WORKER        — routing         (python | rust | shadow)
```

`WorkerGraphsManager.complete_and_route_batch` delegates to
`make_batch_router(self)`, so the worker calls the wrapper and the wrapper
picks the backend. `RoutingWorld` states what a router needs from the worker,
which keeps the dependency one-way.

Routing is one level up from the runtime because it needs what a single worker
graph cannot see: which *local* worker graph owns a destination node, the
request's `ShardingConfig`, and the cross-worker fanout.

Note the shadow router is weaker than the runtime's: routing **mutates** state
(the completion advances the walk; local edges are ingested), so two backends
cannot both run against one world. It only engages once a backend keeps its
own state — i.e. the Rust case.

`_routing_context`'s 5-tuple of closures is gone; it is a `RoutingContext`
object with `dest_wg()`, `fanout()` and `node_and_walk()` methods, built once
per batch and shared with the single-request path.

### 11.2 Why `RustBatchRouter.build()` returns None

Two prerequisites, the first of which is measured rather than assumed:

**The return shape has to stop being per-request Python objects.**
`NodeOutputRouting` holds `GraphEdge`s with real `TensorPointerInfo` lists.
Constructing them is **27% of the batched routing path**:

```
batched path              11695 ns/rid/completion
without GraphEdge.clone    8574 ns      -> 27%
```

A Rust backend handing that shape back pays the 27% *plus* a boundary crossing
*plus* rebuilding the tensor infos — so it starts behind, and §9.2 already
showed a per-request crossing runs 0.5-0.9x. `_send_outputs` and
`_register_outputs` have to consume flat, batch-shaped data first: per-worker
edge groups and flat `(rid, uuid)` arrays, which is what
`mstar_rust.BatchRouting` already returns.

**The Rust runtime has to be per-worker, not per-worker-graph.** Today
`RustGraphRuntime` compiles one section and owns only its own states, and
`shard.rs`'s `ShardMap` is a single trivial group rather than the real
`ShardingConfig`.

So the order is: batch the *consumers* → promote the Rust runtime to
per-worker → then the Rust router is worth building. Building it first would
produce something that is, at best, break-even.

### 11.3 `MSTAR_RUST_GRAPH=1` progress

`WorkerGraphsManager.stop_loops` no longer reaches past the harness for its
`loop_stop_times` snapshot — the harness gained `nested_loop_idxs(rid,
loop_name)` and `loop_names()`, with `nested_loop_idxs_for_loop` on the Rust
side. That removes one of the three §9.3 production reach-ins.

Under `MSTAR_RUST_GRAPH=1`, `test_worker_graphs_manager` now fails 2 rather
than 3, and both remaining failures are *test-side* assertions reaching into
`per_request_queues` directly, not production paths. The two production
reach-ins left are `micro_scheduler` and the worker's speculation path.

---

## 12. The Rust runtime is now per-worker

### 12.1 What changed

`mstar_rust.GraphRuntime` owned one compiled section. It now owns **every
worker graph on a worker**:

```rust
pub struct GraphRuntime {
    graphs: Vec<GraphRef>,                  // was: graph: GraphRef
    states: Vec<Vec<Option<RequestState>>>, // [wg][handle]; None = not registered there
    rid_to_handle: FxHashMap<String, u32>,  // handles are worker-wide
    node_owner: FxHashMap<(Sym, Sym), u32>, // (walk, node) -> owning wg
    ...
}
```

Every method takes the worker-graph index first. The section-compiling code
moved out of the constructor into a free `compile_one`, which the constructor
now runs once per graph.

Two consequences that are the point of the change:

* **`owner_of(walk, node)`** answers "which *local* worker graph owns this
  destination", which routing needs and a per-graph runtime cannot know. An
  edge leaving the worker and an edge crossing to a sibling graph on the same
  worker look identical from inside one graph.
* **Request handles are worker-wide.** One handle addresses a request in every
  graph it belongs to, so the per-graph handle maps are gone.

State is `Option<RequestState>` per (wg, handle) because Python only registers
a request with the graphs its partition uses; `remove_requests` recycles a
handle only once no graph still holds state for it.

### 12.2 Python still talks per worker graph

The harness interface is unchanged — it stays at `WorkerGraphIO` scope, as it
should. What changed is who builds it:

```
RustWorkerRuntimes     one per worker: the extension object, the handles,
                       the uuid interner
  .view(wg_id)   ->    RustGraphRuntime, which passes its own `wg` index
```

`install_graph_runtimes(queues, worker_id)` is the worker-level entry point,
called from `WorkerGraphsManager.__post_init__` — the first place every worker
graph is known. Under `MSTAR_RUST_GRAPH=0` it is a no-op and the queues keep
the Python runtimes they built themselves. Compilation is all-or-nothing
across a worker's graphs: routing needs every local graph, so one unsupported
graph falls the whole worker back to Python rather than leaving a hole.

### 12.3 Verified

Two worker graphs on one worker, `encoder` in wgA feeding `llm` in wgB:

```
owner of ('prefill','llm')     -> 1      (wgB)
owner of ('prefill','encoder') -> 0      (wgA)
owner of ('prefill','remote')  -> None   (not on this worker)
handle shared across views: {'r0': 0}
wgA completes encoder -> emits ('emb','llm') -> wgB ready: {'llm'}
```

All three modes still behave: `0` and `shadow` green (including
`MSTAR_RUST_GRAPH_STRICT=1`), `1` unchanged at 2 test-side reach-in failures.

### 12.4 Next

`complete_and_route_batch` on the Rust side can now resolve `Dest::External`
against `node_owner` and ingest into a sibling graph instead of reporting the
edge as leaving the worker. That plus the `ShardMap` being built from the real
`ShardingConfig` (it is still one trivial group) is what makes the Rust router
in §11 possible — after the return-shape work, which is still the thing that
decides whether it wins.
