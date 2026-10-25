---
name: mstar-way
description: Derive how a feature should be built in mstar from first principles, by expressing it in mstar's own primitives (graph nodes, loops, walks, partitions, streaming edges, declared resources, check_stop) before adding any mechanism. Describe one request's timeline in plain words, translate each sentence through a shape table, write the declaration a model would make, extend a primitive where it falls short instead of building beside it, then run a disguise check and a "free or patched" check on the runtime's machinery. Use before designing any mstar feature, engine/worker/scheduler/conductor/API change, new resource or model port; when a design or PR adds a mechanism in a shared layer; or when the user asks "what's the mstar way", "what primitive is this", or invokes /mstar-way.
argument-hint: "<feature or design to derive>"
---

# The mstar way

mstar is a dataflow runtime. A request is a walk through a graph of nodes. The runtime already
schedules, batches, speculates, pre-plans, captures, streams, evicts, parallelizes and tears down
**anything expressed as graph structure, edges and declared resources**. So the first design
question for a feature is never "what mechanism do I add?" It is "what is this feature, written
as graph structure, edges and resources?"

A feature written that way inherits all of that machinery. A feature built beside the graph has
to reimplement or patch each piece: speculation that can't see it, a scheduler special case, a
cursor dict, outputs held by hand, a lifecycle state. An AI's first instinct is the change that's
easiest locally, usually a mechanism in whichever layer it read first. This skill is the
procedure that gets past that instinct.

Work through §1–§6 in order, in writing. Don't skip §1 because the answer seems obvious: the
point of §1 is to see the shape before the existing code suggests where to bolt something on.

## 0. Learn the primitives from the code, not from precedent

Read [references/primitives.md](references/primitives.md): the catalog, with file pointers on
`main`. Check each primitive you'll rely on in the code itself.

**Precedent isn't proof of fit.** A recent mechanism, an open PR, or an example in `AGENTS.md`
or a doc may itself have been built beside a primitive. Judge every existing mechanism by the
same test in §4 and §5 before you copy its shape. Only the primitives in the catalog are given.

## 1. One request's timeline, in plain words

Write 5–10 sentences about what happens to **one request** over time, with no mstar class names:

- what computations run, and in what order;
- **how many times** each one runs for this request, and whether that count is known when the
  request arrives;
- what each run consumes, and **when that data exists**: all at the start, or arriving over time,
  and from whom (another computation, outside the server, a clock);
- what has to be **remembered between runs**, how large it grows, and whether other requests
  could share it;
- **what decides** that a phase, or the request, is over;
- what goes out, and when: at the end, or as it's produced;
- what really differs per model, and what is the same for every model.

Also write what the feature is **not** about. Scheduling knobs (budgets, fairness, priorities)
decide *how much* work a step does and *which* rows run. They're tuning on top of the shape, not
the shape.

## 2. Translate each sentence through the shape table

| If the timeline says… | …the primitive is |
| --- | --- |
| A computation **runs more than once for one request**, each run continuing from the last | a `Loop` over that node (or section). The per-run state lives in a declared resource, the stop comes from `check_stop` (or `max_iters` when the count is fixed), and per-run outputs go out on edges (`EMIT_TO_CLIENT` each iteration, or `accumulated_outputs` to emit them all at the end). A node that runs N times for a request *is* a loop, whatever triggers each run. |
| The number of runs is known when the request arrives | still a `Loop`. Exit early in `prepare_inputs` on the overshoot step so the async worker's speculation costs no compute. |
| Data **arrives over time**, from any producer (a node, another partition, or outside the server), at a rate the consumer doesn't control | a **streaming edge** into the consuming node: a `Connection` with a chunk policy, buffered in a `StreamBuffer`. The producer is a partition. A node whose stream hasn't delivered isn't ready yet, which is readiness, not a new state. |
| Two parts of a request **advance at their own rates** | **async partitions**, joined by streaming edges. |
| Different phases or inputs need **different computations** | separate **graph walks**, with the conductor moving the request between them. Walks that should share a batch are **combined walks**. |
| Something must be **remembered across runs** (per request, per token, per layer) | a **declared resource** (`get_node_resources`, with each step's effect in `declare_step`). The engine admits, plans, commits, evicts, offloads, prefix-caches and applies backpressure. Never a tensor, dict or cache owned by the model, worker or engine. |
| A resource needs a **different access pattern** (a window, a sparse selection, a different layout, another tier) | a new **resource kind or a variant of one**, in `mstar/engine/resources/`, planned in `plan()` and used through its own access method. Variants of one role (attention flavours, state-space flavours, memory tiers) are one resource with the difference as config or a backend, not sibling copies of a lifecycle. |
| **Which requests** run this step, and **how big** each one's step is | the micro-scheduler (`can_batch`, `input_seq_len`, its budgets). It sizes and picks node steps. It never decides that a node runs again for a request: that's the graph's job. |
| **Something is done** | `check_stop` returning loop names, or the walk ending. Aborts, failures and disconnects use the existing teardown paths. |
| **Where** something runs (GPU, rank, TP width) | config `node_groups`, never code. |
| Behaviour that **differs per model** inside a shared mechanism | a declaration or a policy object with a default, which the engine reads. Never a method every model must implement when the engine could work the answer out from the module tree or a declaration. |
| A fixed-shape step repeated many times | an accelerator-graph config. The engine captures it. |

If a sentence fits no row, write down the missing primitive. Don't build around it yet (§4).

## 3. Write the declaration

Write what a model would declare, as a code sketch: `get_graph_walk_graphs` (nodes, `Loop`s,
edges), `get_partitions` / `get_partition_topology` (with each `Connection`'s chunk policy),
`get_node_resources` and `declare_step`, and `check_stop`. Then write the per-model adoption cost
as a list. The target is declarations only.

If the whole feature fits in declarations, the design is done, and the shared-layer diff should
be close to empty.

## 4. Extend the primitive. Never build beside it

List exactly what the primitives in §3 can't do yet. For example: a loop whose node takes a
different slice of an input each iteration, a loop that waits for more streamed input, a
streaming producer outside the graph, or a resource with a new access pattern. That list is the
shared-layer work, and each item is a **generic extension of an existing primitive** that every
model gets.

A gap in a primitive is a reason to extend it, not to route around it. A design with a short
list of primitive extensions beats one with zero extensions and a mechanism beside the graph:
the mechanism beside the graph will grow a patch per runtime feature (§6).

## 5. Disguise check

For every mechanism the design (or a PR) puts in the worker, engine, scheduler, conductor or API
server, ask which primitive it is in disguise:

| Looks like | Is really |
| --- | --- |
| A node pushed back onto the ready queue for the same request, a per-request cursor or progress counter, outputs held until a final pass | a `Loop` built outside the graph runtime |
| Telling the graph runtime not to speculate into or look at something, or rebuilding readiness by hand | structure the runtime can't see, so put it in the graph |
| Input buffered in the API server or conductor, windows cut by the server, persistent "sessions", polling for more input | a streaming edge |
| A per-request dict, tensor or cache in model, worker or engine code | a declared resource |
| A new lifecycle state ("waiting", "paused", "abandoned", "partial") | readiness on an edge, or the existing abort and teardown paths |
| A data-dependent loop inside `forward`, with a host sync | a graph `Loop` with `check_stop` and a resource |
| A second copy of an existing lifecycle (a second cache, a second manager beside a near-identical one) | one primitive that is generic over the variants |
| A hook every model must implement for a cross-cutting feature | the engine deriving the answer, or a declaration |
| A feature that needs special cases in several layers with no graph declaration | a feature at the wrong layer |

Each hit is the most expensive kind of design error, because later work builds on its shape.
Rewrite the design through §2 until nothing hits.

## 6. Free or patched

Take at least two candidates: the graph-native design from §3–§4, and the locally easiest one.
For each, go down this list and mark every row **free** (works because the runtime already
handles graph structure, edges and resources) or **patched** (needs its own code):

- the async worker: speculating step N+1 and pre-planning it;
- batching with other requests' steps, decode and prefill mixed;
- accelerator-graph capture;
- TP lockstep (every rank takes the same branch);
- the Python and Rust graph runtimes, with no drift between them;
- streaming outputs to the client;
- prefix caching, eviction, offload and backpressure;
- abort, failure, disconnect and teardown, including handle-keyed state;
- placement across GPUs (`node_groups`) and disaggregation;
- a second model adopting it (count what that model writes).

The design with more **free** rows and fewer new concepts wins. Every **patched** row in the
winner needs a stated reason. A design whose patched rows are mostly "tell X about the new
mechanism" is built beside a primitive: go back to §2.

## Output

1. **The timeline** (§1).
2. **The shape:** each timeline sentence, with its primitive (§2).
3. **The declaration sketch,** and the per-model cost (§3).
4. **Primitive extensions needed,** each generic (§4).
5. **New concepts:** one line each on why no existing one does the job. Delete any you can't
   justify.
6. **The free/patched table** for the chosen design and at least one rejected one (§6), plus a
   one-line verdict on why the rejected one loses.

For a review, apply §1–§6 to the PR's feature and report each §5 hit as a finding that names the
primitive.

## Worked examples (existing mstar designs, and two past misses)

- **Autoregressive decode.** Timeline: the LLM runs once per output token, an unknown number of
  times; each run consumes the previous token; the keys and values of every token so far are
  remembered; the request ends at EOS or a token cap; each token streams out as it's produced.
  The shape: a `Loop` over the `LLM` node with a loop-back edge, a KV resource, `check_stop`
  for EOS, and an `EMIT_TO_CLIENT` edge per iteration (`mstar/model/qwen3_5/qwen3_5_model.py`,
  `decode_loop`). Every runtime row in §6 is free.
- **A talker that speaks while the thinker is still writing** (Qwen3-Omni, `nemotron_duplex`).
  Timeline: the talker consumes the thinker's tokens as they're produced, at its own rate, and
  outlives or matches the thinker. The shape: two async partitions and a streaming edge with a
  chunk policy (`get_partition_topology`). No coordinator object.
- **A flow or diffusion denoiser.** It runs a fixed number of steps, known at ingestion: a
  `Loop`, with the overshoot step exited early in `prepare_inputs`.
- **Past miss, multi-LoRA.** First instinct: every model implements LoRA hook methods. The
  timeline says the behaviour is the same for every model (each linear layer adds a per-request
  low-rank delta) and that adapter weights are per-request state. The shape: the engine wraps
  linear layers by name, the adapter pool is a resource, and a model declares at most its
  target-module names.
- **Past miss, a streaming encoder's attention cache.** First instinct: the model keeps a tensor
  of the last W frames' keys and values. The timeline says "remembered across runs, bounded
  window", which is a declared KV resource with a window, so the engine owns it and it gets
  capture, eviction and batching like any other KV.
