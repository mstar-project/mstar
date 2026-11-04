# Preemption under KV pressure

Since M* has no admission control today, requests take pages as they prefill and decode, and free them only when they finish. Once every running request in a full pool needs another page, none can finish, and all wait until clients time out (#307).

Earlier PR (#400) proposed admission control that reserves each request's worst case --- its prompt plus `max_tokens`. It removes the stall completely, but induces severe HoL blocking (e.g. chats queue behind image requests): their TTFT p50 rose from 0.45 s to 17.7 s. Whether this fragments the pool was never measured, but it is nonetheless based on a banker check which felt too strict.

This design adds slim version of admission control by default (A1). A request reserves only its prompt's pages and waits in a queue until they fit. On top of it, the requests are classified as either:

1. **Guaranteed.** A request that streams an output that can't pause, such as audio, or whose pages nothing could take back, also reserves pages for the most it can generate, and is never preempted.
2. **Preemptible.** When a request runs out of pages, newer preemptible requests free pages for it. Each moves to the host when the host pool can hold all of its pages, and is otherwise dropped and later rebuilds its KV from token ids. A request that can't be rebuilt moves only to the host, into room the host set aside for it.

A host tier for the prefix cache, which was on the roadmap regardless, is built alongside, and aims to cut that recompute by copying pages between host and device asynchronously, hiding the offload and reload cost.

| Part | PR | Section |
|---|---|---|
| Reservations and a queue per pool | A1 | §2 |
| The rebuild row | R1 | §5 |
| Preemption on a failed grant | R2 | §3, §4 |
| Guaranteed and preemptible requests | R3 | §1 |
| TP followers | R4 | §6 |

---

## 1. Classes

```python
@dataclass
class KVReqConfig(ResourceReqConfig):
    ...
    unpausable: bool = False   # streams an output that can't pause, set by the conductor
    rebuildable: bool = True   # a drop on this worker can free its pages, set by the worker
```

A dropped request keeps only its token ids, so a request can be dropped only if its node can rebuild all of its KV from them. One that can't be dropped can still move to the host.

A request that nothing can take pages back from must never run short. It reserves its whole footprint up front, its prompt plus its growth bound, and is never preempted. That is what guaranteed means.

The model declares the footprint as KV spans (#832). A span's kind says what its KV is made from:

- `ids`: token ids
- `digest`: an encoder's output
- `rows`: another node's outputs

Each pool decides the class once, at the request's first `ask`, from two facts. The conductor marks a request that streams an output that can't pause, and that mark holds in every pool. The worker marks whether a drop there can free the request's pages: every node of the pool that writes the request's KV on that worker has a policy that can write all of it again. A request no drop can free is preemptible when the host pool can set its footprint aside, and guaranteed otherwise.

| # | A request with | Class | Why |
|---|---|---|---|
| 1 | An output it streams that can't pause, audio unless the model or the deployment says otherwise | guaranteed | a pause is a gap in playback |
| 2 | A `rows` span from another node's outputs | moves only to the host there, or guaranteed | its KV comes from another node's outputs, such as the Thinker's hidden states the Qwen3-Omni Talker reads, and no policy on its node rebuilds it |
| 3 | A `digest` or `rows` span on a walk of its node's rebuild policy | preemptible at one rank, and elsewhere moves only to the host there, or guaranteed | its KV comes from an encoder's output, freed after prefill, such as a Qwen3.5 image's or video's, so a rebuild needs a host copy of it |
| 4 | No rebuild policy on the node that writes its KV | moves only to the host there, or guaranteed | the model declares no way to rebuild the node's KV, as when its decode reads embeddings |
| 5 | Only `ids` spans | preemptible | its token ids reproduce all of its KV |

R5 keeps a host copy of the encoder's output for the rebuild of a row 3 request. R5a covers Qwen3.5 images, and R5b covers Bagel, Qwen3-Omni and Higgs. Without a copy, a rebuild of row 3 would have to run the encoder again, which on Bagel and Qwen3-Omni sits on another worker, so such a request stays guaranteed.

On initial tests, I measured that keeping row 3 guaranteed costs about 9% of TTFT under pressure. On a 48-page pool at c16, Qwen3.5 image requests' TTFT p50 fell from 3.48 s to 3.18 s once R5a made them preemptible.

Why this shape:

- **The class comes from the request.** A priority or a per-request flag would be one more input to tune, when the footprint and the output can be used to make the decision.
- **An output that can't pause holds in every pool.** A Qwen3-Omni request with streamed audio out must not be dropped by the Thinker's worker while its Talker is speaking, so the conductor marks it. A span no rebuild can write again holds only in its node's pools: for an unstreamed reply, a Thinker rebuild only delays the Talker's rows.
- **Only streamed outputs that can't pause.** The model declares them (`Model.UNPAUSABLE_OUTPUTS`, audio by default), and a deployment overrides them with `unpausable_outputs`, as an Orpheus deployment may to keep a fast first chunk. A streamed text reply that pauses is late, not broken, and an unstreamed audio reply is played after it arrives.
- **Room on the host is set aside, not assumed.** A request that can only move to the host must be able to move when it is the oldest one short (§5.5). Room counted at that moment is whatever other moves left, so a full host pool would leave it holding its pages, and a pool of such requests stalls as in #307. Its footprint set aside at its first ask makes its move fit. Past `cpu_offload_pages`, the rest are guaranteed.
- **Not the worst case for every request.** With 32 Bagel text chats beside 16 image chats on 64 pages, chat TTFT p50 was 559 ms with every request guaranteed and 69.7 ms with two classes.
- **Seeded requests stay preemptible.** Guaranteeing them would charge the worst case to every request that sets a seed. A rebuilt seeded request may not repeat its output (§5.4).

---

## 2. Reservations

`KVManager._alloc` ([manager.py](../../mstar/engine/resources/kv/manager.py)) checks every one of the grants against the reservations, under the KV lock which it already holds. A grant is `_alloc` giving a request the next page to write into. Each pool gains:

```python
# per pool, under the KV lock
_allotments   # each admitted request's Allotment, its reservation
_unused       # pages reserved but not yet used, summed over requests
_asks         # the pages each request that asked and has no reservation asks for
```

The code calls a reservation an `Allotment`, since `manager.py` already uses "reservation" for an admit's unwind record.

Rules:

- **Ingest.** A request whose footprint is bigger than its pool gets a 400 at ingest. A request that asks for no length has its default cut to fit the pool (#808).
- **Queue.** A request joins the runner's waiting list at its first readiness check, by its arrival number (§3.1), or behind every numbered request in the order it first asks when it has none. A request that is dropped or moves to the host asks again at once, at its arrival. It is admitted when it is first among the waiting requests of every pool it asks, and its reservation fits in each, in the free and evictable pages no admitted request has reserved.
- **Size.** A preemptible request reserves its prompt. A guaranteed one also reserves its growth.
- **Inside a reservation.** A grant always succeeds. A failure there is an accounting bug, and the manager raises.
- **Past a reservation.** A grant takes only from the room, and fails with `AllocationFailed(pages_short=...)` when the room is short.
- **Across pools.** A request takes its reservation in every pool its node asks, the recurrent pool's slot and the sampler's in-graph row included, or in none, and nothing is retrieved for it before then. A waiting request is asked again only after a pool it waits on has grown.

```
room = free + evictable - _unused

invariant, after every grant, reservation, lease and release:
    _unused <= free + evictable
```

What a cache hit costs:

| A hit on | Costs |
|---|---|
| a page only the index holds | a fresh page, out of the room alone for a guaranteed request until it adopts its lease |
| a page another request holds | nothing, lent |
| a cached prefix copied from the host at readiness | a grant within the request's reservation |

Why this shape:

- **The rule lives in `_alloc`.** Today it grants any free page, so a growing request could take a page a guaranteed request reserved, and that request's next grant would fail.
- **Only the head is admitted.** Nothing newer passes the oldest request, so it can't wait forever behind a stream of small ones (§5.5).
- **Evictable hits are charged as fresh.** Eviction could hand such a page to another request before the probe leases it.
- **A guaranteed request is charged for no hit until it adopts its lease.** A cut lease leaves the walk to write the cut pages, and a cut page another request still holds frees no room, so its reservation keeps every page until then.

---

## 3. Preemption

Preemption starts when a grant past a reservation fails, in `Worker._handle_allocation_failure` ([worker.py](../../mstar/worker/worker.py)), which calls `Worker._preempt`. In place of today's 50 ms hold, the worker moves victims to the host or drops them, moves or drops the request itself, or makes it wait. A victim moves to the host when the host pool can hold all of its pages, since dropping and rebuilding requests the host pool could hold made 32 Bagel requests sharing a prefix take about 4x as long with 256 host pages.

Only KV pages trigger it. Qwen3.5’s recurrent state pool, the sampler's in-graph rows and the ring KV manager give each request one fixed-size slot for its whole life, so a running request never runs short there. A new request waits at admission until a slot is free beyond those already reserved, so it never holds KV pages it can't use.


### 3.1 Arrival numbers

```python
@dataclass
class CurrentForwardPassInfo:
    ...
    arrival: int | None = None   # set by the conductor when it takes the request
```

The conductor numbers each request from a single counter, in the order it takes them, and sends the number with the request. A dropped request keeps its number. The runner keeps the requests in that order, and each pool's queue and its victim choice go by it.

A number set by each worker won't work, since requests reach workers on separate sockets, and so two workers can see the same two requests in different orders. Each will drop a different one, and neither makes progress.

### 3.2 Choosing the victims

Say request R's grant is short by `pages_short` pages, one for a decode row. A request frees the pages only it holds, alone or with the index, plus the unused part of its reservation.

- **Candidates.** The preemptible requests that arrived after R, newest first, other than victims still to move, requests waiting on room set aside for them, and requests with a step or a KV read in flight on their pages. Each holds no page of its own, fits whole in the host pool, or has a drop that isn't refused for a lasting reason (§4). One that holds no page only gives back its reservation and asks again, whatever the host pool. The host's room is its free pages and the cached pages it can evict, less the room set aside for requests that can only move to the host, and each victim chosen for the host takes from it in turn. A request with room set aside moves into it.
- **Victims.** The fewest leading candidates that together free at least `pages_short`, minus any whose move wouldn't change what the set frees.

Then:

1. If the victims free enough, the worker earmarks `pages_short` for R (§3.3), and moves each victim to the host or drops it (§4).
2. Otherwise, if no newer victim is still to move, R can move to the host or be dropped, and that frees any page, R gives up its pages and is requeued.
3. Otherwise, R keeps its reservation and asks the pool for its shortage. It waits in the queue at its arrival, parked as a new prompt is, until the room covers the shortage.

While R's victims, or R itself, haven't moved yet, R waits for those moves and chooses no other victims.

The choice is one `KVManager.preempt` call under the pool's lock.

Why this shape:

- **Newest first.** No request loses its pages so that a newer request gets a page. vLLM's FCFS policy preempts the last entry in `running`, which is where a resumed request sits.
- **A victim frees its pages for one request.** If a second request could choose it too, the two earmarks would count the same pages, and the first would never be met. So a second request passes over it, and waits for the room its move makes rather than moving itself.
- **A request waiting on room set aside for it isn't a victim.** Its victims' moves would grow a reservation it had given up.
- **R is its own victim only as a last resort.** If even that frees nothing, every page is held by older or guaranteed requests, and those finish.
- **A waiting request waits in the queue.** It is measured again only when the room grows, and let go only once the room covers its whole shortage. Its arrival keeps newer prompts from taking that room first.
- **Not the row last in line.** Today's offload picks it, but under `TURNS` it can be the oldest request in the pool, so nothing guarantees progress.
- **Not the largest request.** A long request would be preempted every time it grows.

### 3.3 Earmarks

```python
@dataclass
class Earmark:
    rid: str            # R, the request whose grant failed
    pages: int          # what R was short
    victims: set[str]   # the victims whose moves cover it
```

As each victim's move lands, what it frees joins R's reservation, up to `pages_short`, so R's next grant can't fail again. Without it, the head of the queue, or a victim that asks again at once, could be admitted into the freed pages before R's next step, and R would fail again, as it does today.

Every victim can move in the call that chose it, since no step or read is in flight on its pages. A victim whose move fails comes off the earmark with nothing freed, so R chooses again at its next failed grant.

Lifecycle:

1. Created with its victims when the worker handles R's failed grant.
2. Grows R's reservation as each victim's move lands, and closes once the last one has.
3. Removed with R.

---

## 4. The drop

```python
class KVManager:
    def drop(self, rid: str) -> int:
        """Take rid's KV off the GPU without copying it, and return the number
        of pages that reach the free list. Keyed pages stay in the index as
        evictable pages. Needs no host pool."""

    def residency(self, rid: str) -> Residency:
        """ON_HOST while rid is offloaded, DROPPED from its drop until its
        rebuild lands, and RESIDENT otherwise."""
```

A drop releases the request's whole reservation. `KVManager.check_drop` refuses:

- a guaranteed request
- one with no kept ids
- one not yet decoding
- one with a host copy
- one with a step, a KV read or a sample in flight, until that lands

At a drop, the other resources:

| Resource | At a drop |
|---|---|
| Sampler | kept, since its RNG offset and repetition mask can't be recomputed |
| Position counter | reset, and the rebuild moves it back to the end of the history |
| Qwen3.5's recurrent state | zeroed in place, and the request keeps its slot, so its rebuild never waits for one |
| Attention | kept, since it holds no state a rebuild writes again |
| LoRA adapter pool | kept, so the rebuild runs under the request's adapter, which keeps its slot |
| Anything else | refused at load: a node with a rebuild policy needs every resource it holds to support a drop (`Resource.supports_drop`) |

Why this shape:

- **A drop needs no host pool.** `cpu_offload_pages` is 0 by default, and a full host pool would stall preemption as it stalls offload today.
- **The whole reservation goes.** Keeping its unused part, as the offload of the row last in line does, would leave pages counted that nothing can use until the victim returns.

> Note: a dropped request's keyed pages stay in the index as evictable cache. With write-ahead (H1b) and a host pool, they reach the host once no request holds them, so a later request with that prefix copies them back. A rebuild row doesn't read the host index, though, so it recomputes what only the host holds.

---

## 5. Return and rebuild

### 5.1 States

| State | Held as |
|---|---|
| running | resident, out of the queue |
| on the host | its pages on the host, in the queue at its arrival position, no reservation |
| dropped | not resident, in the queue at its arrival position, no reservation |
| rebuilding | not resident, out of the queue, holding its return reservation, its rebuild under way in the worker's log |

No field holds the state: residency, the host pool, the queue and the log's rebuilds tell the four apart.

Lifecycle:

- A request that was dropped or moved to the host is never chosen again, since it comes back guaranteed (§5.2).
- Removing a request in any state releases its earmark, its pages on the host, its place in the queue, its return reservation, its rebuild and its kept ids. A removal during a step that holds a rebuild row waits for the step.

### 5.2 The return check

A dropped request asks again when it is dropped, and the runner admits it from its arrival position as any request, with this check in place of the fit check:

```
need(V) = V's footprint

return when _unused + need(V) <= free + evictable,
       with every hit charged as fresh
```

When it passes, V reserves `need(V)` as a guaranteed request and readiness starts its rebuild.

A request on the host asks again when it moves there, and is admitted the same way. Its return reservation covers the pages its reload takes and the rest of its footprint. The readiness check for its next step then reloads it within that reservation. Once admitted it holds no newer request back, so a next step that comes only after its REMOVE costs nothing but the reservation.

Why this shape:

- **Its whole footprint, not its pages and one more.** With room for one page, its next page finds none, and as the newest request holding pages it moves itself again, so each round trip copies all its pages both ways for a page of decode. The footprint is the one amount after which it never needs room it didn't reserve, and the ingest 400 bounds it.
- **Guaranteed once back.** As a preemptible request its unused reservation would count toward the next older request's shortage, and it would be that request's first victim. So each request is preempted at most once.
- **Outside `check_ready`.** After a partial cache hit, `admit_retrieve` would move the position counter past what the rebuild restores, so no `admit_retrieve` runs on a request between its drop and the end of its rebuild.

### 5.3 The rebuild row

```python
@dataclass(frozen=True)
class RebuildPolicy:
    """How a node rebuilds a dropped request's KV, as its model declares it."""
    walks: frozenset[str]   # the prefill walks its rebuild rows run, those the request prefilled in
    ids_input: str          # the input a row's ids go under, which the ids sampled since follow
    sample_output: str      # the output that carries each sampled id
    loop: str               # the loop each of whose iterations samples one id
    copy_width: Mapping[str, int]           # a walk that reads more than the ids -> values its copies hold per KV slot
```

A model declares a policy per node (`Model.rebuild_policies`), and the worker checks it against the graphs at startup and reads off them what the node reads in each walk besides the ids. A node without one never drops a request.

A rebuild is a reload whose transfer is computed. The readiness check reads the request's residency: a request on the host is reloaded by copying, and a dropped one is answered with a rebuild. The scheduler then lays the KV out again over the kept ids, and opens the rebuild in the worker's log. Readiness answers the request's decode row from the log until the rebuild lands, without asking the engine, even if the request is let back in for room meanwhile.

The rebuild runs one row per prefill step the request's layout lays out (`StreamLayout.steps`), in order, the last also writing the ids sampled since minus the last, as rows mid-prefill that nothing popped. Each row runs in its step's walk, cut by that walk's budget, and a walk the model doesn't chunk rebuilds whole. The scheduler offers them where it offers a speculation's chunk in flight, so the walk's token budget cuts and counts them as it does a prompt, and they share steps with the walk's decode rows. Their inputs come from the log, which also advances their progress as each chunk lands. The step context names them (`StepContext.rebuild_rids`), so the sampler throws their samples away and leaves the request's last token and RNG offset alone. Postprocess takes them off the step before the graph runtime reads it. Decoding resumes from the last kept id once the last chunk lands.

A request dropped between two chunks of a prompt step reruns the steps before it from the log, then that step's chunks up to where it stood. Its chunk Loop still stands there, and the engine still holds the input it prepared for the step, which waits aside while the earlier rows prepare theirs, so the last row is cut from that input, over the layout the request already has. The prompt's next chunk waits until the rebuild lands and then starts where the Loop stands. Another request may have cached more of the prompt meanwhile, so the KV answers a dropped request's probe with at most a page under what it held.

The worker records each prefill step of a request on the step's first chunk, which carries its whole input: the ids, copied as the step read them to a pinned host pool (`rebuild_host_mb`) once that step completes, every other input the node reads there, such as the encoder's output a Qwen3.5 prompt with images reads (`rebuild_media_host_mb` more, after the step's outputs and within the bytes #876 lets a step copy to the host), and the step's metadata, which the step's rebuild row runs under. A step other than the one the layout says comes next leaves the request with nothing kept, so it is never dropped. Once the last step is recorded, the worker appends each sample from the stop check's host copy. They go back to the device when the row's step is built, on the pool's stream, and only that step waits for them. A request takes one slice of the pool for all of it when it arrives, sized from its layout and the values per slot the policy declares in the node's dtype, so a request the pool can't hold is known before it is admitted and takes the class of one no drop can free. The worker keeps all of this per node and per handle, and lists it in `rid_table.py`.

Why this shape:

- **The graph runtime never sees the rows.** Switching the request's runtime walk for one step would touch the runtime, the output edges, `WORKER_GRAPHS_DONE` and the loop-back. Every call that hands a step's rows to the runtime takes `ScheduledBatch.graph_rows`, which leaves them out.
- **The prefill walk.** Each model's decode walk breaks a multi-token row in its own way. Bagel's declares a span of 1, the Qwen3-Omni Thinker's embeds every id at one position, and Qwen3.5's with device loop-back reads one token off the sampler.
- **The prompt's ids from the walk's input.** A drop can come at any token, after the prompt tensor is freed, so the worker copies the ids after the step that read them, on a stream of the pool's own. Sending the ids from the data worker with every request instead added 0.5 to 1.8 ms of TTFT at Bagel c16 and Qwen3.5 c32.
- **Not a remove and re-ingest.** The sampler's RNG offset and repetition mask would be lost.
- **A prompt mid-prefill reruns its own chunks.** Restarting it at chunk 0 would move its chunk Loop back, which only the graph runtime can do.

> Note: a rebuild's first chunk starts past its cached prefix, measured as a fresh prompt's is on each try, so it recomputes only what the GPU index doesn't hold, and Qwen3.5 restores a recurrent snapshot for the prefix.

### 5.4 Tokens after a rebuild

A rebuild's prefill rounds differently from the decode steps that first wrote the KV. So a rebuilt request continues like a fresh prefill of its context, not like the run it was dropped from, and a seeded request rebuilt under load may not repeat its output.

In 24 cases on Qwen3.5, Bagel, Orpheus and the Qwen3-Omni Thinker, greedy and seeded, captured and eager, the tokens after every rebuild equal a prefill of the kept ids with the same step shape and cached length. 7 of the 24 also match the run without the drop for 64 tokens.

### 5.5 Why every request finishes

A guaranteed request reserves its whole footprint and is never preempted, so it finishes.

Take the oldest preemptible request:

1. When it runs short, newer requests move to the host or are dropped first, and it gives up its own pages only when none of them can free a page for it. It always can: it is dropped, or moves into the room the host set aside for it (§1).
2. It then waits at the head of its queue, where nothing newer is admitted past it.
3. It returns guaranteed once its whole footprint fits, so it is never preempted again.
4. Its footprint fits the pool, so it returns once the older requests finish, and then finishes.

The next oldest request then takes its place in the argument.

For example, an older and a newer request hold 5 pages each of a 10-page pool, and the older needs a sixth. The newer one is dropped and needs its whole footprint to return, so it waits until the older one finishes. Readmitted once its prompt fit, it would take back its own pages and the older one would fail again. Today both requests end with HTTP 500 at the 600 s timeout.

---

## 6. TP followers

Rank 0 decides every reservation and its undo, each earmark, drop and return, and records it in the `OffloadDelta` ([ipc_format.py](../../mstar/utils/ipc_format.py)) it sends with the next `ScheduleTPNode`, next to the offloads and reloads the delta already carries. A follower applies the delta before it builds that step, so every rank admits each step from the same counts.

```python
@dataclass
class OffloadDelta:
    rids: list[str]
    kinds: list[MoveKind]   # offload, reload, reserve, unreserve, earmark, drop or rebuild
    ...                     # the page counts and victims the kinds carry
```

- Every rank checks each grant against the reservations. Only rank 0 queues, runs the fit check, chooses victims and makes a request wait.
- Rank 0 drops a victim as it sends the next step of the victim's node.
- `ScheduleTPNode` carries the step's rebuild rows. Each rank builds them from its own kept ids, which match, since every rank samples rank 0's token.
- A follower whose counts can't take a move exits.

With speculation on, rank 0 decides only where pages already move today, after the admit fence or after a refused step, so a speculative step never carries a move made after it was sent.

Why this shape:

- **Rank 0 decides.** Victims chosen from per-rank state could differ between ranks, which AGENTS.md invariant 11 rules out.
- **The drop goes out with the next step.** Applied on rank 0 the moment its step lands, a drop would free pages before removals a follower applies ahead of the delta, and the ranks' free lists would hand out different pages.
- **No kept ids on the wire.** Every rank already holds them.

---

## 7. Checks

Always on:

- A victim is preemptible, can move to the host or be dropped, and is never older than the request whose grant failed. A breach raises, and so does a forced drop of a guaranteed request.
- A grant that a reservation covers and that can't take its pages raises.
- Counters per node: preemptions, drops, tokens rebuilt, and requests dropped more than once.

Under `MSTAR_KV_DEBUG_ASSERTS`:

- The invariant of §2, on each grant.
- After a drop or a move to the host, the victim holds no page in the pool, and the pool's pages are conserved.

---

## 8. Trade-offs

- **Drops cost a rebuild.** A victim the host pool can't hold is dropped and rebuilt, once at most, since it comes back guaranteed.
- **A request comes back late.** A request moved off the device waits at the head of its queue for its whole footprint, and the prompts behind it wait with it.
- **A lone rebuild isn't speculated.** Speculation targets come from the graph runtime, which holds no rebuild row, so a step of rebuild rows alone pays one scheduling gap per chunk. A rebuild chunk rides speculation when another row of its step gives a target.
- **Kept ids.** A node with a rebuild policy copies each request's prompt ids to a pinned host pool, 64 MB by default, and keeps them until the request finishes: 128 KB for a 32K-token request. A request the pool can't hold is never dropped on that node, though with a host pool it can still move to the host.
- **Shared prefixes with host offload.** A hit on a page only the index holds is charged as fresh, so prompts sharing a prefix run fewer at once. With host offload on, 32 Bagel requests over four shared prefixes on 96 pages took 1.49x to 1.66x as long as they do today.
- **Guaranteed requests queue.** On Orpheus at c16, streamed TTFA p99 rose from 247 ms to 2.58 s against preemption without classes. But, I am fine with the p99 cost since for voice, a later first chunk is better than a gap in the middle of playback.

---

## 9. Next steps (Out of Scope here)

- **Pool sizing.** Fixed, hand-set pool sizes can also cause thrashing, which this work surfaced. Each cache's `max_num_pages` comes from the config, 2048 pages by default, so a pool can fill and preempt long before memory runs out: Command A+'s TP8 config gets about 4 GB of KV per GPU where, by my estimate, about 20 GB is free after weights. Except for M*, every other engine sizes the pool from free memory instead.
- **One pool per GPU.** SGLang has moved from a fixed pool per cache type to one physical pool, where full-attention KV and window or recurrent state grow from opposite ends and compaction remaps pages, since agent contexts vary too much in length for fixed ratios. M*'s caches on a worker could share one pool the same way, but the effects would need to be measured.
