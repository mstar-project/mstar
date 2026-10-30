# Preemption under KV pressure

Since M* has no admission control today, requests take pages as they prefill and decode, and free them only when they finish. Once every running request in a full pool needs another page, none can finish, and all wait until clients time out (#307).

Earlier PR (#400) proposed admission control that reserves each request's worst case --- its prompt plus `max_tokens`. It removes the stall completely, but induces severe HoL blocking (e.g. chats queue behind image requests): their TTFT p50 rose from 0.45 s to 17.7 s. Whether this fragments the pool was never measured, but it is nonetheless based on a banker check which felt too strict.

This design adds slim version of admission control by default (A1). A request reserves only its prompt's pages and waits in a queue until they fit. On top of it, the requests are classified as either:

1. **Guaranteed.** A request that streams audio, or whose KV can't be rebuilt from token ids, also reserves pages for the most it can generate, and is never preempted.
2. **Preemptible.** When a request runs out of pages, newer preemptible requests free pages for it. Each moves to the host when the host pool can hold all of its pages, and is otherwise dropped and later rebuilds its KV from token ids.

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
    guaranteed: bool = False   # reserves its whole footprint and is never dropped
```

A dropped request keeps only its token ids, so a request can be preempted only if its node can rebuild all of its KV from them.

A request that can't be rebuilt must never run short. It reserves its whole footprint up front, its prompt plus its growth bound, and is never preempted. That is what guaranteed means.

The model declares the footprint as KV spans (#832). A span's kind says what its KV is made from:

- `ids`: token ids
- `digest`: an encoder's output
- `rows`: another node's outputs

The conductor sets the class once, at ingest, and the same class holds in every pool.

| # | A request with | Class | Why |
|---|---|---|---|
| 1 | Audio it streams out | guaranteed | a pause is a gap in playback |
| 2 | A `rows` span | guaranteed | its KV comes from another node's outputs, such as the Thinker's hidden states the Qwen3-Omni Talker reads |
| 3 | A `digest` span | guaranteed | its KV comes from an encoder's output, freed after prefill |
| 4 | No rebuild policy on its node | guaranteed there | the model declares no way to rebuild the node's KV, as when its decode reads embeddings |
| 5 | Only `ids` spans | preemptible | its token ids reproduce all of its KV |

Row 3 becomes preemptible with a followup, R5, which keeps a host copy of the encoder's output for the rebuild. R5a covers Qwen3.5 images, and R5b covers Bagel, Qwen3-Omni and Higgs. Until then, a rebuild of row 3 would have to run the encoder again, which on Bagel and Qwen3-Omni sits on another worker.

On initial tests, I measured that keeping row 3 guaranteed costs about 9% of TTFT under pressure. On a 48-page pool at c16, Qwen3.5 image requests' TTFT p50 fell from 3.48 s to 3.18 s once R5a made them preemptible.

Why this shape:

- **The class comes from the request.** A priority or a per-request flag would be one more input to tune, when the footprint and the output can be used to make the decision.
- **The conductor sets it.** It is the one place that sees every resource's layout. A Qwen3-Omni request with audio out has `rows` only on the Talker's worker, so a class each worker set from its own pools would let the Thinker's worker drop a request whose Talker is speaking.
- **Only streamed audio.** A streamed text reply that pauses is late, not broken, and an unstreamed audio reply is played after it arrives.
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
- **Across pools.** A request takes its reservation in every pool its node asks, the recurrent pool's slot included, or in none, and nothing is retrieved for it before then. A waiting request is asked again only after a pool it waits on has grown.

```
room = free + evictable - _unused

invariant, after every grant, reservation, lease and release:
    _unused <= free + evictable
```

What a cache hit costs:

| A hit on | Costs |
|---|---|
| a page only the index holds | a fresh page |
| a page another request holds | nothing, lent to a preemptible request |
| a cached prefix copied from the host at readiness | a grant within the request's reservation |

Why this shape:

- **The rule lives in `_alloc`.** Today it grants any free page, so a growing request could take a page a guaranteed request reserved, and that request's next grant would fail.
- **Only the head is admitted.** Nothing newer passes the oldest request, so it can't wait forever behind a stream of small ones (§5.5).
- **Evictable hits are charged as fresh.** Eviction could hand such a page to another request before the probe leases it.

---

## 3. Preemption

Preemption starts when a grant past a reservation fails, in `Worker._handle_allocation_failure` ([worker.py](../../mstar/worker/worker.py)), which calls `Worker._preempt`. In place of today's 50 ms hold, the worker moves victims to the host or drops them, moves or drops the request itself, or makes it wait. A victim moves to the host when the host pool can hold all of its pages, since dropping and rebuilding requests the host pool could hold made 32 Bagel requests sharing a prefix take about 4x as long with 256 host pages.

Only KV pages trigger it. Qwen3.5’s recurrent state pool and the ring KV manager give each request one fixed-size slot for its whole life, so a running request never runs short there. A new request waits at admission until a slot is free beyond those already reserved, so it never holds KV pages it can't use.


### 3.1 Arrival numbers

```python
@dataclass
class KVReqConfig(ResourceReqConfig):
    ...
    arrival: int | None = None   # set by the conductor when it takes the request
```

The conductor numbers each request from a single counter, in the order it takes them, and sets it on every pool's config. A dropped request keeps its number. Each pool's queue and its victim choice go by it.

A number set by each worker won't work, since requests reach workers on separate sockets, and so two workers can see the same two requests in different orders. Each will drop a different one, and neither makes progress.

### 3.2 Choosing the victims

Say request R's grant is short by `pages_short` pages, one for a decode row. A request frees the pages only it holds, alone or with the index, plus the unused part of its reservation.

- **Candidates.** The preemptible requests that arrived after R and aren't already victims, newest first. Each fits whole in the host pool, or has a drop that isn't refused for a lasting reason (§4). The host's room is its free pages and the cached pages it can evict, and each victim chosen for the host takes from it in turn.
- **Victims.** The fewest leading candidates that together free at least `pages_short`, minus any whose move wouldn't change what the set frees.

Then:

1. If the victims free enough, the worker earmarks `pages_short` for R (§3.3), and moves each victim to the host or drops it (§4).
2. Otherwise, if no newer victim is still to move, R can move to the host or be dropped, and that frees any page, R gives up its pages and is requeued.
3. Otherwise, R waits until the pool's room grows past what its failed grant saw.

While R's victims, or R itself, haven't moved yet, R waits for those moves and chooses no other victims.

The choice is one `KVManager.preempt` call under the pool's lock.

Why this shape:

- **Newest first.** No request loses its pages so that a newer request gets a page. vLLM's FCFS policy preempts the last entry in `running`, which is where a resumed request sits.
- **A victim frees its pages for one request.** If a second request could choose it too, the two earmarks would count the same pages, and the first would never be met. So a second request passes over it, and waits for the room its move makes rather than moving itself.
- **R is its own victim only as a last resort.** If even that frees nothing, every page is held by older or guaranteed requests, and those finish.
- **A waiting request watches the room.** The pool records the room each failed grant saw, so no timer decides when to retry.
- **Not the least recently active request.** Today's offload picks it, but it can be the oldest request in the pool, so nothing guarantees progress.
- **Not the largest request.** A long request would be preempted every time it grows.

### 3.3 Earmarks

```python
@dataclass
class Earmark:
    rid: str            # R, the request whose grant failed
    pages: int          # what R was short
    victims: set[str]   # the victims whose moves cover it
```

When the victims' moves land, the earmark joins R's reservation, so R's next grant can't fail again. Without it, the head of the queue could be admitted into the freed pages before R's next step, and R would fail again, as it does today.

A victim with a step or a KV read in flight moves or is dropped once that lands, and the worker composes it into no further step until then.

Lifecycle:

1. Created with its victims when the worker handles R's failed grant.
2. Joins R's reservation when its victims' moves land.
3. Removed with R.

---

## 4. The drop

```python
class KVManager:
    def drop(self, rid: str) -> int:
        """Take rid's KV off the GPU without copying it, and return the number
        of pages that reach the free list. Keyed pages stay in the index as
        evictable pages. Needs no host pool."""

    def is_resident(self, rid: str) -> bool:
        """False while rid is offloaded, and from its drop until its rebuild
        step lands."""
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
| Everything else | kept, as an offload keeps it |

Why this shape:

- **A drop needs no host pool.** `cpu_offload_pages` is 0 by default, and a full host pool would stall preemption as it stalls offload today.
- **The whole reservation goes.** Keeping its unused part, as the offload of the least recently active request does, would leave pages counted that nothing can use until the victim returns.

> Note: a dropped request's keyed pages stay in the index as evictable cache. With write-ahead (H1b) and a host pool, they reach the host once no request holds them, so a later request with that prefix copies them back. A rebuild row doesn't read the host index, though, so it recomputes what only the host holds.

---

## 5. Return and rebuild

### 5.1 States

| State | Held as |
|---|---|
| running | resident, out of the queue |
| on the host | its pages on the host, in the queue at its arrival position, no reservation |
| dropped | not resident, in the queue at its arrival position, no reservation |
| rebuilding | not resident, out of the queue, holding its return reservation, its rebuild row queued or in a step |

No field holds the state: residency, the host pool, the queue and the rebuild queue tell the four apart.

Lifecycle:

- A rebuilding request can be chosen again. Its queued row leaves the queue, and it gives up its return reservation and waits in the queue again.
- Removing a request in any state releases its earmark, its pages on the host, its place in the queue, its return reservation, its queued rebuild row and its kept ids. A removal during a rebuild step waits for the step.

### 5.2 The return check

A dropped request asks again when it is dropped, and the runner admits it from its arrival position as any request, with this check in place of the fit check:

```
need(V) = for each label, the pages V's kept history fills, plus 1,
          capped at V's footprint

return when _unused + need(V) <= free + evictable,
       with every hit charged as fresh
```

When it passes, V reserves `need(V)`, and the worker queues its rebuild row. The drop counts the history from the tokens V's KV held and the sample its next step writes.

A request on the host asks again when it moves there, and is admitted the same way. Its return reservation covers the pages its reload takes, plus one page per stream, capped at its footprint, or its queued prompt if that is larger. The readiness check for its next step then reloads it within that reservation. Once admitted it holds no newer request back, so a next step that comes only after its REMOVE costs nothing but the reservation.

Why this shape:

- **A whole history and a page, not the prompt.** A victim readmitted once its prompt fit would take back its own pages, and R would fail again.
- **Capped at the footprint.** The ingest 400 bounds the footprint, not the history plus a page.
- **Outside `check_ready`.** After a partial cache hit, `admit_retrieve` would move the position counter past what the rebuild restores, so no `admit_retrieve` runs on a request between its drop and the end of its rebuild.

### 5.3 The rebuild row

```python
@dataclass(frozen=True)
class RebuildPolicy:
    """How a node rebuilds a dropped request's KV, as its model declares it."""
    walk: str            # the prefill walk a rebuild row runs
    ids_input: str       # the walk's input that carries the prompt's ids
    sample_output: str   # the output that carries each sampled id
    loop: str            # the loop each of whose iterations samples one id
    copied_inputs: Mapping[str, tuple[str, ...]]   # another prefill walk -> what it reads besides the ids
```

A model declares a policy per node (`Model.rebuild_policies`), and the worker checks it against the graphs at startup. A node without one never drops a request.

The rebuild is one row of the engine's batch over the kept ids minus the last, under the walk the request prefilled in, in a step of its own. Its sample is discarded, it sends nothing, and its decode loop doesn't advance. Decoding resumes from the last kept id once the step lands.

The worker keeps a request's prompt ids from the policy's input the first time a step runs one of the policy's prefill walks, as one device copy per step shared by its rows, and appends each sample from the stop check's host copy. A walk of `copied_inputs` reads more than the ids, such as the encoder's output a Qwen3.5 prompt with images reads. Once the request's prefill step has run, the worker copies those inputs and the step's metadata to the host, and the rebuild row reads them back under that metadata. The worker keeps all of this per node and per handle, only for a request a rebuild could write again, and lists it in `rid_table.py`. A rebuild copies the prompt to the host once, for the KV's layout and keys.

Why this shape:

- **The graph runtime never sees the row.** Switching the request's runtime walk for one step would touch both runtimes, the output edges, `WORKER_GRAPHS_DONE` and the loop-back.
- **The prefill walk.** Each model's decode walk breaks a multi-token row in its own way. Bagel's declares a span of 1, the Qwen3-Omni Thinker's embeds every id at one position, and Qwen3.5's with device loop-back reads one token off the sampler.
- **The prompt's ids from the walk's input.** A drop can come at any token, after the prompt tensor is freed, so the worker copies the ids when the step runs. The copy stays on the device, and only a rebuild waits on a copy to the host. Sending the ids from the data worker with every request instead added 0.5 to 1.8 ms of TTFT at Bagel c16 and Qwen3.5 c32.
- **Not a remove and re-ingest.** The sampler's RNG offset and repetition mask would be lost.

> Note: the row's cache probe serves the kept ids' cached prefix first, so a rebuild recomputes only what the GPU index doesn't hold. Qwen3.5 rebuilds its whole history until its prefix caching (P2) lands.

### 5.4 Tokens after a rebuild

A rebuild's prefill rounds differently from the decode steps that first wrote the KV. So a rebuilt request continues like a fresh prefill of its context, not like the run it was dropped from, and a seeded request rebuilt under load may not repeat its output.

In 24 cases on Qwen3.5, Bagel, Orpheus and the Qwen3-Omni Thinker, greedy and seeded, captured and eager, the tokens after every rebuild equal a prefill of the kept ids with the same step shape and cached length. 7 of the 24 also match the run without the drop for 64 tokens.

### 5.5 Why every request finishes

A guaranteed request reserves its whole footprint and is never preempted, so it finishes.

Take the oldest preemptible request:

1. When it runs short, newer requests move to the host or are dropped first, and it gives up its own pages only when none of them can free a page for it.
2. It then waits at the head of its queue, where nothing newer is admitted past it.
3. It returns with its whole history and one more page reserved, so each time it is preempted again it has grown by at least a page.
4. Its footprint fits the pool, so it finishes after a bounded number of returns.

The next oldest request then takes its place in the argument.

For example, an older and a newer request hold 5 pages each of a 10-page pool, and the older needs a sixth. The newer one is dropped and needs 6 pages to return, so it waits until the older one finishes. Readmitted once its prompt fit, it would take back its own pages and the older one would fail again. Today both requests end with HTTP 500 at the 600 s timeout.

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

- **Drops cost a rebuild.** A victim the host pool can't hold is dropped and rebuilt. Under sustained overload the newest requests are dropped again and again: on #307's workload one request was dropped four times, although that was a contrived test.
- **Rebuild steps.** A rebuild runs in a step of its own, so the node's decode rows wait one step. A long uncached history rebuilds in one step, eager past the capture buckets, until chunked prefill (#365) splits it. One step of delay on decode rows is small next to a request that never finishes.
- **Kept ids.** A node with a rebuild policy keeps each request's prompt ids on the device, in the walk's input dtype, until the request finishes: 256 KB for a 32K-token request of int64 ids. A rebuild waits on one copy of them to the host.
- **Shared prefixes with host offload.** A hit on a page only the index holds is charged as fresh, so prompts sharing a prefix run fewer at once. With host offload on, 32 Bagel requests over four shared prefixes on 96 pages took 1.49x to 1.66x as long as they do today.
- **Guaranteed requests queue.** On Orpheus at c16, streamed TTFA p99 rose from 247 ms to 2.58 s against preemption without classes. But, I am fine with the p99 cost since for voice, a later first chunk is better than a gap in the middle of playback.

---

## 9. Next steps (Out of Scope here)

- **Pool sizing.** Fixed, hand-set pool sizes can also cause thrashing, which this work surfaced. Each cache's `max_num_pages` comes from the config, 2048 pages by default, so a pool can fill and preempt long before memory runs out: Command A+'s TP8 config gets about 4 GB of KV per GPU where, by my estimate, about 20 GB is free after weights. Except for M*, every other engine sizes the pool from free memory instead.
- **One pool per GPU.** SGLang has moved from a fixed pool per cache type to one physical pool, where full-attention KV and window or recurrent state grow from opposite ends and compaction remaps pages, since agent contexts vary too much in length for fixed ratios. M*'s caches on a worker could share one pool the same way, but the effects would need to be measured.
