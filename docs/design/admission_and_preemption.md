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

A request that nothing can take pages back from must never run short. From the moment it is guaranteed, it reserves its whole footprint, its prompt plus its growth bound, and is never preempted. That is what guaranteed means.

The model declares the footprint as KV spans (#832). A span's kind says what its KV is made from:

- `ids`: token ids
- `digest`: an encoder's output
- `rows`: another node's outputs

Each pool decides the class from two facts. The conductor marks a request that streams an output that can't pause, and that mark holds in every pool. The worker marks whether a drop there can free the request's pages: every node of the pool that writes the request's KV on that worker has a policy that can write all of it again. A request no drop can free is preemptible while the host pool sets aside room for its move: its prompt when it is admitted, and each page it takes after that before it takes it. When the host pool has no room for that, at admission or at any page after, it becomes guaranteed, and the host pool logs and counts it.

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
- **Room on the host is set aside, not assumed.** A request that can only move to the host must be able to move when an older request needs its pages (§5.5). Room counted at that moment is whatever other moves left, so a full host pool would leave it holding its pages, and a pool of such requests stalls as in #307. The room set aside always covers the pages it holds, so its move fits. Past `cpu_offload_pages`, the rest are guaranteed.
- **The room grows with what it holds.** Setting its whole footprint aside at its first ask held host room for pages it might never write, and for requests still waiting in the queue, so few requests got room and the host prefix tier was left little. Turning guaranteed at the page the host can't cover needs the rest of its footprint free on the device, and until it is, the request is short of those pages as a decode step is short of one (§3).
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
- **Queue.** A request joins the runner's waiting list at its first readiness check, by its arrival number (§3.1), or behind every numbered request in the order it first asks when it has none. It is admitted when it is first among the waiting requests of every pool it asks, and its reservation fits in each, in the free and evictable pages no admitted request has reserved.
- **Asking again.** A request that moves to the host, is dropped, gives back a reservation it never wrote to, or waits for its shortage (§3.2) asks again at once and waits in no queue. Each pass tries these requests first, oldest first, once what each didn't fit has grown. While one of them is the oldest live request to have asked a pool it waits for, that pool's queue admits no newer prompt.
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
- **Only the head of a queue is admitted.** No newer prompt passes the oldest one, so it can't wait forever behind a stream of small ones.
- **A request that asks again waits outside the queues, and holds newer prompts back only while it is the oldest in its pool.** Waiting at its arrival for its whole history and a page, it held every newer prompt back: 32 Bagel requests over four shared prefixes on 96 pages, with 512 host pages, ran about 3 at once and took about 58 s, against 41 to 54 s today. Today an offloaded request reloads outside admission once its pages fit, and trying returns first in the same way took that run to 40.8 to 47.9 s on R4. Waiting outside the queues with no such rule, a request on the host or dropped could be passed by a steady stream of newer prompts until it timed out, even once it was the oldest. Every older request in its pool finishes (§5.5), so it becomes the oldest there, and no newer prompt passes it then.
- **Evictable hits are charged as fresh.** Eviction could hand such a page to another request before the probe leases it.
- **A guaranteed request is charged for no hit until it adopts its lease.** A cut lease leaves the walk to write the cut pages, and a cut page another request still holds frees no room, so its reservation keeps every page until then.

---

## 3. Preemption

Preemption starts when a grant past a reservation fails, in `Worker._handle_admit_failure` ([worker.py](../../mstar/worker/worker.py)), which calls `Preemption.preempt` ([preemption.py](../../mstar/worker/preemption.py)) before `Admission.relieve`. In place of today's 50 ms hold, the worker moves victims to the host or drops them, or makes the request wait. A victim moves to the host when the host pool can hold all of its pages, since dropping and rebuilding requests the host pool could hold made 32 Bagel requests sharing a prefix take about 4x as long with 256 host pages.

Only KV pages trigger it. Qwen3.5’s recurrent state pool, the sampler's in-graph rows and the ring KV manager give each request one fixed-size slot for its whole life, so a running request never runs short there. A new request waits at admission until a slot is free beyond those already reserved, so it never holds KV pages it can't use.


### 3.1 Arrival numbers

```python
@dataclass
class CurrentForwardPassInfo:
    ...
    arrival: int | None = None   # set by the conductor when it takes the request
```

The conductor numbers each request from a single counter, in the order it takes them, and sends the number with the request. A dropped request keeps its number. The runner keeps the requests in that order, and each pool's queue, the order it tries requests that ask again in, and its victim choice go by it.

A number set by each worker won't work, since requests reach workers on separate sockets, and so two workers can see the same two requests in different orders. Each will drop a different one, and neither makes progress.

### 3.2 Choosing the victims

Say request R's grant is short by `pages_short` pages, one for a decode row. A request frees the pages only it holds, alone or with the index, plus the unused part of its reservation.

- **Candidates.** The preemptible requests that arrived after R, newest first, other than requests with a step or a KV read in flight on their pages, a dropped request whose rebuild hasn't landed, since the rebuild runs on its reservation, and a request holding a lease, until its walk's admit takes the lease over. The probe that took the lease moved the position counter, which never moves back, and a walk the cache served keeps its KV only in the lease. Each holds no page of its own, fits whole in the host pool, or has a drop that isn't refused for a lasting reason (§4). Any other that holds no page only gives back its reservation and the room the host set aside for it, and asks again, whatever the host pool. The host's room is its free pages and the cached pages it can evict, less the room set aside for requests that can only move to the host, and each victim chosen for the host takes from it in turn. A request with room set aside moves into it.
- **Victims.** The fewest leading candidates that together free at least `pages_short`, minus any whose move wouldn't change what the set frees.

Then:

1. If the victims free enough, `preempt` earmarks what R's grant needed past its reservation (§3.3), and moves each victim to the host or drops it (§4).
2. Otherwise, R asks the pool for every page its grant needed past its reservation. When the host pool can hold all of R's pages, R moves there, as today's offload-and-hold moves a request that can't grow, and returns once its history and those pages fit. Otherwise R keeps its reservation and its pages, and waits, parked as a new prompt is, until the room covers them.

A waiting request chooses victims too. Once each step's admit has settled, the worker walks the waiting requests the pools have refused, oldest first, whether a new prompt, a request that asks again or one waiting for its shortage. It helps the first whose pool chooses victims for it among the newer requests that wait holding pages, newest first, as for a short grant (`KVManager.make_room`), so an older request stuck on another resource's slot, or in a pool that doesn't move, holds up no younger one. It takes the room their moves free as they land (§3.3). The walk stops at the first refused request the room now fits, so no younger request takes that room. The next admission pass tries it, and admits it unless another resource refuses it or a request that asks again holds its queue back. While no request waits holding pages, the worker skips the walk without taking a lock, and a walk that moved nothing runs again only once the refusals, the requests the pools could move or the room moving them would make change.

The choice is one `KVManager.preempt` call under the pool's lock.

Why this shape:

- **Newest first.** No request is chosen as a victim for a newer request. A request that moved itself to the host can see newer prompts start in the pages it freed while an older request runs, but none once it is the oldest in its pool. vLLM's FCFS policy preempts the last entry in `running`, which is where a resumed request sits.
- **A waiting request moves newer ones that wait holding pages.** A rule decided only when a request runs short can't see an older request that asks later, as one held in an encoder does. The newer request that waited first holds the pages it needs and runs no step that could run short, so both waited for good. R moving itself whenever nothing newer covered its shortage avoided that, but R came straight back into the room it freed and ran short again. R now moves itself only to the host, and returns only once its history and every page its grant needed fit.
- **Only requests that wait are moved for one.** A running request is never moved for a waiting one, so prompts still start in arrival order whenever they fit and requests that ask again are tried first. A new prompt is usually the newest request, so it rarely has anyone to move. A waiting request runs no step, so moving it once the admit has settled races nothing.
- **A waiting request waits for its whole shortage.** It is measured again only when the room grows, and let go only once the room covers its whole shortage.
- **Not the row last in line.** Today's offload picks it, but under `TURNS` it can be the oldest request in the pool, so nothing guarantees progress.
- **Not the largest request.** A long request would be preempted every time it grows.

### 3.3 Earmarks

```python
@dataclass
class Earmark:
    rid: str            # R, the request whose grant failed
    pages: int          # what R's grant needed past its reservation
    victims: set[str]   # the victims whose moves cover the rest
```

The room R's grant counted joins R's reservation at once. As each victim's move lands, what it frees joins too, up to what the grant needed past the reservation, so R's next grant can't fail again. Without it, the head of the queue, or a victim that asks again at once, could be admitted into the freed pages before R's next step, and R would fail again, as it does today.

Every victim can move in the call that chose it, since no step or read is in flight on its pages. A victim whose move fails comes off the earmark with nothing freed, so R chooses again at its next failed grant.

When a waiting request W chooses victims, the earmark is W's. W takes the room as its victims' moves land, once that covers all W asks for, so a victim, which asks again at once, can't take it back first. If the host then has no room to set aside for W's move (§1), W turns guaranteed and waits for its whole footprint instead.

Lifecycle:

1. Created with its victims when the worker handles R's failed grant.
2. Grows R's reservation as each victim's move lands, and closes once the last one has.
3. Removed with R.

---

## 4. The drop

```python
class KVManager:
    def drop(self, rid: str, tokens: int) -> int:
        """Take rid's KV off the GPU without copying it, and return the number
        of pages that reach the free list. Keyed pages stay in the index as
        evictable pages. Needs no host pool. tokens is the slots rid's rebuild
        writes: its kept ids but the last and the slots its host copies fill,
        or the chunks its prompt ran."""

    def residency(self, rid: str) -> Residency:
        """ON_HOST while rid is offloaded, DROPPED from its drop until its
        rebuild lands, and RESIDENT otherwise."""
```

A drop releases the request's whole reservation. A drop is refused for:

- one that keeps no ids, or whose ids haven't reached the host or don't cover its KV, unless it is between two chunks of its prompt, which its rebuild runs again (`Rebuilds.refusal`)
- one with a host copy, or one dropped already whose rebuild hasn't landed (`KVManager.check_drop`, as are the rest)
- one with a step, a KV read or a sample in flight, until that lands
- one whose KV can't be recomputed from token ids, or that writes more than one label, span or growth loop
- one whose rebuild would write other than the tokens its KV holds
- any request above one rank, for now

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
| running | resident, admitted |
| on the host | its pages on the host, waiting to return, no reservation |
| dropped | not resident, waiting to return, no reservation |
| rebuilding | not resident, admitted, holding its return reservation, its rebuild under way in the worker's log |

No field holds the state: residency, the host pool, admission and the log's rebuilds tell the four apart.

Lifecycle:

- A rebuilding request is never chosen again: its residency stays dropped until the rebuild lands, and the KV refuses a second drop until then.
- Removing a request in any state releases its earmark, its pages on the host, its wait to return, its return reservation, its rebuild and its kept ids. A removal during a step that holds a rebuild row waits for the step.

### 5.2 The return check

A dropped request asks again when it is dropped, and waits outside the queues (§2), with this check in place of the fit check:

```
need(V) = the larger of V's prompt and, for each label, the pages
          V's kept history fills, plus 1, capped at V's footprint,
          less the whole prompt pages another request holds, up to a
          page under what V's KV held

return when _unused + need(V) <= free + evictable
```

A page another request holds can't be evicted, so the rebuild's probe takes it for nothing, as a prompt's probe does (§2), and the probe takes at most a page under what V's KV held (§5.3). Any other hit is charged as fresh.

When it passes, V reserves `need(V)` and readiness starts its rebuild. The drop counts the history from the tokens V's KV held and the sample its next step writes.

A request on the host asks again when it moves there, and the runner tries it before the queues, oldest first (§2), with the same check. Its return reservation covers the pages its reload takes, plus one page per stream, capped at its footprint, or its queued prompt if that is larger. A request that waited for its shortage before it moved returns with every page its grant needed instead of the one page. The readiness check for its next step then reloads it within that reservation. Once admitted it holds no newer request back, so a next step that comes only after its REMOVE costs nothing but the reservation.

Why this shape:

- **A whole history and a page, not the prompt alone.** A victim readmitted once its prompt fit would take back its own pages, and R would fail again. A victim dropped partway through its prompt still writes the rest of it, so it returns with the whole prompt at least.
- **Capped at the footprint.** The ingest 400 bounds the footprint, not the history plus a page.
- **Not its whole footprint.** A footprint far larger than what the request held kept it at the head of the queue while pages were free, and every prompt behind it waited too. When it runs short again it waits for its shortage instead (§3.2).
- **Outside `check_ready`.** After a partial cache hit, `admit_retrieve` would move the position counter past what the rebuild restores, so no `admit_retrieve` runs on a request between its drop and the end of its rebuild.

### 5.3 The rebuild row

```python
@dataclass(frozen=True)
class RebuildPolicy:
    """How a node rebuilds a dropped request's KV, as its model declares it."""
    walks: frozenset[str]   # the prefill walks a rebuild row runs, the one the request prefilled in
    ids_input: str          # the input that carries the prompt's ids
    sample_output: str      # the output that carries each sampled id
    loop: str               # the loop each of whose iterations samples one id
    copy_width: Mapping[str, int]           # a walk that reads more than the ids -> values its copies hold per KV slot
```

A model declares a policy per node (`Model.rebuild_policies`), and the worker checks it against the graphs at startup and reads off them what the node reads in each walk besides the ids. A node without one never drops a request.

A rebuild is a reload whose transfer is computed. The readiness check reads the request's residency: a request on the host is reloaded by copying, and a dropped one is answered with a rebuild. The scheduler then has the KV lay the label out again to hold the slots its drop checked, and opens the rebuild over those slots in the worker's log. Readiness answers the request's decode row from the log until the rebuild lands, without asking the engine, even if the request is let back in for room meanwhile.

The rebuild prefills the kept ids minus the last, under the walk the request prefilled in, as rows mid-prefill that nothing popped. The scheduler offers them where it offers a speculation's chunk in flight, so the walk's token budget cuts and counts them as it does a prompt, and they share steps with the walk's decode rows. Their inputs come from the log, which also advances their progress as each chunk lands. The step context names them (`StepContext.rebuild_rids`), so the sampler throws their samples away and leaves the request's last token and RNG offset alone. Postprocess takes them off the step before the graph runtime reads it. Decoding resumes from the last kept id once the last chunk lands.

A request dropped between two chunks of its prompt is rebuilt from its prompt instead. Its chunk Loop still stands where the prompt stopped, and the engine still holds the input it prepared for the prompt, so the rebuild runs the prompt's chunks again up to that point, cut from that input, over the layout the request already has. The prompt's next chunk waits until the rebuild lands and then starts where the Loop stands. Another request may have cached more of the prompt meanwhile, so the KV answers a dropped request's probe with at most a page under what it held.

The worker keeps a request's prompt ids from the policy's input the first time a step runs one of the policy's prefill walks, copied as the walk read them to a pinned host pool (`rebuild_host_mb`) once that step completes, and appends each sample from the stop check's host copy. A walk can read more than the ids, such as the encoder's output a Qwen3.5 prompt with images reads. Once the request's prefill step has run, the worker copies those inputs to the same pool (1024 MB by default on such a worker), after the step's outputs, or on a pass of the worker's loop that runs no step once its last copy has landed, and within the bytes #876 lets a step copy to the host. The copies go back to the device when the row's step is built, on a stream of the pool's apart from its copies to the host, and only that step waits for them. The prefill row's step metadata stays out of the pool, in a dict on `KeptIds`, and every rebuild row runs under it. A request takes one slice of the pool for its ids and copies when it arrives. Its size comes from the layout: 8 B for each slot, since a walk may read an id at a copied slot too, and for each copied slot the values the policy declares, in the node's dtype. A request the pool can't hold is then known before it is admitted, and takes the class of one no drop can free. The worker keeps all of this per node and per handle, and lists it in `rid_table.py`.

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

A guaranteed request reserves its whole footprint and is never preempted, so once it is admitted and guaranteed it finishes.

Take the oldest request O, once every older request has finished:

1. No request chooses O as a victim, since a victim is newer than the request it is chosen for.
2. When O runs short, newer requests move to the host or are dropped first. When their moves can't cover the shortage, O moves to the host if the host holds it whole, and otherwise waits. A newer request whose ids or host copies haven't landed yet can't be dropped, the same as one that never can, so O may move itself to the host while that request's copies land. O still finishes, after a trip to the host it could have avoided.
3. While O waits, on the host or holding its pages, every other request that holds pages is newer, and no newer prompt starts in O's pools. Each one that runs either finishes, which grows the room, or runs short and, with nothing newer to cover its shortage, moves to the host or waits too. O then moves the newer requests that wait, chosen together as one set, and takes the room as their moves land. A newer request with images can be dropped only once its host copies have landed, and with every request waiting no step runs, so the worker starts those copies on its passes that run no step. None of them is guaranteed, since a guaranteed request never runs short, and one no drop can free moves into the room the host set aside for it (§1). O's footprint fits the pool, so those pages and the room cover it once every newer request has finished or waits.
4. So O runs again each time it waits, and each time it has grown by at least a page. From the host it returns with every page its grant needed, so that grant can't fail again. Its footprint fits the pool, so it finishes.

The next oldest request then takes its place in the argument. Two requests that hold pages and wait can't wait on each other for good, since the older one, or an older request still, moves the newer one. A dropped request returns with its whole history and one more page reserved, so each time it is preempted again it has grown by at least a page, and once it is the oldest no newer prompt passes it. Newer prompts, guaranteed ones included, can start in the room a move left until every older request has finished, and none starts in the moved request's pools after that, so the argument holds while new requests keep arriving.

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

- **Drops cost a rebuild.** A victim the host pool can't hold is dropped and rebuilt. Under sustained overload the newest requests are dropped again each time an older request runs short.
- **A lone rebuild isn't speculated.** Speculation targets come from the graph runtime, which holds no rebuild row, so a step of rebuild rows alone pays one scheduling gap per chunk. A rebuild chunk rides speculation when another row of its step gives a target.
- **Kept ids.** A node with a rebuild policy copies each request's prompt ids to a pinned host pool, 64 MB by default, and keeps them until the request finishes: 256 KB for a 32K-token request. A request the pool can't hold is never dropped on that node, though with a host pool it can still move to the host.
- **Shared prefixes with host offload.** A hit on a page only the index holds is charged as fresh, so prompts sharing a prefix run fewer at once. With host offload on, 32 Bagel requests over four shared prefixes on 96 pages took 1.49x to 1.66x as long as they do today.
- **Guaranteed requests queue.** On Orpheus at c16, streamed TTFA p99 rose from 247 ms to 2.58 s against preemption without classes. But, I am fine with the p99 cost since for voice, a later first chunk is better than a gap in the middle of playback.

---

## 9. Next steps (Out of Scope here)

- **Pool sizing.** Fixed, hand-set pool sizes can also cause thrashing, which this work surfaced. Each cache's `max_num_pages` comes from the config, 2048 pages by default, so a pool can fill and preempt long before memory runs out: Command A+'s TP8 config gets about 4 GB of KV per GPU where, by my estimate, about 20 GB is free after weights. Except for M*, every other engine sizes the pool from free memory instead.
- **One pool per GPU.** SGLang has moved from a fixed pool per cache type to one physical pool, where full-attention KV and window or recurrent state grow from opposite ends and compaction remaps pages, since agent contexts vary too much in length for fixed ratios. M*'s caches on a worker could share one pool the same way, but the effects would need to be measured.
