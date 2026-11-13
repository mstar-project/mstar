---
orphan: true
---

# Episode state across requests (design, not implemented)

Status: proposed, for agreement on the contract before any code. The motivating case is Praxis-1 (#1460), a robot policy whose next call is conditioned on what the robot did since its last call.

M* has content-keyed caches across requests (the prefix cache, Chatterbox's voice conditioning), but nothing keyed by a client. A robot client over openpi's WebSocket (#1204) sends one request per observation, so anything a policy wants to carry from one call to the next (text embeddings computed once per task, the executed past that conditions the next chunk) is gone when each call ends. AGENTS.md invariant 1 puts session state in the system, not the model. This design adds that state as a system concept, an **episode**: a sequence of requests from one client that share retained arrays, which the model declares and the engine holds, so the model object keeps no mutable per-session state.

| Part | Section |
|---|---|
| What happens today | §0 |
| What the reference keeps | §1 |
| The contract | §2 |
| Identity | §3 |
| Reset | §4 |
| What is retained, and where | §5 |
| Memory limits | §6 |
| Cleanup | §7 |
| Concurrent requests | §8 |
| Praxis-1 and other users | §9 |
| Migration and compatibility | §10 |
| Open questions | §11 |

---

## 0. What happens today

- **A call is a request.** The Python openpi handler submits one request per observation with no request id ([openpi.py](../../mstar/api_server/openpi.py) `_infer`, lines 240-248), so `submit_request` mints a fresh uuid4 ([entrypoint.py](../../mstar/api_server/entrypoint.py) lines 855-856). The Rust handler does the same with `rid("openpi")` ([ws/openpi.rs](../../rust/server/src/ws/openpi.rs) line 521, [serving.rs](../../rust/server/src/serving.rs) lines 328-330). Both send the prompt on every call (`openpi.py` lines 202-206, `openpi.rs` lines 463-467).
- **A socket runs one call at a time.** Both loops receive, wait for the reply and send it before reading again (`openpi.py` lines 281-303, `openpi.rs` lines 563-613). Any failure answers with the error as text and closes the socket (`openpi.py` lines 295-298, `openpi.rs` lines 593-605).
- **A persist edge lasts one request.** Persisted signals live on the request's `RequestData` (`persist_signals`, [conductor.py](../../mstar/conductor/conductor.py) lines 372-375), seeded from what the data worker stored for this request ([data_worker.py](../../mstar/api_server/data_worker.py) lines 1112 and 1124). A later pass replaces what an earlier pass held under a name (`conductor.py` lines 1977-1993). When the request is done, the conductor unpersists everything still held (lines 1894-1902) with `UNPERSIST_TENSORS`, the producer clears the flag ([input_ingest.py](../../mstar/worker/input_ingest.py) lines 282-285), and the tensor store frees the tensor once its references drop ([rust/src/tensors.rs](../../rust/src/tensors.rs) lines 792-797).
- **The plan is decided at ingest, from this request's signals only.** The conductor calls `walk_plan` with `body.initial_signals` (`conductor.py` lines 1397-1401; the contract is [base.py](../../mstar/model/base.py) lines 999-1027). A walk input with no persisted signal gets an empty edge ([request_info.py](../../mstar/conductor/request_info.py) lines 102-110).
- **Engine state is per request.** A declared array is "per request, per label and per declared layer" ([contract.py](../../mstar/engine/state/contract.py) line 213). The engine registers a request's state on its first `NEW_REQUEST` ([input_ingest.py](../../mstar/worker/input_ingest.py) lines 199-216, `Engine.add_request` in [engine.py](../../mstar/engine/engine.py) line 3906), every resource keys it by the worker's request handle, and all of it is purged on `REMOVE_REQUEST` ([rid_table.py](../../mstar/worker/rid_table.py) lines 8-31, `Engine.remove_request`, `engine.py` lines 3937-3954).
- **The only cross-request reuse is content-keyed.** The prefix cache keeps a finished request's KV pages under a hash of what they depend on (`Model.kv_labels`, `base.py` lines 885-895, and `.claude/skills/prefix-caching/SKILL.md`), and Chatterbox keeps an LRU of voice conditioning by the clip's hash ([chatterbox/submodules.py](../../mstar/model/chatterbox/submodules.py) lines 88-108). The replica router places each request on its own (`ReplicaRouter.pick`, [routing.py](../../mstar/conductor/routing.py) line 177). Nothing is keyed by a client.
- **The one long-lived call is a request with open client input** (client streams, [topology.py](../../mstar/streaming/topology.py) lines 30-35; the realtime conversation, [realtime/conversation.rs](../../rust/server/src/realtime/conversation.rs) lines 6-8). At capacity the conductor refuses such a call rather than queueing it (`conductor.py` lines 1291-1298).

So for #1460 as written: `walk_plan` never finds `text_embeds`, `encode_text` runs on every call, and `chunk_latents` is unpersisted when the call ends.

---

## 1. What the reference keeps per episode

The reference server (`praxis-1-oss`) keeps one session per WebSocket connection (`praxis/serve/server.py` lines 9-11), built when the socket opens and dropped in a `finally` when it closes (lines 134-167). Its protocol v2 also mints a `session_id` per connection (lines 148-154) and takes `end_session` (lines 199-205). The table describes a chunked session, where each call answers one chunk; a real-time session (`schedule=rtc`) gets one sample per call and also keeps a sample history and precomputed work (`policy.py` lines 605-711), which this design does not try to cover.

| State | Size | Lifetime |
|---|---|---|
| Executed frame window (`ContiguousStreamContext`, `praxis/robot_io.py` lines 298-336) | float32 clips `(1, 3, T, 256, 832)` on the GPU (tile 256x416, two views, `single_arm.py` lines 94 and 461-466): 2.44 MiB a frame. `T <= 21 + 24 (d - 1)`: 21 frames (51 MiB of window) at the server default of one context chunk (`policy.py` line 64), 69 frames (168 MiB) at three. These are the logical window; the slice keeps the concatenated buffer under it, so the allocation is larger (`robot_io.py` lines 403-419). As uint8 the window would be a quarter of that. | Grows by the executed chunk each call, drops the oldest 24 frames past the cap; cleared by a context restart |
| Boundary latents and poses (`single_arm.py` lines 1094-1097, 1185-1191) | the last `d + 1` single-frame latents and 8-float poses, under 1 MiB | as the window |
| Text embeddings (`praxis/pipeline.py` lines 121, 145-148) | `(1, S, 2048)` bf16 (`text_encoder.py` line 169), 4 KiB a token, plus the blank caption | memoized for the whole process by prompt string, not per session |
| `chunk_index` (`single_arm.py` lines 1093, 1290) | an int | counts completed chunks: seeds chunk k with `base + k` (`pipeline.py` line 469) and counts the budget; reset only at session end |
| Prompt (`single_arm.py` lines 1139-1144) | a string | an observation's `prompt` replaces it without restarting the context |

How it resets and ends:

- **Context restart.** An observation's `reset_context` (or `__reset_context__`) drops the window and the boundary state, but keeps `chunk_index` (`single_arm.py` lines 1105-1115 and 1137-1138). The session also restarts by itself when the frame stack is missing or too short (line 1173), when a real-time gap is off the frame grid (`policy.py` lines 678-691), and on every call in stateless mode (lines 571-572).
- **Episode end.** Closing the socket or `end_session` calls `close`, which drops the context and resets `chunk_index` (`policy.py` lines 769-780); the session object is then discarded. The process-wide text memo survives.
- **Budget.** `max_chunks_per_episode` (default unset, `policy.py` line 72) is checked at the start of a call (lines 567-570). Past it the server sends the error and stops reading (`server.py` lines 267-271).
- **A failed call ends the session**, since it may have changed the context (`server.py` lines 272-279).
- **Concurrency.** All calls of all sessions run one after another on one thread (`server.py` lines 44-45), so there is no batching across sessions. Real-time sessions may have 16 calls in flight (`periodic.py` line 66), and they are served in order.

Two facts here shape the design. The context is the **executed real frames**, which a chunked client sends as a frame stack of up to 25 frames each call; conditioning on the model's own predicted video ("rolling") is refused because the model trained on real frames only (`policy.py` lines 271-276). And the server retains frames only because a context of more than one chunk spans more frames than one observation carries.

---

## 2. The contract

An episode is named by the frontend, owned by the conductor, and held by the engines of the workers its arrays live on. A model declares what it retains as arrays with `lifetime="episode"` in `declare_state`. The engine keeps those arrays across the episode's requests in a fixed slot per episode, a request's steps read what earlier requests kept and append their own, and `walk_plan` sees how much each array holds, so it can skip work whose result is retained. Requests with no episode behave exactly as today.

```python
# mstar/engine/state/contract.py
@dataclass(frozen=True)
class Array:
    ...
    lifetime: Literal["request", "episode"] = "request"   # episode: kept across the episode's requests

# what walk_plan and step_metadata see; built by the conductor, not by the model
class EpisodeView(NamedTuple):
    id: str | None              # None: a request with no episode, which starts cold every call
    calls: int                  # calls of this episode admitted before this one
    held: Mapping[str, int]     # episode array -> entries it holds now (0 after a reset or an eviction)
    cold: str | None            # why held is empty when calls > 0: "reset", "evicted", "idle", "lost"

# mstar/model/base.py
class ProcessPromptOutput(NamedTuple):
    ...
    episode_keys: Mapping[str, str] = MappingProxyType({})   # episode array -> what its entries were made from
    reset_episode: bool = False                              # a context restart, asked from this request's inputs
```

```yaml
# serving config
episodes:
  max_episodes: 16        # slots per replica; every episode array is sized max_episodes x its max_len
  idle_timeout_s: 600     # an episode with no call for this long is ended
  max_calls: null         # the reference's max_chunks_per_episode; null is unbounded
  max_queued_calls: 1     # calls of one episode waiting behind the one in flight
```

---

## 3. Episode identity

**Proposal.** The frontend mints an episode id for each openpi connection when the socket opens (`ep-<uuid4>`), and every call on that socket carries it. It goes to `submit_request` as a system field beside `request_id` (`episode=`; `SubmitArgs.episode` in Rust), not in `model_kwargs`, through the data worker to `NewRequestConductor.episode`. A request with no episode (every HTTP route, and any model that declares no episode arrays) runs as today.

The conductor keeps one record per episode:

```python
@dataclass(slots=True)
class EpisodeRecord:
    calls: int                  # admitted calls
    placement: Placement        # the replica of each node group, fixed by the first call
    slot: int                   # the episode's slot index, the same on every rank of a lockstep instance
    held: dict[str, int]        # entries per array, from the workers' WORKER_GRAPHS_DONE
    keys: dict[str, str]        # what each array's entries were made from
    in_flight: str | None       # the request running now
    queued: deque               # calls waiting behind it (§8)
    last_end: float             # for the idle timeout and LRU (§6, §7)
```

Why this shape:

- **The connection is the episode in both the reference and openpi.** The reference opens a session per connection. openpi's `WebsocketClientPolicy.reset()` is a no-op, so an openpi client tells the server nothing at episode boundaries, and the socket is the only identity it has. A connection-scoped id needs no client change.
- **Minted by the system, never chosen by the client.** A client-chosen id can collide with or guess another client's, so it needs scoping to the caller's credentials, and its only cleanup is a timeout. It also survives a reconnect, which the reference does not attempt: a new connection starts cold there. This is left as an open question (§11), and the field is shaped so it can be added.
- **The conductor owns the record.** It already owns the request lifecycle and the replica pick, and it must decide eviction and slot assignment so every rank of a lockstep instance agrees (invariant 11). A record on a worker would need rank 0 to journal its choices.
- **The placement is pinned.** The retained arrays live on the replica the first call ran on, so later calls skip `ReplicaRouter.pick` and go there.

Rejected:

- **A derived id** (a hash of the prompt, the robot's state keys or the client address): two robots on one task, or behind one NAT, would share an episode.
- **No id, the client sends its context back.** The client already sends the last executed chunk, but a context of more than one chunk needs more frames than an observation carries (69 against 25 at three chunks), and every call would ship and re-check them. It also leaves text embeddings recomputed every call. The maintainers' question was this choice, and the decision was system-owned state.

---

## 4. Reset

**Proposal.** Three ways a context restarts, and one way an episode ends early:

1. **Explicit.** An observation with `reset_context: true` (or `__reset_context__`, as the reference reads it) sets `episode.reset` on its request. Both frontends' openpi parsers read the key.
2. **Model-requested.** `process_prompt` sets `reset_episode` when this request's inputs can't continue the context: the reference's missing or short frame stack is the case. The data worker ORs it with the client's flag.
3. **System.** An evicted, idle-ended or lost episode starts its next call cold (§6, §7), with `EpisodeView.cold` saying why.
4. **Budget.** `episodes.max_calls` refuses the call past it with a 400, `episode reached max_calls=N`. Both frontends answer a failure with its text and close the socket, which ends the episode (§7).

A restart clears every episode array of the episode before the request's first step: the conductor marks the request's `NEW_REQUEST` to the episode's workers, and each engine empties the slot when it binds the request. `calls` keeps counting, as the reference's `chunk_index` does, so the seed sequence and the budget run over the whole connection.

An array also clears when its key changes. A request's `episode_keys` names what each array's entries are made from (for text embeddings, a digest of the prompt's token ids, which the data worker has). When the key differs from the record's, the conductor treats that array as empty for this request (`held[name] = 0`) and the engine clears it on bind, so `walk_plan` runs the producer again. This is what lets a prompt change mid-episode re-encode the text without dropping the frame window, as the reference allows.

Why this shape:

- **A restart is not an episode end.** The reference keeps `chunk_index` across `reset_context`, and a client that reuses one socket for many episodes would otherwise have to reconnect to reset its budget.
- **Applied on bind, not by a separate message.** The flag rides the request it belongs to, so it is ordered with that request on every rank and can't race the previous call's commit.
- **The model asks; the engine clears.** The model can see that a stack is too short, but holds nothing it could clear.

Rejected:

- **Implicit reset on a prompt change.** The reference changes the prompt without restarting the context, and the key rule covers the one array that depends on the prompt.
- **Resetting silently at `max_calls`.** The robot would keep moving on a cold context; the reference stops the session.
- **Keeping some arrays across a restart.** Clearing everything is one rule, and re-encoding a prompt costs one text-encoder pass. §11 asks whether a model needs more.

---

## 5. What is retained, and where

**Proposal.** A model declares each retained thing as an `Array` with `lifetime="episode"`:

- `max_len` bounds the episode, not the request.
- `nodes` say which node reads and writes it; the array lives on the workers that run that node, inside their engines, one slot per episode on every rank of the node's lockstep instance (replicated, or split where a `TensorType.shard` says).
- A step appends with `handle.add` ([calls.py](../../mstar/engine/state/calls.py) line 194) and reads with `handle.read` (line 201). An empty `reads` reads the whole array (`contract.py` line 228); a `Window` read says what stays reachable, and the engine frees what falls out of it, as `Read`'s docstring already promises (lines 158-161).
- A no-op read returns `Reached`, each field's entries packed by row (`contract.py` lines 470-474).

The conductor holds only the record (§3). The model object holds nothing: the calls count reaches a step as `step_metadata["episode_calls"]`, and everything else is in the arrays.

For Praxis-1 (§9) that is two arrays: `text_embeds` on the backbone, entry `{"h": TensorType((2048,), dtype=bf16)}` with labels `cond` and `blank`, and `executed` on the VAE encoder, entry `{"frame": TensorType((3, 256, 832), dtype=uint8), "pose": TensorType((8,), dtype=float32)}`, `max_len` 69.

**Does it fit `declare_state`?** The description fits: `Entry` takes any named tensors with a dtype and a shard, and `max_len`, `labels`, masks and the no-op `Read` say the rest. Five things are missing:

1. **A lifetime.** Arrays are per request today (`contract.py` line 213). `lifetime` is the only new field on the declaration.
2. **A storage.** No storage holds an append-only array that is not K/V or serves a no-op read. `PagedStorage` takes only a `KVEntry` with an `Attend` read ([adapters.py](../../mstar/engine/state/adapters.py) lines 194-199 and 1256), `storage_of` sends every other append-only array with a nonzero `max_len` there (lines 1391-1397), and nothing outside `contract.py` returns `Reached`. This needs an `EpisodeStorage`: a dense buffer of `max_episodes x max_len` entries per layer, a row addressed through its slot index, staged in `plan` with `force_double_buffer` (invariant 7), so eager and captured steps address it alike.
3. **Binding a request to an episode.** The engine registers a request's state on `NEW_REQUEST` and purges it on removal (`engine.py` lines 3906 and 3937-3954). An episode array instead attaches the request to the episode's slot on `NEW_REQUEST` (clearing it on a reset or a key change), detaches it on `REMOVE_REQUEST` without freeing it, and frees the slot on `END_EPISODE`. The slot is keyed by the episode, not the request handle, so `rid_table.py`'s purge rule doesn't apply to it; the binding from handle to slot is handle-keyed and is listed there.
4. **What the plan sees.** `walk_plan` gets only this request's signals (`base.py` lines 999-1006). It gains an `episode: EpisodeView` keyword, passed only to a model that declares episode arrays, so no existing override changes. `held` comes from the workers' `WORKER_GRAPHS_DONE`, as persisted signals already do (`conductor.py` line 1977).
5. **Keys.** `ProcessPromptOutput.episode_keys` (§4), since a retained value made from a prompt is only valid for that prompt.

Why this shape:

- **Where the reader is.** The arrays sit on the node that reads them, so a warm call moves no retained bytes between GPUs. A persisted tensor stays with its producer and is read over the transport on each call.
- **The engine owns the memory.** A slot pool sized at startup is something admission can count (§6); state a model allocated itself would turn a full pool into an OOM (invariant 1, heuristic 4).
- **One contract for per-request and per-episode state.** The same `add` and `read` serve both, so a model moves an array between lifetimes by changing one field.

Rejected:

- **Episode-lifetime persist edges.** It is the smallest change to #1460: keep `persist_signals` on an episode record instead of the request. But the tensors would stay in the producer's tensor store, which admission doesn't count and nothing evicts under pressure (a persisted tensor is collected once the conductor unpersists it, `tensors.rs` lines 792-797, or at the request's hard teardown, [communication/tensors.py](../../mstar/communication/tensors.py) lines 2191-2198), their descriptors would go through the conductor every call, and a window that slides needs the model to re-emit the whole window each call, since a pass replaces a name (`conductor.py` lines 1977-1993).
- **One long-lived request with open client input**, as realtime conversations run. A request's walks are planned once at ingest (`base.py` lines 1007-1008), where an episode is an unbounded run of calls each with its own plan. A long request stays live while the robot is idle, holding what it opened (only some resources can be offloaded under pressure), and the conductor refuses one at capacity (`conductor.py` lines 1291-1298). Per-call timeouts, metrics and errors would also blur into one request.
- **Holding it in the frontend or the API server.** Retained tensors would cross into and out of the GPU workers every call, which is the client-sends-it-back design with one more hop.
- **A content-keyed cache for text embeddings**, as the reference memoizes them for the process (`pipeline.py` line 121). It would share embeddings across robots on one task, but it is a separate feature (an encoder-output cache, with its own eviction across tenants), and an episode already reuses the embeddings for as long as its prompt stays the same, which in a robot episode is usually every call. A content cache would add reuse across episodes, or after a prompt switches away and back. §11 asks whether to build it too.

---

## 6. Memory limits

**Proposal.**

- **Per episode.** An episode's footprint is fixed by its declarations: the sum over its episode arrays of `max_len` x entry bytes x layers, per rank. For Praxis-1 at three context chunks it is 69 x 0.61 MiB of frames plus a few MiB of text, about 44 MiB; at one chunk, about 15 MiB. Per-deployment `max_len` goes in `resources:` as today (invariant 2).
- **Per replica.** `episodes.max_episodes` slots, allocated at startup on each worker that holds an episode array. A worker whose slots don't fit refuses to start, naming the array and the size.
- **Admission.** A call for a new episode, or for one whose slot was taken, needs a slot on the replica the router picks. It takes a free slot, or evicts the least recently used idle episode there (no call in flight or queued). When every slot holds an episode with a call in flight or queued, the call waits in the conductor's queue, in arrival order, as a request over `max_concurrent_requests` waits today (`conductor.py` lines 1288-1305). A call of an episode that has its slot never waits for one.
- **Eviction.** The conductor chooses, LRU by `last_end` over idle episodes on that replica, and sends `END_EPISODE` to the episode's workers before the new episode's `NEW_REQUEST`. An episode with a call in flight or queued is never evicted.
- **What the client sees.** The evicted episode's next call starts cold: `held` is empty, so `walk_plan` runs every producer again, and the call is otherwise correct. Its reply says so (§10): `"episode": {"calls": n, "cold": "evicted"}`. A metric counts evictions and cold starts by reason.

Relation to [admission_and_preemption.md](admission_and_preemption.md): an episode slot is a fixed-size slot like the recurrent state pool and the sampler's rows there (§3: a request waits at admission until a slot is free, and never runs short after), except that it outlives its requests, so its idle time is what eviction takes back. Slots are not KV pages, so KV preemption never touches them, and a KV victim keeps its episode. A node with a rebuild policy needs every resource to support a drop (that doc's §4 table); `EpisodeStorage` supports one by keeping its entries, as attention does.

Why this shape:

- **Slots, not pages.** Each array's per-episode size is known at startup, so a slot per episode makes admission a count and eviction one message, with no fragmentation. The ring KV manager hands out fixed sessions the same way ([ring/manager.py](../../mstar/engine/resources/kv/ring/manager.py) lines 344-349).
- **Idle first.** An idle episode costs a cold start when its robot comes back; a busy one would be corrupted mid-call.
- **The conductor decides.** It is the only process that sees every episode's state on a replica, and its choice reaches every rank in the same order (invariant 11).

Rejected:

- **Sharing the KV pool.** It would let episodes and KV trade memory, but it ties this design to preemption's classes before either has been measured together. §9 of the admission doc already lists one pool per GPU as later work.
- **Refusing a new episode when full.** A busy server would turn away robots while idle ones held their slots.
- **Moving idle episodes to the host.** Worth it if cold starts turn out expensive, and left for later (§11).

---

## 7. Cleanup

**Proposal.** An episode ends, and its slot frees, on the first of:

| Event | Who sends `END_EPISODE` | Notes |
|---|---|---|
| Connection close | the frontend, in the handler's `finally` (Python) or when `serve` returns (Rust) | a call in flight runs to its end first, since the handler notices the close only when it sends the reply (`openpi.py` line 305); the end is applied after the call's teardown |
| A failed or aborted call | the conductor, at the request's teardown | its arrays may hold a partial write, as the reference assumes (`server.py` lines 272-279); openpi closes the socket anyway |
| `max_calls` reached | the conductor | the refused call fails, and the socket closes (§4) |
| Idle for `idle_timeout_s` | the conductor's periodic sweep | the record stays with `cold="idle"` until the connection closes, so the next call reports why it is cold |
| Eviction | the conductor (§6) | as idle, with `cold="evicted"` |
| A worker holding it is lost | the conductor ends every episode placed there | `cold="lost"` on the next call, if the conductor outlives the worker |
| Process restart | nothing to send | every episode is in memory and is gone; connection-scoped ids die with their sockets, so no client can name a stale one |

Leaked episodes are bounded three ways: the slot count is a hard cap on retained memory, which was allocated at startup anyway; the idle timeout returns any slot a lost close left behind; and a record without a slot holds no tensors and lives only as long as its connection.

Why this shape:

- **Close is the normal end.** It is the reference's, and openpi gives nothing else.
- **A failure ends the episode rather than rolling back.** Undoing a partial write needs a copy of every array before every call; ending it costs one cold start in a case where the client has closed the socket anyway.
- **The timeout is a backstop, not the main path.** With connection-scoped ids it only fires for a frontend that died without closing, or a robot paused longer than the timeout, whose next call then starts cold. The reference's hosted deployment caps a session at an hour (`modal_app.py` line 94).

---

## 8. Concurrent requests

**Proposal.**

- **One call per episode at a time.** The conductor admits an episode's call only when none of its calls is in flight. A call that arrives meanwhile waits in the episode's queue, in arrival order, up to `max_queued_calls`; past that it is refused with a 429. A queued call is planned when it is admitted, not when it arrives, so its `walk_plan` sees what the call before it kept.
- **Many episodes on one worker.** Each call is an ordinary request. The micro-scheduler batches calls of different episodes as it batches any requests, through `can_batch` and `forward_batched`; each row reads and writes only its own slot, which the plan addresses per row. Nothing reads across episodes.
- **Speculation.** The async worker speculates within a request (invariant 7). Two calls of one episode are never in flight together, so nothing speculates across the boundary between them.

Why this shape:

- **Calls are ordered by their state.** Each call reads the history and appends to it. Two at once would read the same history and both append, so the second's context would miss the first's chunk. The reference serializes every call (`server.py` lines 44-45).
- **Queue, not refuse.** openpi's frontends never send two calls at once (§0), so for them the queue is always empty. A pipelining client, as the real-time reference (16 in flight) and the Cosmos3 example over `/generate/ws` (`examples/cosmos3_action_ws_client.py`) are, would see a refusal as a dropped observation. A bound keeps a runaway client from building a queue.

Rejected:

- **Running them concurrently on a snapshot.** Last writer wins, and which call wins depends on timing.
- **Serializing on the worker.** The worker would hold a ready request back by state only the conductor has, and every rank would have to agree on it.

---

## 9. Praxis-1 and other users

**Praxis-1 on episodes.**

- `walk_plan` runs `encode_text` only when `episode.held["text_embeds"] == 0` (a new episode, a restart, an eviction or a new prompt), then the chunk walk. On a cold call the text embeddings reach the backbone as today, persisted within the request, and its first step adds them to `text_embeds`. On a warm call the walk's `text_embeds` input is an empty edge (`request_info.py` lines 105-106), and every step reads the array.
- The VAE encoder appends the newly executed rows of the observation's frame stack to `executed` (the whole chunk on the chunk grid, `executed_rows` in real time) and reads the window, which it encodes into the clean context as the reference does each call.
- The seed of call k is the episode's seed plus `episode_calls`, the reference's `base + k`.
- **A correction to #1460.** Its docstring says the predicted video conditions the next call, and it persists `chunk_latents`. The reference conditions on the executed real frames and refuses its own predictions as context (`policy.py` lines 271-276; `robot_io.py` lines 298-317). The array to retain is the executed frame window, not the predicted latents.

Until this lands, Praxis-1 (once its preprocessing and submodules are in) can serve every call cold, which is correct and is the reference's stateless mode, and moves to episodes by changing its declarations and `walk_plan`.

**Other users.** Candidates, not checked against their references:

- Any robot policy that conditions on more history than one observation carries. pi0.5 conditions on the current observation only and is unaffected.
- Interactive world models. Waypoint keeps a ring KV per request today (`ring/manager.py`); a session that spans requests could keep it as an episode array once K/V arrays can have an episode lifetime.
- Multi-turn agent and chat sessions, which today rely on content-keyed prefix reuse. An episode-lifetime KV would keep a session's KV without re-matching it, but it needs the KV manager to hold episode slots and a class under the admission doc, so it is out of scope here (§11).
- Full-duplex models already run a call as one long request (§0) and don't need this.

---

## 10. Migration and compatibility

- **openpi wire.** No new required key. `reset_context` is optional, and today's parsers already ignore keys they don't name (`OpenpiPolicy.request`, `openpi.py` lines 193-215; `request`, `openpi.rs` lines 449-496). The reply gains an optional `"episode"` map; `openpi_client`'s `infer` returns the unpacked reply, so a client that doesn't read it is unaffected. The metadata frame is unchanged.
- **Both frontends change together** (invariant 15): the id per connection, `reset_context`, `END_EPISODE` on close and the reply field land in `api_server/openpi.py` and `rust/server/src/ws/openpi.rs` in one PR, with a case in `test/rust/test_rust_openpi.py` that runs one episode against both and compares what reaches `submit_request` and what the client gets back.
- **Models that declare no episode array are untouched.** The conductor knows at startup whether the model declares any. If not, it makes no record, routes as today and never passes `episode` to `walk_plan`, so the request path and its per-request cost are today's (invariant 6). The frontends' minted id is a string on the submission, dropped at the conductor.
- **Routes other than openpi** pass no episode in this design.

Staging, each step useful alone:

| Step | Contents |
|---|---|
| E1 | `lifetime`, `EpisodeStorage`, bind and detach in the engine, `END_EPISODE`; a modular test drives two requests through one slot |
| E2 | The conductor's record, pinned placement, `EpisodeView` to `walk_plan`, per-episode queue, `max_calls`, keys |
| E3 | Slots, eviction, idle timeout, cleanup on failure and worker loss, metrics |
| E4 | Both openpi frontends: the id, `reset_context`, close, the reply field |
| E5 | Praxis-1 on episodes |

---

## 11. Open questions

1. **Client-supplied ids.** Should a client be able to name its episode, to survive a reconnect or to drive episodes over `/generate/ws`? If so, scoped to its API key, and is the idle timeout then the only cleanup?
2. **The cold-start notice.** Is a top-level `"episode"` key in the openpi reply acceptable, or should it go under `server_timing`, or be left out of the wire and only counted?
3. **Text embeddings across episodes.** Is a content-keyed encoder-output cache, as the reference's process-wide memo, worth building alongside, for many robots on one task?
4. **Idle episodes to host memory** instead of dropping them, once cold-start cost is measured?
5. **Episode-lifetime K/V arrays** for agent sessions: a follow-up design under the admission doc, or out of scope for M*?
6. **Per-connection settings.** The reference takes `max_chunks_per_episode` and sampler settings per session from the URL query. Should openpi connections take any, or is the deployment config enough?
7. **Restart scope.** Is clearing every array on a restart enough, or does some model need an array kept across `reset_context`?
8. **Defaults.** `max_episodes` 16 and `idle_timeout_s` 600 are guesses until measured on a robot workload.
