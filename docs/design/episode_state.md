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

The baseline code observations in this section and §5 were checked at [main 9668497](https://github.com/mstar-project/mstar/tree/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76); file links there are pinned to that snapshot.

- **A call is a request.** The Python openpi handler submits one request per observation with no request id ([openpi.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/api_server/openpi.py) `_infer`), so `submit_request` mints a fresh uuid4 ([entrypoint.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/api_server/entrypoint.py)). The Rust handler does the same with `rid("openpi")` ([ws/openpi.rs](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/rust/server/src/ws/openpi.rs), [serving.rs](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/rust/server/src/serving.rs)). Both send the prompt on every call (`openpi.py`, `openpi.rs`).
- **A socket runs one call at a time.** Both loops receive, wait for the reply and send it before reading again (`openpi.py`, `openpi.rs`). Any failure answers with the error as text and closes the socket (`openpi.py`, `openpi.rs`).
- **A persist edge lasts one request.** Persisted signals live on the request's `RequestData` (`persist_signals`, [conductor.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/conductor/conductor.py)), seeded from what the data worker stored for this request ([data_worker.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/api_server/data_worker.py)). A later pass replaces what an earlier pass held under a name. When the request is done, the conductor unpersists everything still held with `UNPERSIST_TENSORS`, the producer clears the flag ([input_ingest.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/worker/input_ingest.py)), and the tensor store frees the tensor once its references drop ([rust/src/tensors.rs](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/rust/src/tensors.rs)).
- **The plan is decided at ingest, from this request's signals only.** The conductor calls `walk_plan` with `body.initial_signals` (`conductor.py`; the contract is [base.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/model/base.py)). A walk input with no persisted signal gets an empty edge ([request_info.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/conductor/request_info.py)).
- **Engine state is per request.** A declared array is "per request, per label and per declared layer" ([contract.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/engine/state/contract.py)). The engine registers a request's state on its first `NEW_REQUEST` ([input_ingest.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/worker/input_ingest.py), `Engine.add_request` in [engine.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/engine/engine.py)), every resource keys it by the worker's request handle, and all of it is purged on `REMOVE_REQUEST` ([rid_table.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/worker/rid_table.py), `Engine.remove_request`, `engine.py`).
- **The only cross-request reuse is content-keyed.** The prefix cache keeps a finished request's KV pages under a hash of what they depend on (`Model.kv_labels`, `base.py`, and `.claude/skills/prefix-caching/SKILL.md`), and Chatterbox keeps an LRU of voice conditioning by the clip's hash ([chatterbox/submodules.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/model/chatterbox/submodules.py)). The replica router places each request on its own (`ReplicaRouter.pick`, [routing.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/conductor/routing.py)). Nothing is keyed by a client.
- **The one long-lived call is a request with open client input** (client streams, [topology.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/streaming/topology.py); the realtime conversation, [realtime/conversation.rs](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/rust/server/src/realtime/conversation.rs)). At capacity the conductor refuses such a call rather than queueing it (`conductor.py`).

So for #1460 as written: `walk_plan` never finds `text_embeds` from a prior call, and `encode_text` runs on every call. Predicted video latents end at the denoise loop; they do not become context for a later call.

---

## 1. What the reference keeps per episode

The reference server (`praxis-1-oss`) keeps one session per WebSocket connection (`praxis/serve/server.py` lines 9-11), built when the socket opens and dropped in a `finally` when it closes (lines 134-167). Its protocol v2 also mints a `session_id` per connection (lines 148-154) and takes `end_session` (lines 199-205). The table describes a chunked session, where each call answers one chunk; a real-time session (`schedule=rtc`) gets one sample per call and also keeps a sample history and precomputed work (`policy.py` lines 605-711), which this design does not try to cover.

| State | Size | Lifetime |
|---|---|---|
| Executed frame window (`ContiguousStreamContext`, `praxis/robot_io.py` lines 298-336) | float32 clips `(1, 3, T, H, W)` on the GPU. The `praxis-1-droid` checkpoint uses 224x832 (two 224x416 tiles); YAM uses 256x1248 (three 256x416 tiles), from their `video` configs. The earlier DROID-K checkpoint uses 256x832. `T <= 21 + 24 (d - 1)`: at the server default of one context chunk (`policy.py` line 64), 21 frames use about 45 MiB (DROID) or 77 MiB (YAM); at three chunks, 69 frames use about 147 or 252 MiB. These are logical windows; the slice keeps a larger concatenated buffer under them (`robot_io.py` lines 403-419). The proposed uint8 store uses a quarter of the float32 frame bytes. | Grows by the executed chunk each call, drops the oldest 24 frames past the cap; cleared by a context restart |
| Boundary latents and poses (`single_arm.py` lines 1094-1097, 1185-1191) | the last `d + 1` single-frame latents and 8-float poses for DROID, or 16-float poses for YAM; small beside the frame window | as the window |
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

An episode is named by the frontend, owned by the conductor, and held by the engines of the workers its arrays live on. A model declares what it retains as arrays with `lifetime="episode"` in `declare_state`. The conductor gives each episode one logical slot identity for admission and ownership; each storage maps that identity to its own dense, paged or ring allocation. A request's steps read what earlier requests kept and append their own, and `walk_plan` sees how much each array holds, so it can skip work whose result is retained. A request without an episode id uses a temporary identity for any episode arrays its model declares; its data is released on `REMOVE_REQUEST`, so it starts cold on every call. Models with no episode arrays take the existing path without a slot.

```python
# mstar/engine/state/contract.py
@dataclass(frozen=True)
class Array:
    ...
    max_len: int | None           # None: unbounded logical log with a provable resident bound
    lifetime: Literal["request", "episode"] = "request"   # episode: kept across the episode's requests

# what walk_plan and step_metadata see; built by the conductor, not by the model
class EpisodeView(NamedTuple):
    id: str | None              # None: a request with no episode, which starts cold every call
    calls: int                  # calls of this episode admitted before this one
    held: Mapping[str, int]     # episode array -> entries it holds now (0 after a reset, expiry, loss or identity change)
    cold: str | None            # why held is empty when calls > 0: "reset", "idle", "lost", "identity"

# mstar/model/base.py
class ProcessPromptOutput(NamedTuple):
    ...
    episode_keys: Mapping[str, str] = MappingProxyType({})   # episode array -> what its entries were made from
    reset_episode: bool = False                              # a context restart, asked from this request's inputs
```

```yaml
# serving config
episodes:
  max_episodes: 16        # episode identities per replica; capacity follows declarations and read bounds
  idle_timeout_s: 600     # free a connected episode after this much time since call teardown
  max_calls: null         # the reference's max_chunks_per_episode; null is unbounded
  max_queued_calls: 1     # calls of one episode waiting behind the one in flight
```

The id follows a connection; the retained tensors stay on workers. A later call
uses the same placement and slot, while an idless call borrows a slot only for
its own request:

```mermaid
flowchart LR
    C[Positronic session connection] -->|session id on each call| F[Frontend]
    F --> R[Conductor: record and pinned placement]
    R -->|bind request to slot| W[Worker engines: episode arrays]
    W -->|held counts after call| R
    R -->|warm view for next call| P[Model walk plan]
    I[Other route: no episode id] -->|cold request| R
    R -->|temporary slot, freed on request end| W
```

---

## 3. Episode identity

**Proposal.** Once E4 lands, the Positronic frontend mints an episode id for each `/api/v1/session` WebSocket connection when it opens, and every call on that socket carries it. Positronic protocol v2 sends a 32-hex `session_id` in its ready frame, requires that same value in every request and rejects a mismatch (`server.py` lines 148-154 and 199-208); that connection-bound value is also its internal episode id. The id goes to `submit_request` as a system field beside `request_id` (`episode=`; `SubmitArgs.episode` in Rust), not in `model_kwargs`, through the data worker to `NewRequestConductor.episode`. A model with no episode arrays runs as today. An idless request to a model with episode arrays gets a temporary slot and a cold `EpisodeView` (§5), without creating an episode record.

The conductor keeps one record per episode:

```python
@dataclass(slots=True)
class EpisodeRecord:
    calls: int                  # admitted calls
    placement: Placement        # the replica of each node group, fixed by the first call
    slot: int                   # logical ownership/admission id, the same on every lockstep rank
    held: dict[str, int]        # entries per array, from the workers' WORKER_GRAPHS_DONE
    identity: tuple[bytes, bytes | None]  # startup weight epoch and final selected adapter digest
    keys: dict[str, str]        # model input dependencies for individual arrays
    in_flight: str | None       # the request running now
    queued: deque               # calls waiting behind it (§8)
    last_end: float             # last completed call teardown, for the idle timeout (§6, §7)
```

Why this shape:

- **The Positronic connection is the episode.** The reference opens one session per connection and ends it when the connection closes. A connection-scoped id follows that lifecycle.
- **Minted by the system, never chosen by the client.** A client-chosen id can collide with or guess another client's, so it needs scoping to the caller's credentials, and its only cleanup is a timeout. It also survives a reconnect, which the reference does not attempt: a new connection starts cold there. This is left as an open question (§11), and the field is shaped so it can be added.
- **The conductor owns the record.** It already owns the request lifecycle and the replica pick, and it must decide idle expiry and slot assignment so every rank of a lockstep instance agrees (invariant 11). A record on a worker would need rank 0 to journal its choices.
- **The placement is pinned.** The retained arrays live on the replica the first call ran on, so later calls skip `ReplicaRouter.pick` and go there.

Rejected:

- **A derived id** (a hash of the prompt, the robot's state keys or the client address): two robots on one task, or behind one NAT, would share an episode.
- **No id, the client sends its context back.** The client already sends the last executed chunk, but a context of more than one chunk needs more frames than an observation carries (69 against 25 at three chunks), and every call would ship and re-check them. It also leaves text embeddings recomputed every call. The maintainers' question was this choice, and the decision was system-owned state.

---

## 4. Reset

**Proposal.** Three ways a context restarts, and one way an episode ends early:

1. **Explicit.** A truthy `reset_context` or `__reset_context__` in an observation sets `episode.reset` on its request, matching the reference. Both frontends' Positronic handlers read the keys; integer `1` resets and `0` does not.
2. **Model-requested.** `process_prompt` sets `reset_episode` when this request's inputs can't continue the context: the reference's missing or short frame stack is the case. The data worker ORs it with the client's flag.
3. **System.** An idle-ended or lost episode starts its next call cold (§6, §7). A weight or selected adapter identity change also starts a cold call (§4). `EpisodeView.cold` gives the internal reason; whether the wire exposes it is open (§11).
4. **Budget.** `episodes.max_calls` refuses the call past it with a 400, `episode reached max_calls=N`. Both frontends answer a failure with its text and close the socket, which ends the episode (§7).

A restart clears every episode array of the episode before the request's first step: the conductor marks the request's `NEW_REQUEST` to the episode's workers, and each engine empties the slot when it binds the request. `calls` keeps counting, as the reference's `chunk_index` does, so the seed sequence and the budget run over the whole connection.

Before planning a request, the conductor compares its system identity with the episode record's. E2 represents it as `(startup_weight_epoch, final_selected_adapter_digest)`, with `None` for the base model. The data worker settles the adapter after `process_prompt`, including a model-chosen adapter, and forwards its raw weight digest independently of prefix caching. This is the same adapter digest used to make the domain-separated prefix root ([lora/identity.py](../../mstar/engine/resources/lora/identity.py)); the conductor compares the tuple rather than the root. On a mismatch, it marks **every** held episode array empty before `walk_plan`, sets `cold="identity"`, and the engine clears all those arrays when it binds the request. An array without a model-provided key therefore cannot retain values derived from old weights or an old adapter. The episode's `calls` count continues across this restart.

Separately, a request's `episode_keys` names the input dependencies of individual arrays (for text embeddings, a digest of the prompt's token ids, which the data worker has). A changed key clears only that array before planning and on bind. A prompt change can therefore re-encode text without dropping the executed-frame window, as the reference allows. These model keys do not replace the system identity on the whole record.

The weight epoch is a system value, not a model-chosen `episode_key`. Base weights do not hot reload today, so E2 uses a fresh conductor startup epoch and keeps a whole-record identity. The engine's prefix cache has separate checkpoint, preprocessing and resource fingerprints ([engine.py](../../mstar/engine/engine.py) `_open_prefix_caches`); its per-resource root is not available to the conductor. E2 adds an unconditional adapter digest on the data-worker-to-conductor wire, even when prefix caching is off. A future base-weight reload must drain old calls and rotate the epoch, or end all affected episodes before admitting new calls. `checkpoint_identity` reads the index or shard names and sizes, so it alone cannot detect an in-place shard edit of the same size.

Why this shape:

- **A restart is not an episode end.** The reference keeps `chunk_index` across `reset_context`, and a client that reuses one socket for many episodes would otherwise have to reconnect to reset its budget.
- **Applied on bind, not by a separate message.** The flag rides the request it belongs to, so it is ordered with that request on every rank and can't race the previous call's commit.
- **The model asks; the engine clears.** The model can see that a stack is too short, but holds nothing it could clear.

Rejected:

- **Implicit reset on a prompt change.** The reference changes the prompt without restarting the context, and the key rule covers the one array that depends on the prompt.
- **Resetting silently at `max_calls`.** The robot would keep moving on a cold context; the reference stops the session.
- **Keeping some arrays across a restart.** Clearing everything is the initial rule, and re-encoding a prompt costs one text-encoder pass. A different scope needs a concrete model use case.

---

## 5. What is retained, and where

**Proposal.** A model declares each retained thing as an `Array` with `lifetime="episode"`:

- An integer `max_len` keeps its current meaning: the maximum logical log length, with zero still meaning scratch within a step. `max_len=None` declares an unbounded logical log, initially only for append-only dense arrays with no positions and only no-op `Window` reads. Every read is from the current tail; arbitrary or backward queries are refused. The largest declared window bounds resident entries, plus room for an in-flight append and staging. There is no separate retention field.
- `nodes` say which node reads and writes it; the array lives on the workers that run that node, inside their engines, with one episode identity on every rank of the node's lockstep instance (replicated, or split where a `TensorType.shard` says). A paged store need not put that identity in one contiguous physical allocation.
- A step appends with `handle.add` and reads with `handle.read` ([calls.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/engine/state/calls.py)). A `Read(mask=Window(size=69))` describes which executed frames a Praxis step can reach ([contract.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/engine/state/contract.py)). The dense store keeps logical positions monotonic across calls, frees entries outside every declared tail window, and returns retained entries in chronological order. This initial unbounded form rejects mixed or unbounded masks and any position rule that could reach back. [#1503](https://github.com/mstar-project/mstar/pull/1503) derives ring retention for K/V frame windows; it does not provide this dense whole-read store. E1 adds that path without changing the K/V storage.
- A no-op read returns `Reached`, each field's entries packed by row (`contract.py`).

The conductor holds only the record (§3). The model object holds nothing: the calls count reaches a step as `step_metadata["episode_calls"]`, and everything else is in the arrays.

For Praxis-1 (§9) this is three arrays. `text_embeds` lives on the backbone with `cond` and `blank` labels and an `h` entry of the checkpoint's text hidden width (2048 in the published configs, bf16). A required positive deployment `max_text_tokens` bounds its `Window` read; `process_prompt` refuses when either stream's unpadded token count exceeds it. The reference tokenization path has no application-level cap (`truncation=False`), so this is an explicit serving limit. `executed` lives on the VAE encoder, with `frame: TensorType((3, tile_height, tile_width * num_cameras), dtype=uint8)` and `pose: TensorType((8 or 16,), dtype=float32)` plus pose validity. `praxis-1-droid` uses 224x832 with an 8-float model-frame pose; YAM uses 256x1248 with a 16-float rig-frame pose. A separate boundary array keeps the recent poses and history validity needed when the executed-frame ring wraps. At three context chunks `executed` has `max_len=None` and a 69-frame `Window` read. Each new 24-frame chunk makes the oldest 24 unreachable; the fourth addition reaches 93 logical frames but retains at most the latest 69 after it completes. Storage must also budget an in-flight append and staging. `EpisodeView.held["executed"]` is the resident count, not the lifetime append count.

**Does it fit `declare_state`?** `Entry` takes named tensors with a dtype and a shard, and `labels`, masks and no-op `Read` say how a step uses them. Five things are missing:

1. **A lifetime and unbounded logical log.** Arrays are per request and require integer `max_len` today (`contract.py`). `lifetime="episode"` changes who owns the rows and when they are released. `max_len=None` adds an unbounded logical append count for the restricted dense window form above; finite integer arrays keep their append limit, including the scratch meaning of zero. There is no separate retention mode.
2. **A dense storage, independent of lifetime.** No current storage holds an append-only non-K/V array with a no-op read. `PagedStorage` takes only a `KVEntry` with an `Attend` read, and `storage_of` sends every other append-only array with nonzero `max_len` there ([adapters.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/engine/state/adapters.py)). A dense storage can serve bounded request arrays as well as episode arrays. For an unbounded windowed log it needs a ring and a validated read-derived resident bound, with logical-to-physical mapping in `plan` and safe staging under invariant 7. Storage selection stays about entry and operation type, not lifetime; paged and ring K/V storages can use the same episode binding. E1 builds this dense path on [#1503](https://github.com/mstar-project/mstar/pull/1503), whose ring is K/V-specific.
3. **Binding a request to an episode across storages.** The engine registers a request's state on `NEW_REQUEST` and purges it on removal (`engine.py`). A named episode binds its request handle to the episode's rows on `NEW_REQUEST` (clearing all of them on a reset or system identity change, or selected arrays on a model key change), unbinds on `REMOVE_REQUEST` without freeing the rows, and frees them on `END_EPISODE`. A request without an episode id borrows a temporary slot and frees it on `REMOVE_REQUEST`. It receives `EpisodeView(id=None, calls=0, held={...: 0}, cold=None)` and every such request starts cold. The binding and its cleanup are common to dense, paged and ring storage. Both named and temporary slots count against admission (§6); neither can overwrite rows held by a live call. The temporary binding is keyed by request handle and included in `rid_table.py`'s purge rule.
4. **What the plan sees.** `walk_plan` gets only this request's signals (`base.py`). It gains an `episode: EpisodeView` keyword, passed only to a model that declares episode arrays, so no existing override changes. `held` comes from the workers' `WORKER_GRAPHS_DONE`, as persisted signals already do (`conductor.py`).
5. **Keys and system identity.** `ProcessPromptOutput.episode_keys` (§4) names per-array model input dependencies. A separate `(startup_weight_epoch, final_selected_adapter_digest)` identity covers the whole episode record and clears every held array when it changes. E2 adds the adapter digest to the wire and the tuple to the record; future reload work must rotate the epoch or end affected episodes.

Why this shape:

- **Where the reader is.** The arrays sit on the node that reads them, so a warm call moves no retained bytes between GPUs. A persisted tensor stays with its producer and is read over the transport on each call.
- **The engine owns the memory.** A slot pool sized at startup is something admission can count (§6); state a model allocated itself would turn a full pool into an OOM (invariant 1, heuristic 4).
- **One call interface for per-request and per-episode state.** Both use `add` and `read`; lifetime changes the owner of a row, while read masks determine which old entries remain reachable. The storage format follows the entry and operation, so an episode does not introduce a parallel K/V mechanism.

Rejected:

- **Episode-lifetime persist edges.** It is the smallest change to #1460: keep `persist_signals` on an episode record instead of the request. But the tensors would stay in the producer's tensor store, which admission doesn't count and nothing evicts under pressure (a persisted tensor is collected once the conductor unpersists it, `tensors.rs`, or at the request's hard teardown, [communication/tensors.py](https://github.com/mstar-project/mstar/blob/9668497eeaf5dd42bf99e3ce0141d8d0a9c70d76/mstar/communication/tensors.py)), their descriptors would go through the conductor every call, and a window that slides needs the model to re-emit the whole window each call, since a pass replaces a name (`conductor.py`).
- **One long-lived request with open client input**, as realtime conversations run. A request's walks are planned once at ingest (`base.py`), where an episode is an unbounded run of calls each with its own plan. A long request stays live while the robot is idle, holding what it opened (only some resources can be offloaded under pressure), and the conductor refuses one at capacity (`conductor.py`). Per-call timeouts, metrics and errors would also blur into one request.
- **Holding it in the frontend or the API server.** Retained tensors would cross into and out of the GPU workers every call, which is the client-sends-it-back design with one more hop.
- **A content-keyed cache for text embeddings**, as the reference memoizes them for the process (`pipeline.py`). It would share embeddings across robots on one task, but it is a separate feature (an encoder-output cache, with its own eviction across tenants), and an episode already reuses the embeddings for as long as its prompt stays the same, which in a robot episode is usually every call. A content cache would add reuse across episodes, or after a prompt switches away and back. §11 asks whether to build it too.

---

## 6. Memory limits

**Proposal.**

- **Per slot.** A named episode and a temporary idless request need the same capacity. A bounded array contributes its declared `max_len`; an unbounded logical log contributes a resident bound derived from all reads, position rules, maximum in-flight append and capture scratch. If the bound cannot be proved, the declaration is refused before serving. Praxis-1's proposed uint8 69-frame window is about 37 MiB for DROID or 63 MiB for YAM, plus poses, text and storage overhead; this is not the whole slot reservation. Per-deployment sizing still belongs under `resources:` (invariant 2).
- **Per replica.** `episodes.max_episodes` episode identities have capacity reserved or budgeted at startup in each storage that holds an episode array. A worker whose declared capacity cannot fit refuses to start, naming the array and the size.
- **Admission.** A new named episode or an idless call to a model with episode arrays needs a free slot on the replica the router picks. A connected named episode keeps its slot between calls, including while the robot executes its 24 rows. Pressure alone never takes that slot. If no slot is free, either kind of arriving call waits in the conductor's admission queue, in arrival order; an idless cold openpi call does not displace a robot. The idle sweeper frees an eligible slot after `episodes.idle_timeout_s`, then admission retries. A call of an episode that still has its slot never waits for slot admission. E3 must enforce the request deadline while a call waits.
- **Idle expiry.** `last_end` is set only after the call's teardown barrier completes (§8). The sweeper sends `END_EPISODE` only when no call is in flight or queued and at least `idle_timeout_s` has elapsed since `last_end`. A connected episode that has been inactive that long loses its retained state; its next call starts cold. A temporary slot belongs to its live request and cannot expire before that request is removed.
- **Temporary slot release.** An idless request frees its slot on `REMOVE_REQUEST`; no `END_EPISODE`, record, idle timer or later warm call is involved.
- **What the client sees.** An idle-expired episode's next call starts cold: `held` is empty, so `walk_plan` runs every producer again. `EpisodeView.cold` gives the model the reason, and a metric counts idle expiry and cold starts by reason. A wire notice is an open question (§11).

Relation to [admission_and_preemption.md](admission_and_preemption.md): an episode slot is an ownership key that outlives its requests. Admission must budget each underlying dense, paged or ring storage before a call runs (§3), and idle expiry returns that episode's allocations. It is not a second KV mechanism: KV pages retain their existing storage and preemption rules. A node with a rebuild policy needs every resource to support a drop (that doc's §4 table); each storage with episode rows must support that lifecycle.

The two lifetimes share the same episode identity pool. A read mask can make old
entries unreachable within an episode; idle expiry releases an episode's
entire identity and its allocations:

```mermaid
flowchart LR
    Free[Free slot] -->|named episode starts| Named[Named episode slot]
    Named -->|append; release entries no read can reach| Named
    Named -->|idle expiry or close| Free
    Free -->|idless request starts| Temp[Temporary request slot]
    Temp -->|REMOVE_REQUEST| Free
    Named -->|after idle expiry, a later call| Cold[Cold start]
    Cold -->|admit into a slot| Named
```

Why this shape:

- **One identity across storages.** The conductor admits and expires an episode as a unit, while each storage keeps its own layout. A finite resident bound is known from declarations before startup; the ring KV manager already hands out fixed sessions ([ring/manager.py](../../mstar/engine/resources/kv/ring/manager.py) lines 344-349).
- **Keep a connected robot's state.** A robot between calls is idle in the scheduler but still executing its task. Admission waits rather than making that robot's next call cold under pressure.
- **The conductor decides.** It is the only process that sees every episode's state on a replica, and its choice reaches every rank in the same order (invariant 11).

Rejected:

- **One global pool across dense and KV storages.** It would let unrelated resources trade memory, but it ties this design to preemption's classes before those storages have been measured together. §9 of the admission doc lists one pool per GPU as later work; episode ownership can use each storage's existing capacity rules meanwhile.
- **Refusing a new episode when full.** A waiting call can run after a close or idle expiry without forcing an immediate failure.
- **Pressure eviction of a connected episode.** A robot executing its 24 rows is between calls, so this would discard its context and make the next call cold. The existing `idle_timeout_s` is the only time-based release of a connected slot.
- **Moving idle episodes to the host.** Worth it if cold starts turn out expensive, and left for later (§11).

---

## 7. Cleanup

**Proposal.** An episode ends, and its slot frees, on the first of:

| Event | Who sends `END_EPISODE` | Notes |
|---|---|---|
| Connection close | the Positronic frontend, in the handler's `finally` (Python) or when `serve` returns (Rust) | a call in flight runs to its end first; the end is applied after the call's teardown |
| Positronic `end_session` | the frontend, after validating the session id and answering the end request | the reference replies with `{session_id, end_session: true}` and closes the session (`server.py` lines 199-208); E4 tests the acknowledgement and exactly one `END_EPISODE` in both frontends |
| A failed or aborted call | the conductor, at the request's teardown | its arrays may hold a partial write, as the reference assumes (`server.py` lines 272-279); the Positronic handler closes the session |
| `max_calls` reached | the conductor | the refused call fails, and the socket closes (§4) |
| Idle for `idle_timeout_s` after teardown | the conductor's periodic sweep | the record stays with `cold="idle"` until the connection closes, so the next call's internal view says why it is cold |
| A worker holding it is lost | the conductor ends every episode placed there | `cold="lost"` on the next call, if the conductor outlives the worker |
| Process restart | nothing to send | every episode is in memory and is gone; connection-scoped ids die with their sockets, so no client can name a stale one |

Leaked episodes are bounded three ways: the slot count and per-storage admission budgets cap retained memory; the idle timeout returns any slot a lost close left behind; and a record without a slot holds no tensors and lives only as long as its connection.

Why this shape:

- **Close is the normal end.** It follows the reference's connection-scoped session lifecycle.
- **A failure ends the episode rather than rolling back.** Undoing a partial write needs a copy of every array before every call; ending it costs one cold start in a case where the client has closed the socket anyway.
- **The timeout is a backstop, not the main path.** With connection-scoped ids it fires for a frontend that died without closing, or a robot paused longer than the timeout, whose next call then starts cold. The reference's hosted deployment caps a session at an hour (`modal_app.py` line 94).

---

## 8. Concurrent requests

**Proposal.**

- **One call per episode at a time.** The conductor admits an episode's call only when none of its calls is in flight. A call stays in flight after the reply until the `READS_DONE` teardown barrier completes, `REMOVE_REQUEST` goes to participants and `_drop_request` clears the request. A teardown timeout can force that barrier to finish if a reader never answers. A call that arrives meanwhile waits in the episode's queue, in arrival order, up to `max_queued_calls`; past that it is refused with a 429. A queued call is planned when it is admitted, not when it arrives, so its `walk_plan` sees what the call before it kept. The DRAIN/READS_DONE round trip may delay a call sent immediately after the reply, especially on a busy worker; a robot executing 24 rows at 15 Hz usually hides it, but a simulation or back-to-back test may not.
- **Many episodes on one worker.** Each call is an ordinary request. The micro-scheduler batches calls of different episodes as it batches any requests, through `can_batch` and `forward_batched`; each row reads and writes only its own slot, which the plan addresses per row. Nothing reads across episodes.
- **Speculation.** The async worker speculates within a request (invariant 7). Two calls of one episode are never in flight together, so nothing speculates across the boundary between them.

Why this shape:

- **Calls are ordered by their state.** Each call reads the history and appends to it. Two at once would read the same history and both append, so the second's context would miss the first's chunk. The reference serializes every call (`server.py` lines 44-45).
- **Queue, not refuse.** The chunked Positronic session handles one call at a time, so its queue is normally empty. A pipelining client, as the real-time reference (16 in flight) and the Cosmos3 example over `/generate/ws` (`examples/cosmos3_action_ws_client.py`) are, would see a refusal as a dropped observation. A bound keeps a runaway client from building a queue.

Rejected:

- **Running them concurrently on a snapshot.** Last writer wins, and which call wins depends on timing.
- **Serializing on the worker.** The worker would hold a ready request back by state only the conductor has, and every rank would have to agree on it.

---

## 9. Praxis-1 and other users

**Praxis-1 on episodes.**

- `walk_plan` runs `encode_text` only when `episode.held["text_embeds"] == 0` (a new episode, a restart, idle expiry, a weight or adapter change, or a new prompt), then the chunk walk. On a cold call the text embeddings reach the backbone as request edges, and its first step adds them to `text_embeds`. On a warm call the backbone reads the held array. E5 must make that warm graph schedulable without the absent encoder edges and test readiness at the graph boundary; an empty tensor inside a node does not satisfy a required graph input.
- The cold call leaves `executed` empty because no chunk has run yet. The first warm call appends all 21 executed frames supplied by the client, even if its earlier cold observation differed from the oldest executed frame. Later warm calls append 24 rows. The VAE encoder reads that window and encodes it into clean context as the reference does each call.
- The seed of call k is the episode's seed plus `episode_calls`, the reference's `base + k`.
- **Executed frames, not predicted video.** #1460 now leaves predicted video latents inside the denoise loop. The reference conditions on executed real frames and refuses its own predictions as context (`policy.py` lines 271-276; `robot_io.py` lines 298-317). The proposed episode array retains the executed frame window.

**Stage 1, before episode state.** A cold-only openpi bring-up mode starts every call at depth 0, matching the reference's `context_mode="stateless"` (`policy.py` lines 571-573). The reference documents that mode as debugging (`policy.py` lines 271-276); it is not production-faithful serving. Predicted video is never carried into the next call.

The parity gate compares model output and `decode_actions` relative rows with the reference's stateless mode, not response bytes. The [reference's protocol test](https://github.com/runwayml/praxis-1-oss/blob/68f4d7a254b4bf41951d2e97f44b07f6cac2bf69/tests/serve/test_serve_protocol.py#L832-L848) confirms that repeated calls stay at depth 0 and each returns 21 commands. A separate openpi wire test checks the array shape and values. The reference still advances `chunk_index` and seeds call `k` with `base + k`, so each parity call supplies that same explicit seed to M*.

The proposed openpi `RobotIO.action_horizon` is 21 in this mode, so the existing fixed-size check holds. DROID openpi `robot_state` uses the reference's default FR3 hand frame and is converted to the checkpoint's flange frame before proprio; YAM uses its rig world frame. The model's general request path accepts an explicit `pose_descriptor`, but this stage's openpi WebSocket does not expose it. DROID returns a 21x10 array and YAM a 21x20 array of denormalized relative Cartesian rows in the model frame, including clipped gripper values. These are Praxis-specific openpi schemas: the [stock openpi DROID client](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py#L778-L866) expects 10x8 joint-velocity/gripper actions and cannot connect unchanged.

**Stage 2, with episode state:** the new Positronic `/api/v1/session` route follows the reference's session behavior, including pose-composed command dictionaries and 21 rows at depth 0 versus 24 at warm depth above 0. A reset or idle expiry can make a later call cold again, returning 21 rows. M* uses protocol v2 for connection-bound session ids, while the reference's chunked deployment uses v1; E5 must advertise a client stack that the stock v2 client can build and send only actuator command fields. The E4 frontends carry the session and episode lifecycle; E5 supplies Praxis-specific pose composition and executed-frame context. This route waits for agreement on the episode contract. The cold-only openpi mode remains an explicitly separate bring-up path.

**Other users.** Candidates, not checked against their references:

- Any robot policy that conditions on more history than one observation carries. pi0.5 conditions on the current observation only and is unaffected.
- Interactive world models. Waypoint keeps a ring KV per request today (`ring/manager.py`); an episode-lifetime declaration should let the same storage keep a session's KV across requests. A concrete Waypoint session still needs its own integration check.
- Multi-turn agent and chat sessions, which today rely on content-keyed prefix reuse. Episode-lifetime K/V should use the same ownership binding rather than a new KV storage. Their admission and model integration remain later work.
- Full-duplex models already run a call as one long request (§0) and don't need this.

---

## 10. Migration and compatibility

- **Stage 1 openpi bring-up.** Before E1-E5, a cold-only Praxis adapter declares a fixed 21-row horizon and returns relative rows in the openpi `"actions"` array. It carries no episode id or history. Its tests compare repeated calls with the reference's stateless mode; this is not Positronic wire parity. The two existing openpi frontends already require a fixed horizon (`openpi.py` lines 217-225; `openpi.rs` lines 532-540), so neither needs a variable-length exception for this stage.
- **Positronic session wire.** `/api/v1/session` is a new WebSocket API in both frontends, rather than an extension of the openpi reply. It sends a connection-bound `session_id`, checks it on every request, carries reset, `end_session` acknowledgement and errors, and returns pose-composed command dictionaries (`praxis/serve/server.py`, `praxis/single_arm.py` `finish`). E4 adds the route and episode plumbing in Python and Rust together; E5 supplies the model-specific command conversion. Its protocol v2 ready frame uses versioned client-stack components and a 15 Hz chunked schedule. The reply omits the legacy `timestamp` key, which the reference's v1 client strips before giving commands to the robot.
- **Both frontends change together** (invariant 15): the episode id, `reset_context`, `END_EPISODE` on close and reply behavior land in the Python and Rust Positronic handlers in E4. A `test/rust` Positronic parity case uses a stub model reply to compare cold 21-row and warm 24-row protocol behavior across both frontends; E5 tests the real model and pose composition. The cold-only Praxis openpi mode still returns 21 rows on every call.
- **Models that declare no episode array are untouched.** The conductor knows at startup whether the model declares any. If not, it makes no record, routes as today and never passes `episode` to `walk_plan`, so the request path and its per-request cost are today's (invariant 6). The frontends' minted id is a string on the submission, dropped at the conductor.
- **Routes other than episode-capable WebSockets** pass no episode id. A model with episode arrays still runs: each such request borrows a slot, receives an empty `EpisodeView`, and releases the slot at request teardown. It waits at slot admission under the same rules as a new named episode (§6). Two idless calls cannot share state, even when they run on the same worker.

Staging, each step independently reviewable and testable. The cold-only openpi bring-up mode is separate and precedes E1-E5. The episode feature has five work stages because the engine's storage, the conductor's request routing, bounded admission, the two frontends' wire behavior, and Praxis's model wiring have different owners and regression surfaces. A stage may need more than one PR, especially E4's new API. E1 and E2 provide internal contracts; they do not expose episode serving on their own. The episode feature reaches clients only after the lifecycle and frontend work. Praxis's integration waits for those pieces, while independent model component ports can proceed:

| Step | Contents |
|---|---|
| E1 | Stack on #1503; add `lifetime="episode"`, the restricted `max_len=None` dense window, a generic request-handle-to-episode-row binding across storages, and `END_EPISODE`. Modular tests append 21 + 24 + 24 + 24 frames through a 69-frame window, check chronological reads and bounded resident memory, reject unsupported unbounded or backward-reaching reads, and free a temporary slot on `REMOVE_REQUEST` |
| E2 | The conductor's record, pinned placement, `EpisodeView` to `walk_plan`, per-episode queue, `max_calls`, final adapter digest on the wire, startup weight epoch in the conductor, whole-record identity and per-array model keys; tests show a named episode resumes warm, a changed epoch or adapter digest clears every held array even without model keys, a prompt key change clears only its array, and separate idless requests start cold |
| E3 | Shared-slot admission for named and temporary calls, idle expiry, cleanup on failure and worker loss, and metrics; pressure tests show an idless or new named call waits behind connected episodes instead of displacing them, then runs after a close or expiry, with cases on both sides of `idle_timeout_s`. A future base-weight reload must coordinate invalidation or end affected episodes |
| E4 | Both Python and Rust frontends: the new Positronic `/api/v1/session` API, including connection id, reset, close and reply behavior; `test/rust` checks a stubbed Positronic cold/warm session across both frontends |
| E5 | Praxis-1 on episodes, including pose-composed Positronic commands; an integration test checks executed-frame history and text reuse across calls, reset, idle expiry, and 21 rows at cold depth 0 versus 24 when warm |

---

## 11. Open questions

1. **Client-supplied ids.** Should a client be able to name its episode, to survive a reconnect or to drive episodes over `/generate/ws`? If so, scoped to its API key, and is the idle timeout then the only cleanup?
2. **The cold-start notice.** Should a Positronic reply report when a previously warm session expired idle, so the client can tell why its next call is cold? If so, where in the wire reply?
3. **Text embeddings across episodes.** A content-keyed encoder-output cache is separate from episode state. If built, should it include Chatterbox's voice LRU so both share one cache policy?
4. **Idle episodes to host memory** instead of dropping them, once cold-start cost is measured?
5. **Per-connection settings.** The reference takes `max_chunks_per_episode` and sampler settings per session from the URL query. Should the Positronic route accept the same settings, or is the deployment config enough?
6. **Defaults.** `max_episodes` 16 and `idle_timeout_s` 600 are guesses until measured on a robot workload.
