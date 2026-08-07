# Design notes: KV reuse in M*

Companion to the prefix-caching RFC: the alternatives considered, why not, and what changed along the way.

A caching system is five questions: who owns a cached value and when is it freed; how a value is named so two requests can agree they computed the same thing; where the lookup runs; what to evict under memory pressure; and what a router can know.

## Comparison to existing systems

| Mechanism | vLLM | SGLang | TensorRT-LLM | This design |
|---|---|---|---|---|
| Key | chain hash over full blocks | token path in a radix tree | token content in a radix tree | chain hash over full pages |
| Granularity | fixed blocks, full only | token level, page aligned when paged | fixed blocks, partial reuse | fixed pages, full only |
| Trust | SHA-256 digest equality; token ids in the preimage only | exact token comparison on edges | hash equality | hash nominates, stored token ids confirm |
| Lookup site | scheduler at admission | scheduler at batch formation | runtime at request add | engine at first allocation |
| Index scope | one namespace per model | one tree per engine | one tree | one per (node, label, rank) |
| Eviction | free-list LRU | leaf LRU over the tree | priority + LRU, leaf only | LRU run, tail first within run |
| Scheduler coupling | none | queue reordering by matched length | prefix-aware admission | none |

The cross-request mechanism is vLLM's, adapted. Three differences are forced by M*: the lookup site (the micro-scheduler never sees tokens, `worker/micro_scheduler.py:212`), the index scope (a model declares several KV nodes, and labels partition pools), and eviction ordering (M* already offloads live requests under memory pressure, so cached pages must yield before that path runs).

Two differences are choices: an explicit token comparison on a hit, where vLLM relies on a cryptographic digest, and no radix tree.

## Why not a radix tree

The tree's one advantage over a hash map is token-granular partial matching: a request diverging at token 300 reuses tokens 0-299. That advantage does not survive M*'s append mechanics. Decode writes in place into the last page, so a shared partial page would have one owner mutating bytes another reads; only full pages can be shared. At page size 128, a divergence at token 300 lets both structures share pages 0 and 1 and no more. The tree identifies tokens it is then forbidden to share. SGLang concedes the same rule when paged: with page size above 1 it aligns and drops the partial tail.

With hit capability equal, what remains is the tree's costs:

- Structural mutation under the allocator lock. Node splitting, recursive leaf promotion, and lock-reference walks over ancestors would run under the re-entrant lock that already arbitrates the plan thread against the GPU thread (`kv_store.py:474`). A hash map does point inserts and deletes.
- A forest, not a tree. Scoping is per node, label, and rank, so each pool needs its own tree, eviction heap, and reference discipline. With a flat map the scope dimensions are index selection and every page sits under the one counter.
- Leaf-only eviction. A cold run's tail cannot be dropped without walking structure. In a flat map it is a row deletion and a decrement.
- The payoff has no consumer. SGLang's tree pays off in the scheduler, which reorders the queue by matched prefix length, and in the tiering built on the same tree ([HiCache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)). M*'s scheduler never sees tokens.

The prefix guarantee carries over without the tree: chaining makes a hit at page *i* imply the whole prefix matched, the invariant a tree path encodes in pointers.

The bugs in the wild point at keys, not trees. sglang-omni hit collisions between different media behind identical placeholder tokens in its radix cache, fixed by content-hashing media into the token stream ([#260](https://github.com/sgl-project/sglang-omni/pull/260)); vLLM's hash cache shipped its own collision fix. Both ecosystems converged on content-hashed media, which is the digest field in the key here. The tree wins in the system SGLang is: token-granular pages, one KV namespace, a token-aware scheduler. M* is none of the three.

## Decision record

### Page ownership

| Option | Why not |
|---|---|
| Copy reusable KV into a separate arena at teardown | Bandwidth per store and per hit, a second pool for attention to plan over, and double peak memory during the copy. Only necessary if pages cannot outlive requests, and they can |
| Free whole generations at once | Cannot express "a live reader holds this page"; a generation flip during a read is either corruption or a global barrier |
| Recompute reachability periodically | Scans on the allocation path under the same lock; nondeterministic free timing conflicts with the allocator's exact shortage accounting (`pages_short`) |
| One count per cached run rather than per page | Prevents partial eviction of a run's cold tail. The saving is 2048 integers per node |

### Trust

| Option | Why not |
|---|---|
| Trust a cryptographic hash (vLLM today, TensorRT-LLM) | Sound, but forces the expensive hash and rests correctness on collision resistance. vLLM has been on both sides: a live collision under the builtin hash led to token ids in the lookup key, later traded back for a SHA-256 digest with the ids only in the preimage |
| Compare stored token ids on a hit | Chosen. About 512 bytes per page, roughly 1 MB per node at the default pool size. Makes the result exact, demotes the hash to a nominator so a fast hash suffices, and requires digests at full width, never truncated |

The salt stays either way: it addresses timing probes between tenants, not collisions.

### Key fields

One asymmetry decides the defaults: a dependency left out of the key silently corrupts, an extra field only costs hits. Include when in doubt; remove only with an argument.

| Field | In or out | Reason |
|---|---|---|
| Whole prefix, via chaining | in | Hidden states mix causally, so a page's bytes depend on every earlier token |
| Absolute position | in, implicitly | K is rotated before storage; rooting every chain at position 0 pins positions without a field |
| Weights, TP geometry, dtype, backend, tokenizer revision | in, via the startup fingerprint | Constant per process. Hashing them once puts them in every key through the chain, and makes redeployment empty the cache by making old keys unreachable |
| Media digest | in | Placeholder token ids are identical across different images while the substituted embeddings are not. vLLM and SGLang both shipped this after hitting the collision |
| Salt | in, empty by default | Isolation policy, not correctness. Within one salt domain the timing signal remains |
| Sampling parameters | out | They act after the logits; prefill K/V does not depend on them |
| Adapter identity | in, per request, if M* grows adapters | Different effective weights. Joins the chain like media digests; none exist today |
| Graph walk | out | Nodes shared by name across walks share one engine and one cache, so keying by walk would split one physical cache's hits |
| Node, label, rank | out of the hash, in as index selection | Pools are already partitioned this way, so the scope costs nothing |

The failure mode is [documented](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html): TensorRT-LLM requires a per-token extra id under prompt tuning because different requests otherwise present identical fake input ids, and their tracker carries a report of reuse producing wrong outputs under FP8.

### Lookup site

Rejected sites for the lookup; the RFC's Lookup states the chosen one.

| Site | Missing |
|---|---|
| API data worker | index, authority. Keeps its role: it computes the names where media already loads |
| Conductor | index, authority. A remote answer is stale on arrival because eviction is local and continuous |
| Micro-scheduler | tokens. It schedules `(node, request, walk)` tuples |
| Engine, first prefill allocation | nothing |

### Storing

Eager storing during prefill would put index mutations on the hot path to serve the case of identical prompts racing within one request's lifetime. Teardown transfer serves the common case at no cost, at the price that a prefix becomes reusable only after its first request finishes. All full pages of the final sequence are indexed, generated included: the cited traces price exclusion at 1.2 to 3.0 points of hit rate, and in agent sessions one turn's output is a prefix of the next turn's prompt.

### Eviction

| Option | Why not |
|---|---|
| Background reclaimer at a watermark | Needs a threshold nobody can pick, and evicts while the pool is idle |
| Reserve caps per pool | A budget inside a fixed budget. The yield ordering already guarantees live requests win |
| Runtime arbiter balancing across pools | Incoherent: pools are separate fixed tensors allocated once at `max_num_pages` (`kv_cache_engine.py:231`), so pages cannot move between them. Balance is a startup sizing decision |
| Offload cached pages to CPU rather than dropping | Competes with request preemption for the same `cpu_offload_pages` budget and puts an asynchronous retrieve on the hit path |
| Priority and duration retention, as TensorRT-LLM offers | A tuning surface aimed at hit rate. Earns its cost only after measurement shows LRU losing |
| Evict whole runs | Discards a hot shared head to reclaim a cold tail |

Tail-first within a run is a correctness constraint. A lookup walks the chain and stops at the first missing link, so evicting a middle page makes every page after it unreachable while it still occupies the pool. LRU produces this ordering anyway, since every long match also touches the earlier pages.

### Replica choice

| Option | Why not |
|---|---|
| Global directory of which replica holds what | Replicas evict independently and continuously, so directory entries decay. Pays a lookup or a round trip per request for something agreement provides free |
| Query replicas at admission | Round trip on the critical path, and the answer can be stale by dispatch |
| Move pages to the chosen replica | Cross-rank transfer to serve a prefix that could be recomputed. The transfer engines exist for disaggregation, where the data has no other source |
| Route by session id | Covers multi-turn chat only, and is blind to prefixes shared across users, which is the dominant case |
| Keep the random draw | Divides hit rate by the replica count and holds duplicate copies of the same entry |

Deterministic routing on the prefix holds no cache state, so nothing it holds can go stale. Its failure mode is a hot prefix skewing load, handled by preferring the target until its queue passes a threshold.

## Does this cover images and audio

Covered from day one, for the exact case: media content enters the key as a digest of raw bytes and preprocessing parameters, so when the same image recurs behind the same prefix, its placeholder tokens' KV pages hit like any text pages. What phase 2 does not do is reuse the encoder's output itself - skipping the vision tower rather than the language prefill - which is phase 3 under the same digest.

## Why nothing approximate

Exact and approximate are different kinds of object; the boundary here is the one a test can check. Exact reuse is invisible: outputs are bit-identical, so it needs no consent, no quality budget, no per-model tuning, and the equality gate the repo already runs validates it. Approximate reuse changes outputs, so someone owns a tolerance, tunes it per model, and judges it with quality metrics instead of equality. Ship both in one layer and a quality regression has two suspects, the cache and the budget, with no way to separate them. I went through the approximate diffusion side at length separately - TeaCache-style step caching, which reuses a step's residual when the modulated input barely moved, and the cache-dit line of block-level caching - and pinned it: it stays a separate opt-in lane, with the native-versus-vendor call gated on a cache-dit source read. Nothing in this design blocks it.

## Can KV cross modalities

Four cases. A text prefix shared between a text request and an image request: yes, keys match until content diverges. The same image in two requests: yes, that is what the digest is for, and the encoder output itself becomes cacheable under the same digest. Equivalent content in different modalities, a picture of a dog against audio saying "dog": no. These are different-length sequences in separately trained spaces with no positional correspondence, so the mismatch is structural, not a matter of precision. Across different KV nodes: no, different weights.

One adjacent case is currently missed. A byte digest fails when the same image arrives re-encoded or resized. A perceptual hash could nominate candidates with an exact comparison of the preprocessed tensor confirming, which is the same nominate-and-confirm split used for tokens. Left for the phase 3 PR.

## Corrections

| Earlier | Evidence | Now |
|---|---|---|
| Longest-prefix matching in the micro-scheduler | It schedules `(node, request, walk)` tuples and never sees tokens (`worker/micro_scheduler.py:212`) | Lookup is engine-side at first allocation |
| Token-granular radix with node splitting | Decode appends in place into the last page, so only full pages can be shared | Page granularity, which also removes the tree's advantage |
| A runtime arbiter balancing cached pages across pools | Pools are separate fixed tensors allocated once | Balance is `max_num_pages` at startup; eviction is intra-pool |
| Key scoped by graph walk | Nodes shared by name across walks share one engine and one KV cache | Walk dropped from the key |
| A trace study gates all cross-request work | Trace measurement already exists for text agent traffic and reports 97.54% redundant prefill at page size 128 | The measurement phase now targets realized reuse against that ceiling, and whether multimodal traffic behaves the same |
| Reserve caps and a low-water mark for the cache | The pool is prepaid and fixed, so caching costs nothing until it is full | No reserve, no watermark; release happens on shortfall |
| Generated pages excluded from the index as rare hits | The cited artifact's own ablation prices retention at 1.2 to 3.0 points, and in agent sessions one turn's output is a prefix of the next turn's prompt | Every full page of the final sequence is indexed; the exclusion was a provenance filter with no correctness basis |
| vLLM cited as the trust-the-hash counterexample | Its lookup key on current main is a SHA-256 digest with token ids in the preimage; it had earlier carried ids in the key after a live collision, then traded the comparison for the cryptographic hash | Characterized precisely: shared concern, different remedy |

## Deferred

| Item | Reason |
|---|---|
| Cross-worker fetch of cached pages | Needs a directory and an asynchronous retrieve on the hit path. The transfer engines exist, but for disaggregation |
| Sharing pages across a label fork instead of copying (`snapshot_all`, `cache_manager.py:632`) | Becomes possible once counts exist. Correctness under guidance needs its own argument |
| Position-independent reuse | K is rotated before storage; re-rotation is not bit exact and degrades at large positions in bf16 |
| Cache-aware batch ordering | Requires the scheduler to know matched lengths, which requires the lookup to move |
| Perceptual media hashing | Speculative until duplicate-media rate is measured |
| In-flight sharing of identical concurrent prefixes | Teardown indexing forfeits in-burst reuse; phase 2 measures the forfeit before machinery is added |
| A determinism mode pairing the cache with batch-invariant kernels | Orthogonal to reuse; relevant only to users needing bit-stable outputs across batch compositions |
| Loop-invariant value reuse inside `Loop` | Parked: I want an evidence pass across the shipped models first - which values, their cost, iteration counts - before proposing the mechanism |
| Approximate diffusion caching (TeaCache, cache-dit) | Pinned as a separate opt-in lane; native reimplementation versus vendoring is gated on a cache-dit source read |
