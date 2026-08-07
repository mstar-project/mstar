# RFC: Cross-request prefix KV reuse

- Status: draft
- Area: engine (KV_CACHE path), worker, conductor
- Related: open PR #198 (SWA + KV lifecycle) touches the same files
- Design notes: alternatives and the questions behind each decision, posted as the first comment on this issue

## Summary

Keep the KV pages of a prompt's prefix after the request finishes, and let the next request with the same prefix use them instead of recomputing. Pages become reference-counted: freed when the count hits zero, not when the request ends.

Reuse is exact only; approximate reuse is kept out of scope. A hit is confirmed by comparing the stored token ids, and outputs stay bit-identical under the gate below. Images and audio work from phase 2 - the media content is hashed into the key, so the same image behind the same prefix hits like text. Caching encoder outputs themselves comes in phase 3.

## Motivation

M* has no cross-request reuse of any kind. KV pages are allocated per request, per node, per label, and freed at teardown (`PagedAllocationManager.remove_request`, `mstar/engine/kv_store.py:677`).

vLLM ([automatic prefix caching](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)), SGLang ([RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)), and TensorRT-LLM ([KV cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)) all have prefix caching; M* does not. The gap shows up wherever prompts repeat: multi-turn re-pays prefill for the whole history every turn, and multi-tenant deployments re-pay the shared system prompt on every request. Trace measurement on 500 public mini-swe-agent sessions (25,494 requests, 401.8M prefill tokens) puts redundant prefill at 97.54% at page size 128, which is M*'s page size, and 91.35% on τ-bench retail ([kvshare](https://github.com/vasilevklart/kvshare)).

## What already exists

The design changes existing code paths rather than adding subsystems.

| Seam | Location | Used for |
|---|---|---|
| Page free list, no per-page metadata | `kv_store.py:25` | the one primitive is added here |
| Per-request state: `page_indices`, `seq_len`, `position_id_start` | `kv_store.py:156` | a hit seeds these three fields |
| Single free path | `kv_store.py:677`, from `kv_cache_engine.py:1258` | retention intercepts here |
| Allocator lock covering plan thread vs GPU thread | `kv_store.py:474` | lookup and eviction run under it |
| Allocation shortfall path | `kv_store.py:534` → `worker.py:1252`, `:789`, `:832` | eviction inserts ahead of request offload |
| Replica choice among data-parallel ranks | `conductor.py:497` | affinity replaces a random draw |
| Reference counting precedent | `conductor.py:578`, `communication/tensors.py:42` | same pattern, applied to pages |

## Design

### The primitive: counted page ownership

Today every page has exactly one owner, recorded nowhere: the page id appears in one request's `page_indices`, and teardown returns it to the free list. Sharing a page under that rule is unsafe, because the second reader is invisible to the first owner's teardown.

Add one integer per page and change the free rule: a page returns to the free list when its owner count reaches zero. An owner is any scope with a defined end - a live request, the prefix index. Today's behavior is the case where the count is always one, so nothing changes until a second owner exists.

One constraint follows from how decode writes. Appends go in place into the last page until it fills (`advance_seq_len`, `cache_manager.py:539`), so a partially filled page under two owners would have one owner mutating bytes the other reads. Only full pages can have more than one owner. The tail page is always exclusive.

### Cross-request prefix reuse

#### Names

A hit substitutes a stored page's data for a fresh computation, so the key must name everything that data depends on. The dependence is wider than the page's own 128 tokens: K and V at a position are projections of that position's hidden state, and the hidden state is built by attending over every position before it. Change any earlier token and the stored values change. So a page depends on the whole prefix from position 0, plus the weights, the tensor-parallel slicing (each rank holds its own heads; caches are not shared across TP groups, `kv_cache_engine.py:406`), the pool dtype, and the attention backend. Position is baked in too: K is rotated before it is written (`plan_rope`, `cache_manager.py:346`).

Constants of the process collapse into a fingerprint hashed once at startup, which forms the root of a chain:

```
root  = H(model + weights rev, tp geometry, dtype, backend, tokenizer rev, salt)
key_i = H(key_{i-1}, tokens_i [, media digests in span])
```

H is xxHash at 64 bits; nothing stronger is needed, because every hit is verified by token comparison below.

Chaining makes each page key a commitment to the whole prefix, so a match at page *i* implies the prefix matched. Rooting every chain at position 0 pins positions, which is what makes rotated bytes valid on reuse. Redeploying anything in the fingerprint changes the root, so old keys become unmatchable and the cache empties itself with no flush path to write.

Multimodal prompts need one more field. Placeholder token ids are identical across images while the embeddings substituted at those positions are not, so a digest `H(raw bytes ‖ preprocessor params)` folds into the key of the page holding the item's first placeholder and propagates forward by chaining. Both vLLM and SGLang shipped this after hitting the collision ([sglang-omni #260](https://github.com/sgl-project/sglang-omni/pull/260)).

Sampling parameters are excluded: they act after the logits, so prefill K/V does not depend on them. If M* grows per-request adapters, adapter identity joins the chain the way media digests do; today there are none. A per-request salt, empty by default, keeps entries from being shared between tenants who should not probe each other's timing. Within one salt domain the timing signal remains; deployments needing isolation give each tenant a distinct salt.

Node, label, and rank are not hashed. Pools are already per node, per label, per rank, so those dimensions select which index is consulted and cost nothing.

#### Lookup

A lookup needs the tokens, the index, and the authority to mutate request state under the allocator lock. Only one place has all three: the engine, at the first prefill allocation for a `(request, node, label)`. The conductor has tokens but not the index. The micro-scheduler never sees tokens; it schedules `(node, request, walk)` tuples (`worker/micro_scheduler.py:212`).

The path is one pass: walk the chain until the first miss, compare stored token ids to confirm each nominee, increment owner counts, then write `page_indices`, `seq_len = 128·k` for k matched pages, and `position_id_start = 0` - positions are absolute and RoPE is already baked into stored K, so a hit shifts nothing. Existing allocation extends the tail; prefill computes only the remaining tokens. Because matching is page aligned, the write frontier opens a fresh page, so no copy-on-write is needed. Name computation runs outside the lock, in the data worker that already computes admission keys; the lock holds only probe, compare, increment.

Comparing token ids on a hit means the hash only nominates a candidate, so a collision costs one wasted comparison, not a wrong answer. That is what lets H stay cheap: vLLM, which trusts its hash with no comparison, defaults to SHA-256 and warns that its fast-hash option can leak private information between tenants ([design doc](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md?plain=1#L25)).

After seeding, the request's state has the same shape as one that prefilled normally, which is also the shape used for decode with prior KV and for disaggregated read-in. Decode, CUDA graph replay, and speculation need no awareness of the hit. Rollback is ordinary teardown: removes are already deferred while a batch or its speculation is in flight (`worker.py:288`, `:1963`), and teardown decrements the counts seeding took.

Under tensor parallelism, admission runs independently per rank (`_handle_allocation_failure` docstring, `worker.py:1252`), and followers must run what rank 0 committed to. Matched length must therefore be identical across the group. Two mechanisms work: deterministic lockstep indexes, or a minimum across the group carried on the existing rank-0 fan-out. The agreement ships in phase 4; phases 1-3 run at `tp_size = 1`. TP deployments are unaffected meanwhile: they run exactly as today, without cross-request hits, so nothing regresses. Sequence parallelism adds nothing further: under Ulysses each rank caches the full sequence for its own head slice, heads divided by tp×sp (`kv_cache_engine.py:220`), so an SP rank looks identical to a TP rank here and one agreed length covers the whole instance.

#### Storing

Pages enter the index at teardown, where ownership already changes hands. Every full page of the request's final sequence transfers to the index instead of being freed - generated pages included, one rule, no provenance filter. Prompt-page keys were computed at admission and are on the state; keys for generated pages extend the same chain over the recorded output ids at teardown, off the hot path, and those ids are stored regardless for verification. Generated pages matter in agent traffic: one turn's output is a prefix of the next turn's prompt, and the cited traces measure what excluding them costs.

#### Release

Without the index, every page returns to the free list at request end. With it, cached pages accumulate until the pool is full, and the first failure is a live request that cannot allocate. Eviction restores the old invariant: a live request must never fail to allocate because of cached data.

The pool is a fixed tensor allocated once at `max_num_pages` (`kv_cache_engine.py:231`), so cached pages occupy prepaid headroom and never grow memory. Caching is free until the pool is full, and the first eviction happens when a live request needs a page. No watermark, no background reclaimer, no reserve.

On shortfall, before the exception escapes and under the same lock: take what is free, then release cold tails - from the least-recently-hit run, drop tail pages down to the first page another owner still holds, as one batch, repeating with the next run until the shortfall is covered - and only then fall through to today's behavior of offloading or holding a request (`worker.py:789`, `:832`). Batching by cold tail keeps saturation from degrading to one eviction per allocation, with no absolute threshold introduced. The ordering follows from cost: dropping a cached page costs a possible future recompute, no copy, and cannot fail; offloading a live request costs certain latency and two PCIe transfers.

Tail first within a run is a garbage invariant, not correctness: a mid-run eviction cannot corrupt anything - a lookup stops at the first missing link - it only strands the pages behind the hole. LRU yields tail-first for free: any match that touches page i+1 also touched page i, so recency is monotone along a chain and a run's least-recent page is always its tail.

Offload composes with counts by one rule: suspension is not teardown. A request being offloaded moves only pages it exclusively owns; a shared page is decremented, stays resident, and is exactly the page worth keeping on GPU.

Pool size is the one measured constraint: on the same traces, collected redundancy rises from 1.99% at a 64k-token pool to 61.8% at 128k to 97.9% at 256k, so the pool must exceed concurrency times final context, completions included. M*'s default of 2048 pages at 128 tokens sits at that scale, shared with live requests.

### Replica choice

When a `node_group` lists more ranks than one instance needs, those ranks are data-parallel replicas, and the conductor picks one at random per request (`np.random.randint`, `conductor.py:497`). With R replicas, a prefix cached on one is consulted with probability 1/R, and R copies of the same entry consume R times the pages for the same benefit.

Replacing the random draw with a deterministic function of the request's prefix makes two requests with the same prefix agree on a replica without communicating. The router holds no cache state, so nothing it holds can go stale; the engine still decides on arrival, and a wrong guess costs a miss. The `_group_id` coordination and whole-instance lockstep are untouched: this changes which replica, never how many ranks run together. A hot prefix skews load, so the target is preferred only until its queue passes a threshold, after which the behavior is today's.

The source already anticipates this seam: "TODO: smarter assignment that minimizes cross-graph-walk tensor transfer" (`conductor.py:483`).

Current configs ship no replication, so this is inactive until someone scales out.

#### Windowed labels (PR #198)

PR #198's windowed generation (`protect_prefix`, `release_oldest`, `prefix_epoch`) composes with reuse under two rules. Storing: a windowed label transfers only its protected-prefix run at teardown; pages beyond the first released token are chain-orphans - a lookup walks from the root and stops at the gap - so they are freed as today. Releasing: `release_oldest` goes through the same counted free as teardown, so a shared page is decremented, not freed, and a seeded prefix cannot be pulled out from under the index. The protected prefix is the model's stable conditioning, which is exactly the part that repeats, so what windowing preserves and what reuse wants are the same pages. Windowing is per label, and pools are already per node, label, and rank, so counts stay per page; no per-layer residency arises.

#### Cost

On a miss, teardown extends the chain over the output ids: one hash per generated page, off the request's critical path. Metadata is the verifying ids themselves, about 1MB per node at 2048 pages, ~0.005% of the pool they describe. On a hit, in-lock work is probe, one 128-int compare, and an increment per matched page - microseconds even for a thousand-page match, once per admission. Eviction is batched by cold tail, priced above.

## Correctness

Reuse is exact. A hit means the same tokens at the same absolute positions under the same fingerprint, verified by token comparison before any page is used. The mechanism is testable by running with and without and comparing bits.

The gate needs one qualification. Continuous batching makes reductions outside attention sensitive to batch composition, so a cached run and an uncached run can differ if their batch schedules differ. The test compares a cached run against an uncached run with the same schedule. The same gate scopes dtypes: if M* grows quantized KV, reuse for those dtypes stays off until the comparison passes.

Labels stay physically separate. A `["main"]` page is never shared with `["main","cfg_img"]`. Sharing pages across a label fork instead of copying them (`snapshot_all`, `cache_manager.py:632`) becomes possible once counts exist, but is not proposed here.

## Phases

**1. Names without pages.** Compute page names at admission, keep a table of names and their verifying tokens, log how many pages each request would have matched. No page is retained, nothing is reused, no lifetime changes, so it cannot affect an output. This measures realized reuse against the measured ceiling, tells us whether multimodal traffic resembles text agent traffic, and reports duplicate-media rate and replica spread as a side effect. It records, per miss, whether an in-flight request already held a matching prefix, which prices teardown indexing against eager sharing.

*No gate on phase 2:* parity and the cited traces already justify it, and M* has no production traffic to measure against. Phase 1's numbers instead set pool sizing and the phase 3 triggers, taken by replaying the Benchmarking workloads.

**2. Keep the pages.** Owner counts, teardown transfer, seeding, release. These cannot be separated: retaining without releasing is a leak, reusing without counting is a correctness bug. Ships at `tp_size = 1`, single worker - which still covers the single-rank nodes composite walks run (text encoders, VAE) even in TP'd deployments.

**3. Wider reach.** Two independent follow-ups on the same keys, in any order. (a) Reuse encoder outputs, vision towers and text encoders alike (umT5 on the diffusion path), across requests under the same digest, skipping the encoder rather than the prefill; triggered when phase 1's duplicate rate shows the same media or prompt actually recurring. The CFG negative branch makes the trigger structural for generation: the same negative tokens ride nearly every image and video request, so diffusion traffic repeats without any user repeating a prompt. sglang-omni's [#258](https://github.com/sgl-project/sglang-omni/issues/258), encoder state leaking across sequential requests, is the accidental version of this reuse; the digest makes it deliberate and keyed. (b) Replica affinity when a deployment runs copies of a node; triggered when phase 1's replica spread shows repeat prefixes landing on different copies; acceptance is that tail time-to-first-output under a hot-prefix skew does not regress against random routing.

**4. Standard extensions.** Both exist in mainline engines; both land once phase 2 holds. (a) TP/SP agreement on matched length: rank 0 announces its match on the existing fan-out, a follower missing a page replies shorter, the minimum wins; the cost is multi-GPU testing. (b) A CPU tier for evicted cached pages, behind the readiness gate disaggregated read-in already uses (`read_in_progress`), with an explicit split of `cpu_offload_pages` between the tier and preemption; SGLang's [HiCache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/) is the reference shape.

## Benchmarking

Text: vLLM's prefix-caching harness ([benchmark_prefix_caching.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_prefix_caching.py)) and SGLang's [Mooncake trace replay](https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/datasets/mooncake.py), plus the kvshare sessions themselves so realized reuse is read against the same traces as the cited ceiling; cache on and off, same model and GPU, against vLLM and SGLang with their caches on. Metrics: time to first token, inter-token latency, throughput.

Multimodal has no settled suite. The omni forks each cover a slice - vLLM-Omni a [diffusion serving benchmark](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/diffusion/diffusion_benchmark_serving.py) and [per-modality SLO metrics](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/metrics.md), SGLang-Omni a [time-to-first-audio benchmark](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/eval/benchmark_omni_streaming_ttft.py) - and neither has repeat-structure workloads or cache metrics.

For M*, one workload per modality: VLM chat (SGLang's [image and MMMU datasets](https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/serving.py#L1632)), image generation (DiffusionDB prompts), video (cosmos3 and Wan2.2), audio ([SeedTTS](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/eval/benchmark_omni_seedtts.py) prompts on Qwen3-Omni, plus voice-session replay and one-reference cloning for repeat structure). Metrics, each tied to its claim: first token and first audio (what hits move), goodput at fixed SLO (the same win under load), per-image and per-video latency on the stateless models (the overhead guard; SGLang measured none and defaults its cache on, [ablation](https://www.lmsys.org/blog/2024-01-17-sglang/)), and phase 1's three - matched-page share, hit-versus-miss latency, zero live-request stalls. The text track carries phases 1-2; the multimodal track feeds phase 3's triggers and can wait (open question 6).

## Non-goals

- Reusing a prefix at positions other than the ones it was written at. K is rotated in place before it is stored, so this needs re-rotation and is not bit exact.
- Approximate reuse across denoising steps. It changes outputs, needs a per-model tolerance and quality metrics rather than an equality test, and mixing it with exact reuse makes a quality regression impossible to attribute. It belongs in its own RFC and nothing here blocks it.
- Fetching cached pages from another worker. The transfer engines exist for disaggregation (`kv_store.py:264`, `:354`), but a cross-worker cache needs a directory and an async retrieve on the hit path.
- Reordering batches to chase matches. It imports token-awareness into a deliberately token-blind scheduler; a separate proposal if ever wanted.

## Open questions

1. Which replayed workloads count as representative for phase 1's numbers, absent production traffic.
2. Which TP agreement mechanism to use: lockstep indexes or a group minimum.
3. Sizing guidance: `max_num_pages` must exceed concurrency times final context, but the split between live and cached pages under load is unmeasured.
4. Windowed labels seeded from a hit: whether `protect_prefix` always covers the matched length once PR #198 lands, or the decrement rule alone carries it.
5. Whether identical prompts concurrently in flight are worth sharing. Teardown indexing forfeits in-burst reuse; phase 2 measures the forfeit before any machinery is added.
6. Whether the multimodal suite is worth building before phase 3: its cost is mostly the two missing workloads, and its numbers only feed phase 3's triggers.
