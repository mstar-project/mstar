# Design notes

Companion to RFC #210.

## Why not a radix tree

### Radix tree

SGLang stores cached prefixes in a tree. Each edge is a run of tokens, so two prompts that share a prefix share a path and split where they diverge. Each node carries a reference count; pages are freed only from leaves whose count is zero.

The tree exists so that a lookup can find the longest cached prefix at any token boundary.

### Page granularity

The tree's advantage is matching at token granularity, and it depends on SGLang storing one token per page: a prompt that diverges from a cached one at token 300 gets tokens 0-299 back.

At page size 128 that advantage cannot exist, whatever the index. A page can be shared only if it is full, because decode writes into the last page in place; a half-shared page would have one request mutating bytes another reads. So the same prompt shares pages 0 and 1 (tokens 0-255) and nothing else. SGLang's own code does this: its prefix match truncates the key to a page multiple whenever the page size is above 1. The tree finds tokens it is then forbidden to share; a hash map keyed by page finds the same pages.

### Cost of page size 1

Page size 1 would make the tree's advantage real, at a price that has nothing to do with caching:

- The page index a request carries grows 128×, and it is rebuilt and shipped to the attention kernel every step.
- The pool stores a page as a contiguous slab, so kernels lose coalescing at 1.
- Disaggregation, offload, and any CPU tier move pages; one-token pages make every transfer latency-bound.
- An image is thousands of tokens that hit or miss as one unit under one content digest. There is no divergence inside an image to split on, so granularity buys nothing for media.

SGLang itself defaults to 1 only on CUDA and sets 64 wherever the kernel layout demands it - ROCm, musa, and its own HiCache benchmarks.

### Cache-aware scheduling

The tree's real payoff is the scheduler. SGLang's paper (Theorem 3.1) shows that ordering the waiting queue by longest matched prefix is hit-optimal, and its scheduler matches every waiting request against the tree to sort them. That needs the scheduler to see tokens.

M*'s scheduler does not: a KV step carries a request id, a label, and a span, and node scheduling is round-robin. SGLang itself turns this ordering off above 128 queued requests because the matching is expensive. vLLM never had it in the engine - its scheduler is first-come or priority - and does cache-awareness at the router, which is what M*'s conductor is.

### Eviction of shared prefixes

A shared prefix cannot be evicted from under live requests in either design: a page with a nonzero count is not evictable, and SGLang's eviction only touches leaves with a zero count.

Leaf-first ordering falls out of the chain without a tree. Any match that touches page i+1 also touched page i, so recency is monotone along a chain, and plain LRU releases tails before trunks. Counts and recency protect the trunk; pointers do not.

### Forks

Children forking off one parent share a prefix. In the tree they split a node; in the chain they hash to the same keys and raise the count. Same pages either way.

The real problem with forks is timing: they arrive together, before the parent's pages are indexed. SGLang indexes at every chunked-prefill boundary and keeps a second tree over the waiting queue so that only one of several identical requests runs first; vLLM indexes on block fill. The RFC's fill-time indexing is the same fix. A subagent whose context is a slice of the parent's rather than a prefix is reuse at a different position, which RoPE forbids for everyone.

### Trade-offs of a hash map

- Sub-page hits. vLLM added them with a partial entry plus a copy-on-write redirect to a private block. It works, costs a copy per partial hit, and is the one capability declined here.
- Subtree operations. Without child pointers there is no cheap "how much does this session hold" or "evict this whole subtree". Nothing in matching or eviction needs them; an index by session id would serve if policy ever does.

### Hashing in SGLang

SGLang's router ships a `prefix_hash` policy beside the tree policy: a hash of the first N tokens onto a consistent-hash ring, with a load fallback. That is the RFC's replica-affinity follow-up. Its storage tier keys pages by token hash, not by tree position. The tree lives on the GPU index of one system that chose one-token pages and a token-aware scheduler; M* has neither, and everything below the GPU index is hashed anyway.

### Other page sizes

At any page size above 1 the argument is unchanged: sharing is whole-page, SGLang's match truncates to page multiples, and only the waste per divergence shrinks, equally for both. At exactly 1 the granularity argument disappears, but the tree still pays structural mutation for a scheduler payoff M* has no consumer for, and page size 1 is ruled out on kernel and transfer grounds regardless of caching. A smaller default changes hit rates by a few points and the decision not at all.

## Decision record

| Decision | Alternative | Why |
|---|---|---|
| SHA-256, no token compare on a hit | Fast hash plus comparing stored token ids | The compare protects only the fields it compares: identical placeholder tokens with different images pass it. It costs a memcmp under the allocator lock on every hit. SHA-256 makes accidental collisions a non-event (about 2⁻²¹⁷ at a million pages) and is vLLM's default for the same reason. |
| Canonical byte encoding for the preimage | pickle, CBOR | pickle output drifts across Python and library versions, so keys would silently stop matching across processes. Fixed-width little-endian ints with length-prefixed fields is a few lines; canonical CBOR is acceptable if reviewers prefer a standard. |
| Index a page when it fills | Index at teardown | Teardown misses every simultaneous fork and every request that arrives mid-prefill. Full pages are immutable, so early indexing is safe under counts. |
| Label is part of the key | Separate index per label | Labels share one arena after the refactor; a `["main"]` page and a `["main","cfg_img"]` page with the same tokens must not collide. |
| TP by group minimum of matched length | Assuming per-rank indexes stay identical | Symmetry is checked only for free-page count at warmup; async offload and transfer can diverge residency at runtime. Each rank probes locally, the group takes the minimum, every rank pins that many. A miss on one rank shortens the common prefix; it can never make ranks compute different tokens. |
| Release cold tails on shortfall, inside allocation | Watermark with a background reclaimer | A watermark needs a number nobody can pick and evicts while the pool is idle. The pool is prepaid, so early release buys nothing; releasing when a live request is short is the only moment it is needed. |
| Release backward while the predecessor is index-only and no newer than the tail | Stop on ownership alone | After A→B→D is hit and finishes, A and B are back to one owner while D is hot; a walk from cold C that stopped on owners would strand D. Recency is monotone along a chain, so it is the right stop condition; ownership is not. |
| Per-request salt for tenant isolation | Hash strength | Isolation is a namespace question, not a collision question. SGLang's cache salt and extra key are the same mechanism. |
| Cross-worker fetch is a non-goal | A directory plus async retrieve on the hit path | The transfer path exists for disaggregation; a cross-worker cache adds a directory that can go stale. Routing by prefix sends the request to the worker that has it instead. |
| Hit resolved before step declaration, via a pinned lease | Seed the stream at allocation time | A step's span and CUDA-graph bucket are fixed from prepare_inputs before admit, so an allocation-time hit cannot trim the prompt. The lease pins matched pages until admit consumes or preplan cancels it. |
