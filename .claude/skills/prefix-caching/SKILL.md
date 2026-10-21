---
name: prefix-caching
description: Things to keep in mind for changes that touch the KV cache, with respect to prefix caching.
---

# Cross-request prefix reuse

Prefix caching retains a finished request's KV pages so the next request with the same prefix skips recomputing them. Design and rationale: [RFC #210](https://github.com/mstar-project/mstar/issues/210). The rules that make it safe:

- **Sharing is by owner count, and only sealed pages may be shared.** `PageArena` tracks `num_owners` per page and is the only path to the allocator; a page is freed when the count hits zero, not when a request ends. Appends land in the last page until it fills, so a partially filled page under two owners would have one owner mutating bytes another reads — the tail page therefore stays exclusive, and there is an assertion for it. Sealed pages are never written again; a stream that rewinds across one drops its pages and reacquires private ones.
- **The key must name everything the stored bytes depend on.** SHA-256, as a chain (`resources/kv/keys.py`: `fingerprint`, `page_key`, `chain`) rooted in a process fingerprint covering weights, TP head slice, dtype, KV layout, page size, attention backend and position config. Change any of those and every old key becomes unreachable, so the cache empties itself. Media folds in a digest of raw bytes plus preprocessor params plus encoder fingerprint — placeholder token ids alone are identical across images, which is a collision both vLLM and SGLang have shipped. **Adding anything that changes hidden state without adding it to the chain is a correctness bug, not a cache-miss bug.**
- **The matched length is the minimum across the node's resources** (`StepRunner.resolve_cached_prefix`; a resource with no opinion is not an answer of zero), resolved on the CPU before `prepare_inputs`. Bucket selection, attention planning, positions and commit all see a shorter request and never learn a hit happened. Keep it that way.
- **Eligibility is narrow deliberately:** a fresh stream, a keyed walk, a linear position scheme with no custom position ids (asserted), and no extra tensor inputs, kwargs or resource step info. Widening it means proving the stored bytes are valid at the positions they will be read at.
- **A live request must never fail to allocate because of cached data.** Release runs where the shortfall surfaces, in `_alloc` under the arena lock, and removes **leaves only**, so parent pointers never go stale.
- **TP > 1 is not correct yet** — each rank probes its own index and nothing reconciles the matched length across a lockstep instance. Tracked in [#308](https://github.com/mstar-project/mstar/issues/308); don't file it again, and don't assume rank agreement in new code.