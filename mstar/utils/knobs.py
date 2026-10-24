"""Environment knobs read by more than one package."""
import os


def device_loopback_enabled() -> bool:
    """``MSTAR_DEVICE_LOOPBACK`` (default 1): whether a node may keep its
    loop-back token on the device (the sampler's slot master) instead of
    routing a tensor per request; also whether the sampler persists the
    last token at all."""
    return os.environ.get("MSTAR_DEVICE_LOOPBACK", "1") == "1"


def sampler_ingraph_scatter() -> bool:
    """``MSTAR_SAMPLER_INGRAPH_SCATTER`` (default 0): the captured sampler
    gathers its RNG offsets from the slot masters and scatters the advanced
    offsets and the sampled tokens back inside the graph, so a replayed step
    issues none of those launches on the gpu thread (each is a GIL release
    the main thread's postprocess can win). Padding rows then address a trash
    master row, and the masters are allocated once for
    ``MSTAR_SAMPLER_SLOTS`` (default 4096) concurrent requests, since a
    captured graph holds their addresses."""
    return os.environ.get("MSTAR_SAMPLER_INGRAPH_SCATTER", "0") == "1"


def sampler_slots() -> int:
    """``MSTAR_SAMPLER_SLOTS`` (default 4096): master rows allocated up front
    when the in-graph scatter is on (see ``sampler_ingraph_scatter``)."""
    return int(os.environ.get("MSTAR_SAMPLER_SLOTS", "4096"))


def kv_chain_lazy_steps() -> int:
    """``MSTAR_KV_CHAIN_LAZY_STEPS`` (default 16): how many decode steps of
    sampled tokens the KV manager batches before extending the requests'
    prefix chains, one extend per request per batch instead of one per step
    (the per-row Python of that extension was 0.3-0.4 ms a step at 128
    rows). A page key then appears up to that many steps late, and a later
    commit indexes it, which the single-request path already allows for.
    ``1`` (the default) extends every step: 16 measured as noise at c64/c128
    on 0.8B (the extension is a small part of the stop check's tail)."""
    return max(1, int(os.environ.get("MSTAR_KV_CHAIN_LAZY_STEPS", "1")))


def kv_plan_cache() -> int:
    """``MSTAR_KV_PLAN_CACHE`` (default 0): the KV manager plans a leased
    decode step from the previous step's plan when the batch is the same
    rows in the same order and nothing but that step's commit touched the
    streams: lengths advance by one, pages change only where a row crossed
    a page boundary, and the five index arrays come from numpy instead of a
    per-row Python pass. ``0`` (the default until the reworked path is
    measured) plans every step from the streams; ``2`` does both and logs any
    difference (a correctness check for runs)."""
    return int(os.environ.get("MSTAR_KV_PLAN_CACHE", "0"))


def kv_pinned_indptrs() -> bool:
    """``MSTAR_KV_PINNED_INDPTRS`` (default 0): the KV plan writes FlashInfer's
    index arrays (qo_indptr, kv_indptr, kv_indices, last_page_len) into a
    ring of page-locked host buffers, and the decode wrapper copies the
    indices straight into its captured device buffer from there. The copies
    are then really asynchronous; from pageable memory the driver stages each
    one, and FlashInfer makes the indices copy blocking. Off by default: no
    measurable gain at c32-c128 on 0.8B. The cost is one CUDA event per plan,
    which a buffer set waits on before it is rewritten."""
    return os.environ.get("MSTAR_KV_PINNED_INDPTRS", "0") == "1"
