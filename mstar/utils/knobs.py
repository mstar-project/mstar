"""Environment knobs read by more than one package."""
import os


def device_loopback_enabled() -> bool:
    """``MSTAR_DEVICE_LOOPBACK`` (default 1): whether a node may keep its
    loop-back token on the device (the sampler's slot master) instead of
    routing a tensor per request; also whether the sampler persists the
    last token at all."""
    return os.environ.get("MSTAR_DEVICE_LOOPBACK", "1") == "1"


def launch_signal_after_commit() -> bool:
    """``MSTAR_LAUNCH_SIGNAL_AFTER_COMMIT`` (default 0): release the thread
    that submitted a step only once the gpu thread has also committed the
    step and staged its outputs, instead of at the replay launch. Every torch
    call after the launch releases the GIL, and the woken main thread keeps
    it for its whole postprocess stretch, so the first such call waits that
    long; signalling later costs the main thread the uncontended commit and
    collect (a fraction of a millisecond) instead."""
    return os.environ.get("MSTAR_LAUNCH_SIGNAL_AFTER_COMMIT", "0") == "1"


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


def tp_early_spec() -> bool:
    """``MSTAR_TP_EARLY_SPEC`` (default 0): let a tensor-parallel leader build
    its next speculation early (during the current step's postprocess,
    committing only once it holds a head). Off by default: with the device
    loop-back on a 4-way group this stalled every request for ~1.2 s once
    (27B TP4 c8 674 vs 1009 tok/s with it off), and the TP cells are GPU-bound
    anyway. TP1 nodes build early regardless (``MSTAR_EARLY_SPEC``)."""
    return os.environ.get("MSTAR_TP_EARLY_SPEC", "0") == "1"


def ingraph_decode_tokens() -> bool:
    """``MSTAR_INGRAPH_DECODE_TOKENS`` (default 0): a captured decode step
    gathers its input ids off the sampler's slot master inside the graph
    instead of staging a gathered tensor before the replay. Experiment knob:
    the combined form of this and ``ingraph_decode_rope`` produced wrong
    outputs once; each half is gated on its own."""
    return os.environ.get("MSTAR_INGRAPH_DECODE_TOKENS", "0") == "1"


def ingraph_decode_rope() -> bool:
    """``MSTAR_INGRAPH_DECODE_ROPE`` (default 0): a captured decode step
    builds its rotary tables inside the graph from the position resource's
    planned positions instead of staging them before the replay."""
    return os.environ.get("MSTAR_INGRAPH_DECODE_ROPE", "0") == "1"


def kv_chain_lazy_steps() -> int:
    """``MSTAR_KV_CHAIN_LAZY_STEPS`` (default 16): how many decode steps of
    sampled tokens the KV manager batches before extending the requests'
    prefix chains, one extend per request per batch instead of one per step
    (the per-row Python of that extension was 0.3-0.4 ms a step at 128
    rows). A page key then appears up to that many steps late, and a later
    commit indexes it, which the single-request path already allows for.
    ``1`` extends every step as before."""
    return max(1, int(os.environ.get("MSTAR_KV_CHAIN_LAZY_STEPS", "16")))
