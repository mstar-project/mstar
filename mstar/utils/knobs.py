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
