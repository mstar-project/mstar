"""Engine knobs read by more than one package.

Each knob is set per deployment in the serving config's ``knobs:`` block
(``configure`` installs it once per worker process, before the engine is
built) and can be overridden for a run by its environment variable: a
non-empty ``MSTAR_*`` value wins over the config, which wins over the
platform's default (``Platform.knob_defaults``), then the knob's own default.
"""
import functools
import os
from collections.abc import Mapping

# config key in the ``knobs:`` block -> the environment variable that overrides it
KNOBS: dict[str, str] = {
    "device_loopback": "MSTAR_DEVICE_LOOPBACK",
    "sampler_ingraph_scatter": "MSTAR_SAMPLER_INGRAPH_SCATTER",
    "sampler_slots": "MSTAR_SAMPLER_SLOTS",
    "ingraph_decode_rope": "MSTAR_INGRAPH_DECODE_ROPE",
    "kv_chain_lazy_steps": "MSTAR_KV_CHAIN_LAZY_STEPS",
    "kv_plan_cache": "MSTAR_KV_PLAN_CACHE",
    "kv_pinned_indptrs": "MSTAR_KV_PINNED_INDPTRS",
    "gc_gen0_threshold": "MSTAR_GC_GEN0_THRESHOLD",
    "step_thread_core": "MSTAR_STEP_THREAD_CORE",
}

# this process's ``knobs:`` block, as strings in the env vars' format
_CONFIGURED: dict[str, str] = {}


def check(settings: Mapping[str, object] | None) -> dict[str, str]:
    """``settings`` (a ``knobs:`` block) in the env vars' string format.

    Raises on a key that names no knob, so a misspelled setting fails at
    startup instead of being silently ignored."""
    if settings is None:
        return {}
    if not isinstance(settings, Mapping):
        raise ValueError(f"`knobs:` must be a mapping, got {type(settings).__name__}")
    unknown = sorted(set(settings) - KNOBS.keys())
    if unknown:
        raise ValueError(
            f"`knobs:` sets {unknown}, which are not knobs; known: {sorted(KNOBS)}"
        )
    out = {}
    for key, value in settings.items():
        if key == "step_thread_core":
            out[key] = _step_thread_core_value(value)
        elif isinstance(value, bool):
            out[key] = "1" if value else "0"
        elif isinstance(value, int):
            out[key] = str(value)
        else:
            raise ValueError(
                f"`knobs: {key}` must be a bool or an integer, got {value!r}"
            )
    return out


def _step_thread_core_value(value: object) -> str:
    # yaml reads a bare ``off`` as False
    if isinstance(value, bool):
        return "auto" if value else "off"
    if isinstance(value, int) and value >= 0:
        return str(value)
    if isinstance(value, str) and value.strip().lower() in ("auto", "off"):
        return value.strip().lower()
    raise ValueError(
        f"`knobs: step_thread_core` must be auto, off or a CPU number, got {value!r}"
    )


def configure(settings: Mapping[str, object] | None) -> None:
    """Install the deployment config's ``knobs:`` block for this process."""
    checked = check(settings)
    _CONFIGURED.clear()
    _CONFIGURED.update(checked)
    _platform_defaults.cache_clear()


@functools.cache
def _platform_defaults() -> dict[str, str]:
    from mstar.platforms import current_platform

    return current_platform.knob_defaults()


def _setting(key: str, default: str) -> str:
    env = os.environ.get(KNOBS[key])
    if env:
        return env
    if key in _CONFIGURED:
        return _CONFIGURED[key]
    return _platform_defaults().get(key, default)


def device_loopback_enabled() -> bool:
    """``device_loopback`` / ``MSTAR_DEVICE_LOOPBACK`` (default on): whether a
    node may keep its loop-back token on the device (the sampler's slot
    master) instead of routing a tensor per request; also whether the sampler
    persists the last token at all."""
    return _setting("device_loopback", "1") == "1"


def sampler_ingraph_scatter() -> bool:
    """``sampler_ingraph_scatter`` / ``MSTAR_SAMPLER_INGRAPH_SCATTER`` (default
    off, on for AMD GPUs): the captured sampler gathers its RNG offsets from
    the slot masters and scatters the advanced offsets and the sampled tokens
    back inside the graph, so a replayed step issues none of those launches on
    the gpu thread (each is a GIL release the main thread's postprocess can
    win). Padding rows then address a trash master row, and the masters are
    allocated once for ``sampler_slots`` concurrent requests, since a captured
    graph holds their addresses."""
    return _setting("sampler_ingraph_scatter", "0") == "1"


def sampler_slots() -> int:
    """``sampler_slots`` / ``MSTAR_SAMPLER_SLOTS`` (default 4096): master rows
    allocated up front when the in-graph scatter is on (see
    ``sampler_ingraph_scatter``)."""
    return int(_setting("sampler_slots", "4096"))


def ingraph_decode_rope() -> bool:
    """``ingraph_decode_rope`` / ``MSTAR_INGRAPH_DECODE_ROPE`` (default off, on
    for AMD GPUs): a captured decode step builds its rotary tables inside the
    graph from the position resource's planned positions instead of staging
    them before the replay."""
    return _setting("ingraph_decode_rope", "0") == "1"


def kv_chain_lazy_steps() -> int:
    """``kv_chain_lazy_steps`` / ``MSTAR_KV_CHAIN_LAZY_STEPS`` (default 1): how
    many decode steps of sampled tokens the KV manager batches before
    extending the requests' prefix chains, one extend per request per batch
    instead of one per step (the per-row Python of that extension was
    0.3-0.4 ms a step at 128 rows). A page key then appears up to that many
    steps late, and a later commit indexes it, which the single-request path
    already allows for. ``1`` extends every step: 16 measured as noise at
    c64/c128 on 0.8B (the extension is a small part of the stop check's
    tail)."""
    return max(1, int(_setting("kv_chain_lazy_steps", "1")))


def kv_plan_cache() -> int:
    """``kv_plan_cache`` / ``MSTAR_KV_PLAN_CACHE`` (default 0): the KV manager
    plans a leased decode step from the previous step's plan when the batch
    is the same rows in the same order and nothing but that step's commit
    touched the streams: lengths advance by one, pages change only where a
    row crossed a page boundary, and the five index arrays come from numpy
    instead of a per-row Python pass. ``0`` plans every step from the
    streams; ``2`` does both and logs any difference (a correctness check for
    runs)."""
    return int(_setting("kv_plan_cache", "0"))


def kv_pinned_indptrs() -> bool:
    """``kv_pinned_indptrs`` / ``MSTAR_KV_PINNED_INDPTRS`` (default off): the
    KV plan writes FlashInfer's index arrays (qo_indptr, kv_indptr,
    kv_indices, last_page_len) into a ring of page-locked host buffers, and
    the decode wrapper copies the indices straight into its captured device
    buffer from there. The copies are then really asynchronous; from pageable
    memory the driver stages each one, and FlashInfer makes the indices copy
    blocking. Off by default: no measurable gain at c32-c128 on 0.8B. The
    cost is one CUDA event per plan, which a buffer set waits on before it is
    rewritten."""
    return _setting("kv_pinned_indptrs", "0") == "1"


def gc_gen0_threshold() -> int:
    """``gc_gen0_threshold`` / ``MSTAR_GC_GEN0_THRESHOLD`` (default 0, keep
    CPython's 700): the generation-0 threshold a worker sets after warmup.
    A decode step at 128 rows keeps more than 700 young container objects
    alive at its peak, so at 700 a collection runs almost every step and
    escalates to a full one (about 10 ms) every few hundred; the time lands
    in whichever span is allocating. 50000 removed them on Qwen3.5 0.8B at
    c=128 (8% shorter decode step) for at most that many young objects held
    between collections."""
    return max(0, int(_setting("gc_gen0_threshold", "0")))


def step_thread_core() -> str | int:
    """``step_thread_core`` / ``MSTAR_STEP_THREAD_CORE`` (default ``auto``):
    the CPU a worker's main, GPU and plan threads share (and the one the API
    server's result threads share). ``auto`` picks one per worker from its
    GPU's local CPUs, ``off`` leaves them on the process mask, and a number is
    the first worker's CPU (see ``mstar.utils.step_threads``)."""
    value = _setting("step_thread_core", "auto").strip().lower()
    if value in ("auto", "off"):
        return value
    if value.isdigit():
        return int(value)
    raise ValueError(
        f"MSTAR_STEP_THREAD_CORE must be auto, off or a CPU number, got {value!r}"
    )
