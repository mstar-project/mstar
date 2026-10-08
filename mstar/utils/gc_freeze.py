"""Move a process's set-up heap out of the garbage collector's reach.

A serving process builds a large, long-lived heap while it starts (configs,
tokenizers, weights' Python wrappers, capture buffers). Every gen-2
collection afterwards walks all of it: measured at 200 ms per pass in the
API server process, every ~20 s under load, each pass stalling request
admission and token delivery for that long. Collecting once and freezing
moves those objects to the permanent generation, so later collections only
walk what the process created since. Refcounting still frees frozen objects;
only a reference cycle alive at the freeze would be retained for good.
"""
import gc
import logging
import os

logger = logging.getLogger(__name__)


def freeze_after_setup(who: str) -> int:
    """``gc.collect()`` then ``gc.freeze()`` for a process that finished its
    set-up. Returns the number of objects frozen; 0 and no change when
    ``MSTAR_GC_FREEZE=0``."""
    if os.environ.get("MSTAR_GC_FREEZE", "1") != "1":
        return 0
    gc.collect()
    gc.freeze()
    frozen = gc.get_freeze_count()
    logger.info("%s: gc.freeze() after setup, %d objects in the permanent generation", who, frozen)
    return frozen
