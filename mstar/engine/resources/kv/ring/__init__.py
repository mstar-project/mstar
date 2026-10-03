"""Ring KV storage: a fixed horizon of frame slots, overwritten in place.

The storage strategy Waypoint declares. ``KVSpec.resource_class`` dispatches
here on seeing a ``RingKVConfig``, so a paged model never imports it and this
package never imports FlashInfer.
"""

from mstar.engine.resources.kv.ring.cache import LayerRingCache, ring_scatter
from mstar.engine.resources.kv.ring.manager import RingKVManager

__all__ = ["LayerRingCache", "RingKVManager", "ring_scatter"]
