"""Waypoint-1.5-1B component modules.

The DiT (``dit.py``, ``attention.py``, ``layers.py``, ``rope.py``) is a native
port of ``world_engine/src/model/world_model.py``. ``apply_inference_patches``
runs unconditionally in the reference's ``WorldEngine.__init__``, so the
*patched* model is the shipped one -- but the port does not follow it uniformly:
it takes the patched **fused QKV** and keeps the unpatched **packed**
``MLPFusion.fc1``, because that patch *splits* the packed weight rather than
merging it, and packed is the checkpoint's own storage. Both forms are
algebraically identical (measured 0.0 either way). See
``docs/waypoint/CONTRACTS.md`` section 5.

``kv_backend.py`` is deliberately *not* part of the module tree: the ring KV
cache is plain classes holding eagerly-allocated tensors, so a meta build and
``to_empty`` cannot touch it, and so it can be swapped for an engine-owned cache
behind ``WaypointKVBackend`` without the DiT noticing (backlog B1). The TAEHV
streaming VAE (``taehv.py``) is phase 7 and does not exist yet.
"""

from mstar.model.waypoint.components.attention import WaypointAttention
from mstar.model.waypoint.components.dit import (
    WaypointDiT,
    WaypointDiTBlock,
    WaypointPosIds,
)
from mstar.model.waypoint.components.kv_backend import (
    FlexRingBackend,
    LayerRingCache,
    WaypointKVBackend,
    describe_ring_memory,
    flex_attention_masked,
    make_block_mask,
    ring_memory_bytes,
)
from mstar.model.waypoint.components.layers import (
    FP32_MODULE_PATHS,
    MLP,
    AdaLN,
    CondHead,
    ControllerInputEmbedding,
    DeviceTableCache,
    MLPFusion,
    NoiseConditioner,
    ada_gate,
    ada_rmsnorm,
    rms_norm,
)
from mstar.model.waypoint.components.rope import (
    OrthoRoPE,
    OrthoRoPEAngles,
    apply_ortho_rope,
)

__all__ = [
    "FP32_MODULE_PATHS",
    "MLP",
    "AdaLN",
    "CondHead",
    "ControllerInputEmbedding",
    "DeviceTableCache",
    "FlexRingBackend",
    "LayerRingCache",
    "MLPFusion",
    "NoiseConditioner",
    "OrthoRoPE",
    "OrthoRoPEAngles",
    "WaypointAttention",
    "WaypointDiT",
    "WaypointDiTBlock",
    "WaypointKVBackend",
    "WaypointPosIds",
    "ada_gate",
    "ada_rmsnorm",
    "apply_ortho_rope",
    "describe_ring_memory",
    "flex_attention_masked",
    "make_block_mask",
    "ring_memory_bytes",
    "rms_norm",
]
