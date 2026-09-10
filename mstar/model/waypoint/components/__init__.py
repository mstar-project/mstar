"""Waypoint-1.5-1B component modules.

The DiT (``dit.py``, ``attention.py``, ``layers.py``, ``rope.py``) is a native
port of ``world_engine/src/model/world_model.py``. ``apply_inference_patches``
runs unconditionally in the reference's ``WorldEngine.__init__``, so the
*patched* model is the shipped one -- but the port does not follow it uniformly:
it takes the patched **fused QKV** and keeps the unpatched **packed**
``MLPFusion.fc1``, because that patch *splits* the packed weight rather than
merging it, and packed is the checkpoint's own storage. Both forms are
algebraically identical (measured 0.0 either way).

**Nothing here owns the world state.** The ring KV cache and the FlexAttention
kernel are engine resources (``engine/resources/kv/ring/``,
``engine/resources/attn/flex.py``), bound onto ``WaypointDiT`` and its 24
``WaypointAttention`` layers at load by ``NodeSubmodule.bind_node_resources``.
That keeps the rings out of the module tree, where ``to_empty(device)``,
``state_dict()`` and the weight loader would each have a buffer of ours to leave
holding garbage. The superseded model-owned implementation (``kv_backend.py``,
with its ``FlexRingBackend`` and ``WaypointKVBackend`` protocol) was deleted
once the equivalence gate against it passed bit-exactly; ``git show
d31c3e70:mstar/model/waypoint/components/kv_backend.py`` is the last version.

``ring_memory_bytes`` / ``describe_ring_memory`` moved up a level to
``waypoint/ring_geometry.py``: they are ``WaypointConfig`` arithmetic for
sizing a deployment and never needed a component to exist. The TAEHV streaming
VAE (``taehv.py``) is phase 7 and does not exist yet.
"""

from mstar.model.waypoint.components.attention import WaypointAttention
from mstar.model.waypoint.components.dit import (
    WaypointDiT,
    WaypointDiTBlock,
    WaypointPosIds,
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
    "MLPFusion",
    "NoiseConditioner",
    "OrthoRoPE",
    "OrthoRoPEAngles",
    "WaypointAttention",
    "WaypointDiT",
    "WaypointDiTBlock",
    "WaypointPosIds",
    "ada_gate",
    "ada_rmsnorm",
    "apply_ortho_rope",
    "rms_norm",
]
