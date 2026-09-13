"""Waypoint-1.5-1B component modules.

The DiT is a native port of ``world_engine/src/model/world_model.py``.
``apply_inference_patches`` runs unconditionally in the reference, so the
*patched* model is the shipped one -- but the port does not follow it
uniformly: it takes the patched **fused QKV** and keeps the unpatched
**packed** ``MLPFusion.fc1``, since that patch splits a weight the checkpoint
stores packed. The two forms are algebraically identical.

**Nothing here owns the world state.** The ring KV cache and the FlexAttention
kernel are engine resources (``engine/resources/kv/ring/``,
``engine/resources/attn/flex.py``) bound onto ``WaypointDiT`` and its 24
``WaypointAttention`` layers at load by ``NodeSubmodule.bind_node_resources``.
That keeps the rings out of the module tree, where ``to_empty(device)``,
``state_dict()`` and the weight loader would each have a buffer of ours to
leave holding garbage.

``taehv.py`` is the exception: a streaming AE session is per-request Python
state, held in ``PerRequestState`` by the two VAE nodes and dropped with the
request. Its ``taehv`` imports are deferred, so this package imports without
the upstream package installed.
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
from mstar.model.waypoint.components.taehv import ChunkedStreamingTAEHV, load_taehv

__all__ = [
    "FP32_MODULE_PATHS",
    "MLP",
    "AdaLN",
    "ChunkedStreamingTAEHV",
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
    "load_taehv",
    "rms_norm",
]
