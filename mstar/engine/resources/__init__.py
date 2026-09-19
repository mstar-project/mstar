"""The resource layer's declaration surface.

Everything re-exported here is a declaration — a spec, a per-request config, a
step, or the generic step envelope — and every module behind it is free of the
managers and their kernels. So a model can import all of it in one line without
dragging FlashInfer or Triton in behind it; the concrete resources are reached
by their own paths.

Nothing inside the package may import from here: during this module's own
execution the package is only half-initialized. Import siblings by their
submodule path instead.
"""

from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    CrossAttentionConfig,
    CrossAttentionSpec,
)
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
)
from mstar.engine.resources.base import CGSlotSpec, PublishedInfo, Resource
from mstar.engine.resources.kv.config import (
    KVConfig,
    KVLayout,
    KVReqConfig,
    KVSpec,
    KVStep,
)
from mstar.engine.resources.position.config import (
    PosBackend,
    PositionConfig,
    PositionSpec,
    PositionStep,
    PosScheme,
)
from mstar.engine.resources.runner import StepRunner, topo_sort
from mstar.engine.resources.sampler.config import (
    SamplerSpec,
    SamplerStep,
    SamplingReqConfig,
)
from mstar.engine.resources.spec import (
    NodeResourceSpec,
    ResourceReqConfig,
    apply_yaml_overrides,
    resolve_spec_dependencies,
)
from mstar.engine.resources.step import (
    AdmitFailedReason,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
    BucketKey,
    FullAdmitOutcome,
    RequestOffloading,
    ResourceStep,
    Segment,
    SlotLease,
    StepContext,
    SubmoduleStep,
)

__all__ = [
    "AdmitFailedReason",
    "AdmitOutcome",
    "AdmitRuntimeError",
    "AllocationFailed",
    "FullAdmitOutcome",
    "RequestOffloading",
    "AttentionConfig",
    "AttentionSpec",
    "AttentionStep",
    "AttnBackend",
    "BucketKey",
    "CGSlotSpec",
    "CrossAttentionConfig",
    "CrossAttentionSpec",
    "KVConfig",
    "KVLayout",
    "KVReqConfig",
    "KVSpec",
    "KVStep",
    "NodeResourceSpec",
    "PosBackend",
    "PosScheme",
    "PositionConfig",
    "PositionSpec",
    "PositionStep",
    "PublishedInfo",
    "RaggedAttentionConfig",
    "RaggedAttentionSpec",
    "Resource",
    "ResourceReqConfig",
    "ResourceStep",
    "SamplerSpec",
    "SamplerStep",
    "SamplingReqConfig",
    "Segment",
    "SlotLease",
    "StepContext",
    "StepRunner",
    "SubmoduleStep",
    "apply_yaml_overrides",
    "resolve_spec_dependencies",
    "topo_sort",
]
