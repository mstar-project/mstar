from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, ResourceType
from mstar.engine.resources.step import ADMIT_OK, AdmitOutcome, BucketKey, ResourceStep, StepContext

if TYPE_CHECKING:
    # the config reaches back here through the submodule base, so keep the
    # import out of module exec
    from mstar.engine.cuda_graph_config import (
        CudaGraphConfig,
        PiecewiseCudaGraphConfig,
    )
    from mstar.engine.resources.kv.transfer import TransferEngineInfo


@dataclass(frozen=True)
class CGSlotSpec:
    bucket: BucketKey
    slot: int
    # a whole forward's capture, or one piecewise region's
    config: CudaGraphConfig | PiecewiseCudaGraphConfig
    config_idx: int | None = None

    @property
    def bs(self):
        return self.bucket.bs

    @property
    def num_tokens(self):
        return self.bucket.num_tokens

    def __str__(self) -> str:
        return f"{self.bucket} slot={self.slot}"


@dataclass(frozen=True)
class CGSlotKey:
    bucket: BucketKey
    slot: int
    label: str


class Resource(ABC):
    @classmethod
    @abstractmethod
    def build(
        cls, spec: NodeResourceSpec,
        device: torch.device,
        comm_group: CommGroup | None,
        **engine_kwargs
    ) -> "Resource":
        ...

    def depends_on(self) -> set[str]:
        return set()


    # Request lifecycle

    def ingest_request(self, rid: str, overrides: ResourceReqConfig | None):
        return

    def remove_request(self, rid: str):
        return

    def admit_retrieve(
        self, rid: str,
        node_name: str,
        graph_walk: str,
        published: "PublishedInfo | None"
    ) -> AdmitOutcome:
        """
        Takes the output of publish, possibly from another device, and kicks
        of a retrieval if needed (e.g., PD disaggregation KV transfer).
        Returns whether the retrieve has completed.
        """
        return ADMIT_OK

    # Step lifecycle

    def admit(self, step: ResourceStep, ctx: StepContext) -> AdmitOutcome:
        return ADMIT_OK

    def plan(self, step: ResourceStep, ctx: StepContext) -> Any:
        """ret is immutable and opaque to runner; only gives to `ctx.plan_results`"""
        return None

    def commit(self, step: ResourceStep, ctx: StepContext) -> None:
        """record step consumption"""
        return

    def publish(self, request_id: str) -> "PublishedInfo | None":
        return None

    def reset_request(self, rid: str, free: bool=False):
        """For clearing dummy RIDs during cuda graph capture"""
        return

    # Pre-planning

    @property
    def supports_preplan(self):
        return False

    def clear_preplan(self):
        return

    # Eviction
    #
    # A resource that holds enough per-request state to be worth reclaiming
    # (the KV cache, today) opts in here; the worker picks victims and drives
    # the move. Which requests exist and how recently they ran is the
    # scheduler's knowledge, so the resource only answers for its own state.

    @property
    def supports_eviction(self):
        return False

    def is_offloaded(self, rid: str) -> bool:
        return False

    def offload(self, rid: str) -> int:
        """Move the request's state off-device. Returns what was reclaimed."""
        return 0

    def reload(self, rid: str) -> bool:
        """Bring it back. False when it doesn't fit on device yet."""
        return True

    def get_offload_priority(self, rid: str) -> float:
        """How much this resource wants ``rid`` gone, higher being more.

        Only consulted under a PRIORITY eviction policy, which names the
        resource to ask; LRU never calls it.
        """
        return 0.0

    # Engine lifecycle

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec],
        max_bs: int, max_seq_len: int
    ) -> None:
        """Size whatever the captured replays will read.

        Called once per runner that captures against this node — the whole
        forward's, and one per piecewise region — so it must tolerate repeated
        calls: grow to the largest shape asked for, never clobber what an
        earlier call already sized.
        """
        # NOTE @nsagan: this should probably be refined; it was just the first
        # thing that came to mind
        return

    def post_warmup_validate(self):
        """
        For, e.g., the KV cache to check that num_free_pages is identical
        across TP ranks after cuda graph capture.

        Raises an error (fails loudly) if invalid.
        """
        return

    def cleanup(self):
        return


class AttentionResource(Resource):
    """A resource a layer stack calls per layer, under one plan label.

    Adds the label / layer-index cursors: a caller running the whole stack sets
    them once instead of threading them through every call, and an explicit
    argument still supersedes. Class-level defaults so a subclass picks them up
    without touching its __init__; subclasses that read them clear them in
    `plan`, so a step that never binds cannot inherit the previous step's.

    The KV, attention and cross-attention resources; not the sampler or the
    position resource, which are called once per step rather than per layer.
    """

    _default_label: str = "main"
    _default_layer_idx: int | None = None

    @property
    def default_label(self) -> str:
        return self._default_label

    def set_default_label(self, label: str) -> None:
        self._default_label = label

    def set_default_layer_idx(self, layer_idx: int) -> None:
        self._default_layer_idx = layer_idx

    def reset_default_cursors(self) -> None:
        self._default_label = "main"
        self._default_layer_idx = None


class PublishedInfo(ABC):
    @abstractmethod
    def update(self, other: "PublishedInfo") -> None:
        ...


def build_resource(
    spec: NodeResourceSpec,
    device: torch.device,
    joint_comm_group: JointGroups,
    transfer_engine_info: TransferEngineInfo,
    kv_dtype: torch.dtype
) -> Resource:
    if spec.resource_type == ResourceType.KV_CACHE:
        from mstar.engine.resources.kv.manager import KVManager
        return KVManager.build(
            spec=spec,
            device=device,
            joint_comm_group=joint_comm_group,
            transfer_engine_info=transfer_engine_info,
            dtype=kv_dtype
        )
    if spec.resource_type == ResourceType.SAMPLER:
        from mstar.engine.resources.sampler.resource import SamplerResource
        return SamplerResource.build(
            spec=spec,
            device=device,
            joint_comm_group=joint_comm_group
        )
    if spec.resource_type == ResourceType.ATTENTION:
        from mstar.engine.resources.attn.manager import AttentionManager
        return AttentionManager.build(
            spec=spec,
            device=device,
            joint_comm_group=joint_comm_group,
            dtype=kv_dtype
        )
    if spec.resource_type == ResourceType.CROSS_ATTENTION:
        from mstar.engine.resources.attn.manager import CrossAttentionManager
        return CrossAttentionManager.build(
            spec=spec,
            device=device,
            joint_comm_group=joint_comm_group,
            dtype=kv_dtype
        )
    if spec.resource_type == ResourceType.POSITIONS:
        from mstar.engine.resources.position.manager import PositionManager
        return PositionManager.build(
            spec=spec, device=device
        )
