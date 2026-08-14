

from dataclasses import dataclass
from enum import Enum
import os
from typing import NamedTuple

import torch

from mstar.engine.resources.base import CGSlotSpec, Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceType
from mstar.engine.resources.step import AttentionStep, BucketKey, StepContext
from mstar.engine.v1.kv_cache import KVConfig
from mstar.engine.v1.kv_manager import KVManager, KVPlanOutput, SequenceView
from mstar.engine.v1.attention_wrappers import FlashInferDecodeWrapper, FlashInferPrefillWrapper


class AttnBackend(Enum):
    FLASHINFER = "flashinfer"

@dataclass
class AttentionConfig:
    backend: AttnBackend=AttnBackend.FLASHINFER
    kv_cache: str # name of the KV cache
    flashinfer_backend: str = "auto"


@dataclass
class AttentionSpec(NodeResourceSpec):
    config: AttentionConfig
    kv_config: KVConfig

    @property
    def resource_type(self):
        return ResourceType.ATTENTION


class AttentionManager(Resource):
    # Remains abstract except for build; will build based
    # on the attention backend

    @classmethod
    def build(
        cls, spec: AttentionSpec,
        device: torch.device,
        dtype=torch.bfloat16,
        **engine_kwargs
    ):
        if spec.config.backend == AttnBackend.FLASHINFER:
            return FlashInferManager(
                kv_cache=spec.config.kv_cache,
                device=device,
                dtype=dtype,
                kv_config=spec.kv_config,
                backend=spec.config.flashinfer_backend,
            )


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable."""
    q_seq_lens: tuple
    page_indices: tuple
    last_page_lens: tuple


@dataclass(frozen=True)
class CGSlotKey:
    bucket: BucketKey
    slot: int
    label: str


AttentionWrapper = FlashInferPrefillWrapper | FlashInferDecodeWrapper


class FlashInferManager(AttentionManager):
    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: KVConfig,
        backend: str="auto",
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype

        # label to plan state
        self._current_plan_states: dict[str, AttentionWrapper] = {}

        self._cg_plan_states: dict[CGSlotKey, AttentionWrapper] = {}

        self._kv_config = kv_config
        self._wrapper_kv_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            num_kv_heads=kv_config.num_kv_heads,
            head_dim=kv_config.head_dim,
            page_size=kv_config.page_size,
            max_num_pages=kv_config.max_num_pages,
            device=device,
            backend=backend
        )

        self._buffer_size = int(
            os.environ.get("MSTAR_WORKSPACE_BUFFER_MB", "512")
        ) * 1024 * 1024
        self._workspace_buffers: dict[str, torch.Tensor] = {}

    def depends_on(self):
        return self._kv_cache_name

    def _get_workspace_buffer(
        self, label: str, cg_slot: int | None=None
    ):
        key = label if cg_slot is not None else f"{label}_{cg_slot}"
        if key not in self._workspace_buffers:
            self._workspace_buffers[key] = torch.empty(
                self._buffer_size, dtype=torch.uint8, device=self._device
            )
        return self._workspace_buffers[key]

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ):
        del max_bs, max_seq_len
        for slot in slots:
            bucket = slot.bucket
            is_decode = bucket.bs == bucket.num_tokens

            for label in slot.config.labels:
                buffer = self._get_workspace_buffer(label, slot.slot)
                if is_decode:
                    wrapper = FlashInferDecodeWrapper(
                        workspace_buffer=buffer,
                        batch_size=bucket.bs,
                        use_cuda_graph=True,
                        **self._wrapper_kv_kwargs,
                    )
                else:
                    wrapper = FlashInferPrefillWrapper(
                        workspace_buffer=buffer,
                        batch_size=bucket.bs,
                        max_total_tokens=bucket.num_tokens,
                        use_cuda_graph=True,
                        **self._wrapper_kv_kwargs,
                    )
                self._cg_plan_states[CGSlotKey(
                    bucket=slot.bucket,
                    slot=slot.slot,
                    label=label
                )] = wrapper

    
    def plan(self, step: AttentionStep, ctx: StepContext):
        plan_outputs: dict[str, KVPlanOutput] = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, f"Attention Manager expected plan result from {self._kv_cache_name}"

        self._current_plan_states.clear()
        for label, kv_out in plan_outputs.items():
            indptrs = kv_out.cpu_indptrs
            if ctx.slot_lease is not None:
                wrapper = self._cg_plan_states[CGSlotKey(
                    bucket=ctx.slot_lease.bucket,
                    slot=ctx.slot_lease.slot,
                    label=label
                )]
            else:
                is_decode = indptrs.qo_indptr[-1] == len(indptrs.qo_indptr) - 1
                buffer = self._get_workspace_buffer(label)
                if is_decode:
                    wrapper = FlashInferDecodeWrapper(
                        workspace_buffer=buffer,
                        **self._wrapper_kv_kwargs,
                    )
                else:
                    wrapper = FlashInferPrefillWrapper(
                        workspace_buffer=buffer,
                        **self._wrapper_kv_kwargs,
                    )

            # TODO: cache the latest plan state
            wrapper.plan(
                causal=step.causal,
                dtype=self._dtype,
                **indptrs.to_kwargs_dict()
            )
            self._current_plan_states[label] = wrapper

    ### Submodule-level functionality
    def qo_indptr_buf(self, label: str) -> torch.Tensor | None:
        if label not in self._current_plan_states:
            return
        return self._current_plan_states[label]._qo_indptr_buf
    
    def run(
        self, q: torch.Tensor, label: str,
        kv_cache_layer: torch.Tensor
    ) -> torch.Tensor:
        return self._current_plan_states[label].run(q, kv_cache_layer)

        
# TODO: cross attention, dense attention