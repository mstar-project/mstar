"""Cacheless varlen attention through FlashInfer's ragged prefill wrapper."""

import torch

from mstar.engine.resources.attn.base import WorkspacePool
from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.attn.ragged.base import (
    RaggedBlockCausalAttnManager,
    _RaggedCrossManagerBase,
    _RaggedManagerBase,
    _RaggedSelfManagerBase,
)
from mstar.engine.resources.attn.ragged.block_causal import RaggedBlockCausalWrapper
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig
from mstar.engine.resources.attn.ragged.wrappers import (
    FlashInferRaggedCrossPrefillWrapper,
    FlashInferRaggedPrefillWrapper,
    _RaggedPrefillBase,
)
from mstar.engine.resources.step import StepContext


class _FlashInferRaggedManagerBase(_RaggedManagerBase):
    """FlashInfer extras on top of the shared manager: the kernel backend, and a
    workspace per (label, slot) shared by that slot's eager and captured wrappers."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: RaggedAttentionConfig,
    ):
        super().__init__(device=device, dtype=dtype, config=config)
        self._workspaces = WorkspacePool(device, config.workspace_mb)
        self._kwargs["backend"] = config.flashinfer_backend

    def _build_wrapper(self, label: str, slot: int, **graph_kwargs) -> _RaggedPrefillBase:
        return super()._build_wrapper(
            label, slot, workspace_buffer=self._workspaces.get(label, slot), **graph_kwargs,
        )


class FlashInferRaggedManager(_FlashInferRaggedManagerBase, _RaggedSelfManagerBase):
    wrapper_class = FlashInferRaggedPrefillWrapper


class FlashInferRaggedCrossManager(_FlashInferRaggedManagerBase, _RaggedCrossManagerBase):
    wrapper_class = FlashInferRaggedCrossPrefillWrapper


class FlashInferRaggedBlockCausalManager(_FlashInferRaggedManagerBase, RaggedBlockCausalAttnManager):
    """Block-causal self-attention: every label in the step is its own layout,
    each segment attending block-causally within itself."""

    wrapper_class = RaggedBlockCausalWrapper

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: RaggedAttentionConfig,
        block_size: int,
    ):
        super().__init__(device=device, dtype=dtype, config=config)
        self._kwargs["block_size"] = block_size

    def plan(self, step: AttentionStep, ctx: StepContext):
        refuse_context_only(step, type(self).__name__)
        super().plan(step, ctx)

    def _plan_layouts(self, step: AttentionStep, ctx: StepContext, plan_states) -> None:
        if step.causal:
            raise ValueError(
                "block-causal attention is its own mask; declare it with "
                "AttentionStep(causal=False)"
            )
        for label, segments in self._group_segments_by_label(step.segments or ()).items():
            wrapper = self._wrapper_for(ctx.slot_lease, ctx.slot, label)
            wrapper.plan([seg.span for seg in segments])
            plan_states[label] = wrapper
