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
    padded_head_dim,
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
        key: str,
    ):
        super().__init__(device=device, dtype=dtype, config=config, key=key)
        self._workspaces = WorkspacePool(device, config.workspace_mb)
        self._kwargs["backend"] = config.flashinfer_backend

    def _build_wrapper(self, label: str, slot: int, **graph_kwargs) -> _RaggedPrefillBase:
        return super()._build_wrapper(
            label, slot, workspace_buffer=self._workspaces.get(label, slot), **graph_kwargs,
        )

    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        label: str | None = None, return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._config.compile_op and not return_lse and torch.compiler.is_compiling():
            # A head dim the kernels lack is padded here, in the traced graph,
            # where the pad fuses into whatever produced q/k/v and the slice
            # into whatever reads the output; inside the op both are
            # full-size copies per layer.
            d = q.shape[-1]
            padded = padded_head_dim(d)
            if padded != d:
                q, k, v = (torch.nn.functional.pad(t, (0, padded - d)) for t in (q, k, v))
                return super().run(q, k, v, label)[..., :d]
        return super().run(q, k, v, label, return_lse)


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
        key: str,
        block_size: int,
    ):
        super().__init__(device=device, dtype=dtype, config=config, key=key)
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
