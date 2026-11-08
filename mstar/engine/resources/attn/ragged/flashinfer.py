"""Cacheless varlen attention through FlashInfer's ragged prefill wrapper, or
the kernel a ``ragged_attention`` table row names for the layout."""

import functools
import logging

import torch

from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.attn.ragged.base import (
    RaggedBlockCausalAttnManager,
    _RaggedCrossManagerBase,
    _RaggedManagerBase,
    _RaggedSelfManagerBase,
)
from mstar.engine.resources.attn.ragged.block_causal import RaggedBlockCausalWrapper
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig
from mstar.engine.resources.attn.ragged.cudnn_wrappers import CudnnRaggedPrefillWrapper, uniform_segments
from mstar.engine.resources.attn.ragged.wrappers import (
    FlashInferRaggedCrossPrefillWrapper,
    FlashInferRaggedPrefillWrapper,
    _RaggedPrefillBase,
    padded_head_dim,
)
from mstar.engine.resources.attn.workspace import WorkspacePool
from mstar.engine.resources.step import StepContext
from mstar.kernels.registry import describe_rows
from mstar.kernels.table import Rows

logger = logging.getLogger(__name__)


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
        self._workspaces = WorkspacePool(device, config.workspace_mb, owner=key)
        self._kwargs["backend"] = config.flashinfer_backend

    def _build_wrapper(
        self, label: str, slot: int, cu_seqlens: torch.Tensor | None = None, **graph_kwargs,
    ) -> _RaggedPrefillBase:
        return super()._build_wrapper(
            label, slot, cu_seqlens, workspace_buffer=self._workspaces.get(label, slot), **graph_kwargs,
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
    """With ``kernel_rows`` (a ``ragged_attention`` table's, or the ``cudnn``
    backend's one row), each wrapper runs the row its first layout's longest
    query segment falls in, for its lifetime: a captured graph can't switch
    kernels, and an eager wrapper keeps the captured ones' rule. A ``cudnn``
    row needs equal-length segments, and under capture a config that declares
    ``fixed_bucket_layouts``; otherwise the wrapper runs on FlashInfer."""

    wrapper_class = FlashInferRaggedPrefillWrapper

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: RaggedAttentionConfig,
        key: str,
        kernel_rows: Rows | None = None,
    ):
        super().__init__(device=device, dtype=dtype, config=config, key=key)
        self._kernel_rows = kernel_rows
        if kernel_rows is not None:
            logger.info("ragged attention %r (%d/%d heads, head dim %d, %s): %s", key, config.num_qo_heads,
                        config.num_kv_heads, config.head_dim, str(dtype).removeprefix("torch."),
                        describe_rows(kernel_rows))

    def _build_wrapper(
        self, label: str, slot: int, cu_seqlens: torch.Tensor | None = None, **graph_kwargs,
    ) -> _RaggedPrefillBase:
        if self._kernel_rows is None or cu_seqlens is None:
            return super()._build_wrapper(label, slot, cu_seqlens, **graph_kwargs)
        from mstar.engine.resources.attn.ragged.kernels import choice_for

        seq_len = int(torch.diff(cu_seqlens.to(device="cpu", dtype=torch.int64)).max()) if cu_seqlens.numel() > 1 else 0
        choice = choice_for(self._kernel_rows, seq_len)
        captured = bool(graph_kwargs.get("accelerator_graph"))
        if (choice.impl == "cudnn" and uniform_segments(cu_seqlens) is not None
                and (not captured or self._config.fixed_bucket_layouts)):
            fallback = None if graph_kwargs else functools.partial(super()._build_wrapper, label, slot)
            return CudnnRaggedPrefillWrapper(
                num_qo_heads=self._config.num_qo_heads, num_kv_heads=self._config.num_kv_heads,
                head_dim=self._config.head_dim, device=self._device, sm_scale=self._config.sm_scale,
                q_data_type=self._dtype, fallback=fallback, **graph_kwargs,
            )
        backend = (choice.config or {}).get("backend") if choice.impl == "flashinfer" else None
        if backend is None:
            return super()._build_wrapper(label, slot, cu_seqlens, **graph_kwargs)
        return self.wrapper_class(
            **{**self._kwargs, "backend": backend}, workspace_buffer=self._workspaces.get(label, slot), **graph_kwargs,
        )


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
