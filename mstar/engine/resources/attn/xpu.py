"""Paged attention through vllm-xpu-kernels."""

import functools
from dataclasses import dataclass

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.plan import SINK_PAGE, KVPlanOutput, KVPlanOutputs
from mstar.engine.resources.step import StepContext


@functools.cache
def _xpu_paged_unavailable_reason() -> str | None:
    """Return why the XPU paged-attention extension cannot be loaded."""
    try:
        import vllm_xpu_kernels._C  # noqa: F401
        import vllm_xpu_kernels._xpu_C  # noqa: F401
        from vllm_xpu_kernels.flash_attn_interface import (  # noqa: F401
            flash_attn_varlen_func,
        )
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


@dataclass(frozen=True)
class XPUPagedPlan:
    block_table: torch.Tensor
    cu_q: torch.Tensor
    host_kv_lens: torch.Tensor
    max_q: int
    max_k: int
    causal: bool


class XPUPagedAttentionManager(AttentionManager):
    """Paged KV attention backed by ``vllm-xpu-kernels`` FlashAttention."""

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._current_plans: dict[str, XPUPagedPlan] = {}

    def depends_on(self):
        return {self._kv_cache_name}

    def plan(self, step: AttentionStep, ctx: StepContext):
        if ctx.slot_lease is not None or ctx.is_preplan:
            raise RuntimeError(
                "xpu_paged attention is eager-only: its plan metadata is "
                "rebuilt for every step and does not have capture-stable addresses"
            )
        self.reset_default_cursors()
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"XPU attention expected plan result from {self._kv_cache_name}"
        )
        self._current_plans = {
            label: self._build_plan(kv_out, step.causal)
            for label, kv_out in plan_outputs.items()
        }

    def _build_plan(
        self, kv_out: KVPlanOutput, causal: bool,
    ) -> XPUPagedPlan:
        views = kv_out.views
        max_blocks = max((len(view.page_idxs) for view in views), default=1)
        block_table = [
            view.page_idxs
            + [SINK_PAGE] * (max_blocks - len(view.page_idxs))
            for view in views
        ]

        cu_q = [0]
        kv_lens = []
        for view in views:
            cu_q.append(cu_q[-1] + view.to_compute)
            kv_lens.append(view.length)

        return XPUPagedPlan(
            block_table=torch.tensor(
                block_table, dtype=torch.int32, device=self._device,
            ),
            cu_q=torch.tensor(cu_q, dtype=torch.int32, device=self._device),
            # vllm-xpu-kernels consumes KV lengths from host memory.
            host_kv_lens=torch.tensor(kv_lens, dtype=torch.int32),
            max_q=max((view.to_compute for view in views), default=0),
            max_k=max(kv_lens, default=0),
            causal=causal,
        )

    @torch.compiler.disable
    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        """Return this eager step's query indptr.

        Unlike CUDA-graph FlashInfer buffers, this tensor is newly allocated
        by every plan and therefore has no address-stability guarantee.
        """
        plan = self._current_plans.get(label)
        return None if plan is None else plan.cu_q

    def select_last_hidden(
        self, hidden: torch.Tensor, label: str = "main",
    ) -> torch.Tensor:
        last_token_indices = (self.qo_indptr_buf(label)[1:] - 1).long()
        return hidden.index_select(0, last_token_indices)

    def run(
        self,
        q: torch.Tensor,
        label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        del k, v, layer_idx
        if label is None:
            label = self._default_label
        return self._attend(q, label, kv_cache_layer)

    @torch.compiler.disable
    def _attend(
        self,
        q: torch.Tensor,
        label: str,
        kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        import vllm_xpu_kernels._C  # noqa: F401
        import vllm_xpu_kernels._xpu_C  # noqa: F401
        from vllm_xpu_kernels.flash_attn_interface import (
            flash_attn_varlen_func,
        )

        plan = self._current_plans[label]
        output = flash_attn_varlen_func(
            q,
            kv_cache_layer[:, 0],
            kv_cache_layer[:, 1],
            max_seqlen_q=plan.max_q,
            cu_seqlens_q=plan.cu_q,
            max_seqlen_k=plan.max_k,
            host_kv_lens=plan.host_kv_lens,
            block_table=plan.block_table,
            causal=plan.causal,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.to(q.dtype)
