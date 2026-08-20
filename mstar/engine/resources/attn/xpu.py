"""Paged attention through vllm-xpu-kernels."""

import functools
import math
from dataclasses import dataclass

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import KVConfig
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


@dataclass
class XPUPagedPlan:
    block_table: torch.Tensor
    cu_q: torch.Tensor
    host_kv_lens: torch.Tensor
    max_q: int
    max_k: int
    causal: bool

    def copy_(self, other: "XPUPagedPlan") -> None:
        """Refresh graph-stable plan buffers without changing their addresses."""
        for name in ("block_table", "cu_q", "host_kv_lens"):
            dst = getattr(self, name)
            src = getattr(other, name)
            if dst.shape != src.shape:
                raise ValueError(
                    f"XPU graph {name} shape changed from "
                    f"{tuple(dst.shape)} to {tuple(src.shape)}"
                )
            dst.copy_(src)


class XPUPagedAttentionManager(AttentionManager):
    """Paged KV attention backed by ``vllm-xpu-kernels`` FlashAttention."""

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        kv_config: KVConfig,
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._kv_config = kv_config
        self._current_plans: dict[str, XPUPagedPlan] = {}
        self._cg_plans: dict[CGSlotKey, XPUPagedPlan] = {}
        self._preplan_states: dict[str, XPUPagedPlan] = {}
        self._preplanned = False

    def depends_on(self):
        return {self._kv_cache_name}

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires an accelerator graph step"
        )
        if self._preplanned:
            self._current_plans = self._preplan_states
            self._preplan_states = {}
            self._preplanned = False
            return

        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"XPU attention expected plan result from {self._kv_cache_name}"
        )
        plan_states = (
            self._preplan_states if ctx.is_preplan else self._current_plans
        )
        plan_states.clear()
        for label, kv_out in plan_outputs.items():
            new_plan = self._build_plan(
                kv_out, step.causal, graph_mode=lease is not None
            )
            if lease is None:
                plan = new_plan
            else:
                key = CGSlotKey(
                    bucket=lease.bucket, slot=lease.slot, label=label
                )
                plan = self._cg_plans.get(key)
                if plan is None:
                    self._cg_plans[key] = new_plan
                    plan = new_plan
                else:
                    plan.copy_(new_plan)
            plan_states[label] = plan
        self._preplanned = ctx.is_preplan

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_states = {}

    def _build_plan(
        self, kv_out: KVPlanOutput, causal: bool, graph_mode: bool,
    ) -> XPUPagedPlan:
        views = kv_out.views
        max_blocks = max((len(view.page_idxs) for view in views), default=1)
        if graph_mode:
            max_blocks = max(
                max_blocks,
                math.ceil(
                    self._kv_config.max_seq_len / self._kv_config.page_size
                ),
            )
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
            host_kv_lens=torch.tensor(
                kv_lens,
                dtype=torch.int32,
                device=self._device if graph_mode else None,
            ),
            max_q=max((view.to_compute for view in views), default=0),
            max_k=(
                self._kv_config.max_seq_len
                if graph_mode
                else max(kv_lens, default=0)
            ),
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
