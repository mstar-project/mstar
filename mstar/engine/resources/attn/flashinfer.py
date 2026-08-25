"""Paged attention through FlashInfer's prefill/decode wrappers."""

import torch

from mstar.engine.resources.attn.base import (
    AttentionManager,
    AttentionWrapper,
    WorkspacePool,
    _register_attender,
)
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.step import SlotLease, StepContext


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
        self._attend_handle = _register_attender(self)

        # label to plan state
        self._current_plan_states: dict[str, AttentionWrapper] = {}

        # (label, is_decode) -> wrapper; persistent, because constructing one
        # allocates FlashInfer's own workspaces and is far from free per step
        self._eager_plan_states: dict[tuple[str, bool], AttentionWrapper] = {}
        self._cg_plan_states: dict[CGSlotKey, AttentionWrapper] = {}

        self._preplan_states: dict[str, AttentionWrapper] = {}
        self._preplanned = False

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

        self._workspaces = WorkspacePool(device)

    def depends_on(self):
        return {self._kv_cache_name}

    def _cg_wrapper(self, lease: SlotLease, label: str, num_rows: int) -> AttentionWrapper:
        """The captured-graph wrapper for one (bucket, slot, label).

        Built on the first plan for that key rather than up front: which labels
        a walk plans under is the step's to declare, and the first plan for a
        bucket runs during capture, before the graph is recorded, so the wrapper
        is just as persistent either way.

        ``num_rows`` is the qo_indptr row count this label actually plans —
        the FlashInfer batch_size for CUDA-graph mode, which must stay fixed
        for the wrapper's lifetime. It equals ``bucket.bs`` (one row per
        request) for an ordinary label, but a label that combines several
        source labels into one plan (e.g. cond+uncond batched CFG) always
        contributes several rows per request; using ``bucket.bs`` there
        under-allocates the wrapper's fixed buffers and every later plan()
        call overruns them with a "runtime batch size" mismatch.
        """
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        wrapper = self._cg_plan_states.get(key)
        if wrapper is not None:
            return wrapper

        bucket = lease.bucket
        buffer = self._workspaces.get(label, lease.slot)
        if bucket.bs == bucket.num_tokens:
            wrapper = FlashInferDecodeWrapper(
                workspace_buffer=buffer,
                batch_size=num_rows,
                use_cuda_graph=True,
                **self._wrapper_kv_kwargs,
            )
        else:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=buffer,
                batch_size=num_rows,
                max_total_tokens=bucket.num_tokens,
                use_cuda_graph=True,
                **self._wrapper_kv_kwargs,
            )
        self._cg_plan_states[key] = wrapper
        return wrapper

    def _eager_wrapper(self, label: str, is_decode: bool) -> AttentionWrapper:
        """The persistent eager wrapper for one (label, kind)."""
        key = (label, is_decode)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            cls = FlashInferDecodeWrapper if is_decode else FlashInferPrefillWrapper
            wrapper = self._eager_plan_states[key] = cls(
                workspace_buffer=self._workspaces.get(label),
                **self._wrapper_kv_kwargs,
            )
        return wrapper

    @property
    def supports_preplan(self):
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        assert not ctx.is_preplan or ctx.slot_lease is not None, (
            "preplan requires a cuda graph step: eager wrappers share one "
            "workspace per label with the forward still in flight"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            # the wrappers were planned a step early against this same step's
            # KV plan; nothing left to do but promote them
            self._current_plan_states = self._preplan_states
            self._preplan_states = {}
            self._preplanned = False
            return

        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, f"Attention Manager expected plan result from {self._kv_cache_name}"

        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states

        plan_states.clear()
        for label, kv_out in plan_outputs.items():
            indptrs = kv_out.cpu_indptrs
            if ctx.slot_lease is not None:
                num_rows = indptrs.qo_indptr.shape[0] - 1
                wrapper = self._cg_wrapper(ctx.slot_lease, label, num_rows)
            else:
                is_decode = bool(
                    indptrs.qo_indptr[-1] == len(indptrs.qo_indptr) - 1
                )
                wrapper = self._eager_wrapper(label, is_decode)

            # TODO: cache the latest plan state
            wrapper.plan(
                causal=step.causal,
                dtype=self._dtype,
                **indptrs.to_kwargs_dict()
            )
            plan_states[label] = wrapper

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    ### Submodule-level functionality
    def qo_indptr_buf(self, label: str="main") -> torch.Tensor | None:
        if label not in self._current_plan_states:
            return
        # decode wrappers never set one
        return getattr(self._current_plan_states[label], "_qo_indptr_buf", None)

    def select_last_hidden(
        self, hidden: torch.Tensor, label: str="main"
    ) -> torch.Tensor:
        """
        Select last token of the hidden vector per request, used for sampling
        from prefill.
        """
        qo_indptr_buf = self.qo_indptr_buf(label)
        last_token_indices = (qo_indptr_buf[1:] - 1).long()
        return hidden.index_select(0, last_token_indices)

    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # `k`/`v`/`layer_idx` are the dense backend's; here the K/V is already
        # in the pages (the layer wrote it through the KV resource) and the
        # planned wrapper carries the layout. Accepted and ignored so one
        # layer body serves either backend — see `requires_kv_write`.
        del k, v, layer_idx
        if label is None:
            label = self._default_label
        return torch.ops.mstar.flashinfer_attend(
            self._attend_handle, label, q, kv_cache_layer,
        )

    def attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        """The eager body of ``run``, called from inside the op."""
        return self._current_plan_states[label].run(q, kv_cache_layer)
