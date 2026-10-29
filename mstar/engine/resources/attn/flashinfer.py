"""Paged attention through FlashInfer's prefill/decode wrappers."""

import functools
from collections.abc import Mapping

import torch

from mstar.engine.resources.attn.base import (
    AttentionManager,
    AttentionWrapper,
    EagerSlotKey,
    WorkspacePool,
    captured_page_capacity,
)
from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferMixedWrapper,
    FlashInferPrefillWrapper,
    check_flashinfer_head_dim,
)
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.kv.rows import decode_indptrs
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import SlotLease, StepContext
from mstar.engine.rows import RowFills
from mstar.utils.profiler import PhaseLaps


class FlashInferManager(AttentionManager):
    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: PagedKVConfig,
        backend: str="auto",
        splits: tuple[CombinedSplit, ...] = (),
        max_rows_in_combined: Mapping[str, int | None] | None = None,
        decode_tensor_cores: bool = True,
        sliding_window: int | None = None,
    ):
        self._kv_cache_name = kv_cache
        self._splits = {split.combined_walk: split for split in splits}
        self._max_rows_in_combined = max_rows_in_combined or {}
        self._device = device
        self._dtype = dtype
        # FlashInfer counts preceding positions; M* counts the current token too.
        self._window_left = -1 if sliding_window is None else sliding_window - 1

        # label to plan state
        self._current_plan_states: dict[str, AttentionWrapper] = {}

        # persistent, because constructing one allocates FlashInfer's own
        # workspaces and is far from free per step
        self._eager_plan_states: dict[EagerSlotKey, AttentionWrapper] = {}
        self._cg_plan_states: dict[CGSlotKey, AttentionWrapper] = {}

        self._preplan_states: dict[str, AttentionWrapper] = {}
        self._preplanned = False
        # persistent walk -> its graph's fills of the wrappers' arrays
        self._row_fills: dict[str, RowFills] = {}

        self._kv_config = kv_config
        check_flashinfer_head_dim(kv_config.head_dim, device)     # at load, not at the first plan
        self._wrapper_kv_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            num_kv_heads=kv_config.num_kv_heads,
            head_dim=kv_config.head_dim,
            page_size=kv_config.page_size,
            max_num_pages=kv_config.max_num_pages,
            device=device,
            backend=backend
        )
        self._decode_tensor_cores = decode_tensor_cores

        self._workspaces = WorkspacePool(device)

    def depends_on(self):
        return {self._kv_cache_name}

    def _cg_wrapper(
        self, lease: SlotLease, label: str, num_rows: int, force_prefill: bool,
    ) -> AttentionWrapper:
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
        kv_kwargs = {
            **self._wrapper_kv_kwargs,
            "max_num_pages": captured_page_capacity(num_rows, bucket.num_tokens, self._kv_config),
        }
        split = self._splits.get(bucket.graph_walk)
        if split is not None:
            # its own workspaces: both wrappers are planned for the one step
            wrapper = FlashInferMixedWrapper(
                decode_workspace=self._workspaces.get(f"{label}.split_decode", lease.slot),
                prefill_workspace=self._workspaces.get(f"{label}.split_prefill", lease.slot),
                batch_size=num_rows,
                max_total_tokens=bucket.num_tokens,
                max_prefill_rows=split.max_prefill_rows(
                    self._max_rows_in_combined, num_rows,
                ),
                use_tensor_cores=self._decode_tensor_cores,
                **kv_kwargs,
            )
        elif bucket.bs == bucket.num_tokens and not force_prefill:
            wrapper = FlashInferDecodeWrapper(
                workspace_buffer=buffer,
                batch_size=num_rows,
                accelerator_graph=True,
                use_tensor_cores=self._decode_tensor_cores,
                **kv_kwargs,
            )
        else:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=buffer,
                batch_size=num_rows,
                max_total_tokens=bucket.num_tokens,
                accelerator_graph=True,
                **kv_kwargs,
            )
        self._cg_plan_states[key] = wrapper
        return wrapper

    def _eager_wrapper(
        self, label: str, is_decode: bool, slot: int
    ) -> AttentionWrapper:
        """The persistent eager wrapper for one (label, kind, slot).

        Takes the workspace this slot's captured wrappers already hold: the
        two never run at once, since the pipeline retires step N before the
        host stages N+2 and a slot repeats no sooner than that.
        """
        key = EagerSlotKey(label=label, slot=slot, is_decode=is_decode)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            workspace = self._workspaces.get(label, slot)
            if is_decode:
                wrapper = FlashInferDecodeWrapper(
                    workspace_buffer=workspace,
                    use_tensor_cores=self._decode_tensor_cores,
                    **self._wrapper_kv_kwargs,
                )
            else:
                wrapper = FlashInferPrefillWrapper(workspace_buffer=workspace, **self._wrapper_kv_kwargs)
            self._eager_plan_states[key] = wrapper
        return wrapper

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # FlashInfer's plan stages the schedule into a pinned buffer it holds
        # per wrapper and H2Ds it on the stream (scheduler.cuh's
        # cudaMemcpyAsync out of page_locked_int_workspace_buffer), so one
        # wrapper per slot is what keeps plan(N+1) off step N's queued copy.
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        refuse_context_only(step, type(self).__name__)
        if self._window_left >= 0 and not step.causal:
            raise ValueError("sliding_window requires causal attention")
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a captured graph step: an eager wrapper shares its "
            "workspace with the captured one on the same slot"
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
        laps = PhaseLaps(f"{ctx.plan_phase}.attn.")
        for label, kv_out in plan_outputs.items():
            indptrs = kv_out.cpu_indptrs
            if kv_out.device_rows is not None and lease is not None:
                plan_states[label] = self._plan_rows(label, kv_out, lease, ctx, step.causal)
                laps.lap("rows")
                continue
            if lease is not None:
                num_rows = indptrs.qo_indptr.shape[0] - 1
                wrapper = self._cg_wrapper(lease, label, num_rows, ctx.force_prefill)
            else:
                is_decode = not ctx.force_prefill and bool(
                    indptrs.qo_indptr[-1] == len(indptrs.qo_indptr) - 1
                )
                wrapper = self._eager_wrapper(label, is_decode, ctx.slot)

            kwargs = indptrs.to_kwargs_dict()
            laps.lap("wrapper")
            # TODO: cache the latest plan state
            wrapper.plan(
                causal=step.causal, dtype=self._dtype, window_left=self._window_left,
                **kwargs,
            )
            laps.lap("wrapper_plan")
            plan_states[label] = wrapper

        self._preplanned = ctx.is_preplan

    def row_state(self, space, node_name: str, graph_walk: str) -> RowFills:
        # a step of the walk reads its arrays off the KV rows (`_plan_rows`)
        fills = self._row_fills[graph_walk] = RowFills()
        return fills

    def _plan_rows(
        self, label: str, kv_out, lease: SlotLease, ctx: StepContext, causal: bool,
    ) -> FlashInferDecodeWrapper:
        """A step of a persistent walk: its graph builds the wrapper's paged
        arrays from the KV rows before the forward, so the host runs only
        FlashInfer's planner, on the host indptr and lengths the KV plan made
        off the rows' mirror. The capture plans the wrapper in full first,
        which settles the planner's other arguments."""
        indptrs = kv_out.cpu_indptrs
        num_rows = indptrs.qo_indptr.shape[0] - 1
        wrapper = self._cg_wrapper(lease, label, num_rows, ctx.force_prefill)
        if not isinstance(wrapper, FlashInferDecodeWrapper) or self._window_left >= 0:
            raise ValueError(
                f"attention {label!r} on persistent walk {ctx.graph_walk!r} needs a "
                "plain decode wrapper (one token a row, no sliding window)"
            )
        if ctx.capture:
            wrapper.plan(causal=causal, dtype=self._dtype, **indptrs.to_kwargs_dict())
            self._row_fills[ctx.graph_walk].arm(lease, label, functools.partial(
                decode_indptrs, kv_out.device_rows, num_rows, wrapper._paged_kv_indptr_buf,
                wrapper._paged_kv_indices_buf, wrapper._paged_kv_last_page_len_buf,
            ))
        else:
            wrapper.plan_schedule(indptrs.paged_kv_indptr, indptrs.kv_lens)
        return wrapper

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    ### Submodule-level functionality
    @torch.compiler.disable
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
        return self._attend(q, label, kv_cache_layer)

    @torch.compiler.disable
    def _attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        o = self._current_plan_states[label].run(q, kv_cache_layer)
        if o.dtype != q.dtype:
            o = o.to(q.dtype)
        return o
