"""Paged attention through FlashInfer's prefill/decode wrappers."""

from collections.abc import Mapping

import torch

from mstar.engine.resources.attn.base import (
    AttentionManager,
    AttentionWrapper,
    EagerSlotKey,
    WorkspacePool,
    captured_page_capacity,
)
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferMixedWrapper,
    FlashInferPrefillWrapper,
    FmhaDecodeWrapper,
    check_flashinfer_head_dim,
    fmha_decode_applies,
)
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import KVPlanOutputs, PagedIndptrs, build_paged_indptrs, hide_tail_views
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import SlotLease, StepContext
from mstar.utils.profiler import PhaseLaps

# Under device rewind a verify block (at most this many tokens a row) plans FA2, whose lengths the
# device can move (`FlashInferPrefillWrapper.correct_kv_lens`); FA3 keeps them in its plan. FA2
# also reads a row's KV once a KV head, not once a query head: 0.5-0.9x FA3's time at these rows.
_SHORT_BLOCK = 16


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
        decode_fmha: bool = False,
        layers: tuple[int, ...] | None = None,
    ):
        self._kv_cache_name = kv_cache
        self._splits = {split.combined_walk: split for split in splits}
        self._max_rows_in_combined = max_rows_in_combined or {}
        self._device = device
        self._dtype = dtype
        # FlashInfer counts preceding positions; M* counts the current token too.
        self._window_left = -1 if sliding_window is None else sliding_window - 1
        # The index of the group of the pool's layers this reads (`PagedKVConfig.groups`): the
        # group of the layers it declares, else the first, the layers no window is kept for.
        groups = kv_config.groups
        self._group_idx = next(g for g, group in enumerate(groups) if layers[0] in group.layers) if layers else 0
        self._group = groups[self._group_idx]
        # the layers it runs, where the pool's layers are in more than one group: the ones it
        # declares, or the first group's
        self._layers = None
        if len(groups) > 1:
            self._layers = frozenset(layers or self._group.layers)

        # label to plan state
        self._current_plan_states: dict[str, AttentionWrapper] = {}

        # persistent, because constructing one allocates FlashInfer's own
        # workspaces and is far from free per step
        self._eager_plan_states: dict[EagerSlotKey, AttentionWrapper] = {}
        self._cg_plan_states: dict[CGSlotKey, AttentionWrapper] = {}

        self._preplan_states: dict[str, AttentionWrapper] = {}
        self._preplanned = False

        self._kv_config = kv_config
        check_flashinfer_head_dim(kv_config.head_dim, device)     # at load, not at the first plan
        self._wrapper_kv_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            num_kv_heads=kv_config.num_kv_heads,
            head_dim=kv_config.head_dim,
            page_size=self._group.page_size,
            max_num_pages=kv_config.num_units,
            device=device,
            backend=backend
        )
        self._decode_tensor_cores = decode_tensor_cores
        self._decode_fmha = (
            decode_fmha and dtype == torch.bfloat16 and sliding_window is None
            and fmha_decode_applies(device)
        )

        self._workspaces = WorkspacePool(device)

    def depends_on(self):
        return {self._kv_cache_name}

    def _check_layer(self, layer_idx: int) -> None:
        """Over a pool of more than one group, ``layer_idx`` must be a layer this attention runs,
        since its plans address only that group's pages. Checked in Python, so at eager and
        capture time only: warmup and capture raise a miswired layer at startup."""
        if self._layers is not None and layer_idx not in self._layers:
            raise ValueError(
                f"layer {layer_idx} of KV {self._kv_cache_name!r} isn't one this attention runs "
                f"({sorted(self._layers)})"
            )

    @torch.compiler.disable
    def set_default_layer_idx(self, layer_idx: int) -> None:
        self._check_layer(layer_idx)
        self._default_layer_idx = layer_idx

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
            "max_num_pages": captured_page_capacity(num_rows, bucket.num_tokens, self._kv_config, self._group),
        }
        split = self._splits.get(bucket.graph_walk)
        if split is not None:
            # its own workspaces: both wrappers are planned for the one step
            decode_workspace = self._workspaces.get(f"{label}.split_decode", lease.slot)
            wrapper = FlashInferMixedWrapper(
                decode_workspace=decode_workspace,
                prefill_workspace=self._workspaces.get(f"{label}.split_prefill", lease.slot),
                batch_size=num_rows,
                max_total_tokens=bucket.num_tokens,
                max_prefill_rows=split.max_prefill_rows(
                    self._max_rows_in_combined, num_rows,
                ),
                use_tensor_cores=self._decode_tensor_cores,
                decode_wrapper=self._fmha_wrapper(decode_workspace, num_rows),
                **kv_kwargs,
            )
        elif bucket.bs == bucket.num_tokens and not force_prefill and not self.device_correct:
            # (under device rewind every step plans the prefill wrapper, whose rows' lengths
            # `correct_on_device` can move; the decode kernels split them by length)
            wrapper = self._fmha_wrapper(buffer, num_rows) or FlashInferDecodeWrapper(
                workspace_buffer=buffer,
                batch_size=num_rows,
                accelerator_graph=True,
                use_tensor_cores=self._decode_tensor_cores,
                **kv_kwargs,
            )
        else:
            kv_kwargs["backend"] = self._prefill_backend(bucket.num_tokens, num_rows)
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=buffer,
                batch_size=num_rows,
                max_total_tokens=bucket.num_tokens,
                accelerator_graph=True,
                **kv_kwargs,
            )
        self._cg_plan_states[key] = wrapper
        return wrapper

    def _fmha_wrapper(self, workspace: torch.Tensor, num_rows: int) -> FmhaDecodeWrapper | None:
        """The fmha decode wrapper for a capture at ``num_rows``, or None to
        stay on the paged decode kernel (not asked for, or none for this geometry)."""
        if not self._decode_fmha:
            return None
        config = self._kv_config
        wrapper = FmhaDecodeWrapper(
            workspace_buffer=workspace,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            page_size=self._group.page_size,
            batch_size=num_rows,
            max_seq_len=config.max_seq_len,
            device=self._device,
        )
        return wrapper if wrapper.has_kernel(config.head_dim, config.num_qo_heads) else None

    def _prefill_backend(self, tokens: int, rows: int) -> str:
        """The prefill wrapper's FlashInfer backend: FA2 for a short block under device rewind
        (`_SHORT_BLOCK`), else the configured one."""
        if self.device_correct and tokens <= _SHORT_BLOCK * rows:
            return "fa2"
        return self._wrapper_kv_kwargs["backend"]

    def _eager_wrapper(
        self, label: str, is_decode: bool, slot: int, backend: str | None = None,
    ) -> AttentionWrapper:
        """The persistent eager wrapper for one (label, kind, slot), and prefill backend.

        Takes the workspace this slot's captured wrappers already hold: the
        two never run at once, since the pipeline retires step N before the
        host stages N+2 and a slot repeats no sooner than that.
        """
        backend = backend or self._wrapper_kv_kwargs["backend"]
        key = EagerSlotKey(label=label, slot=slot, is_decode=is_decode, backend=backend)
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
                wrapper = FlashInferPrefillWrapper(
                    workspace_buffer=workspace, **{**self._wrapper_kv_kwargs, "backend": backend},
                )
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

    def _indptrs(self, kv_out, step: AttentionStep) -> PagedIndptrs:
        """The index arrays a label plans from: the KV plan's, or with ``hide_last`` its
        views' keys short of each row's last rows."""
        if not step.hide_last:
            return kv_out.cpu_indptrs
        page_size = self._group.page_size
        return build_paged_indptrs(hide_tail_views(kv_out.views, step.hide_last, page_size), page_size)

    def plan(self, step: AttentionStep, ctx: StepContext):
        if step.context_only:
            raise ValueError(
                f"{type(self).__name__} does not read AttentionStep.context_only; only the "
                "flashinfer_mla backend does"
            )
        if step.hide_last and step.causal:
            raise ValueError("AttentionStep.hide_last is for non-causal attention")
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
        if self._group_idx:
            plan_outputs = plan_outputs.groups[self._group_idx]

        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states

        plan_states.clear()
        laps = PhaseLaps(f"{ctx.plan_phase}.attn.")
        for label, kv_out in plan_outputs.items():
            indptrs = self._indptrs(kv_out, step)
            if lease is not None:
                num_rows = indptrs.qo_indptr.shape[0] - 1
                wrapper = self._cg_wrapper(lease, label, num_rows, ctx.force_prefill)
            else:
                rows = len(indptrs.qo_indptr) - 1
                tokens = int(indptrs.qo_indptr[-1])
                is_decode = not ctx.force_prefill and not self.device_correct and tokens == rows
                wrapper = self._eager_wrapper(
                    label, is_decode, ctx.slot, None if is_decode else self._prefill_backend(tokens, rows),
                )

            kwargs = indptrs.to_kwargs_dict()
            if (lease is None and self.device_correct
                    and getattr(getattr(wrapper, "attn_wrapper", None), "_backend", None) == "fa2"):
                # an eager plan, whose split arrays no graph re-plans (`correct_kv_lens`): one KV
                # chunk a row, so a row the device shortens keeps its chunk count
                kwargs["disable_split_kv"] = True
            laps.lap("wrapper")
            # TODO: cache the latest plan state
            wrapper.plan(
                causal=step.causal, dtype=self._dtype, window_left=self._window_left,
                **kwargs,
            )
            if self.device_correct:
                wrapper.locate_kv_lens()
            laps.lap("wrapper_plan")
            plan_states[label] = wrapper

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    corrects_preplan = True

    def correct_preplan(self, verdicts, step: AttentionStep, ctx: StepContext) -> bool:
        """Each staged wrapper with a rewound row, put right from the corrected KV plan: under
        FA3 by rewriting its rows' lengths in the plan (`patch_kv_lens`), else planned again."""
        if not self._preplanned:
            return True
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        if self._group_idx:
            plan_outputs = plan_outputs.groups[self._group_idx]
        rewound = set(zip(verdicts.request_ids, verdicts.labels, strict=True))
        for label, wrapper in self._preplan_states.items():
            kv_out = plan_outputs[label]
            rows = kv_out.rows
            if rows is None:
                rows = [(v.request_id, v.label, v.to_compute) for v in kv_out.views]
            if not any((rid, lab) in rewound for rid, lab, _ in rows):
                continue
            indptrs = self._indptrs(kv_out, step)
            if isinstance(wrapper, FlashInferPrefillWrapper) and (
                wrapper.planned_for(indptrs.kv_lens) or wrapper.patch_kv_lens(indptrs.kv_lens)
            ):
                continue
            wrapper.plan(
                causal=step.causal, dtype=self._dtype, window_left=self._window_left,
                **indptrs.to_kwargs_dict(),
            )
        return True

    @property
    def corrects_on_device(self) -> bool:
        # a short block's FA2 prefill kernel reads each row's KV length on the device, from its
        # last page's length (`FlashInferPrefillWrapper.correct_kv_lens`)
        return self._device.type == "cuda" and self._wrapper_kv_kwargs["backend"] in ("auto", "fa2")

    @torch.compiler.disable
    def correct_on_device(self, rejected) -> None:
        """Each row's keys end ``rejected`` rows earlier than planned."""
        for wrapper in self._current_plan_states.values():
            wrapper.correct_kv_lens(rejected.rows)

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
        if label not in self._current_plan_states:
            raise RuntimeError(f"FlashInfer: no plan for label {label!r} to select last tokens by")
        qo_indptr_buf = self.qo_indptr_buf(label)
        if qo_indptr_buf is None:
            # planned as a decode: one token per row, so every token is a row's last
            return hidden
        last_token_indices = (qo_indptr_buf[1:] - 1).long()
        return hidden.index_select(0, last_token_indices)

    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # `k`/`v` are the dense backend's; here the K/V is already in the pages
        # (the layer wrote it through the KV resource) and the planned wrapper
        # carries the layout. Accepted and ignored so one layer body serves
        # either backend — see `requires_kv_write`. `layer_idx` matters only over
        # a pool of more than one group (`set_default_layer_idx`).
        del k, v
        if label is None:
            label = self._default_label
        return self._attend(q, label, kv_cache_layer, layer_idx)

    @torch.compiler.disable
    def _attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor, layer_idx: int | None = None,
    ) -> torch.Tensor:
        if self._layers is not None:
            # an explicit layer over the cursor, checked as the cursor is
            if layer_idx is not None:
                self._check_layer(layer_idx)
            elif self._default_layer_idx is None:
                raise ValueError(
                    f"KV {self._kv_cache_name!r} has more than one group, so its attention needs the "
                    "layer it runs: set_layer_idx, or run(layer_idx=)"
                )
        o = self._current_plan_states[label].run(q, kv_cache_layer)
        if o.dtype != q.dtype:
            o = o.to(q.dtype)
        return o
