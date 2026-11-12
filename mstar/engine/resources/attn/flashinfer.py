"""Paged attention through FlashInfer's prefill/decode wrappers."""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from mstar.engine.resources.attn.base import (
    AttentionWrapper,
    EagerSlotKey,
    PlacedAttention,
    captured_page_capacity,
    one_token_rows,
)
from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.attn.workspace import Workspace
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferMixedWrapper,
    FlashInferPrefillWrapper,
    FmhaDecodeWrapper,
    check_flashinfer_head_dim,
    fmha_decode_applies,
)
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.context_parallel import KeyRows, OwnTokensKernel, SPPlacement
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.resident_hooks import ANY_REPEATS
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import SlotLease, StepContext
from mstar.platforms import current_platform
from mstar.utils.profiler import PhaseLaps

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mstar.engine.resources.context_parallel import ShareCapture
    from mstar.engine.resources.resident import CapturedResidentStep


class FlashInferManager(PlacedAttention):
    supports_split_steps = True
    supports_drop = True

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
        sm_scale: float | None = None,
    ):
        super().__init__(kv_cache, device, dtype, kv_config)
        self._splits = {split.combined_walk: split for split in splits}
        self._max_rows_in_combined = max_rows_in_combined or {}
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

        # persistent, because constructing one allocates FlashInfer's own
        # workspaces and is far from free per step
        self._eager_plan_states: dict[EagerSlotKey, AttentionWrapper] = {}
        self._cg_plan_states: dict[CGSlotKey, AttentionWrapper] = {}

        check_flashinfer_head_dim(kv_config.head_dim, device)     # at load, not at the first plan
        self._wrapper_kv_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            num_kv_heads=kv_config.num_kv_heads,
            head_dim=kv_config.head_dim,
            page_size=self._group.page_size,
            max_num_pages=kv_config.num_units,
            device=device,
            backend=backend,
            sm_scale=sm_scale,
        )
        self._decode_tensor_cores = decode_tensor_cores
        self._decode_fmha = (
            decode_fmha and dtype == torch.bfloat16 and sliding_window is None
            and fmha_decode_applies(device)
        )
        self._sm_scale = sm_scale

        # walk -> whether its captured decode kernels see a row's length per replay
        self._repeat_walks: dict[str, bool] = {}

    def _bind_placement(self, geometry: dict[str, int]) -> None:
        if geometry["num_qo_heads"] != self._kv_config.num_qo_heads and not self._decode_tensor_cores:
            # a rank attending every partner's query heads over its one KV head (TP placement) forms
            # a GQA group FlashInfer's CUDA-core decode can have no kernel for
            self._decode_tensor_cores = True
            logger.info(
                "attention over context-parallel KV pool %r decodes on tensor cores although "
                "decode_tensor_cores is off: a TP-placed rank's query heads over its one KV head can form "
                "a group FlashInfer's CUDA-core decode has no kernel for", self._kv_cache_name,
            )
        if self._decode_fmha:
            logger.info(
                "attention over context-parallel KV pool %r decodes on FlashInfer's paged kernel although "
                "decode_fmha is on: the merge needs each partial's lse", self._kv_cache_name,
            )
        over_sp = isinstance(self._placement, SPPlacement)
        capability = current_platform.cuda_capability(self._device)
        # FlashInfer has fa3 for SM90 only (`_check_window_pool`)
        fa3 = capability is not None and capability[0] == 9
        if capability is not None and capability >= (10, 0) and self._wrapper_kv_kwargs["backend"] == "auto":
            # from SM100 auto can send short rows to cuDNN, whose lse the merge isn't known to
            # read right (`_check_window_pool` resolves it the same way)
            self._wrapper_kv_kwargs["backend"] = "fa2"
            logger.info(
                "attention over context-parallel KV pool %r plans on fa2 for flashinfer_backend 'auto'",
                self._kv_cache_name,
            )
        # one SP-placed server hung at its first 32k prefill on fa2 and the cause isn't known (#1363)
        backend = self._wrapper_kv_kwargs["backend"]
        # an explicit fa2 on any GPU; `auto` where the capability says FlashInfer has no fa3
        on_fa2 = (backend == "fa2" and self._device.type == "cuda") or (
            backend == "auto" and capability is not None and not fa3
        )
        if over_sp and on_fa2:
            logger.warning(
                "attention over SP-placed KV pool %r prefills on FlashInfer's fa2 kernel; an unexplained one-off "
                "lockstep hang in an fa2 prefill is still open (#1363)",
                self._kv_cache_name,
            )
        self._wrapper_kv_kwargs.update(geometry)
        cfg = self._kv_config
        self._new_kwargs = dict(
            num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
            device=self._device, backend=self._wrapper_kv_kwargs["backend"], q_data_type=self._dtype,
            sm_scale=self._sm_scale,
        )

    def max_repeats(self, step, ctx) -> int:
        """Whether every wrapper of the walk's captures sees each replay's lengths
        (`supports_repeats`)."""
        walk = ctx.graph_walk
        allowed = self._repeat_walks.get(walk)
        if allowed is None:
            wrappers = [w for key, w in self._cg_plan_states.items() if key.bucket.graph_walk == walk]
            allowed = self._repeat_walks[walk] = bool(wrappers) and all(w.supports_repeats for w in wrappers)
        return ANY_REPEATS if allowed else 1

    def resident_refusal(self) -> str | None:
        if self._window_left >= 0:
            return "a sliding window reads pages a resident row no longer holds"
        return None

    def resident_gather(self, step: "CapturedResidentStep") -> None:
        # planned per label of its KV cache's plan, so its wrapper is under that stream's label
        pool, stream = self._kv_cache_name, step.streams[self._kv_cache_name]
        wrapper = self._cg_plan_states.get(CGSlotKey(bucket=step.bucket, slot=step.slot, label=stream.label))
        if wrapper is not None and wrapper.plans_in_graph and not stream.reads:
            table = step.block_tables[pool]
            wrapper.gather_resident(step.store, step.slots, step.buffers.kv_len[pool], table.stride, table.col)
            step.reads_tables = True
            return
        step.host_planned = True
        if wrapper is not None and wrapper.supports_gather_resident and not stream.reads:
            # the paged arrays come off the tables; the host plans the schedule alone
            table = step.block_tables[pool]
            wrapper.gather_resident(step.store, step.slots, step.buffers.kv_len[pool], table.stride, table.col)
            step.lengths_only.add(pool)

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

    def fingerprint(self) -> bytes:
        if self._window_left == -1 and self._sm_scale is None:
            return super().fingerprint()
        # the window and scale shape every later layer's K/V
        return fingerprint(type(self).__name__, self._window_left, self._sm_scale)

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
        workspace = self._workspaces.get(label, lease.slot)
        kv_kwargs = {
            **self._wrapper_kv_kwargs,
            "max_num_pages": captured_page_capacity(num_rows, bucket.num_tokens, self._kv_config, self._group),
        }
        split = self._splits.get(bucket.graph_walk)
        if self._placement is not None:
            # the mixed wrapper is not used: the context wrapper takes rows of any length
            wrapper = self._context_parallel_wrapper(
                label, lease.slot, one_token_rows(bucket.bs, bucket.num_tokens, force_prefill),
                dict(batch_size=num_rows, accelerator_graph=True, **kv_kwargs), bucket.num_tokens,
            )
        elif split is not None:
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
        elif one_token_rows(bucket.bs, bucket.num_tokens, force_prefill):
            wrapper = self._fmha_wrapper(workspace, num_rows) or FlashInferDecodeWrapper(
                workspace_buffer=workspace,
                batch_size=num_rows,
                accelerator_graph=True,
                use_tensor_cores=self._decode_tensor_cores,
                **kv_kwargs,
            )
        else:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=workspace,
                batch_size=num_rows,
                max_total_tokens=bucket.num_tokens,
                accelerator_graph=True,
                **kv_kwargs,
            )
        self._cg_plan_states[key] = wrapper
        return wrapper

    def _fmha_wrapper(self, workspace: Workspace, num_rows: int) -> FmhaDecodeWrapper | None:
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
            sm_scale=self._sm_scale,
        )
        return wrapper if wrapper.has_kernel(config.head_dim, config.num_qo_heads) else None

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
        if wrapper is None and self._placement is not None:
            wrapper = self._eager_plan_states[key] = self._context_parallel_wrapper(
                label, slot, is_decode, dict(self._wrapper_kv_kwargs),
            )
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

    def paged_kernel(self, workspace, decode, kwargs, max_total_tokens):
        if decode:
            # on tensor cores under TP placement (`_bind_placement`)
            return FlashInferDecodeWrapper(
                workspace_buffer=workspace, use_tensor_cores=self._decode_tensor_cores, **kwargs,
            )
        return FlashInferPrefillWrapper(workspace_buffer=workspace, max_total_tokens=max_total_tokens, **kwargs)

    def own_tokens_kernel(self, workspace, kwargs, captured, whole):
        if not whole:
            return super().own_tokens_kernel(workspace, kwargs, captured, whole)
        # whole rows read their keys as contiguous runs, which the ragged prefill takes 1.06-1.16x
        # faster than one-slot pages on a captured step (H200, FA2 and FA3)
        from mstar.engine.resources.attn.ragged.wrappers import FlashInferRaggedPrefillWrapper

        ragged = FlashInferRaggedPrefillWrapper(
            workspace, max_num_segments=kwargs.get("batch_size"), max_total_tokens=captured and captured.tokens,
            accelerator_graph=kwargs.get("accelerator_graph", False), owns_padding=True, **self._new_kwargs,
        )
        return _WholeRows(ragged, captured)

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
        if ctx.resident is not None and ctx.slot_lease.bucket in ctx.resident.host_plan_free:
            return
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
        self._check_sequence_shard(ctx)
        if self._group_idx:
            plan_outputs = plan_outputs.groups[self._group_idx]

        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states

        plan_states.clear()
        laps = PhaseLaps(f"{ctx.plan_phase}.attn.")
        for label, kv_out in plan_outputs.items():
            indptrs = kv_out.cpu_indptrs
            if lease is not None:
                num_rows = indptrs.qo_indptr.shape[0] - 1
                wrapper = self._cg_wrapper(lease, label, num_rows, ctx.force_prefill)
            else:
                is_decode = one_token_rows(len(indptrs.qo_indptr) - 1, int(indptrs.qo_indptr[-1]), ctx.force_prefill)
                wrapper = self._eager_wrapper(label, is_decode, ctx.slot)

            if self._placement is not None:
                laps.lap("wrapper")
                wrapper.plan(kv_out, causal=step.causal, real_rows=ctx.real_rows())
                laps.lap("wrapper_plan")
                plan_states[label] = wrapper
                continue
            # a placed pool never runs resident (`KVManager.resident_refusal`)
            if ctx.resident is not None and (lease.bucket, self._kv_cache_name) in ctx.resident.lengths_only:
                wrapper.plan_schedule(indptrs.paged_kv_indptr, indptrs.kv_lens)
                laps.lap("schedule")
                plan_states[label] = wrapper
                continue
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

    ### Submodule-level functionality
    @torch.compiler.disable
    def qo_indptr_buf(self, label: str="main") -> torch.Tensor | None:
        if label not in self._current_plan_states:
            return
        # decode wrappers never set one
        return getattr(self._current_plan_states[label], "_qo_indptr_buf", None)

    # Eager as a whole, as AttentionManager.select_last_hidden explains. FlashInfer's
    # own: a label planned as a decode has no query offsets, and a missing one is an error.
    @torch.compiler.disable
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
            # an eager step of one-token rows plans the decode wrapper, which
            # keeps no qo_indptr: each row is its own last token
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
        # The K/V is already in the pages (the layer wrote it through the KV
        # resource) and the planned wrapper carries the layout, so `k`/`v` are
        # read only over a placed pool, where a step's own tokens attend each
        # other from them. Accepted so one layer body serves either backend —
        # see `requires_kv_write`. `layer_idx` matters only over a pool of more
        # than one group (`set_default_layer_idx`).
        if label is None:
            label = self._default_label
        if self._placement is not None:
            return self._attend_placed(label, q, kv_cache_layer, k, v)
        return self._attend(q, label, kv_cache_layer, layer_idx)

    @torch.compiler.disable
    def _attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor, layer_idx: int | None = None,
    ) -> torch.Tensor:
        if self._shard is not None:
            self._shard.check_bypass(label, "attention run")
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


class _WholeRows(OwnTokensKernel):
    """FlashInfer's ragged prefill as the own-token kernel for whole rows, each over its own tokens
    (``KeyRows.whole``): their keys are contiguous runs of the step's, which it reads in place. It
    takes no other rows, whose keys would have to be gathered."""

    whole_rows_only = True

    def __init__(self, ragged, captured: "ShareCapture | None"):
        self._ragged, self.captured = ragged, captured

    def plan(self, rows: KeyRows, causal: bool) -> None:
        self._ragged.plan(torch.from_numpy(rows.qo_indptr.astype(np.int32)), causal=causal)

    def run(self, q, k, v):
        return self._ragged.run(q, k, v, return_lse=True)

    def forget_plan(self) -> None:
        # it skips a layout it already holds (`_RaggedPrefillBase.forget_plan`)
        self._ragged.forget_plan()
