import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from mstar.engine.kv_store import (
    CrossAttnPool,
    KVCacheConfig,
    KVRequestState,
    PagedAllocationManager,
)
from mstar.engine.resources import KVCachePool, RopeEmbedder, Segment
from mstar.engine.resources.attention import (
    CrossAttentionManager,
    DenseGenAttentionManager,
    FlashInferAttentionManager,
    PagedIndptrs,
    PlanCacheKey,
    WorkspaceBufferManager,
    _PlanState,
    build_paged_indptrs,
    cross_attn_label,
)

__all__ = [
    "ATTENTION_BACKENDS",
    "BatchedCacheManager",
    "BatchedCfgInfo",
    "DenseGenCacheManager",
    "FlashInferCacheManager",
    "PagedIndptrs",
    "PlanCacheKey",
    "WorkspaceBufferManager",
    "_PlanState",
    "build_paged_indptrs",
    "create_cache_manager",
    "cross_attn_label",
]

logger = logging.getLogger(__name__)


@dataclass
class BatchedCfgInfo:
    per_label_seq_len: dict[str, list[int]]


class BatchedCacheManager(ABC):
    """Internal: the runner's per-step surface over the engine's resources.

    Holds the step's addressing (which requests, which label each is on)
    and dispatches every call into the resources behind it: the KV cache
    pool (admission, views, commits, forks), the attention manager (plans,
    wrappers, workspaces), the positional embedder, and one cross-attention
    manager per declared source. The domain state lives on those resources;
    what stays here is per-step bookkeeping (active labels, the batched-CFG
    advance info, the pre-plan short-circuit set).

    Adopted models (cosmos3, qwen3_omni, bagel) never sequence through this
    surface: they declare their step (``NodeSubmodule.declare_step``), the
    runner drives the plans, forks, and commits, and their forwards call
    only the device-side ops with explicit labels (``run_attention``,
    ``apply_rope``). The plan and advance methods here are transitional
    surface for the families that have not adopted declarations yet; new
    model code declares instead of calling them.

    Concrete backends pick the attention manager kind via
    ``ATTENTION_MANAGER_CLS``; ``ATTENTION_BACKENDS`` maps
    ``KVCacheConfig.attention_backend`` names to backend classes and
    ``create_cache_manager`` instantiates the configured one.

    Constructed per batch: one facade serves the whole batch with a single
    attention call per layer instead of N per-request calls.
    """

    ATTENTION_MANAGER_CLS: type[FlashInferAttentionManager] = FlashInferAttentionManager

    def __init__(
        self,
        request_ids: list[str],
        active_labels_per_request: dict[str, str],
        kv_cache: torch.Tensor,
        alloc_manager: PagedAllocationManager,
        buffer_manager: WorkspaceBufferManager,
        kv_cache_config: KVCacheConfig,
        device,
        cuda_graph_plan_states: dict[str, _PlanState] | None = None,
        auto_write_store: bool=False,
        enable_nvtx: bool=False,
        cross_pools: dict[str, CrossAttnPool] | None = None,
        kv_pool: KVCachePool | None = None,
        cross_kv_pools: dict[str, KVCachePool] | None = None,
        rope_embedder: RopeEmbedder | None = None,
    ):
        self.request_ids = request_ids
        self.active_labels = active_labels_per_request  # {req_id: label}
        self.kv_cache = kv_cache
        self.alloc_manager = alloc_manager
        self.buffer_manager = buffer_manager
        self.kv_cache_config = kv_cache_config
        self.device = device
        self.layer_idx = 0
        self.enable_nvtx = enable_nvtx
        # source name -> CrossAttnPool (see KVCacheConfig.cross_attn)
        self.cross_pools = cross_pools or {}

        # The resources this facade dispatches into. The engine passes its
        # persistent pools and embedder; callers that construct the facade
        # standalone (the graph runner, tests) get fresh fronts over the
        # same allocation manager, which hold no state of their own.
        self.kv_pool = (
            kv_pool if kv_pool is not None
            else KVCachePool(alloc_manager) if alloc_manager is not None
            else None
        )
        self.cross_kv_pools: dict[str, KVCachePool] = (
            cross_kv_pools if cross_kv_pools is not None
            else {
                source: KVCachePool(pool.alloc_manager)
                for source, pool in self.cross_pools.items()
            }
        )
        # Position semantics live on the embedder; the pool's counters are
        # read at plan time and advanced only through commit.
        self.rope_embedder = rope_embedder if rope_embedder is not None else RopeEmbedder()

        self.auto_write_store = auto_write_store

        # CUDA graph mode: persistent plan states passed in from
        # CudaGraphRunner. The attention manager plans onto them, so a
        # persistent wrapper's plan() updates static buffers via .copy_()
        # instead of creating a new wrapper each call.
        self._cuda_graph_mode = cuda_graph_plan_states is not None
        self.attention = self.ATTENTION_MANAGER_CLS(
            kv_cache=kv_cache,
            kv_cache_config=kv_cache_config,
            buffer_manager=buffer_manager,
            device=device,
            states=cuda_graph_plan_states,
            cuda_graph_mode=self._cuda_graph_mode,
            enable_nvtx=enable_nvtx,
        )
        # One manager per cross-attention source, sharing the per-step plan
        # store (cross plans live under the resolved cross label).
        self.cross_attention: dict[str, CrossAttentionManager] = {
            source: CrossAttentionManager(
                source=source,
                pool=pool,
                kv_pool=self.cross_kv_pools[source],
                buffer_manager=buffer_manager,
                device=device,
                states=self.attention.states,
                enable_nvtx=enable_nvtx,
            )
            for source, pool in self.cross_pools.items()
        }

        # Labels the Worker's plan_executor has pre-planned for the
        # next batch. Each entry causes the matching plan_attention(label=L)
        # call to short-circuit — the captured graph's preprocess skips the
        # heavy FlashInfer wrapper.plan() (GIL-contended with main thread's
        # fast/slow post in the speculative path). Populated by
        # CudaGraphRunner.pre_plan_for_batch with one entry per label in the
        # captured config; each entry is consumed by the matching
        # plan_attention call and removed.
        #
        # Set-of-labels (rather than single bool) is required for multi-label
        # captures (e.g. BAGEL CFG decode, labels=["main", "cfg_img"]): the
        # earlier single-bool primitive short-circuited whichever label ran
        # first regardless of whether that label was the one pre-planned, so
        # only one of N labels got the overlap benefit.
        self._pre_planned_labels: set[str] = set()
        # CUDA event recorded on the plan-overlap stream after the pre-planned
        # plan() calls completed. The captured graph's replay must wait on
        # this before reading any wrapper's static buffers (the pre-plan
        # wrote them on a different stream). One event covers all labels —
        # they're sequential on the same plan_stream, so the final event
        # signals "all label wrappers' writes visible". None when no
        # pre-plan was applied.
        self._plan_done_event: "torch.cuda.Event | None" = None

        self._batched_cfg_info: BatchedCfgInfo | None = None

    @property
    def _plan_states(self) -> dict[str, _PlanState]:
        """The per-label plan store, owned by the attention manager (the
        graph runner's per-slot store when one was passed in)."""
        return self.attention.states

    @property
    def is_captured(self) -> bool:
        """Whether this step surface fronts a captured graph's persistent
        per-slot plan store rather than a per-step eager one."""
        return self._cuda_graph_mode

    @torch.compiler.disable
    def _get_state(self, request_id: str, label: str | None = None) -> KVRequestState:
        label = label or self.active_labels.get(request_id, "main")
        return self.alloc_manager.get_state(request_id, label)

    def _active_label(self) -> str:
        """The single label all requests are currently on (asserts uniformity)."""
        labels = list(self.active_labels.values())
        assert len(set(labels)) == 1, f"All active labels must be the same, got {labels}"
        return labels[0]

    @torch.compiler.disable
    def set_active_labels(self, labels: dict[str, str]) -> None:
        """Switch active cache labels for all requests at once."""
        self.active_labels = labels

    @torch.compiler.disable
    def set_active_label(self, label: str) -> None:
        """Switch all requests to the same cache label."""
        self.active_labels = {rid: label for rid in self.request_ids}

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int):
        self.layer_idx = layer_idx

    @torch.compiler.disable
    def get_qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        """The attention manager's ``qo_indptr_buf`` for ``label`` (see
        ``FlashInferAttentionManager.qo_indptr_buf``)."""
        return self.attention.qo_indptr_buf(label)

    @abstractmethod
    def plan_attention(
        self,
        seq_lens: list[int] | None = None,
        dtype: torch.dtype | None = None,
        is_causal=True,
        write_store: bool=True,
        label: str | None = None,
        **kwargs,
    ):
        """Pre-compute the attention plan for a cache label.

        Backend-specific planning hints (e.g. ``DenseGenCacheManager``'s
        ``dense_gen``) pass through ``**kwargs``; backends ignore hints they
        don't implement.

        Args:
            seq_lens: number of new tokens per request.
            dtype: query data type.
            is_causal: whether attention is causal.
            write_store: whether run_attention may flush this label to store.
            label: cache label to plan for. If None, uses the current active label.
        """

    @abstractmethod
    def plan_attention_batched_cfg(
        self,
        labels: list[str],
        seq_lens: list[int] | dict[str, list[int]],
        is_causal: bool = False,
        write_store: bool = False,
        dtype=torch.bfloat16,
        combined_label: str = "_cfg_batched",
        **kwargs,
    ):
        """Plan a single attention batch across multiple cache labels.

        Each (label, request_id) pair becomes one entry in the batch. For
        3-branch CFG with 1 request this creates a 3-entry batch, enabling all
        CFG branches to execute in a single forward pass. Backend-specific
        planning hints pass through ``**kwargs`` as in ``plan_attention``.

        Args:
            labels: cache labels to batch (e.g. ["main", "cfg_text", "cfg_img"]).
            seq_lens: number of new tokens per request (same for all labels),
                or a per-label dict of such lists.
            is_causal: whether attention is causal.
            write_store: whether to write to mooncake store (False for image_gen).
            combined_label: key for the combined _PlanState.
        """

    @abstractmethod
    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int | None=None,
        label: str | None = None,
    ) -> torch.Tensor:
        """Run the pre-planned attention for a cache label.

        Args:
            q: [total_tokens, num_q_heads, head_dim]
            k: [total_tokens, num_kv_heads, head_dim]
            v: [total_tokens, num_kv_heads, head_dim]
            layer_idx: transformer layer index
            label: plan key to run (a cache label, or a combined key such
                as ``_cfg_batched``). If None, uses the active label.
        Returns:
            output: [total_tokens, num_q_heads, head_dim]
        """

    def plan_rope(
        self,
        seq_lens: list[int],
        pos_ids: torch.Tensor | None = None,
        label: str | None = None,
    ):
        """Pre-compute position IDs for RoPE for a cache label.

        In CUDA graph mode, updates the static pos_ids tensor via .copy_()
        so that the same GPU address is used during graph replay.

        Args:
            seq_lens: number of new tokens per request.
            pos_ids: explicit position IDs. If None, computed from
                each request's position_id_start.
            label: cache label. If None, uses the current active label.
        """
        from mstar.utils.profiler import range_pop, range_push

        if self.enable_nvtx:
            range_push("cache.plan_rope", synchronize=False)
        try:
            self._plan_rope_impl(seq_lens=seq_lens, pos_ids=pos_ids, label=label)
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def _plan_rope_impl(
        self,
        seq_lens: list[int],
        pos_ids: torch.Tensor | None = None,
        label: str | None = None,
    ):
        from mstar.utils.profiler import range_pop, range_push

        effective_label = label if label is not None else self._active_label()

        if effective_label not in self._plan_states:
            self._plan_states[effective_label] = _PlanState()
        ps = self._plan_states[effective_label]

        # Fast path: cuda-graph mode with the static pos_ids buffer already
        # allocated and no caller-supplied pos_ids. Build the position list on
        # CPU and copy straight into the static buffer — skipping the
        # intermediate device-side allocation + GPU→GPU copy the eager path
        # would do.
        static_copy_from_cpu = (
            self._cuda_graph_mode and ps.pos_ids is not None and pos_ids is None
        )

        computed_pos_ids = pos_ids
        if computed_pos_ids is None:
            # The embedder builds the position list on CPU (1 int per output
            # token); placement is decided here.
            if self.enable_nvtx:
                range_push("cache.plan_rope.build_pos_ids", synchronize=False)
            try:
                segments = [
                    Segment(rid, effective_label, sl)
                    for rid, sl in zip(self.request_ids, seq_lens, strict=True)
                ]
                position_plan = self.rope_embedder.plan(segments, self.kv_pool)
                computed_pos_ids = position_plan.pos_ids
                if not static_copy_from_cpu:
                    computed_pos_ids = computed_pos_ids.to(self.device)
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)

        if self._cuda_graph_mode:
            if ps.pos_ids is not None:
                n = computed_pos_ids.shape[0]
                if self.enable_nvtx:
                    range_push("cache.plan_rope.copy_pos_ids", synchronize=False)
                try:
                    # CPU→GPU when static_copy_from_cpu, else GPU→GPU. Both
                    # are stream-ordered before any subsequent graph replay.
                    ps.pos_ids[:n].copy_(computed_pos_ids, non_blocking=True)
                finally:
                    if self.enable_nvtx:
                        range_pop(synchronize=False)
            else:
                # First plan_rope on this label: adopt the just-built tensor
                # as the static buffer. Must live on the device.
                if computed_pos_ids.device != self.device:
                    computed_pos_ids = computed_pos_ids.to(self.device)
                ps.pos_ids = computed_pos_ids
        else:
            ps.pos_ids = computed_pos_ids

    @torch.compiler.disable
    def plan_rope_batched_cfg(
        self,
        labels: list[str],
        seq_lens: list[int] | dict[str, list[int]],
        per_label_pos_ids: dict[str, list[torch.Tensor]] | None = None,
        combined_label: str = "_cfg_batched",
    ):
        """Concatenate position IDs across multiple labels for batched CFG.

        Args:
            labels: cache labels in batch order.
            seq_lens: new tokens per request (same for all labels).
            per_label_pos_ids: {label: [pos_ids_tensor per request]}.
                If None or a label is missing, computed from position_id_start.
            combined_label: key for the combined _PlanState (must already exist
                from plan_attention_batched_cfg).
        """
        if isinstance(seq_lens, list):
            seq_lens = {
                key: seq_lens for key in labels
            }

        # Build one tensor *per label* in `labels` order, then concat. Order
        # matters: downstream attention indexes these positions by
        # (label_i * per_label_len + within_label_offset), so reordering the
        # labels silently corrupts the attention computation. For labels with
        # explicit pos_ids we concat their list as given; for computed labels
        # we accumulate Python ints on CPU and do one H2D per label.
        parts: list[torch.Tensor] = []
        for label in labels:
            if per_label_pos_ids and label in per_label_pos_ids:
                parts.append(torch.cat(per_label_pos_ids[label]))
            else:
                segments = [
                    Segment(rid, label, sl)
                    for rid, sl in zip(self.request_ids, seq_lens[label], strict=True)
                ]
                plan = self.rope_embedder.plan(segments, self.kv_pool)
                parts.append(plan.pos_ids.to(self.device))
        combined_pos_ids = parts[0] if len(parts) == 1 else torch.cat(parts)
        self._plan_states[combined_label].pos_ids = combined_pos_ids

    @torch.compiler.disable
    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rotary_dim: int | None = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 10000.0,
        rope_dtype=None,
        label: str | None = None,
        **kwargs
    ):
        """Apply RoPE using a label's pre-computed position IDs (the
        active label when ``label`` is None)."""
        if label is None:
            label = self._active_label()

        ps = self._plan_states[label]
        assert ps.pos_ids is not None

        return self.rope_embedder.apply(
            q, k, ps.pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            rope_dtype=rope_dtype,
            **kwargs,
        )

    @torch.compiler.disable
    def advance_seq_len(self, n: int | None = None, pos_id_n: int | None = None) -> None:
        """Advance seq_len for all requests.

        Uses provided n and/or pos_id_n if they exist, then falls back to
        per-request seq_lens from the last plan_attention call. Errors if
        both n and planned seq_lens are None.
        """
        if n is None:
            return self.advance_seq_lens(pos_id_n)
        for rid in self.request_ids:
            label = self.active_labels.get(rid, "main")
            self.kv_pool.commit(
                Segment(rid, label, n),
                pos_advance=pos_id_n if pos_id_n is not None else n,
            )

    @torch.compiler.disable
    def advance_seq_lens(self, pos_id_ns: list[int] | int | None = None) -> None:
        """Advance seq_len for each request by different amounts.

        ``pos_id_ns`` overrides the position-id advance for walks whose
        position span differs from seq_len; None advances positions by
        each request's planned seq_len.
        """

        if self._batched_cfg_info:
            for label, seq_lens in self._batched_cfg_info.per_label_seq_len.items():
                for i, rid in enumerate(self.request_ids):
                    n = seq_lens[i]
                    if pos_id_ns is None:
                        pos_advance = n
                    elif isinstance(pos_id_ns, int):
                        pos_advance = pos_id_ns
                    else:
                        pos_advance = pos_id_ns[i]
                    self.kv_pool.commit(
                        Segment(rid, label, n), pos_advance=pos_advance
                    )
        else:
            for i, rid in enumerate(self.request_ids):
                label = self.active_labels[rid]
                ps = self._plan_states[label]
                if ps.seq_lens is None:
                    continue
                n = ps.seq_lens[i]
                if pos_id_ns is None:
                    pos_advance = n
                elif isinstance(pos_id_ns, int):
                    pos_advance = pos_id_ns
                else:
                    pos_advance = pos_id_ns[i]
                self.kv_pool.commit(
                    Segment(rid, label, n), pos_advance=pos_advance
                )

    @torch.compiler.disable
    def snapshot_all(
        self, from_label: str,
        to_label: str,
        realloc: bool=False,
        write_store: bool=True
    ) -> None:
        """Snapshot KV cache for all requests in batch."""
        for rid in self.request_ids:
            self.kv_pool.fork(rid, from_label, to_label, realloc=realloc)
            if write_store:
                self.alloc_manager.flush_to_store(
                    rid, label=to_label
                )

    @torch.compiler.disable
    def flush_to_store(self):
        for rid in self.request_ids:
            for label in self.alloc_manager.request_states[rid]:
                ps = self._plan_states.get(label)
                if ps is None or not ps.write_store:
                    continue
                self.alloc_manager.flush_to_store(rid, label)


class FlashInferCacheManager(BatchedCacheManager):
    """Facade over the paged FlashInfer attention manager (the default).

    Admits one segment per request on the KV pool, hands the resulting
    sequence views to the attention manager, and issues a single FlashInfer
    call per layer instead of N per-request calls. K/V for every planned
    token is written to the paged cache.
    """

    ATTENTION_MANAGER_CLS = FlashInferAttentionManager

    def plan_attention(
        self,
        seq_lens: list[int] | None = None,
        dtype: torch.dtype | None = None,
        is_causal=True,
        write_store: bool=True,
        label: str | None = None,
        **kwargs,
    ):
        """Pre-compute FlashInfer plan and page positions for a cache label.

        Admits one segment per request on the KV pool (reserving pages) and
        has the attention manager plan against the resulting views. All
        state is stored in the per-label plan store.

        In CUDA graph mode, the manager plans onto the persistent wrapper
        from the store (pre-built by CudaGraphRunner), whose plan() updates
        static buffers via .copy_(). In eager mode, it creates a new wrapper
        each call.

        Planning hints for other backends arriving via **kwargs are ignored.
        """
        from mstar.utils.profiler import range_pop, range_push
        self._batched_cfg_info = None

        if self.enable_nvtx:
            range_push("cache.plan_attention", synchronize=False)
        try:
            effective_label = label if label is not None else self._active_label()
            if effective_label in self._pre_planned_labels:
                # Fast path: plan was pre-computed by Worker.plan_executor
                # against the same persistent wrapper. The wrapper's static
                # buffers and FlashInfer scheduling state are already correct
                # for this iter's seq_lens. We only need to record ps.seq_lens
                # / ps.write_store on the matching label so downstream
                # run_attention sees them.
                self._pre_planned_labels.discard(effective_label)
                ps = self._plan_states.get(effective_label)
                if ps is not None:
                    ps.seq_lens = seq_lens
                    ps.write_store = write_store
                if self.enable_nvtx:
                    range_push("cache.plan_attention.skipped_pre_planned", synchronize=False)
                    range_pop(synchronize=False)
                return
            assert self.kv_cache is not None
            segments = []
            views = []
            for i, rid in enumerate(self.request_ids):
                segment = Segment(rid, effective_label, seq_lens[i])
                self.kv_pool.admit(segment)
                segments.append(segment)
                views.append(self.kv_pool.view(segment))
            self.attention.plan(
                label=effective_label,
                segments=segments,
                views=views,
                dtype=dtype,
                is_causal=is_causal,
                write_store=write_store,
            )
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

    @torch.compiler.disable
    def plan_attention_batched_cfg(
        self,
        labels: list[str],
        seq_lens: list[int] | dict[str, list[int]],
        is_causal: bool = False,
        write_store: bool = False,
        dtype=torch.bfloat16,
        combined_label: str = "_cfg_batched",
        **kwargs,
    ):
        """Plan a single FlashInfer batch across multiple cache labels.

        Planning hints for other backends arriving via **kwargs are ignored.
        See ``BatchedCacheManager.plan_attention_batched_cfg`` for the argument
        contract.
        """
        assert self.kv_cache is not None
        if isinstance(seq_lens, list):
            seq_lens = {
                key: seq_lens for key in labels
            }

        self._batched_cfg_info = BatchedCfgInfo(
            per_label_seq_len=seq_lens
        )

        # Label-major segment order: every (label, request) pair in batch
        # order, the packed layout the CFG forward uses.
        segments = []
        views = []
        for label in labels:
            for i, rid in enumerate(self.request_ids):
                segment = Segment(rid, label, seq_lens[label][i])
                self.kv_pool.admit(segment)
                segments.append(segment)
                views.append(self.kv_pool.view(segment))

        self.attention.plan_batched_cfg(
            combined_label=combined_label,
            segments=segments,
            views=views,
            dtype=dtype,
            is_causal=is_causal,
            write_store=write_store,
        )

    @torch.compiler.disable
    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int | None=None,
        label: str | None = None,
    ) -> torch.Tensor:
        """Run pre-planned FlashInfer attention with KV cache write.

        Uses the given label's plan state (the active label when ``label``
        is None; set up by a prior plan_attention call). Writes K and V to
        the paged KV cache at pre-computed page positions, then runs the
        FlashInfer wrapper for batched attention.

        In CUDA graph mode, the wrapper's set_kv_cache() + run() operate on
        pre-computed token_to_page/token_to_cache or kv_cache_locations
        tensors (static GPU addresses). In eager mode, direct fancy indexing
        for KV writes and the raw FlashInfer wrapper's run().
        """
        if layer_idx is None:
            layer_idx = self.layer_idx

        orig_dtype = q.dtype

        if label is None:
            label = self._active_label()
        ps = self._plan_states[label]

        assert self.kv_cache is not None and ps.wrapper is not None

        self.attention.write_kv(k, v, layer_idx, label)

        if self.auto_write_store and ps.write_store:
            for req_id in self.request_ids:
                self.alloc_manager.flush_to_store(
                    req_id, label=label, layers=layer_idx
                )

        return self.attention.run(q, layer_idx, label).to(orig_dtype)

    # ------------------------------------------------------------------
    # Cross-attention: dispatched to the per-source CrossAttentionManager
    # (a read-only context pool plus its own wrapper machinery). The label
    # namespace under which cross plans are stored is that manager's
    # concern; callers keep addressing by base label and source.
    # ------------------------------------------------------------------

    def _get_cross_manager(self, source: str) -> CrossAttentionManager:
        manager = self.cross_attention.get(source)
        if manager is None:
            raise KeyError(
                f"No cross-attention pool for source {source!r}; declare it in "
                f"KVCacheConfig.cross_attn (available: {list(self.cross_attention)})"
            )
        return manager

    def _active_base_label(self, label: str | None) -> str:
        if label is not None:
            return label
        labels = list(self.active_labels.values())
        assert len(set(labels)) == 1, f"All active labels must be the same, got {labels}"
        return next(iter(self.active_labels.values()))

    @torch.compiler.disable
    def add_cross_attn_kv(
        self,
        request_ids: list[str],
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        seq_lens: list[int] | None = None,
        source: str = "default",
        label: str | None = None,
    ) -> None:
        """Write encoder-context K/V for one layer into ``source``'s pool.

        See ``CrossAttentionManager.write_context``.
        """
        self._get_cross_manager(source).write_context(
            request_ids=request_ids,
            k=k,
            v=v,
            layer_idx=layer_idx,
            seq_lens=seq_lens,
            base_label=self._active_base_label(label),
        )

    def plan_cross_attention(
        self,
        q_seq_lens: list[int],
        dtype: torch.dtype | None = None,
        source: str = "default",
        label: str | None = None,
    ) -> None:
        """Plan the cross-attention wrapper for this step's decoder queries.

        ``q_seq_lens`` matches the self-attention plan's seq_lens; ``label``
        is the base (self-attention) label. See
        ``CrossAttentionManager.plan``.
        """
        self._get_cross_manager(source).plan(
            request_ids=self.request_ids,
            q_seq_lens=q_seq_lens,
            dtype=dtype,
            base_label=self._active_base_label(label),
        )

    # Like run_attention: kept out of the compiled decoder — its query's leading
    # dim varies per step, which would blow dynamo's recompile limit (#160).
    @torch.compiler.disable
    def run_cross_attn(
        self,
        q: torch.Tensor,
        layer_idx: int | None = None,
        source: str = "default",
    ) -> torch.Tensor:
        """Run pre-planned cross-attention against ``source``'s context pool.

        Unlike ``run_attention``, the context K/V were written once by
        ``add_cross_attn_kv`` — nothing is written here.

        Args:
            q: [total_query_tokens, num_qo_heads, head_dim]
        Returns:
            output: [total_query_tokens, num_qo_heads, head_dim]
        """
        if layer_idx is None:
            layer_idx = self.layer_idx

        orig_dtype = q.dtype
        return self._get_cross_manager(source).run(
            q, layer_idx, self._active_base_label(None),
        ).to(orig_dtype)

class DenseGenCacheManager(FlashInferCacheManager):
    """Facade over the dense generation-attention manager.

    Runs non-causal generation attention (planned with ``dense_gen=True``) as a
    dense FlashAttention-3 pass over a contiguous [frozen-prefix | fresh]
    sequence instead of the paged FlashInfer prefill. Eager-only and
    single-request only (see ``_dense_gen_applies``); everything else —
    prefill, captured graphs, multi-request batches — falls through to the
    inherited paged FlashInfer path.
    """

    ATTENTION_MANAGER_CLS = DenseGenAttentionManager

    def _dense_gen_applies(self) -> bool:
        """Dense generation attention is eager-only (captured paths keep the
        persistent paged wrapper the graph was planned with) and single-request
        only (the launch overhead it removes is what matters at bs=1; bs>1
        amortizes the plan and grows the per-request prefix gather, so it stays
        on the paged path)."""
        return not self._cuda_graph_mode and len(self.request_ids) == 1

    def _dense_entries(
        self, labels: list[str], seq_lens: dict[str, list[int]],
    ) -> list[tuple]:
        """The dense plan's per-segment inputs, in the same (label, request)
        batch order the generation tokens are packed in. Each frozen prefix
        is read without extending its stream (the generation K/V never
        enters the pages), so it is planned from a zero-span view; the
        persistent request state rides along so the manager can cache the
        gathered prefix across denoise steps."""
        entries = []
        for label in labels:
            for i, rid in enumerate(self.request_ids):
                view = self.kv_pool.view(Segment(rid, label, 0))
                entries.append((view, seq_lens[label][i], self._get_state(rid, label)))
        return entries

    def plan_attention(
        self,
        seq_lens: list[int] | None = None,
        dtype: torch.dtype | None = None,
        is_causal=True,
        write_store: bool=True,
        label: str | None = None,
        dense_gen: bool = False,
        **kwargs,
    ):
        """As ``FlashInferCacheManager.plan_attention``, plus ``dense_gen``:
        the caller's declaration that this plan covers recomputed-every-step
        generation attention, eligible for the dense path when applicable."""
        if not (dense_gen and self._dense_gen_applies()):
            return super().plan_attention(
                seq_lens=seq_lens,
                dtype=dtype,
                is_causal=is_causal,
                write_store=write_store,
                label=label,
                **kwargs,
            )
        from mstar.utils.profiler import range_pop, range_push
        self._batched_cfg_info = None

        if self.enable_nvtx:
            range_push("cache.plan_attention", synchronize=False)
        try:
            # Lean dense generation-attention path: the gen K/V is never
            # written to pages and attention is one varlen FA3 over the
            # frozen prefix + fresh gen, so the whole paged-FlashInfer plan
            # (per-step page alloc, index tensors, and wrapper.plan()'s
            # radix-sort/fills) is dead work — run_dense reads none of it.
            # Build only the dense gather/varlen plan.
            effective_label = label if label is not None else self._active_label()
            self.attention.plan_dense(
                label=effective_label,
                entries=self._dense_entries(
                    [effective_label], {effective_label: seq_lens}
                ),
                seq_lens=seq_lens,
                write_store=write_store,
            )
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

    @torch.compiler.disable
    def plan_attention_batched_cfg(
        self,
        labels: list[str],
        seq_lens: list[int] | dict[str, list[int]],
        is_causal: bool = False,
        write_store: bool = False,
        dtype=torch.bfloat16,
        combined_label: str = "_cfg_batched",
        dense_gen: bool = False,
        **kwargs,
    ):
        """As ``FlashInferCacheManager.plan_attention_batched_cfg``, plus
        ``dense_gen`` (see ``plan_attention``)."""
        if not (dense_gen and self._dense_gen_applies()):
            return super().plan_attention_batched_cfg(
                labels=labels,
                seq_lens=seq_lens,
                is_causal=is_causal,
                write_store=write_store,
                dtype=dtype,
                combined_label=combined_label,
                **kwargs,
            )
        if isinstance(seq_lens, list):
            seq_lens = {
                key: seq_lens for key in labels
            }

        self._batched_cfg_info = BatchedCfgInfo(
            per_label_seq_len=seq_lens
        )

        # Lean dense generation-attention path (see plan_attention): skip the
        # paged-FlashInfer plan entirely and build only the per-segment dense
        # gather/varlen plan.
        self.attention.plan_dense(
            label=combined_label,
            entries=self._dense_entries(labels, seq_lens),
            seq_lens=seq_lens,
            write_store=write_store,
        )

    @torch.compiler.disable
    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int | None=None,
        label: str | None = None,
    ) -> torch.Tensor:
        """Route the label (the active one when None) to its dense plan
        when one was built, else run the inherited paged FlashInfer
        attention."""
        if label is None:
            label = self._active_label()
        ps = self._plan_states[label]
        if ps.dense_gen is not None:
            if layer_idx is None:
                layer_idx = self.layer_idx
            return self.attention.run_dense(q, k, v, layer_idx, ps.dense_gen).to(q.dtype)
        return super().run_attention(q, k, v, layer_idx=layer_idx, label=label)


# Backend registry: KVCacheConfig.attention_backend names one of these.
ATTENTION_BACKENDS: dict[str, type[BatchedCacheManager]] = {
    "flashinfer": FlashInferCacheManager,
    "dense_gen": DenseGenCacheManager,
}


@functools.cache
def _fa3_unavailable_reason() -> str | None:
    """None when the FlashAttention-3 forward kernel (``fa3-fwd``) imports in
    this environment, else the import failure. The wheel is ABI-tied to the
    installed torch/CUDA build, so a mismatch fails here rather than at the
    first attention call."""
    try:
        import fa3_fwd_interface  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


@functools.cache
def _warn_dense_gen_fallback(reason: str) -> None:
    logger.warning(
        "Attention backend 'dense_gen' requested but the fa3-fwd kernel is "
        "unavailable (%s); using the paged 'flashinfer' backend instead.",
        reason,
    )


def create_cache_manager(
    *, kv_cache_config: KVCacheConfig, **kwargs
) -> BatchedCacheManager:
    """Instantiate the cache-manager backend named by
    ``kv_cache_config.attention_backend``. Takes the same keyword arguments as
    ``BatchedCacheManager.__init__``.

    A dense-gen backend needs the FlashAttention-3 kernel; when that is not
    importable it degrades to the paged ``flashinfer`` backend (one warning
    per process) so serving works on environments without a matching
    ``fa3-fwd`` wheel."""
    backend_cls = ATTENTION_BACKENDS.get(kv_cache_config.attention_backend)
    if backend_cls is None:
        raise ValueError(
            f"Unknown attention backend {kv_cache_config.attention_backend!r}; "
            f"available: {sorted(ATTENTION_BACKENDS)}"
        )
    if issubclass(backend_cls, DenseGenCacheManager):
        reason = _fa3_unavailable_reason()
        if reason is not None:
            _warn_dense_gen_fallback(reason)
            backend_cls = FlashInferCacheManager
    return backend_cls(kv_cache_config=kv_cache_config, **kwargs)
