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
from mstar.utils.flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper

logger = logging.getLogger(__name__)


def cross_attn_label(label: str, source: str = "default") -> str:
    """Resolve the cache label under which a cross-attention plan/state for
    ``source`` is stored, relative to the base (self-attention) label."""
    return f"{label}::CROSS_ATTN::{source}"


@dataclass
class _PlanState:
    """Pre-computed state from plan_attention/plan_rope for a single cache label.

    Stored per-label so that preprocess can plan for all relevant labels
    upfront (plan operations are CUDA graph incompatible). During forward,
    run_attention/apply_rope look up the active label's plan state.

    In CUDA graph mode, wrapper is a persistent FlashInferPrefillWrapper or
    FlashInferDecodeWrapper created once during capture. plan_attention()
    calls wrapper.plan() which updates static buffers via .copy_().

    ``custom_pos_advance`` is a generic out-of-band channel for prefill
    walks whose position-id span differs from the seq_len being prefilled
    (e.g. Qwen3-Omni's ``prefill_vision``, where the 3D-grid MRoPE span is
    larger than the number of tokens). The submodule writes a per-request
    list here via ``BatchedCacheManager.set_custom_pos_advance``;
    ``advance_seq_lens`` reads it when ``pos_id_ns`` is None and advances
    ``position_id_start`` by these values instead of by ``seq_len``.
    Auto-cleared by ``advance_seq_lens`` so it doesn't leak across calls.
    The CUDA-graph runner's post-replay ``advance_seq_lens()`` call is what
    actually consumes this — the model's inner ``advance_seq_lens(pos_id_ns=...)``
    runs at capture time only and is not replayed.
    """
    wrapper: FlashInferPrefillWrapper | FlashInferDecodeWrapper | None = None
    pos_ids: torch.Tensor | None = None
    seq_lens: list[int] | None = None
    write_store: bool = True
    custom_pos_advance: list[int] | None = None
    # Set when DenseGenCacheManager planned this label dense: the per-segment
    # gather indices + varlen cu_seqlens needed to attend each generation
    # segment over its contiguous frozen prefix. None on paged plans, which
    # keep the FlashInfer path. See DenseGenCacheManager._build_dense_gen_plan.
    dense_gen: dict | None = None


class WorkspaceBufferManager:
    def __init__(
        self, size, device
    ):
        self.size = size
        self.device = device
        self.buffers = {}

    def get(self, label: str="main"):
        if label not in self.buffers:
            self.buffers[label] = torch.empty(
                self.size, dtype=torch.uint8, device=self.device
            )
        return self.buffers[label]


@dataclass
class BatchedCfgInfo:
    per_label_seq_len: dict[str, list[int]]


class BatchedCacheManager(ABC):
    """Attention/KV-cache backend interface for batched multi-request forwards.

    Owns the backend-agnostic machinery: per-label plan state, active-label
    switching, RoPE position planning/application, sequence-length stepping,
    KV snapshots, store flushes, and the qo_indptr accessor. Concrete backends
    implement the attention ops (``plan_attention``,
    ``plan_attention_batched_cfg``, ``run_attention``); ``ATTENTION_BACKENDS``
    maps ``KVCacheConfig.attention_backend`` names to backend classes and
    ``create_cache_manager`` instantiates the configured one.

    Replaces per-request CacheHandle for decode and simple prefill batches where
    all requests use the same graph_walk. Constructed per batch: one manager
    serves the whole batch with a single attention call per layer instead of N
    per-request calls. Complex paths like image_gen (3-pass CFG with label
    switching) continue using per-request CacheHandle.
    """

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

        self.auto_write_store = auto_write_store

        # CUDA graph mode: persistent wrappers passed in from CudaGraphRunner.
        # When set, plan_attention() uses the persistent wrapper's plan()
        # method instead of creating a new wrapper each call.
        self._cuda_graph_mode = cuda_graph_plan_states is not None

        # Per-label plan state: plan_attention/plan_rope store results here,
        # run_attention/apply_rope look up by active label.
        if cuda_graph_plan_states is not None:
            self._plan_states: dict[str, _PlanState] = cuda_graph_plan_states
        else:
            self._plan_states: dict[str, _PlanState] = {}

        self.base_pos_ids = torch.arange(
            kv_cache_config.max_seq_len, dtype=torch.long, device=device
        )

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
        """Return the persistent qo_indptr static buffer for a CUDA-graph
        prefill wrapper, or None if not in CUDA-graph mode / wrong wrapper.

        Captured prefill paths read this to recover per-request token boundaries
        from inside the captured region — plan_attention updates the buffer via
        .copy_() outside the graph, so the address stays stable across replay.
        """
        ps = self._plan_states.get(label)
        if ps is None or ps.wrapper is None:
            return None
        return getattr(ps.wrapper, "_qo_indptr_buf", None)

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
    ) -> torch.Tensor:
        """Run the pre-planned attention for the active label.

        Args:
            q: [total_tokens, num_q_heads, head_dim]
            k: [total_tokens, num_kv_heads, head_dim]
            v: [total_tokens, num_kv_heads, head_dim]
            layer_idx: transformer layer index
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
            # CPU-accumulate the position list (1 int per output token). The
            # old `torch.cat([torch.arange(...) + start for ...])` launched
            # 2 GPU kernels per request.
            if self.enable_nvtx:
                range_push("cache.plan_rope.build_pos_ids", synchronize=False)
            try:
                pos_ids_list: list[int] = []
                for rid, sl in zip(self.request_ids, seq_lens, strict=True):
                    start = self._get_state(rid, effective_label).position_id_start
                    pos_ids_list.extend(range(start, start + sl))
                computed_pos_ids = torch.tensor(
                    pos_ids_list,
                    dtype=torch.long,
                    device=None if static_copy_from_cpu else self.device,
                )
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
                pos_ids_list: list[int] = []
                for rid, sl in zip(self.request_ids, seq_lens[label], strict=True):
                    start = self._get_state(rid, label).position_id_start
                    pos_ids_list.extend(range(start, start + sl))
                parts.append(torch.tensor(
                    pos_ids_list, dtype=torch.long, device=self.device,
                ))
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
        **kwargs
    ):
        """Apply RoPE using the active label's pre-computed position IDs."""
        label = self._active_label()

        ps = self._plan_states[label]
        assert ps.pos_ids is not None

        orig_dtype = q.dtype

        if rope_dtype is not None:
            q, k = q.to(rope_dtype), k.to(rope_dtype)
        elif torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            q, k = q.to(dtype), k.to(dtype)
        elif q.dtype == torch.float32:
            dtype = torch.bfloat16
            q, k = q.to(dtype), k.to(dtype)

        llama31_params = {}
        for key, value in kwargs.items():
            if key in ['low_freq_factor', 'high_freq_factor', 'old_context_len']:
                llama31_params[key] = value

        import flashinfer

        if not llama31_params:
            flashinfer.rope.apply_rope_pos_ids_inplace(
                q, k, ps.pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_scale=rope_scale,
                rope_theta=rope_theta,

            )
        else:
            flashinfer.rope.apply_llama31_rope_pos_ids_inplace(
                q, k, ps.pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
                **llama31_params
            )
        return q.to(orig_dtype), k.to(orig_dtype)

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
            state = self._get_state(rid)
            state.seq_len += n
            state.position_id_start += (pos_id_n if pos_id_n is not None else n)

    @torch.compiler.disable
    def set_custom_pos_advance(
        self, pos_advance: list[int] | None, label: str | None = None,
    ) -> None:
        """Stash a per-request position-id advance for the next
        ``advance_seq_lens()`` call to consume.

        Resolves ``label`` the same way ``plan_attention`` does: explicit
        label wins; otherwise the (single) currently-active label is used.

        Used by submodules whose forward advances ``position_id_start`` by
        something other than ``seq_len`` (e.g. Qwen3-Omni's prefill_vision
        passes the MRoPE 3D-grid span here). Auto-cleared by
        ``advance_seq_lens`` after use, so it does not leak across calls.

        Pass ``pos_advance=None`` to clear an earlier set explicitly.
        """
        effective_label = label
        if effective_label is None:
            labels = list(self.active_labels.values())
            if not labels:
                return
            assert len(set(labels)) == 1, (
                f"All active labels must be the same to omit ``label``, got {labels}"
            )
            effective_label = labels[0]
        ps = self._plan_states.get(effective_label)
        if ps is None:
            return
        ps.custom_pos_advance = (
            list(pos_advance) if pos_advance is not None else None
        )

    @torch.compiler.disable
    def advance_seq_lens(self, pos_id_ns: list[int] | int | None = None) -> None:
        """Advance seq_len for each request by different amounts.

        When ``pos_id_ns`` is None, falls back to a per-label side-channel
        (``_PlanState.custom_pos_advance``, set via
        ``set_custom_pos_advance``) for walks whose position-id span
        differs from seq_len (e.g. Qwen3-Omni prefill_vision). The
        side-channel is auto-cleared after use so it doesn't leak across
        calls.
        """

        if self._batched_cfg_info:
            for label, seq_lens in self._batched_cfg_info.per_label_seq_len.items():
                for i, rid in enumerate(self.request_ids):
                    n = seq_lens[i]
                    state = self._get_state(rid, label=label)
                    state.seq_len += n
                    if pos_id_ns is None:
                        state.position_id_start += n
                    elif isinstance(pos_id_ns, int):
                        state.position_id_start += pos_id_ns
                    else:
                        state.position_id_start += pos_id_ns[i]
        else:
            for i, rid in enumerate(self.request_ids):
                label = self.active_labels[rid]
                ps = self._plan_states[label]
                n = ps.seq_lens[i]
                state = self._get_state(rid, label=label)
                state.seq_len += n
                if pos_id_ns is None:
                    if ps.custom_pos_advance is not None:
                        state.position_id_start += ps.custom_pos_advance[i]
                    else:
                        state.position_id_start += n
                elif isinstance(pos_id_ns, int):
                    state.position_id_start += pos_id_ns
                else:
                    state.position_id_start += pos_id_ns[i]
        # Clear the side-channel on every consumer so a stale value can't
        # bleed into a subsequent walk.
        for ps in self._plan_states.values():
            ps.custom_pos_advance = None

    @torch.compiler.disable
    def snapshot_all(
        self, from_label: str,
        to_label: str,
        realloc: bool=False,
        write_store: bool=True
    ) -> None:
        """Snapshot KV cache for all requests in batch."""
        for rid in self.request_ids:
            from_state = self._get_state(rid, from_label)

            if realloc:
                self.alloc_manager.reset_label(rid, to_label)

            to_state = self._get_state(rid, to_label)
            start_pos =  to_state.seq_len // self.kv_cache_config.page_size
            self.alloc_manager.alloc(
                rid, to_label, seq_len=from_state.seq_len
            )

            to_state.seq_len = from_state.seq_len
            to_state.position_id_start = from_state.position_id_start

            for src_page, dst_page in zip(
                from_state.page_indices[start_pos:],
                to_state.page_indices[start_pos:],
                strict=True
            ):
                self.kv_cache[:, dst_page] = self.kv_cache[:, src_page]
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
    """Paged FlashInfer attention backend (the default).

    Constructs batch-level FlashInfer index tensors (qo_indptr, paged_kv_indptr,
    paged_kv_indices) and issues a single FlashInfer call per layer instead of
    N separate calls. K/V for every planned token is written to the paged cache.
    """

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

        Allocates pages, computes page_indices/page_offsets/token_offsets for
        vectorized KV writes, builds FlashInfer index tensors, and plans the
        wrapper. All state is stored in _plan_states[label].

        In CUDA graph mode, uses the persistent wrapper from _plan_states
        (pre-built by CudaGraphRunner) and calls its plan() method which
        updates static buffers via .copy_(). In eager mode, creates a new
        wrapper each call.

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
            self._plan_attention_impl(
                seq_lens=seq_lens,
                dtype=dtype,
                is_causal=is_causal,
                write_store=write_store,
                label=label,
            )
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def _plan_attention_impl(
        self,
        seq_lens: list[int] | None = None,
        dtype: torch.dtype | None = None,
        is_causal=True,
        write_store: bool=True,
        label: str | None = None,
    ):
        from mstar.utils.profiler import range_pop, range_push

        assert self.kv_cache is not None

        # Default the FlashInfer wrapper's dtype to whatever dtype the KV
        # cache tensor was actually allocated in. Hardcoding bf16 here breaks
        # any model that runs in fp32 (the wrapper would try to write
        # bf16-cast K/V into an fp32 cache and torch raises a dtype mismatch
        # in flashinfer_utils.set_kv_cache).
        if dtype is None:
            dtype = self.kv_cache.dtype

        effective_label = label if label is not None else self._active_label()

        cfg = self.kv_cache_config
        page_size = cfg.page_size
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim
        num_qo_heads = cfg.num_qo_heads
        device = self.device

        # CPU-side accumulation. The old implementation launched 4-5 tiny GPU
        # kernels per request (arange, tensor(state.page_indices), indexing,
        # mod) to build page_indices/page_offsets/token_offsets — all of which
        # turn out to be unused bookkeeping (grep: no reader in the codebase).
        # We only need the four int32 tensors the FlashInfer wrapper consumes,
        # so do the arithmetic in pure Python and send them over in one H2D
        # each.
        if self.enable_nvtx:
            range_push("cache.plan_attention.build_lists", synchronize=False)
        try:
            qo_indptr_list = [0]
            kv_indptr_list = [0]
            all_page_indices = []
            kv_last_page_lens = []
            kv_cache_locations_list = []

            for i, rid in enumerate(self.request_ids):
                state = self._get_state(rid, effective_label)
                sl = seq_lens[i]
                total_len = state.seq_len + sl

                self.alloc_manager.alloc(
                    rid, label=effective_label, seq_len=total_len
                )

                qo_indptr_list.append(qo_indptr_list[-1] + sl)
                all_page_indices.extend(state.page_indices)
                kv_indptr_list.append(kv_indptr_list[-1] + len(state.page_indices))

                last_page_len = total_len % page_size or page_size
                kv_last_page_lens.append(last_page_len)
                if sl == 1:
                    kv_cache_locations_list.append([state.page_indices[-1], last_page_len - 1])
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

        # Build batched FlashInfer index tensors on CPU so wrapper.plan()
        # doesn't trigger a synchronous D→H inside its body. FlashInfer
        # calls ``indptr.to("cpu")`` / ``last_page_len.to("cpu")`` near the
        # top of ``plan()`` to get host views of those metadata tensors;
        # if we hand them GPU tensors that ``.to("cpu")`` becomes a
        # synchronous default-stream sync that waits for the entire
        # outstanding stream — including the speculatively-queued next
        # decode step. By creating these on CPU directly, ``.to("cpu")``
        # is a no-op. FlashInfer later copies the tiny int32 metadata to
        # the device when it needs it; the source is pageable CPU memory, so
        # ``non_blocking=True`` does not make that H2D copy asynchronous, but
        # the tensors are batch-size length and the cost is inconsequential.
        if self.enable_nvtx:
            range_push("cache.plan_attention.make_tensors", synchronize=False)
        try:
            qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32)
            paged_kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32)
            paged_kv_indices = torch.tensor(all_page_indices, dtype=torch.int32)
            paged_kv_last_page_len = torch.tensor(kv_last_page_lens, dtype=torch.int32)
            kv_cache_locations = (
                torch.tensor(kv_cache_locations_list, dtype=torch.long)
                if len(kv_cache_locations_list) == len(self.request_ids)
                else None
            )
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)


        is_decode = all([sl == 1 for sl in seq_lens])
        ps = self._plan_states.get(effective_label)
        if ps is not None and ps.wrapper is not None:
            wrapper = ps.wrapper
        elif is_decode:
            wrapper = FlashInferDecodeWrapper(
                workspace_buffer=self.buffer_manager.get(effective_label),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                device=self.device,
                enable_nvtx=self.enable_nvtx,
            )
            ps = _PlanState(wrapper=wrapper)
            self._plan_states[effective_label] = ps
        else:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self.buffer_manager.get(effective_label),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                device=self.device,
                enable_nvtx=self.enable_nvtx,
            )
            ps = _PlanState(wrapper=wrapper)
            self._plan_states[effective_label] = ps

        if self.enable_nvtx:
            range_push("cache.plan_attention.wrapper_plan", synchronize=False)
        try:
            if isinstance(wrapper, FlashInferDecodeWrapper):
                wrapper.plan(
                    paged_kv_indptr=paged_kv_indptr,
                    paged_kv_indices=paged_kv_indices,
                    paged_kv_last_page_len=paged_kv_last_page_len,
                    kv_cache_locations=kv_cache_locations,
                    dtype=dtype,
                )
            else:
                wrapper.plan(
                    qo_indptr=qo_indptr,
                    paged_kv_indptr=paged_kv_indptr,
                    paged_kv_indices=paged_kv_indices,
                    paged_kv_last_page_len=paged_kv_last_page_len,
                    causal=is_causal,
                    dtype=dtype,
                )
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)
        # seq_lens is read by the flush_to_store path; write_store by
        # run_attention. The page_indices / page_offsets / token_offsets /
        # per_req_page_indices fields were legacy bookkeeping and had no
        # reader — dropped along with their per-rid GPU construction above.
        ps.seq_lens = seq_lens
        ps.write_store = write_store
        # A paged plan clears any prior dense plan (set by DenseGenCacheManager)
        # so run_attention routes this label back through the wrapper.
        ps.dense_gen = None

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

        cfg = self.kv_cache_config
        page_size = cfg.page_size
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim
        num_qo_heads = cfg.num_qo_heads
        device = self.device

        # CPU-side accumulation (see plan_attention for the same pattern).
        qo_indptr_list = [0]
        kv_indptr_list = [0]
        all_page_indices = []
        kv_last_page_lens = []
        combined_seq_lens = []

        for label in labels:
            for i, rid in enumerate(self.request_ids):
                state = self._get_state(rid, label)
                sl = seq_lens[label][i]
                total_len = state.seq_len + sl

                self.alloc_manager.alloc(rid, label=label, seq_len=total_len)

                qo_indptr_list.append(qo_indptr_list[-1] + sl)
                all_page_indices.extend(state.page_indices)
                kv_indptr_list.append(
                    kv_indptr_list[-1] + len(state.page_indices)
                )

                last_page_len = total_len % page_size or page_size
                kv_last_page_lens.append(last_page_len)
                combined_seq_lens.append(sl)

        # CPU tensors — see comment in ``plan_attention`` above. FlashInfer
        # async-H2Ds these inside ``plan()``; passing GPU tensors would
        # trigger a synchronous default-stream sync via the internal
        # ``.to("cpu")`` call.
        qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32)
        paged_kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32)
        paged_kv_indices = torch.tensor(all_page_indices, dtype=torch.int32)
        paged_kv_last_page_len = torch.tensor(kv_last_page_lens, dtype=torch.int32)

        ps = self._plan_states.get(combined_label)
        if self._cuda_graph_mode and ps is not None and ps.wrapper is not None:
            # CUDA-graph mode: reuse the persistent wrapper across denoise steps.
            # plan() updates its static buffers via .copy_() so the captured
            # kernel picks up each step's page table without reallocating.
            wrapper = ps.wrapper
        elif self._cuda_graph_mode:
            # First call under capture: build the persistent wrapper sized for the
            # fixed batch (labels x requests) and token budget.
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self.buffer_manager.get(combined_label),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                batch_size=len(labels) * len(self.request_ids),
                max_total_tokens=sum(combined_seq_lens),
                max_num_pages=cfg.max_num_pages,
                device=self.device,
                use_cuda_graph=True,
                enable_nvtx=self.enable_nvtx,
            )
            ps = _PlanState(wrapper=wrapper)
            self._plan_states[combined_label] = ps
        else:
            # Eager mode: a fresh wrapper each call (the cache manager is rebuilt
            # per forward, so there is nothing persistent to reuse).
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self.buffer_manager.get(combined_label),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                enable_nvtx=self.enable_nvtx,
            )
            ps = _PlanState(wrapper=wrapper)
            self._plan_states[combined_label] = ps

        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            causal=is_causal,
            dtype=dtype,
        )
        ps.seq_lens = combined_seq_lens
        ps.write_store = write_store
        # A paged plan clears any prior dense plan (set by DenseGenCacheManager)
        # so run_attention routes this label back through the wrapper.
        ps.dense_gen = None

    @torch.compiler.disable
    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int | None=None,
    ) -> torch.Tensor:
        """Run pre-planned FlashInfer attention with KV cache write.

        Uses the active label's plan state (set up by a prior plan_attention
        call). Writes K and V to the paged KV cache at pre-computed page
        positions, then runs the FlashInfer wrapper for batched attention.

        In CUDA graph mode, uses wrapper.set_kv_cache() + wrapper.run()
        which operates on pre-computed token_to_page/token_to_cache or
        kv_cache_locations tensors (static GPU addresses).

        In eager mode, uses direct fancy indexing for KV writes and
        the raw FlashInfer wrapper's run().
        """
        if layer_idx is None:
            layer_idx = self.layer_idx

        orig_dtype = q.dtype

        label = self._active_label()
        ps = self._plan_states[label]

        assert self.kv_cache is not None and ps.wrapper is not None

        ps.wrapper.set_kv_cache(self.kv_cache[layer_idx], k, v)

        if self.auto_write_store and ps.write_store:
            for req_id in self.request_ids:
                self.alloc_manager.flush_to_store(
                    req_id, label=label, layers=layer_idx
                )

        return ps.wrapper.run(q, self.kv_cache[layer_idx]).to(orig_dtype)

    # ------------------------------------------------------------------
    # Cross-attention (issue #160)
    #
    # Cross-attention is non-causal attention over a separate, fixed
    # encoder-context KV: written once at encode time (add_cross_attn_kv),
    # planned per step against the decoder's query lengths
    # (plan_cross_attention), and executed per layer (run_cross_attn). The
    # context KV lives in per-source pools (KVCacheConfig.cross_attn) whose
    # head config may differ from the decoder's self-attention; plan/run
    # state rides the existing per-label _PlanState machinery under the
    # resolved ``{label}::CROSS_ATTN::{source}`` label. Cross labels never
    # plan RoPE — context positions, if any, are baked in at encode time.
    # ------------------------------------------------------------------

    def _get_cross_pool(self, source: str) -> CrossAttnPool:
        pool = self.cross_pools.get(source)
        if pool is None:
            raise KeyError(
                f"No cross-attention pool for source {source!r}; declare it in "
                f"KVCacheConfig.cross_attn (available: {list(self.cross_pools)})"
            )
        return pool

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

        Called once per request per layer at encode time. ``k``/``v`` are
        packed ``(total_context_tokens, num_kv_heads, head_dim)`` across
        ``request_ids``; ``seq_lens`` gives per-request context lengths
        (defaults to a single request owning the whole tensor). Pages are
        allocated on first write (layer 0) and reused for the rest.
        """
        pool = self._get_cross_pool(source)
        base_label = self._active_base_label(label)
        cross_label = cross_attn_label(base_label, source)
        page_size = pool.alloc_config.page_size

        if seq_lens is None:
            assert len(request_ids) == 1, (
                "add_cross_attn_kv needs seq_lens for multi-request batches"
            )
            seq_lens = [k.shape[0]]

        offset = 0
        for rid, ctx_len in zip(request_ids, seq_lens, strict=True):
            state = pool.alloc_manager.get_state(rid, cross_label)
            if state.seq_len == 0:
                pool.alloc_manager.alloc(rid, label=cross_label, seq_len=ctx_len)
                state.seq_len = ctx_len
            else:
                assert state.seq_len == ctx_len, (
                    f"cross-attn context for {rid!r}/{source!r} already written "
                    f"with length {state.seq_len}, got {ctx_len}"
                )

            positions = torch.arange(ctx_len, device=self.device)
            page_indices = torch.tensor(
                state.page_indices, dtype=torch.long, device=self.device,
            )
            token_to_page = page_indices[
                torch.div(positions, page_size, rounding_mode="floor")
            ]
            token_to_cache = positions % page_size

            layer_cache = pool.kv_cache[layer_idx]
            dtype = pool.kv_cache.dtype
            layer_cache[token_to_page, 0, token_to_cache] = \
                k[offset:offset + ctx_len].to(dtype)
            layer_cache[token_to_page, 1, token_to_cache] = \
                v[offset:offset + ctx_len].to(dtype)
            offset += ctx_len

    def plan_cross_attention(
        self,
        q_seq_lens: list[int],
        dtype: torch.dtype | None = None,
        source: str = "default",
        label: str | None = None,
    ) -> None:
        """Plan the cross-attention wrapper for this step's decoder queries.

        ``q_seq_lens`` is the number of decoder query tokens per request
        (matches the self-attention plan's seq_lens). The context side is
        read from the pool state written by ``add_cross_attn_kv`` — pages
        are fixed, so unlike ``plan_attention`` nothing is allocated here.
        ``label`` is the base (self-attention) label; the plan is stored
        under the resolved cross label. Always uses the prefill wrapper
        with ``causal=False`` (decode-style single queries still attend to
        the full context).
        """
        pool = self._get_cross_pool(source)
        base_label = self._active_base_label(label)
        cross_label = cross_attn_label(base_label, source)
        cfg = pool.alloc_config
        page_size = cfg.page_size

        if dtype is None:
            dtype = pool.kv_cache.dtype

        qo_indptr_list = [0]
        kv_indptr_list = [0]
        all_page_indices: list[int] = []
        kv_last_page_lens: list[int] = []
        for i, rid in enumerate(self.request_ids):
            state = pool.alloc_manager.get_state(rid, cross_label)
            assert state.seq_len > 0, (
                f"plan_cross_attention before add_cross_attn_kv for {rid!r} "
                f"(source {source!r})"
            )
            qo_indptr_list.append(qo_indptr_list[-1] + q_seq_lens[i])
            all_page_indices.extend(state.page_indices)
            kv_indptr_list.append(kv_indptr_list[-1] + len(state.page_indices))
            kv_last_page_lens.append(state.seq_len % page_size or page_size)

        ps = self._plan_states.get(cross_label)
        if ps is None or ps.wrapper is None:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self.buffer_manager.get(cross_label),
                num_qo_heads=cfg.num_qo_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                page_size=page_size,
                device=self.device,
                enable_nvtx=self.enable_nvtx,
            )
            ps = _PlanState(wrapper=wrapper)
            self._plan_states[cross_label] = ps

        ps.wrapper.plan(
            qo_indptr=torch.tensor(qo_indptr_list, dtype=torch.int32),
            paged_kv_indptr=torch.tensor(kv_indptr_list, dtype=torch.int32),
            paged_kv_indices=torch.tensor(all_page_indices, dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor(kv_last_page_lens, dtype=torch.int32),
            causal=False,
            dtype=dtype,
        )
        ps.seq_lens = q_seq_lens
        ps.write_store = False

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
        pool = self._get_cross_pool(source)
        base_label = self._active_base_label(None)
        cross_label = cross_attn_label(base_label, source)

        orig_dtype = q.dtype
        ps = self._plan_states.get(cross_label)
        assert ps is not None and ps.wrapper is not None, (
            f"run_cross_attn before plan_cross_attention (label {cross_label!r})"
        )
        return ps.wrapper.run(q, pool.kv_cache[layer_idx]).to(orig_dtype)

class DenseGenCacheManager(FlashInferCacheManager):
    """FlashInfer backend with a dense generation-attention fast path.

    Runs non-causal generation attention (planned with ``dense_gen=True``) as a
    dense FlashAttention-3 pass over a contiguous [frozen-prefix | fresh]
    sequence instead of the paged FlashInfer prefill. Diffusion recomputes every
    generation K/V each step (only the tiny text prefix is reused), so the paged
    path's per-step full-buffer K/V write is pure overhead here; a dense pass
    gathers the small prefix, concatenates it with the freshly projected K/V,
    and runs one varlen kernel — which is also the faster attention kernel at
    these shapes. Eager-only and single-request only (see ``_dense_gen_applies``);
    everything else — prefill, captured graphs, multi-request batches — falls
    through to the inherited paged FlashInfer path.
    """

    def _dense_gen_applies(self) -> bool:
        """Dense generation attention is eager-only (captured paths keep the
        persistent paged wrapper the graph was planned with) and single-request
        only (the launch overhead it removes is what matters at bs=1; bs>1
        amortizes the plan and grows the per-request prefix gather, so it stays
        on the paged path)."""
        return not self._cuda_graph_mode and len(self.request_ids) == 1

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
            # radix-sort/fills) is dead work — _run_dense_gen reads none of
            # it. Build only the dense gather/varlen plan.
            effective_label = label if label is not None else self._active_label()
            ps = self._plan_states.get(effective_label) or _PlanState()
            self._plan_states[effective_label] = ps
            ps.seq_lens = seq_lens
            ps.write_store = write_store
            ps.dense_gen = self._build_dense_gen_plan([effective_label], seq_lens)
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
        # paged-FlashInfer plan (per-label page alloc, index tensors, and
        # wrapper.plan()'s radix-sort/fills) — _run_dense_gen reads none of
        # it. Build only the per-segment dense gather/varlen plan.
        ps = self._plan_states.get(combined_label) or _PlanState()
        self._plan_states[combined_label] = ps
        ps.seq_lens = seq_lens
        ps.write_store = write_store
        ps.dense_gen = self._build_dense_gen_plan(labels, seq_lens)

    @torch.compiler.disable
    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int | None=None,
    ) -> torch.Tensor:
        """Route the active label to its dense plan when one was built, else
        run the inherited paged FlashInfer attention."""
        label = self._active_label()
        ps = self._plan_states[label]
        if ps.dense_gen is not None:
            if layer_idx is None:
                layer_idx = self.layer_idx
            return self._run_dense_gen(q, k, v, layer_idx, ps.dense_gen).to(q.dtype)
        return super().run_attention(q, k, v, layer_idx=layer_idx)

    def _build_dense_gen_plan(
        self, labels: list[str],
        seq_lens: list[int] | dict[str, list[int]]
    ) -> dict:
        """Pre-compute the per-segment gather + varlen layout for the dense
        generation-attention path, in the same (label, request) batch order the
        generation tokens are packed in. Each segment attends its fresh
        generation tokens over its frozen text prefix; the prefix lives in the
        pages written at prefill, so we record the page indices to gather it from
        (the same across all layers) and the cumulative-sequence-length tensors a
        single varlen kernel needs. Built once per denoise step, reused by every
        layer's run_attention."""

        if isinstance(seq_lens, list):
            seq_lens = {
                key: seq_lens for key in labels
            }

        cfg = self.kv_cache_config
        page_size = cfg.page_size
        segs = []  # (prefix_page_indices, prefix_len, gen_len)
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        for label in labels:
            for i, rid in enumerate(self.request_ids):
                state = self._get_state(rid, label)
                prefix_len = state.seq_len
                gen_len = seq_lens[label][i]
                n_pages = (prefix_len + page_size - 1) // page_size
                idx = torch.tensor(
                    state.page_indices[:n_pages], dtype=torch.long, device=self.device
                )
                # Carry the persistent KVRequestState so run_attention can cache
                # the gathered frozen prefix on it across denoise steps (the
                # manager itself is rebuilt every forward).
                segs.append((idx, prefix_len, gen_len, state))
                cu_q.append(cu_q[-1] + gen_len)
                cu_k.append(cu_k[-1] + prefix_len + gen_len)
                max_q = max(max_q, gen_len)
                max_k = max(max_k, prefix_len + gen_len)
        return {
            "segs": segs,
            "cu_q": torch.tensor(cu_q, dtype=torch.int32, device=self.device),
            "cu_k": torch.tensor(cu_k, dtype=torch.int32, device=self.device),
            "max_q": max_q,
            "max_k": max_k,
        }

    @torch.compiler.disable
    def _run_dense_gen(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_idx: int, dg: dict
    ) -> torch.Tensor:
        """Dense generation attention: per segment, take the frozen text-prefix
        K/V, concatenate it with this segment's fresh K/V, and attend
        non-causally with one FlashAttention-3 varlen kernel. Bypasses the paged
        write entirely (the generation K/V is recomputed every step, so
        persisting it is wasted work). The frozen prefix is gathered from the
        paged cache once per layer and cached on the request state, then reused
        across denoise steps (it never changes during denoise)."""
        from fa3_fwd_interface import flash_attn_varlen_func

        cfg = self.kv_cache_config
        num_kv_heads, head_dim = cfg.num_kv_heads, cfg.head_dim
        kv_layer = self.kv_cache[layer_idx]  # [max_pages, 2, page_size, num_kv_heads, head_dim]

        k_parts, v_parts = [], []
        offset = 0
        for idx, prefix_len, gen_len, state in dg["segs"]:
            prefix_cache = state.dense_prefix_kv
            if prefix_cache is None:
                prefix_cache = state.dense_prefix_kv = {}
            cached = prefix_cache.get(layer_idx)
            if cached is None:
                sub = kv_layer[idx]  # [n_pages, 2, page_size, num_kv_heads, head_dim]
                k_pref = sub[:, 0].reshape(-1, num_kv_heads, head_dim)[:prefix_len].clone()
                v_pref = sub[:, 1].reshape(-1, num_kv_heads, head_dim)[:prefix_len].clone()
                prefix_cache[layer_idx] = (k_pref, v_pref)
            else:
                k_pref, v_pref = cached
            k_parts.append(k_pref)
            k_parts.append(k[offset:offset + gen_len])
            v_parts.append(v_pref)
            v_parts.append(v[offset:offset + gen_len])
            offset += gen_len
        key = torch.cat(k_parts, dim=0)
        val = torch.cat(v_parts, dim=0)
        if q.dtype != key.dtype:
            q = q.to(key.dtype)

        out = flash_attn_varlen_func(
            q, key, val, dg["cu_q"], dg["cu_k"], dg["max_q"], dg["max_k"], causal=False,
        )
        return out[0] if isinstance(out, tuple) else out


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
