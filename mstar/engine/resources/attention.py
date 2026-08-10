"""Attention manager resources.

An attention manager owns the attention execution plan for its set of
layers and the static buffers behind it: the FlashInfer wrappers, the
workspace buffers, and the per-label plan state. It plans against the
sequence views handed to it rather than reading cache internals, and it
exposes the device-side write and run calls the model's forward uses.
Cross-attention gets its own manager over a read-only context pool.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from mstar.engine.kv_store import CrossAttnPool, KVCacheConfig, KVRequestState
from mstar.engine.resources.base import Segment, SequenceView
from mstar.engine.resources.kv_pool import KVCachePool
from mstar.utils.flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper

logger = logging.getLogger(__name__)


def cross_attn_label(label: str, source: str = "default") -> str:
    """Resolve the cache label under which a cross-attention plan/state for
    ``source`` is stored, relative to the base (self-attention) label."""
    return f"{label}::CROSS_ATTN::{source}"


class PagedIndptrs(NamedTuple):
    """The four int32 index tensors a FlashInfer prefill/decode wrapper's
    ``plan`` consumes, built on CPU (so wrapper.plan's ``.to("cpu")`` is a
    no-op — see ``FlashInferAttentionManager.plan``)."""
    qo_indptr: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor


def build_paged_indptrs(
    q_seq_lens: list[int],
    page_indices_per_request: list[Sequence[int]],
    context_lens: list[int],
    page_size: int,
) -> PagedIndptrs:
    """Assemble FlashInfer paged-attention index tensors from per-request
    query lengths + already-allocated page lists. Shared by the self- and
    cross-attention plan paths (the difference is only where the pages come
    from: self grows them per step, cross reads a fixed context)."""
    qo_indptr = [0]
    kv_indptr = [0]
    all_pages: list[int] = []
    last_page_lens: list[int] = []
    for q_len, pages, ctx_len in zip(
        q_seq_lens, page_indices_per_request, context_lens, strict=True,
    ):
        qo_indptr.append(qo_indptr[-1] + q_len)
        all_pages.extend(pages)
        kv_indptr.append(kv_indptr[-1] + len(pages))
        last_page_lens.append(ctx_len % page_size or page_size)
    return PagedIndptrs(
        qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32),
        paged_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
        paged_kv_indices=torch.tensor(all_pages, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor(last_page_lens, dtype=torch.int32),
    )


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable. Used by cross-attention (context
    pages are immutable after encode); the mechanism is label-generic, so a
    fixed-shape self-attention label could reuse it (see the field on
    ``_PlanState``)."""
    q_seq_lens: tuple
    page_indices: tuple
    last_page_lens: tuple
    dtype: torch.dtype


@dataclass
class _PlanState:
    """Pre-computed state from plan_attention/plan_rope for a single cache label.

    Stored per-label so that preprocess can plan for all relevant labels
    upfront (plan operations are CUDA graph incompatible). During forward,
    run_attention/apply_rope look up the active label's plan state.

    In CUDA graph mode, wrapper is a persistent FlashInferPrefillWrapper or
    FlashInferDecodeWrapper created once during capture. plan_attention()
    calls wrapper.plan() which updates static buffers via .copy_().
    """
    wrapper: FlashInferPrefillWrapper | FlashInferDecodeWrapper | None = None
    pos_ids: torch.Tensor | None = None
    seq_lens: list[int] | None = None
    write_store: bool = True
    # Plan memo: fingerprint of the last wrapper.plan() inputs for this label;
    # when it matches, the re-plan is skipped. Only the cross-attention path
    # sets it today (its context pages are immutable after add_cross_attn_kv),
    # and only where plan states persist across steps (the CUDA-graph runner);
    # the eager path rebuilds the plan surface per step and still re-plans.
    #
    # Future reference — this is a general-purpose tool, not cross-attn only.
    # A regular (self-attention) label can memo its plan too; the extra care
    # there is invalidation, since self-attn pages grow every decode step.
    # The fingerprint would need to include the per-request seq_len (so appending
    # a token misses the memo and re-plans), and any page-table remap (eviction /
    # reallocation) must also bust the key. Given that, decode could skip the
    # re-plan on the common "seq_len += 1, same pages" step. Deferred until a
    # model needs it; the eager-path persistence noted above is the prerequisite.
    plan_cache_key: "PlanCacheKey | None" = None
    # Set when the dense-gen manager planned this label dense: the per-segment
    # gather indices + varlen cu_seqlens needed to attend each generation
    # segment over its contiguous frozen prefix. None on paged plans, which
    # keep the FlashInfer path. See DenseGenAttentionManager.build_dense_plan.
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


class FlashInferAttentionManager:
    """Paged FlashInfer attention (the default backend).

    Builds batch-level FlashInfer index tensors from the sequence views it
    is handed, plans the label's wrapper, and runs one FlashInfer call per
    layer. The per-label ``_PlanState`` store belongs here; in CUDA-graph
    mode the graph runner passes in its own per-slot store so the persistent
    wrappers survive across steps.
    """

    def __init__(
        self,
        kv_cache: torch.Tensor | None,
        kv_cache_config: KVCacheConfig,
        buffer_manager: WorkspaceBufferManager | None,
        device,
        states: dict[str, _PlanState] | None = None,
        cuda_graph_mode: bool = False,
        enable_nvtx: bool = False,
    ):
        self.kv_cache = kv_cache
        self.kv_cache_config = kv_cache_config
        self.buffer_manager = buffer_manager
        self.device = device
        self.states = states if states is not None else {}
        self.cuda_graph_mode = cuda_graph_mode
        self.enable_nvtx = enable_nvtx

    def plan(
        self,
        label: str,
        segments: list[Segment],
        views: list[SequenceView],
        dtype: torch.dtype | None = None,
        is_causal: bool = True,
        write_store: bool = True,
    ) -> None:
        """Plan one label's attention over the given segment/view pairs."""
        from mstar.utils.profiler import range_pop, range_push

        assert self.kv_cache is not None

        # Default the FlashInfer wrapper's dtype to whatever dtype the KV
        # cache tensor was actually allocated in. Hardcoding bf16 here breaks
        # any model that runs in fp32 (the wrapper would try to write
        # bf16-cast K/V into an fp32 cache and torch raises a dtype mismatch
        # in flashinfer_utils.set_kv_cache).
        if dtype is None:
            dtype = self.kv_cache.dtype

        cfg = self.kv_cache_config
        page_size = cfg.page_size

        # CPU-side accumulation: only the four int32 tensors the FlashInfer
        # wrapper consumes are built, in pure Python, one H2D each.
        if self.enable_nvtx:
            range_push("cache.plan_attention.build_lists", synchronize=False)
        try:
            qo_indptr_list = [0]
            kv_indptr_list = [0]
            all_page_indices = []
            kv_last_page_lens = []
            kv_cache_locations_list = []

            for segment, view in zip(segments, views, strict=True):
                qo_indptr_list.append(qo_indptr_list[-1] + segment.span)
                all_page_indices.extend(view.page_indices)
                kv_indptr_list.append(kv_indptr_list[-1] + len(view.page_indices))

                last_page_len = view.length % page_size or page_size
                kv_last_page_lens.append(last_page_len)
                if segment.span == 1:
                    kv_cache_locations_list.append([view.page_indices[-1], last_page_len - 1])
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
                if len(kv_cache_locations_list) == len(segments)
                else None
            )
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

        seq_lens = [segment.span for segment in segments]
        is_decode = all([sl == 1 for sl in seq_lens])
        ps = self.states.get(label)
        if ps is not None and ps.wrapper is not None:
            wrapper = ps.wrapper
        elif is_decode:
            wrapper = FlashInferDecodeWrapper(
                workspace_buffer=self.buffer_manager.get(label),
                num_qo_heads=cfg.num_qo_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                page_size=page_size,
                device=self.device,
                enable_nvtx=self.enable_nvtx,
                backend=cfg.flashinfer_backend,
            )
            ps = _PlanState(wrapper=wrapper)
            self.states[label] = ps
        else:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self.buffer_manager.get(label),
                num_qo_heads=cfg.num_qo_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                page_size=page_size,
                device=self.device,
                enable_nvtx=self.enable_nvtx,
                backend=cfg.flashinfer_backend,
            )
            ps = _PlanState(wrapper=wrapper)
            self.states[label] = ps

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
        # run_attention.
        ps.seq_lens = seq_lens
        ps.write_store = write_store
        # A paged plan clears any prior dense plan (set by the dense-gen
        # manager) so run routes this label back through the wrapper.
        ps.dense_gen = None

    def plan_batched_cfg(
        self,
        combined_label: str,
        segments: list[Segment],
        views: list[SequenceView],
        dtype=torch.bfloat16,
        is_causal: bool = False,
        write_store: bool = False,
    ) -> None:
        """Plan a single FlashInfer batch across multiple cache labels.

        ``segments``/``views`` arrive label-major: every (label, request)
        pair in batch order, exactly the packed layout the CFG forward uses.
        """
        assert self.kv_cache is not None

        cfg = self.kv_cache_config
        page_size = cfg.page_size

        # CPU-side accumulation (see plan for the same pattern).
        qo_indptr_list = [0]
        kv_indptr_list = [0]
        all_page_indices = []
        kv_last_page_lens = []
        combined_seq_lens = []

        for segment, view in zip(segments, views, strict=True):
            qo_indptr_list.append(qo_indptr_list[-1] + segment.span)
            all_page_indices.extend(view.page_indices)
            kv_indptr_list.append(
                kv_indptr_list[-1] + len(view.page_indices)
            )

            last_page_len = view.length % page_size or page_size
            kv_last_page_lens.append(last_page_len)
            combined_seq_lens.append(segment.span)

        # CPU tensors — see comment in ``plan`` above. FlashInfer
        # async-H2Ds these inside ``plan()``; passing GPU tensors would
        # trigger a synchronous default-stream sync via the internal
        # ``.to("cpu")`` call.
        qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32)
        paged_kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32)
        paged_kv_indices = torch.tensor(all_page_indices, dtype=torch.int32)
        paged_kv_last_page_len = torch.tensor(kv_last_page_lens, dtype=torch.int32)

        ps = self.states.get(combined_label)
        if self.cuda_graph_mode and ps is not None and ps.wrapper is not None:
            # CUDA-graph mode: reuse the persistent wrapper across denoise steps.
            # plan() updates its static buffers via .copy_() so the captured
            # kernel picks up each step's page table without reallocating.
            wrapper = ps.wrapper
        elif self.cuda_graph_mode:
            # First call under capture: build the persistent wrapper sized for the
            # fixed batch (labels x requests) and token budget.
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self.buffer_manager.get(combined_label),
                num_qo_heads=cfg.num_qo_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                page_size=page_size,
                batch_size=len(segments),
                max_total_tokens=sum(combined_seq_lens),
                max_num_pages=cfg.max_num_pages,
                device=self.device,
                use_cuda_graph=True,
                enable_nvtx=self.enable_nvtx,
                backend=cfg.flashinfer_backend,
            )
            ps = _PlanState(wrapper=wrapper)
            self.states[combined_label] = ps
        else:
            # Eager mode: a fresh wrapper each call (the plan surface is rebuilt
            # per forward, so there is nothing persistent to reuse).
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self.buffer_manager.get(combined_label),
                num_qo_heads=cfg.num_qo_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                page_size=page_size,
                enable_nvtx=self.enable_nvtx,
                backend=cfg.flashinfer_backend,
            )
            ps = _PlanState(wrapper=wrapper)
            self.states[combined_label] = ps

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
        ps.dense_gen = None

    def qo_indptr_buf(self, label: str) -> torch.Tensor | None:
        """The persistent qo_indptr static buffer of ``label``'s CUDA-graph
        prefill wrapper, or None outside that mode. Captured prefill paths
        read it to recover per-request token boundaries from inside the
        captured region: ``plan`` updates the buffer via ``.copy_()``
        outside the graph, so the address stays stable across replay."""
        ps = self.states.get(label)
        if ps is None or ps.wrapper is None:
            return None
        return getattr(ps.wrapper, "_qo_indptr_buf", None)

    def write_kv(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, label: str) -> None:
        """Write this step's K/V into the paged cache at the label's planned
        positions. Separate from ``run`` so strategies that attend without
        storing (or store without attending) stay expressible."""
        self.states[label].wrapper.set_kv_cache(self.kv_cache[layer_idx], k, v)

    def run(self, q: torch.Tensor, layer_idx: int, label: str) -> torch.Tensor:
        """Run the label's pre-planned paged attention for one layer."""
        return self.states[label].wrapper.run(q, self.kv_cache[layer_idx])


class DenseGenAttentionManager(FlashInferAttentionManager):
    """FlashInfer manager with a dense generation-attention fast path.

    Runs non-causal generation attention as a dense FlashAttention-3 pass
    over a contiguous [frozen-prefix | fresh] sequence instead of the paged
    FlashInfer prefill. Diffusion recomputes every generation K/V each step
    (only the tiny text prefix is reused), so the paged path's per-step
    full-buffer K/V write is pure overhead here; a dense pass gathers the
    small prefix, concatenates it with the freshly projected K/V, and runs
    one varlen kernel.
    """

    def plan_dense(
        self,
        label: str,
        entries: list[tuple[SequenceView, int, KVRequestState]],
        seq_lens,
        write_store: bool,
    ) -> None:
        """Build the dense gather/varlen plan for ``label`` and store it on
        the label's plan state. ``entries`` are (frozen-prefix view, fresh
        generation length, request state) per batch element, in the packed
        batch order; the request state carries the cross-step cache slot for
        the gathered prefix."""
        ps = self.states.get(label) or _PlanState()
        self.states[label] = ps
        ps.seq_lens = seq_lens
        ps.write_store = write_store
        ps.dense_gen = self._build_dense_plan(entries)

    def _build_dense_plan(
        self, entries: list[tuple[SequenceView, int, KVRequestState]]
    ) -> dict:
        """Pre-compute the per-segment gather + varlen layout for the dense
        generation-attention path. Each segment attends its fresh generation
        tokens over its frozen text prefix; the prefix lives in the pages
        written at prefill, so we record the page indices to gather it from
        (the same across all layers) and the cumulative-sequence-length
        tensors a single varlen kernel needs. Built once per denoise step,
        reused by every layer's run."""
        page_size = self.kv_cache_config.page_size
        segs = []  # (prefix_page_indices, prefix_len, gen_len, state)
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        for view, gen_len, state in entries:
            prefix_len = view.length
            n_pages = (prefix_len + page_size - 1) // page_size
            idx = torch.tensor(
                view.page_indices[:n_pages], dtype=torch.long, device=self.device
            )
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
    def run_dense(
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


class CrossAttentionManager:
    """Cross-attention over a fixed encoder context (issue #160).

    Cross-attention is non-causal attention over a separate, read-only
    context KV: written once at encode time (``write_context``), planned per
    step against the decoder's query lengths (``plan``), and executed per
    layer (``run``). The context lives in a per-source pool whose head
    config may differ from the decoder's self-attention; plan state rides
    the shared per-label store under the resolved
    ``{label}::CROSS_ATTN::{source}`` label. Cross labels never plan RoPE —
    context positions, if any, are baked in at encode time.
    """

    def __init__(
        self,
        source: str,
        pool: CrossAttnPool,
        kv_pool: KVCachePool,
        buffer_manager: WorkspaceBufferManager | None,
        device,
        states: dict[str, _PlanState],
        enable_nvtx: bool = False,
    ):
        self.source = source
        self.pool = pool
        self.kv_pool = kv_pool
        self.buffer_manager = buffer_manager
        self.device = device
        self.states = states
        self.enable_nvtx = enable_nvtx

    def write_context(
        self,
        request_ids: list[str],
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        seq_lens: list[int] | None,
        base_label: str,
    ) -> None:
        """Write encoder-context K/V for one layer into the source's pool.

        Called once per request per layer at encode time. ``k``/``v`` are
        packed ``(total_context_tokens, num_kv_heads, head_dim)`` across
        ``request_ids``; ``seq_lens`` gives per-request context lengths
        (defaults to a single request owning the whole tensor). Pages are
        allocated on first write (layer 0) and reused for the rest.
        """
        cross_label = cross_attn_label(base_label, self.source)
        page_size = self.pool.alloc_config.page_size

        if seq_lens is None:
            assert len(request_ids) == 1, (
                "add_cross_attn_kv needs seq_lens for multi-request batches"
            )
            seq_lens = [k.shape[0]]

        offset = 0
        for rid, ctx_len in zip(request_ids, seq_lens, strict=True):
            resident = self.kv_pool.view(Segment(rid, cross_label, 0)).length
            if resident == 0:
                # First write for this context: reserve its pages and commit
                # the extent up front; later layers reuse them. The position
                # counter stays untouched (context positions are baked in at
                # encode time).
                segment = Segment(rid, cross_label, ctx_len)
                self.kv_pool.admit(segment)
                self.kv_pool.commit(segment, pos_advance=0)
            else:
                assert resident == ctx_len, (
                    f"cross-attn context for {rid!r}/{self.source!r} already "
                    f"written with length {resident}, got {ctx_len}"
                )
            view = self.kv_pool.view(Segment(rid, cross_label, 0))

            positions = torch.arange(ctx_len, device=self.device)
            page_indices = torch.tensor(
                view.page_indices, dtype=torch.long, device=self.device,
            )
            token_to_page = page_indices[
                torch.div(positions, page_size, rounding_mode="floor")
            ]
            token_to_cache = positions % page_size

            layer_cache = self.pool.kv_cache[layer_idx]
            dtype = self.pool.kv_cache.dtype
            layer_cache[token_to_page, 0, token_to_cache] = \
                k[offset:offset + ctx_len].to(dtype)
            layer_cache[token_to_page, 1, token_to_cache] = \
                v[offset:offset + ctx_len].to(dtype)
            offset += ctx_len

    def plan(
        self,
        request_ids: list[str],
        q_seq_lens: list[int],
        dtype: torch.dtype | None,
        base_label: str,
    ) -> None:
        """Plan the cross-attention wrapper for this step's decoder queries.

        ``q_seq_lens`` is the number of decoder query tokens per request
        (matches the self-attention plan's seq_lens). The context side is
        read from the pool state written by ``write_context`` — pages are
        fixed, so nothing is allocated here. Always uses the prefill wrapper
        with ``causal=False`` (decode-style single queries still attend to
        the full context).
        """
        cross_label = cross_attn_label(base_label, self.source)
        cfg = self.pool.alloc_config
        page_size = cfg.page_size

        if dtype is None:
            dtype = self.pool.kv_cache.dtype

        page_indices_per_request: list[tuple[int, ...]] = []
        context_lens: list[int] = []
        for rid in request_ids:
            view = self.kv_pool.view(Segment(rid, cross_label, 0))
            assert view.length > 0, (
                f"plan_cross_attention before add_cross_attn_kv for {rid!r} "
                f"(source {self.source!r})"
            )
            page_indices_per_request.append(view.page_indices)
            context_lens.append(view.length)

        indptrs = build_paged_indptrs(
            q_seq_lens, page_indices_per_request, context_lens, page_size,
        )

        # Skip the re-plan when its inputs are unchanged (common decode case).
        # Keyed on the page indices so a reused request id with a new context
        # can't alias a stale plan.
        plan_key = PlanCacheKey(
            q_seq_lens=tuple(q_seq_lens),
            page_indices=tuple(indptrs.paged_kv_indices.tolist()),
            last_page_lens=tuple(indptrs.paged_kv_last_page_len.tolist()),
            dtype=dtype,
        )

        ps = self.states.get(cross_label)
        if ps is not None and ps.wrapper is not None and ps.plan_cache_key == plan_key:
            # plan_key carries q_seq_lens, so a match means ps.seq_lens (set at
            # plan time below) already equals the current q_seq_lens.
            return
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
            self.states[cross_label] = ps

        ps.wrapper.plan(
            qo_indptr=indptrs.qo_indptr,
            paged_kv_indptr=indptrs.paged_kv_indptr,
            paged_kv_indices=indptrs.paged_kv_indices,
            paged_kv_last_page_len=indptrs.paged_kv_last_page_len,
            causal=False,
            dtype=dtype,
        )
        ps.seq_lens = q_seq_lens
        ps.write_store = False
        ps.plan_cache_key = plan_key

    def run(self, q: torch.Tensor, layer_idx: int, base_label: str) -> torch.Tensor:
        """Run pre-planned cross-attention against the source's context pool.

        The context K/V were written once by ``write_context``; nothing is
        written here.
        """
        cross_label = cross_attn_label(base_label, self.source)
        ps = self.states.get(cross_label)
        assert ps is not None and ps.wrapper is not None, (
            f"run_cross_attn before plan_cross_attention (label {cross_label!r})"
        )
        return ps.wrapper.run(q, self.pool.kv_cache[layer_idx])
