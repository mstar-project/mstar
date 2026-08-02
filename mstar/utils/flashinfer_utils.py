"""FlashInfer utility wrappers for batched attention.

Provides:
- run_rms_norm / run_attention: simple single-request helpers
- FlashInferPrefillWrapper: batched prefill with paged KV cache, optional CUDA graph mode
- FlashInferDecodeWrapper: batched decode with paged KV cache, optional CUDA graph mode
- RaggedPrefillWrapper: varlen self-attention with NO KV cache (encoder-style),
  optional CUDA graph mode

CUDA graph mode requires:
- Static buffer pointers passed at construction (qo_indptr_buf, paged_kv_indptr_buf, etc.)
- plan() updates values via .copy_() without reallocating
- The same wrapper object must be used during both capture and replay

The paged wrappers are adapted from VoxServe's flashinfer_utils.py for our KV
cache layout:
  [num_layers, max_pages, 2, page_size, num_kv_heads, head_dim]
(VoxServe uses [n_pages, 2, page_size, n_heads, head_dim] without layer dim.)
"""

import logging

import torch

logger = logging.getLogger(__name__)


@torch.compiler.disable
def run_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-06,
    rms_norm_dtype=None
):
    orig_dtype = input.dtype
    if rms_norm_dtype is not None:
        input = input.to(rms_norm_dtype)
    elif torch.is_autocast_enabled():
        dtype = torch.get_autocast_dtype("cuda")
        input = input.to(dtype)
    elif input.dtype == torch.float32:
        # Unsupported dtype; must recast
        input = input.to(torch.bfloat16)

    # flashinfer.norm.rmsnorm requires matching input/weight dtypes; cast weight
    # to match whatever input ended up as.
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)

    import flashinfer
    return flashinfer.norm.rmsnorm(
        input, weight, eps=eps
    ).to(orig_dtype)


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float=1.0,
    causal: bool=True,
):
    import flashinfer
    return flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=causal,
        sm_scale=scale,
    )


class FlashInferPrefillWrapper:
    """Batched prefill attention with paged KV cache.

    Wraps flashinfer.BatchPrefillWithPagedKVCacheWrapper with:
    - Pre-computed token_to_page / token_to_cache for vectorized KV writes
    - Optional CUDA graph mode with static buffers

    Args:
        workspace_buffer: FlashInfer workspace (256MB+ recommended)
        num_qo_heads: number of query/output heads
        num_kv_heads: number of key/value heads
        head_dim: dimension per head
        page_size: KV cache page size
        batch_size: required for CUDA graph mode (max requests in batch)
        max_total_tokens: required for CUDA graph mode (max total new tokens across batch)
        max_num_pages: required for CUDA graph mode (max pages across all requests)
        device: torch device
        use_cuda_graph: if True, pre-allocate static buffers for graph capture
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        batch_size: int | None = None,
        max_total_tokens: int | None = None,
        max_num_pages: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        enable_nvtx: bool = False,
        backend: str = "auto",
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.enable_nvtx = enable_nvtx
        self.batch_size = batch_size
        self.max_total_tokens = max_total_tokens
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = None

        import flashinfer

        if self.use_cuda_graph:
            assert batch_size is not None, "batch_size required for CUDA graph mode"
            assert max_total_tokens is not None, "max_total_tokens required for CUDA graph mode"
            assert max_num_pages is not None, "max_num_pages required for CUDA graph mode"

            # Pre-allocate static index buffers
            self._qo_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._paged_kv_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._paged_kv_indices_buf = torch.zeros(
                max_num_pages, dtype=torch.int32, device=device
            )
            self._paged_kv_last_page_len_buf = torch.ones(
                batch_size, dtype=torch.int32, device=device
            )

            self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=self._qo_indptr_buf,
                paged_kv_indptr_buf=self._paged_kv_indptr_buf,
                paged_kv_indices_buf=self._paged_kv_indices_buf,
                paged_kv_last_page_len_buf=self._paged_kv_last_page_len_buf,
                backend=backend,
            )

            # Static buffers for vectorized KV cache writes
            self.token_to_page = torch.zeros(
                max_total_tokens, dtype=torch.long, device=device
            )
            self.token_to_cache = torch.zeros(
                max_total_tokens, dtype=torch.long, device=device
            )
        else:
            self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD", backend=backend
            )
            self.token_to_page = None
            self.token_to_cache = None

        self._total_tokens = 0

    @torch.compiler.disable
    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Plan attention and compute KV write indices.

        In CUDA graph mode, updates static buffers via .copy_() so that
        the same GPU addresses are used during graph replay.

        Inputs may be on CPU — that's preferred because FlashInfer's
        ``BatchPrefillWithPagedKVCacheWrapper.plan`` does ``indptr.to("cpu")``
        / ``last_page_len.to("cpu")`` internally; passing GPU tensors there
        triggers a synchronous default-stream sync that drains the
        speculatively-queued next decode step. We let the inner plan
        consume them as CPU and async-H2D copy to the device for our own
        per-token bookkeeping below.
        """
        self.dtype = dtype
        self.attn_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=causal,
            q_data_type=dtype,
        )

        # Async H2D for the GPU-side per-token bookkeeping that follows.
        if qo_indptr.device.type != "cuda":
            qo_indptr = qo_indptr.to(self.device, non_blocking=True)
            paged_kv_indptr = paged_kv_indptr.to(self.device, non_blocking=True)
            paged_kv_indices = paged_kv_indices.to(self.device, non_blocking=True)
            paged_kv_last_page_len = paged_kv_last_page_len.to(self.device, non_blocking=True)

        # Allow the qo_indptr to be accessible by BatchedCacheManager.get_qo_indptr_buf,
        # even if we're not in a cuda graph
        if not self.use_cuda_graph:
            self._qo_indptr_buf = qo_indptr

        # Compute per-token page and offset for vectorized KV writes
        n_req = qo_indptr.shape[0] - 1
        starts = qo_indptr[:-1].to(torch.int32)
        lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)
        total_tokens = int(lens.sum().item())
        self._total_tokens = total_tokens

        # Pages/lengths AFTER append
        num_pages_after = (
            paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        ).to(torch.int32)
        kv_len_after = (
            (num_pages_after - 1) * self.page_size + paged_kv_last_page_len
        )

        # Flatten to per-token indices
        seg = torch.repeat_interleave(
            torch.arange(n_req, dtype=torch.int32, device=self.device), lens
        )
        intra = torch.arange(
            total_tokens, dtype=torch.int32, device=self.device
        ) - torch.repeat_interleave(starts, lens)

        # Absolute KV position per token
        start_new = kv_len_after[seg] - lens[seg]
        g = start_new + intra

        # Map to page + offset
        page_off = torch.div(g, self.page_size, rounding_mode="floor").to(
            torch.int32
        )
        off_in_page = (g - page_off * self.page_size).to(torch.int32)
        abs_page_ptr = paged_kv_indptr[:-1][seg] + page_off

        token_to_page = paged_kv_indices[abs_page_ptr].to(torch.long)
        token_to_cache = off_in_page.to(torch.long)

        if self.use_cuda_graph:
            self.token_to_page[:total_tokens].copy_(token_to_page)
            self.token_to_cache[:total_tokens].copy_(token_to_cache)
            if total_tokens < self.max_total_tokens:
                self.token_to_page[total_tokens:] = 0
                self.token_to_cache[total_tokens:] = 0
        else:
            self.token_to_page = token_to_page
            self.token_to_cache = token_to_cache

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer: torch.Tensor) -> torch.Tensor:
        """Run planned batched prefill attention.

        Args:
            q: [total_tokens, num_qo_heads, head_dim]
            kv_cache_layer: [max_pages, 2, page_size, num_kv_heads, head_dim]
                (single layer slice of the full KV cache)
        Returns:
            output: [total_tokens, num_qo_heads, head_dim]
        """
        return self.attn_wrapper.run(q.to(self.dtype), kv_cache_layer)

    def set_kv_cache(
        self,
        kv_cache_layer: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Write K, V to the paged KV cache at pre-computed positions.

        Args:
            kv_cache_layer: [max_pages, 2, page_size, num_kv_heads, head_dim]
            k: [total_tokens, num_kv_heads, head_dim]
            v: [total_tokens, num_kv_heads, head_dim]
        """
        n = self._total_tokens
        page_idx = self.token_to_page[:n]
        cache_idx = self.token_to_cache[:n]
        kv_cache_layer[page_idx, 0, cache_idx] = k[:n].to(self.dtype)
        kv_cache_layer[page_idx, 1, cache_idx] = v[:n].to(self.dtype)


class FlashInferDecodeWrapper:
    """Batched decode attention with paged KV cache.

    Optimized for the common decode case where each request appends
    exactly 1 new token. Uses BatchDecodeWithPagedKVCacheWrapper.

    Args:
        workspace_buffer: FlashInfer workspace
        num_qo_heads: number of query/output heads
        num_kv_heads: number of key/value heads
        head_dim: dimension per head
        page_size: KV cache page size
        batch_size: required for CUDA graph mode (max requests in batch)
        max_num_pages: required for CUDA graph mode (max pages across all requests)
        device: torch device
        use_cuda_graph: if True, pre-allocate static buffers for graph capture
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        batch_size: int | None = None,
        max_num_pages: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        enable_nvtx: bool = False,
        backend: str = "auto",
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.enable_nvtx = enable_nvtx
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = None

        import flashinfer

        if self.use_cuda_graph:
            assert batch_size is not None, "batch_size required for CUDA graph mode"
            assert max_num_pages is not None, "max_num_pages required for CUDA graph mode"

            self._paged_kv_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._paged_kv_indices_buf = torch.zeros(
                max_num_pages, dtype=torch.int32, device=device
            )
            self._paged_kv_last_page_len_buf = torch.ones(
                batch_size, dtype=torch.int32, device=device
            )

            self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=self._paged_kv_indptr_buf,
                paged_kv_indices_buffer=self._paged_kv_indices_buf,
                paged_kv_last_page_len_buffer=self._paged_kv_last_page_len_buf,
                backend=backend,
            )

            # Static buffer for KV write locations: [batch_size, 2] = (page_idx, pos_idx)
            self.kv_cache_locations = torch.zeros(
                batch_size, 2, dtype=torch.long, device=device
            )
        else:
            self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, "NHD",
                use_tensor_cores=True,
                backend=backend,
            )
            self.kv_cache_locations = None

    def plan(
        self,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        kv_cache_locations: torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Plan decode attention and compute KV write locations.

        For decode, each request appends exactly 1 token. The write
        location is the last page at position = last_page_len (before
        the append; after append it becomes last_page_len).

        Inputs may be on CPU; see prefill wrapper's plan docstring.
        """
        n_req = paged_kv_indptr.shape[0] - 1

        if self.enable_nvtx:
            from mstar.utils.profiler import range_pop, range_push

            range_push("flashinfer.decode.plan_inner", synchronize=False)
        try:
            self.attn_wrapper.plan(
                indptr=paged_kv_indptr,
                indices=paged_kv_indices,
                last_page_len=paged_kv_last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=dtype,
            )
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

        # Async H2D before our own per-rid bookkeeping.
        if paged_kv_indptr.device.type != "cuda":
            if self.enable_nvtx:
                range_push("flashinfer.decode.metadata_h2d", synchronize=False)
            try:
                paged_kv_indptr = paged_kv_indptr.to(self.device, non_blocking=True)
                paged_kv_indices = paged_kv_indices.to(self.device, non_blocking=True)
                paged_kv_last_page_len = paged_kv_last_page_len.to(self.device, non_blocking=True)
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)

        if kv_cache_locations is not None:
            locations = kv_cache_locations
            if locations.device.type != "cuda":
                if self.enable_nvtx:
                    range_push("flashinfer.decode.kv_location_h2d", synchronize=False)
                try:
                    locations = locations.to(self.device, non_blocking=True)
                finally:
                    if self.enable_nvtx:
                        range_pop(synchronize=False)
        else:
            # Compute KV write locations: page and position for each request's new token
            if self.enable_nvtx:
                range_push("flashinfer.decode.kv_location_compute", synchronize=False)
            try:
                page_idx = paged_kv_indices[paged_kv_indptr[1:] - 1]
                pos_idx = paged_kv_last_page_len - 1

                locations = torch.stack([page_idx.to(torch.long), pos_idx.to(torch.long)], dim=1)
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)

        if self.use_cuda_graph:
            if self.enable_nvtx:
                range_push("flashinfer.decode.kv_location_copy", synchronize=False)
            try:
                self.kv_cache_locations[:n_req].copy_(locations)
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)
        else:
            self.kv_cache_locations = locations

        self._n_req = n_req
        self.dtype = dtype

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer: torch.Tensor) -> torch.Tensor:
        """Run planned batched decode attention.

        Args:
            q: [n_req, num_qo_heads, head_dim]
            kv_cache_layer: [max_pages, 2, page_size, num_kv_heads, head_dim]
        Returns:
            output: [n_req, num_qo_heads, head_dim]
        """
        return self.attn_wrapper.run(q.to(self.dtype), kv_cache_layer)

    def set_kv_cache(
        self,
        kv_cache_layer: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Write K, V for decode (1 token per request).

        Args:
            kv_cache_layer: [max_pages, 2, page_size, num_kv_heads, head_dim]
            k: [n_req, num_kv_heads, head_dim]
            v: [n_req, num_kv_heads, head_dim]
        """
        n = self._n_req
        pages = self.kv_cache_locations[:n, 0]
        positions = self.kv_cache_locations[:n, 1]
        kv_cache_layer[pages, 0, positions] = k[:n].to(self.dtype)
        kv_cache_layer[pages, 1, positions] = v[:n].to(self.dtype)


# Head dims the FlashInfer prefill kernels are instantiated for. An unsupported
# one fails to BUILD (SM90: static_assert in hopper/prefill_sm90.cuh), so q/k/v
# are zero-padded up to the next supported size. Exact: padded lanes contribute
# 0 to QK^T, and the extra output lanes are sums of zeros. Qwen3-Omni's AuT and
# the SigLIP2-style ViTs use head_dim=72.
SUPPORTED_HEAD_DIMS = (64, 128, 256)


def padded_head_dim(head_dim: int) -> int:
    """Smallest FlashInfer-supported head dim >= ``head_dim``."""
    for supported in SUPPORTED_HEAD_DIMS:
        if head_dim <= supported:
            return supported
    raise ValueError(
        f"head_dim {head_dim} exceeds the largest supported ({SUPPORTED_HEAD_DIMS[-1]})"
    )


class RaggedPrefillWrapper:
    """Varlen self-attention over packed segments, with no KV cache.

    The encoder counterpart to ``FlashInferPrefillWrapper``: audio/vision towers
    attend within variable-length segments packed into one ``[total_tokens, H, D]``
    tensor. ``cu_seqlens`` is both ``qo_indptr`` and ``kv_indptr``.

    A "segment" is one independently-attending span, NOT one request — the audio
    tower window-chunks one clip into many. Size the wrapper by segments.

    Kernel-selecting args (``causal``, ``sm_scale``, dtype, head counts) are fixed
    at construction, not per-plan: a captured graph records one kernel. ``plan``
    takes only the layout.

    CUDA graph mode: capture once per bucket, re-plan freely before each replay.
    Segment count is fixed at ``max_num_segments`` (``plan`` pads fewer with
    zero-length segments, rejects more), and two buckets need two wrappers —
    re-planning a wrapper another graph recorded corrupts that graph.

    Args:
        workspace_buffer: FlashInfer workspace (128MB+ recommended)
        num_qo_heads / num_kv_heads: head counts (equal for these encoders)
        head_dim: TRUE head dim; padded internally to a supported size
        max_num_segments / max_total_tokens: required for CUDA graph mode
        device: torch device
        use_cuda_graph: if True, pre-allocate static indptr buffers for capture
        causal: fixed for the wrapper's lifetime (encoders are bidirectional)
        sm_scale: defaults to ``head_dim ** -0.5`` on the TRUE head dim. Passed
            explicitly because FlashInfer would otherwise derive it from the
            PADDED head dim and silently apply the wrong scale.
        q_data_type: dtype the kernel is planned for; q/k/v are cast to it
        kv_layout: "NHD" or "HND"
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_num_segments: int | None = None,
        max_total_tokens: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        causal: bool = False,
        sm_scale: float | None = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_layout: str = "NHD",
    ):
        self.device = device
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.padded_head_dim = padded_head_dim(head_dim)
        self.causal = causal
        self.sm_scale = float(sm_scale) if sm_scale is not None else head_dim ** -0.5
        self.q_data_type = q_data_type
        self.use_cuda_graph = use_cuda_graph
        self.max_num_segments = max_num_segments
        self.max_total_tokens = max_total_tokens
        self._num_segments = 0

        import flashinfer

        if use_cuda_graph:
            assert max_num_segments is not None, "max_num_segments required for CUDA graph mode"
            assert max_total_tokens is not None, "max_total_tokens required for CUDA graph mode"

            self._qo_indptr_buf = torch.zeros(
                max_num_segments + 1, dtype=torch.int32, device=device
            )
            self._kv_indptr_buf = torch.zeros(
                max_num_segments + 1, dtype=torch.int32, device=device
            )
            self.attn_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer,
                kv_layout,
                use_cuda_graph=True,
                qo_indptr_buf=self._qo_indptr_buf,
                kv_indptr_buf=self._kv_indptr_buf,
            )
            # Host staging for the padded indptr; FlashInfer's plan moves it to
            # host anyway, so building it here avoids a D2H sync per plan.
            self._cu_host = torch.zeros(
                max_num_segments + 1,
                dtype=torch.int32,
                pin_memory=torch.cuda.is_available(),
            )
            # FlashInfer latches max rows on the FIRST plan; prime at the
            # bucket ceiling so a small first plan can't cap it.
            self.plan(self._max_layout_cu_seqlens())
        else:
            self._qo_indptr_buf = None
            self._kv_indptr_buf = None
            self._cu_host = None
            self.attn_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer, kv_layout
            )

    @property
    def num_segments(self) -> int:
        """Real (unpadded) segment count from the most recent ``plan``."""
        return self._num_segments

    def _max_layout_cu_seqlens(self) -> torch.Tensor:
        """``max_total_tokens`` spread over all segments, remainder on the first."""
        n, total = self.max_num_segments, self.max_total_tokens
        lens = [total // n] * n
        lens[0] += total % n
        cu = [0]
        for seg_len in lens:
            cu.append(cu[-1] + seg_len)
        return torch.tensor(cu, dtype=torch.int32)

    def _prepare_cu_seqlens(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        n_seg = int(cu_seqlens.numel()) - 1
        if not self.use_cuda_graph:
            self._num_segments = n_seg
            return cu_seqlens.to(torch.int32)

        if n_seg > self.max_num_segments:
            raise ValueError(
                f"RaggedPrefillWrapper: {n_seg} segments exceeds the "
                f"{self.max_num_segments} this graph-mode wrapper was built for"
            )
        host = cu_seqlens.to(device="cpu", dtype=torch.int32)
        total_tokens = int(host[-1])
        if total_tokens > self.max_total_tokens:
            raise ValueError(
                f"RaggedPrefillWrapper: {total_tokens} tokens exceeds the "
                f"{self.max_total_tokens} this graph-mode wrapper was built for"
            )
        self._cu_host[: n_seg + 1].copy_(host)
        # Repeating the final offset appends zero-length segments — pads the
        # segment count to the fixed size without adding tokens.
        self._cu_host[n_seg + 1:] = total_tokens
        self._num_segments = n_seg
        return self._cu_host

    @torch.compiler.disable
    def plan(self, cu_seqlens: torch.Tensor) -> None:
        """Plan one packed layout. ``cu_seqlens``: ``[num_segments + 1]``, [0] == 0.

        CPU tensor preferred; a GPU one costs a sync. Safe to call before every
        replay — values are copied through the static buffers, not rebound.
        """
        cu = self._prepare_cu_seqlens(cu_seqlens)
        self.attn_wrapper.plan(
            cu,
            cu,
            self.num_qo_heads,
            self.num_kv_heads,
            self.padded_head_dim,
            causal=self.causal,
            sm_scale=self.sm_scale,
            q_data_type=self.q_data_type,
        )

    def _pad_head_dim(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-1] == self.padded_head_dim:
            return t.contiguous()
        return torch.nn.functional.pad(t, (0, self.padded_head_dim - t.shape[-1]))

    @torch.compiler.disable
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run planned varlen self-attention.

        Args:
            q, k, v: [total_tokens, num_heads, head_dim], packed by cu_seqlens
        Returns:
            output: [total_tokens, num_qo_heads, head_dim]

        Rows past the planned ``cu_seqlens[-1]`` are untouched, so an oversized
        static buffer replays fine — the caller slices.
        """
        qp, kp, vp = (self._pad_head_dim(t.to(self.q_data_type)) for t in (q, k, v))
        out = self.attn_wrapper.run(qp, kp, vp)
        if self.padded_head_dim != self.head_dim:
            return out[..., : self.head_dim].contiguous()
        return out
