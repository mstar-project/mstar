"""FlashInfer utility wrappers for batched paged attention.

Provides:
- run_rms_norm / run_attention: simple single-request helpers
- FlashInferPrefillWrapper: batched prefill with paged KV cache, optional CUDA graph mode
- FlashInferDecodeWrapper: batched decode with paged KV cache, optional CUDA graph mode

CUDA graph mode requires:
- Static buffer pointers passed at construction (qo_indptr_buf, paged_kv_indptr_buf, etc.)
- plan() updates values via .copy_() without reallocating
- The same wrapper object must be used during both capture and replay

Adapted from VoxServe's flashinfer_utils.py for our KV cache layout:
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
        dtype = torch.get_autocast_gpu_dtype()
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
                workspace_buffer, "NHD"
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
            )

            # Static buffer for KV write locations: [batch_size, 2] = (page_idx, pos_idx)
            self.kv_cache_locations = torch.zeros(
                batch_size, 2, dtype=torch.long, device=device
            )
        else:
            self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, "NHD",
                use_tensor_cores=True,
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
