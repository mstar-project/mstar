"""FlashInfer utility wrappers for batched paged attention.

Provides:
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

import functools
import logging

import torch

logger = logging.getLogger(__name__)


class FlashInferPrefillWrapper:
    """Batched prefill attention with paged KV cache.

    Wraps flashinfer.BatchPrefillWithPagedKVCacheWrapper with optional
    CUDA graph mode using static buffers. KV writes are the KVManager's job.

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

        self._qo_indptr_buf = None
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
        **kwargs
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

        # Allow the qo_indptr to be accessible by BatchedCacheManager.get_qo_indptr_buf,
        # even if we're not in a cuda graph
        if not self.use_cuda_graph:
            # TODO: take the cuda version as a kwarg
            if qo_indptr.device.type != "cuda":
                qo_indptr = qo_indptr.to(self.device, non_blocking=True)
            self._qo_indptr_buf = qo_indptr

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
        else:
            self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, "NHD",
                use_tensor_cores=True,
                backend=backend,
            )

    def plan(
        self,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs
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


class FlashInferMLAWrapper:
    """FlashInfer MLA wrapper for the 4D latent cache.

    The kernel only supports real Kimi dims (ckv=512, kpe=64); callers must gate
    before construction because off-dim calls can corrupt the CUDA context. CUDA
    graph mode updates static index buffers with ``copy_()``.

    Attention only: the caller scatters the step's latents through the KV plan's
    own addressing (``KVPlanState``) before ``run``.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        *,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        sm_scale: float,
        batch_size: int | None = None,
        max_num_pages: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        backend: str = "auto",
        enable_nvtx: bool = False,
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.enable_nvtx = enable_nvtx
        self.num_heads = num_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.page_size = page_size
        self.sm_scale = sm_scale
        self.batch_size = batch_size
        self.dtype = None

        import flashinfer

        if self.use_cuda_graph:
            assert batch_size is not None, "batch_size required for CUDA graph mode"
            assert max_num_pages is not None, "max_num_pages required for CUDA graph mode"

            # Stable addresses for graph replay.
            self._qo_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._kv_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._kv_indices_buf = torch.zeros(
                max_num_pages, dtype=torch.int32, device=device
            )
            self._kv_len_arr_buf = torch.zeros(
                batch_size, dtype=torch.int32, device=device
            )

            self.attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self._qo_indptr_buf,
                kv_indptr=self._kv_indptr_buf,
                kv_indices=self._kv_indices_buf,
                kv_len_arr=self._kv_len_arr_buf,
                backend=backend,
            )
        else:
            self.attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                workspace_buffer, backend=backend,
            )

    @torch.compiler.disable
    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        *,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Plan the MLA kernel. ``kv_len_arr`` is each request's KV length
        *after* this step's tokens land, one entry per ``qo_indptr`` row."""
        self.dtype = dtype
        self.attn_wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            self.num_heads,
            self.head_dim_ckv,
            self.head_dim_kpe,
            self.page_size,
            causal,
            self.sm_scale,
            dtype,
            dtype,
        )

    @torch.compiler.disable
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Run the planned MLA kernel."""
        return self.attn_wrapper.run(
            q_nope.to(self.dtype), q_pe.to(self.dtype),
            ckv_cache, kpe_cache, return_lse=False,
        )


# FlashAttention-3's MLA decode (``qv``), from a source install or the
# kernels-community/flash-attn3 hub build. Pinned by the build's kernel version.
FA3_KERNEL_REPO = "kernels-community/flash-attn3"
FA3_KERNEL_VERSION = 1


@functools.cache
def load_fa3():
    """The FA3 module, or ``None`` when neither source is available."""
    try:
        import flash_attn_interface as fa3  # a hopper/ source build
        return fa3
    except ImportError:
        pass
    try:
        from kernels import get_kernel

        return get_kernel(FA3_KERNEL_REPO, version=FA3_KERNEL_VERSION)
    except Exception as ex:  # noqa: BLE001 -- not installed, offline, no matching build
        logger.warning("FA3 MLA unavailable (%s); MLA decode stays on FlashInfer", ex)
        return None


class FA3MLAWrapper:
    """FA3 MLA decode over the 4D latent cache, behind ``FlashInferMLAWrapper``'s
    plan/run interface.

    Decode only (one query token per request), so ``max_seqlen_q`` is 1 at
    capture and replay alike. FA3 takes a padded ``page_table`` rather than
    CSR indices and derives ``max_seqlen_k`` from its width, so the table is
    sized to ``max_seq_len`` once and the captured kernel's arguments never
    change. Splitting follows each step's real lengths through the AOT
    scheduler metadata, which ``plan`` recomputes into a persistent buffer
    (zero-padded past its live part), as vLLM's FlashAttnMLA backend does.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        sm_scale: float,
        max_seq_len: int,
        batch_size: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        num_splits: int = 32,
    ):
        self.fa3 = load_fa3()
        assert self.fa3 is not None, "FA3MLAWrapper needs FA3; gate on load_fa3()"
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.num_heads = num_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.page_size = page_size
        self.sm_scale = sm_scale
        self.max_pages_per_seq = -(-max_seq_len // page_size)
        # captured: a fixed split count sizes FA3's intermediates at capture;
        # eager: 0 lets FA3 choose per call
        self.num_splits = num_splits if use_cuda_graph else 0
        self.batch_size = batch_size
        self.dtype = None
        self._rows = 0
        if use_cuda_graph:
            assert batch_size is not None, "batch_size required for CUDA graph mode"
            self._qo_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            self._seqlens_buf = torch.zeros(batch_size, dtype=torch.int32, device=device)
            self._page_table_buf = torch.zeros(
                batch_size, self.max_pages_per_seq, dtype=torch.int32, device=device)
            self._sched_buf: torch.Tensor | None = None
        self._sched: torch.Tensor | None = None

    def _page_table(self, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, rows: int) -> torch.Tensor:
        counts = kv_indptr[1:rows + 1] - kv_indptr[:rows]
        width = max(int(counts.max()), 1) if rows else 1
        table = torch.zeros(rows, width, dtype=torch.int32)
        for i in range(rows):
            n = int(counts[i])
            if n:
                start = int(kv_indptr[i])
                table[i, :n] = kv_indices[start:start + n]
        return table

    @torch.compiler.disable
    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        *,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Same arguments as ``FlashInferMLAWrapper.plan`` (CPU tensors): stage
        the page table and lengths, then schedule the splits on the device."""
        del causal  # one query per request: causal and non-causal coincide
        self.dtype = dtype
        rows = qo_indptr.shape[0] - 1
        assert int(qo_indptr[-1]) == rows, "FA3MLAWrapper serves decode steps only"
        self._rows = rows
        table = self._page_table(kv_indptr, kv_indices, rows)
        if self.use_cuda_graph:
            assert rows <= self.batch_size and table.shape[1] <= self.max_pages_per_seq
            self._qo_indptr_buf[: rows + 1].copy_(qo_indptr)
            self._seqlens_buf[:rows].copy_(kv_len_arr)
            self._page_table_buf[:rows, : table.shape[1]].copy_(table)
            qo, seqlens, pt = self._qo_indptr_buf[: rows + 1], self._seqlens_buf[:rows], self._page_table_buf[:rows]
        else:
            qo = qo_indptr.to(self.device)
            seqlens = kv_len_arr.to(self.device)
            pt = table.to(self.device)
        self._qo, self._seqlens, self._pt = qo, seqlens, pt
        sched = self.fa3.get_scheduler_metadata(
            rows, 1, pt.shape[1] * self.page_size, self.num_heads, 1, self.head_dim_kpe, seqlens,
            dtype, headdim_v=self.head_dim_ckv, cu_seqlens_q=qo, page_size=self.page_size,
            causal=True, num_splits=self.num_splits,
        )
        if self.use_cuda_graph:
            if self._sched_buf is None:
                # sized by the bucket's row count, fixed for this wrapper
                self._sched_buf = torch.zeros(sched.numel(), dtype=sched.dtype, device=self.device)
            n = sched.numel()
            assert n <= self._sched_buf.numel(), (n, self._sched_buf.numel())
            self._sched_buf[:n].copy_(sched)
            self._sched_buf[n:].zero_()
            sched = self._sched_buf
        self._sched = sched

    @torch.compiler.disable
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Run the planned decode: ``[T, H, ckv]`` like the FlashInfer wrapper."""
        return self.fa3.flash_attn_with_kvcache(
            q_pe.to(self.dtype), kpe_cache.unsqueeze(-2), ckv_cache.unsqueeze(-2),
            qv=q_nope.to(self.dtype), cache_seqlens=self._seqlens, page_table=self._pt,
            cu_seqlens_q=self._qo, max_seqlen_q=1, softmax_scale=self.sm_scale, causal=True,
            scheduler_metadata=self._sched, num_splits=self.num_splits,
        )
