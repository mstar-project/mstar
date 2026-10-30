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

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# The head dims FlashInfer instantiates its paged prefill/decode kernels (and
# the split-KV merge, DISPATCH_HEAD_DIM) for. Other sizes raise inside
# FlashInfer (32, 48), crash with an illegal memory access (80, 96, 112) or --
# worst -- run and return wrong values: head_dim 72 gives max |diff| 2.25
# against a torch reference where 128 gives 0.004 (FlashInfer 0.6.18, H100,
# 2026-09-20; 192 computes exactly but has no merge kernel). Refuse them up front.
FLASHINFER_HEAD_DIMS = frozenset({64, 128, 256})


def check_flashinfer_head_dim(head_dim: int, device: torch.device | None = None) -> None:
    """Raise unless FlashInfer's paged attention computes ``head_dim`` exactly.
    Only CUDA runs FlashInfer, so a non-CUDA ``device`` (CPU tests driving the
    wrappers with a stand-in module and toy head dims) is not checked."""
    if device is not None and device.type != "cuda":
        return
    if head_dim not in FLASHINFER_HEAD_DIMS:
        raise ValueError(
            f"FlashInfer paged attention: head_dim {head_dim} is not one of "
            f"{sorted(FLASHINFER_HEAD_DIMS)}. FlashInfer builds its paged prefill/decode and "
            "split-KV merge kernels for these sizes only; other head dims fail inside FlashInfer "
            "or run and return wrong values (72 does). Zero-pad the model's q/k/v to the next "
            "supported size (exact: the padded dims add nothing to the scores, the padded output "
            "dims are zeros to drop) and declare that size as the KV cache's head_dim."
        )


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
        accelerator_graph: if True, pre-allocate static buffers for graph capture
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
        accelerator_graph: bool = False,
        enable_nvtx: bool = False,
        backend: str = "auto",
    ):
        check_flashinfer_head_dim(head_dim, device)
        self.device = device
        self.accelerator_graph = accelerator_graph
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
        if self.accelerator_graph:
            assert batch_size is not None, "batch_size required for accelerator graph mode"
            assert max_total_tokens is not None, "max_total_tokens required for accelerator graph mode"
            assert max_num_pages is not None, "max_num_pages required for accelerator graph mode"

            # Pre-allocate static index buffers
            self._qo_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._paged_kv_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._paged_kv_indices_buf = torch.zeros(
                max_num_pages + batch_size, dtype=torch.int32, device=device
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
        # page-locked staging for an inline plan's host arrays (`_stage_pinned`)
        self._pinned: dict[str, torch.Tensor] = {}
        self._pinned_done = None

    def _stage_pinned(self, qo_indptr, kv_indptr, last_page_len, kv_lens, indices):
        """Copy the plan's host arrays into this wrapper's page-locked buffers,
        once the previous plan's copies out of them have landed."""
        if self._pinned_done is None:
            self._pinned_done = torch.cuda.Event()
        else:
            # a plan or more back on this wrapper's slot: long since done
            self._pinned_done.synchronize()
        if kv_lens is None:
            # FlashInfer's `get_seq_lens`, here so it is staged with the rest
            kv_lens = (kv_indptr[1:] - kv_indptr[:-1] - 1).clamp_min(0) * self.page_size \
                + last_page_len
        out = []
        for name, t in (
            ("qo", qo_indptr), ("kv", kv_indptr), ("last", last_page_len),
            ("lens", kv_lens), ("indices", indices),
        ):
            n = t.shape[0]
            buf = self._pinned.get(name)
            if buf is None or buf.shape[0] < n:
                buf = self._pinned[name] = torch.empty(
                    max(n, 2 * (buf.shape[0] if buf is not None else 0), 256),
                    dtype=torch.int32, pin_memory=True,
                )
            view = buf[:n]
            view.copy_(t)
            out.append(view)
        qo, kv, last, lens, idx = out
        # keep the fast row iteration FlashInfer's `max(seq_lens)` relies on
        from mstar.engine.resources.kv.plan import HostLens
        return qo, kv, last, lens.as_subclass(HostLens), idx.shape[0]

    @torch.compiler.disable
    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        kv_lens: torch.Tensor | None = None,
        window_left: int = -1,
        **kwargs
    ):
        """Plan attention and compute KV write indices.

        ``window_left`` is the number of preceding positions each query sees;
        -1 is unbounded.

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
        if self.accelerator_graph and paged_kv_indices.device.type != "cuda":
            # planned inline behind the step in flight, which a pageable copy
            # would wait for: stage page-locked, copy the indices in ourselves
            qo_indptr, paged_kv_indptr, paged_kv_last_page_len, kv_lens, n = (
                self._stage_pinned(
                    qo_indptr, paged_kv_indptr, paged_kv_last_page_len,
                    kv_lens, paged_kv_indices,
                )
            )
            self._paged_kv_indices_buf[:n].copy_(
                self._pinned["indices"][:n], non_blocking=True,
            )
            paged_kv_indices = self._paged_kv_indices_buf[:n]
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
            # host lengths, so plan neither rebuilds them nor copies pageable ones
            seq_lens=kv_lens,
            window_left=window_left,
        )
        if self._pinned_done is not None:
            # every copy out of the staging buffers is issued: the next
            # restage waits for them
            self._pinned_done.record()

        # Allow the qo_indptr to be accessible by BatchedCacheManager.get_qo_indptr_buf,
        # even if we're not in a cuda graph
        if not self.accelerator_graph:
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
        accelerator_graph: if True, pre-allocate static buffers for graph capture
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
        accelerator_graph: bool = False,
        enable_nvtx: bool = False,
        backend: str = "auto",
        use_tensor_cores: bool = True,
    ):
        check_flashinfer_head_dim(head_dim, device)
        self.device = device
        self.accelerator_graph = accelerator_graph
        self.enable_nvtx = enable_nvtx
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = None

        import flashinfer

        if self.accelerator_graph:
            assert batch_size is not None, "batch_size required for accelerator graph mode"
            assert max_num_pages is not None, "max_num_pages required for accelerator graph mode"

            self._paged_kv_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            # one index per resident page, plus one SINK_PAGE per padding row
            # (a full arena under a wide bucket needs the extra `batch_size`)
            self._paged_kv_indices_buf = torch.zeros(
                max_num_pages + batch_size, dtype=torch.int32, device=device
            )
            self._paged_kv_last_page_len_buf = torch.ones(
                batch_size, dtype=torch.int32, device=device
            )

            self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=use_tensor_cores,
                paged_kv_indptr_buffer=self._paged_kv_indptr_buf,
                paged_kv_indices_buffer=self._paged_kv_indices_buf,
                paged_kv_last_page_len_buffer=self._paged_kv_last_page_len_buf,
                backend=backend,
            )
        else:
            self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, "NHD",
                use_tensor_cores=use_tensor_cores,
                backend=backend,
            )

    def plan(
        self,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        kv_lens: torch.Tensor | None = None,
        window_left: int = -1,
        **kwargs
    ):
        """Plan decode attention and compute KV write locations.

        For decode, each request appends exactly 1 token. The write
        location is the last page at position = last_page_len (before
        the append; after append it becomes last_page_len).
        ``window_left`` is as in the prefill wrapper's ``plan``.

        Inputs may be on CPU; see prefill wrapper's plan docstring.
        """
        n_req = paged_kv_indptr.shape[0] - 1
        if (
            self.accelerator_graph and paged_kv_indices.device.type != "cuda"
            and paged_kv_indices.is_pinned()
        ):
            # FlashInfer copies host indices with a blocking copy; page-locked
            # indices (the KV plan's ring, MSTAR_KV_PINNED_INDPTRS) go in one
            # non-blocking copy into its own buffer instead, and its copy_ of
            # that buffer onto itself is a no-op. Pageable indices keep
            # FlashInfer's path, so the knob is a full toggle.
            n = paged_kv_indices.shape[0]
            self._paged_kv_indices_buf[:n].copy_(paged_kv_indices, non_blocking=True)
            paged_kv_indices = self._paged_kv_indices_buf[:n]

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
                # host lengths, so plan skips rebuilding them (`get_seq_lens`)
                seq_lens=kv_lens,
                window_left=window_left,
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


class FlashInferMixedWrapper:
    """A step mixing one-token rows (decode) with prompt rows, on two plans.

    The engine orders a split walk's rows decode first (``CombinedSplit``),
    so the step's leading run of one-token rows is its first tokens, row i
    token i. The decode half plans the bucket's ``batch_size`` rows: a row of
    the run attends its own pages, and any other reads the sink page and its
    output is dropped (with tensor cores those rows get no query, except the
    last, which takes the tokens past the run). The prefill wrapper takes every
    token: the run as one row on the sink page, at most ``max_prefill_rows``
    prompt rows, and a sink-page tail row over the bucket's padding. Both
    read the step's queries in place, at static shapes, and the decode
    outputs overwrite the run's.

    A mixed step plans inline on the gpu thread, behind the step in flight,
    so every host array is staged page-locked: the CUDA-core decode half's in
    this wrapper's buffers, the prefill wrapper's and the tensor-core half's in
    their own (``_stage_pinned``).
    """

    def __init__(
        self,
        decode_workspace: torch.Tensor,
        prefill_workspace: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        batch_size: int,
        max_total_tokens: int,
        max_num_pages: int,
        max_prefill_rows: int,
        device: torch.device,
        backend: str = "auto",
        use_tensor_cores: bool = True,
    ):
        from mstar.engine.resources.kv.plan import SINK_PAGE

        self._sink_page = SINK_PAGE
        self._page_size = page_size
        self._bs = batch_size
        self._tokens = max_total_tokens
        self._max_prefill_rows = max_prefill_rows
        self._device = device
        kv = dict(
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, page_size=page_size, device=device, backend=backend,
        )
        self._tensor_cores = use_tensor_cores
        if use_tensor_cores:
            # The kernel FlashInfer's tensor-core decode runs (FA2 paged
            # prefill), planned with this step's qo_indptr: its KV split budget
            # (2 CTAs per SM over the KV heads) charges each row with a query,
            # so the rows outside the run take none, where the decode wrapper
            # gives each one token. One row carries the outside rows' tokens.
            self._decode = FlashInferPrefillWrapper(
                workspace_buffer=decode_workspace, batch_size=batch_size,
                max_total_tokens=batch_size, max_num_pages=max_num_pages,
                accelerator_graph=True,
                **{**kv, "backend": "fa2" if backend == "auto" else backend},
            )
        else:
            self._decode = FlashInferDecodeWrapper(
                workspace_buffer=decode_workspace, batch_size=batch_size,
                max_num_pages=max_num_pages, accelerator_graph=True,
                use_tensor_cores=False, **kv,
            )
        # a row ahead of the prompt rows takes the decode run's tokens, and one
        # behind them the bucket's padding (see `plan`)
        self._prefill = FlashInferPrefillWrapper(
            workspace_buffer=prefill_workspace, batch_size=max_prefill_rows + 2,
            max_total_tokens=max_total_tokens, max_num_pages=max_num_pages,
            accelerator_graph=True, **kv,
        )
        self._dec_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # the whole step's qo_indptr, which `select_last_hidden` reads
        self._qo_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self._pinned: dict[str, torch.Tensor] = {}
        self._done = None
        self.dtype = torch.bfloat16

    def _pin(self, name: str, values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        """``values`` in this wrapper's page-locked buffer ``name``."""
        n = len(values)
        buf = self._pinned.get(name)
        if buf is None or buf.shape[0] < n:
            buf = self._pinned[name] = torch.empty(
                max(n, 256, 2 * (buf.shape[0] if buf is not None else 0)),
                dtype=dtype, pin_memory=True,
            )
        view = buf[:n]
        view.numpy()[:] = values
        return view

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        kv_lens: torch.Tensor | None = None,
        window_left: int = -1,
        **kwargs,
    ):
        from mstar.engine.resources.kv.plan import HostLens

        self.dtype = dtype
        if self._done is None:
            self._done = torch.cuda.Event()
        else:
            # this wrapper's previous plan, a step or more back: long since done
            self._done.synchronize()

        qo = qo_indptr.numpy().astype(np.int64)
        kvp = paged_kv_indptr.numpy().astype(np.int64)
        idx = paged_kv_indices.numpy()
        last = paged_kv_last_page_len.numpy()
        npages = np.diff(kvp)
        lens = (
            np.asarray(kv_lens, dtype=np.int64) if kv_lens is not None
            # FlashInfer's `get_seq_lens`
            else np.maximum(npages - 1, 0) * self._page_size + last
        )
        spans = np.diff(qo)
        rows = len(spans)
        tokens = max(self._tokens, int(qo[-1]))
        one = spans == 1  # a one-token row is a decode query, state or not
        run = rows if one.all() else int(np.argmin(one))
        prompt = run + np.flatnonzero(spans[run:] > 0)
        if len(prompt) > self._max_prefill_rows:
            raise RuntimeError(
                f"FlashInferMixedWrapper: {len(prompt)} prompt rows in a step "
                f"captured for {self._max_prefill_rows}. Possible causes: more than "
                "the scheduler lets in, a decode row behind a prompt row (a broken "
                "row order), or a CombinedSplit that leaves out a multi-token walk"
            )
        dec = np.arange(rows) < run

        # decode: every row of the bucket; a row outside the run reads one
        # token of the sink page, and its output is dropped
        cnt = np.where(dec, npages, 1)
        d_indptr = np.zeros(rows + 1, dtype=np.int32)
        np.cumsum(cnt, out=d_indptr[1:])
        d_indices = np.full(int(d_indptr[-1]), self._sink_page, dtype=np.int32)
        page_row = np.repeat(np.arange(rows), npages)
        keep = dec[page_row]
        within = np.arange(len(idx)) - kvp[page_row]
        d_indices[d_indptr[page_row[keep]] + within[keep]] = idx[keep]
        d_last = np.where(dec, last, 1).astype(np.int32)
        d_lens = np.where(dec, lens, 1).astype(np.int32)

        # prefill: a sink-page row over the run's tokens, the prompt rows,
        # empty padding rows, then a sink-page row over the bucket's padding,
        # so qo_indptr ends at the token count FlashInfer checks
        pad = self._max_prefill_rows - len(prompt)
        tail = tokens - run - int(spans[prompt].sum())

        def sink_row(n):
            n = max(n, 1)
            pages = -(-n // self._page_size)
            return pages, n - (pages - 1) * self._page_size, n

        lead_pages, lead_last, lead_kv = sink_row(run)
        tail_pages, tail_last, tail_kv = sink_row(tail)
        p_spans = np.concatenate([[run], spans[prompt], np.zeros(pad, dtype=np.int64), [tail]])
        p_qo = np.zeros(self._max_prefill_rows + 3, dtype=np.int32)
        np.cumsum(p_spans, out=p_qo[1:])
        p_cnt = np.concatenate([[lead_pages], npages[prompt], np.ones(pad, dtype=np.int64), [tail_pages]])
        p_kvp = np.zeros(self._max_prefill_rows + 3, dtype=np.int32)
        np.cumsum(p_cnt, out=p_kvp[1:])
        p_idx = np.concatenate(
            [np.full(lead_pages, self._sink_page, dtype=idx.dtype)]
            + [idx[kvp[r]:kvp[r + 1]] for r in prompt]
            + [np.full(pad + tail_pages, self._sink_page, dtype=idx.dtype)]
        ).astype(np.int32)
        p_last = np.concatenate([[lead_last], last[prompt], np.ones(pad), [tail_last]]).astype(np.int32)
        p_lens = np.concatenate([[lead_kv], lens[prompt], np.ones(pad), [tail_kv]]).astype(np.int32)

        self._dec_mask[:rows].copy_(self._pin("dec_mask", dec, torch.bool), non_blocking=True)
        self._qo_indptr_buf[: rows + 1].copy_(
            self._pin("qo", qo.astype(np.int32), torch.int32), non_blocking=True,
        )
        self._num_rows = rows

        if self._tensor_cores:
            # the run's rows one token each, and the last row the rest, so
            # qo_indptr ends at the row count FlashInfer checks
            d_qo = np.zeros(rows + 1, dtype=np.int32)
            d_qo[1 : run + 1] = np.arange(1, run + 1)
            d_qo[run + 1 :] = run
            d_qo[-1] = rows
            self._decode.plan(
                qo_indptr=torch.from_numpy(d_qo),
                paged_kv_indptr=torch.from_numpy(d_indptr),
                paged_kv_indices=torch.from_numpy(d_indices),
                paged_kv_last_page_len=torch.from_numpy(d_last),
                kv_lens=torch.from_numpy(d_lens).as_subclass(HostLens),
                causal=False, dtype=dtype, window_left=window_left,
            )
        else:
            self._decode.plan(
                paged_kv_indptr=self._pin("d_indptr", d_indptr, torch.int32),
                paged_kv_indices=self._pin("d_indices", d_indices, torch.int32),
                paged_kv_last_page_len=self._pin("d_last", d_last, torch.int32),
                kv_lens=self._pin("d_lens", d_lens, torch.int32).as_subclass(HostLens),
                dtype=dtype, window_left=window_left,
            )
        self._prefill.plan(
            qo_indptr=torch.from_numpy(p_qo),
            paged_kv_indptr=torch.from_numpy(p_kvp),
            paged_kv_indices=torch.from_numpy(p_idx),
            paged_kv_last_page_len=torch.from_numpy(p_last),
            kv_lens=torch.from_numpy(p_lens).as_subclass(HostLens),
            causal=causal, dtype=dtype, window_left=window_left,
        )
        self._done.record()

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer: torch.Tensor) -> torch.Tensor:
        n = self._num_rows
        rows = q[:n] if q.shape[0] >= n else torch.cat((q, q.new_zeros((n - q.shape[0], *q.shape[1:]))))
        out = self._prefill.run(q, kv_cache_layer)
        out_rows = self._decode.run(rows, kv_cache_layer)
        m = min(n, out.shape[0])
        out[:m] = torch.where(
            self._dec_mask[:m].view(-1, 1, 1), out_rows[:m].to(out.dtype), out[:m],
        )
        return out[: q.shape[0]]
