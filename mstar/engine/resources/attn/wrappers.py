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

import numpy as np
import torch

logger = logging.getLogger(__name__)

# A captured wrapper's later plans call FlashInfer's planner module directly
# (``_plan_direct``): most of FlashInfer's ``plan()`` is Python that gives the
# same answer every step of a captured bucket. vLLM and SGLang do the same
# (vLLM's fast_plan_decode over FlashInfer's fast_decode_plan, SGLang's
# fast_prefill_plan) with FlashInfer pinned to one version, because the
# direct call restates FlashInfer's planner arguments and per-plan state.
# So it runs only on the versions here, which test_flashinfer_direct_plan.py
# and test_attn_mixed_split.py have compared with ``plan()`` on a GPU; on any
# other, a captured wrapper plans with ``plan()``. pyproject.toml pins the
# same version.
DIRECT_PLAN_FLASHINFER_VERSIONS = frozenset({"0.7.2"})


@functools.cache
def _direct_plan_supported() -> bool:
    import flashinfer

    # a stand-in module (CPU tests) has no version: not a checked one
    version = getattr(flashinfer, "__version__", None)
    if version in DIRECT_PLAN_FLASHINFER_VERSIONS:
        return True
    checked = ", ".join(sorted(DIRECT_PLAN_FLASHINFER_VERSIONS))
    logger.warning(
        "FlashInfer %s is not a version the attention wrappers' direct planner calls were "
        "checked against (%s), so captured wrappers plan with FlashInfer's plan(), which is "
        "slower. Install the pinned version (pip install flashinfer-python==%s) to use them.",
        version, checked, max(
            DIRECT_PLAN_FLASHINFER_VERSIONS,
            key=lambda v: tuple(int(p) for p in v.split(".") if p.isdigit()),
        ),
    )
    return False


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


def _prefill_plan_is_direct(w) -> bool:
    """Whether FlashInfer prefill wrapper ``w``'s plan resolved to the one
    path ``_plan_direct`` reproduces, on a checked FlashInfer version: fa2 or
    fa3 under a graph, no custom mask, prefix items, window, soft cap or
    position encoding, a 16-bit KV."""
    return (
        _direct_plan_supported()
        and w._backend in ("fa2", "fa3") and w.is_cuda_graph_enabled and w._jit_module is None
        and w._cached_module is not None and w._custom_mask_buf is None
        and w._prefix_len_ptr is None and w._window_left == -1
        and w._logits_soft_cap == 0.0 and w._pos_encoding_mode == "NONE"
        and w._cached_kv_data_type in (torch.bfloat16, torch.float16)
        and w._max_total_num_rows is not None
    )


# The two FlashInfer decode planners ``_plan_direct`` reproduces: the batch
# prefill module's (tensor-core decode) and the batch decode module's (CUDA-core
# decode, which an MHA model such as Whisper takes: the tensor-core kernel pads
# one query head per KV head to a tile).
_TENSOR_CORE_DECODE, _CUDA_CORE_DECODE = "tensor_cores", "cuda_cores"


def _decode_direct_variant(w) -> str | None:
    """Which planner FlashInfer decode wrapper ``w``'s plan resolved to, when
    ``_plan_direct`` reproduces it on a checked FlashInfer version: fa2
    tensor-core or generic CUDA-core decode, under a graph, one query token per
    row, no window, no soft cap. None otherwise. CUDA-core decode takes the
    generic module under "auto" (unless it resolved to cuDNN) and "fa2" alike."""
    if not (
        _direct_plan_supported() and w.is_cuda_graph_enabled
        and not getattr(w, "_uses_cudnn", False) and w._jit_module is None
        and getattr(w, "_q_len_per_req", 1) in (None, 1) and w._window_left == -1
    ):
        return None
    if w.use_tensor_cores:
        return _TENSOR_CORE_DECODE if w._backend == "fa2" else None
    if w._backend in ("auto", "fa2") and not w._logits_soft_cap:
        return _CUDA_CORE_DECODE
    return None


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
        # set by the first staged plan when ``_plan_direct`` may take the later
        # ones, for the (dtype, causal) it was built for
        self._planner_args: list | None = None
        self._planner_key = None

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
        if self.accelerator_graph and paged_kv_indices.device.type != "cuda":
            # planned inline behind the step in flight, which a pageable copy
            # would wait for: stage page-locked, copy the indices in ourselves
            qo_indptr, paged_kv_indptr, paged_kv_last_page_len, kv_lens, n = (
                self._stage_pinned(
                    qo_indptr, paged_kv_indptr, paged_kv_last_page_len,
                    kv_lens, paged_kv_indices,
                )
            )
            try:
                self._plan_staged(
                    qo_indptr, paged_kv_indptr, self._pinned["indices"][:n],
                    paged_kv_last_page_len, kv_lens, causal, dtype, window_left,
                )
            finally:
                # also on a raise: copies already issued still read the pinned buffers
                self._pinned_done.record()
            return
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
            # host lengths, so plan neither rebuilds them nor copies pageable ones
            seq_lens=kv_lens,
            window_left=window_left,
        )

        # Allow the qo_indptr to be accessible by BatchedCacheManager.get_qo_indptr_buf,
        # even if we're not in a cuda graph
        if not self.accelerator_graph:
            # TODO: take the cuda version as a kwarg
            if qo_indptr.device.type != "cuda":
                qo_indptr = qo_indptr.to(self.device, non_blocking=True)
            self._qo_indptr_buf = qo_indptr

    def _plan_staged(
        self, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
        kv_lens, causal: bool, dtype: torch.dtype, window_left: int = -1,
    ) -> None:
        """Plan a captured wrapper from page-locked int32 host arrays, which
        the caller keeps unchanged until this plan's copies have landed."""
        n = paged_kv_indices.shape[0]
        self.dtype = dtype
        key = (dtype, causal, window_left)
        if self._planner_key == key:
            self._plan_direct(
                qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, kv_lens,
            )
            return
        self._paged_kv_indices_buf[:n].copy_(paged_kv_indices, non_blocking=True)
        self.attn_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=self._paged_kv_indices_buf[:n],
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=causal,
            q_data_type=dtype,
            seq_lens=kv_lens,
            window_left=window_left,
        )
        self._planner_args = self._direct_planner_args(causal)
        self._planner_key = None if self._planner_args is None else key

    def _direct_planner_args(self, causal: bool) -> list | None:
        """FlashInfer's planner arguments for this wrapper, fixed but for the
        host indptrs and lengths, when its plan resolved to the one path
        ``_plan_direct`` reproduces (``_prefill_plan_is_direct``); else None.
        Restating them is what vLLM and SGLang do too, on a pinned FlashInfer
        (see ``DIRECT_PLAN_FLASHINFER_VERSIONS``)."""
        w = self.attn_wrapper
        if not _prefill_plan_is_direct(w):
            return None
        args = [
            w._float_workspace_buffer, w._int_workspace_buffer, w._pin_memory_int_workspace_buffer,
            None, None, None,
            w._max_total_num_rows, w._fixed_batch_size,
            self.num_qo_heads, self.num_kv_heads, self.page_size, True,
            self.head_dim, self.head_dim,
            causal,
            -1,  # window_left
        ]
        if w._backend == "fa2":
            args += [
                -1,  # fixed_split_size
                False,  # disable_split_kv (FlashInfer sets it for an NVFP4 KV only)
                0,  # num_colocated_ctas
                0,  # uniform_q_len
            ]
        return args

    def _plan_direct(self, qo_indptr, kv_indptr, indices, last_page_len, kv_lens) -> None:
        """What FlashInfer's ``plan`` does per step on the path
        ``_direct_planner_args`` checked: its checks, the four index copies
        into the captured buffers, the per-step lengths it keeps, then its
        planner. Not its ``_kv_lens_buffer`` copy, which fa2/fa3 do not read."""
        w = self.attn_wrapper
        n = indices.shape[0]
        qo = qo_indptr.numpy()
        total = int(qo[-1])
        if n > w._paged_kv_indices_buf.shape[0]:
            raise ValueError("more page indices than the captured buffer holds")
        if total > w._max_total_num_rows:
            raise ValueError(
                f"{total} query rows in a wrapper captured for {w._max_total_num_rows}"
            )
        w._qo_indptr_buf.copy_(qo_indptr, non_blocking=True)
        w._paged_kv_indptr_buf.copy_(kv_indptr, non_blocking=True)
        w._paged_kv_last_page_len_buf.copy_(last_page_len, non_blocking=True)
        w._paged_kv_indices_buf[:n].copy_(indices, non_blocking=True)
        w._qo_indptr_last = total
        w._max_q_len = int((qo[1:] - qo[:-1]).max())
        w._max_kv_len = int(kv_lens.numpy().max())
        args = self._planner_args
        args[3] = qo_indptr
        args[4] = kv_indptr
        args[5] = kv_lens
        w._plan_info = w._cached_module.plan(*args)

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
        # set by the first plan when ``_plan_direct`` may take the later ones,
        # with where in them the per-step indptr and lengths go
        self._planner_args: list | None = None
        self._planner_slots: tuple[int, int | None] = (4, 5)

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

        if (
            self._planner_args is not None and dtype is self.dtype and kv_lens is not None
            and window_left == -1
            # the planner reads host int32 by raw pointer; plan() converts anything else
            and kv_lens.dtype == torch.int32 and kv_lens.device.type == "cpu"
        ):
            self._plan_direct(paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, kv_lens)
            return

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
        if self.accelerator_graph and kv_lens is not None:
            self._planner_args = self._direct_planner_args(n_req)

    def _direct_planner_args(self, n_req: int) -> list | None:
        """FlashInfer's planner arguments for this wrapper, fixed but for the
        indptr (and, for tensor-core decode, the lengths), when its plan
        resolved to a path ``_plan_direct`` reproduces
        (``_decode_direct_variant``); else None. Restating them is what vLLM
        and SGLang do too, on a pinned FlashInfer (see
        ``DIRECT_PLAN_FLASHINFER_VERSIONS``). As FlashInfer's own
        ``fast_decode_plan``, the direct path sets no per-step state but the
        plan."""
        w = self.attn_wrapper
        variant = _decode_direct_variant(w)
        if variant is None:
            return None
        if variant == _CUDA_CORE_DECODE:
            self._planner_slots = (3, None)
            return [
                w._float_workspace_buffer, w._int_workspace_buffer, w._pin_memory_int_workspace_buffer,
                None,  # indptr
                n_req, self.num_qo_heads, self.num_kv_heads, self.page_size, True,
                -1,  # window_left
                0.0,  # logits_soft_cap
                self.head_dim, self.head_dim,
                torch.empty(0, dtype=w._cached_q_data_type),
                torch.empty(0, dtype=w._cached_kv_data_type),
            ]
        from flashinfer.utils import _get_range_buf

        self._planner_slots = (4, 5)
        return [
            w._float_workspace_buffer, w._int_workspace_buffer, w._pin_memory_int_workspace_buffer,
            _get_range_buf(n_req + 1, "cpu"), None, None,
            n_req, n_req, self.num_qo_heads, self.num_kv_heads, self.page_size, True,
            self.head_dim, self.head_dim,
            False,  # causal
            -1,  # window_left
            -1,  # fixed_split_size
            False,  # disable_split_kv
            0,  # num_colocated_ctas
            0,  # uniform_q_len
        ]

    def _plan_direct(self, indptr, indices, last_page_len, kv_lens) -> None:
        """What FlashInfer's ``plan`` does per step on the path
        ``_direct_planner_args`` checked: the three index copies into the
        captured buffers, then its planner."""
        w = self.attn_wrapper
        n = indices.shape[0]
        if n > w._paged_kv_indices_buf.shape[0]:
            raise ValueError("more page indices than the captured buffer holds")
        w._paged_kv_indptr_buf.copy_(indptr, non_blocking=True)
        w._paged_kv_last_page_len_buf.copy_(last_page_len, non_blocking=True)
        dst = w._paged_kv_indices_buf[:n]
        if indices.data_ptr() != dst.data_ptr():
            dst.copy_(indices, non_blocking=indices.device == dst.device)
        self.plan_schedule(indptr, kv_lens)

    def plan_schedule(self, indptr: torch.Tensor, kv_lens: torch.Tensor) -> None:
        """FlashInfer's planner alone, from the host indptr and lengths, for a
        captured wrapper whose paged arrays the graph writes itself. Its other
        arguments are the ones the first full ``plan`` settled."""
        args = self._planner_args
        if args is None:
            raise RuntimeError(
                "this decode wrapper's plan resolved to a path the direct planner "
                "does not reproduce, so it can't plan the schedule alone"
            )
        at_indptr, at_lens = self._planner_slots
        args[at_indptr] = indptr
        if at_lens is not None:
            args[at_lens] = kv_lens
        w = self.attn_wrapper
        w._plan_info = w._cached_module.plan(*args)

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

    The packed prefill planner over every row costs per row about twice what
    the decode planner does, and its kernel runs the one-token rows as
    prefill queries. Here the decode wrapper takes every row of the bucket
    (a one-token row attends its own pages; any other row reads one token of
    the sink page, and its output is dropped), and the prefill wrapper takes
    at most ``max_prefill_rows`` prompt rows, their tokens moved to the front
    by a planned permutation. Both are captured at static shapes: the
    bucket's rows and tokens, and ``max_prefill_rows`` sequences.

    Every host array is staged in page-locked buffers this wrapper owns,
    sized for the bucket: a mixed step plans inline on the gpu thread, behind
    the step in flight.
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
        from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens

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
        self._decode = FlashInferDecodeWrapper(
            workspace_buffer=decode_workspace, batch_size=batch_size,
            max_num_pages=max_num_pages, accelerator_graph=True,
            use_tensor_cores=use_tensor_cores, **kv,
        )
        # one row past the prompt rows takes the step's other tokens (see `plan`)
        self._prefill = FlashInferPrefillWrapper(
            workspace_buffer=prefill_workspace, batch_size=max_prefill_rows + 1,
            max_total_tokens=max_total_tokens, max_num_pages=max_num_pages,
            accelerator_graph=True, **kv,
        )
        # the token permutation and the decode rows' gather/scatter, one
        # device block filled by one copy: [perm | dec_src | dec_dst]
        tokens, rows = max_total_tokens, batch_size
        self._idx64 = torch.zeros(tokens + 2 * rows, dtype=torch.int64, device=device)
        self._perm = self._idx64[:tokens]
        self._dec_src = self._idx64[tokens:tokens + rows]
        self._dec_dst = self._idx64[tokens + rows:]
        pin = device.type == "cuda"
        self._host64 = torch.empty(tokens + 2 * rows, dtype=torch.int64, pin_memory=pin)
        # the whole step's qo_indptr, which `select_last_hidden` reads
        self._qo_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        # an index row holds the step's pages plus one sink page per padding
        # row; the prefill one, its prompt pages plus the sink pages of its
        # empty rows and tail row
        pages = max_num_pages + batch_size
        tail_pages = -(-max(tokens, 1) // page_size)
        sizes = dict(
            qo=rows + 1, d_indptr=rows + 1, d_indices=pages, d_last=rows, d_lens=rows,
            p_qo=max_prefill_rows + 2, p_kvp=max_prefill_rows + 2,
            p_last=max_prefill_rows + 1, p_lens=max_prefill_rows + 1,
            p_idx=pages + max_prefill_rows + tail_pages,
        )
        self._host = {
            k: torch.empty(n, dtype=torch.int32, pin_memory=pin) for k, n in sizes.items()
        }
        self._np = {k: t.numpy() for k, t in self._host.items()}
        self._d_lens = self._host["d_lens"].as_subclass(HostLens)
        self._p_lens = self._host["p_lens"].as_subclass(HostLens)
        self._iota = np.arange(tokens, dtype=np.int64)
        self._sinks = np.full(max_prefill_rows + tail_pages, SINK_PAGE, dtype=np.int32)
        self._done = None
        self.dtype = torch.bfloat16

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
        self.dtype = dtype
        if self._done is None:
            self._done = torch.cuda.Event()
        else:
            # this wrapper's previous plan, a step or more back: long since done
            self._done.synchronize()

        page = self._page_size
        sink = self._sink_page
        qo = qo_indptr.numpy()
        kvp = paged_kv_indptr.numpy()
        idx = paged_kv_indices.numpy()
        last = paged_kv_last_page_len.numpy()
        rows = len(qo) - 1
        tokens = self._tokens
        if rows != self._bs or int(qo[-1]) > tokens:
            raise RuntimeError(
                f"FlashInferMixedWrapper: a step of {rows} rows and {int(qo[-1])} tokens "
                f"in a bucket captured for {self._bs} and {tokens}"
            )
        npages = np.diff(kvp)
        lens = (
            kv_lens.numpy() if kv_lens is not None
            # FlashInfer's `get_seq_lens`
            else np.maximum(npages - 1, 0) * page + last
        )
        spans = np.diff(qo)
        other = np.flatnonzero(spans != 1)  # prompt rows and padding rows
        prompt = other[spans[other] > 1].tolist()
        if len(prompt) > self._max_prefill_rows:
            raise RuntimeError(
                f"FlashInferMixedWrapper: {len(prompt)} prompt rows in a step "
                f"captured for {self._max_prefill_rows}; the scheduler caps them"
            )

        # decode: every row of the bucket; a row that is not one token reads
        # one token of the sink page, and its output lands on a scratch row.
        # The KV plan has put padding rows there already, so only the prompt
        # rows (and any other row not yet on the sink) move.
        h = self._np
        d_last, d_lens = h["d_last"], h["d_lens"]
        d_last[:] = last
        d_lens[:] = lens
        move = other
        if len(other) and len(idx):
            on_sink = (
                (npages[other] == 1) & (last[other] == 1) & (lens[other] == 1)
                & (idx[np.minimum(kvp[other], len(idx) - 1)] == sink)
            )
            move = other[~on_sink]
        if len(move) == 0:
            h["d_indptr"][:] = kvp
            n_dec = len(idx)
            h["d_indices"][:n_dec] = idx
        else:
            cnt = npages.copy()
            cnt[move] = 1
            h["d_indptr"][0] = 0
            np.cumsum(cnt, out=h["d_indptr"][1:])
            n_dec = int(h["d_indptr"][-1])
            pieces, prev = [], 0
            for r in move.tolist():
                pieces.append(idx[prev:kvp[r]])
                pieces.append(self._sinks[:1])
                prev = kvp[r + 1]
            pieces.append(idx[prev:])
            np.concatenate(pieces, out=h["d_indices"][:n_dec])
            d_last[move] = 1
            d_lens[move] = 1

        # prefill: the prompt rows, then empty rows on the sink page up to
        # `max_prefill_rows`, then a sink-page tail row over the step's other
        # tokens, so qo_indptr ends at the token count FlashInfer checks
        p_qo, p_kvp, p_last, p_lens = [0], [0], [], []
        pieces, front = [], []
        for r in prompt:
            a, b, s, e = int(qo[r]), int(qo[r + 1]), int(kvp[r]), int(kvp[r + 1])
            p_qo.append(p_qo[-1] + b - a)
            p_kvp.append(p_kvp[-1] + e - s)
            p_last.append(int(last[r]))
            p_lens.append(int(lens[r]))
            pieces.append(idx[s:e])
            front.append((a, b))
        pad = self._max_prefill_rows - len(prompt)
        tail_kv = max(tokens - p_qo[-1], 1)
        tail_pages = -(-tail_kv // page)
        for _ in range(pad):
            p_qo.append(p_qo[-1])
            p_kvp.append(p_kvp[-1] + 1)
            p_last.append(1)
            p_lens.append(1)
        p_qo.append(tokens)
        p_kvp.append(p_kvp[-1] + tail_pages)
        p_last.append(tail_kv - (tail_pages - 1) * page)
        p_lens.append(tail_kv)
        h["p_qo"][:] = p_qo
        h["p_kvp"][:] = p_kvp
        h["p_last"][:] = p_last
        h["p_lens"][:] = p_lens
        n_pre = p_kvp[-1]
        pieces.append(self._sinks[:pad + tail_pages])
        np.concatenate(pieces, out=h["p_idx"][:n_pre])

        # the prompt tokens first, then every other position in order
        host64 = self._host64.numpy()
        perm = host64[:tokens]
        if front:
            iota = self._iota
            rest, prev = [], 0
            for a, b in front:
                rest.append(iota[prev:a])
                prev = b
            rest.append(iota[prev:tokens])
            np.concatenate([iota[a:b] for a, b in front] + rest, out=perm)
        else:
            perm[:] = self._iota
        starts = qo[:-1]
        np.minimum(starts, max(tokens - 1, 0), out=host64[tokens:tokens + rows])
        dec_dst = host64[tokens + rows:]
        dec_dst[:] = starts
        dec_dst[other] = tokens
        h["qo"][:] = qo

        t = self._host
        try:
            self._idx64.copy_(self._host64, non_blocking=True)
            self._qo_indptr_buf.copy_(t["qo"], non_blocking=True)
            self._num_rows = rows
            self._decode.plan(
                paged_kv_indptr=t["d_indptr"],
                paged_kv_indices=t["d_indices"][:n_dec],
                paged_kv_last_page_len=t["d_last"],
                kv_lens=self._d_lens,
                dtype=dtype, window_left=window_left,
            )
            self._prefill._plan_staged(
                t["p_qo"], t["p_kvp"], t["p_idx"][:n_pre], t["p_last"], self._p_lens,
                causal, dtype, window_left,
            )
        finally:
            # also on a raise: copies already issued still read the pinned buffers
            self._done.record()

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer: torch.Tensor) -> torch.Tensor:
        n = q.shape[0]
        front = q.index_select(0, self._perm[:n])
        rows = q.index_select(0, self._dec_src[: self._num_rows])
        out_front = self._prefill.run(front, kv_cache_layer)
        out_rows = self._decode.run(rows, kv_cache_layer)
        out = out_front.new_empty((n + 1, *out_front.shape[1:]))
        out.index_copy_(0, self._perm[:n], out_front[:n])
        out.index_copy_(0, self._dec_dst[: self._num_rows], out_rows.to(out_front.dtype))
        return out[:n]
