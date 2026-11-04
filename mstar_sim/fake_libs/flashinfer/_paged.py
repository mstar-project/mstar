"""What FlashInfer's attention wrappers share: a plan, and a run priced by it.

``plan()`` reads its host arrays once, checks them as FlashInfer does, copies
them where FlashInfer keeps them (the caller's static buffers under
``use_cuda_graph``, else device copies) and prices the attention the plan
describes from each request's query and key counts. ``run`` is one kernel
(``clock.kernel_fn``) charged the latest plan's price, so a graph replay
prices whatever the last plan was.

FlashInfer's conventions, which the pricing follows: a paged request's length
is ``(num_pages - 1) * page_size + last_page_len``, and a request with ``n``
queries over ``m`` keys aligns its queries to the end of its keys (causal
query ``i`` sees keys ``j <= i + m - n``; ``window_left = w >= 0`` adds
``j >= i + m - n - w``).
"""

from __future__ import annotations

import torch

from mstar_sim import clock


def canonical_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, dtype)


def check_layout(kv_layout: str) -> str:
    if kv_layout not in ("NHD", "HND"):
        raise KeyError(f"Invalid kv_layout {kv_layout}")
    return kv_layout


def element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def paged_kv_len(num_pages: int, last_page_len: int, page_size: int) -> int:
    return (num_pages - 1) * page_size + last_page_len if num_pages > 0 else 0


def visible_keys(n: int, m: int, causal: bool, window_left: int) -> int:
    """How many (query, key) pairs a request's mask lets through: what a
    tiled kernel computes, near enough, since it skips fully masked tiles."""
    if n <= 0 or m <= 0:
        return 0
    if causal:
        full = n * m - n * (n - 1) // 2 if m >= n else m * (m + 1) // 2
    else:
        full = n * m
    if window_left >= 0:
        full = min(full, n * (window_left + 1))
    return full


def _outputs(wrapper, out, lse):
    return out, lse


def attention_kernel(name: str):
    """A wrapper's run as kernel ``name`` (a method): priced by the wrapper's
    latest plan, read when charged, so a replay prices its own step's plan."""
    return clock.kernel_fn(name, cost=lambda wrapper, out, lse: lambda: wrapper._cost)(_outputs)


def _not_supported(what: str):
    raise NotImplementedError(f"mstar_sim: flashinfer.{what} has no stand-in yet")


def table_features(rows: list, heads: int, kv_heads: int, head_dim_qk: int, head_dim_vo: int,
                   kv_dtype) -> dict:
    """What :func:`mstar_sim.cost.kernel_seconds` needs to price attention from measured
    tables: ``rows`` [(queries, keys, visible pairs)] of the requests with queries."""
    return {"rows": rows, "heads": heads, "kv_heads": kv_heads,
            "head_dim": head_dim_qk if head_dim_qk == head_dim_vo else None, "kv_dtype": kv_dtype}


def _copy(src: torch.Tensor, device) -> torch.Tensor:
    """A device copy of a plan array (a snapshot, as FlashInfer's H2D is)."""
    return src.to(device=device, dtype=torch.int32, copy=True)


class PagedAttention:
    """The plan and run of a paged-KV wrapper. Subclasses parse their own
    ``plan`` arguments and call ``_set_plan``."""

    _kernel = attention_kernel("flashinfer.batch_paged_attention")

    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str, use_cuda_graph: bool,
        qo_indptr_buf, paged_kv_indptr_buf, paged_kv_indices_buf, paged_kv_last_page_len_buf,
        backend: str,
    ) -> None:
        self._kv_layout = check_layout(kv_layout)
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._backend = backend
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            for name, buf in (
                ("paged_kv_indptr_buf", paged_kv_indptr_buf),
                ("paged_kv_indices_buf", paged_kv_indices_buf),
                ("paged_kv_last_page_len_buf", paged_kv_last_page_len_buf),
            ):
                if not torch.is_tensor(buf):
                    raise ValueError(f"{name} should be a torch.Tensor in cuda graph mode")
            self._fixed_batch_size = len(paged_kv_last_page_len_buf)
            if len(paged_kv_indptr_buf) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The length of paged_kv_indptr_buf should be batch_size + 1."
                )
            if qo_indptr_buf is not None and len(qo_indptr_buf) != self._fixed_batch_size + 1:
                raise ValueError("The length of qo_indptr_buf should be batch_size + 1.")
        else:
            self._fixed_batch_size = 0
        self._qo_indptr_buf = qo_indptr_buf
        self._paged_kv_indptr_buf = paged_kv_indptr_buf
        self._paged_kv_indices_buf = paged_kv_indices_buf
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buf
        self._max_total_num_rows: int | None = None
        self._planned = False
        self._cost = (0.0, 0.0)

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    # ------------------------------------------------------------------ plan

    def _set_plan(
        self, qo_indptr, indptr, indices, last_page_len, *, num_qo_heads: int, num_kv_heads: int,
        head_dim_qk: int, head_dim_vo: int | None, page_size: int, causal: bool, window_left: int,
        q_data_type, kv_data_type, o_data_type, q_len_per_req: int = 1,
    ) -> None:
        """Store the plan. ``qo_indptr`` None means ``q_len_per_req`` queries a row (decode)."""
        if num_qo_heads % num_kv_heads:
            raise ValueError(
                f"num_qo_heads {num_qo_heads} must be a multiple of num_kv_heads {num_kv_heads}"
            )
        batch = len(last_page_len)
        if len(indptr) != batch + 1:
            raise ValueError(f"indptr has {len(indptr)} entries for a batch of {batch}")
        qo_host = (
            [i * q_len_per_req for i in range(batch + 1)] if qo_indptr is None
            else [int(x) for x in qo_indptr.tolist()]
        )
        if len(qo_host) != batch + 1:
            raise ValueError(f"qo_indptr has {len(qo_host)} entries for a batch of {batch}")
        kv_host = [int(x) for x in indptr.tolist()]
        last_host = [int(x) for x in last_page_len.tolist()]
        total_rows = qo_host[-1]

        if self._use_cuda_graph:
            if self._max_total_num_rows is None:
                self._max_total_num_rows = total_rows
            elif total_rows > self._max_total_num_rows:
                raise ValueError(
                    f"The total number of rows in qo_indptr {total_rows} in cuda graph mode cannot "
                    f"exceed the number of rows set during initialization {self._max_total_num_rows}."
                )
            if batch != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed during the lifecycle of the wrapper in "
                    f"cuda graph mode, the runtime batch size {batch} mismatches the batch size "
                    f"{self._fixed_batch_size}  set during initialization."
                )
            if len(indices) > len(self._paged_kv_indices_buf):
                raise ValueError("The length of paged_kv_indices exceeds the allocated buffer size.")
            if qo_indptr is not None:
                self._qo_indptr_buf.copy_(qo_indptr)
            self._paged_kv_indptr_buf.copy_(indptr)
            self._paged_kv_last_page_len_buf.copy_(last_page_len)
            self._paged_kv_indices_buf[: len(indices)].copy_(indices)
        else:
            if qo_indptr is not None:
                self._qo_indptr_buf = _copy(qo_indptr, self.device)
            self._paged_kv_indptr_buf = _copy(indptr, self.device)
            self._paged_kv_indices_buf = _copy(indices, self.device)
            self._paged_kv_last_page_len_buf = _copy(last_page_len, self.device)

        self._batch_size = batch
        self._q_len_per_req = q_len_per_req
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim_qk = head_dim_qk
        self._head_dim_vo = head_dim_vo or head_dim_qk
        self._page_size = page_size
        self._causal = bool(causal)
        self._window_left = int(window_left)
        self._cached_q_data_type = canonical_dtype(q_data_type)
        self._cached_kv_data_type = canonical_dtype(kv_data_type or q_data_type)
        self._cached_o_data_type = canonical_dtype(o_data_type or q_data_type)
        self._qo_indptr_last = total_rows
        self._planned = True

        pairs, pages, rows = 0, 0, []
        for r in range(batch):
            n = qo_host[r + 1] - qo_host[r]
            npages = kv_host[r + 1] - kv_host[r]
            m = paged_kv_len(npages, last_host[r], page_size)
            seen = visible_keys(n, m, self._causal, self._window_left)
            pairs += seen
            pages += npages if n > 0 else 0
            if n > 0:
                rows.append((n, m, seen))
        q_el = element_size(self._cached_q_data_type)
        kv_el = element_size(self._cached_kv_data_type)
        o_el = element_size(self._cached_o_data_type)
        flops = 2.0 * pairs * num_qo_heads * (head_dim_qk + self._head_dim_vo)
        nbytes = (
            total_rows * num_qo_heads * (head_dim_qk * q_el + self._head_dim_vo * o_el)
            + pages * page_size * num_kv_heads * (head_dim_qk + self._head_dim_vo) * kv_el
        )
        self._cost = (flops, float(nbytes), table_features(
            rows, num_qo_heads, num_kv_heads, head_dim_qk, self._head_dim_vo,
            self._cached_kv_data_type))

    # ------------------------------------------------------------------- run

    def run(
        self, q: torch.Tensor, paged_kv_cache, *args, q_scale=None, k_scale=None, v_scale=None,
        out: torch.Tensor | None = None, lse: torch.Tensor | None = None, return_lse: bool = False,
        enable_pdl=None, window_left: int | None = None, sinks=None, kv_cache_sf=None,
        lse_base: str = "log2", **kwargs,
    ):
        if not self._planned:
            raise RuntimeError("plan() must be called before run()")
        if args:
            _not_supported("custom-kernel run(*args)")
        if sinks is not None:
            _not_supported("attention sinks")
        if kv_cache_sf is not None:
            _not_supported("fp4 KV cache scale factors")
        if out is None:
            out = torch.empty(
                (q.shape[0], q.shape[1], self._head_dim_vo), dtype=self._cached_o_data_type,
                device=q.device,
            )
        if return_lse and lse is None:
            lse = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        out, lse = self._kernel(out, lse)
        return (out, lse) if return_lse else out
