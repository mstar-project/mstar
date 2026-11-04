"""Stand-in for FlashInfer's MLA paged attention wrapper (absorbed latents)."""

from __future__ import annotations

import torch

from mstar_sim.device import gpu_error

from ._paged import _copy, _not_supported, attention_kernel, canonical_dtype, element_size, visible_keys


class BatchMLAPagedAttentionWrapper:
    """Attention of ``q_nope``/``q_pe`` over paged ``ckv``/``kpe`` latents:
    logits ``(q_nope . ckv + q_pe . kpe) * sm_scale``, values ``ckv``. Each
    row's length is its ``kv_len_arr`` entry."""

    _kernel = attention_kernel("flashinfer.batch_mla_paged_attention")

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
        qo_indptr: torch.Tensor | None = None,
        kv_indptr: torch.Tensor | None = None,
        kv_indices: torch.Tensor | None = None,
        kv_len_arr: torch.Tensor | None = None,
        backend: str = "auto",
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._backend = backend
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            for name, buf in (("qo_indptr", qo_indptr), ("kv_indptr", kv_indptr),
                              ("kv_indices", kv_indices), ("kv_len_arr", kv_len_arr)):
                if not torch.is_tensor(buf):
                    raise gpu_error(ValueError(f"{name} should be a torch.Tensor in cuda graph mode"))
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr
        self._planned = False
        self._cost = (0.0, 0.0)

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
    ) -> None:
        qo_host = [int(x) for x in qo_indptr.tolist()]
        lens_host = [int(x) for x in kv_len_arr.tolist()]
        batch = len(qo_host) - 1
        if self._use_cuda_graph:
            if batch + 1 != len(self._qo_indptr_buf):
                raise gpu_error(ValueError(
                    f"The batch size should be fixed in cudagraph mode: {batch} rows planned, "
                    f"{len(self._qo_indptr_buf) - 1} captured"
                ))
            if len(kv_indices) > len(self._kv_indices_buf):
                raise gpu_error(ValueError("The length of kv_indices exceeds the allocated buffer size."))
            self._qo_indptr_buf.copy_(qo_indptr)
            self._kv_indptr_buf.copy_(kv_indptr)
            self._kv_indices_buf[: len(kv_indices)].copy_(kv_indices)
            self._kv_len_arr_buf[: len(kv_len_arr)].copy_(kv_len_arr)
        else:
            self._qo_indptr_buf = _copy(qo_indptr, self.device)
            self._kv_indptr_buf = _copy(kv_indptr, self.device)
            self._kv_indices_buf = _copy(kv_indices, self.device)
            self._kv_len_arr_buf = _copy(kv_len_arr, self.device)
        self._batch_size = batch
        self._num_heads = num_heads
        self._head_dim_ckv = head_dim_ckv
        self._head_dim_kpe = head_dim_kpe
        self._page_size = page_size
        self._causal = bool(causal)
        self._cached_q_data_type = canonical_dtype(q_data_type)
        self._cached_kv_data_type = canonical_dtype(kv_data_type)
        self._planned = True

        pairs = sum(
            visible_keys(qo_host[r + 1] - qo_host[r], lens_host[r], self._causal, -1)
            for r in range(batch)
        )
        width = head_dim_ckv + head_dim_kpe
        q_el = element_size(self._cached_q_data_type)
        kv_el = element_size(self._cached_kv_data_type)
        rows = qo_host[-1]
        kv_tokens = sum(lens_host[r] for r in range(batch) if qo_host[r + 1] > qo_host[r])
        flops = 2.0 * pairs * num_heads * (width + head_dim_ckv)
        nbytes = rows * num_heads * (width + head_dim_ckv) * q_el + kv_tokens * width * kv_el
        self._cost = (flops, float(nbytes))

    def run(
        self, q_nope: torch.Tensor, q_pe: torch.Tensor, ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor, out: torch.Tensor | None = None, lse: torch.Tensor | None = None,
        return_lse: bool = False, profiler_buffer=None, kv_len: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None, return_lse_base_on_e: bool = False,
        o_scale: float | None = None, *, ckv_scale=None, ckv_scale_arr=None, kpe_scale=None,
    ):
        if not self._planned:
            raise gpu_error(RuntimeError("plan() must be called before run()"))
        if kv_len is not None or page_table is not None:
            _not_supported("mla run(kv_len=, page_table=) (the trtllm-gen path)")
        if ckv_scale is not None or ckv_scale_arr is not None or kpe_scale is not None:
            _not_supported("mla run with fp8 latent scales")
        if out is None:
            out = torch.empty(q_nope.shape, dtype=self._cached_q_data_type, device=q_nope.device)
        if return_lse and lse is None:
            lse = torch.empty(q_nope.shape[:2], dtype=torch.float32, device=q_nope.device)
        out, lse = self._kernel(out, lse)
        return (out, lse) if return_lse else out
