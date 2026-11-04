"""Stand-ins for FlashInfer's batch decode wrapper and trtllm-gen decode."""

from __future__ import annotations

import torch

from mstar_sim import clock
from mstar_sim.device import gpu_error

from ._paged import PagedAttention, _not_supported, check_layout, element_size, table_features


class BatchDecodeWithPagedKVCacheWrapper(PagedAttention):
    """Batch decode attention over a paged KV cache: ``q_len_per_req`` queries
    per row (1 by default), causal within a row's block when more."""

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: torch.Tensor | None = None,
        paged_kv_indices_buffer: torch.Tensor | None = None,
        paged_kv_last_page_len_buffer: torch.Tensor | None = None,
        backend: str = "auto",
        jit_args=None,
    ) -> None:
        if jit_args is not None:
            _not_supported("BatchDecodeWithPagedKVCacheWrapper(jit_args=...)")
        super().__init__(
            float_workspace_buffer, kv_layout, use_cuda_graph, None, paged_kv_indptr_buffer,
            paged_kv_indices_buffer, paged_kv_last_page_len_buffer, backend,
        )
        self._use_tensor_cores = use_tensor_cores

    @property
    def use_tensor_cores(self) -> bool:
        return self._use_tensor_cores

    def _cuda_core_decode(self) -> bool:
        return not self._use_tensor_cores and self._q_len_per_req == 1

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *deprecated_positional_args,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        window_right: int = 0,
        logits_soft_cap: float | None = None,
        q_data_type="float16",
        kv_data_type=None,
        o_data_type=None,
        data_type=None,
        sm_scale: float | None = None,
        rope_scale: float | None = None,
        rope_theta: float | None = None,
        non_blocking: bool = True,
        block_tables: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
        q_len_per_req: int = 1,
    ) -> None:
        if deprecated_positional_args:
            _not_supported("decode plan() positional arguments past page_size")
        if pos_encoding_mode != "NONE":
            _not_supported(f"decode pos_encoding_mode={pos_encoding_mode!r}")
        if window_right != 0:
            _not_supported("decode window_right")
        if q_len_per_req < 1:
            raise gpu_error(ValueError(f"q_len_per_req must be >= 1, got {q_len_per_req}"))
        if data_type is not None:
            q_data_type = kv_data_type = data_type
        if self._use_tensor_cores and self._backend == "auto":
            self._backend = "fa2"  # tensor-core decode plans with the fa2 prefill module
        self._logits_soft_cap = logits_soft_cap or 0.0
        self._set_plan(
            None, indptr, indices, last_page_len, num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads, head_dim_qk=head_dim, head_dim_vo=None,
            page_size=page_size, causal=q_len_per_req > 1, window_left=window_left,
            q_data_type=q_data_type, kv_data_type=kv_data_type, o_data_type=o_data_type,
            q_len_per_req=q_len_per_req,
        )


def _trtllm_decode_cost(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len,
                        *args, **kwargs):
    # each row's real KV length, read at each replay; max_seq_len only if the lengths carry no values
    batch, heads, dim = query.shape
    k = kv_cache[0] if isinstance(kv_cache, (tuple, list)) else kv_cache[:, 0]
    kv_heads = k.shape[1] if kwargs.get("kv_layout", "HND") == "HND" else k.shape[2]
    el, kv_dtype = element_size(query.dtype), k.dtype
    window = kwargs.get("window_left", args[2] if len(args) > 2 else -1)

    def terms() -> tuple:
        lens = _row_lengths(seq_lens, batch, max_seq_len)
        rows = [(1, n, min(n, window + 1) if window >= 0 else n) for n in lens]
        kv = sum(lens)
        features = table_features(rows, heads, kv_heads, dim, dim, kv_dtype)
        return 4.0 * kv * heads * dim, float(el * (2 * batch * heads * dim + 2 * kv * kv_heads * dim)), features

    return terms


def _row_lengths(seq_lens, batch: int, cap: int) -> list[int]:
    """The rows' lengths, read without a host sync (the kernel's own read)."""
    data = getattr(seq_lens, "_data", seq_lens)
    if not isinstance(data, torch.Tensor) or data.device.type == "meta":
        return [cap] * batch
    return [max(1, min(int(n), cap)) for n in data[:batch].tolist()]


@clock.kernel_fn("flashinfer.trtllm_batch_decode", cost=_trtllm_decode_cost)
def trtllm_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    window_left: int = -1,
    out: torch.Tensor | None = None,
    out_dtype=None,
    o_sf_scale=None,
    o_sf_vec_size=None,
    sinks=None,
    kv_layout: str = "HND",
    enable_pdl=None,
    backend: str = "auto",
    q_len_per_req: int | None = 1,
    o_scale: float | None = 1.0,
    mask=None,
    max_q_len=None,
    cum_seq_lens_q=None,
    skip_softmax_threshold_scale_factor=None,
    kv_cache_sf=None,
    uses_shared_paged_kv_idx: bool = True,
    lse: torch.Tensor | None = None,
    return_lse: bool = False,
    bmm1_scale_log2=None,
    multi_ctas_kv_counter_buffer=None,
    enable_block_sparse_attention: bool = False,
):
    """One query per row (``query [batch, heads, dim]``) over the pages
    ``block_tables[row, :ceil(seq_lens[row] / page_size)]``."""
    if (q_len_per_req or 1) != 1 or sinks is not None or mask is not None or kv_cache_sf is not None:
        _not_supported("trtllm_batch_decode_with_kv_cache beyond one bf16/fp16 query per row")
    if bmm1_scale_log2 is not None or torch.is_tensor(bmm1_scale) or torch.is_tensor(bmm2_scale):
        _not_supported("trtllm_batch_decode_with_kv_cache with tensor scales")
    if return_lse:
        _not_supported("trtllm_batch_decode_with_kv_cache(return_lse=True)")
    check_layout(kv_layout)
    return out if out is not None else torch.empty(
        query.shape, dtype=out_dtype or query.dtype, device=query.device)
