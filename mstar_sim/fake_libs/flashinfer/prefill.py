"""Stand-ins for FlashInfer's batch prefill wrappers (paged and ragged KV)."""

from __future__ import annotations

import torch

from mstar_sim.device import gpu_error

from ._paged import (
    PagedAttention,
    PrefillPlanner,
    _copy,
    _not_supported,
    attention_kernel,
    canonical_dtype,
    check_layout,
    element_size,
    table_features,
    visible_keys,
)
from .utils import determine_attention_backend


def _check_plan_extras(custom_mask, packed_custom_mask, pos_encoding_mode, **extras) -> None:
    if custom_mask is not None or packed_custom_mask is not None:
        _not_supported("prefill custom_mask")
    if pos_encoding_mode != "NONE":
        _not_supported(f"prefill pos_encoding_mode={pos_encoding_mode!r}")
    for name, value in extras.items():
        if value is not None:
            _not_supported(f"prefill plan({name}=...)")


def get_batch_prefill_module(backend: str, *args) -> PrefillPlanner:
    """FlashInfer's batch prefill module for ``backend`` (its dtypes and head dims, ``args``,
    don't change the planner): its ``plan`` and ``workspace_size``."""
    return PrefillPlanner(backend)


def _resolve_backend(backend: str, device, use_fp16_qk_reduction: bool, q_data_type, kv_data_type, head_dim_qk: int,
                     head_dim_vo: int | None) -> str:
    """What a wrapper's plan runs on: FlashInfer's choice, for ``"auto"``."""
    if backend != "auto":
        return backend
    return determine_attention_backend(
        device, 0, use_fp16_qk_reduction, False, canonical_dtype(q_data_type),
        canonical_dtype(kv_data_type or q_data_type), head_dim_qk=head_dim_qk, head_dim_vo=head_dim_vo or head_dim_qk)


class BatchPrefillWithPagedKVCacheWrapper(PagedAttention):
    """Batch prefill/append attention over a paged KV cache."""

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: torch.Tensor | None = None,
        paged_kv_indptr_buf: torch.Tensor | None = None,
        paged_kv_indices_buf: torch.Tensor | None = None,
        paged_kv_last_page_len_buf: torch.Tensor | None = None,
        custom_mask_buf: torch.Tensor | None = None,
        mask_indptr_buf: torch.Tensor | None = None,
        backend: str = "auto",
        jit_args=None,
        jit_kwargs=None,
    ) -> None:
        if jit_args is not None:
            _not_supported("BatchPrefillWithPagedKVCacheWrapper(jit_args=...)")
        if use_cuda_graph and not torch.is_tensor(qo_indptr_buf):
            raise gpu_error(ValueError("qo_indptr_buf should be a torch.Tensor in cuda graph mode"))
        super().__init__(
            float_workspace_buffer, kv_layout, use_cuda_graph, qo_indptr_buf,
            paged_kv_indptr_buf, paged_kv_indices_buf, paged_kv_last_page_len_buf, backend,
        )

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        head_dim_vo: int | None = None,
        custom_mask: torch.Tensor | None = None,
        packed_custom_mask: torch.Tensor | None = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: float | None = None,
        window_left: int = -1,
        logits_soft_cap: float | None = None,
        rope_scale: float | None = None,
        rope_theta: float | None = None,
        q_data_type="float16",
        kv_data_type=None,
        o_data_type=None,
        non_blocking: bool = True,
        prefix_len_ptr: torch.Tensor | None = None,
        token_pos_in_items_ptr: torch.Tensor | None = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        seq_lens_q: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
        max_token_per_sequence: int | None = None,
        max_sequence_kv: int | None = None,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
    ) -> None:
        # seq_lens, block_tables and the max_* hints feed other backends'
        # schedulers; the fa2/fa3 kernels read lengths off indptr/last_page_len
        _check_plan_extras(
            custom_mask, packed_custom_mask, pos_encoding_mode,
            prefix_len_ptr=prefix_len_ptr, token_pos_in_items_ptr=token_pos_in_items_ptr,
            max_item_len_ptr=max_item_len_ptr,
        )
        self._backend = _resolve_backend(self._backend, self.device, use_fp16_qk_reduction, q_data_type, kv_data_type,
                                         head_dim_qk, head_dim_vo)
        self._logits_soft_cap = logits_soft_cap or 0.0
        self._set_plan(
            qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo, page_size=page_size, causal=causal, window_left=window_left,
            q_data_type=q_data_type, kv_data_type=kv_data_type, o_data_type=o_data_type,
            fixed_split_size=fixed_split_size, disable_split_kv=disable_split_kv,
        )


class BatchPrefillWithRaggedKVCacheWrapper:
    """Batch prefill attention over packed (ragged) q/k/v: segment ``i`` of the
    queries attends segment ``i`` of the keys."""

    _kernel = attention_kernel("flashinfer.batch_prefill_ragged")

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: torch.Tensor | None = None,
        kv_indptr_buf: torch.Tensor | None = None,
        custom_mask_buf: torch.Tensor | None = None,
        mask_indptr_buf: torch.Tensor | None = None,
        backend: str = "auto",
        jit_args=None,
        jit_kwargs=None,
    ) -> None:
        if jit_args is not None:
            _not_supported("BatchPrefillWithRaggedKVCacheWrapper(jit_args=...)")
        self._kv_layout = check_layout(kv_layout)
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._requested_backend = backend  # auto chooses again at each plan, as FlashInfer's ragged wrapper does
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            if not torch.is_tensor(qo_indptr_buf) or not torch.is_tensor(kv_indptr_buf):
                raise gpu_error(ValueError(
                    "qo_indptr_buf and kv_indptr_buf should be torch.Tensors in cuda graph mode"
                ))
            self._fixed_batch_size = len(qo_indptr_buf) - 1
            if len(kv_indptr_buf) != self._fixed_batch_size + 1:
                raise gpu_error(ValueError("The length of kv_indptr_buf should be batch_size + 1."))
        else:
            self._fixed_batch_size = 0
        self._qo_indptr_buf = qo_indptr_buf
        self._kv_indptr_buf = kv_indptr_buf
        self._max_total_num_rows: int | None = None
        self._planned = False
        self._cost = (0.0, 0.0)
        self._cached_module = None

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int | None = None,
        custom_mask: torch.Tensor | None = None,
        packed_custom_mask: torch.Tensor | None = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: float | None = None,
        sm_scale: float | None = None,
        rope_scale: float | None = None,
        rope_theta: float | None = None,
        q_data_type="float16",
        kv_data_type=None,
        o_data_type=None,
        non_blocking: bool = True,
        prefix_len_ptr: torch.Tensor | None = None,
        token_pos_in_items_ptr: torch.Tensor | None = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: torch.Tensor | None = None,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
        **kwargs,
    ) -> None:
        _check_plan_extras(
            custom_mask, packed_custom_mask, pos_encoding_mode,
            prefix_len_ptr=prefix_len_ptr, token_pos_in_items_ptr=token_pos_in_items_ptr,
            max_item_len_ptr=max_item_len_ptr,
        )
        if num_qo_heads % num_kv_heads:
            raise gpu_error(ValueError(
                f"num_qo_heads {num_qo_heads} must be a multiple of num_kv_heads {num_kv_heads}"
            ))
        qo_host = [int(x) for x in qo_indptr.tolist()]
        kv_host = [int(x) for x in kv_indptr.tolist()]
        if len(qo_host) != len(kv_host):
            raise gpu_error(ValueError("qo_indptr and kv_indptr must have the same length"))
        batch = len(qo_host) - 1
        total_rows = qo_host[-1]
        if self._use_cuda_graph:
            if self._max_total_num_rows is None:
                self._max_total_num_rows = total_rows
            elif total_rows > self._max_total_num_rows:
                raise gpu_error(ValueError(
                    f"The total number of rows in qo_indptr {total_rows} in cuda graph mode cannot "
                    f"exceed the number of rows set during initialization {self._max_total_num_rows}."
                ))
            if batch != self._fixed_batch_size:
                raise gpu_error(ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size "
                    f"{batch}  mismatches the batch size set during initialization "
                    f"{self._fixed_batch_size}."
                ))
            self._qo_indptr_buf.copy_(qo_indptr)
            self._kv_indptr_buf.copy_(kv_indptr)
        else:
            self._qo_indptr_buf = _copy(qo_indptr, self.device)
            self._kv_indptr_buf = _copy(kv_indptr, self.device)

        self._backend = _resolve_backend(self._requested_backend, self.device, use_fp16_qk_reduction, q_data_type,
                                         kv_data_type, head_dim_qk, head_dim_vo)
        self._cached_module = PrefillPlanner(self._backend)
        fa2_args = [fixed_split_size or -1, disable_split_kv, 0, 0] if self._backend == "fa2" else []
        lens = torch.tensor([kv_host[r + 1] - kv_host[r] for r in range(batch)], dtype=torch.int32)
        self._cached_module.plan(  # FlashInfer plans a ragged batch as pages of one token
            self._float_workspace_buffer, None, None, qo_indptr, kv_indptr, lens,
            self._max_total_num_rows or total_rows, batch, num_qo_heads, num_kv_heads, 1,
            self._use_cuda_graph, head_dim_qk, head_dim_vo or head_dim_qk, causal, window_left, *fa2_args)
        self._batch_size = batch
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim_qk = head_dim_qk
        self._head_dim_vo = head_dim_vo or head_dim_qk
        self._causal = bool(causal)
        self._window_left = int(window_left)
        self._cached_q_data_type = canonical_dtype(q_data_type)
        self._cached_kv_data_type = canonical_dtype(kv_data_type or q_data_type)
        self._cached_o_data_type = canonical_dtype(o_data_type or q_data_type)
        self._planned = True

        rows = [
            (n, m, visible_keys(n, m, self._causal, self._window_left))
            for n, m in ((qo_host[r + 1] - qo_host[r], kv_host[r + 1] - kv_host[r]) for r in range(batch))
            if n > 0
        ]
        pairs = sum(seen for _, _, seen in rows)
        kv_tokens = kv_host[-1] - kv_host[0]
        flops = 2.0 * pairs * num_qo_heads * (head_dim_qk + self._head_dim_vo)
        nbytes = (
            total_rows * num_qo_heads * (
                head_dim_qk * element_size(self._cached_q_data_type)
                + self._head_dim_vo * element_size(self._cached_o_data_type)
            )
            + kv_tokens * num_kv_heads * (head_dim_qk + self._head_dim_vo)
            * element_size(self._cached_kv_data_type)
        )
        self._cost = (flops, float(nbytes), table_features(
            rows, num_qo_heads, num_kv_heads, head_dim_qk, self._head_dim_vo,
            self._cached_kv_data_type))

    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, q_scale=None,
        k_scale=None, v_scale=None, o_scale=None, out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None, return_lse: bool = False, enable_pdl=None,
        kv_cache_sf=None, lse_base: str = "log2", **kwargs,
    ):
        if not self._planned:
            raise gpu_error(RuntimeError("plan() must be called before run()"))
        if args:
            _not_supported("custom-kernel run(*args)")
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
