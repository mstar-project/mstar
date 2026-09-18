"""FlashInfer's absorbed-MLA kernel behind the wrapper shape the attention
resource plans and runs.
"""

import functools
import logging

import torch

logger = logging.getLogger(__name__)


@functools.cache
def _mla_kernel_available_cached(ckv: int, kpe: int, sm_major: int) -> bool:
    if not (ckv == 512 and kpe == 64):
        return False
    if sm_major != 9:
        return False
    try:
        import flashinfer.mla  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def mla_kernel_available(ckv: int, kpe: int, device: torch.device) -> bool:
    """Whether ``flashinfer.mla.BatchMLAPagedAttentionWrapper`` can serve these latent dims
    on this device.
    """
    if device.type != "cuda":
        return False
    sm_major = torch.cuda.get_device_capability(device)[0]
    return _mla_kernel_available_cached(ckv, kpe, sm_major)


class FlashInferMLAWrapper:
    """One planned MLA kernel: eager (fresh plan buffers) or CUDA-graph
    (static ``qo_indptr``/``kv_indptr``/``kv_indices``/``kv_len_arr`` buffers
    the captured replay reads at fixed addresses)."""

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
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
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

        # Fence between consecutive plans on this wrapper. FlashInfer's plan()
        self._fence_ok = torch.cuda.is_available() and device.type == "cuda"
        self._plan_event: torch.cuda.Event | None = None

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
        """Plan the MLA kernel for one batch."""
        self.dtype = dtype
        if self._plan_event is not None and not self._plan_event.query():
            self._plan_event.synchronize()

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

        if self._fence_ok:
            if self._plan_event is None:
                self._plan_event = torch.cuda.Event()
            self._plan_event.record(torch.cuda.current_stream())

    @torch.compiler.disable
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Run the planned kernel."""
        return self.attn_wrapper.run(
            q_nope.to(self.dtype), q_pe.to(self.dtype),
            ckv_cache, kpe_cache, return_lse=False,
        )
