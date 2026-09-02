import torch

# Head dims the FlashInfer prefill kernels are instantiated for. An unsupported
# one fails to BUILD (SM90: static_assert in hopper/prefill_sm90.cuh).
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
    """Varlen self-attention over packed segments, with no KV cache; attention
    is within variable-length segments packed into one ``[total_tokens, H, D]``
    tensor. ``cu_seqlens`` is both ``qo_indptr`` and ``kv_indptr``.
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
        sm_scale: float | None = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_layout: str = "NHD",
        backend: str = "auto",
    ):
        self.device = device
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.padded_head_dim = padded_head_dim(head_dim)
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
            assert max_num_segments > 0, "max_num_segments must be positive"

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
                backend=backend,
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
                workspace_buffer, kv_layout, backend=backend
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
    def plan(self, cu_seqlens: torch.Tensor, causal: bool=False) -> None:
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
            causal=causal,
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
