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
        self._total_tokens = 0

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
            # Own the output: the kernel writes only the planned rows, and it
            # reads KV past cu_seqlens[-1] to the last segment's tile boundary,
            # masking additively. A NaN/Inf left in that tail by another
            # graph's freed pool block survives the mask and poisons the last
            # segment. ``plan`` keeps the window finite.
            self._out_buf = torch.zeros(
                max_total_tokens, num_qo_heads, self.padded_head_dim,
                dtype=q_data_type, device=device,
            )
            # FlashInfer latches max rows on the FIRST plan; prime at the
            # bucket ceiling so a small first plan can't cap it.
            self.plan(self._max_layout_cu_seqlens())
        else:
            self._qo_indptr_buf = None
            self._kv_indptr_buf = None
            # eager callers pass exact-size q/k/v; no tail to read
            self._out_buf = None
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
        self._num_segments = n_seg
        self._total_tokens = total_tokens
        # FlashInfer's plan copies this into the static device buffer with a
        # non-blocking H2D that can still be in flight when the next step plans,
        # so the source must be a fresh buffer per plan (never reused) and, to
        # stay async, pinned — the caching host allocator then holds it until the
        # copy retires. The graph step already declares a layout padded to the
        # captured segment count (padding rows attend nothing), so the fresh
        # pinned buffer the caller hands us is already the right size: use it.
        if n_seg == self.max_num_segments:
            return host
        # A shorter layout is staged into a fresh pinned buffer padded to size.
        cu = torch.empty(
            self.max_num_segments + 1, dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )
        cu[: n_seg + 1].copy_(host)
        # Repeating the final offset appends zero-length segments — pads the
        # segment count to the fixed size without adding tokens.
        cu[n_seg + 1:] = total_tokens
        return cu

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
        if self._out_buf is not None:
            # rows this layout leaves unwritten, zeroed outside the graph where
            # the real token count is known; see __init__
            self._out_buf[self._total_tokens:].zero_()

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

        Only rows before the planned ``cu_seqlens[-1]`` are computed; the rest
        read back zero. An oversized static buffer replays fine, but the caller
        must still slice — the padding rows are not a valid result.
        """
        qp, kp, vp = (self._pad_head_dim(t.to(self.q_data_type)) for t in (q, k, v))
        if self._out_buf is None:
            out = self.attn_wrapper.run(qp, kp, vp)
        else:
            n = qp.shape[0]
            assert n <= self.max_total_tokens, (
                f"RaggedPrefillWrapper: {n} rows exceeds the "
                f"{self.max_total_tokens} this graph-mode wrapper was built for"
            )
            out = self.attn_wrapper.run(qp, kp, vp, out=self._out_buf[:n])
        if self.padded_head_dim != self.head_dim:
            return out[..., : self.head_dim].contiguous()
        # `out` is the shared buffer the next call overwrites; the padded
        # branch above already returns a copy
        return out.clone() if self._out_buf is not None else out
