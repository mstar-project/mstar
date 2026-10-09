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


class _RaggedPrefillBase:
    """FlashInfer's ragged prefill with no KV cache: queries and keys packed by
    segment into ``[total_tokens, H, D]`` tensors, laid out by ``cu_seqlens``.

    Holds what the self- and cross-attention wrappers share: the graph-mode static
    buffers and their padding, the plan, and ``run``. A subclass's ``plan`` decides
    where the key layout comes from.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_num_segments: int | None = None,
        max_total_tokens: int | None = None,
        max_total_kv_tokens: int | None = None,
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
        self.max_total_kv_tokens = (
            max_total_kv_tokens if max_total_kv_tokens is not None else max_total_tokens
        )
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
            self._prime()
        else:
            self._qo_indptr_buf = None
            self._kv_indptr_buf = None
            # eager callers pass exact-size q/k/v; no tail to read
            self._out_buf = None
            self.attn_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer, kv_layout, backend=backend
            )

    def _prime(self) -> None:
        raise NotImplementedError

    @property
    def num_segments(self) -> int:
        """Real (unpadded) segment count from the most recent ``plan``."""
        return self._num_segments

    def _max_layout_cu_seqlens(self, total: int) -> torch.Tensor:
        """``total`` tokens spread over all segments, remainder on the first."""
        n = self.max_num_segments
        lens = [total // n] * n
        lens[0] += total % n
        cu = [0]
        for seg_len in lens:
            cu.append(cu[-1] + seg_len)
        return torch.tensor(cu, dtype=torch.int32)

    def _prepare_cu_seqlens(
        self, cu_seqlens: torch.Tensor, max_tokens: int | None = None, side: str = "",
    ) -> tuple[torch.Tensor, int]:
        """The layout to hand FlashInfer, and its real token count."""
        n_seg = int(cu_seqlens.numel()) - 1
        if not self.use_cuda_graph:
            # the token count is only needed to zero a graph wrapper's tail, and
            # reading it off a device layout would sync
            return cu_seqlens.to(torch.int32), -1

        if n_seg > self.max_num_segments:
            raise ValueError(
                f"{type(self).__name__}: {n_seg} segments exceeds the "
                f"{self.max_num_segments} this graph-mode wrapper was built for"
            )
        host = cu_seqlens.to(device="cpu", dtype=torch.int32)
        total_tokens = int(host[-1])
        if total_tokens > max_tokens:
            raise ValueError(
                f"{type(self).__name__}: {total_tokens} {side}tokens exceeds the "
                f"{max_tokens} this graph-mode wrapper was built for"
            )
        # FlashInfer's plan copies this into the static device buffer with a
        # non-blocking H2D that can still be in flight when the next step plans,
        # so the source must be a fresh buffer per plan (never reused) and, to
        # stay async, pinned — the caching host allocator then holds it until the
        # copy retires. The graph step already declares a layout padded to the
        # captured segment count (padding rows attend nothing), so the fresh
        # pinned buffer the caller hands us is already the right size: use it.
        if n_seg == self.max_num_segments:
            return host, total_tokens
        # A shorter layout is staged into a fresh pinned buffer padded to size.
        cu = torch.empty(
            self.max_num_segments + 1, dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )
        cu[: n_seg + 1].copy_(host)
        # Repeating the final offset appends zero-length segments — pads the
        # segment count to the fixed size without adding tokens.
        cu[n_seg + 1:] = total_tokens
        return cu, total_tokens

    def _plan(self, cu_seqlens: torch.Tensor, kv_cu_seqlens: torch.Tensor, causal: bool) -> None:
        cu, total_tokens = self._prepare_cu_seqlens(cu_seqlens, self.max_total_tokens)
        if kv_cu_seqlens is cu_seqlens:
            kv_cu = cu
        else:
            kv_cu, _ = self._prepare_cu_seqlens(kv_cu_seqlens, self.max_total_kv_tokens, "key ")
        self._num_segments = int(cu_seqlens.numel()) - 1
        if self.use_cuda_graph:
            self._total_tokens = total_tokens
        self.attn_wrapper.plan(
            cu,
            kv_cu,
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
        """Run the planned attention.

        Args:
            q: [total_tokens, num_qo_heads, head_dim], packed by the query layout
            k, v: [total_kv_tokens, num_kv_heads, head_dim], packed by the key layout
                (the query layout itself, for self-attention)
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
                f"{type(self).__name__}: {n} rows exceeds the "
                f"{self.max_total_tokens} this graph-mode wrapper was built for"
            )
            out = self.attn_wrapper.run(qp, kp, vp, out=self._out_buf[:n])
        if self.padded_head_dim != self.head_dim:
            return out[..., : self.head_dim].contiguous()
        # `out` is the shared buffer the next call overwrites; the padded
        # branch above already returns a copy
        return out.clone() if self._out_buf is not None else out


class RaggedPrefillWrapper(_RaggedPrefillBase):
    """Varlen self-attention: each segment attends within itself.
    ``cu_seqlens`` is both ``qo_indptr`` and ``kv_indptr``."""

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_num_segments: int | None = None,
        max_total_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__(
            workspace_buffer, num_qo_heads, num_kv_heads, head_dim,
            max_num_segments=max_num_segments, max_total_tokens=max_total_tokens, **kwargs,
        )

    def _prime(self) -> None:
        self.plan(self._max_layout_cu_seqlens(self.max_total_tokens))

    @torch.compiler.disable
    def plan(self, cu_seqlens: torch.Tensor, causal: bool = False) -> None:
        """Plan one packed layout. ``cu_seqlens``: ``[num_segments + 1]``, [0] == 0.

        CPU tensor preferred; a GPU one costs a sync. Safe to call before every
        replay — values are copied through the static buffers, not rebound.
        """
        self._plan(cu_seqlens, cu_seqlens, causal)


class RaggedCrossPrefillWrapper(_RaggedPrefillBase):
    """Varlen cross-attention between two packed layouts: query segment ``i``
    attends key segment ``i`` (one pair per request). Never causal.

    The key side has its own token ceiling in graph mode, ``max_total_kv_tokens``
    (defaults to ``max_total_tokens``)."""

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_num_segments: int | None = None,
        max_total_tokens: int | None = None,
        max_total_kv_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__(
            workspace_buffer, num_qo_heads, num_kv_heads, head_dim,
            max_num_segments=max_num_segments, max_total_tokens=max_total_tokens,
            max_total_kv_tokens=max_total_kv_tokens, **kwargs,
        )

    def _prime(self) -> None:
        self.plan(
            self._max_layout_cu_seqlens(self.max_total_tokens),
            self._max_layout_cu_seqlens(self.max_total_kv_tokens),
        )

    @torch.compiler.disable
    def plan(self, q_cu_seqlens: torch.Tensor, kv_cu_seqlens: torch.Tensor) -> None:
        """Plan one query layout against one key layout, each ``[num_segments + 1]``
        with [0] == 0 and the same segment count. CPU tensors preferred."""
        n_q, n_kv = int(q_cu_seqlens.numel()) - 1, int(kv_cu_seqlens.numel()) - 1
        if n_q != n_kv:
            raise ValueError(
                f"RaggedCrossPrefillWrapper: {n_q} query segments but {n_kv} key "
                "segments; a cross-attention plan pairs them one to one"
            )
        self._plan(q_cu_seqlens, kv_cu_seqlens, causal=False)
