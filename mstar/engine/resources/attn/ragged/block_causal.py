"""Block-causal cacheless attention through FlashInfer's paged prefill.

Within each declared span, a token in block ``b`` (``block_size`` tokens a
block, counted from the span's start) attends every key of blocks ``0..b`` of
that span: causal between blocks, bidirectional within one. Whisper-style
encoders run this way so a streaming client sees the same features a
whole-clip forward would.

Prefixes of one span overlap, which a ragged ``kv_indptr`` (one contiguous key
range per query segment) cannot express. The paged kernel can: the packed
keys are viewed as pages of one token, and each query block is one paged
"request" whose page list is its prefix. No key is copied; the page lists are
host-side index arithmetic done at plan time.
"""

import torch

from mstar.engine.resources.attn.ragged.wrappers import padded_head_dim


def block_causal_layout(
    seg_lens: list[int], block_size: int,
) -> tuple[list[int], list[int], torch.Tensor]:
    """``(qo_indptr, kv_indptr, kv_indices)`` for spans packed end to end.

    One entry per query block. Block ``i`` of a span starting at ``off`` has
    queries ``[off + i*B, off + min((i+1)*B, L))`` and keys
    ``[off, off + min((i+1)*B, L))``.
    """
    qo, kv = [0], [0]
    starts: list[int] = []
    lens: list[int] = []
    off = 0
    for n in seg_lens:
        for s in range(0, n, block_size):
            e = min(s + block_size, n)
            qo.append(qo[-1] + e - s)
            kv.append(kv[-1] + e)
            starts.append(off)
            lens.append(e)
        off += n
    if not lens:
        return qo, kv, torch.zeros(0, dtype=torch.int32)
    lens_t = torch.tensor(lens, dtype=torch.int64)
    # each block's prefix as a run of consecutive token ids, all runs end to end
    run_start = torch.tensor(starts, dtype=torch.int64) - torch.tensor(kv[:-1], dtype=torch.int64)
    indices = torch.arange(kv[-1], dtype=torch.int64) + run_start.repeat_interleave(lens_t)
    return qo, kv, indices.to(torch.int32)


def max_blocks(max_segments: int, max_tokens: int, block_size: int) -> int:
    """Most query blocks a layout of at most this many spans and tokens has:
    every span may end on a partial block."""
    return max_tokens // block_size + max_segments


def max_prefix_tokens(max_tokens: int, block_size: int) -> int:
    """Most page indices such a layout needs: one span holding every token,
    which maximizes the sum of its blocks' prefixes."""
    n = -(-max_tokens // block_size)
    return block_size * n * (n + 1) // 2


class RaggedBlockCausalWrapper:
    """FlashInfer's paged prefill over a packed, cacheless ``[total, H, D]``
    layout, planned block-causally per segment; see the module docstring.

    In graph mode the block count is padded to the bucket's ceiling with
    zero-length blocks, and the index buffer is sized for the worst prefix
    sum the bucket's token count allows, so any layout inside the bucket
    replays through the same captured kernel.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        max_num_segments: int | None = None,
        max_total_tokens: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        sm_scale: float | None = None,
        q_data_type: torch.dtype = torch.bfloat16,
        backend: str = "auto",
    ):
        import flashinfer

        self.device = device
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.padded_head_dim = padded_head_dim(head_dim)
        self.block_size = block_size
        self.sm_scale = float(sm_scale) if sm_scale is not None else head_dim ** -0.5
        self.q_data_type = q_data_type
        self.use_cuda_graph = use_cuda_graph
        self.max_num_segments = max_num_segments
        self.max_total_tokens = max_total_tokens
        self._num_segments = 0
        self._total_tokens = 0

        if not use_cuda_graph:
            self._out_buf = None
            self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD", backend=backend,
            )
            return

        assert max_num_segments is not None, "max_num_segments required for CUDA graph mode"
        assert max_total_tokens is not None, "max_total_tokens required for CUDA graph mode"
        assert max_num_segments > 0, "max_num_segments must be positive"
        self.max_blocks = max_blocks(max_num_segments, max_total_tokens, block_size)
        self.max_indices = max_prefix_tokens(max_total_tokens, block_size)
        i32 = dict(dtype=torch.int32, device=device)
        self._qo_indptr_buf = torch.zeros(self.max_blocks + 1, **i32)
        self._kv_indptr_buf = torch.zeros(self.max_blocks + 1, **i32)
        self._kv_indices_buf = torch.zeros(self.max_indices, **i32)
        self._last_page_len_buf = torch.ones(self.max_blocks, **i32)
        self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD", use_cuda_graph=True,
            qo_indptr_buf=self._qo_indptr_buf,
            paged_kv_indptr_buf=self._kv_indptr_buf,
            paged_kv_indices_buf=self._kv_indices_buf,
            paged_kv_last_page_len_buf=self._last_page_len_buf,
            backend=backend,
        )
        # Own the output so a replay's unplanned tail is a known value; see
        # `_RaggedPrefillBase.__init__`.
        self._out_buf = torch.zeros(
            max_total_tokens, num_qo_heads, self.padded_head_dim,
            dtype=q_data_type, device=device,
        )
        # FlashInfer latches its row ceilings on the first plan; prime at the
        # bucket's worst case so a small first plan can't cap it.
        self.plan([max_total_tokens])

    @property
    def num_segments(self) -> int:
        """Real (unpadded) segment count from the most recent ``plan``."""
        return self._num_segments

    @staticmethod
    def _pinned(values) -> torch.Tensor:
        # fresh and pinned per plan: FlashInfer H2Ds these non-blocking, so a
        # reused source could be overwritten while its copy is in flight
        t = values if torch.is_tensor(values) else torch.tensor(values, dtype=torch.int32)
        return t.pin_memory() if torch.cuda.is_available() else t

    @torch.compiler.disable
    def plan(self, seg_lens: list[int]) -> None:
        """Plan one packed layout: ``seg_lens[i]`` tokens in span ``i``, end to end."""
        n_seg = len(seg_lens)
        total = sum(seg_lens)
        qo, kv, indices = block_causal_layout(seg_lens, self.block_size)
        n_blocks = len(qo) - 1
        if self.use_cuda_graph:
            if n_seg > self.max_num_segments:
                raise ValueError(
                    f"RaggedBlockCausalWrapper: {n_seg} segments exceeds the "
                    f"{self.max_num_segments} this graph-mode wrapper was built for"
                )
            if total > self.max_total_tokens:
                raise ValueError(
                    f"RaggedBlockCausalWrapper: {total} tokens exceeds the "
                    f"{self.max_total_tokens} this graph-mode wrapper was built for"
                )
            # zero-length blocks pad the count to what the graph was captured with
            pad = self.max_blocks - n_blocks
            qo = qo + [qo[-1]] * pad
            kv = kv + [kv[-1]] * pad
            n_blocks = self.max_blocks
        self._num_segments = n_seg
        self._total_tokens = total
        self.attn_wrapper.plan(
            self._pinned(qo),
            self._pinned(kv),
            self._pinned(indices),
            # one token a page: every page is full
            self._pinned(torch.ones(n_blocks, dtype=torch.int32)),
            self.num_qo_heads,
            self.num_kv_heads,
            self.padded_head_dim,
            1,
            causal=False,
            sm_scale=self.sm_scale,
            q_data_type=self.q_data_type,
        )
        if self._out_buf is not None:
            self._out_buf[total:].zero_()

    def _pad_head_dim(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-1] == self.padded_head_dim:
            return t.contiguous()
        return torch.nn.functional.pad(t, (0, self.padded_head_dim - t.shape[-1]))

    @torch.compiler.disable
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``[total, H, D]`` q/k/v packed by the planned spans -> ``[total, H, D]``.

        As with the ragged wrapper, rows past the planned total are not a
        valid result; the caller slices.
        """
        qp, kp, vp = (self._pad_head_dim(t.to(self.q_data_type)) for t in (q, k, v))
        # each token is a page of one: [num_pages, page_size=1, H, D]
        kv = (kp.unsqueeze(1), vp.unsqueeze(1))
        if self._out_buf is None:
            out = self.attn_wrapper.run(qp, kv)
        else:
            n = qp.shape[0]
            assert n <= self.max_total_tokens, (
                f"RaggedBlockCausalWrapper: {n} rows exceeds the "
                f"{self.max_total_tokens} this graph-mode wrapper was built for"
            )
            out = self.attn_wrapper.run(qp, kv, out=self._out_buf[:n])
        if self.padded_head_dim != self.head_dim:
            return out[..., : self.head_dim].contiguous()
        return out.clone() if self._out_buf is not None else out
