import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE

__all__ = ["LayerRingCache", "ring_scatter"]


@torch.library.custom_op("mstar::ring_scatter", mutates_args={"cache", "written"})
def ring_scatter(
    cache: torch.Tensor, written: torch.Tensor, dst: torch.Tensor,
    kv: torch.Tensor, mark: bool,
) -> None:
    """Write one frame of K/V into ``dst``'s token slots, optionally marking it.

    An op rather than plain in-place indexing for exactly the reason
    ``mstar::kv_scatter_nhd`` (``kv/cache.py``) is one: the forward reaches
    ``cache`` through an attribute chain, so dynamo lifts it as a graph
    attribute and AOTAutograd functionalizes the mutation into a copy of the
    WHOLE ring. At 720P that is 816 MiB per world copied 120 times per frame
    (24 layers x 5 passes) -- a throughput collapse, not an error, so nothing
    tells you.
    Declaring the mutation keeps the write in place and in-graph, with no break
    to recompile the layer body once per layer.

    ``index_fill_`` and not ``written[dst] = True``, which is what this was.
    The subscript form lowers to ``index_put_`` with the Python ``True``
    materialized as a CPU scalar tensor and copied to the device, and an H2D
    copy from pageable host memory invalidates CUDA graph capture.
    """
    cache.index_copy_(3, dst, kv)
    if mark:
        written.index_fill_(0, dst, True)


@ring_scatter.register_fake
def _ring_scatter_fake(
    cache: torch.Tensor, written: torch.Tensor, dst: torch.Tensor,
    kv: torch.Tensor, mark: bool,
) -> None:
    return None


class LayerRingCache:
    """One attention layer's ring: ``ring_frames`` frame slots of history plus
    one scratch frame at the tail, times ``num_worlds`` resident worlds.

    Storage is a single ``[2, 1, H_kv, num_worlds * capacity, D]`` tensor so
    that a commit is one ``index_copy_`` for K and V together and the read is
    one ``unbind(0)`` into two views -- no copy on the read path.

    ``capacity`` is ONE world's token slots; ``total_slots`` is the token-dim
    length of the buffer.
    """

    def __init__(
        self,
        num_worlds: int,
        n_kv_heads: int,
        ring_frames: int,
        ring_buckets: int,
        d_head: int,
        tokens_per_frame: int,
        pinned_dilation: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        if num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1; got {num_worlds}.")
        if pinned_dilation < 1:
            raise ValueError(f"pinned_dilation must be >= 1; got {pinned_dilation}.")
        if not 1 <= ring_buckets <= ring_frames:
            raise ValueError(
                f"ring_buckets ({ring_buckets}) must be in [1, ring_frames ({ring_frames})]; "
                "the addressable slots have to fit inside the allocated ring."
            )
        if tokens_per_frame % _DEFAULT_SPARSE_BLOCK_SIZE:
            raise ValueError(
                f"tokens_per_frame ({tokens_per_frame}) must be a multiple of the sparse "
                f"block size ({_DEFAULT_SPARSE_BLOCK_SIZE}); the BlockMask has no partial blocks."
            )

        self.num_worlds = num_worlds
        self.tokens_per_frame = tokens_per_frame
        self.ring_frames = ring_frames
        self.ring_buckets = ring_buckets
        self.pinned_dilation = pinned_dilation
        # ring_len is the reference's `L`: one world's history region, scratch
        # excluded.
        self.ring_len = ring_frames * tokens_per_frame
        self.capacity = self.ring_len + tokens_per_frame
        self.total_slots = num_worlds * self.capacity

        self.kv = torch.zeros(
            2, 1, n_kv_heads, self.total_slots, d_head, dtype=dtype, device=device
        )

        written = torch.zeros(self.total_slots, dtype=torch.bool, device=device)
        written.view(num_worlds, self.capacity)[:, self.ring_len :] = True
        self.written = written
        # Preallocated scratch for the per-call visibility mask. Allocating it
        # inside upsert would put a fresh buffer in the compiled region on every
        # one of the 120 upserts per frame.
        self._mask_written = torch.empty_like(written)
        self._world_of_slot = (
            torch.arange(self.total_slots, dtype=torch.long, device=device)
            // self.capacity
        )
        self.frame_offsets = torch.arange(tokens_per_frame, dtype=torch.long, device=device)
        self._current_base = self.frame_offsets + self.ring_len

    @property
    def memory_bytes(self) -> int:
        """Resident bytes of KV storage, all worlds (the bool/index buffers are
        noise)."""
        return self.kv.numel() * self.kv.element_size()

    def world_span(self, world_idx: int) -> tuple[int, int]:
        """``[lo, hi)`` token slots owned by ``world_idx``.

        A host ``int`` here, unlike everywhere on the forward path: the two
        callers below are host-side lifecycle (a rollout ending, capture
        tearing down), never inside a captured region.
        """
        if not 0 <= world_idx < self.num_worlds:
            raise IndexError(
                f"world_idx {world_idx} out of range for {self.num_worlds} worlds."
            )
        lo = world_idx * self.capacity
        return lo, lo + self.capacity

    def reset(self, world_idx: int) -> None:
        """Drop ONE world's state and re-arm its scratch tail."""
        lo, hi = self.world_span(world_idx)
        self.kv[:, :, :, lo:hi].zero_()
        self.written[lo:hi].zero_()
        self.written[lo + self.ring_len : hi].fill_(True)

    def upsert(
        self,
        kv: Tensor,
        frame_pos: Tensor,
        commit: bool,
        world_idx: Tensor,
        *,
        build_visibility: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``kv`` is ``[2, 1, H_kv, tokens_per_frame, D]`` for exactly one frame
        of one world;
        ``commit`` writes the frame into its ring slot; without it the frame
        lands in that world's scratch tail only, visible to itself and to
        nothing later.

        Returns ``(k, v, visible)``
        """
        tokens = self.tokens_per_frame

        if not torch.compiler.is_compiling():
            torch._check(
                kv.size(3) == tokens,
                lambda: f"ring cache expects exactly one frame per upsert; got {kv.size(3)} tokens",
            )
            torch._check(
                frame_pos.ndim == 0 and frame_pos.dtype == torch.int64,
                lambda: f"frame_pos must be a [] int64 tensor; got {tuple(frame_pos.shape)} "
                f"{frame_pos.dtype}",
            )
            torch._check(
                tuple(world_idx.shape) == (1,) and world_idx.dtype == torch.int64,
                lambda: f"world_idx must be a [1] int64 tensor; got "
                f"{tuple(world_idx.shape)} {world_idx.dtype}",
            )
        world_base = world_idx * self.capacity
        bucket = (frame_pos + (self.pinned_dilation - 1)) // self.pinned_dilation
        slot = bucket % self.ring_buckets
        ring_idx = self.frame_offsets + slot * tokens + world_base
        current_idx = self._current_base + world_base
        ring_scatter(self.kv, self.written, current_idx, kv, False)

        write_step = frame_pos.remainder(self.pinned_dilation) == 0
        mask_written = self._mask_written
        if build_visibility:
            mask_written.copy_(self.written)
            mask_written &= self._world_of_slot == world_idx
            mask_written[ring_idx] = mask_written[ring_idx] & ~write_step

        if commit:
            dst = torch.where(write_step, ring_idx, current_idx)
            ring_scatter(self.kv, self.written, dst, kv, True)

        k, v = self.kv.unbind(0)
        # ALIASING HAZARD. When ``build_visibility`` is true, the third return
        # value IS `self._mask_written`, this
        # layer's preallocated scratch, handed out by reference and overwritten
        # in place by the next `upsert` on this layer. A consumer that stashes
        # it and reads it later reads some *later* frame's visibility -- which
        # is a mask off by one or more frames, and with worlds resident it can
        # now also be another world's mask entirely.
        #
        # When false, the value is intentionally stale and must be ignored by
        # the planned attention backend. The obligation on a fallback consumer:
        # read it (build the block mask, or
        # clone it) before the next upsert on this same layer. The 4+1 schedule
        # satisfies that trivially.
        return k, v, mask_written
