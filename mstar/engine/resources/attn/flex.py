"""Attention over a ring KV cache, through a compiled ``flex_attention``.

The visibility of a ring cannot be expressed as a sequence length: the slots a
step may read are scattered across a fixed buffer that is overwritten in place,
so there is no prefix to pass a kernel. FlexAttention's ``BlockMask`` names the
readable KV blocks directly, which is what makes the ring expressible at all.

The mask this builds is query-uniform and full-blocks-only — every query block
sees the same KV blocks, and a visible block is wholly visible — because writes
are whole frames of tokens. That is what makes "capacity must be a multiple of
128" a real constraint here rather than a convenience.
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask,
    flex_attention,
)

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.base import CGSlotSpec
from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.kv.ring.manager import RingPlan
from mstar.engine.resources.step import StepContext

__all__ = ["FlexAttentionManager", "flex_attention_masked", "make_block_mask"]


# CORRECTNESS, not speed. Our BlockMask carries a NO-OP `mask_mod`: we pass
# `mask_mod=None` to `from_kv_blocks` and it substitutes `noop_mask`, so
# `bm.mask_mod` is a function returning True everywhere, not `None`. Visibility
# is therefore encoded *entirely* in the block index lists (`full_kv_indices`
# truncated to `full_kv_num_blocks`), which is what makes a ring expressible at
# all. The compiled kernel iterates exactly those blocks. The eager path does
# not -- it rebuilds the mask by evaluating `mask_mod` over the grid, and a noop
# mask_mod means "everything is visible", so eager attention silently reads
# every unwritten slot in the ring as a zero K/V and blends it in.
#
# Measured on the ported cache: eager output diverges from a masked-dense
# reference by 2.7e-01, while the compiled path matches it to 1.2e-07. Nothing
# raises. This is why the reference wraps both of its regions in
# @torch.compile(fullgraph=True) -- compilation is load-bearing for the *result*
# there too, not just the throughput.
#
# So the compile is pinned here rather than left to the caller: correctness must
# not depend on whether someone set `WaypointConfig.compile_dit`.
flex_attention_masked = torch.compile(flex_attention, dynamic=False)


def make_block_mask(q_len: int, kv_len: int, written: Tensor) -> BlockMask:
    """Build the query-uniform, full-blocks-only ``BlockMask`` over ``written``.

    ``written`` is ``[kv_len]`` bool, True where the ring holds valid KV. Both
    lengths must be exact multiples of the 128-token sparse block size and
    ``written`` must be block-aligned -- both hold because writes are whole
    frames of 512 (or 128) tokens.

    Two properties are load-bearing. Every query block sees the same KV blocks,
    so the ``[1, 1, 1, num_kv_blocks]`` row broadcasts over query blocks and
    ``compute_q_blocks=False`` is safe. And every visible block is *full*
    (no partial blocks at all), because frame-granular writes
    mean "any token in the block is written" and "all of them are" coincide.
    """
    block_size = _DEFAULT_SPARSE_BLOCK_SIZE

    if not torch.compiler.is_compiling():
        torch._check(
            q_len % block_size == 0,
            lambda: f"q_len ({q_len}) must be a multiple of block size ({block_size})",
        )
        torch._check(
            kv_len % block_size == 0,
            lambda: f"kv_len ({kv_len}) must be a multiple of block size ({block_size})",
        )

    q_blocks = q_len // block_size
    kv_blocks = kv_len // block_size

    written_blocks = written.view(kv_blocks, block_size)
    block_any = written_blocks.any(-1)
    if not torch.compiler.is_compiling():
        assert torch.equal(block_any, written_blocks.all(-1)), "written must be block-aligned"

    full_bm = block_any[None, :].expand(q_blocks, kv_blocks)
    full_kv_num_blocks = full_bm.sum(dim=-1, dtype=torch.int32)[None, None].contiguous()
    # Stable descending argsort: the visited list is this truncated to
    # full_kv_num_blocks, so unwritten blocks sort to the tail and are never
    # read. That is exactly why compacting the global ring is bit-exact.
    full_kv_indices = (
        full_bm.argsort(dim=-1, descending=True, stable=True)
        .to(torch.int32)[None, None]
        .contiguous()
    )

    # No partial blocks at all -- these two exist only to satisfy the signature.
    kv_num_blocks = torch.zeros(
        (1, 1, q_blocks), dtype=torch.int32, device=written.device
    )
    kv_indices = torch.zeros(
        (1, 1, q_blocks, kv_blocks), dtype=torch.int32, device=written.device
    )

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=None,
        seq_lengths=(q_len, kv_len),
        compute_q_blocks=False,
    )


def _empty_block_mask(q_len: int, kv_len: int, device: torch.device) -> BlockMask:
    """Allocate a fixed-address mask whose visible prefix is staged in plan."""
    block_size = _DEFAULT_SPARSE_BLOCK_SIZE
    q_blocks, kv_blocks = q_len // block_size, kv_len // block_size
    full_kv_num_blocks = torch.zeros(
        (1, 1, q_blocks), dtype=torch.int32, device=device
    )
    full_kv_indices = torch.zeros(
        (1, 1, q_blocks, kv_blocks), dtype=torch.int32, device=device
    )
    kv_num_blocks = torch.zeros_like(full_kv_num_blocks)
    kv_indices = torch.zeros_like(full_kv_indices)
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=None,
        seq_lengths=(q_len, kv_len),
        compute_q_blocks=False,
    )

class FlexAttentionManager(AttentionManager):
    """Attention against a ring KV cache, through ``flex_attention``.

    **Why FlexAttention and not the paged FlashInfer path** that already
    exists here: numerical parity with the reference implementation, which
    runs ``flex_attention`` with a full-block-only ``BlockMask``. A paged
    kernel changes the accumulation order over the KV blocks, so a port on top
    of it could never be compared bit-exactly against the reference — and for
    the model this exists to serve, that comparison is the only check there
    is: a mask or position bug does not raise, it produces plausible, smoothly
    drifting video. The FlashInfer alternative was measured at ~9% on the
    attention kernel, against the 2x that had motivated trying it.

    Stateless across steps. The world state is the ring, and the ring belongs
    to the KV resource this one names in ``depends_on``; nothing here survives
    a call except the label/layer cursors the base class defines.
    """

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: KVConfig,
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._planned_masks: dict[tuple[int, tuple[int, int, int]], BlockMask] = {}
        self._visibility_tables: dict[
            tuple[int, int, int], tuple[Tensor, Tensor, int]
        ] = {}
        self._active_slot: int | None = None

    def depends_on(self) -> set[str]:
        return {self._kv_cache_name}

    @property
    def requires_kv_write(self) -> bool:
        """False: the layer has already written this step's K/V.

        A ring ``upsert`` is not the paged backends' ``write_kv``. It writes
        the frame *and* returns the whole ring view to attend against, in one
        call, because which slots the write makes visible is part of its
        result — so there is no separate write step for a layer to perform and
        calling one would commit the frame twice.

        A ``@property``, not a class attribute, to keep the base's read-only
        contract: ``AttentionManager.requires_kv_write`` is a property and
        ``DenseAttentionManager`` overrides it as one. Shadowing it with a
        plain ``False`` would work at runtime and quietly make the attribute
        writable on this subclass alone.
        """
        return False

    @property
    def needs_token_visibility(self) -> bool:
        """Whether the ring must construct its legacy token-level mask.

        Engine execution stages a block mask in ``plan`` and does not need the
        per-layer row. The fallback remains useful for standalone numerical
        calls that invoke ``attend`` without a resource plan.
        """
        return self._active_slot is None

    @staticmethod
    def _geometry(layer) -> tuple[int, int, int]:
        return (layer.ring_frames, layer.ring_buckets, layer.pinned_dilation)

    def _mask_for(
        self, slot: int, geometry: tuple[int, int, int], *, create: bool = False,
    ) -> BlockMask:
        key = (slot, geometry)
        mask = self._planned_masks.get(key)
        if mask is None and create:
            ring_frames, _, _ = geometry
            capacity = (ring_frames + 1) * self._kv_config.tokens_per_frame
            capacity *= self._kv_config.num_worlds
            mask = _empty_block_mask(
                self._kv_config.tokens_per_frame, capacity, self._device
            )
            self._planned_masks[key] = mask
            self._visibility_table_for(geometry)
        if mask is None:
            raise RuntimeError(
                f"FlexAttention mask for slot={slot}, geometry={geometry} was not "
                "allocated before CUDA graph capture"
            )
        return mask

    def _visibility_table_for(
        self, geometry: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor, int]:
        """Return immutable device rows for every distinct ring phase.

        Before the first wrap, visibility grows with ``frame_pos``. Once every
        addressable bucket has been written, it is periodic over
        ``ring_buckets * pinned_dilation`` frames. Keeping both the growing and
        periodic phases makes the lookup finite even though rollout clocks are
        not capped by the resource itself.
        """
        existing = self._visibility_tables.get(geometry)
        if existing is not None:
            return existing

        ring_frames, ring_buckets, dilation = geometry
        period = ring_buckets * dilation
        blocks_per_frame = (
            self._kv_config.tokens_per_frame // _DEFAULT_SPARSE_BLOCK_SIZE
        )
        total_blocks = (
            (ring_frames + 1) * blocks_per_frame * self._kv_config.num_worlds
        )
        counts: list[list[int]] = []
        rows: list[list[list[int]]] = []
        for world_idx in range(self._kv_config.num_worlds):
            world_counts = []
            world_rows = []
            for frame_pos in range(2 * period):
                visible = self._visible_blocks_for(
                    geometry, world_idx=world_idx, frame_pos=frame_pos
                )
                world_counts.append(len(visible))
                world_rows.append(visible + [0] * (total_blocks - len(visible)))
            counts.append(world_counts)
            rows.append(world_rows)

        table = (
            torch.tensor(counts, dtype=torch.int32, device=self._device),
            torch.tensor(rows, dtype=torch.int32, device=self._device),
            period,
        )
        self._visibility_tables[geometry] = table
        return table

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del max_bs, max_seq_len
        geometries = {self._geometry(layer) for layer in self._kv_config.layers}
        for slot in {spec.slot for spec in slots}:
            for geometry in geometries:
                self._mask_for(slot, geometry, create=True)

    def _visible_blocks_for(
        self,
        geometry: tuple[int, int, int],
        *,
        world_idx: int,
        frame_pos: int,
    ) -> list[int]:
        tokens = self._kv_config.tokens_per_frame
        blocks_per_frame = tokens // _DEFAULT_SPARSE_BLOCK_SIZE
        ring_frames, ring_buckets, dilation = geometry
        capacity_blocks = (ring_frames + 1) * blocks_per_frame
        world_base = world_idx * capacity_blocks

        committed = range(0, frame_pos, dilation)
        slots = {
            (frame // dilation) % ring_buckets for frame in committed
        }
        if frame_pos % dilation == 0:
            slots.discard((frame_pos // dilation) % ring_buckets)

        visible: list[int] = []
        for ring_slot in sorted(slots):
            start = world_base + ring_slot * blocks_per_frame
            visible.extend(range(start, start + blocks_per_frame))
        scratch = world_base + ring_frames * blocks_per_frame
        visible.extend(range(scratch, scratch + blocks_per_frame))
        return visible

    def _stage(
        self,
        mask: BlockMask,
        geometry: tuple[int, int, int],
        plan: RingPlan,
    ) -> None:
        counts, indices, period = self._visibility_table_for(geometry)
        phase = (
            plan.frame_pos
            if plan.frame_pos < period
            else period + plan.frame_pos % period
        )
        mask.full_kv_num_blocks.copy_(counts[plan.world_idx, phase])
        mask.full_kv_indices.copy_(indices[plan.world_idx, phase])

    def plan(self, step: AttentionStep, ctx: StepContext) -> None:
        """Stage one local/global visibility mask for this frame and slot."""
        del step
        self.reset_default_cursors()
        ring_plan = ctx.plan_results.get(self._kv_cache_name)
        if ring_plan is None:
            # Keeps standalone manager tests and non-ring diagnostic calls able
            # to use the visibility tensor supplied directly to attend().
            self._active_slot = None
            return
        if not isinstance(ring_plan, RingPlan):
            raise TypeError(
                f"FlexAttention expected RingPlan from {self._kv_cache_name!r}; "
                f"got {type(ring_plan).__name__}"
            )
        self._active_slot = ctx.slot
        geometries = {self._geometry(layer) for layer in self._kv_config.layers}
        for geometry in geometries:
            mask = self._mask_for(ctx.slot, geometry, create=not ctx.capture)
            self._stage(mask, geometry, ring_plan)

    def attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        visible: Tensor,
        *,
        enable_gqa: bool,
        layer_idx: int | None = None,
    ) -> Tensor:
        """One layer's attention over the ring.

        ``q`` is ``[B, H_q, T, D]``; ``k``/``v`` are ``[B, H_kv, kv_len, D]``,
        the whole ring view (history slots plus the scratch frame) as the KV
        resource returned it. ``visible`` is ``[kv_len]`` bool, True where that
        view holds KV this step may read — it is not derivable from a length,
        which is the entire reason this backend exists. Returns
        ``[B, H_q, T, D]``.
        """
        if self._active_slot is None or layer_idx is None:
            block_mask = make_block_mask(q.size(-2), k.size(-2), visible)
        else:
            geometry = self._geometry(self._kv_config.layers[layer_idx])
            block_mask = self._mask_for(self._active_slot, geometry)
        # `flex_attention_masked`, never bare `flex_attention`: with a no-op
        # `mask_mod` the eager path ignores the block mask entirely and attends
        # to unwritten ring slots. See the note at its definition.
        return flex_attention_masked(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)
