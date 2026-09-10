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
from mstar.engine.resources.kv.config import KVConfig
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
    kv_num_blocks = torch.zeros((1, 1, q_blocks), dtype=torch.int32, device=written.device)
    kv_indices = torch.zeros((1, 1, q_blocks, kv_blocks), dtype=torch.int32, device=written.device)

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

    def plan(self, step: AttentionStep, ctx: StepContext) -> None:
        """Nothing to plan: the mask is a function of the ring's own
        visibility row, which the KV resource hands to ``attend`` per layer.

        The cursors are still cleared, per ``AttentionResource``: a step that
        never binds them must not inherit the previous step's.
        """
        del step, ctx
        self.reset_default_cursors()

    def attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        visible: Tensor,
        *,
        enable_gqa: bool,
    ) -> Tensor:
        """One layer's attention over the ring.

        ``q`` is ``[B, H_q, T, D]``; ``k``/``v`` are ``[B, H_kv, kv_len, D]``,
        the whole ring view (history slots plus the scratch frame) as the KV
        resource returned it. ``visible`` is ``[kv_len]`` bool, True where that
        view holds KV this step may read — it is not derivable from a length,
        which is the entire reason this backend exists. Returns
        ``[B, H_q, T, D]``.
        """
        # Rebuilt every call. The mask is pass-invariant by construction (all
        # passes over one frame see byte-identical KV), so a cache keyed on
        # (layer_idx, frame_pos) would collapse 120 rebuilds per frame to 24.
        # Deliberately NOT taken: there is no measurement of what the rebuild
        # costs against the rest of the frame, and a stale mask is
        # exactly the failure this backend is here to prevent. Measure first.
        block_mask = make_block_mask(q.size(-2), k.size(-2), visible)
        # `flex_attention_masked`, never bare `flex_attention`: with a no-op
        # `mask_mod` the eager path ignores the block mask entirely and attends
        # to unwritten ring slots. See the note at its definition.
        return flex_attention_masked(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)
