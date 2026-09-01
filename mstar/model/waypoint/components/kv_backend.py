"""Waypoint's ring KV cache and the model-local attention backend seam.

The cache is not an optimization here, it *is* the world state: Waypoint
denoises one latent frame at a time and everything the model knows about the
past lives in these rings. Evicting a slot is not a cache miss, it is amnesia,
and a position bug does not raise -- it produces plausible, smoothly drifting
video. See ``docs/waypoint/CONTRACTS.md`` sections 1-3, which this file is the
implementation of.

Facts the rest of the port depends on:

  * **Per-layer heterogeneous geometry.** 18 local layers hold 16 consecutive
    frames; 6 global layers hold 16 frames spaced 8 apart, spanning 128 frames
    of history. Every layer carries one extra *scratch* frame at the tail, so
    capacity is ``(ring_frames + 1) * tokens_per_frame``. All of it comes from
    ``WaypointConfig``; nothing is re-derived here.
  * **4+1 passes per frame.** The four Euler denoise passes run with
    ``is_frozen=True`` and touch only the scratch frame -- they must not mutate
    the ring, because each attends to a different noisy version of the same
    frame. The fifth pass (sigma=0, ``is_frozen=False``) is the only writer.
  * **The scratch write is unconditional**, frozen passes included. It is the
    entire mechanism by which the frame being denoised attends to itself.
  * **Global layers commit on one frame in eight.**
    ``torch.where(write_step, ring_idx, current_idx)`` redirects a
    non-committing global write back into the scratch slot it just wrote, so it
    commits nothing. That redirect *is* the dilation, expressed without
    data-dependent control flow.
  * **The mask hides the ring slot this frame is about to overwrite**, on
    frozen and unfrozen passes alike, so the current frame never attends to the
    stale frame it is replacing -- and so all five passes of a frame see
    byte-identical KV.
  * The ``BlockMask`` is query-uniform and full-blocks-only (no partial blocks,
    and a no-op ``mask_mod``), which is what makes "capacity must be a multiple
    of 128" a real constraint rather than a convenience.

Deviation from the reference (``world_engine/src/model/kv_cache.py``), argued in
CONTRACTS section 2.4: global rings are **compacted** to their 16 addressable
slots. The reference allocates 128 frame slots per global layer but
``slot = bucket % 16`` can never address past the 16th, so 7/8 of every global
ring is permanently unwritten and permanently masked off. Dropping never-written
blocks is bit-exact -- ``BlockMask.from_kv_blocks`` orders visited blocks by a
*stable* descending argsort truncated to the visited count, so trailing ``False``
entries neither enter the visited list nor perturb the order of the ``True``
ones, and attention accumulates over the same blocks in the same order. It saves
1.31 GiB (see ``describe_ring_memory``). Set ``WaypointConfig.full_global_ring``
to restore the reference allocation for an A/B parity run.

The one thing compaction changes structurally: the bucket count can no longer be
derived from the buffer length. The reference computes
``num_buckets = (L // tpf) // dilation``, which is only correct because ``L`` is
8x oversized; against a compacted ring it would yield 2 instead of 16 and shred
the history. ``LayerRingCache`` therefore takes ``ring_buckets`` as an explicit
argument, sourced from ``WaypointConfig.ring_buckets``.

Nothing in this module is an ``nn.Module``. The rings are DERIVED state, never
checkpoint state: keeping them out of the DiT's module tree means
``to_empty(device)``, ``state_dict()`` and the weight loader have no buffer of
ours to leave holding garbage (the stance ``wan22.components.dit.Wan22RoPE3D``
takes for its RoPE tables). Storage is therefore allocated **eagerly and
explicitly** on the device handed to the constructor rather than lazily on first
use -- the backend is built after the DiT has been materialized, so there is no
meta-device phase to defer past, and eager allocation means a rollout cannot
discover halfway through that it is 800 MiB short of VRAM.
"""

import dataclasses
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask,
    flex_attention,
)

from mstar.model.waypoint.config import WaypointConfig

__all__ = [
    "FlexRingBackend",
    "LayerRingCache",
    "WaypointKVBackend",
    "describe_ring_memory",
    "flex_attention_masked",
    "make_block_mask",
    "ring_memory_bytes",
]


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
# not depend on whether someone set `WaypointConfig.compile_dit`. See
# docs/waypoint/CONTRACTS.md section 2.3 and DECISIONS.md D10.
flex_attention_masked = torch.compile(flex_attention, dynamic=False)


@runtime_checkable
class WaypointKVBackend(Protocol):
    """The seam between ``WaypointAttn`` and whatever owns the world state.

    ``mstar/engine/resources/attn/`` is deliberately not involved: Waypoint's
    cache is model-owned, ``get_node_resources()`` returns ``[]``, and the
    engine builds no KV resource for it. The node/walk topology is invariant
    across that decision, so if engine-owned paged KV later becomes viable only
    the implementation behind this protocol is replaced -- no graph reshape, no
    submodule signature change.

    The opaque ``meta`` returned by ``upsert`` and consumed by ``attend`` is
    what keeps the protocol independent of FlexAttention: ``FlexRingBackend``
    puts a ``BlockMask`` there, a paged backend would put a page table there,
    and the attention module never has to know which.
    """

    def upsert(
        self,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        frame_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Any]:
        """Commit one frame's K/V for ``layer_idx`` and return what to attend to.

        ``k``/``v`` are ``[B, H_kv, tokens_per_frame, D]``. ``k`` is already
        RoPE'd and RMS-normed and ``v`` is post value-residual lerp: the cache
        stores post-RoPE keys, so replayed history is never re-rotated (see
        CONTRACTS section 4.2). Returns ``(k_all, v_all, meta)`` spanning the
        whole ring plus the scratch frame.

        **Why ``frame_pos`` is passed explicitly** and never derived from an
        internal slot cursor: it is a ``[]`` int64 device tensor holding the
        ring clock -- not a slot id -- and it alone determines both the ring
        slot written and the visibility mask. Desynchronizing it from the
        caller's clock does not raise; it silently rewrites history. An input
        with that failure mode has to be an argument, not hidden state.
        """
        ...

    def attend(self, q: Tensor, k: Tensor, v: Tensor, meta: Any, *, enable_gqa: bool) -> Tensor:
        """Attend ``q`` ``[B, H_q, T, D]`` against the ``(k, v, meta)`` triple
        returned by ``upsert``. Returns ``[B, H_q, T, D]``."""
        ...

    def set_frozen(self, frozen: bool) -> None:
        """``True`` for the four denoise passes, ``False`` for the committing
        pass. Python-level state on purpose: it gates a real branch, and
        branching on it is graph-safe."""
        ...

    def reset(self) -> None:
        """Drop the world state and re-freeze. A new rollout starts here."""
        ...

    def get_state(self) -> dict: ...

    def load_state(self, state: dict) -> None: ...


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


class LayerRingCache:
    """One attention layer's ring: ``ring_frames`` frame slots of history plus
    one scratch frame at the tail.

    Storage is a single ``[2, B, H_kv, capacity, D]`` tensor so that a commit is
    one ``index_copy_`` for K and V together and the read is one ``unbind(0)``
    into two views -- no copy on the read path.

    ``ring_buckets`` is the number of *addressable* slots and is passed in, not
    derived from ``ring_frames``: under the compacted global allocation the two
    differ from the reference's relationship (16 slots over a 16-frame buffer at
    stride 8, where the reference had 16 slots over a 128-frame buffer), and
    re-deriving it would silently produce 2.
    """

    def __init__(
        self,
        batch: int,
        n_kv_heads: int,
        ring_frames: int,
        ring_buckets: int,
        d_head: int,
        tokens_per_frame: int,
        pinned_dilation: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
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

        self.tokens_per_frame = tokens_per_frame
        self.ring_frames = ring_frames
        self.ring_buckets = ring_buckets
        self.pinned_dilation = pinned_dilation
        # ring_len is the reference's `L`: the history region, scratch excluded.
        self.ring_len = ring_frames * tokens_per_frame
        self.capacity = self.ring_len + tokens_per_frame

        self.kv = torch.zeros(
            2, batch, n_kv_heads, self.capacity, d_head, dtype=dtype, device=device
        )

        # The tail frame is permanently visible: it always holds the frame
        # currently being denoised, so masking it would remove self-attention.
        written = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        written[self.ring_len :] = True
        self.written = written
        # Preallocated scratch for the per-call visibility mask. Allocating it
        # inside upsert would put a fresh buffer in the compiled region on every
        # one of the 120 upserts per frame.
        self._mask_written = torch.empty_like(written)

        self.frame_offsets = torch.arange(tokens_per_frame, dtype=torch.long, device=device)
        self.current_idx = self.frame_offsets + self.ring_len

    @property
    def memory_bytes(self) -> int:
        """Resident bytes of KV storage (the bool/index buffers are noise)."""
        return self.kv.numel() * self.kv.element_size()

    def reset(self) -> None:
        self.kv.zero_()
        self.written.zero_()
        self.written[self.ring_len :].fill_(True)

    def upsert(
        self, kv: Tensor, frame_pos: Tensor, is_frozen: bool
    ) -> tuple[Tensor, Tensor, BlockMask]:
        """``kv`` is ``[2, B, H_kv, tokens_per_frame, D]`` for exactly one frame;
        ``frame_pos`` is a ``[]`` int64 device tensor (the ring clock).

        Ported statement for statement from the reference; CONTRACTS section 2.2
        lists the five ways to get it subtly wrong, all of which drift instead of
        raising. Everything below is index arithmetic on device tensors --
        ``torch.where`` and ``index_copy_`` rather than a Python ``if`` -- because
        this runs inside ``torch.compile(fullgraph=True)`` and a branch on a
        tensor value would graph-break. Only ``is_frozen`` is Python-level.
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

        # Bucket rounds UP (+ dilation - 1), copying the reference verbatim.
        #
        # The rounding is in fact DEAD arithmetic here, and the comment that
        # used to sit on this line -- "flooring rotates the whole history by one
        # slot" -- was measured false: swapping ceil for floor changes nothing,
        # 0.0 across a 26-frame rollout through two ring wraps. `ring_idx` is
        # only ever *read* under `write_step` (below), and where
        # `frame_pos % dilation == 0` the ceil and the floor agree. It is kept
        # because the reference has it and this port does not silently
        # "simplify" expressions it merely believes to be dead -- but do not
        # mistake it for a live invariant.
        bucket = (frame_pos + (self.pinned_dilation - 1)) // self.pinned_dilation
        slot = bucket % self.ring_buckets
        ring_idx = self.frame_offsets + slot * tokens

        # Unconditional, frozen passes included: this is how the frame being
        # denoised attends to itself between Euler steps.
        self.kv.index_copy_(3, self.current_idx, kv)

        # Hide the ring slot this frame is about to take over, so the current
        # frame never attends to the stale frame it is replacing. Done on every
        # pass -- the four frozen passes must see exactly the KV the committing
        # pass will see -- but only where write_step, hence the `& ~write_step`.
        write_step = frame_pos.remainder(self.pinned_dilation) == 0
        mask_written = self._mask_written
        mask_written.copy_(self.written)
        mask_written[ring_idx] = mask_written[ring_idx] & ~write_step
        bm = make_block_mask(tokens, self.capacity, mask_written)

        if not is_frozen:
            # On a global layer's 7-in-8 non-committing frames this redirects
            # the commit onto the scratch slot that was just written with the
            # same data, i.e. it commits nothing. That is the dilation, and it
            # is a select rather than a branch so the graph stays whole.
            #
            # `is_frozen` is REDUNDANT under the shipped 4+1 schedule, measured:
            # ignoring it entirely (committing on all five passes) gives 0.0 at
            # the output AND a byte-identical ring, because the mask-hide above
            # already blinds the current frame to this slot and the fifth pass
            # rewrites the same `dst` last. That is a genuine no-op and not an
            # untested patch: the same monkeypatch produces 5.8e-02 when it
            # suppresses the commit instead, so it was demonstrably live.
            #
            # Do not delete it on that basis. It stops being redundant the
            # moment anything reads the ring between the frozen passes and the
            # cache pass, or the pass order changes, or two frames are in
            # flight at once (B8). Redundant-under-this-schedule is not the
            # same as unnecessary.
            dst = torch.where(write_step, ring_idx, self.current_idx)
            self.kv.index_copy_(3, dst, kv)
            self.written[dst] = True

        k, v = self.kv.unbind(0)
        return k, v, bm


class FlexRingBackend:
    """``WaypointKVBackend`` over per-layer ring caches plus FlexAttention.

    **Why FlexAttention and not FlashInfer**, given that mstar has a paged
    FlashInfer path: numerical parity. The reference runs ``flex_attention``
    with a full-block-only ``BlockMask``; a paged kernel changes the
    accumulation order, which would make bit-exact comparison against the
    reference impossible and defeat the point of the parity harness.

    Constructed after the DiT is materialized, with a real device -- see the
    module docstring on why none of this rides through ``to_empty``.
    """

    def __init__(
        self,
        config: WaypointConfig,
        device: torch.device | str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 1,
    ):
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        self.batch_size = batch_size
        # Convenience mirror of the config fact, so the attention module can
        # read it off the backend; `attend` still takes it explicitly because
        # the protocol says so and the caller owns its own head counts.
        self.enable_gqa = config.enable_gqa

        self.layers = [
            LayerRingCache(
                batch=batch_size,
                n_kv_heads=config.n_kv_heads,
                ring_frames=config.ring_frames(i),
                ring_buckets=config.ring_buckets(i),
                d_head=config.d_head,
                tokens_per_frame=config.tokens_per_frame,
                pinned_dilation=config.pinned_dilation(i),
                dtype=dtype,
                device=self.device,
            )
            for i in range(config.n_layers)
        ]
        # A fresh backend is frozen: nothing may commit until the model's cache
        # pass explicitly unfreezes it.
        self._is_frozen = True

    # ---- WaypointKVBackend ------------------------------------------------

    def upsert(
        self, k: Tensor, v: Tensor, layer_idx: int, frame_pos: Tensor
    ) -> tuple[Tensor, Tensor, BlockMask]:
        # `layer_idx` is Python-level (it indexes a list of differently-shaped
        # rings), so indexing on it is graph-safe.
        kv = torch.stack([k, v], dim=0)
        return self.layers[layer_idx].upsert(kv, frame_pos, self._is_frozen)

    def attend(self, q: Tensor, k: Tensor, v: Tensor, meta: BlockMask, *, enable_gqa: bool) -> Tensor:
        # `flex_attention_masked`, never bare `flex_attention`: with a no-op
        # `mask_mod` the eager path ignores the block mask entirely and attends
        # to unwritten ring slots. See the note at its definition.
        return flex_attention_masked(q, k, v, block_mask=meta, enable_gqa=enable_gqa)

    def set_frozen(self, frozen: bool) -> None:
        self._is_frozen = bool(frozen)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self._is_frozen = True

    @torch.no_grad()
    def get_state(self) -> dict:
        """Snapshot the world state. Cloned, so the caller can hold it across
        further rollout steps that mutate the rings in place."""
        return {
            "_is_frozen": self._is_frozen,
            "layers": [
                (layer.kv.detach().clone(), layer.written.detach().clone())
                for layer in self.layers
            ],
        }

    @torch.no_grad()
    def load_state(self, state: dict) -> None:
        layers = state["layers"]
        if len(layers) != len(self.layers):
            raise ValueError(
                f"state has {len(layers)} layers, backend has {len(self.layers)}."
            )
        for i, (layer, (kv, written)) in enumerate(zip(self.layers, layers, strict=True)):
            # Geometry mismatch (e.g. a 360p state into a 720p backend, or a
            # compacted state into a full_global_ring backend) would otherwise
            # surface as a copy_ broadcast error deep in the loop.
            if tuple(kv.shape) != tuple(layer.kv.shape):
                raise ValueError(
                    f"layer {i} state shape {tuple(kv.shape)} != ring shape "
                    f"{tuple(layer.kv.shape)}."
                )
            layer.kv.copy_(kv)
            layer.written.copy_(written)
        self._is_frozen = bool(state.get("_is_frozen", True))

    # ---- Introspection ----------------------------------------------------

    def memory_bytes(self) -> int:
        """Total resident ring bytes across all layers."""
        return sum(layer.memory_bytes for layer in self.layers)

    def describe(self) -> str:
        """Human-readable geometry/footprint table, including what the other
        setting of ``full_global_ring`` would cost."""
        return describe_ring_memory(self.config, batch_size=self.batch_size, dtype=self.dtype)


def ring_memory_bytes(
    config: WaypointConfig, *, batch_size: int = 1, dtype: torch.dtype = torch.bfloat16
) -> list[int]:
    """Per-layer ring bytes for ``config``, computed without allocating anything
    (so it can be called on a laptop while sizing a deployment)."""
    per_slot = 2 * batch_size * config.n_kv_heads * config.d_head * dtype.itemsize
    return [per_slot * config.kv_capacity(i) for i in range(config.n_layers)]


def _fmt_bytes(n: int) -> str:
    return f"{n / 2**20:.1f} MiB" if n < 2**30 else f"{n / 2**30:.2f} GiB"


def describe_ring_memory(
    config: WaypointConfig, *, batch_size: int = 1, dtype: torch.dtype = torch.bfloat16
) -> str:
    """Geometry and footprint of every ring, grouped local vs global, with the
    counterfactual under the opposite ``full_global_ring`` setting.

    The compacted default is what a reviewer should see; the ``full_global_ring``
    line is the reference's allocation, 8/9ths of whose global storage is
    permanently unwritten (CONTRACTS section 2.4).
    """
    per_layer = ring_memory_bytes(config, batch_size=batch_size, dtype=dtype)
    total = sum(per_layer)

    lines = [
        f"Waypoint ring KV  variant={config.variant}  batch={batch_size}  dtype={dtype}  "
        f"full_global_ring={config.full_global_ring}"
    ]
    groups = (
        ("local ", [i for i in range(config.n_layers) if not config.is_global_layer(i)]),
        ("global", sorted(config.global_layers)),
    )
    for name, indices in groups:
        if not indices:
            continue
        i = indices[0]
        lines.append(
            f"  {name} x{len(indices):2d}  "
            f"{config.ring_frames(i):3d} ring frames @ stride {config.pinned_dilation(i)} "
            f"({config.ring_buckets(i)} addressable) + 1 scratch  "
            f"= {config.kv_capacity(i):6d} tok  "
            f"= {_fmt_bytes(per_layer[i]):>9s}/layer  "
            f"= {_fmt_bytes(sum(per_layer[j] for j in indices)):>9s}"
        )
    lines.append(f"  total {_fmt_bytes(total)}  ({total} bytes)")

    other = dataclasses.replace(config, full_global_ring=not config.full_global_ring)
    other_total = sum(ring_memory_bytes(other, batch_size=batch_size, dtype=dtype))
    delta = other_total - total
    lines.append(
        f"  full_global_ring={other.full_global_ring} would use {_fmt_bytes(other_total)} "
        f"({'+' if delta > 0 else '-'}{_fmt_bytes(abs(delta))})"
    )
    return "\n".join(lines)
