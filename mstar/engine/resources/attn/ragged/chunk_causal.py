"""Teacher-forcing attention over a chunked two-stream layout, through a
compiled ``flex_attention``.

The mask (``RaggedChunkCausalAttentionSpec``) depends on the query's stream as
well as its chunk: a noised query of chunk ``c`` reads clean keys of chunks
``< c`` while a clean query of chunk ``c`` reads clean keys of chunks ``<= c``
and no noised key at all. A query block that holds both streams therefore
reads different keys per row, which no list of key ranges or pages per query
block expresses. FlexAttention can: the ``BlockMask`` lists which tiles to
visit, and its ``mask_mod`` decides each pair inside a partial tile.

Every token carries one int32 *tag*: its segment (request) in the high half
and its code in the low half, the code being 0 for a global token, ``2c + 1``
for a clean token of chunk ``c`` and ``2c + 2`` for a noised one. The whole
rule is then four integer tests on two tags (``chunk_causal_visible``), the
same function the host evaluates per run to list tiles and the kernel
evaluates per pair inside one. One tag per token rather than a table per
property lets the same predicate serve both tile planning and kernel masking.

Unlike ``attn.flex``'s ring mask, this ``mask_mod`` is the exact predicate, so
eager and compiled ``flex_attention`` agree; the compile is for speed and for
not materialising the scores.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE, BlockMask

from mstar.engine.resources.attn.flex import _FLEX_BACKEND, flex_attention_masked
from mstar.engine.resources.attn.ragged.base import RaggedChunkCausalAttnManager, _RaggedManagerBase
from mstar.engine.resources.attn.ragged.config import (
    ChunkCausalAttentionStep,
    ChunkCausalLayout,
    RaggedAttentionConfig,
)
from mstar.engine.resources.step import StepContext
from mstar.platforms import current_platform

__all__ = [
    "FlexChunkCausalManager",
    "FlexChunkCausalWrapper",
    "chunk_causal_blocks",
    "chunk_causal_runs",
    "chunk_causal_visible",
]

# A tag is ``segment * _SEGMENT_UNIT + code``. Masks rather than shifts, so the
# predicate needs only the integer ops every flex backend lowers (the reference
# mask_mod uses the same ones).
_SEGMENT_UNIT = 1 << 16
_CODE_MASK = _SEGMENT_UNIT - 1
_SEGMENT_MASK = 0x7FFF_0000
# the segment of a captured bucket's padding tokens, past any real one: they
# attend only each other, so their rows stay finite, and no real row sees them
PAD_SEGMENT = 0x7FFF
_PAD_TAG = PAD_SEGMENT * _SEGMENT_UNIT
# chunk c's codes are 2c+1 and 2c+2, which must stay below the segment half
MAX_CHUNKS = (_CODE_MASK - 2) // 2


def chunk_causal_visible(q_tag, k_tag):
    """Whether key ``k_tag`` is visible to query ``q_tag``; elementwise over
    tensors or numpy arrays alike.

    Within one segment, ``k`` is visible when it is global (code 0), when both
    are the same stream and chunk (``k == q``: a noised chunk is bidirectional
    within itself), or when ``k`` is clean and ``k <= hi(q)``: a clean query
    takes the clean keys of its own and earlier chunks (``hi = q``), a noised
    query those of earlier chunks only (``hi = q - 3``), a global query none
    (``hi = -3``). Clean codes are odd and noised codes even, so a clean query
    never matches a noised key.
    """
    q_code = q_tag & _CODE_MASK
    k_code = k_tag & _CODE_MASK
    # q for an odd (clean) code, q - 3 otherwise, without a where: one formula
    # for torch and numpy, and arithmetic is what every backend's mask lowers
    hi = q_code - 3 + 3 * (q_code & 1)
    same_segment = (q_tag & _SEGMENT_MASK) == (k_tag & _SEGMENT_MASK)
    rule = (k_code == q_code) | (k_code == 0) | (((k_code & 1) == 1) & (k_code <= hi))
    return same_segment & rule


def _check_layouts(layouts: Sequence[ChunkCausalLayout]) -> None:
    if len(layouts) >= PAD_SEGMENT:
        raise ValueError(f"chunk-causal attention: {len(layouts)} segments, at most {PAD_SEGMENT - 1}")
    for i, lay in enumerate(layouts):
        if min(lay.global_len, lay.noised_len, lay.clean_len) < 0 or lay.chunk_len <= 0:
            raise ValueError(f"chunk-causal attention: segment {i} has an invalid layout {lay}")
        chunks = max(-(-lay.noised_len // lay.chunk_len), -(-lay.clean_len // lay.chunk_len))
        if chunks > MAX_CHUNKS:
            raise ValueError(f"chunk-causal attention: segment {i} has {chunks} chunks, at most {MAX_CHUNKS}")


def chunk_causal_runs(
    layouts: Sequence[ChunkCausalLayout], length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(starts, ends, tags)`` of the maximal runs of equal tag in the packed
    layout of ``layouts``, padded to ``length`` tokens with ``PAD_SEGMENT``.

    A run is the unit of the mask: every token of it sees the same keys. There
    are a few per chunk, so the host plans in runs, never in tokens.
    """
    tags: list[int] = []
    lens: list[int] = []

    def run(n: int, tag: int) -> None:
        if n > 0:
            lens.append(n)
            tags.append(tag)

    for seg, lay in enumerate(layouts):
        base = seg * _SEGMENT_UNIT
        run(lay.global_len, base)
        for c, s in enumerate(range(0, lay.noised_len, lay.chunk_len)):
            run(min(lay.chunk_len, lay.noised_len - s), base + 2 * c + 2)
        for c, s in enumerate(range(0, lay.clean_len, lay.chunk_len)):
            run(min(lay.chunk_len, lay.clean_len - s), base + 2 * c + 1)
    total = sum(lens)
    if total > length:
        raise ValueError(f"chunk-causal attention: {total} tokens exceeds the {length} planned for")
    run(length - total, _PAD_TAG)
    ends = np.cumsum(np.array(lens, dtype=np.int64))
    return ends - np.array(lens, dtype=np.int64), ends, np.array(tags, dtype=np.int64)


def chunk_causal_blocks(
    starts: np.ndarray, ends: np.ndarray, tags: np.ndarray, length: int, block_size: int, full_blocks: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The tiles of the mask over runs ``(starts, ends, tags)`` covering
    ``[0, length)``: ``(any, partial, full)``, each ``[num_blocks, num_blocks]``
    bool over (query block, key block).

    Exact, not a bound: a tile is listed iff some pair in it is visible, and
    full iff every pair is and both blocks are whole. ``full_blocks=False``
    lists every visible tile as partial (``full`` None), which is what the
    TRITON backend runs: the kernel visits partial tiles and full tiles in two
    passes, so splitting them changes the accumulation order, and the
    reference's Triton path does not split.
    """
    num_blocks = -(-length // block_size)
    lo = np.arange(num_blocks, dtype=np.int64) * block_size
    hi = np.minimum(lo + block_size, length)
    # run r lies (partly) in block b
    incidence = ((starts[None, :] < hi[:, None]) & (ends[None, :] > lo[:, None])).astype(np.float64)
    visible = chunk_causal_visible(tags[:, None], tags[None, :])
    # counts of (query run, key run) pairs per tile, exact in float64 at any size here
    any_ = (incidence @ visible.astype(np.float64) @ incidence.T) > 0
    if not full_blocks:
        return any_, any_, None
    whole = (hi - lo) == block_size
    hidden = incidence @ (~visible).astype(np.float64) @ incidence.T
    full = any_ & (hidden == 0) & whole[:, None] & whole[None, :]
    return any_, any_ & ~full, full


def _packed(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per query block, the count and the ascending indices of its listed key
    blocks, left-packed; the tail of each row is never read."""
    counts = adjacency.sum(axis=1).astype(np.int32)
    # stable argsort of "not listed" puts the listed columns first, in order
    indices = np.argsort(~adjacency, axis=1, kind="stable").astype(np.int32)
    return counts[None, None], indices[None, None]


def resolve_mask_block_size(declared: int | None, device: torch.device) -> int:
    """The ``BlockMask`` tile: the declared size, else flex's default, rounded
    up to a multiple of 256 where FLASH runs on Blackwell (SM100, SM110), whose
    FA4 kernel processes a query tile as two 128-row stages. The reference makes
    the same choice (``praxis.attention.flash_mask_block_size``)."""
    block_size = declared if declared is not None else _DEFAULT_SPARSE_BLOCK_SIZE
    capability = current_platform.cuda_capability(device)
    if _FLEX_BACKEND == "FLASH" and capability is not None and capability[0] in (10, 11):
        block_size = -(-block_size // 256) * 256
    return block_size


def full_blocks_for_backend() -> bool:
    """Whether to list whole visible tiles as full: FLASH needs them, TRITON is
    left with partial tiles only (see ``chunk_causal_blocks``)."""
    return _FLEX_BACKEND == "FLASH"


class FlexChunkCausalWrapper:
    """One (label, slot)'s plan: the ``BlockMask`` of a packed
    ``[total, H, D]`` layout and the token tags its ``mask_mod`` reads.

    Eagerly the mask covers exactly the planned tokens and is rebuilt when the
    layout changes. In graph mode every buffer is allocated once at
    ``max_total_tokens``, the width the bucket replays at, and each plan stages
    into it: the captured kernel holds their addresses. The tokens past the
    layout are padding in ``PAD_SEGMENT``, and their output rows are zeroed in
    the graph, as the ragged wrappers zero theirs.
    """

    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        full_blocks: bool,
        max_num_segments: int | None = None,
        max_total_tokens: int | None = None,
        device: torch.device = torch.device("cuda"),
        accelerator_graph: bool = False,
        sm_scale: float | None = None,
        q_data_type: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.full_blocks = full_blocks
        self.sm_scale = float(sm_scale) if sm_scale is not None else head_dim ** -0.5
        self.q_data_type = q_data_type
        self.accelerator_graph = accelerator_graph
        self.max_num_segments = max_num_segments
        self.max_total_tokens = max_total_tokens
        self._num_segments = 0
        # the layouts this wrapper's buffers describe: a plan of the same ones
        # (every step of a denoise loop) stages nothing
        self._planned: tuple[ChunkCausalLayout, ...] | None = None
        self._length = 0
        self._tags: torch.Tensor | None = None
        self.block_mask: BlockMask | None = None
        if not accelerator_graph:
            return

        assert max_num_segments is not None and max_num_segments > 0, "max_num_segments required for graph mode"
        assert max_total_tokens is not None and max_total_tokens > 0, "max_total_tokens required for graph mode"
        self._length = max_total_tokens
        n = -(-max_total_tokens // block_size)
        i32 = dict(dtype=torch.int32, device=device)
        self._tags = torch.full((max_total_tokens,), _PAD_TAG, **i32)
        self.block_mask = self._mask_over(
            self._tags, max_total_tokens,
            torch.zeros((1, 1, n), **i32), torch.zeros((1, 1, n, n), **i32),
            torch.zeros((1, 1, n), **i32) if full_blocks else None,
            torch.zeros((1, 1, n, n), **i32) if full_blocks else None,
        )

    @property
    def num_segments(self) -> int:
        """Real (unpadded) segment count from the most recent ``plan``."""
        return self._num_segments

    def forget_plan(self) -> None:
        """A dropped pre-plan's H2Ds went on the plan stream, which nothing
        orders the replacing step against, so the next plan must stage again."""
        self._planned = None

    def _mask_over(self, tags, length, kv_num, kv_idx, full_num, full_idx) -> BlockMask:
        def mask_mod(b, h, q_idx, kv_idx):
            return chunk_causal_visible(tags[q_idx], tags[kv_idx])

        return BlockMask.from_kv_blocks(
            kv_num, kv_idx, full_num, full_idx,
            BLOCK_SIZE=self.block_size, mask_mod=mask_mod,
            seq_lengths=(length, length), compute_q_blocks=False,
        )

    def _host(self, array: np.ndarray) -> torch.Tensor:
        # fresh and pinned per plan: the copy out of it is non-blocking, and a
        # reused source could be overwritten while it is in flight
        t = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int32))
        return t.pin_memory() if self.device.type == "cuda" else t

    def _to_device(self, array: np.ndarray | None) -> torch.Tensor | None:
        return None if array is None else self._host(array).to(self.device, non_blocking=True)

    def _stage(self, buffer: torch.Tensor | None, array: np.ndarray | None) -> None:
        if buffer is not None:
            buffer.copy_(self._host(array), non_blocking=True)

    @torch.compiler.disable
    def plan(self, layouts: Sequence[ChunkCausalLayout]) -> None:
        layouts = tuple(layouts)
        if layouts == self._planned:
            return
        _check_layouts(layouts)
        total = sum(lay.span for lay in layouts)
        if self.accelerator_graph:
            if len(layouts) > self.max_num_segments:
                raise ValueError(
                    f"FlexChunkCausalWrapper: {len(layouts)} segments exceeds the "
                    f"{self.max_num_segments} this graph-mode wrapper was built for"
                )
            length = self.max_total_tokens
        else:
            length = total
        starts, ends, tags = chunk_causal_runs(layouts, length)
        if length == 0:
            self._length, self.block_mask, self._tags = 0, None, None
            self._num_segments = len(layouts)
            self._planned = layouts
            return
        _, partial, full = chunk_causal_blocks(starts, ends, tags, length, self.block_size, self.full_blocks)
        kv_num, kv_idx = _packed(partial)
        full_num, full_idx = _packed(full) if full is not None else (None, None)
        token_tags = np.repeat(tags, ends - starts)
        if not self.accelerator_graph:
            device_tags = self._to_device(token_tags)
            block_mask = self._mask_over(
                device_tags, length, *(self._to_device(a) for a in (kv_num, kv_idx, full_num, full_idx)),
            )
            self._length = length
            self._tags = device_tags
            self.block_mask = block_mask
            self._num_segments = len(layouts)
            self._planned = layouts
            return
        mask = self.block_mask
        # A partial stage invalidates the old mask, even if the failed plan is
        # retried with the old layout.
        self._planned = None
        self._stage(self._tags, token_tags)
        self._stage(mask.kv_num_blocks, kv_num)
        self._stage(mask.kv_indices, kv_idx)
        self._stage(mask.full_kv_num_blocks, full_num)
        self._stage(mask.full_kv_indices, full_idx)
        self._num_segments = len(layouts)
        self._planned = layouts

    @torch.compiler.disable
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False) -> torch.Tensor:
        """``[total, H, D]`` q/k/v packed by the planned layouts -> ``[total, H, D]``.

        ``total`` is the planned token count eagerly and ``max_total_tokens`` in
        graph mode: the kernel is compiled for one length, and a length the mask
        does not describe would read tags and tiles of another layout.
        """
        if return_lse:
            raise ValueError("chunk-causal attention does not return the log-sum-exp")
        if q.shape[0] != self._length:
            raise ValueError(
                f"FlexChunkCausalWrapper: q has {q.shape[0]} rows, the plan describes {self._length}"
                + (" (a graph-mode plan is the bucket's width)" if self.accelerator_graph else "")
            )
        if self._length == 0:
            return q.new_empty(q.shape, dtype=self.q_data_type)
        q, k, v = (t.to(self.q_data_type).transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        out = flex_attention_masked(
            q, k, v, block_mask=self.block_mask, scale=self.sm_scale,
            enable_gqa=self.num_qo_heads != self.num_kv_heads,
        )
        out = out[0].transpose(0, 1)
        if self.accelerator_graph:
            # a select, not a multiply: padding rows may hold inf, and inf * 0 is NaN
            real = (self._tags & _SEGMENT_MASK) != _PAD_TAG
            out = torch.where(real[:, None, None], out, torch.zeros((), dtype=out.dtype, device=out.device))
        return out.contiguous()


class FlexChunkCausalManager(_RaggedManagerBase, RaggedChunkCausalAttnManager):
    """Chunk-causal teacher-forcing attention: every label in the step is its
    own packed layout, each segment laid out by its ``ChunkCausalLayout``.

    The shared ragged machinery supplies what the other kinds have: a wrapper
    per (bucket, slot, label) and per eager (label, slot), the pre-plan
    protocol, and ``force_double_buffer``, since plan stages into buffers a
    captured replay reads.
    """

    wrapper_class = FlexChunkCausalWrapper

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: RaggedAttentionConfig,
        key: str,
        block_size: int,
        full_blocks: bool,
    ):
        super().__init__(device=device, dtype=dtype, config=config, key=key)
        self._kwargs.update(block_size=block_size, full_blocks=full_blocks)

    @property
    def block_size(self) -> int:
        return self._kwargs["block_size"]

    def _plan_layouts(self, step: ChunkCausalAttentionStep, ctx: StepContext, plan_states, shard_plan) -> None:
        # takes no split steps (`supports_split_steps`), so has no shard to read
        if not isinstance(step, ChunkCausalAttentionStep):
            raise TypeError(
                f"chunk-causal attention is stepped with ChunkCausalAttentionStep, got {type(step).__name__}"
            )
        segments = list(step.segments or ())
        if len(step.layouts) != len(segments):
            raise ValueError(
                f"ChunkCausalAttentionStep: {len(step.layouts)} layouts for {len(segments)} segments"
            )
        by_label: dict[str, list[ChunkCausalLayout]] = {}
        for seg, lay in zip(segments, step.layouts, strict=True):
            if lay.span != seg.span:
                raise ValueError(
                    f"ChunkCausalAttentionStep: request {seg.request_id!r}'s layout spans {lay.span} "
                    f"tokens, its segment {seg.span}"
                )
            by_label.setdefault(seg.label, []).append(lay)
        for label, layouts in by_label.items():
            wrapper = self._wrapper_for(ctx.slot_lease, ctx.slot, label)
            wrapper.plan(layouts)
            plan_states[label] = wrapper


def dense_visibility(layouts: Sequence[ChunkCausalLayout]) -> torch.Tensor:
    """``[total, total]`` bool: the mask of ``layouts`` written out per pair.
    For tests and references only; quadratic in the token count."""
    starts, ends, tags = chunk_causal_runs(layouts, sum(lay.span for lay in layouts))
    t = torch.from_numpy(np.repeat(tags, ends - starts))
    return chunk_causal_visible(t[:, None], t[None, :])
