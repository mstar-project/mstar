"""Video sparse attention (VSA) as a ragged self-attention backend.

VSA (Zhang et al., "VSA: Faster Video Diffusion with Trainable Sparse Attention") cuts a video
segment's tokens into 4x4x4 tiles of its (t, h, w) token grid, scores every query tile against
every key tile on tile means, and lets each video query tile attend only its top-k key tiles; a
second, coarse branch attends the tile means densely and is added back through a learned
per-token gate. A checkpoint distilled with it (FastVideo's FastH3) expects exactly this math, so
on such a checkpoint VSA is the model's own attention rather than an approximation of dense.

A segment opts in by carrying a ``VideoSparseLayout`` (``Segment.layout``): its rows in packed
order as runs of dense rows (text, audio, ...) and one video token grid. Following FastVideo's
MiniMax-H3 recipe ("exempt" mode), dense rows query every key and every query keeps every dense
key tile; only video-to-video tile pairs are selected. A segment without a layout (a DiT's
padding rows, say) attends densely within itself.

The block-sparse kernel is FastVideo's (``fastvideo_kernel``'s ``ptx_blackwell`` forward,
sm_100a / sm_103a, bf16, head dim 128, 64-token tiles); the tiling, scoring, selection and gate
around it are here, in torch. Eager only: the kernel's per-layer index lists are data-dependent.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from mstar.engine.resources.attn.ragged.config import VideoSparseLayout
from mstar.engine.resources.attn.ragged.wrappers import RaggedPrefillWrapper

logger = logging.getLogger(__name__)

VSA_TILE = (4, 4, 4)
VSA_TILE_ELEMS = math.prod(VSA_TILE)
VSA_HEAD_DIM = 128
# FastVideo's FastH3 8-Step V2 contract (``fastvideo_inference.json`` ``vsa_sparsity``)
DEFAULT_VSA_SPARSITY = 0.8


def _axis_tiles(n: int, tile: int) -> list[tuple[int, int]]:
    return [(start, min(start + tile, n)) for start in range(0, n, tile)]


@dataclass(frozen=True)
class VsaGeometry:
    """A layout's tiles: dense runs chunked into the leading tiles (never across a run's end),
    then the video grid's 3-D tiles in (t, h, w) tile order, each tile's tokens in raster order
    (FastVideo's ``get_tile_partition_indices``). Padded with one empty tile to an even count,
    which the kernel's paired query tiles need. On the device."""

    num_prefix_tiles: int
    num_video_tiles: int
    # [tiles_padded] valid tokens per tile (0 for the pairing tile)
    block_sizes: torch.Tensor
    # [rows] each segment row's slot in the tiled sequence, and its tile
    slot_of_row: torch.Tensor
    tile_of_row: torch.Tensor
    # [tiles_padded * 64] the row in each slot, -1 for a pad slot
    row_of_slot: torch.Tensor

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles

    @property
    def tiles_padded(self) -> int:
        return int(self.block_sizes.numel())


def vsa_tile_order(layout: VideoSparseLayout) -> tuple[list[int], list[torch.Tensor]]:
    """``(tile sizes, rows of each tile)`` over the layout's rows, in tile order (CPU)."""
    sizes: list[int] = []
    tiles: list[torch.Tensor] = []
    video: list[torch.Tensor] = []
    cursor = 0
    for part in layout.parts:
        if isinstance(part, tuple):
            t, h, w = part
            grid = torch.arange(cursor, cursor + t * h * w).view(t, h, w)
            for t0, t1 in _axis_tiles(t, VSA_TILE[0]):
                for h0, h1 in _axis_tiles(h, VSA_TILE[1]):
                    for w0, w1 in _axis_tiles(w, VSA_TILE[2]):
                        video.append(grid[t0:t1, h0:h1, w0:w1].flatten())
            cursor += t * h * w
            continue
        for start in range(0, part, VSA_TILE_ELEMS):
            rows = torch.arange(cursor + start, cursor + min(start + VSA_TILE_ELEMS, part))
            tiles.append(rows)
            sizes.append(int(rows.numel()))
        cursor += part
    sizes += [int(rows.numel()) for rows in video]
    return sizes, tiles + video


@functools.lru_cache(maxsize=16)
def vsa_geometry(layout: VideoSparseLayout, device: torch.device) -> VsaGeometry:
    sizes, tiles = vsa_tile_order(layout)
    num_video = sum(1 for _ in _video_tiles(layout))
    num_prefix = len(sizes) - num_video
    padded = len(sizes) + len(sizes) % 2
    block_sizes = torch.zeros(padded, dtype=torch.int32)
    block_sizes[:len(sizes)] = torch.tensor(sizes, dtype=torch.int32)
    rows = torch.cat(tiles)
    slot = torch.cat([i * VSA_TILE_ELEMS + torch.arange(n) for i, n in enumerate(sizes)])
    tile = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
    slot_of_row = torch.empty(layout.rows, dtype=torch.long)
    tile_of_row = torch.empty(layout.rows, dtype=torch.long)
    slot_of_row[rows], tile_of_row[rows] = slot, tile
    if rows.numel() != layout.rows or torch.unique(rows).numel() != layout.rows:
        raise AssertionError(f"VSA tiling of {layout} is not a permutation of its rows")
    row_of_slot = torch.full((padded * VSA_TILE_ELEMS,), -1, dtype=torch.long)
    row_of_slot[slot_of_row] = torch.arange(layout.rows)
    logger.info("VSA layout %s: %d rows, %d dense + %d video tiles", layout.parts, layout.rows, num_prefix, num_video)
    return VsaGeometry(num_prefix, num_video, block_sizes.to(device), slot_of_row.to(device), tile_of_row.to(device),
                       row_of_slot.to(device))


def _video_tiles(layout: VideoSparseLayout):
    (grid,) = [p for p in layout.parts if isinstance(p, tuple)]
    for t in _axis_tiles(grid[0], VSA_TILE[0]):
        for h in _axis_tiles(grid[1], VSA_TILE[1]):
            for w in _axis_tiles(grid[2], VSA_TILE[2]):
                yield t, h, w


def vsa_keep(sparsity: float, num_video_tiles: int) -> int:
    """Key tiles each video query tile keeps of the video's (FastVideo's ``compute_topk``)."""
    return max(1, min(math.ceil((1.0 - sparsity) * num_video_tiles), num_video_tiles))


def _block_sparse_kernel():
    try:
        from fastvideo_kernel.block_sparse_attn_ptx_blackwell import (
            _HAS_VSA_PTX_BLACKWELL,
            block_sparse_attn_ptx_blackwell,
        )
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "ragged attention backend 'vsa' runs FastVideo's block-sparse kernel: install fastvideo-kernel built "
            "for this GPU (data-center Blackwell, sm_100a / sm_103a; see docs/models.rst, MiniMax-H3)"
        ) from e
    if not _HAS_VSA_PTX_BLACKWELL:
        raise ImportError("fastvideo-kernel was built without its ptx_blackwell VSA forward (TORCH_CUDA_ARCH_LIST "
                          "must name 10.0a or 10.3a)")
    return block_sparse_attn_ptx_blackwell


@triton.jit
def _pack_kernel(src_ptr, row_of_slot_ptr, tiled_ptr, pooled_ptr, inv_size_ptr, heads, slots, n_tiles,
                 TILE: tl.constexpr, D: tl.constexpr):
    """One (tile, head): gather the tile's packed ``[rows, H, D]`` rows into head-major ``tiled``
    ``[H, slots, D]`` (zeros in pad slots), and their fp32 mean into ``pooled`` ``[H, n_tiles, D]``."""
    tile, h = tl.program_id(0), tl.program_id(1)
    slot = tile * TILE + tl.arange(0, TILE)
    cols = tl.arange(0, D)
    row = tl.load(row_of_slot_ptr + slot)
    x = tl.load(src_ptr + (row.to(tl.int64)[:, None] * heads + h) * D + cols[None, :],
                mask=(row >= 0)[:, None], other=0.0)
    tl.store(tiled_ptr + (h.to(tl.int64) * slots + slot[:, None]) * D + cols[None, :], x)
    if tile < n_tiles:
        mean = tl.sum(x.to(tl.float32), axis=0) * tl.load(inv_size_ptr + tile)
        tl.store(pooled_ptr + (h.to(tl.int64) * n_tiles + tile) * D + cols, mean)


@triton.jit
def _untile_kernel(out_ptr, slot_of_row_ptr, tile_of_row_ptr, coarse_ptr, gate_ptr, res_ptr, rows, heads, slots,
                   n_tiles, HAS_GATE: tl.constexpr, BLOCK: tl.constexpr, D: tl.constexpr):
    """A block of packed rows for one head: the kernel's head-major output at each row's slot,
    plus the gate times the row's tile's coarse output (``[H, n_tiles, D]``, the output dtype)."""
    block, h = tl.program_id(0), tl.program_id(1)
    row = block * BLOCK + tl.arange(0, BLOCK)
    live = row < rows
    cols = tl.arange(0, D)
    slot = tl.load(slot_of_row_ptr + row, mask=live, other=0)
    o = tl.load(out_ptr + (h.to(tl.int64) * slots + slot[:, None]) * D + cols[None, :], mask=live[:, None])
    dst = (row.to(tl.int64)[:, None] * heads + h) * D + cols[None, :]
    if HAS_GATE:
        tile = tl.load(tile_of_row_ptr + row, mask=live, other=0)
        c = tl.load(coarse_ptr + (h.to(tl.int64) * n_tiles + tile[:, None]) * D + cols[None, :], mask=live[:, None])
        g = tl.load(gate_ptr + dst, mask=live[:, None])
        o = (o.to(tl.float32) + c.to(tl.float32) * g.to(tl.float32)).to(o.dtype)
    tl.store(res_ptr + dst, o, mask=live[:, None])


@triton.jit
def _compact_kernel(mask_ptr, index_ptr, num_video, prefix, mask_h, mask_r, index_h, index_r, VIDEO: tl.constexpr):
    """One (video query tile, head): the video key tiles its row of ``mask`` picks, ascending, after the ``prefix``
    dense ones in its ``index`` row; the row of ``mask`` is left zero for the next layer."""
    r, h = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, VIDEO)
    live = cols < num_video
    row = mask_ptr + h.to(tl.int64) * mask_h + r.to(tl.int64) * mask_r
    picked = tl.load(row + cols, mask=live, other=0).to(tl.int32)
    slot = tl.cumsum(picked, axis=0) - 1
    out = index_ptr + h.to(tl.int64) * index_h + (r + prefix).to(tl.int64) * index_r
    tl.store(out + prefix + slot, cols + prefix, mask=live & (picked != 0))
    tl.store(row + cols, tl.zeros_like(picked).to(tl.int8), mask=live)


class VsaWorkspace:
    """Per-geometry scratch for one head count: the tiled q / k / v, and the kernel's index
    lists, whose dense rows (and video rows' dense columns) never change; a layer writes only
    the video rows' selected tiles."""

    def __init__(self, geometry: VsaGeometry, heads: int, keep: int, dtype: torch.dtype, device: torch.device):
        n, n_pad, prefix = geometry.num_tiles, geometry.tiles_padded, geometry.num_prefix_tiles
        self.tiled = torch.empty(3, heads, n_pad * VSA_TILE_ELEMS, VSA_HEAD_DIM, dtype=dtype, device=device)
        self.pooled = torch.empty(3, heads, n, VSA_HEAD_DIM, dtype=torch.float32, device=device)
        self.inv_size = 1.0 / geometry.block_sizes[:n].float()
        # the video query tiles' picks as a 0/1 map, zero between layers (``_compact_kernel``)
        v = geometry.num_video_tiles
        self.picked = torch.zeros(heads, v, v, dtype=torch.int8, device=device)
        self.index = torch.zeros(heads, n_pad, n, dtype=torch.int32, device=device)
        self.index[:, :prefix] = torch.arange(n, dtype=torch.int32, device=device)
        self.index[:, prefix:n, :prefix] = torch.arange(prefix, dtype=torch.int32, device=device)
        self.count = torch.zeros(heads, n_pad, dtype=torch.int32, device=device)
        self.count[:, :prefix] = n
        self.count[:, prefix:n] = prefix + keep
        if keep == geometry.num_video_tiles:
            self.index[:, prefix:n] = torch.arange(n, dtype=torch.int32, device=device)


def video_sparse_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor | None, geometry: VsaGeometry,
    workspace: VsaWorkspace, keep: int, kernel=None, out: torch.Tensor | None = None,
) -> torch.Tensor:
    """VSA over one segment's packed ``[rows, H, D]`` q / k / v (bf16), with the compression
    branch through ``gate`` (``[rows, H, D]``) when given. Returns ``[rows, H, D]``, written into
    ``out`` (contiguous) when given."""
    kernel = kernel or _block_sparse_kernel()
    heads, d = q.shape[1], q.shape[2]
    n, prefix = geometry.num_tiles, geometry.num_prefix_tiles
    tiled, pooled = workspace.tiled, workspace.pooled
    slots = tiled.shape[2]
    if q.is_cuda:
        for i, x in enumerate((q, k, v)):
            _pack_kernel[(geometry.tiles_padded, heads)](
                x.contiguous(), geometry.row_of_slot, tiled[i], pooled[i], workspace.inv_size, heads, slots, n,
                TILE=VSA_TILE_ELEMS, D=d,
            )
    else:  # the same in torch, for CPU tests
        live = geometry.row_of_slot >= 0
        for i, x in enumerate((q, k, v)):
            staged = x.new_zeros(slots, heads, d)
            staged[live] = x[geometry.row_of_slot[live]]
            tiled[i] = staged.transpose(0, 1)
            pooled[i] = (staged.unflatten(0, (-1, VSA_TILE_ELEMS))[:n].sum(1, dtype=torch.float32)
                         * workspace.inv_size[:, None, None]).transpose(0, 1)
    scores = None
    if keep < geometry.num_video_tiles or gate is not None:
        scores = (pooled[0] @ pooled[1].transpose(-2, -1)) * d ** -0.5
    if keep < geometry.num_video_tiles:
        picked = scores[:, prefix:, prefix:].topk(keep, dim=-1, sorted=False).indices
        # listed ascending, as FastVideo's ``map_to_index`` lists them (the kernel accumulates in list order)
        if q.is_cuda:
            mask, index = workspace.picked, workspace.index
            mask.scatter_(-1, picked, 1)
            num_video = geometry.num_video_tiles
            _compact_kernel[(num_video, heads)](
                mask, index, num_video, prefix, mask.stride(0), mask.stride(1), index.stride(0), index.stride(1),
                VIDEO=triton.next_power_of_2(num_video),
            )
        else:
            workspace.index[:, prefix:n, prefix:prefix + keep] = picked.sort(dim=-1).values.to(torch.int32) + prefix
    out_tiled, _ = kernel(
        tiled[0:1], tiled[1:2], tiled[2:3], workspace.index[None], workspace.count[None], geometry.block_sizes,
        need_lse=False,
    )
    dst, out = out, out_tiled[0]
    coarse = None
    if gate is not None:
        coarse = (torch.softmax(scores, dim=-1) @ pooled[2]).to(out.dtype)
    rows = q.shape[0]
    if q.is_cuda:
        res = torch.empty_like(q) if dst is None else dst
        block = 64
        _untile_kernel[(triton.cdiv(rows, block), heads)](
            out, geometry.slot_of_row, geometry.tile_of_row, out if coarse is None else coarse,
            res if gate is None else gate.contiguous(), res, rows, heads, slots, n,
            HAS_GATE=gate is not None, BLOCK=block, D=d,
        )
        return res
    res = out[:, geometry.slot_of_row].transpose(0, 1)
    if gate is not None:
        res = torch.addcmul(res, coarse[:, geometry.tile_of_row].transpose(0, 1), gate)
    if dst is None:
        return res.contiguous()
    dst.copy_(res)
    return dst


class VsaRaggedPrefillWrapper(RaggedPrefillWrapper):
    """Self-attention within each segment: VSA over a segment that carries a
    ``VideoSparseLayout`` (``plan_layouts``), dense (SDPA) over any other. ``run`` takes the
    compression branch's ``gate``, packed like ``q``. Eager only."""

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int, sparsity: float, kernel=None, **kwargs):
        if kwargs.get("accelerator_graph"):
            raise ValueError(f"{type(self).__name__}: VSA's index lists are data-dependent; it runs eagerly only")
        if num_qo_heads != num_kv_heads or head_dim != VSA_HEAD_DIM:
            raise ValueError(f"{type(self).__name__} runs MHA at head dim {VSA_HEAD_DIM}, got {num_qo_heads} q / "
                             f"{num_kv_heads} kv heads of {head_dim}")
        super().__init__(num_qo_heads, num_kv_heads, head_dim, **kwargs)
        if not math.isclose(self.sm_scale, head_dim ** -0.5):
            raise ValueError(f"{type(self).__name__}: the VSA kernel scales by head_dim ** -0.5, not {self.sm_scale}")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"VSA sparsity must be in [0, 1), got {sparsity}")
        self.sparsity = float(sparsity)
        # the block-sparse kernel (``_block_sparse_kernel``), resolved on first use when not given
        self._kernel = kernel
        self._layouts: tuple[VideoSparseLayout | None, ...] = ()
        self._spans: list[tuple[int, int]] = []
        self._workspaces: dict[tuple, VsaWorkspace] = {}

    def plan_layouts(self, layouts: tuple[VideoSparseLayout | None, ...]) -> None:
        """Each segment's layout, in the order ``plan``'s ``cu_seqlens`` lists them; None for dense."""
        self._layouts = tuple(layouts)

    def _plan_kernel(self, cu: torch.Tensor, kv_cu: torch.Tensor, total_tokens: int, causal: bool) -> None:
        if causal:
            raise ValueError(f"{type(self).__name__}: VSA is bidirectional")
        bounds = cu.to(device="cpu", dtype=torch.int64).tolist()
        spans = list(zip(bounds[:-1], bounds[1:], strict=True))
        if len(spans) != len(self._layouts):
            raise ValueError(f"{type(self).__name__}: {len(spans)} segments, {len(self._layouts)} layouts")
        for (start, end), layout in zip(spans, self._layouts, strict=True):
            if layout is not None and layout.rows != end - start:
                raise ValueError(f"{type(self).__name__}: a segment of {end - start} rows carries a layout of "
                                 f"{layout.rows} ({layout})")
        self._spans = spans

    def _layout_key(self, cu_seqlens, kv_cu_seqlens, causal):
        # the segments' layouts are part of what a plan means
        return RaggedPrefillWrapper._layout_key(cu_seqlens, kv_cu_seqlens, causal), self._layouts

    def _workspace(self, layout: VideoSparseLayout, geometry: VsaGeometry, keep: int, dtype) -> VsaWorkspace:
        key = (layout, keep, dtype)
        ws = self._workspaces.get(key)
        if ws is None:
            # one geometry's scratch at a time: a new bucket's replaces the last's
            self._workspaces.clear()
            ws = self._workspaces[key] = VsaWorkspace(geometry, self.num_qo_heads, keep, dtype, self.device)
        return ws

    @torch.compiler.disable
    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if return_lse:
            raise ValueError(f"{type(self).__name__} returns no log-sum-exp")
        out = q.new_empty(q.shape[0], self.num_qo_heads, self.head_dim, dtype=self.q_data_type)
        if self._spans and self._spans[-1][1] < q.shape[0]:
            # rows past the layout read back zero, as on the other backends
            out[self._spans[-1][1]:].zero_()
        for (start, end), layout in zip(self._spans, self._layouts, strict=True):
            if end == start:
                continue
            qs, ks, vs = (t[start:end].to(self.q_data_type) for t in (q, k, v))
            if layout is None:
                out[start:end] = F.scaled_dot_product_attention(
                    *(t.transpose(0, 1)[None] for t in (qs, ks, vs)), scale=self.sm_scale,
                )[0].transpose(0, 1)
                continue
            geometry = vsa_geometry(layout, self.device)
            keep = vsa_keep(self.sparsity, geometry.num_video_tiles)
            ws = self._workspace(layout, geometry, keep, self.q_data_type)
            g = None if gate is None else gate[start:end].to(self.q_data_type)
            if self._kernel is None:
                self._kernel = _block_sparse_kernel()
            video_sparse_attention(qs, ks, vs, g, geometry, ws, keep, self._kernel, out=out[start:end])
        return out


def _vsa_manager_class():
    """``VsaRaggedManager``, built on first use: FlashInfer's manager imports ``base``, which
    imports this module's build path."""
    from mstar.engine.resources.attn.ragged.flashinfer import FlashInferRaggedManager

    class VsaRaggedManager(FlashInferRaggedManager):
        """The ``vsa`` backend: a label whose segments carry a ``VideoSparseLayout`` runs VSA
        (``VsaRaggedPrefillWrapper``); every other label runs as on ``auto`` (the kernel table, or
        FlashInfer). Runs unsplit and uncaptured: it refuses a split step and a captured slot."""

        # the model may add a ``gate`` for VSA's compression branch beside k and v
        takes_gate = True

        def __init__(self, *args, sparsity: float = DEFAULT_VSA_SPARSITY, **kwargs):
            super().__init__(*args, **kwargs)
            # at build, so a missing kernel fails the boot, not the first request
            self._kernel = _block_sparse_kernel()
            self._sparsity = float(sparsity)
            self._vsa_wrappers: dict[tuple[str, int], VsaRaggedPrefillWrapper] = {}

        def _plan_layouts(self, step, ctx, plan_states, shard_plan) -> None:
            import dataclasses

            sparse, dense = {}, []
            for label, segments in self._group_segments_by_label(step.segments or ()).items():
                layouts = tuple(getattr(seg, "layout", None) for seg in segments)
                if any(isinstance(layout, VideoSparseLayout) for layout in layouts):
                    sparse[label] = (segments, layouts)
                else:
                    dense.extend(segments)
            if sparse and ctx.slot_lease is not None:
                raise ValueError(f"ragged attention {self._key!r}: VSA runs eagerly, not in a captured graph")
            if sparse and shard_plan is not None and set(shard_plan.ranges) & set(sparse):
                raise ValueError(f"ragged attention {self._key!r}: VSA label(s) {sorted(sparse)} split over SP ranks; "
                                 "VSA runs on one rank's whole sequence")
            super()._plan_layouts(dataclasses.replace(step, segments=tuple(dense)), ctx, plan_states, shard_plan)
            for label, (segments, layouts) in sparse.items():
                key = (label, ctx.slot)
                wrapper = self._vsa_wrappers.get(key)
                if wrapper is None:
                    wrapper = self._vsa_wrappers[key] = VsaRaggedPrefillWrapper(
                        **self._kwargs, sparsity=self._sparsity, kernel=self._kernel,
                    )
                wrapper.plan_layouts(layouts)
                wrapper.plan(cu_seqlens=self._cu_seqlens(segments), causal=step.causal)
                plan_states[label] = wrapper

        def run(self, q, k, v, label=None, return_lse=False, gate=None):
            if gate is None:
                return super().run(q, k, v, label, return_lse)
            return self._run_gated(q, k, v, label, gate)

        @torch.compiler.disable
        def _run_gated(self, q, k, v, label, gate):
            wrapper = self._wrapper(label)
            if not isinstance(wrapper, VsaRaggedPrefillWrapper):
                raise ValueError(f"ragged attention {self._key!r}: a gate for label {label!r}, whose segments carry "
                                 "no VideoSparseLayout")
            return wrapper.run(q, k, v, gate=gate)

    return VsaRaggedManager
