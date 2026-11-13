"""The ``vsa`` ragged backend's tiling and math (``attn.ragged.vsa``): the tile order FastVideo's
VSA kernels and FastH3 were trained on, the exempt top-k, the compression gate, and the plan's
checks. CPU except the kernel test, which needs a data-center Blackwell GPU and fastvideo-kernel."""

from __future__ import annotations

import math

import pytest
import torch

from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend, VideoSparseLayout
from mstar.engine.resources.attn.ragged.vsa import (
    VSA_TILE_ELEMS,
    VsaRaggedPrefillWrapper,
    VsaWorkspace,
    video_sparse_attention,
    vsa_geometry,
    vsa_keep,
    vsa_tile_order,
)
from mstar.engine.resources.step import Segment

CPU = torch.device("cpu")


def _fastvideo_tile_rows(grid):
    """FastVideo's ``get_tile_partition_indices`` (fastvideo_kernel.vsa_utils), restated."""
    t, h, w = grid
    idx = torch.arange(t * h * w).reshape(t, h, w)
    return [idx[a:a + 4, b:b + 4, c:c + 4].flatten()
            for a in range(0, t, 4) for b in range(0, h, 4) for c in range(0, w, 4)]


def test_layout_rows_and_refusals():
    layout = VideoSparseLayout((70, 0, 45, (5, 6, 10)))
    assert layout.rows == 70 + 45 + 300
    for bad in ((70, 45), (70, (5, 6, 10), (1, 1, 1)), ((0, 4, 4),), (-1, (1, 1, 1))):
        with pytest.raises(ValueError):
            VideoSparseLayout(bad)
    # a segment carries its layout; one without stays a 3-field segment as before
    assert Segment("r", "rows", 415, layout).layout == layout and Segment("r", "rows", 8).layout is None


def test_tile_order_dense_runs_first_then_fastvideos_video_tiles():
    layout = VideoSparseLayout((70, 0, 45, (5, 6, 10)))
    sizes, tiles = vsa_tile_order(layout)
    # dense runs chunk by 64 within each run, never across one: 70 -> 64 + 6, 45 -> 45
    assert sizes[:3] == [64, 6, 45]
    assert torch.equal(tiles[1], torch.arange(64, 70)) and torch.equal(tiles[2], torch.arange(70, 115))
    video = _fastvideo_tile_rows((5, 6, 10))
    assert len(tiles) == 3 + len(video) == 3 + 2 * 2 * 3
    for mine, theirs in zip(tiles[3:], video, strict=True):
        assert torch.equal(mine, theirs + 115)
    assert sizes[3:] == [len(rows) for rows in video]


def test_geometry_is_a_permutation_padded_to_even_tiles():
    layout = VideoSparseLayout((70, 0, 45, (5, 6, 10)))
    geo = vsa_geometry(layout, CPU)
    assert (geo.num_prefix_tiles, geo.num_video_tiles) == (3, 12)
    assert geo.tiles_padded == 16 and int(geo.block_sizes[-1]) == 0
    assert int(geo.block_sizes.sum()) == layout.rows
    slots = geo.slot_of_row
    assert torch.unique(slots).numel() == layout.rows
    assert torch.equal(slots // VSA_TILE_ELEMS, geo.tile_of_row)
    assert bool((slots % VSA_TILE_ELEMS < geo.block_sizes[geo.tile_of_row]).all())


def test_keep_is_fastvideos_compute_topk():
    assert vsa_keep(0.8, 528) == math.ceil(0.2 * 528) == 106
    assert vsa_keep(0.8, 3) == 1 and vsa_keep(0.0, 12) == 12


def test_workspace_lists_dense_rows_and_dense_columns():
    geo = vsa_geometry(VideoSparseLayout((70, 0, 45, (5, 6, 10))), CPU)
    ws = VsaWorkspace(geo, 2, keep=3, dtype=torch.float32, device=CPU)
    n, p = geo.num_tiles, geo.num_prefix_tiles
    assert torch.equal(ws.count[:, :p], torch.full((2, p), n, dtype=torch.int32))
    assert torch.equal(ws.count[:, p:n], torch.full((2, n - p), p + 3, dtype=torch.int32))
    assert int(ws.count[:, n:].sum()) == 0
    assert torch.equal(ws.index[:, p:n, :p], torch.arange(p, dtype=torch.int32).expand(2, n - p, p))


def _masked_kernel(q, k, v, index, count, block_sizes, need_lse=False):
    """The block-sparse kernel's contract, densely: each query tile attends its listed key tiles'
    valid tokens."""
    _, heads, s, d = q.shape
    n_pad = block_sizes.numel()
    valid = (torch.arange(VSA_TILE_ELEMS)[None] < block_sizes[:, None]).flatten()
    allowed = torch.zeros(heads, n_pad, n_pad, dtype=torch.bool)
    for h in range(heads):
        for i in range(n_pad):
            allowed[h, i, index[0, h, i, :count[0, h, i]].long()] = True
    tok = allowed.repeat_interleave(VSA_TILE_ELEMS, 1).repeat_interleave(VSA_TILE_ELEMS, 2) & valid[None, None]
    scores = (q[0].float() @ k[0].float().transpose(-2, -1)) * d ** -0.5
    probs = scores.masked_fill(~tok, float("-inf")).softmax(-1).nan_to_num()
    return (probs @ v[0].float()).to(q.dtype)[None], None


def _reference(q, k, v, gate, layout, keep):
    """VSA in tile order from FastVideo's MiniMax-H3 recipe, independent of the module's geometry."""
    sizes, tiles = vsa_tile_order(layout)
    n = len(sizes)
    p = n - sum(1 for t in tiles if t.min() >= 115)
    order, tile_of = torch.cat(tiles), torch.cat([torch.full((len(t),), i) for i, t in enumerate(tiles)])
    qf, kf, vf = (x.float()[order] for x in (q, k, v))
    count = torch.tensor(sizes, dtype=torch.float32)[:, None, None]
    pool = lambda x: torch.zeros(n, *x.shape[1:]).index_add_(0, tile_of, x) / count  # noqa: E731
    qp, kp, vp = pool(qf), pool(kf), pool(vf)
    scores = torch.einsum("ihd,jhd->hij", qp, kp) * q.shape[-1] ** -0.5
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[:, p:].scatter_(-1, scores[:, p:, p:].topk(keep, -1).indices + p, True)
    mask[:, :, :p] = True
    mask[:, :p] = True
    att = torch.einsum("ihd,jhd->hij", qf, kf) * q.shape[-1] ** -0.5
    att = att.masked_fill(~mask[:, tile_of][:, :, tile_of], float("-inf")).softmax(-1)
    out = torch.einsum("hij,jhd->ihd", att, vf)
    coarse = torch.einsum("hij,jhd->ihd", scores.softmax(-1), vp)
    out = out + gate.float()[order] * coarse[tile_of]
    res = torch.empty_like(out)
    res[order] = out
    return res


@pytest.mark.parametrize("gated", [True, False])
def test_video_sparse_attention_matches_the_recipe(gated):
    torch.manual_seed(0)
    layout = VideoSparseLayout((70, 0, 45, (5, 6, 10)))
    geo = vsa_geometry(layout, CPU)
    keep = vsa_keep(0.8, geo.num_video_tiles)
    q, k, v, g = (torch.randn(layout.rows, 2, 128) for _ in range(4))
    ws = VsaWorkspace(geo, 2, keep, torch.float32, CPU)
    out = video_sparse_attention(q, k, v, g if gated else None, geo, ws, keep, kernel=_masked_kernel)
    ref = _reference(q, k, v, g if gated else torch.zeros_like(g), layout, keep)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_wrapper_plan_checks_layouts_against_segments():
    w = VsaRaggedPrefillWrapper(2, 2, 128, sparsity=0.8, device=CPU, q_data_type=torch.float32)
    layout = VideoSparseLayout((70, 0, 45, (5, 6, 10)))
    w.plan_layouts((layout, None))
    w.plan(torch.tensor([0, 415, 427], dtype=torch.int32))
    w.plan_layouts((layout,))
    with pytest.raises(ValueError, match="2 segments, 1 layouts"):
        w.plan(torch.tensor([0, 415, 427], dtype=torch.int32))
    w.plan_layouts((layout,))
    with pytest.raises(ValueError, match="carries a layout"):
        w.plan(torch.tensor([0, 400], dtype=torch.int32))
    with pytest.raises(ValueError, match="eagerly"):
        VsaRaggedPrefillWrapper(2, 2, 128, sparsity=0.8, device=CPU, accelerator_graph=True,
                                max_num_segments=2, max_total_tokens=512)
    with pytest.raises(ValueError, match="sparsity"):
        VsaRaggedPrefillWrapper(2, 2, 128, sparsity=1.0, device=CPU)


def test_backend_is_named_vsa():
    assert RaggedAttnBackend("vsa") is RaggedAttnBackend.VSA


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
                    reason="FastVideo's block-sparse kernel runs on data-center Blackwell")
def test_kernel_matches_the_recipe_on_gpu():
    pytest.importorskip("fastvideo_kernel")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    layout = VideoSparseLayout((70, 0, 45, (5, 6, 10)))
    geo = vsa_geometry(layout, dev)
    keep = vsa_keep(0.8, geo.num_video_tiles)
    q, k, v, g = (torch.randn(layout.rows, 4, 128, device=dev, dtype=torch.bfloat16) for _ in range(4))
    out = video_sparse_attention(q, k, v, g, geo, VsaWorkspace(geo, 4, keep, torch.bfloat16, dev), keep)
    ref = _reference(q.cpu(), k.cpu(), v.cpu(), g.cpu(), layout, keep)
    assert float((out.float().cpu() - ref).norm() / ref.norm()) < 1e-2
