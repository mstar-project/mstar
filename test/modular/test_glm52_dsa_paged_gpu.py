"""dsa/paged.py's CUDA selection (score kernel, flashinfer top-k -> slots) against its CPU path on
the same data, at GLM-5.2's indexer dims (a key per token). Sparse MLA itself:
test_sparse_mla_gpu.py."""
import dataclasses

import pytest
import torch

from mstar.engine.resources.dsa import paged
from mstar.engine.resources.dsa.paged import DSAContext
from mstar.utils.h2d import PinnedStager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA paths")

NH, D, PAGE, TOPK = 32, 128, 64, 256
DEV = torch.device("cuda")
# several score chunks for 700 rows: 256 rows of the longest request's 1500 keys' scores
SCORE_BYTES = 4 * 1500 * 256


def _ctx(lens, rows_per_req, gen, device):
    """Two requests on shuffled pages, the index keys on the latents' pages; rows are the last
    ``rows_per_req[i]`` positions of request i."""
    pages = [-(-n // PAGE) for n in lens]
    total = sum(pages) + 4
    perm = torch.randperm(total, generator=gen).tolist()
    starts = [sum(pages[:i]) for i in range(len(lens))]
    row_req, positions, spans = [], [], []
    for i, (n, r) in enumerate(zip(lens, rows_per_req, strict=True)):
        spans.append((len(row_req), r, i))
        row_req += [i] * r
        positions += list(range(n - r, n))
    return total, DSAContext(
        pos=torch.tensor(positions, dtype=torch.int32, device=device),
        row_req=torch.tensor(row_req, dtype=torch.int32, device=device),
        pages=torch.tensor(perm[:sum(pages)], dtype=torch.int32, device=device),
        page_start=torch.tensor(starts, dtype=torch.int32, device=device),
        host_pos=positions, spans=spans, host_page_start=starts,
        max_pools=max(positions) + 1, page_size=PAGE, topk=TOPK, kpool=1)


def _to(ctx, device):
    return dataclasses.replace(
        ctx, pos=ctx.pos.to(device), row_req=ctx.row_req.to(device),
        pages=ctx.pages.to(device), page_start=ctx.page_start.to(device))


@pytest.mark.parametrize("rows_per_req", [(1, 1), (3, 5), (1, 700)])  # decode, prefill, mixed
def test_cuda_selection_matches_the_cpu_path(rows_per_req):
    gen = torch.Generator().manual_seed(0)
    total, ctx = _ctx([700, 1500], list(rows_per_req), gen, DEV)
    index = torch.randn(total, PAGE, D, generator=gen).bfloat16()
    rows = ctx.row_req.numel()
    q = torch.randn(rows, NH, D, generator=gen).bfloat16()
    w = torch.randn(rows, NH, generator=gen) * (D ** -0.5 * NH ** -0.5)
    got = paged.select(q.to(DEV), w.to(DEV), index.to(DEV), ctx, score_bytes=SCORE_BYTES).cpu()
    ref = paged.select(q, w, index, _to(ctx, "cpu"), score_bytes=SCORE_BYTES)
    same = sum(len(set(got[r, :n].tolist()) & set(ref[r, :n].tolist()))
               for r, n in enumerate(ctx.attn_lens()))
    # fp32 summation order can swap two near-equal scores at the selection boundary
    assert same >= 0.999 * sum(ctx.attn_lens())


def test_one_row_spans_copy_nothing_from_pageable_memory():
    """An eager step's one-row spans (a mixed step's decode rows, ``pick``'s rows) reach the
    device through the resource's pinned stager. A pageable host-to-device copy would wait for
    every kernel queued before it, once per indexer layer."""
    gen = torch.Generator().manual_seed(1)
    total, ctx = _ctx([700, 1500], [1, 700], gen, DEV)
    ctx = dataclasses.replace(ctx, stager=PinnedStager(torch.int32))
    index = torch.randn(total, PAGE, D, generator=gen).bfloat16().to(DEV)
    rows = ctx.row_req.numel()
    q = torch.randn(rows, NH, D, generator=gen).bfloat16().to(DEV)
    w = (torch.randn(rows, NH, generator=gen) * (D ** -0.5 * NH ** -0.5)).to(DEV)
    ends = [0, rows - 1]
    q_ends, w_ends = q[ends], w[ends]

    def step():
        mixed = paged.select(q, w, index, ctx, score_bytes=SCORE_BYTES)
        return mixed, paged.select(q_ends, w_ends, index, ctx.pick(ends))

    want = step()  # warm: kernel builds, the stager's arena
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        got = step()
        torch.cuda.synchronize()
    copies = [e.name for e in prof.events() if "Memcpy HtoD" in e.name]
    assert copies and not [c for c in copies if "Pageable" in c], copies
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))


def test_a_row_with_fewer_keys_than_topk_lists_them_first():
    """Captured decode sends every row through the sparse path, rows short of topk too;
    attention reads each row's first min(lens, topk) slots, which must be all its keys."""
    gen = torch.Generator().manual_seed(2)
    total, ctx = _ctx([40, 300], [1, 1], gen, DEV)  # 40 < TOPK = 256 < 300
    rows = ctx.row_req.numel()
    width = 512
    scores = torch.randn(rows, width, generator=gen)
    lens = ctx.pos.cpu()[:, None] + 1
    scores = scores.masked_fill(torch.arange(width) >= lens, float("-inf"))
    got = paged.top_pools(scores.to(DEV), ctx).cpu()
    ref = paged.top_pools(scores, _to(ctx, "cpu"))
    for r, n in enumerate(ctx.attn_lens()):
        assert set(got[r, :n].tolist()) == set(ref[r, :n].tolist()), r


@pytest.mark.parametrize("world", [3, 8])
def test_sharded_selection_is_bitwise_the_unsharded_one(world):
    """Ranks' blocks of a long prefill span start their score chunks at other rows than one
    rank selecting the whole span; every row's selection must still be the same."""
    gen = torch.Generator().manual_seed(4)
    total, ctx = _ctx([700, 1500], [1, 700], gen, DEV)  # a decode row and a sharded span
    index = torch.randn(total, PAGE, D, generator=gen).bfloat16().to(DEV)
    rows = ctx.row_req.numel()
    q = torch.randn(rows, NH, D, generator=gen).bfloat16().to(DEV)
    w = (torch.randn(rows, NH, generator=gen) * (D ** -0.5 * NH ** -0.5)).to(DEV)
    want = paged.select(q, w, index, ctx, score_bytes=SCORE_BYTES)
    blocks = {}

    class Group:  # one rank per select call; a rank not yet run gathers -7s
        world_size, rank = world, 0

        def all_gather(self, t, dim=0):
            blocks[self.rank] = t
            return torch.cat([blocks.get(r, torch.full_like(t, -7)) for r in range(world)])

    group = Group()
    for group.rank in range(world):
        got = paged.select(q, w, index, ctx, group, score_bytes=SCORE_BYTES)
    assert torch.equal(got, want)
