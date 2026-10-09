"""dsa_paged's CUDA selection (score kernel, flashinfer top-k -> slots) against its CPU path on
the same data, at GLM-5.2's indexer dims. Sparse MLA itself: test_sparse_mla_gpu.py."""
import pytest
import torch

from mstar.model.glm52 import dsa_paged
from mstar.model.glm52.dsa_paged import Glm52DsaPagedContext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA paths")

NH, D, PAGE, TOPK = 32, 128, 64, 256
DEV = torch.device("cuda")


def _ctx(lens, rows_per_req, gen, device):
    """Two requests; rows are the last ``rows_per_req[i]`` positions of request i."""
    pages = [-(-n // PAGE) for n in lens]
    total = sum(pages) + 4
    perm = torch.randperm(total, generator=gen).tolist()
    perm2 = torch.randperm(total, generator=gen).tolist()
    width = max(pages)
    kv, ix, row_req, row_lens, spans = [], [], [], [], []
    for i, (n, r) in enumerate(zip(lens, rows_per_req, strict=True)):
        lo = sum(pages[:i])
        kv.append(perm[lo: lo + pages[i]] + [0] * (width - pages[i]))
        ix.append(perm2[lo: lo + pages[i]] + [0] * (width - pages[i]))
        spans.append((len(row_req), r, i))
        row_req += [i] * r
        row_lens += list(range(n - r + 1, n + 1))
    return total, Glm52DsaPagedContext(
        row_req=torch.tensor(row_req, dtype=torch.int32, device=device),
        lens=torch.tensor(row_lens, dtype=torch.int32, device=device), host_lens=row_lens,
        kv_table=torch.tensor(kv, dtype=torch.int32, device=device),
        index_table=torch.tensor(ix, dtype=torch.int32, device=device),
        spans=spans, width=max(row_lens), page_size=PAGE, topk=TOPK, needs_selection=True)


def _to(ctx, device):
    return Glm52DsaPagedContext(
        row_req=ctx.row_req.to(device), lens=ctx.lens.to(device), host_lens=ctx.host_lens,
        kv_table=ctx.kv_table.to(device), index_table=ctx.index_table.to(device),
        spans=ctx.spans, width=ctx.width, page_size=ctx.page_size, topk=ctx.topk,
        needs_selection=True)


@pytest.mark.parametrize("rows_per_req", [(1, 1), (3, 5), (1, 700)])  # decode, prefill, mixed
def test_cuda_selection_matches_the_cpu_path(rows_per_req, monkeypatch):
    monkeypatch.setattr(dsa_paged, "PREFILL_CHUNK_ROWS", 256)  # several chunks for 700 rows
    gen = torch.Generator().manual_seed(0)
    total, ctx = _ctx([700, 1500], list(rows_per_req), gen, DEV)
    index = torch.randn(total, PAGE, D, generator=gen).bfloat16()
    rows = ctx.row_req.numel()
    q = torch.randn(rows, NH, D, generator=gen).bfloat16()
    w = torch.randn(rows, NH, generator=gen) * (D ** -0.5 * NH ** -0.5)
    got = dsa_paged.select(q.to(DEV), w.to(DEV), index.to(DEV), ctx).cpu()
    ref = dsa_paged.select(q, w, index, _to(ctx, "cpu"))
    same = sum(len(set(got[r, :n].tolist()) & set(ref[r, :n].tolist()))
               for r, n in enumerate(ctx.attn_lens))
    # fp32 summation order can swap two near-equal scores at the selection boundary
    assert same >= 0.999 * sum(ctx.attn_lens)


def test_a_row_with_fewer_keys_than_topk_lists_them_first():
    """Captured decode sends every row through the sparse path, rows short of topk too;
    attention reads each row's first min(lens, topk) slots, which must be all its keys."""
    gen = torch.Generator().manual_seed(2)
    total, ctx = _ctx([40, 300], [1, 1], gen, DEV)  # 40 < TOPK = 256 < 300
    rows = ctx.row_req.numel()
    width = 512
    scores = torch.randn(rows, width, generator=gen)
    scores = scores.masked_fill(torch.arange(width) >= ctx.lens.cpu()[:, None], float("-inf"))
    got = dsa_paged.select_slots(scores.to(DEV), ctx.kv_table, ctx.row_req, ctx.lens, PAGE,
                                 TOPK).cpu()
    cpu = _to(ctx, "cpu")
    ref = dsa_paged.select_slots(scores, cpu.kv_table, cpu.row_req, cpu.lens, PAGE, TOPK)
    for r, n in enumerate(ctx.attn_lens):
        assert set(got[r, :n].tolist()) == set(ref[r, :n].tolist()), r


@pytest.mark.parametrize("world", [2, 3, 8])
def test_sharded_selection_is_bitwise_the_unsharded_one(world, monkeypatch):
    """Ranks' blocks of a long prefill span start their score chunks at other rows than one
    rank selecting the whole span; every row's selection must still be the same."""
    monkeypatch.setattr(dsa_paged, "PREFILL_CHUNK_ROWS", 256)
    gen = torch.Generator().manual_seed(4)
    total, ctx = _ctx([700, 1500], [1, 700], gen, DEV)  # a decode row and a sharded span
    index = torch.randn(total, PAGE, D, generator=gen).bfloat16().to(DEV)
    rows = ctx.row_req.numel()
    q = torch.randn(rows, NH, D, generator=gen).bfloat16().to(DEV)
    w = (torch.randn(rows, NH, generator=gen) * (D ** -0.5 * NH ** -0.5)).to(DEV)
    want = dsa_paged.select(q, w, index, ctx)
    blocks = {}

    class Group:  # one rank per select call; a rank not yet run gathers -7s
        world_size, rank = world, 0

        def all_gather(self, t, dim=0):
            blocks[self.rank] = t
            return torch.cat([blocks.get(r, torch.full_like(t, -7)) for r in range(world)])

    group = Group()
    for group.rank in range(world):
        got = dsa_paged.select(q, w, index, ctx, group)
    assert torch.equal(got, want)
