"""dsa_kernels vs the indexer's fp32 score math (components/indexer.py) on a paged key store."""
import pytest
import torch

from mstar.model.glm52.components.indexer import select_topk_causal

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")

NH, D, PAGE, TOPK = 32, 128, 128, 256
DEV = torch.device("cuda")


def _store(num_pages, gen):
    return torch.randn(num_pages, PAGE, D, device=DEV, generator=gen).bfloat16()


def _reference(q, w, keys, bound):
    """compute_selection's scores: fp32 per-head relu dots, weighted head sum; -inf past
    ``bound`` (keys a row sees)."""
    dots = torch.einsum("hd,sd->hs", q.float(), keys.float()).relu()
    sc = w @ dots
    sc[bound:] = float("-inf")
    return sc


def _rows_agree(got, ref, bounds):
    """Same finite scores (summation order aside), same -inf tail, same top-k selection.
    Columns past a row's bound are the top-k's to skip, so the kernels may leave them."""
    visible = torch.arange(got.shape[1], device=DEV) < torch.tensor(bounds, device=DEV)[:, None]
    got = got.masked_fill(~visible, float("-inf"))
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(got), fin)
    torch.testing.assert_close(got[fin], ref[fin], rtol=1e-5, atol=1e-5)
    pos = torch.tensor(bounds, device=DEV) - 1
    a, b = select_topk_causal(got, pos, TOPK), select_topk_causal(ref, pos, TOPK)
    # fp32 summation order can swap two scores within ~1e-7 at the selection boundary
    same = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b, strict=True))
    assert same >= 0.999 * a.numel()


def test_decode_scores_follow_each_rows_request_and_bound():
    from mstar.model.glm52.dsa_kernels import decode_scores

    gen = torch.Generator(device="cuda").manual_seed(0)
    lens = [300, 1000, 2500]
    max_pages = -(-max(lens) // PAGE)
    k_pages = _store(len(lens) * max_pages + 3, gen)
    table = torch.randperm(k_pages.shape[0], device=DEV, generator=gen)[: len(lens) * max_pages]
    table = table.view(len(lens), max_pages).to(torch.int32)
    q = torch.randn(len(lens), NH, D, device=DEV, generator=gen).bfloat16()
    w = torch.randn(len(lens), NH, device=DEV, generator=gen) * (D ** -0.5 * NH ** -0.5)
    row_req = torch.tensor([2, 0, 1], dtype=torch.int32, device=DEV)  # rows out of request order
    bounds = [lens[int(i)] for i in row_req]
    max_len = max_pages * PAGE
    got = decode_scores(q, w, k_pages, table, row_req,
                        torch.tensor(bounds, dtype=torch.int32, device=DEV), max_len)
    keys = k_pages[table.long()].view(len(lens), max_len, D)
    ref = torch.stack([_reference(q[r], w[r], keys[int(row_req[r])], bounds[r])
                       for r in range(len(lens))])
    _rows_agree(got, ref, bounds)


def test_decode_scores_stride_over_more_blocks_than_programs():
    """64 rows leave a few programs per row, each striding over many key blocks."""
    from mstar.model.glm52.dsa_kernels import decode_scores

    gen = torch.Generator(device="cuda").manual_seed(2)
    rows, max_pages = 64, 40
    lens = torch.randint(TOPK, max_pages * PAGE, (rows,), device=DEV, generator=gen)
    k_pages = _store(rows * max_pages, gen)
    table = torch.randperm(k_pages.shape[0], device=DEV, generator=gen)
    table = table.view(rows, max_pages).to(torch.int32)
    q = torch.randn(rows, NH, D, device=DEV, generator=gen).bfloat16()
    w = torch.randn(rows, NH, device=DEV, generator=gen) * (D ** -0.5 * NH ** -0.5)
    row_req = torch.arange(rows, dtype=torch.int32, device=DEV)
    max_len = max_pages * PAGE
    got = decode_scores(q, w, k_pages, table, row_req, lens.to(torch.int32), max_len)
    keys = k_pages[table.long()].view(rows, max_len, D)
    bounds = lens.tolist()
    ref = torch.stack([_reference(q[r], w[r], keys[r], bounds[r]) for r in range(rows)])
    _rows_agree(got, ref, bounds)


def test_prefill_scores_are_causal_per_row():
    from mstar.model.glm52.dsa_kernels import prefill_scores

    gen = torch.Generator(device="cuda").manual_seed(1)
    ctx, rows = 3000, 70  # a chunk that ends the prompt; 70 is not a multiple of the row block
    pages = -(-ctx // PAGE)
    k_pages = _store(pages + 2, gen)
    table = torch.randperm(pages + 2, device=DEV, generator=gen)[:pages].to(torch.int32)
    positions = torch.arange(ctx - rows, ctx, dtype=torch.int32, device=DEV)
    q = torch.randn(rows, NH, D, device=DEV, generator=gen).bfloat16()
    w = torch.randn(rows, NH, device=DEV, generator=gen) * (D ** -0.5 * NH ** -0.5)
    max_len = pages * PAGE
    got = prefill_scores(q, w, k_pages, table, positions, max_len)
    keys = k_pages[table.long()].view(max_len, D)
    bounds = (positions + 1).tolist()
    ref = torch.stack([_reference(q[r], w[r], keys, bounds[r]) for r in range(rows)])
    _rows_agree(got, ref, bounds)
