"""glm5_next DSA's CUDA paths against its torch ones (dsa.py) at the real indexer and latent dims:
the pool-key and score kernels, flashinfer's top-k + slot transform, and a captured decode
against the same rows eager."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.attn import sparse_mla
from mstar.model.glm5_next import dsa

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NH, D, KP, TOPK, PAGE, W, HEADS = 32, 128, 4, 2048, 128, 512, 8
DEV = "cuda"


def _store(lens, seed=0, pages_total=None):
    """A bf16 index plane with random k/gate rows for requests of ``lens`` tokens on shuffled
    pages, pool keys written by the torch path. Returns (plane, page lists)."""
    g = torch.Generator().manual_seed(seed)
    need = [-(-n // PAGE) for n in lens]
    total = pages_total or sum(need) + 3
    perm = torch.randperm(total, generator=g).tolist()
    tables, at = [], 0
    for n in need:
        tables.append(perm[at:at + n])
        at += n
    plane = torch.zeros(total, PAGE, W, dtype=torch.bfloat16)
    ape = torch.randn(KP, D, generator=g)
    for table, n in zip(tables, lens, strict=True):
        for t in range(n):
            plane[table[t // PAGE], t % PAGE, :2 * D] = torch.randn(2 * D, generator=g).bfloat16()
    return plane, tables, ape


def _ctx(tables, row_req, positions, device, max_pools=None, graph=False):
    flat, starts = [], []
    for t in tables:
        starts.append(len(flat))
        flat.extend(t)
    spans = None if graph else [(r, 1, q) for r, q in enumerate(row_req)]
    return dsa.Glm5NextDsaContext(
        pos=torch.tensor(positions, dtype=torch.int32, device=device),
        row_req=torch.tensor(row_req, dtype=torch.int32, device=device),
        pages=torch.tensor(flat, dtype=torch.int32, device=device),
        page_start=torch.tensor(starts, dtype=torch.int32, device=device),
        host_pos=None if graph else list(positions), spans=spans,
        host_page_start=None if graph else starts,
        max_pools=max_pools or (max(positions) + 1) // KP, page_size=PAGE, topk=TOPK, kpool=KP)


def _write_pools(plane, tables, lens, ape, device):
    rows_req = [i for i, n in enumerate(lens) for _ in range(n)]
    rows_pos = [p for n in lens for p in range(n)]
    ctx = _ctx(tables, rows_req, rows_pos, device)
    view = plane.to(device)
    if device == "cpu":
        dsa._pool_keys_torch(view, ctx, ape, D)
    else:
        from mstar.model.glm5_next.dsa_kernels import pool_keys

        pool_keys(view, ctx.pages, ctx.page_start, ctx.row_req, ctx.pos, ape.to(device), D)
    return view


def test_pool_keys_kernel_matches_torch():
    lens = [1000, 4096 + 3, 517]
    plane, tables, ape = _store(lens)
    ref = _write_pools(plane.clone(), tables, lens, ape, "cpu")
    got = _write_pools(plane.clone(), tables, lens, ape, DEV).cpu()
    a, b = ref[..., 2 * D:3 * D].float(), got[..., 2 * D:3 * D].float()
    # exp/divide rounding and the 4-term sum's order may move a bf16 value by an ulp or two
    assert ((a - b).abs() <= a.abs() * 2 ** -6 + 2 ** -10).all()
    assert (a == b).float().mean() > 0.999


@pytest.mark.parametrize("lens", [[9000, 2049, 100, 40000], [131072]])
def test_decode_selection_matches_torch(lens):
    plane, tables, ape = _store(lens, seed=1)
    view = _write_pools(plane, tables, lens, ape, DEV)
    torch.manual_seed(2)
    positions = [n - 1 for n in lens]
    q = torch.randn(len(lens), NH, D, device=DEV).bfloat16()
    w = torch.randn(len(lens), NH, device=DEV)
    ctx = _ctx(tables, list(range(len(lens))), positions, DEV)
    scores = dsa._decode_scores(q, w, view, ctx)
    ref = dsa._scores_torch(q, w, view, ctx, ctx.row_req.long(), ctx.pos, ctx.max_pools)
    for r, p in enumerate(positions):
        n = (p + 1) // KP
        assert torch.allclose(scores[r, :n], ref[r, :n], rtol=1e-4, atol=1e-3)
    got = dsa.select(q, w, view, ctx)
    cpu_ctx = _ctx(tables, list(range(len(lens))), positions, "cpu")
    want = dsa.select(q.cpu(), w.cpu(), view.cpu(), cpu_ctx)
    for r, n in enumerate(dsa.attn_lens(positions, TOPK, KP)):
        g, h = set(got[r, :n].tolist()), set(want[r, :n].tolist())
        assert len(g) == n and len(g & h) >= n - 2 * KP  # a near-tie may swap one pool


def test_prefill_selection_matches_torch():
    lens = [20000]
    plane, tables, ape = _store(lens, seed=3)
    view = _write_pools(plane, tables, lens, ape, DEV)
    torch.manual_seed(4)
    positions = list(range(19000, 20000))
    q = torch.randn(len(positions), NH, D, device=DEV).bfloat16()
    w = torch.randn(len(positions), NH, device=DEV)
    ctx = _ctx(tables, [0] * len(positions), positions, DEV)
    ctx.spans = [(0, len(positions), 0)]
    got = dsa.select(q, w, view, ctx)
    cpu_ctx = _ctx(tables, [0] * len(positions), positions, "cpu")
    cpu_ctx.spans = [(0, len(positions), 0)]
    want = dsa.select(q.cpu(), w.cpu(), view.cpu(), cpu_ctx)
    same = 0
    for r, n in enumerate(dsa.attn_lens(positions, TOPK, KP)):
        g, h = set(got[r, :n].tolist()), set(want[r, :n].tolist())
        assert len(g) == n and len(g & h) >= n - 2 * KP
        same += g == h
    assert same >= 0.95 * len(positions)


def _compiled(fn) -> int:
    caches = getattr(fn, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton keeps no per-device kernel cache")
    return sum(len(c[0]) for c in caches.values())


def test_prefill_scores_compile_once_whatever_the_split():
    """The split count follows the rows and the context, so one long prompt's chunks take many
    values of it; none may compile the kernel again mid-request."""
    from mstar.model.glm5_next.dsa_kernels import _prefill_scores_kernel, prefill_scores

    lens = [64000]
    plane, tables, ape = _store(lens, seed=9)
    view = _write_pools(plane, tables, lens, ape, DEV)
    ctx = _ctx(tables, [0], [0], DEV)
    before = _compiled(_prefill_scores_kernel)
    # rows and pools all multiples of 16 (one integer specialization), splits 25, 50, 16, 4
    for rows, pools in [(16, 1600), (64, 3200), (256, 6400), (1024, 16000)]:
        pos = torch.arange(pools * KP - rows, pools * KP, dtype=torch.int32, device=DEV)
        q = torch.randn(rows, NH, D, device=DEV).bfloat16()
        w = torch.randn(rows, NH, device=DEV)
        prefill_scores(q, w, view, ctx.pages, 0, pos, pools, KP)
    torch.cuda.synchronize()
    assert _compiled(_prefill_scores_kernel) - before <= 1


def test_expand_slots_kernel_matches_torch():
    torch.manual_seed(8)
    tables = [list(range(i * 80, i * 80 + 80)) for i in range(5)]
    positions = [0, 2, 3, 2050, 9000]
    ctx = _ctx(tables, list(range(5)), positions, DEV)
    k = TOPK // KP
    first = torch.full((5, k), -1, dtype=torch.int32, device=DEV)
    for r, p in enumerate(positions):
        n = min((p + 1) // KP, k)
        pools = torch.randperm((p + 1) // KP)[:n] if n else torch.empty(0, dtype=torch.long)
        first[r, :n] = torch.tensor([tables[r][int(j) * KP // PAGE] * PAGE + int(j) * KP % PAGE
                                     for j in pools], dtype=torch.int32)
    got = dsa.expand_slots(first, ctx)
    want = dsa.expand_slots(first.cpu(), _ctx(tables, list(range(5)), positions, "cpu"))
    assert torch.equal(got.cpu(), want)


def test_captured_decode_matches_eager():
    lens = [3000, 9001, 65536, 800]
    plane, tables, ape = _store(lens, seed=6)
    view = _write_pools(plane, tables, lens, ape, DEV)
    latent = torch.randn(view.shape[0], PAGE, W, device=DEV).bfloat16()
    rows = len(lens)
    window = 1 << 20
    ctx = _ctx(tables, list(range(rows)), [0] * rows, DEV, max_pools=window // KP, graph=True)
    ctx.pages = torch.cat([ctx.pages, ctx.pages.new_zeros(rows * window // PAGE)])
    plan = sparse_mla.SparseGraphPlan(
        rows, ctx.width, torch.empty(sparse_mla.WORKSPACE_BYTES, dtype=torch.uint8, device=DEV))
    ctx.sparse_plan = plan
    q = torch.zeros(rows, NH, D, device=DEV).bfloat16()
    w = torch.zeros(rows, NH, device=DEV)
    q_nope = torch.zeros(rows, HEADS, W, device=DEV).bfloat16()
    q_pe = q_nope.new_zeros(rows, HEADS, 0)

    def step():
        slots = dsa.select(q, w, view, ctx)
        return dsa.sparse_attend(q_nope, q_pe, latent, slots, ctx, 0.0625)

    plan.plan([1] * rows, HEADS, W, 0, 0.0625)
    step()  # warm up the kernels before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    torch.manual_seed(7)
    for trial in range(2):
        positions = [n - 1 - trial for n in lens]
        ctx.pos.copy_(torch.tensor(positions, dtype=torch.int32))
        q.copy_(torch.randn(rows, NH, D).bfloat16())
        w.copy_(torch.randn(rows, NH))
        q_nope.copy_(torch.randn(rows, HEADS, W).bfloat16())
        plan.plan(dsa.attn_lens(positions, TOPK, KP), HEADS, W, 0, 0.0625)
        graph.replay()
        eager_ctx = _ctx(tables, list(range(rows)), positions, DEV)
        slots = dsa.select(q, w, view, eager_ctx)
        want = dsa.sparse_attend(q_nope, q_pe, latent, slots, eager_ctx, 0.0625)
        assert torch.allclose(out.float(), want.float(), atol=1e-2, rtol=1e-2), trial
