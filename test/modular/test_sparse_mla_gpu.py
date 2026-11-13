"""sparse_mla's FlashInfer plans against the torch reference, at DeepSeek's latent dims (ckv 512,
kpe 64) and GLM-5.3's NoPE ones (kpe 0)."""
import pytest
import torch

from mstar.engine.resources.attn.workspace import WorkspacePool
from mstar.engine.resources.dsa import sparse_mla

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CKV, HEADS, PAGE, PAGES, WIDTH = 512, 8, 64, 200, 300
DEV = torch.device("cuda")


def _case(rows, kpe, seed):
    gen = torch.Generator().manual_seed(seed)
    latent = (torch.randn(PAGES, PAGE, CKV + kpe, generator=gen) * 0.5).bfloat16().to(DEV)
    lens = torch.randint(1, WIDTH + 1, (rows,), generator=gen).tolist()
    slots = torch.full((rows, WIDTH), -1, dtype=torch.int32)
    for r, n in enumerate(lens):
        slots[r, :n] = torch.randperm(PAGES * PAGE, generator=gen)[:n].int()
    q_nope = torch.randn(rows, HEADS, CKV, generator=gen).bfloat16().to(DEV)
    q_pe = torch.randn(rows, HEADS, kpe, generator=gen).bfloat16().to(DEV)
    return latent, lens, slots.to(DEV), q_nope, q_pe


def _workspaces() -> WorkspacePool:
    return WorkspacePool(DEV, size_mb=128, owner="dsa")


def _reference(latent, lens, slots, q_nope, q_pe, scale):
    return sparse_mla.reference(q_nope.float(), q_pe.float(), latent.float(), slots, lens, scale)


@pytest.mark.parametrize("kpe", [64, 0])
@pytest.mark.parametrize("max_rows", [8192, 3])  # one launch; row chunks of 3
def test_eager_plan_matches_the_reference(kpe, max_rows, monkeypatch):
    monkeypatch.setattr(sparse_mla, "MAX_ROWS", max_rows)
    latent, lens, slots, q_nope, q_pe = _case(7, kpe, seed=1)
    scale = 192 ** -0.5
    plan = sparse_mla.EagerSparsePlans(_workspaces()).plan(lens, WIDTH, HEADS, CKV, kpe, scale)
    for flip in (False, True):  # a second layer refills the same plan's indices
        layer_slots = slots.flip(1) if flip else slots
        layer_slots = torch.stack([
            torch.cat([row[row >= 0], row[row < 0]]) for row in layer_slots])
        got = plan.attend(q_nope, q_pe, latent, layer_slots)
        ref = _reference(latent, lens, layer_slots, q_nope, q_pe, scale)
        torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("kpe", [64, 0])
def test_graph_plan_replays_each_steps_lengths(kpe):
    rows, scale = 5, 192 ** -0.5
    latent, _, _, _, _ = _case(rows, kpe, seed=2)
    workspaces = _workspaces()
    plan = sparse_mla.SparseGraphPlan(rows, WIDTH, workspaces.get("decode", 0))
    q_nope = torch.zeros(rows, HEADS, CKV, dtype=torch.bfloat16, device=DEV)
    q_pe = torch.zeros(rows, HEADS, kpe, dtype=torch.bfloat16, device=DEV)
    slots = torch.zeros(rows, WIDTH, dtype=torch.int32, device=DEV)
    plan.plan([1] * rows, HEADS, CKV, kpe, scale)
    plan.attend(q_nope, q_pe, latent, slots)  # warm up before capture
    torch.cuda.synchronize()
    workspaces.seal()  # as the engine does: the graph holds the workspace's address
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = plan.attend(q_nope, q_pe, latent, slots)
    for seed in (3, 4):
        _, lens, new_slots, new_q, new_pe = _case(rows, kpe, seed)
        q_nope.copy_(new_q)
        q_pe.copy_(new_pe)
        slots.copy_(new_slots)
        plan.plan(lens, HEADS, CKV, kpe, scale)
        graph.replay()
        ref = _reference(latent, lens, new_slots, new_q, new_pe, scale)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("max_rows", [8192, 3])  # one launch; row chunks of 3
def test_reused_eager_plans_keep_each_windows_plan(max_rows, monkeypatch):
    """The resource's eager plans (two sets of wrappers taken in turn, re-planned without a
    stream sync): windows queued behind other work each attend with their own plan."""
    monkeypatch.setattr(sparse_mla, "MAX_ROWS", max_rows)
    cases = [_case(rows, 64, seed) for rows, seed in ((4, 7), (2, 8), (5, 9))]
    scale = 192 ** -0.5
    plans = sparse_mla.EagerSparsePlans(_workspaces())

    def windows():
        return [plans.plan(lens, WIDTH, HEADS, CKV, 64, scale).attend(q_nope, q_pe, latent, slots)
                for latent, lens, slots, q_nope, q_pe in cases]

    windows()  # the first plans allocate: no cudaMalloc (a device sync) below
    torch.cuda.synchronize()
    torch.cuda._sleep(2_000_000_000)  # ~1 s of GPU work ahead of the plans' copies
    got = windows()
    torch.cuda.synchronize()
    for (latent, lens, slots, q_nope, q_pe), out in zip(cases, got, strict=True):
        ref = _reference(latent, lens, slots, q_nope, q_pe, scale)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
