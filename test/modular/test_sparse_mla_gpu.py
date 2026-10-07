"""sparse_mla's FlashInfer plans against the torch reference, at DeepSeek's latent dims (ckv 512,
kpe 64) and GLM-5.3's NoPE ones (kpe 0)."""
import pytest
import torch

from mstar.engine.resources.attn import sparse_mla

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


def _reference(latent, lens, slots, q_nope, q_pe, scale):
    return sparse_mla.reference(q_nope.float(), q_pe.float(), latent.float(), slots, lens, scale)


@pytest.mark.parametrize("kpe", [64, 0])
@pytest.mark.parametrize("max_rows", [8192, 3])  # one launch; row chunks of 3
def test_eager_plan_matches_the_reference(kpe, max_rows, monkeypatch):
    monkeypatch.setattr(sparse_mla, "MAX_ROWS", max_rows)
    latent, lens, slots, q_nope, q_pe = _case(7, kpe, seed=1)
    scale = 192 ** -0.5
    plan = sparse_mla.EagerSparsePlan(lens, WIDTH, HEADS, CKV, kpe, scale, DEV)
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
    plan = sparse_mla.SparseGraphPlan(
        rows, WIDTH, torch.empty(sparse_mla.WORKSPACE_BYTES, dtype=torch.uint8, device=DEV))
    q_nope = torch.zeros(rows, HEADS, CKV, dtype=torch.bfloat16, device=DEV)
    q_pe = torch.zeros(rows, HEADS, kpe, dtype=torch.bfloat16, device=DEV)
    slots = torch.zeros(rows, WIDTH, dtype=torch.int32, device=DEV)
    plan.plan([1] * rows, HEADS, CKV, kpe, scale)
    plan.attend(q_nope, q_pe, latent, slots)  # warm up before capture
    torch.cuda.synchronize()
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


def test_an_eager_plan_queued_behind_other_work_keeps_its_own_plan():
    """FlashInfer's plan() copies its pinned host buffer with an async memcpy torch does not
    track. A windowed prefill drops a window's plan while that copy may still be queued and
    plans the next window, which can reuse the freed pinned buffer: the queued attention must
    still run with its own plan."""
    latent, lens, slots, q_nope, q_pe = _case(4, 64, seed=5)
    _, o_lens, o_slots, o_q, o_pe = _case(2, 64, seed=6)
    scale = 192 ** -0.5

    def two_windows():
        first = sparse_mla.EagerSparsePlan(lens, WIDTH, HEADS, CKV, 64, scale, DEV)
        out = first.attend(q_nope, q_pe, latent, slots)
        del first  # its wrappers, and the pinned buffer a queued copy may still read
        sparse_mla.EagerSparsePlan(o_lens, WIDTH, HEADS, CKV, 64, scale, DEV).attend(
            o_q, o_pe, latent, o_slots)
        return out

    want = two_windows()  # also warms the allocators: no cudaMalloc (a device sync) below
    torch.cuda.synchronize()
    torch.cuda._sleep(2_000_000_000)  # ~1 s of GPU work ahead of the plans' copies
    got = two_windows()
    torch.cuda.synchronize()
    assert torch.equal(got, want)
