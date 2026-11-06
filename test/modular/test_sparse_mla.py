"""sparse_mla on CPU: the torch reference against a masked dense softmax, a sharded row
selection against the unsharded one, and the plan's packed layout."""
import pytest
import torch

from mstar.engine.resources.dsa import sparse_mla


def test_reference_is_masked_dense_softmax():
    torch.manual_seed(2)
    heads, rank, kpe, page = 4, 32, 8, 8
    latent = torch.randn(7, page, rank + kpe)
    picks = [[0, 1, 3], [4, 5, 6, 7, 12, 13, 14, 15, 20], list(range(16)) + [48, 49]]
    slots = torch.full((3, 20), -1, dtype=torch.int32)
    for r, ps in enumerate(picks):
        slots[r, :len(ps)] = torch.tensor(ps)
    q_nope, q_pe = torch.randn(3, heads, rank), torch.randn(3, heads, kpe)
    out = sparse_mla.reference(q_nope, q_pe, latent, slots, [len(p) for p in picks], 0.3)
    flat = latent.view(-1, rank + kpe)
    for r, ps in enumerate(picks):
        keys = flat[ps]
        query = torch.cat([q_nope[r], q_pe[r]], dim=-1)
        ref = ((query @ keys.T) * 0.3).softmax(-1) @ keys[:, :rank]
        assert torch.allclose(out[r], ref, atol=1e-5)


def test_reference_runs_fp32_under_autocast():
    torch.manual_seed(3)
    latent = torch.randn(4, 8, 40)
    slots = torch.randperm(32)[:12].int().view(2, 6)
    q_nope, q_pe = torch.randn(2, 4, 32), torch.randn(2, 4, 8)
    want = sparse_mla.reference(q_nope, q_pe, latent, slots, [6, 4], 0.3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = sparse_mla.reference(q_nope, q_pe, latent, slots, [6, 4], 0.3)
    assert torch.equal(got, want)


class _FakeGroup:
    """A TP group of ``world`` ranks run one after another: all_gather returns the blocks
    every rank has produced so far, a filler for the rest."""

    def __init__(self, world):
        self.world_size, self.rank, self.blocks = world, 0, {}

    def all_gather(self, t, dim=0):
        self.blocks[self.rank] = t
        filler = torch.full_like(t, -7)
        return torch.cat([self.blocks.get(r, filler) for r in range(self.world_size)], dim=dim)


@pytest.mark.parametrize("world, r0, r1", [(4, 10, 333), (3, 0, 1000), (8, 5, 517)])
def test_a_sharded_selection_comes_back_as_each_ranks_own_slots(world, r0, r1):
    """The ranks hold the request in different pages: each selects a block of rows in its own
    slots, and every rank gets the whole selection back in its own slots."""
    page, n_pages = 4, 256
    pages = [torch.randperm(400, generator=torch.Generator().manual_seed(r))[:n_pages].int()
             for r in range(world)]
    # row r picks positions r, r - 7, r // 2, then nothing
    rows = torch.arange(n_pages * page)[:, None]
    picks = torch.cat([rows, rows - 7, rows // 2, torch.full_like(rows, -1)], 1).clamp_min(-1)
    picks = picks.to(torch.int32)

    def slots_on(rank, positions):
        return sparse_mla.RequestPages(pages[rank], page).to_slots(positions)

    group = _FakeGroup(world)
    for rank in range(world):  # the last rank's call sees every block
        group.rank = rank
        got = sparse_mla.select_rows(
            lambda a, b: slots_on(group.rank, picks[a:b]), r0, r1, 4,
            torch.device("cpu"), group, sparse_mla.RequestPages(pages[rank], page))
    assert torch.equal(got, slots_on(world - 1, picks[r0:r1]))


def test_the_plan_runs_on_one_intra_op_thread():
    """The plan narrows its thread to one intra-op thread and gives the caller its count back,
    also when the plan raises."""
    seen = []

    class Wrapper:
        def plan(self, *args):
            seen.append(torch.get_num_threads())

    class Fails:
        def plan(self, *args):
            raise RuntimeError("plan failed")

    before = torch.get_num_threads()
    torch.set_num_threads(2)  # so CI's OMP_NUM_THREADS=1 can't pass it without the narrowing
    try:
        sparse_mla._plan(Wrapper(), torch.zeros(4, dtype=torch.int32), [2, 1], 4, 512, 64, 0.1)
        assert seen == [1]
        assert torch.get_num_threads() == 2
        with pytest.raises(RuntimeError, match="plan failed"):
            sparse_mla._plan(Fails(), torch.zeros(4, dtype=torch.int32), [2, 1], 4, 512, 64, 0.1)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)


def test_pack_lays_the_slots_out_as_the_plan_describes():
    """The plan describes packed rows, which FlashInfer >= 0.7 checks against each row's
    length, and ``_pack`` lays the slots out the same way."""
    seen = {}

    class Wrapper:
        def plan(self, qo, kv, indices, lens, *args):
            seen.update(kv=kv, lens=lens)

    sparse_mla._plan(Wrapper(), torch.zeros(10, dtype=torch.int32), [2, 3, 1], 4, 512, 64, 0.1)
    slots = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32)
    dst = torch.full((3 * 3 + 1,), -7, dtype=torch.int32)
    sparse_mla._pack(dst, slots, seen["kv"], seen["lens"])
    # what a row leaves unused lands on the spare last entry
    assert dst[:9].tolist() == [10, 11, 20, 21, 22, 30, -7, -7, -7]
