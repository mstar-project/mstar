"""sparse_mla on CPU: the torch reference against a masked dense softmax, and a sharded row
selection against the unsharded one."""
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


def test_select_rows_shards_into_the_unsharded_selection():
    rows = torch.arange(1000, dtype=torch.int32)[:, None].expand(-1, 5).contiguous()
    request = sparse_mla.RequestPages(torch.arange(250, dtype=torch.int32), 4)  # slot = position

    def select(a, b):
        return rows[a:b]

    for world, r0, r1 in [(4, 10, 333), (3, 0, 1000), (8, 5, 517)]:
        group = _FakeGroup(world)
        for rank in range(world):  # the last rank's call sees every block
            group.rank = rank
            got = sparse_mla.select_rows(select, r0, r1, 5, torch.device("cpu"), group, request)
        assert torch.equal(got, rows[r0:r1]), (world, r0, r1)


def test_a_sharded_selection_comes_back_as_each_ranks_own_slots():
    """The ranks hold the request in different pages: each selects its block in its own slots,
    the blocks cross as positions, and every rank gets the whole selection in its own slots.
    Gathering slots instead gave a rank another rank's pages, other tokens or other requests'."""
    page, n_pages, world = 4, 80, 4
    pages = [torch.randperm(200, generator=torch.Generator().manual_seed(r))[:n_pages].int()
             for r in range(world)]
    # row r picks positions r, r - 7, r // 2, then nothing
    rows = torch.arange(n_pages * page)[:, None]
    picks = torch.cat([rows, rows - 7, rows // 2, torch.full_like(rows, -1)], 1).clamp_min(-1)
    picks = picks.to(torch.int32)

    def slots_on(rank, positions):
        return sparse_mla.RequestPages(pages[rank], page).to_slots(positions)

    group = _FakeGroup(world)
    for rank in range(world):
        group.rank = rank
        got = sparse_mla.select_rows(
            lambda a, b: slots_on(group.rank, picks[a:b]), 0, picks.shape[0], 4,
            torch.device("cpu"), group, sparse_mla.RequestPages(pages[rank], page))
    assert torch.equal(got, slots_on(world - 1, picks))
    assert torch.equal(sparse_mla.RequestPages(pages[1], page).to_positions(slots_on(1, picks)), picks)


def test_a_sharded_selection_needs_the_request_pages():
    with pytest.raises(ValueError, match="request's pages"):
        sparse_mla.select_rows(lambda a, b: torch.zeros(b - a, 2, dtype=torch.int32), 0, 1000, 2,
                               torch.device("cpu"), _FakeGroup(4))


def test_select_rows_keeps_short_spans_on_every_rank():
    group = _FakeGroup(8)
    calls = []

    def select(a, b):
        calls.append((a, b))
        return torch.zeros(b - a, 2, dtype=torch.int32)

    sparse_mla.select_rows(select, 0, 8 * sparse_mla.SHARD_MIN_ROWS - 1, 2, torch.device("cpu"),
                           group)
    assert calls == [(0, 8 * sparse_mla.SHARD_MIN_ROWS - 1)]


def test_the_plan_describes_packed_rows():
    """FlashInfer >= 0.7 checks each row's index count against its length (one-token
    pages): a fixed row stride failed every plan with a short row."""
    seen = {}

    class Wrapper:
        def plan(self, qo, kv, indices, lens, *args):
            seen.update(kv=kv.tolist(), lens=lens.tolist())

    sparse_mla._plan(Wrapper(), torch.zeros(10, dtype=torch.int32), [2, 3, 1], 4, 512, 64, 0.1)
    assert seen == {"kv": [0, 2, 5, 6], "lens": [2, 3, 1]}


def test_the_plan_runs_on_one_intra_op_thread():
    """FlashInfer's plan stages through a host copy ATen would split over every intra-op
    thread: the plan narrows its own thread to one and gives the caller its count back, also
    when the plan raises. The caller starts at two threads, so CI's OMP_NUM_THREADS=1 can't
    pass it without the narrowing."""
    seen = []

    class Wrapper:
        def plan(self, *args):
            seen.append(torch.get_num_threads())

    class Fails:
        def plan(self, *args):
            raise RuntimeError("plan failed")

    before = torch.get_num_threads()
    torch.set_num_threads(2)
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
    slots = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32)
    dst = torch.full((3 * 3 + 1,), -7, dtype=torch.int32)
    sparse_mla._pack(dst, slots, torch.tensor([0, 2, 5, 6], dtype=torch.int32),
                     torch.tensor([2, 3, 1], dtype=torch.int32))
    # what a row leaves unused lands on the spare last entry
    assert dst[:9].tolist() == [10, 11, 20, 21, 22, 30, -7, -7, -7]


def test_a_row_attends_at_most_its_width():
    # packed, a length past the width read index entries no slot was written to
    assert sparse_mla._lens([1, 5, 9], 5) == [1, 5, 5]


def test_the_kernel_serves_only_its_latent_shape(monkeypatch):
    # a reduced model's latent went to the kernel, which read out of bounds
    q = torch.empty(2, 4, 32, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    assert not sparse_mla.uses_kernel(q, torch.empty(8, 1, 32 + 16, dtype=torch.bfloat16))
    q = torch.empty(2, 4, 512, dtype=torch.bfloat16)
    assert sparse_mla.uses_kernel(q, torch.empty(8, 1, 512 + 64, dtype=torch.bfloat16))
