"""``ExpertSharding``: the placement of a MoE layer's experts over a comm group (tensor
parallelism on the intermediate dim, expert parallelism over whole experts, or both), its
checkpoint routing and local expert map, and the alignment fallback's handling of assignments a
rank does not hold."""
import pytest
import torch

from mstar.model.components.expert_sharding import ExpertSharding


def test_geometry_tp_ep_and_hybrid():
    tp = ExpertSharding(224, 3072, 8, 5)
    assert (tp.tp_size, tp.ep_size, tp.local_experts, tp.inter_local) == (8, 1, 224, 384)
    assert (tp.expert_offset, tp.inter_offset, tp.is_partial) == (0, 5 * 384, False)
    ep = ExpertSharding(224, 3072, 8, 5, ep_size=8)
    assert (ep.tp_size, ep.local_experts, ep.inter_local, ep.expert_offset, ep.inter_offset) == (1, 28, 3072, 140, 0)
    assert ep.is_partial and ep.invalid_id == 28
    hy = ExpertSharding(224, 3072, 8, 5, ep_size=4)  # ep rank 2 (ranks 4, 5), tp rank 1
    assert (hy.tp_size, hy.ep_rank, hy.tp_rank) == (2, 2, 1)
    assert (hy.local_experts, hy.inter_local, hy.expert_offset, hy.inter_offset) == (56, 1536, 112, 1536)
    assert hy.owns(112) and hy.owns(167) and not hy.owns(111) and not hy.owns(168)
    assert hy.local_expert(150) == 38
    with pytest.raises(ValueError):
        hy.local_expert(0)
    for bad in (3, 16):
        with pytest.raises(ValueError):
            ExpertSharding(224, 3072, 8, 0, ep_size=bad)
    with pytest.raises(ValueError):
        ExpertSharding(10, 3072, 8, 0, ep_size=8)  # experts not divisible
    with pytest.raises(ValueError):
        ExpertSharding(224, 100, 8, 0)  # intermediate not divisible by tp
    with pytest.raises(ValueError):
        ExpertSharding(224, 3072, 8, 8)
    assert "ep 4 x tp 2" in hy.describe()


def test_from_group_and_expert_map():
    from mstar.distributed.communication import CommGroup

    assert ExpertSharding.from_group(None, 8, 256) == ExpertSharding(8, 256, 1, 0, 1)
    assert ExpertSharding.from_group(CommGroup.trivial(), 8, 256, ep_size=1).world_size == 1
    grp = CommGroup(my_global_rank=6, my_group_rank=2, group_members=[4, 5, 6, 7])
    sh = ExpertSharding.from_group(grp, 8, 256, ep_size=4)  # rank 2 of 4 holds experts 4, 5
    assert (sh.rank, sh.expert_offset, sh.local_experts) == (2, 4, 2)
    assert sh.expert_map("cpu").tolist() == [2, 2, 2, 2, 0, 1, 2, 2]
    idx = torch.tensor([[0, 5], [4, 7]], dtype=torch.int32)
    loc = sh.localize(idx)
    assert loc.tolist() == [[2, 1], [0, 2]] and loc.dtype == torch.int32
    assert torch.equal(sh.localize(idx), loc)  # cached map, same answer
    tp = ExpertSharding.from_group(grp, 8, 256)
    assert tp.localize(idx) is idx  # every expert local: no work, no copy


def _route(sh, gu, dn, w1, w3, w2, per_col=1):
    for e in range(w1.shape[0]):
        sh.load_gate_up(gu, w1[e], "gate", e)
        sh.load_gate_up(gu, w3[e], "up", e)
        sh.load_down(dn, w2[e], e, per_col)


@pytest.mark.parametrize("world,ep", [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4)])
def test_loaders_route_each_ranks_slice(world, ep):
    e, inter, k = 8, 64, 32
    torch.manual_seed(0)
    w1, w3, w2 = torch.randn(e, inter, k), torch.randn(e, inter, k), torch.randn(e, k, inter)
    # packed-style tensors: 2 intermediate channels per stored column of the down projection
    w2p = torch.randint(0, 256, (e, k, inter // 2), dtype=torch.uint8)
    for rank in range(world):
        sh = ExpertSharding(e, inter, world, rank, ep_size=ep)
        gu = torch.zeros(sh.local_experts, 2 * sh.inter_local, k)
        dn = torch.zeros(sh.local_experts, k, sh.inter_local)
        dnp = torch.zeros(sh.local_experts, k, sh.inter_local // 2, dtype=torch.uint8)
        _route(sh, gu, dn, w1, w3, w2)
        for ex in range(e):
            sh.load_down(dnp, w2p[ex], ex, per_col=2)
        for le in range(sh.local_experts):
            ge = sh.expert_offset + le
            cols = slice(sh.inter_offset, sh.inter_offset + sh.inter_local)
            assert torch.equal(gu[le, :sh.inter_local], w1[ge, cols])
            assert torch.equal(gu[le, sh.inter_local:], w3[ge, cols])
            assert torch.equal(dn[le], w2[ge][:, cols])
            assert torch.equal(dnp[le], w2p[ge][:, sh.inter_offset // 2:(sh.inter_offset + sh.inter_local) // 2])


@pytest.mark.parametrize("world,ep", [(2, 1), (2, 2), (4, 2), (4, 4)])
def test_rank_partials_sum_to_the_full_moe(world, ep):
    """Every placement gives per-rank partial sums whose total is the single-rank result: the
    combining all-reduce is the same collective for TP, EP and the hybrid."""
    from mstar.model.kimi_k3.reference.moe import routed_experts_loop

    e, inter, k, t, top_k = 8, 64, 32, 5, 3
    torch.manual_seed(1)
    w1, w3, w2 = torch.randn(e, inter, k), torch.randn(e, inter, k), torch.randn(e, k, inter)
    z = torch.randn(t, k)
    idx = torch.stack([torch.randperm(e)[:top_k] for _ in range(t)])
    w = torch.softmax(torch.randn(t, top_k), -1)
    full = routed_experts_loop(z, idx, w, torch.cat([w1, w3], 1), w2, 4.0, 25.0)
    total = torch.zeros_like(full)
    for rank in range(world):
        sh = ExpertSharding(e, inter, world, rank, ep_size=ep)
        gu = torch.zeros(sh.local_experts, 2 * sh.inter_local, k)
        dn = torch.zeros(sh.local_experts, k, sh.inter_local)
        _route(sh, gu, dn, w1, w3, w2)
        part = routed_experts_loop(z, sh.localize(idx), w, gu, dn, 4.0, 25.0)
        if sh.is_partial:  # a token routed to none of this rank's experts gets exactly zero
            none_here = (sh.localize(idx) == sh.invalid_id).all(1)
            assert torch.equal(part[none_here], torch.zeros_like(part[none_here]))
        total += part
    torch.testing.assert_close(total, full, rtol=1e-4, atol=1e-4)


def test_align_fallback_skips_assignments_held_elsewhere():
    from mstar.utils.fused_moe.align import _moe_align_block_size_torch

    torch.manual_seed(2)
    e, block, t, top_k = 6, 4, 9, 3
    ids = torch.randint(0, e + 1, (t, top_k), dtype=torch.int32)  # id 6 = held by another rank
    ids[0] = e  # a token with no local expert at all
    n = ids.numel() + e * (block - 1)
    sorted_ids = torch.empty(n, dtype=torch.int32)
    expert_ids = torch.empty((n + block - 1) // block, dtype=torch.int32)
    post = torch.empty(1, dtype=torch.int32)
    _moe_align_block_size_torch(ids, block, e, sorted_ids, expert_ids, post)
    flat = ids.reshape(-1)
    valid = (flat < e).nonzero().reshape(-1).tolist()
    placed = sorted_ids[: int(post)].tolist()
    real = [p for p in placed if p != ids.numel()]
    assert sorted(real) == sorted(valid)  # every local assignment once, none of the foreign ones
    for slot, pos in enumerate(placed):
        if pos != ids.numel():
            assert int(flat[pos]) == int(expert_ids[slot // block])
    counts = torch.bincount(flat[valid], minlength=e)
    assert int(post) == int(((counts + block - 1) // block * block).sum())
