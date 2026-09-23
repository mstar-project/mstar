"""``KimiRMSNorm.forward_sharded``: the rows spread over the ranks of a group, each rank normalizing its
own columns with the rows' statistics summed over the group, against ``forward`` on the whole rows.
CPU, with a stand-in group whose ranks are run one after the other."""
import torch

from mstar.model.kimi_k3.components.common import KimiRMSNorm


class _Group:
    """``world_size`` ranks whose ``all_reduce_plain`` sums what every rank hands in (the ranks run in
    turn: rank ``r``'s call gets the shards of all the others too)."""

    def __init__(self, world_size, rank, shards):
        self.world_size, self.rank, self._shards = world_size, rank, shards

    def all_reduce_plain(self, x):
        return sum(self._shards)


def test_sharded_norm_matches_the_whole_row():
    torch.manual_seed(0)
    d, tp, t = 96, 4, 5
    norm = KimiRMSNorm(d, eps=1e-5)
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(t, d)
    want = norm(x)
    chunk = d // tp
    shards = [x[:, r * chunk:(r + 1) * chunk] for r in range(tp)]
    stats = [s.float().pow(2).sum(-1, keepdim=True) for s in shards]
    got = torch.cat([norm.forward_sharded(shards[r], _Group(tp, r, stats), chunk) for r in range(tp)], dim=-1)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    # a single rank holds whole rows: the two are the same computation
    whole = _Group(1, 0, [x.float().pow(2).sum(-1, keepdim=True)])
    torch.testing.assert_close(norm.forward_sharded(x, whole, d), want)
