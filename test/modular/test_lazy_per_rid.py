"""The stop check's per-rid host dicts are built only when something reads
them."""
import torch

from mstar.worker.worker import Worker, _LazyPerRid


def test_lazy_per_rid_builds_on_first_read_only():
    host = {"new_token": torch.arange(3).view(3, 1), "flag": torch.zeros(3, dtype=torch.bool)}
    rids = [7, 8, 9]
    lazy = _LazyPerRid(host, rids)
    assert lazy._built is None
    assert len(lazy) == 3 and bool(lazy)
    assert lazy._built is None, "len and truth must not build the dicts"
    eager = Worker._rows_to_per_rid(host, rids)
    assert lazy.get(8).keys() == eager[8].keys()
    for name in eager[8]:
        assert torch.equal(lazy.get(8)[name][0], eager[8][name][0])
    assert lazy.get(42) is None
    assert 7 in lazy and list(lazy) == rids
    assert lazy[9]["new_token"][0].item() == 2
    assert not _LazyPerRid(host, [])
