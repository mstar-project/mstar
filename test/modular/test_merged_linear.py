"""``MergedParallelLinear``: one GEMM for several projections of the same input, with
column-parallel and replicated segments, loaded per segment through the stacked-shard
``weight_loader`` and read back as views."""
import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.merged_linear import COLUMN, REPLICATED, MergedParallelLinear, Segment


def _group(rank, world):
    return CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(world)))


@pytest.mark.parametrize("world", [1, 2, 4])
def test_segments_load_and_project_like_separate_linears(world):
    torch.manual_seed(0)
    k = 16
    w_g, w_b, w_f = torch.randn(8, k), torch.randn(4, k), torch.randn(6, k)
    x = torch.randn(3, k)
    for rank in range(world):
        m = MergedParallelLinear(_group(rank, world), k, [("g", 8, COLUMN), ("b", 4, COLUMN), Segment("f_a", 6, REPLICATED)])
        # segments start on 8-element boundaries (aligned views); the width covers them all
        assert all(off % 8 == 0 for off in m.offsets.values())
        assert m.weight.shape[0] >= 8 // world + 4 // world + 6 and m.weight.shape[0] % 8 == 0
        assert m.offsets["b"] >= m.offsets["g"] + 8 // world and m.offsets["f_a"] >= m.offsets["b"] + 4 // world
        m.weight.weight_loader(m.weight, w_g, "g")
        m.weight.weight_loader(m.weight, w_b, "b")
        m.weight.weight_loader(m.weight, w_f, "f_a")
        out = m.project(x)
        assert set(out) == {"g", "b", "f_a"}
        g_rows = slice(rank * (8 // world), (rank + 1) * (8 // world))
        b_rows = slice(rank * (4 // world), (rank + 1) * (4 // world))
        torch.testing.assert_close(out["g"], x @ w_g[g_rows].T)
        torch.testing.assert_close(out["b"], x @ w_b[b_rows].T)
        torch.testing.assert_close(out["f_a"], x @ w_f.T)
        # views of one GEMM output: contiguous for a single row, column slices otherwise
        assert m.project(x[:1])["g"].is_contiguous()
        if world == 1:
            assert not out["b"].is_contiguous()


def test_validation_and_loader_survive_apply():
    with pytest.raises(ValueError):
        MergedParallelLinear(_group(0, 2), 8, [("a", 4, COLUMN), ("a", 4, COLUMN)])
    with pytest.raises(ValueError):
        MergedParallelLinear(_group(0, 2), 8, [("a", 4, "diagonal")]).weight_loader
    m = MergedParallelLinear(_group(1, 2), 8, [("a", 4, COLUMN), ("r", 3, REPLICATED)])
    with pytest.raises(ValueError):
        m.weight.weight_loader(m.weight, torch.zeros(4, 8))  # segment name required
    with pytest.raises(KeyError):
        m.weight.weight_loader(m.weight, torch.zeros(4, 8), "nope")
    m = m.to(torch.bfloat16)  # _apply re-attaches the loader to the new parameter object
    assert m.weight.weight_loader is not None and m.weight.dtype == torch.bfloat16
    m.weight.weight_loader(m.weight, torch.ones(3, 8, dtype=torch.bfloat16), "r")
    assert torch.equal(m.project(torch.ones(1, 8, dtype=torch.bfloat16))["r"], torch.full((1, 3), 8.0, dtype=torch.bfloat16))
    assert "r=3r" in repr(m)
