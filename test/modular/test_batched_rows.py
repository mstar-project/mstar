"""BatchedRows: each request gets its own row of a stacked forward."""

from types import SimpleNamespace

import pytest
import torch

from mstar.model.components.batched_rows import BatchedRows


class _Doubler(BatchedRows):
    output_keys = ("y", "z")

    def run_batch(self, x, **kwargs):
        return {"y": 2 * x, "z": x + 1}


@pytest.mark.parametrize("keep_batch_dim", [False, True])
def test_rows_match_the_stacked_batch(keep_batch_dim):
    node = _Doubler()
    node.keep_batch_dim = keep_batch_dim
    x = torch.arange(12.0).reshape(3, 4)
    out = node.forward_batched("walk", SimpleNamespace(request_ids=[7, 8, 9]), x=x)
    assert list(out) == [7, 8, 9]
    for i, rid in enumerate([7, 8, 9]):
        row = out[rid]["y"][0]
        assert row.shape == ((1, 4) if keep_batch_dim else (4,))
        torch.testing.assert_close(row.reshape(4), 2 * x[i])
        torch.testing.assert_close(out[rid]["z"][0].reshape(4), x[i] + 1)


@pytest.mark.parametrize("keep_batch_dim", [False, True])
def test_forward_matches_a_batch_of_one(keep_batch_dim):
    node = _Doubler()
    node.keep_batch_dim = keep_batch_dim
    x = torch.ones(1, 4)
    single = node.forward("walk", SimpleNamespace(request_ids=[0]), x=x)
    batched = node.forward_batched("walk", SimpleNamespace(request_ids=[0]), x=x)[0]
    for key in node.output_keys:
        torch.testing.assert_close(single[key][0], batched[key][0])


def test_forward_passes_a_stacked_batch_through_when_keeping_the_batch_dim():
    node = _Doubler()
    node.keep_batch_dim = True
    x = torch.arange(8.0).reshape(2, 4)
    torch.testing.assert_close(node.forward("walk", None, x=x)["y"][0], 2 * x)
