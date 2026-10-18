"""Engine._collect_outputs: the decode fast path (row outputs only, no per-rid
filtering) hands out the same per-rid row views as the generic walk, and the
stop buffer shares the row clone instead of taking a second copy."""
from __future__ import annotations

import torch

from mstar.engine.engine import Engine
from mstar.model.submodule_base import BatchedModelOutput, NodeSubmodule


class _Sub(NodeSubmodule):
    """The base filter: a no-op, which is what takes the fast path."""

    def __init__(self):
        pass

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _Filtering(_Sub):
    """Overrides the filter, so the generic per-rid walk has to run."""

    def filter_batched_output(self, request_info, outputs):
        return {k: v for k, v in outputs.items() if k != "aux"}


def _raw(bs=4, with_per_rid=False):
    tok = torch.arange(bs, dtype=torch.int64) + 100
    aux = torch.arange(bs * 2, dtype=torch.float32).reshape(bs, 2)
    raw = BatchedModelOutput(
        row_outputs={"new_token": tok, "aux": aux},
        check_stop_buffers={"new_token": tok},
    )
    if with_per_rid:
        raw.per_rid_outputs = {1: {"extra": [torch.tensor([7])]}}
    return raw


def _collect(sub, raw, rids):
    eng = Engine.__new__(Engine)
    outputs = {}
    row_clones, _ = eng._merge_per_rid(outputs, raw, rids, rids, sub, {})
    return outputs, row_clones


def _generic_reference(raw, rids, sub):
    """The per-rid walk as it was before the fast path."""
    row_clones = raw.clone_row_outputs(len(rids))
    outputs = {}
    for i, rid in enumerate(rids):
        candidates = {
            name: [t[i:i + 1]] for name, t in row_clones.items()
            if isinstance(t, torch.Tensor) and t.dim() and i < t.shape[0]
        }
        rid_out = sub.filter_batched_output(None, candidates)
        outputs[rid] = {k: v for k, v in rid_out.items()}
    return outputs


def _same(a, b):
    assert a.keys() == b.keys()
    for rid in a:
        assert a[rid].keys() == b[rid].keys(), rid
        for name in a[rid]:
            va, vb = a[rid][name], b[rid][name]
            assert len(va) == len(vb) == 1
            assert va[0].shape == vb[0].shape
            assert torch.equal(va[0], vb[0])


def test_fast_path_matches_the_generic_walk():
    rids = [10, 11, 12, 13]
    raw = _raw()
    fast, _ = _collect(_Sub(), raw, rids)
    _same(fast, _generic_reference(raw, rids, _Sub()))
    # row views, not copies: a row is a [1] view into the batch clone
    assert fast[11]["new_token"][0].shape == (1,) and fast[11]["new_token"][0].item() == 101
    assert fast[12]["aux"][0].shape == (1, 2)


def test_fast_path_handles_fewer_rows_than_requests_and_padding():
    rids = [1, 2]
    raw = _raw(bs=4)  # a padded bucket: four rows, two real requests
    fast, clones = _collect(_Sub(), raw, rids)
    assert set(fast) == {1, 2}
    assert clones["new_token"].shape == (2,), "narrowed to the real rows"
    assert fast[2]["new_token"][0].item() == 101


def test_per_rid_entries_or_a_filtering_submodule_take_the_generic_walk():
    rids = [1, 2, 3]
    raw = _raw(bs=3, with_per_rid=True)
    out, _ = _collect(_Sub(), raw, rids)
    assert "extra" in out[1] and torch.equal(out[1]["extra"][0], torch.tensor([7]))
    assert out[1]["extra"][0] is not raw.per_rid_outputs[1]["extra"][0], "per-rid entries are cloned"
    filt, _ = _collect(_Filtering(), _raw(bs=3), rids)
    assert all("aux" not in filt[r] and "new_token" in filt[r] for r in rids)


def test_stop_buffer_shares_the_row_clone():
    raw = _raw(bs=4)
    clones = raw.clone_row_outputs(3)
    stop = raw.clone_check_stop_buffers(reuse=clones)
    assert stop["new_token"] is clones["new_token"]
    assert stop["new_token"].shape == (3,)
    # without reuse, or for a buffer that is not a row output, it is still a fresh copy
    fresh = raw.clone_check_stop_buffers()
    assert fresh["new_token"] is not raw.check_stop_buffers["new_token"]
    assert torch.equal(fresh["new_token"], raw.check_stop_buffers["new_token"])
    other = BatchedModelOutput(
        row_outputs={"new_token": torch.zeros(2, dtype=torch.int64)},
        check_stop_buffers={"new_token": torch.ones(2, dtype=torch.int64)},
    )
    stop2 = other.clone_check_stop_buffers(reuse=other.clone_row_outputs(2))
    assert torch.equal(stop2["new_token"], torch.ones(2, dtype=torch.int64))


def test_rows_only_keeps_the_views_and_builds_no_per_rid_dicts():
    """A step whose outputs all stay on the device: the row views and the
    stop buffer are what its readers use, so no per-rid dict is built."""
    rids = [1, 2, 3, 4]
    raw = _raw(bs=4)
    eng = Engine.__new__(Engine)
    outputs = {}
    row_clones, row_views = eng._merge_per_rid(
        outputs, raw, rids, rids, _Sub(), {}, rows_only=True,
    )
    assert outputs == {}
    assert set(row_views) == {"new_token", "aux"} and len(row_views["new_token"]) == 4
    assert row_clones["new_token"].tolist() == [100, 101, 102, 103]
    # the ordinary path still fills them
    outputs = {}
    eng._merge_per_rid(outputs, _raw(bs=4), rids, rids, _Sub(), {})
    assert set(outputs) == set(rids)


def test_rows_only_collect_takes_no_clone():
    """On a rows-only step the gpu thread copies the rows to the host right
    behind the step, so the graph's own output buffer is handed over as is."""
    from types import SimpleNamespace

    rids = [1, 2, 3, 4]
    raw = _raw(bs=4)
    eng = Engine.__new__(Engine)
    out = eng._collect_outputs(
        SimpleNamespace(submodule=_Sub(), cuda_graph_runner=None), None, raw,
        [], {}, request_ids=rids, step_request_ids=tuple(rids), rows_only=True,
    )
    assert out.rows_only and out.per_rid_outputs == {} and out.row_views is None
    assert out.check_stop_buffers["new_token"] is raw.check_stop_buffers["new_token"]
    assert out.row_request_ids == (1, 2, 3, 4)
