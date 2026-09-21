"""The batched-output contract, and the D2H it exists to collapse.

A step's stop check reads one scalar per request. Addressed per rid that is a
device-to-host copy each — 48 of them at a decode batch of 16, every one a
launch plus a completion the host waits on, carrying a few bytes. A submodule
that hands the whole batch tensor over instead pays one.

The risk in that is silent: if row i stops belonging to request i, the stop
check reads someone else's token and requests end at the wrong place while
everything still looks like it works. So the row mapping is what these pin.
"""

from __future__ import annotations

import pytest
import torch

from mstar.model.submodule_base import BatchedModelOutput
from mstar.worker.worker import Worker


def test_coerce_splits_on_the_dunder_convention():
    out = BatchedModelOutput.coerce(
        {"r0": {"new_token": [1]}, "__packed__": "batch", "r1": {}}
    )
    assert set(out.per_rid_outputs) == {"r0", "r1"}
    assert out.packed_outputs == {"__packed__": "batch"}
    # already coerced is a no-op, not a re-split
    assert BatchedModelOutput.coerce(out) is out


def test_readers_are_safe_on_a_default_instance():
    """Both dicts are always dicts — a None would mean every reader on the
    step's critical path has to guard before touching them."""
    out = BatchedModelOutput()
    assert out.get("nobody") is None
    assert out.get("nobody", "fallback") == "fallback"
    assert out.pop("nobody") is None
    assert out.get_check_stop_input() == {}
    assert out.clone_check_stop_buffers() is None


def test_update_accepts_the_old_dict_contract():
    """A producer that still returns a plain dict is valid; `coerce` is the
    whole point of having one entry point."""
    out = BatchedModelOutput(per_rid_outputs={"r0": {"a": 1}})
    out.update({"r1": {"b": 2}, "__packed__": 7})
    assert set(out.per_rid_outputs) == {"r0", "r1"}
    assert out.packed_outputs == {"__packed__": 7}


def test_pop_leaves_batch_addressed_buffers_alone():
    """A dropped request is removed from the batch, not from a row-addressed
    tensor — punching a hole in one would shift every later row."""
    buf = torch.arange(4)
    out = BatchedModelOutput(
        per_rid_outputs={"r0": {}, "r1": {}}, check_stop_buffers={"tok": buf},
    )
    out.pop("r0")
    assert set(out.per_rid_outputs) == {"r1"}
    assert out.check_stop_buffers["tok"] is buf


def test_clone_detaches_and_keeps_non_tensors():
    """The buffers are a captured graph's, overwritten by the next replay, so
    the clone is what makes them safe to read later."""
    src = torch.arange(3)
    out = BatchedModelOutput(
        check_stop_buffers={"tok": src, "nested": {"x": [src], "flag": "keep"}},
    )
    cloned = out.clone_check_stop_buffers()
    src.fill_(99)
    assert cloned["tok"].tolist() == [0, 1, 2], "clone must not alias the buffer"
    assert cloned["nested"]["x"][0].tolist() == [0, 1, 2]
    assert cloned["nested"]["flag"] == "keep", "non-tensors must survive"


def test_check_stop_falls_back_to_per_rid():
    per_rid = {"r0": {"new_token": [torch.tensor([5])]}}
    out = BatchedModelOutput(per_rid_outputs=per_rid)
    assert out.get_check_stop_input() is per_rid
    out.check_stop_buffers = {"new_token": torch.tensor([5])}
    assert out.get_check_stop_input() is out.check_stop_buffers


class _StubWorker:
    """Just enough of Worker to drive the batched D2H."""

    _pinned_d2h_buffers: dict = {}

    def __init__(self):
        from collections import defaultdict
        self._pinned_d2h_buffers = defaultdict(list)

    _get_pinned_d2h_buffer = Worker._get_pinned_d2h_buffer
    _prematerialize_batched = Worker._prematerialize_batched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_batched_prematerialize_keeps_row_to_request_order():
    """Row i must land on request i — the bug this would hide is requests
    stopping on each other's tokens."""
    rids = [f"r{i}" for i in range(6)]
    # a padded replay leaves rows past the real ones; they must be ignored
    tokens = torch.arange(100, 110, device="cuda")
    worker = _StubWorker()
    got = worker._prematerialize_batched(
        {"new_token": tokens}, torch.cuda.Stream(), rids,
    )
    assert list(got) == rids
    for i, rid in enumerate(rids):
        assert got[rid]["new_token"][0].item() == 100 + i
        assert not got[rid]["new_token"][0].is_cuda, "must land on the host"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_batched_matches_what_the_per_rid_walk_would_give():
    rids = [f"r{i}" for i in range(4)]
    tokens = torch.tensor([7, 8, 9, 10], device="cuda")
    worker = _StubWorker()
    got = worker._prematerialize_batched(
        {"new_token": tokens}, torch.cuda.Stream(), rids,
    )
    want = {
        rid: {"new_token": [tokens[i : i + 1].cpu()]}
        for i, rid in enumerate(rids)
    }
    for rid in rids:
        assert got[rid]["new_token"][0].item() == want[rid]["new_token"][0].item()


def test_qwen3_5_hands_over_a_batch_tensor():
    """The producer side: without this the worker silently keeps paying the
    per-request copies."""
    import inspect

    from mstar.model.qwen3_5 import submodules

    src = inspect.getsource(submodules.LLMSubmodule.forward_batched)
    assert "check_stop_buffers" in src
    assert "BatchedModelOutput" in src


def test_rows_map_through_the_forward_order_not_the_batch_list():
    """The worker rewrites its batch's request list between the forward and
    the stop check (dropping requests whose loops stopped, and until recently
    through a set, so in hash order). Rows have to be sliced by the order the
    forward ran, which the engine stamps on the output."""
    host = {"new_token": torch.tensor([[248046], [53031]])}
    forward_order = ["p1", "p3"]
    out = Worker._rows_to_per_rid(host, forward_order)
    assert out["p1"]["new_token"][0].item() == 248046
    assert out["p3"]["new_token"][0].item() == 53031
    # the batch list after a stop dropped p1: row 0 still belongs to p1, not p3
    shrunk = Worker._rows_to_per_rid(host, ["p3"])
    assert shrunk["p3"]["new_token"][0].item() == 248046, (
        "slicing by the shrunk list is the bug: p3 would stop on p1's token"
    )


def test_row_request_ids_default_none_and_survive_a_single_merge():
    a = BatchedModelOutput(per_rid_outputs={"a": {}, "b": {}})
    assert a.row_request_ids is None
    b = BatchedModelOutput(
        per_rid_outputs={"a": {}, "b": {}},
        check_stop_buffers={"new_token": torch.tensor([1, 2])},
        row_request_ids=("a", "b"),
    )
    a.update(b)
    assert a.row_request_ids == ("a", "b")
    assert a.check_stop_buffers["new_token"].tolist() == [1, 2]


def test_merging_two_row_addressed_outputs_drops_the_buffers():
    """Two forwards merged into one output cannot share one row-addressed
    buffer, so the merge keeps neither and the stop check falls back to the
    per-rid outputs, which are always present."""
    merged = BatchedModelOutput(per_rid_outputs={"a": {}, "b": {}})
    merged.update(BatchedModelOutput(
        per_rid_outputs={"a": {"new_token": [torch.tensor([1])]}},
        check_stop_buffers={"new_token": torch.tensor([1])}, row_request_ids=("a",),
    ))
    merged.update(BatchedModelOutput(
        per_rid_outputs={"b": {"new_token": [torch.tensor([2])]}},
        check_stop_buffers={"new_token": torch.tensor([2])}, row_request_ids=("b",),
    ))
    assert merged.check_stop_buffers is None and merged.row_request_ids is None
    assert merged.get_check_stop_input()["b"]["new_token"][0].item() == 2
