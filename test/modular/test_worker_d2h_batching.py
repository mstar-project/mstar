"""The stop check's device-to-host copies are batched: one copy per kind of
output per step, rows get views, and a tensor under two names goes once."""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, ".")

import pytest
import torch

from mstar.worker.worker import Worker

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device to copy from")


def test_grouping_dedupes_aliases_and_groups_by_dtype_and_shape() -> None:
    a, b, c = torch.tensor([1]), torch.tensor([2]), torch.zeros(2, 3)
    outputs = {
        "r0": {"new_token": [a], "text_inputs": [a], "code": [c], "meta": "x"},
        "r1": {"new_token": [b], "text_inputs": [b]},
        "r2": "not a dict",
    }
    groups = Worker._group_device_tensors(outputs, device_type="cpu")
    assert [[id(t) for t in g] for g in groups.values()] == [[id(a), id(b)], [id(c)]]
    assert list(groups) == [(torch.int64, (1,)), (torch.float32, (2, 3))]

    ca, cb = torch.tensor([10]), torch.tensor([20])
    out = Worker._substitute_tensors(outputs, {id(a): ca, id(b): cb})
    assert out["r0"]["new_token"][0] is ca and out["r0"]["text_inputs"][0] is ca
    assert out["r1"]["new_token"][0] is cb and out["r0"]["code"][0] is c
    assert out["r0"]["meta"] == "x" and out["r2"] == "not a dict"


def _fake_worker() -> Worker:
    w = Worker.__new__(Worker)
    w.device = torch.device("cuda")
    w._d2h_stream = None
    w._pinned_d2h_buffers = defaultdict(list)
    return w


@requires_cuda
def test_prematerialize_copies_each_kind_once_and_rows_read_their_own_values() -> None:
    w = _fake_worker()
    rows = [torch.tensor([7 * i], device="cuda") for i in range(5)]
    code = torch.full((3,), 2.5, device="cuda")
    outputs = {f"r{i}": {"new_token": [t], "text_inputs": [t]} for i, t in enumerate(rows)}
    outputs["r0"]["code"] = [code]
    event = torch.cuda.Event()
    event.record()
    cpu = Worker._prematerialize_for_check_stop(w, outputs, event)
    for i in range(5):
        assert not cpu[f"r{i}"]["new_token"][0].is_cuda
        assert int(cpu[f"r{i}"]["new_token"][0].item()) == 7 * i
        assert cpu[f"r{i}"]["text_inputs"][0] is cpu[f"r{i}"]["new_token"][0]
    assert torch.equal(cpu["r0"]["code"][0], code.cpu())
    # one pinned buffer for the five tokens (stacked), one for the code
    assert sorted(len(v) for v in w._pinned_d2h_buffers.values()) == [1, 1]
    assert ("check_stop", torch.int64, (5, 1)) in w._pinned_d2h_buffers
    # the GPU outputs themselves are untouched
    assert all(outputs[f"r{i}"]["new_token"][0].is_cuda for i in range(5))
