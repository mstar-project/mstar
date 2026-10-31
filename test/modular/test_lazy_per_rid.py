"""The stop check's per-rid host dicts are built only when something reads
them."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from mstar.model.submodule_base import BatchedModelOutput
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import PendingBatch, Worker, _LazyHostOutputs, _LazyPerRid


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


def test_lazy_host_outputs_copy_only_the_names_read():
    """A stop check that reads the token copies the token alone, and one that
    reads nothing (an iteration-count stop) copies nothing."""
    source = {
        1: {"new_token": [torch.tensor([5])], "latents": [torch.ones(4, 4)]},
        2: {"new_token": [torch.tensor([6])], "latents": [torch.ones(4, 4)]},
        3: "not-a-dict",
    }
    copied = []

    def copy_name(name):
        copied.append(name)
        return {rid: [t.clone() for t in row[name]] for rid, row in source.items()
                if isinstance(row, dict) and name in row}

    lazy = _LazyHostOutputs(source, copy_name)
    row = lazy.get(1)
    assert row and "latents" in row and len(row) == 2 and set(row) == {"new_token", "latents"}
    assert copied == [], "presence, length and iteration must not copy"
    assert row["new_token"][0].item() == 5
    assert lazy[2]["new_token"][0].item() == 6
    assert copied == ["new_token"], "a name is copied once, for every row"
    assert lazy.get(3) == "not-a-dict" and lazy.get(4) is None
    assert set(lazy.landed()) == {1, 2}
    assert all(set(names) == {"new_token"} for names in lazy.landed().values())


class _ReachedStoreError(Exception):
    pass


@pytest.mark.parametrize("host_transport", [True, False], ids=["host", "device"])
def test_a_host_transport_gets_each_routed_output_in_one_copy(host_transport):
    """A host transport (SHM) gets every routed output already copied, one copy
    per name, rather than copying each tensor itself in the send; a device
    transport gets only what the stop check read."""
    source = {rid: {"new_token": [torch.tensor([rid])], "latents": [torch.ones(2)]} for rid in (1, 2)}
    copied = []

    def copy_name(name):
        copied.append(name)
        return {rid: [t.clone() for t in row[name]] for rid, row in source.items()}

    lazy = _LazyHostOutputs(source, copy_name)
    assert lazy[1]["new_token"][0].item() == 1  # the stop check's read
    landed = {}

    def store(rids, outputs, signals, cpu_tensors=None, **kwargs):
        landed.update(cpu_tensors)
        raise _ReachedStoreError

    worker = Worker.__new__(Worker)
    worker._phase_period = 0
    worker._graph_runtime = SimpleNamespace(supports_inline_emit=False)
    worker.tensor_manager = SimpleNamespace(
        needs_cpu_tensor=host_transport, store_and_return_tensor_info_batch=store,
    )
    pending = PendingBatch(
        batch=ScheduledBatch(
            node_name="n", graph_walk="w", request_to_worker_graph={1: 0, 2: 0},
            output_signals=["new_token", "latents"],
        ),
        node_batch=SimpleNamespace(completion_event=None), node_name="n", partition="p",
        graph_walk="w", future=Future(),
    )
    with pytest.raises(_ReachedStoreError):
        worker._store_and_route(pending, BatchedModelOutput(per_rid_outputs=source), [1, 2], ["w"], None, lazy)
    names = ["new_token", "latents"] if host_transport else ["new_token"]
    assert copied == names, "a name is copied once"
    assert {rid: list(row) for rid, row in landed.items()} == {1: names, 2: names}
