"""The worker stores a batch's outputs in one call, then refills each request's
output edges from the flat columns that call returns.

What has to hold: every edge gets exactly its own request's tensors for its
signal, in mint order, and gets the descriptor the store holds -- not a copy --
so the arena's in-place ``shm_segment`` stamp at registration reaches the edge
that is about to go on the wire.
"""
from types import SimpleNamespace

import torch

from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.graph.base import GraphEdge
from mstar.worker.worker import Worker


class _NullCommunicator:
    def send(self, *args, **kwargs):
        pass


def _worker(tmp_path):
    w = Worker.__new__(Worker)
    w.tensor_manager = SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=_NullCommunicator(), shm_dir=str(tmp_path),
    )
    return w


def _batch(*rids):
    def node():
        return SimpleNamespace(outputs=[
            GraphEdge(name="h", next_node="a"),
            GraphEdge(name="tok", next_node="b"),
            GraphEdge(name="h", next_node="c"),
        ])
    return SimpleNamespace(node_objects={rid: node() for rid in rids})


def test_output_signals_come_from_the_edges_in_order_and_once():
    assert Worker._output_signals(_batch(0, 1)) == ["h", "tok"]
    assert Worker._output_signals(_batch()) == []


def test_flat_columns_regroup_per_request_and_signal(tmp_path):
    w = _worker(tmp_path)
    signals = ["h", "tok"]
    outputs = {
        0: {"h": [torch.zeros(2), torch.ones(2)], "tok": [torch.tensor([7])]},
        1: {"tok": [torch.tensor([9])], "unrouted": [torch.zeros(1)]},
    }
    stored = w.tensor_manager.store_and_return_tensor_info_batch(
        [0, 1], outputs, signals,
    )
    grouped = w._output_infos_by_signal(stored, signals)

    assert set(grouped) == {0, 1}
    assert [len(grouped[0]["h"]), len(grouped[0]["tok"])] == [2, 1]
    assert list(grouped[1]) == ["tok"], "no routed h for 1; unrouted stays out"
    # Mint order within a signal, and each descriptor names its own tensor.
    h0, h1 = grouped[0]["h"]
    tm = w.tensor_manager
    assert torch.equal(tm.get_tensor(h0.uuid), torch.zeros(2))
    assert torch.equal(tm.get_tensor(h1.uuid), torch.ones(2))
    assert torch.equal(tm.get_tensor(grouped[1]["tok"][0].uuid), torch.tensor([9]))


def test_edges_carry_the_stores_own_descriptor(tmp_path):
    """Registration stamps the store's descriptor; the edge must be that same
    object or the stamp never reaches the wire."""
    w = _worker(tmp_path)
    stored = w.tensor_manager.store_and_return_tensor_info_batch(
        [0], {0: {"h": [torch.zeros(2)]}}, ["h"],
    )
    info = w._output_infos_by_signal(stored, ["h"])[0]["h"][0]
    assert info is w.tensor_manager.tensor_store.get_info(info.uuid)


def test_an_output_no_signal_carries_is_not_stored(tmp_path):
    """No edge will ever read it, and a stored tensor with no references is
    only freed when its request is torn down -- storing it would hold the
    memory for the rest of the request."""
    w = _worker(tmp_path)
    store = w.tensor_manager.tensor_store
    before = set(store._tensors)
    stored = w.tensor_manager.store_and_return_tensor_info_batch(
        [0], {0: {"h": [torch.zeros(2)], "unrouted": [torch.ones(3)]}}, ["h"],
    )
    assert set(store._tensors) - before == set(stored.flat_uuids)
    assert store.get_all_uuids(0) == stored.flat_uuids
