"""The worker stores a batch's outputs in one call, for the output signals the
node's edges carry."""
import torch

from mstar.communication.tensors import SharedMemoryCommunicationManager
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


def test_new_token_counts_dedup_per_rid_not_across_rids(tmp_path):
    """A signal routed to two destinations appears twice in the batch with
    the same tensors, so it must be counted once -- but the de-dup is per
    request. Sharing one seen-set across rids silently drops every rid after
    the first, which reads as those requests emitting no tokens at all.
    """
    w = _worker(tmp_path)
    stored = w.tensor_manager.store_and_return_tensor_info_batch(
        [7, 9], {7: {"token": [torch.ones(3)]}, 9: {"token": [torch.ones(5)]}},
        ["token"],
    )
    signals = ["token"]
    # rid 7's token is on two edges; rid 9's on one. Columns are rid-major.
    flat_rids = [7, 7, 9]
    flat_uuids = [stored.flat_uuids[0], stored.flat_uuids[0],
                  stored.flat_uuids[1]]
    signal_idxs = [0, 0, 0]

    counts = w._count_new_tokens(
        [0, 1, 2], flat_rids, flat_uuids, signals, signal_idxs,
    )

    assert counts == {7: {"token": 3}, 9: {"token": 5}}
