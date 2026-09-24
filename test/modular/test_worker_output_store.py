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
