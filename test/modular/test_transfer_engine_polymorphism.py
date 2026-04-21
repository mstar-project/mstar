"""TensorTransferEngine ABC contract tests.

Pins the contract that kv_store.py and TensorCommunicationManager depend on
across all engine subclasses (Mooncake, Local, NVSHMEM). Single-process,
no GPU required.
"""

import pytest
import torch

from mminf.communication.communicator import CommProtocol
from mminf.communication.tensors import (
    LocalTransferEngine,
    MooncakeTransferEngine,
    NVSHMEMTransferEngine,
    TensorTransferEngine,
)


def _try_make_mooncake() -> TensorTransferEngine | None:
    try:
        return MooncakeTransferEngine(
            hostname="localhost", protocol=CommProtocol.RDMA,
        )
    except Exception:
        return None  # mooncake not installed or no RDMA available


@pytest.mark.parametrize("engine_factory,expects_real_reader", [
    pytest.param(
        _try_make_mooncake, True,
        marks=pytest.mark.skipif(
            _try_make_mooncake() is None,
            reason="mooncake not installed or unavailable",
        ),
    ),
    (lambda: LocalTransferEngine(hostname="local-test"),    False),
    (lambda: NVSHMEMTransferEngine(my_entity_id="worker_0"), False),
])
def test_engine_satisfies_contract(engine_factory, expects_real_reader):
    """All TensorTransferEngine subclasses must implement the contract that
    kv_store.py and the manager base class depend on: register/unregister a
    real memory region, return the right async-reader shape, expose a
    non-empty session id."""
    engine = engine_factory()

    # Use a small real allocation so the Mooncake path doesn't reject
    # zero-length regions; no-op engines (Local, NVSHMEM) ignore the values.
    if torch.cuda.is_available() and not isinstance(engine, LocalTransferEngine):
        buf = torch.zeros(64, dtype=torch.uint8, device="cuda")
    else:
        buf = torch.zeros(64, dtype=torch.uint8)
    assert engine.register_memory(buf.data_ptr(), buf.nbytes) == 0
    assert engine.unregister_memory(buf.data_ptr()) == 0

    # Mooncake's AsyncMooncakeReader requires a CUDA device; no-op engines
    # ignore the argument.
    reader_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    reader = engine.get_async_reader(reader_device)
    assert (reader is not None) == expects_real_reader
    assert isinstance(engine.get_session_id(), str)
    assert engine.get_session_id()  # non-empty
