"""``test/modular/test_partition_done_report.py`` on the Rust runtime, read off
the conductor's real inbox."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")
# the shared script lives beside the Python-runtime test; "test" itself can
# resolve to the standard library's package, so import it by directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular"))

import pytest

pytest.importorskip("mstar_rust")

from mstar_rust import ZmqCommunicator
from test_delivery_chunk_counts import RUNTIME_KWARGS, WORKER, tensor_info, worker_graphs  # noqa: E402
from test_partition_done_report import EXPECTED, drive  # noqa: E402

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.communication.wire import decode
from mstar.graph.runtime import rust as rust_runtime
from mstar.utils.ipc_format import ConductorMessageType


def _reports(box):
    out = []
    raw = box.recv_timeout(1000)
    while raw is not None:
        message = decode(raw)
        if message.message_type == ConductorMessageType.WORKER_GRAPHS_DONE:
            out.append(message.body)
        raw = box.recv_timeout(50)
    return out


@pytest.mark.parametrize("speculative", [False, True], ids=["real", "speculative"])
def test_partition_done_rides_the_next_real_report(speculative, tmp_path):
    # the api server's inbox only has to exist: the prefill emits to the client
    inboxes = {peer: ZmqCommunicator(peer, str(tmp_path)) for peer in ("conductor", "api_server")}
    book = RustTensorBookkeeping()
    runtime = rust_runtime.RustGraphRuntime(
        my_worker_graphs=worker_graphs(1), bookkeeping=book,
        communicator=SimpleNamespace(_inner=ZmqCommunicator(WORKER, str(tmp_path))),
        **RUNTIME_KWARGS,
    )

    def put(uuid):
        book.put_tensor(uuid, tensor_info(uuid))
        book.increment_ref(uuid, 1)  # the worker's safety hold

    drive(runtime, put, speculative)
    assert [r.partition_done for r in _reports(inboxes["conductor"])] == EXPECTED[speculative]
