"""``test/modular/test_delivery_chunk_counts.py`` on the Rust runtime: the
frames it actually sends (read off real inboxes) must let the conductor count
every chunk, so the API server holds the request until the last one arrives.
Includes the inline-token path, which only the Rust runtime takes.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")
# the shared scenario lives beside the Python-runtime test; "test" itself can
# resolve to the standard library's package, so import it by directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular"))

import pytest

pytest.importorskip("mstar_rust")

from mstar_rust import ZmqCommunicator
from test_delivery_chunk_counts import (  # noqa: E402
    RUNTIME_KWARGS,
    SHAPES,
    WORKER,
    assert_holds_until_last_chunk,
    drive,
    tensor_info,
    worker_graphs,
)

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.api_server.request_types import ResultTokens
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.communication.wire import decode
from mstar.graph.runtime import rust as rust_runtime
from mstar.utils.ipc_format import ConductorMessageType


def _collect(box, first_wait_ms=1000):
    out = []
    first = box.recv_timeout(first_wait_ms)
    if first is None:
        return out
    out.append(decode(first))
    while (more := box.recv_timeout(50)) is not None:
        out.append(decode(more))
    return out


@pytest.mark.parametrize("inline", [False, True], ids=["tensor", "inline"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_request_is_held_until_its_last_chunk(shape, inline, tmp_path):
    prefills, decode_steps = SHAPES[shape]
    inboxes = {peer: ZmqCommunicator(peer, str(tmp_path)) for peer in ("conductor", "api_server")}
    book = RustTensorBookkeeping()
    runtime = rust_runtime.RustGraphRuntime(
        my_worker_graphs=worker_graphs(decode_steps), bookkeeping=book,
        communicator=SimpleNamespace(_inner=ZmqCommunicator(WORKER, str(tmp_path))),
        **RUNTIME_KWARGS,
    )

    def put(uuid):
        book.put_tensor(uuid, tensor_info(uuid))
        book.increment_ref(uuid, 1)  # the worker's safety hold

    drive(runtime, put, prefills, decode_steps, inline=inline)
    chunks = [m.body for m in _collect(inboxes["api_server"])]
    reports = [
        m.body for m in _collect(inboxes["conductor"])
        if m.message_type == ConductorMessageType.WORKER_GRAPHS_DONE
    ]
    assert len(chunks) == prefills + decode_steps
    tokens = sum(isinstance(c, ResultTokens) for c in chunks)
    assert tokens == (decode_steps if inline else 0)
    assert_holds_until_last_chunk(chunks, reports)
