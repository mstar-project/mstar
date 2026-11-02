"""A ``RustGraphRuntime`` for a test: a real bookkeeper and, when the test reads
what the runtime sends, a real transport with an inbox per peer."""
from __future__ import annotations

from types import SimpleNamespace

from mstar_rust import ZmqCommunicator

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.tensor_store import RustTensorBookkeeping, TensorStore
from mstar.communication.wire import decode
from mstar.graph.runtime.rust import RustGraphRuntime

_END_OF_READ = b"end of read"


class Inboxes:
    """Every peer's inbox for frames ``worker_id`` sends.

    ``sent`` is what has landed so far, as ``(peer, message)``, decoded after
    the real transport. Pass ``communicator`` to the runtime.
    """

    def __init__(self, worker_id: str, path, peers=("conductor", "api_server")):
        self._worker = ZmqCommunicator(worker_id, str(path))
        self._boxes = {peer: ZmqCommunicator(peer, str(path)) for peer in peers}
        self.communicator = SimpleNamespace(_inner=self._worker)
        self._landed: list[tuple[str, object]] = []

    @property
    def sent(self) -> list[tuple[str, object]]:
        # One socket keeps its order per peer, so a marker sent now ends each
        # read without guessing how long a quiet peer needs.
        for peer, box in self._boxes.items():
            self._worker.send(peer, _END_OF_READ)
            while (frame := box.recv_timeout(10_000)) != _END_OF_READ:
                assert frame is not None, f"the end marker to {peer} never arrived"
                self._landed.append((peer, decode(frame)))
        return self._landed

    def take(self) -> list[tuple[str, object]]:
        """What landed since the last ``take``."""
        landed, self._landed = self.sent, []
        return landed

    def bodies(self, peer: str) -> list:
        return [message.body for entity, message in self.sent if entity == peer]


def build_runtime(*, bookkeeping=None, inboxes: Inboxes | None = None, **kwargs) -> RustGraphRuntime:
    """The runtime, plus ``tensor_store`` over the bookkeeper it holds a share
    of, where a test stages outputs as the worker's prologue does."""
    book = RustTensorBookkeeping() if bookkeeping is None else bookkeeping
    runtime = RustGraphRuntime(
        bookkeeping=book, communicator=inboxes.communicator if inboxes else None, **kwargs,
    )
    runtime.tensor_store = TensorStore(bookkeeping=book)
    return runtime
