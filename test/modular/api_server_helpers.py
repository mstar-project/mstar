"""An ``APIServer`` for tests, built without its ``__init__``.

``APIServer.__init__`` creates the upload dir, starts the ``PreprocessWorker``
and opens the ZMQ communicator. ``make_test_api_server`` skips those and calls
``APIServer._init_request_state``, the method ``__init__`` uses for the
in-memory request bookkeeping, so a field added there reaches every test that
builds its server here. ``test_api_server_helpers.py`` fails if the object
lacks any field a real ``APIServer`` has.

Import it by module name (``from api_server_helpers import ...``): on CI the
stdlib ``test`` package shadows the repo's ``test/`` directory.
"""
import select
import tempfile
import time
from pathlib import Path

from mstar.api_server.entrypoint import APIServer
from mstar.communication.event import EventWakeup


def make_test_api_server(**overrides) -> APIServer:
    """An ``APIServer`` with every field ``__init__`` sets and no processes,
    sockets or files. ``preprocess_worker``, ``communicator`` and
    ``_msg_thread`` are None unless passed in. ``overrides`` replace fields
    by name; a name ``APIServer`` does not have raises ``TypeError``.
    """
    server = object.__new__(APIServer)
    server._init_request_state()
    server.upload_dir = Path(tempfile.gettempdir())
    server.timeout_seconds = 5.0
    server.enable_nvtx = False
    server.log_stats = False
    server.log_stats_file = None
    server.metrics = None
    server.model = None
    server.model_name = "dummy"
    server.client_streams = frozenset()
    server.takes_stream_input = False
    server._model_config = None   # max_call_input_bytes reads it on first use
    server.preprocess_worker = None
    server.communicator = None
    server._msg_thread = None
    server._result_cpu = None  # never pins a test's thread
    server._process_cpus = frozenset()
    for name, value in overrides.items():
        if not hasattr(server, name):
            raise TypeError(f"APIServer has no field {name!r}")
        setattr(server, name, value)
    return server


class StandInSocket:
    """The communicator surface ``APIServer._process_messages`` uses, over a
    test's own ``messages`` (or an override of ``get_all_new_messages``). Its
    poll ends on the registered wakeup as the real communicators' does, and
    otherwise after at most ``tick_ms``: messages a test queues signal
    nothing."""

    def __init__(self, messages=lambda: [], tick_ms: float = 1.0):
        self._messages = messages
        self.tick_ms = tick_ms
        self.event: EventWakeup | None = None
        self.timeouts: list[int] = []  # each poll's timeout_ms, as the loop asked

    def get_all_new_messages(self) -> list:
        return self._messages()

    def register_event_for_poll(self, event: EventWakeup) -> None:
        self.event = event

    def poll_for_messages(self, timeout_ms: int = 20) -> bool:
        self.timeouts.append(timeout_ms)
        timeout_s = min(timeout_ms, self.tick_ms) / 1000
        if self.event is None:
            time.sleep(timeout_s)
        elif select.select([self.event.fd], [], [], timeout_s)[0]:
            self.event.drain()
        return False
