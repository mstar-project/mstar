"""DRAFT (RFC #130 Step 1): ``ZMQCommunicator`` as a thin wrapper over the
mstar-rs Rust communicator (``mstar_rs._core.ZmqCommunicator``).

Drop-in for :class:`mstar.communication.communicator.ZMQCommunicator` — same
constructor, same methods, same semantics — with the transport moved to Rust:

* **Wire-compatible with the existing pyzmq communicator.** The Rust transport
  moves opaque byte frames (no added framing) over the same endpoints
  (``ipc://{prefix}/{id}.ipc`` / the same TCP host+port scheme), and the
  default codec is pickle — so a wrapped entity talks to unwrapped pyzmq
  entities in both directions. Migration can proceed one process at a time.
* **encode/decode seam.** ``codec`` is a ``(dumps, loads)`` pair defaulting to
  pickle (today's wire). Swapping to msgpack later is a codec change on both
  ends of an edge, never a transport change.
* **eventfd wakeup.** ``register_event_for_poll`` forwards the ``EventWakeup``
  fd to the Rust poller (``register_wakeup_fd``): a completed compute future
  wakes ``wait_for_work`` / ``poll_for_messages`` immediately, not on the poll
  timeout. Drain semantics are identical (the event is drained here, in
  Python, exactly as before).
* **Readiness without consumption.** The pyzmq poller reports "a message is
  readable" without receiving it; the Rust ``recv_or_wake`` consumes. The
  wrapper bridges the two with an internal deque: a message consumed during a
  poll is buffered and handed out by the next ``get_all_new_messages`` — no
  message is ever dropped or reordered.

Requires the vendored extension: ``maturin develop`` (or ``pip install .``) in ``rust/``.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections import deque

from mstar_rust import ZmqCommunicator as _RustZmq

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.event import EventWakeup

logger = logging.getLogger(__name__)

#: The encode/decode seam. Pickle matches today's ``send_pyobj`` wire, so a
#: wrapped entity interoperates with unwrapped pyzmq entities. Migrate an edge
#: to msgpack by giving both endpoints a msgpack codec.
PickleCodec = (pickle.dumps, pickle.loads)


class RustZMQCommunicator(BaseCommunicator):
    """The pyzmq ``ZMQCommunicator`` surface over the Rust transport."""

    def __init__(
        self,
        my_id: str,
        push_ids: list[str],
        protocol: CommProtocol = CommProtocol.IPC,
        ipc_socket_path_prefix: str = "/tmp/mstar/",
        codec=PickleCodec,
    ):
        transport = os.getenv("MSTAR_ZMQ_TRANSPORT", protocol.value).upper()
        self.protocol = CommProtocol(transport)
        self.my_id = my_id
        self.ipc_socket_path_prefix = ipc_socket_path_prefix
        self._dumps, self._loads = codec
        self.event: EventWakeup | None = None
        # Messages consumed by a readiness poll, awaiting get_all_new_messages.
        self._buffered: deque = deque()

        if self.protocol == CommProtocol.IPC:
            os.makedirs(ipc_socket_path_prefix, exist_ok=True)
            self._inner = _RustZmq(my_id, ipc_socket_path_prefix)
        elif self.protocol == CommProtocol.TCP:
            self._inner = _RustZmq.bind_endpoint(my_id, self._endpoint(my_id))
            # TCP has no directory scheme: register every peer explicitly
            # (lazily extended in send() for peers not known up front).
            self._registered: set[str] = set()
            for peer in push_ids:
                if peer != my_id:
                    self._register(peer)
        else:
            raise NotImplementedError(f"Protocol {protocol} not yet supported yet")

    # -- endpoint scheme (identical to the pyzmq communicator's) ------------

    def _endpoint(self, entity_id: str) -> str:
        if self.protocol == CommProtocol.IPC:
            return f"ipc://{self.ipc_socket_path_prefix}/{entity_id}.ipc"
        host = os.getenv("MSTAR_ZMQ_TCP_HOST", "127.0.0.1")
        return f"tcp://{host}:{self._tcp_port(entity_id)}"

    @staticmethod
    def _tcp_port(entity_id: str) -> int:
        base_port = int(os.getenv("MSTAR_ZMQ_TCP_BASE_PORT", "19000"))
        if entity_id == "api_server":
            return base_port
        if entity_id == "conductor":
            return base_port + 1
        if entity_id == "api_server_preprocess_worker":
            return base_port + 2
        if entity_id.startswith("worker_"):
            rank = entity_id.removeprefix("worker_")
            if rank.isdigit():
                return base_port + 100 + int(rank)
        return base_port + 1000 + (sum(entity_id.encode("utf-8")) % 1000)

    def _register(self, entity_id: str) -> None:
        if entity_id not in self._registered:
            self._inner.register_peer(entity_id, self._endpoint(entity_id))
            self._registered.add(entity_id)

    # -- the wrapped surface -------------------------------------------------

    def register_event_for_poll(self, event: EventWakeup) -> None:
        self._inner.register_wakeup_fd(event.fd)
        self.event = event

    def _poll_once(self, timeout_ms: int) -> None:
        """One wake-aware poll. A consumed message goes to the buffer; a
        wakeup drains the event (same place the pyzmq path drains it)."""
        kind, payload = self._inner.recv_or_wake(timeout_ms)
        if kind == "msg":
            self._buffered.append(payload)
        elif kind == "wake" and self.event is not None:
            self.event.drain()

    def wait_for_work(self, timeout_ms: int = 50) -> None:
        if self._buffered:
            return  # work is already waiting
        self._poll_once(timeout_ms)

    def poll_for_messages(self, timeout_ms: int = 20) -> bool:
        """Block up to ``timeout_ms`` for a readable message; True when one is
        available (buffered here, delivered by ``get_all_new_messages``)."""
        if self._buffered:
            return True
        self._poll_once(timeout_ms)
        return bool(self._buffered)

    def send(self, entity_id: str, msg) -> None:
        logger.debug("%s to send a message %s to entity %s", self.my_id, str(msg), entity_id)
        if self.protocol == CommProtocol.TCP:
            self._register(entity_id)
        self._inner.send(entity_id, self._dumps(msg))

    def get_all_new_messages(self, blocking: bool = False) -> list:
        messages = [self._loads(b) for b in self._buffered]
        self._buffered.clear()
        while (b := self._inner.try_recv()) is not None:
            messages.append(self._loads(b))
        return messages
