"""DRAFT (RFC #130 Step 1): ``ZMQCommunicator`` as a thin wrapper over the
mstar-rs Rust communicator (``mstar_rs._core.ZmqCommunicator``).

Drop-in for :class:`mstar.communication.communicator.ZMQCommunicator` — same
constructor, same methods, same semantics — with the transport moved to Rust:

* **Wire-compatible with the existing pyzmq communicator.** The Rust transport
  moves opaque byte frames (no added framing) over the same endpoints
  (``ipc://{prefix}/{id}.ipc`` / the same TCP host+port scheme), and the
  default codec is pickle — so a wrapped entity talks to unwrapped pyzmq
  entities in both directions. Migration can proceed one process at a time.
* **encode/decode seam.** ``codec`` is a :class:`Codec` (mirroring the Rust
  ``Codec`` trait) defaulting to :class:`PickleCodec` (today's wire).
  Swapping to msgpack later is a codec change on both ends of an edge,
  never a transport change.
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

Requires the vendored extension — from ``rust/``: ``maturin develop --release``
(or ``pip install .``). Use ``--release``: debug builds cost real latency on
the hot receive path. See ``docs/installation.rst``.
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

class Codec:
    """The encode/decode seam, mirroring the Rust ``Codec`` trait::

        pub trait Codec<M> {
            fn encode(msg: &M) -> Result<Vec<u8>, CommError>;
            fn decode(bytes: &[u8]) -> Option<M>;
        }

    The transport never looks inside the bytes; migrating an edge to another
    encoding (e.g. msgpack) means giving both endpoints that codec — never a
    transport change.
    """

    @staticmethod
    def encode(msg) -> bytes:
        raise NotImplementedError

    @staticmethod
    def decode(data: bytes):
        raise NotImplementedError


class PickleCodec(Codec):
    """Pickle matches today's ``send_pyobj`` wire, so a wrapped entity
    interoperates with unwrapped pyzmq entities in both directions."""

    encode = staticmethod(pickle.dumps)
    decode = staticmethod(pickle.loads)


class RustZMQCommunicator(BaseCommunicator):
    """The pyzmq ``ZMQCommunicator`` surface over the Rust transport."""

    def __init__(
        self,
        my_id: str,
        push_ids: list[str],
        protocol: CommProtocol = CommProtocol.IPC,
        ipc_socket_path_prefix: str = "/tmp/mstar/",
        codec: type[Codec] = PickleCodec,
    ):
        transport = os.getenv("MSTAR_ZMQ_TRANSPORT", protocol.value).upper()
        self.protocol = CommProtocol(transport)
        self.my_id = my_id
        self.ipc_socket_path_prefix = ipc_socket_path_prefix
        self.codec = codec
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

    # endpoint scheme: `_endpoint` / `_tcp_port` come from BaseCommunicator
    # (shared with the pyzmq ZMQCommunicator).

    def _register(self, entity_id: str) -> None:
        if entity_id not in self._registered:
            self._inner.register_peer(entity_id, self._endpoint(entity_id))
            self._registered.add(entity_id)

    # -- the wrapped surface -------------------------------------------------

    def register_event_for_poll(self, event: EventWakeup) -> None:
        self._inner.register_wakeup_fd(event.fd)
        self.event = event

    def _poll_once(self, timeout_ms: int) -> str:
        """One wake-aware poll; returns ``"msg"`` / ``"wake"`` / ``"timeout"``.
        A consumed message goes to the buffer; a wakeup drains the event
        (same place the pyzmq path drains it)."""
        kind, payload = self._inner.recv_or_wake(timeout_ms)
        if kind == "msg":
            self._buffered.append(payload)
        elif kind == "wake" and self.event is not None:
            self.event.drain()
        return kind

    def wait_for_work(self, timeout_ms: int = 50) -> None:
        if self._buffered:
            return  # work is already waiting
        self._poll_once(timeout_ms)

    def poll_for_messages(self, timeout_ms: int = 20) -> bool:
        """Block until a message is readable, a registered wakeup event
        fires, or ``timeout_ms`` elapses — whichever comes first. True when
        a message is available (buffered here, delivered by
        ``get_all_new_messages``); a wakeup ends the poll early with False
        (the event is drained, exactly as in ``wait_for_work``)."""
        if self._buffered:
            return True
        self._poll_once(timeout_ms)
        return bool(self._buffered)

    def send(self, entity_id: str, msg) -> None:
        logger.debug("%s to send a message %s to entity %s", self.my_id, str(msg), entity_id)
        if self.protocol == CommProtocol.TCP:
            self._register(entity_id)
        self._inner.send(entity_id, self.codec.encode(msg))

    def get_all_new_messages(self, blocking: bool = False) -> list:
        if blocking and not self._buffered:
            # Wait until a message is readable before draining — same
            # semantics as the pyzmq communicator: a registered wakeup
            # event also ends the wait, so a completed compute future can
            # interrupt a blocking receive.
            while self._poll_once(50) == "timeout":
                pass
        messages = [self.codec.decode(b) for b in self._buffered]
        self._buffered.clear()
        while (b := self._inner.try_recv()) is not None:
            messages.append(self.codec.decode(b))
        return messages
