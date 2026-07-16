"""RFC #130 Step 1: ``ZMQCommunicator`` over the Rust transport vendored in
``rust/`` (``mstar_rust.ZmqCommunicator``) — drop-in for the pyzmq class:
same constructor, methods, endpoints, and pickle wire, so wrapped and
unwrapped entities interoperate and migration can proceed one process at a
time. The codec is a :class:`Codec` (pickle default; msgpack for
cross-language edges); ``register_event_for_poll`` forwards the eventfd to
the Rust poller; a readiness poll buffers any consumed message so nothing is
dropped or reordered. Build: ``maturin develop --release`` in ``rust/``
(see ``docs/installation.rst``)."""

from __future__ import annotations

import logging
import os
import pickle
import time
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


class MsgpackCodec(Codec):
    """msgpack is language-neutral: edges using it can terminate in (future)
    Rust processes — the migration's target wire. Both endpoints of an edge
    must use the same codec. Requires the ``msgpack`` package."""

    @staticmethod
    def encode(msg) -> bytes:
        import msgpack

        return msgpack.packb(msg, default=str)

    @staticmethod
    def decode(data: bytes):
        import msgpack

        return msgpack.unpackb(data, raw=False)


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

    def get_all_new_messages(
        self, blocking: bool = False, timeout_s: float | None = None,
    ) -> list:
        if blocking and not self._buffered:
            # Wait until a message is readable before draining — same
            # semantics as the pyzmq communicator: a registered wakeup
            # event also ends the wait, so a completed compute future can
            # interrupt a blocking receive. `timeout_s` bounds the wait
            # (None = wait indefinitely); on expiry, drain whatever is there.
            deadline = None if timeout_s is None else (
                time.monotonic() + timeout_s)
            while self._poll_once(50) == "timeout":
                if deadline is not None and time.monotonic() >= deadline:
                    break
        messages = [self.codec.decode(b) for b in self._buffered]
        self._buffered.clear()
        # One FFI call for the whole queued batch (vs try_recv per message).
        messages.extend(self.codec.decode(b) for b in self._inner.drain())
        return messages
