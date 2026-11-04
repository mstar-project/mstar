import collections
import fcntl
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum

from mstar.communication.event import EventWakeup

logger = logging.getLogger(__name__)

#: The ``mstar_rust`` extension version this tree expects (the vendored
#: ``rust/`` crate's version). Every process sends on the extension's
#: transport, so a mismatching install - e.g. a stale wheel after an upgrade -
#: is refused at startup rather than left to carry the mesh.
EXPECTED_MSTAR_RUST_VERSION = "0.1.1"


#: The entity whose endpoint stands for the whole deployment: the one every
#: deployment has exactly one of. Two deployments that agree on this endpoint
#: ARE the same deployment as far as anything keyed off it is concerned --
#: notably the tensor-file namespace in ``mstar.communication.tensors``.
DEPLOYMENT_ANCHOR_ENTITY = "conductor"


class CommProtocol(Enum):
    IPC = "IPC"
    TCP = "TCP"
    RDMA = "RDMA"
    SHM = "SHM"


class BaseCommunicator(ABC):
    @abstractmethod
    def send(self, entity_id: str, msg):
        """
        entity_id: worker_xyz, conductor, or api_server
        """
        pass

    def try_send_bytes(self, entity_id: str, data: bytes) -> bool:
        """Send a message already encoded with ``self.codec`` without
        blocking: False, with nothing sent, when the peer is at its high-water
        mark. A loop that must keep reading its own input while a peer is slow
        sends with this and retries later (see ``OutboxCommunicator``)."""
        raise NotImplementedError(f"{type(self).__name__} has no non-blocking send")

    @abstractmethod
    def get_all_new_messages(self) -> list:
        pass

    @abstractmethod
    def register_event_for_poll(self, event: EventWakeup) -> None:
        """Have ``poll_for_messages`` (and ``wait_for_work``) also end when
        ``event`` is signalled."""

    @abstractmethod
    def poll_for_messages(self, timeout_ms: int = 20) -> bool:
        """Block until a message is readable, the registered event fires, or
        ``timeout_ms`` elapses, whichever comes first. True when a message is
        waiting for ``get_all_new_messages``; a fired event is drained."""

    # -- endpoint scheme ------------------------------------------------------
    # Subclasses set ``self.protocol`` and ``self.ipc_socket_path_prefix``.

    def _endpoint(self, entity_id: str) -> str:
        if self.protocol == CommProtocol.IPC:
            return f"ipc://{self.ipc_socket_path_prefix}/{entity_id}.ipc"
        if self.protocol == CommProtocol.TCP:
            host = os.getenv("MSTAR_ZMQ_TCP_HOST", "127.0.0.1")
            return f"tcp://{host}:{self._tcp_port(entity_id)}"
        raise NotImplementedError(f"Protocol {self.protocol} not yet supported yet")

    def _lock_deployment(self, my_id: str) -> None:
        """Claim this deployment, so a second one on the same IPC prefix fails
        here instead of corrupting the first.

        Only the anchor entity locks, and only over IPC. TCP needs no lock: two
        deployments sharing a base port collide on ``bind`` already. IPC does
        not -- libzmq UNLINKS an existing socket file and binds a fresh one, so
        the second conductor comes up, silently steals every connection aimed
        at the first, and derives the same tensor-file namespace (uuids are
        per-entity counters, so both deployments then write, read and unlink
        each other's ``mstar_<ns>_worker_0_<n>``).

        An ``flock`` on a file beside the socket is what turns that into an
        error. It is released by the kernel when the process exits however it
        exits, so a crashed deployment leaves nothing to clean up; the fd is
        kept on the instance purely so it outlives this call.
        """
        if self.protocol != CommProtocol.IPC or my_id != DEPLOYMENT_ANCHOR_ENTITY:
            return
        path = self._endpoint(my_id).removeprefix("ipc://") + ".lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(
                f"another mstar deployment already holds {path}: its "
                f"{my_id} is bound to the same IPC prefix "
                f"({self.ipc_socket_path_prefix!r}). Two deployments cannot "
                "share a prefix -- the second would take over the first's "
                "sockets and tensor files. Give this one its own "
                "ipc_socket_path_prefix, or run it over TCP."
            ) from exc
        # Never unlinked: another process may already hold an fd on this inode,
        # and removing it would let a third lock a different one.
        self._deployment_lock_fd = fd

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

    # @abstractmethod
    # def get_session_id(self) -> str:
    #     pass


def make_communicator(*args, nonblocking_send: bool = False, **kwargs) -> BaseCommunicator:
    """Construct the process's communicator: ``RustZMQCommunicator``, on the
    vendored ``rust/`` extension (see ``communication/rust_communicator.py``).

    Raises when the extension is missing or is not the version this tree
    expects. ``nonblocking_send`` is for a communicator ``OutboxCommunicator``
    wraps, which needs ``try_send_bytes``: an extension built before
    ``ZmqCommunicator.try_send`` is refused for it.
    """
    try:
        import mstar_rust
    except ImportError as e:
        raise ImportError(
            "The control mesh runs on the mstar_rust extension, which this "
            "environment does not have. The Linux wheels of mstar-ai on PyPI "
            "carry it; from a checkout, build it with `maturin develop "
            "--release -m rust/Cargo.toml` (see docs/installation.rst)."
        ) from e
    from mstar.communication.rust_communicator import RustZMQCommunicator

    # A support bundle must be able to tell what a mesh was running. The path
    # says which copy loaded: a maturin-built mstar-rust package shadows the
    # module an mstar-ai Linux wheel ships.
    version = getattr(mstar_rust, "__version__", "<pre-versioning>")
    logger.info("control mesh transport: rust %s from %s", version, mstar_rust.__file__)
    if version != EXPECTED_MSTAR_RUST_VERSION:
        raise RuntimeError(
            f"mstar_rust version {version} at {mstar_rust.__file__} does not "
            f"match the expected {EXPECTED_MSTAR_RUST_VERSION}: in a checkout, "
            "rebuild it with `maturin develop --release -m rust/Cargo.toml`; in "
            "a pip install, upgrade mstar-ai and `pip uninstall mstar-rust` if "
            "a leftover build is shadowing the wheel's copy"
        )
    if nonblocking_send and not hasattr(mstar_rust.ZmqCommunicator, "try_send"):
        raise RuntimeError(
            f"mstar_rust {version} predates ZmqCommunicator.try_send, which the "
            "API server's preprocess thread sends with: rebuild it with "
            "`maturin develop --release -m rust/Cargo.toml`"
        )
    return RustZMQCommunicator(*args, **kwargs)


class OutboxCommunicator:
    """A communicator whose sends never block: a send the peer can't take now
    (at its high-water mark) is queued, in order behind any earlier one to the
    same peer, and retried by ``flush``.

    For a loop that peers block on while it sends to them (the API server's
    preprocess thread, against the conductor and the workers): if it blocked on
    a full peer it would stop reading the input that peer waits to deliver, and
    the two would wait on each other for good. Everything else is the wrapped
    communicator's.
    """

    def __init__(self, inner: BaseCommunicator):
        self._inner = inner
        self._outbox: dict[str, collections.deque[bytes]] = {}

    def send(self, entity_id: str, msg) -> None:
        self.send_each((entity_id,), msg)

    def send_each(self, entity_ids: Iterable[str], msg) -> None:
        """``send`` to each of ``entity_ids``, encoding ``msg`` once."""
        # Encoded here, queued or not: a message the codec refuses fails its
        # sender now, as a blocking send would, rather than at the head of the
        # queue where it would hold up every later message.
        data = self._inner.codec.encode(msg)
        for entity_id in entity_ids:
            backlog = self._outbox.get(entity_id)
            if backlog or not self._inner.try_send_bytes(entity_id, data):
                self._outbox.setdefault(entity_id, collections.deque()).append(data)

    def flush(self) -> bool:
        """Send what peers can take now; True if anything is still queued.

        A queued message whose send fails outright (not just a full peer) is
        dropped and logged: there is no caller left to raise to, and keeping
        it would stop every message behind it."""
        for entity_id, backlog in self._outbox.items():
            while backlog:
                try:
                    if not self._inner.try_send_bytes(entity_id, backlog[0]):
                        break
                except Exception:  # noqa: BLE001 — that message's failure alone
                    logger.exception("Dropping a queued message to %s that failed to send", entity_id)
                backlog.popleft()
        return any(self._outbox.values())

    def __getattr__(self, name):
        return getattr(self._inner, name)
