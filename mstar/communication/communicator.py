import logging
import os
from abc import ABC, abstractmethod
from enum import Enum

import zmq

from mstar.communication.event import EventWakeup

logger = logging.getLogger(__name__)


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

    @abstractmethod
    def get_all_new_messages(self) -> list:
        pass

    # -- endpoint scheme (shared by every ZMQ-based communicator) ------------
    # Subclasses set ``self.protocol`` and ``self.ipc_socket_path_prefix``.

    def _endpoint(self, entity_id: str) -> str:
        if self.protocol == CommProtocol.IPC:
            return f"ipc://{self.ipc_socket_path_prefix}/{entity_id}.ipc"
        if self.protocol == CommProtocol.TCP:
            host = os.getenv("MSTAR_ZMQ_TCP_HOST", "127.0.0.1")
            return f"tcp://{host}:{self._tcp_port(entity_id)}"
        raise NotImplementedError(f"Protocol {self.protocol} not yet supported yet")

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


class ZMQCommunicator(BaseCommunicator):
    def __init__(
        self,
        my_id: str,
        push_ids: list[str],
        protocol: CommProtocol=CommProtocol.IPC,
        ipc_socket_path_prefix: str="/tmp/mstar/",
        # TODO: for TCP
    ):
        self.context = zmq.Context.instance()
        transport = os.getenv("MSTAR_ZMQ_TRANSPORT", protocol.value).upper()
        self.protocol = CommProtocol(transport)
        self.pull_socket = self.context.socket(zmq.PULL)
        if self.protocol == CommProtocol.IPC:
            os.makedirs(ipc_socket_path_prefix, exist_ok=True)

        # TODO: maybe only open sockets as we need them, and close sockets
        # when we no longer need them
        self.push_sockets: dict[str, zmq.SyncSocket] = {}
        self.my_id = my_id
        self.ipc_socket_path_prefix = ipc_socket_path_prefix

        if self.protocol == CommProtocol.IPC:
            self.pull_socket.bind(self._endpoint(my_id))
            self.pull_socket.setsockopt(zmq.LINGER, 0)
        elif self.protocol == CommProtocol.TCP:
            self.pull_socket.bind(self._endpoint(my_id))
            self.pull_socket.setsockopt(zmq.LINGER, 0)
        else:
            raise NotImplementedError(f"Protocol {protocol} not yet supported yet")

        for id in push_ids:
            if id == my_id:
                continue
            self.push_sockets[id] = self.context.socket(zmq.PUSH)
            self.push_sockets[id].connect(self._endpoint(id))
            self.push_sockets[id].setsockopt(zmq.LINGER, 0)
        self.poller = zmq.Poller()
        self.poller.register(self.pull_socket, zmq.POLLIN)
        self.event = None

    def register_event_for_poll(self, event: EventWakeup):
        self.poller.register(event.fd,  zmq.POLLIN)
        self.event = event

    def wait_for_work(self, timeout_ms=50):
        events = dict(self.poller.poll(timeout=timeout_ms))
        if self.event.fd in events:
            self.event.drain()

    # def get_session_id(self) -> str:
    #     return self.session_id

    def send(self, entity_id: str, msg):
        # TODO: maybe serialize to JSON instead if more efficient
        logger.debug(
            "%s to send a message %s to entity %s",
            self.my_id, str(msg), entity_id
        )
        if entity_id not in self.push_sockets:
            sock = self.context.socket(zmq.PUSH)
            sock.connect(self._endpoint(entity_id))
            sock.setsockopt(zmq.LINGER, 0)
            self.push_sockets[entity_id] = sock
        self.push_sockets[entity_id].send_pyobj(msg)

    def get_all_new_messages(self, blocking=False) -> list:
        messages = []
        if blocking:
            # Wait until the pull socket is readable before draining. A
            # registered wakeup event also ends the wait (and is drained
            # here, exactly as in wait_for_work), so a completed compute
            # future can interrupt a blocking receive.
            events = dict(self.poller.poll())
            if self.event is not None and self.event.fd in events:
                self.event.drain()
        while True:
            try:
                # zmq.NOBLOCK means zmq doesn't wait for a new message to be
                # available, it returns a message if it exists or raises an error
                # if no messages are available (error is caught below)
                messages.append(self.pull_socket.recv_pyobj(
                    flags=zmq.NOBLOCK
                ))
                logger.debug(
                    "%s to received message %s",
                    self.my_id, str(messages[-1])
                )
            except zmq.Again:
                # zmq.Again actually means no messages left to read
                break
        return messages


def make_communicator(*args, **kwargs) -> BaseCommunicator:
    """Construct the process's communicator, selecting the transport.

    ``MSTAR_RUST_ZMQ`` selects it (see ``docs/environment_variables.rst``):

    * ``0`` (default) — the pyzmq ``ZMQCommunicator``.
    * ``1`` — the Rust-backed ``RustZMQCommunicator`` (vendored ``rust/``
      extension; see ``communication/rust_communicator.py``); raises if the
      extension is not installed.
    * ``AUTO`` — the Rust communicator when the extension imports
      successfully, pyzmq otherwise.

    The two are wire-compatible (same endpoints, same pickle frames), so the
    flag can be set per-process — one entity at a time — while the rest of
    the mesh stays on pyzmq.
    """
    choice = os.getenv("MSTAR_RUST_ZMQ", "0").upper()
    if choice not in ("0", "1", "AUTO"):
        raise ValueError(f"MSTAR_RUST_ZMQ must be 0, 1, or AUTO; got {choice!r}")
    if choice != "0":
        try:
            from mstar.communication.rust_communicator import RustZMQCommunicator
        except ImportError:
            if choice == "1":
                raise
            logger.debug("MSTAR_RUST_ZMQ=AUTO: mstar_rust not installed, using pyzmq")
        else:
            return RustZMQCommunicator(*args, **kwargs)
    return ZMQCommunicator(*args, **kwargs)
