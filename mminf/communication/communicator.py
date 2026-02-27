from abc import ABC, abstractmethod
from enum import Enum

import zmq


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

    @abstractmethod
    def get_session_id(self) -> str:
        pass


class CommProtocol(Enum):
    IPC = "IPC"
    RDMA = "RDMA"


class ZMQCommunicator(BaseCommunicator):
    def __init__(
        self,
        my_id: str,
        push_ids: list[str],
        protocol: CommProtocol=CommProtocol.IPC,
        ipc_socket_path_prefix: str="/tmp/mminf/",
        # TODO: for TCP
    ):
        self.context = zmq.Context()
        self.protocol = protocol
        self.pull_socket = self.context.socket(zmq.PULL)

        # TODO: maybe only open sockets as we need them, and close sockets
        # when we no longer need them
        self.push_sockets: dict[str, zmq.SyncSocket] = {}
        self.session_id = f"ipc://{ipc_socket_path_prefix}/{my_id}.ipc"

        if protocol == CommProtocol.IPC:
            self.pull_socket.bind(f"ipc://{ipc_socket_path_prefix}/{my_id}.ipc")
            self.pull_socket.setsockopt(zmq.LINGER, 0)
        else:
            raise NotImplementedError(f"Protocol {protocol} not yet supported yet")
            
        for id in push_ids:
            if id == my_id:
                continue
            self.push_sockets[id] = self.context.socket(zmq.PUSH)
            self.push_sockets[id].connect(
                f"ipc://{ipc_socket_path_prefix}/{id}.ipc"
            )
            self.push_sockets[id].setsockopt(zmq.LINGER, 0)
    
    def get_session_id(self) -> str:
        return self.session_id

    def send(self, entity_id: str, msg):
        # TODO: maybe serialize to JSON instead if more efficient
        self.push_sockets[entity_id].send_pyobj(msg)
    
    def get_all_new_messages(self) -> list:
        messages = []
        while True:
            try:
                # zmq.NOBLOCK means zmq doesn't wait for a new message to be
                # available, it returns a message if it exists or raises an error 
                # if no messages are available (error is caught below)
                messages.append(self.pull_socket.recv_pyobj(
                    flags=zmq.NOBLOCK
                ))
            except zmq.Again:
                # zmq.Again actually means no messages left to read
                break 
        return messages
        