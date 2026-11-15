"""Two deployments must not share one IPC socket prefix.

Over TCP a second deployment on the same base port fails to bind. Over IPC it
does not: libzmq unlinks an existing socket file and binds a fresh one, so the
second conductor comes up, takes over every connection aimed at the first, and
derives the same tensor-file namespace (uuids are per-entity counters, so both
then write, read and unlink each other's files). An flock beside the socket is
what turns that into an error at startup.
"""

import os
import tempfile

import pytest

from mstar.communication.communicator import DEPLOYMENT_ANCHOR_ENTITY, CommProtocol
from mstar.communication.rust_communicator import RustZMQCommunicator


def _comm(prefix: str, my_id: str, protocol=CommProtocol.IPC) -> RustZMQCommunicator:
    # Its sockets close when it is dropped, hence the `del`s below.
    return RustZMQCommunicator(
        my_id=my_id, push_ids=[], protocol=protocol,
        ipc_socket_path_prefix=prefix,
    )


def test_second_conductor_on_one_prefix_is_refused():
    with tempfile.TemporaryDirectory() as prefix:
        first = _comm(prefix, DEPLOYMENT_ANCHOR_ENTITY)
        with pytest.raises(RuntimeError, match="already holds"):
            _comm(prefix, DEPLOYMENT_ANCHOR_ENTITY)
        del first


def test_a_second_api_server_is_refused_before_it_binds_over_the_first():
    """Its conductor, which would be refused, starts only after the API server
    has bound its sockets in the first deployment's prefix."""
    with tempfile.TemporaryDirectory() as prefix:
        first = _comm(prefix, "api_server_preprocess_worker")
        with pytest.raises(RuntimeError, match="already holds"):
            _comm(prefix, "api_server_preprocess_worker")
        del first


def test_conductors_on_different_prefixes_both_start():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ca = _comm(a, DEPLOYMENT_ANCHOR_ENTITY)
        cb = _comm(b, DEPLOYMENT_ANCHOR_ENTITY)
        del ca, cb


def test_the_lock_is_released_when_the_holder_exits():
    """flock lives on the open file description, so the kernel drops it
    however the process dies -- a crashed deployment leaves nothing to clean
    up by hand."""
    with tempfile.TemporaryDirectory() as prefix:
        first = _comm(prefix, DEPLOYMENT_ANCHOR_ENTITY)
        os.close(first._deployment_lock_fd)  # stands in for the process exiting
        second = _comm(prefix, DEPLOYMENT_ANCHOR_ENTITY)
        del first, second


def test_workers_do_not_lock():
    """One deployment has many workers on one prefix; they must not lock each
    other out."""
    with tempfile.TemporaryDirectory() as prefix:
        a = _comm(prefix, "worker_0")
        b = _comm(prefix, "worker_1")
        assert not hasattr(a, "_deployment_lock_fd")
        del a, b


def test_tcp_does_not_lock():
    """The bind already fails for a duplicate; a lock file keyed on a prefix
    TCP ignores would refuse unrelated deployments."""
    comm = _comm("/tmp/mstar_unused/", DEPLOYMENT_ANCHOR_ENTITY,
                 protocol=CommProtocol.TCP)
    assert not hasattr(comm, "_deployment_lock_fd")
