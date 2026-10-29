"""The conductor's main loop waits between passes on its communicator's poll:
a message ends the wait, and with none it is the 1 ms backstop the loop's nap
was, on either transport."""
import time
from types import SimpleNamespace

import pytest

from mstar.conductor.conductor import Conductor
from mstar.utils.ipc_format import ConductorMessage, ConductorMessageType, SetupDone


def _communicators(kind, tmp_path):
    if kind == "rust":
        pytest.importorskip("mstar_rust")
        from mstar.communication.rust_communicator import RustZMQCommunicator as Comm
    else:
        from mstar.communication.communicator import ZMQCommunicator as Comm
    prefix = str(tmp_path) + "/"
    return Comm("conductor", push_ids=[], ipc_socket_path_prefix=prefix), \
        Comm("worker_0", push_ids=["conductor"], ipc_socket_path_prefix=prefix)


@pytest.mark.parametrize("kind", ["python", "rust"])
def test_the_pass_wait_ends_on_a_message(kind, tmp_path):
    conductor = Conductor.__new__(Conductor)
    conductor.communicator, peer = _communicators(kind, tmp_path)
    t0 = time.monotonic()
    for _ in range(20):
        assert conductor._pass_wait() is False
    assert 0.015 < time.monotonic() - t0 < 1.0
    peer.send("conductor", ConductorMessage(
        message_type=ConductorMessageType.SETUP_DONE, body=SetupDone(worker_id="worker_0"),
    ))
    deadline = time.monotonic() + 5.0
    while not conductor._pass_wait():
        assert time.monotonic() < deadline, "the message never ended the wait"
    messages = conductor.communicator.get_all_new_messages()
    assert [m.message_type for m in messages] == [ConductorMessageType.SETUP_DONE]


def test_a_stand_in_without_the_poll_keeps_the_nap():
    conductor = Conductor.__new__(Conductor)
    conductor.communicator = SimpleNamespace()
    t0 = time.monotonic()
    assert conductor._pass_wait() is False
    assert time.monotonic() - t0 >= 0.001
