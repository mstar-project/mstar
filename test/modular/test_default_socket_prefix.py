"""Each server entry point defaults to an IPC socket prefix of its own (#490).

Every process of a deployment binds fixed names under its prefix, and libzmq
replaces an existing socket file on bind. With one shared default, a second
server's API server binds over the first one's ``api_server.ipc`` before the
deployment lock refuses its conductor, and the first server's replies land in
the second one's inbox.
"""

import os
import shutil
import stat
import sys
import time
import types

import pytest

from mstar.communication.communicator import ZMQCommunicator
from mstar.utils import socket_prefix


class _StopError(Exception):
    pass


def _entrypoint_prefix(monkeypatch, tmp_path, *argv) -> str:
    """The prefix ``mstar-serve`` hands the API server, without starting one."""
    from mstar.api_server import entrypoint
    from mstar.model import registry

    seen = {}

    def fake_api_server(**kwargs):
        seen.update(kwargs)
        raise _StopError

    class _Model:
        PREPROCESS_TORCH_THREADS = None
        API_SWITCH_INTERVAL_S = None

        def __init__(self, **kwargs):
            pass

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: fake\n")
    monkeypatch.setattr(registry, "get_model_class", lambda name: _Model)
    monkeypatch.setattr(entrypoint, "model_init_kwargs", lambda name: {})
    monkeypatch.setattr(entrypoint, "APIServer", fake_api_server)
    # sizing torch's pool would outlive the test in this process
    monkeypatch.setattr(entrypoint, "_set_preprocess_threads", lambda model, config: None)
    with pytest.raises(_StopError):
        entrypoint.main(["--config", str(cfg), *argv])
    return seen["socket_path_prefix"]


def test_mstar_serve_entrypoint_keys_the_default_prefix_on_the_port(monkeypatch, tmp_path):
    a = _entrypoint_prefix(monkeypatch, tmp_path, "--port", "8000")
    b = _entrypoint_prefix(monkeypatch, tmp_path, "--port", "8001")
    assert a != b
    assert a == socket_prefix.per_port_socket_prefix(8000)


def test_mstar_serve_entrypoint_keeps_an_explicit_prefix(monkeypatch, tmp_path):
    got = _entrypoint_prefix(monkeypatch, tmp_path, "--socket-path-prefix", "/x/y/")
    assert got == "/x/y/"


def test_mstar_serve_cli_uses_the_same_rule(monkeypatch):
    from mstar.cli import main as cli

    seen = []
    monkeypatch.setitem(
        sys.modules, "mstar.api_server.entrypoint",
        types.SimpleNamespace(main=seen.append),
    )
    cli.main(["serve", "kokoro", "--port", "9123"])
    argv = seen[0]
    assert argv[argv.index("--socket-path-prefix") + 1] == socket_prefix.per_port_socket_prefix(9123)


def _dynamo_prefix(monkeypatch, *argv) -> dict:
    """The prefix the Dynamo worker builds its server on, and whether that
    directory still exists after ``main`` returns."""
    from mstar.integrations.dynamo import worker

    seen = {}

    def fake_build_server(args):
        seen["prefix"] = args.socket_path_prefix
        seen["mode"] = stat.S_IMODE(os.stat(args.socket_path_prefix).st_mode)
        raise _StopError

    monkeypatch.setattr(worker, "build_server", fake_build_server)
    with pytest.raises(_StopError):
        worker.main(["--config", "c.yaml", "--model-path", "/m", *argv])
    return seen


def test_dynamo_workers_get_a_private_prefix_per_launch_removed_at_exit(monkeypatch):
    a = _dynamo_prefix(monkeypatch)
    b = _dynamo_prefix(monkeypatch)
    assert a["prefix"] != b["prefix"]
    assert a["mode"] == 0o700
    assert not os.path.exists(a["prefix"]) and not os.path.exists(b["prefix"])


def test_dynamo_worker_keeps_and_does_not_remove_an_explicit_prefix(monkeypatch, tmp_path):
    got = _dynamo_prefix(monkeypatch, "--socket-path-prefix", str(tmp_path))
    assert got["prefix"] == str(tmp_path)
    assert tmp_path.is_dir()


def test_two_servers_on_their_default_prefixes_keep_their_own_inboxes():
    """The #490 repro: on a shared prefix, the second server's api_server binds
    over the first one's, so the first conductor's reply reaches the second."""
    a = socket_prefix.per_launch_socket_prefix()
    b = socket_prefix.per_launch_socket_prefix()
    comms = []
    try:
        for prefix in (a, b):
            comms.append(ZMQCommunicator(
                my_id="api_server", push_ids=["conductor"], ipc_socket_path_prefix=prefix))
            comms.append(ZMQCommunicator(
                my_id="conductor", push_ids=["api_server"], ipc_socket_path_prefix=prefix))
        a_api, a_cond, b_api, _ = comms
        a_cond.send("api_server", {"reply": "a"})
        got = []
        deadline = time.monotonic() + 5
        while not got and time.monotonic() < deadline:
            got = a_api.get_all_new_messages()
            time.sleep(0.01)
        assert got == [{"reply": "a"}]
        assert b_api.get_all_new_messages() == []
    finally:
        for c in comms:
            c.pull_socket.close()
            for s in c.push_sockets.values():
                s.close()
        for prefix in (a, b):
            shutil.rmtree(prefix, ignore_errors=True)


@pytest.mark.parametrize("name, kept", [("alice", True), ("a" * 32, True), ("a" * 33, False),
                                        ("\u00e9" * 30, False), ("a b", False), ("", False)])
def test_the_prefix_holds_the_user_name_only_when_it_is_short_and_ascii(monkeypatch, name, kept):
    monkeypatch.setattr(socket_prefix.getpass, "getuser", lambda: name)
    prefix = socket_prefix.per_port_socket_prefix(65535)
    assert prefix == f"/tmp/mstar_{name if kept else os.getuid()}_65535/"
    # a 45-byte socket name, longer than any a deployment binds, still fits a socket path
    assert len((prefix + "x" * 41 + ".ipc").encode()) <= 107
