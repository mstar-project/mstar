"""``mstar serve`` argument wiring, with the server launch stubbed out."""

import sys
import types

import pytest

from mstar.cli import main as cli


@pytest.fixture
def launched(monkeypatch):
    calls: list[list[str]] = []
    fake = types.ModuleType("mstar.api_server.entrypoint")
    fake.main = calls.append
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake)
    return calls


def _socket_prefix(argv: list[str]) -> str:
    return argv[argv.index("--socket-path-prefix") + 1]


def test_servers_on_different_ports_get_different_socket_prefixes(launched):
    for port in ("8001", "8002"):
        cli._serve(cli.build_parser().parse_args(["serve", "kokoro", "--port", port]))
    first, second = (_socket_prefix(argv) for argv in launched)
    assert first != second and first.endswith("_8001/") and second.endswith("_8002/")


def test_an_explicit_socket_prefix_is_kept(launched):
    cli._serve(cli.build_parser().parse_args(["serve", "kokoro", "--socket-path-prefix", "/tmp/mine/"]))
    assert _socket_prefix(launched[0]) == "/tmp/mine/"
