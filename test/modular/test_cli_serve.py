"""``mstar serve``: where a server's IPC sockets and uploads go."""

import getpass
import os

import pytest

from mstar import platforms
from mstar.api_server import entrypoint
from mstar.cli import main as cli
from mstar.utils.socket_prefix import per_port_socket_prefix


@pytest.fixture
def served(monkeypatch):
    """Runs mstar serve without starting a server. Gives back, per run, the
    socket prefix, the upload dir and whether the upload dir existed while
    serving."""
    runs = []

    def serve_main(argv):
        prefix = argv[argv.index("--socket-path-prefix") + 1]
        uploads = argv[argv.index("--upload-dir") + 1]
        runs.append((prefix, uploads, os.path.isdir(uploads)))

    monkeypatch.setattr(entrypoint, "main", serve_main)
    monkeypatch.setattr(platforms, "apply_visible_devices", lambda devices: None)
    return runs


def test_explicit_dirs_pass_through_unchanged(served, tmp_path):
    prefix, uploads = f"{tmp_path}/sock/", f"{tmp_path}/up/"
    cli.main(["serve", "kokoro", "--socket-path-prefix", prefix, "--upload-dir", uploads])
    assert served == [(prefix, uploads, False)]
    assert not os.path.exists(prefix) and not os.path.exists(uploads)


def test_each_server_gets_its_own_upload_dir_removed_at_exit(served):
    """The socket prefix is per port; the upload dir is per server."""
    cli.main(["serve", "kokoro", "--port", "9123"])
    cli.main(["serve", "kokoro", "--port", "9123"])
    (prefix_a, uploads_a, existed_a), (prefix_b, uploads_b, existed_b) = served
    assert prefix_a == prefix_b == per_port_socket_prefix(9123)
    assert uploads_a != uploads_b
    assert uploads_a.startswith(f"/tmp/mstar_uploads_{getpass.getuser()}_") and uploads_a.endswith("/")
    assert existed_a and existed_b
    assert not os.path.exists(uploads_a) and not os.path.exists(uploads_b)


def test_the_upload_dir_is_removed_when_the_server_stops_with_an_interrupt(served, monkeypatch):
    def interrupted(argv):
        served.append(argv[argv.index("--upload-dir") + 1])
        raise KeyboardInterrupt

    monkeypatch.setattr(entrypoint, "main", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["serve", "kokoro"])
    assert not os.path.exists(served[0])
