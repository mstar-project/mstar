"""``mstar serve``: where a server's IPC sockets and uploads go."""

import getpass
import os

import pytest

from mstar import platforms
from mstar.api_server import entrypoint
from mstar.cli import main as cli


@pytest.fixture
def served(monkeypatch):
    """Runs mstar serve without starting a server. Gives back, per run, the
    socket prefix, the upload dir and whether both existed while serving."""
    runs = []

    def serve_main(argv):
        prefix = argv[argv.index("--socket-path-prefix") + 1]
        uploads = argv[argv.index("--upload-dir") + 1]
        runs.append((prefix, uploads, os.path.isdir(prefix) and os.path.isdir(uploads)))

    monkeypatch.setattr(entrypoint, "main", serve_main)
    monkeypatch.setattr(platforms, "apply_visible_devices", lambda devices: None)
    return runs


def test_explicit_dirs_pass_through_unchanged(served, tmp_path):
    prefix, uploads = f"{tmp_path}/sock/", f"{tmp_path}/up/"
    cli.main(["serve", "kokoro", "--socket-path-prefix", prefix, "--upload-dir", uploads])
    assert served == [(prefix, uploads, False)]
    assert not os.path.exists(prefix) and not os.path.exists(uploads)


def test_each_server_gets_its_own_dirs_removed_at_exit(served):
    cli.main(["serve", "kokoro"])
    cli.main(["serve", "kokoro"])
    (prefix_a, uploads_a, existed_a), (prefix_b, uploads_b, existed_b) = served
    who = getpass.getuser()
    assert prefix_a != prefix_b and uploads_a != uploads_b
    assert prefix_a.startswith(f"/tmp/mstar_{who}_") and prefix_a.endswith("/")
    assert uploads_a.startswith(f"/tmp/mstar_uploads_{who}_")
    assert existed_a and existed_b
    for path in (prefix_a, uploads_a, prefix_b, uploads_b):
        assert not os.path.exists(path)


def test_dirs_are_removed_when_the_server_stops_with_an_interrupt(served, monkeypatch):
    def interrupted(argv):
        served.append(argv[argv.index("--socket-path-prefix") + 1])
        raise KeyboardInterrupt

    monkeypatch.setattr(entrypoint, "main", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["serve", "kokoro"])
    assert not os.path.exists(served[0])
