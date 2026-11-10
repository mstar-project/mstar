"""The optional API key (``--api-key`` / ``MSTAR_API_KEY``): how the launcher
resolves it and keeps it from every child but mstar-server, which checks it
(``rust/server/src/auth.rs``; its tests are in test/rust/).
"""

from __future__ import annotations

import os
import sys

import pytest

from mstar.api_server import auth, entrypoint

KEY = "s3cret-key"


def test_flag_wins_over_env_and_empty_is_no_key(monkeypatch):
    monkeypatch.setenv(auth.API_KEY_ENV, "from-env")
    assert auth.resolve_api_key() == "from-env"
    assert auth.resolve_api_key("from-flag") == "from-flag"
    assert auth.resolve_api_key("") is None
    monkeypatch.setenv(auth.API_KEY_ENV, "")
    assert auth.resolve_api_key() is None


@pytest.mark.parametrize("via", ["flag", "env"])
def test_serve_main_keeps_the_key_from_child_processes(monkeypatch, tmp_path, via):
    argv = ["--config", str(tmp_path / "missing.yaml")]
    if via == "flag":
        monkeypatch.delenv(auth.API_KEY_ENV, raising=False)
        argv += ["--api-key", KEY]
    else:
        monkeypatch.setenv(auth.API_KEY_ENV, KEY)
    monkeypatch.setattr("sys.argv", ["mstar", "serve", "kokoro", f"--api-key={KEY}", "--api-key", KEY])
    # main() would hand pytest's log handlers to a writer thread, which repeats every later record
    monkeypatch.setattr(entrypoint, "log_off_thread", lambda: None)
    # The config is read right after the key is applied; a missing one stops
    # main() before it spawns anything.
    with pytest.raises(FileNotFoundError):
        entrypoint.main(argv)
    # The conductor and workers are spawned later, from this environment
    # and (spawn start method) this sys.argv.
    assert auth.API_KEY_ENV not in os.environ
    assert not any(KEY in a for a in sys.argv)


def test_dynamo_worker_keeps_an_exported_key_from_child_processes(monkeypatch, tmp_path):
    """The Dynamo worker starts its conductor without ``entrypoint.main()``."""
    from types import SimpleNamespace

    from mstar.integrations.dynamo import worker as dynamo_worker

    monkeypatch.setenv(auth.API_KEY_ENV, KEY)
    # The config is read before the model is built or anything is spawned.
    with pytest.raises(FileNotFoundError):
        dynamo_worker.build_server(SimpleNamespace(config=str(tmp_path / "missing.yaml")))
    assert auth.API_KEY_ENV not in os.environ


def test_redaction_replaces_only_the_key_argument(monkeypatch):
    monkeypatch.setattr("sys.argv", ["mstar-serve", "--config", "a.yaml", "--api-key", "a",
                                     "--api-k=a", "--model-path=/models/a"])
    auth.keep_api_key_from_children("a")
    assert sys.argv == ["mstar-serve", "--config", "a.yaml", "--api-key", "<redacted>",
                        "--api-k=<redacted>", "--model-path=/models/a"]


def test_mstar_serve_forwards_the_key_and_never_prints_it(monkeypatch, capsys):
    from mstar.cli import main as cli

    seen = {}
    monkeypatch.setattr(entrypoint, "main", lambda argv: seen.setdefault("argv", argv))
    cli.main(["serve", "kokoro", "--api-key", KEY])
    argv = seen["argv"]
    assert argv[argv.index("--api-key") + 1] == KEY
    printed = capsys.readouterr().err
    assert KEY not in printed and 'api_key="<your key>"' in printed
