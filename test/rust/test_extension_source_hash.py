"""A checkout refuses an mstar_rust or mstar-server built from other sources
than its rust/.

Run against an extension built from this checkout, as CI builds it.
"""
import logging
import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

mstar_rust = pytest.importorskip(
    "mstar_rust", reason="mstar_rust not built (maturin develop --release in rust/)",
)
from mstar.api_server.rust_frontend import find_server_binary  # noqa: E402
from mstar.graph.runtime.utils import require_rust_extension  # noqa: E402

RUST = Path(__file__).resolve().parents[2] / "rust"
if not (RUST / "Cargo.toml").is_file():
    pytest.skip("needs the checkout's rust/ (the wheel job copies test/ alone)", allow_module_level=True)


def test_a_fresh_build_matches_its_checkout():
    """build.rs's hash at build time and the extension's over the checkout."""
    assert mstar_rust.source_hash(str(RUST)) == mstar_rust.BUILD_SOURCE_HASH
    require_rust_extension(RUST)


@pytest.fixture
def rust_copy(tmp_path):
    copy = tmp_path / "rust"
    shutil.copytree(RUST, copy, ignore=shutil.ignore_patterns("target", "server"))
    return copy


def _sources(rust_dir):
    return [p for p in rust_dir.rglob("*") if p.is_file()]


def test_sources_that_differ_from_the_build_are_refused(rust_copy):
    """However old the sources look: a build copied in after them is newer."""
    with (rust_copy / "src" / "graph" / "runtime.rs").open("a") as f:
        f.write("// changed after the build\n")
    for p in _sources(rust_copy):
        os.utime(p, (1000, 1000))
    current = mstar_rust.source_hash(str(rust_copy))
    rebuild = (f"`touch {rust_copy / 'build.rs'} && "
               f"maturin develop --release -m {rust_copy / 'Cargo.toml'}`")
    with pytest.raises(ImportError, match=re.escape(rebuild)) as refusal:
        require_rust_extension(rust_copy)
    assert mstar_rust.BUILD_SOURCE_HASH in str(refusal.value)
    assert current in str(refusal.value)


def test_a_source_rewritten_unchanged_is_not_refused(rust_copy):
    """What a branch switch or a stash pop does to the files it puts back."""
    for p in _sources(rust_copy):
        p.write_bytes(p.read_bytes())
    require_rust_extension(rust_copy)


def test_a_directory_that_is_not_the_mstar_rust_crate_is_not_checked(tmp_path):
    require_rust_extension(tmp_path / "rust")
    (tmp_path / "rust").mkdir()
    (tmp_path / "rust" / "Cargo.toml").write_text('[package]\nname = "other"\n')
    require_rust_extension(tmp_path / "rust")


SERVER = RUST / "server" / "target" / "release" / "mstar-server"
# For the cargo runs, as found before a test empties PATH.
BUILD_ENV = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}


@pytest.fixture
def no_installed_server(tmp_path, monkeypatch):
    """Leaves find_server_binary only the in-repo build of the rust_dir it is given."""
    monkeypatch.delenv("MSTAR_SERVER_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(tmp_path / "no-bin"))


@pytest.mark.skipif(not SERVER.is_file(), reason="mstar-server not built (cargo build --release in rust/server/)")
def test_a_fresh_server_build_matches_its_checkout(no_installed_server):
    reported = subprocess.run([SERVER, "--source-hash"], capture_output=True, text=True, check=True).stdout
    assert reported.split() == [mstar_rust.source_hash(str(RUST)), mstar_rust.source_hash(str(RUST / "server"))]
    assert find_server_binary(rust_dir=RUST) == str(SERVER)


def _fake_checkout(rust):
    """rust/ and rust/server/ with the real build scripts and hashing but no
    dependencies, so they build in seconds."""
    (rust / "src").mkdir(parents=True)
    (rust / "server" / "src").mkdir(parents=True)
    for f in ("build.rs", "src/source_hash.rs", "server/build.rs"):
        shutil.copy(RUST / f, rust / f)
    (rust / "Cargo.toml").write_text(
        '[package]\nname = "mstar-rust"\nversion = "0.0.0"\nedition = "2021"\n\n[lib]\nname = "mstar_rust"\n')
    (rust / "src" / "lib.rs").write_text('pub const BUILD_SOURCE_HASH: &str = env!("MSTAR_RUST_SOURCE_HASH");\n')
    (rust / "server" / "Cargo.toml").write_text(
        '[package]\nname = "mstar-server"\nversion = "0.0.0"\nedition = "2021"\n\n'
        '[dependencies]\nmstar-rust = { path = ".." }\n')
    # What the real one prints on no arguments.
    (rust / "server" / "src" / "main.rs").write_text(
        'fn main() {\n'
        '    let server = env!("MSTAR_SERVER_SOURCE_HASH");\n'
        '    eprintln!("built from sources {} {server}", mstar_rust::BUILD_SOURCE_HASH);\n'
        '    std::process::exit(2);\n'
        '}\n')
    return rust


def _sh(command):
    # From the checkout, so rustup takes its pinned toolchain.
    subprocess.run(command, shell=True, check=True, cwd=RUST.parent, env=BUILD_ENV)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")
def test_a_stale_server_is_refused_until_the_printed_command_rebuilds_it(tmp_path, no_installed_server):
    """Sources with other content but older mtimes than the build, as `cp -p`,
    `rsync -a` or tar leave them: cargo goes by mtimes, so a plain rebuild
    keeps the old hashes."""
    rust = _fake_checkout(tmp_path / "rust")
    binary = str(rust / "server" / "target" / "release" / "mstar-server")
    build = f"cargo build --release --manifest-path {rust / 'server' / 'Cargo.toml'}"
    _sh(build)
    assert find_server_binary(rust_dir=rust) == binary
    for f in (rust / "src" / "lib.rs", rust / "server" / "src" / "main.rs"):
        f.write_text(f.read_text() + "// changed after the build\n")
        os.utime(f, (1000, 1000))
    _sh(build)
    with pytest.raises(RuntimeError, match=re.escape(f"{binary} was built from")) as refusal:
        find_server_binary(rust_dir=rust)
    _sh(re.search(r"`(touch .+)`", str(refusal.value)).group(1))
    assert find_server_binary(rust_dir=rust) == binary


def test_a_server_from_before_the_check_is_used_only_when_named(
        rust_copy, tmp_path, no_installed_server, monkeypatch, caplog):
    old = tmp_path / "old" / "mstar-server"
    old.parent.mkdir()
    # What an mstar-server from before the check does with no arguments.
    old.write_text("#!/bin/sh\necho 'usage: mstar-server <model_name>' >&2\nexit 101\n")
    old.chmod(0o755)
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(old.parent))
    with pytest.raises(RuntimeError, match=re.escape(f"{old} predates the source check")):
        find_server_binary(rust_dir=rust_copy)
    with caplog.at_level(logging.WARNING):
        assert find_server_binary(str(old), rust_dir=rust_copy) == str(old)
        monkeypatch.setenv("MSTAR_SERVER_BIN", str(old))
        assert find_server_binary(rust_dir=rust_copy) == str(old)
    assert caplog.text.count(f"{old}, which predates the source check") == 2
