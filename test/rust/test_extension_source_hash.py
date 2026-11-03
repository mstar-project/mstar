"""A checkout refuses an mstar_rust built from other sources than its rust/.

Run against an extension built from this checkout, as CI builds it.
"""
import os
import re
import shutil
from pathlib import Path

import pytest

mstar_rust = pytest.importorskip(
    "mstar_rust", reason="mstar_rust not built (maturin develop --release in rust/)",
)
from mstar.graph.runtime.utils import require_rust_extension  # noqa: E402

RUST = Path(__file__).resolve().parents[2] / "rust"


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
    rebuild = f"`maturin develop --release -m {rust_copy / 'Cargo.toml'}`"
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
