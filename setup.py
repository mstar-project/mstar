"""Build shim: copy the repo-root configs/ into mstar/default_configs/ so the
default deployment YAMLs ship inside the wheel under the mstar namespace, and
compile the ``mstar_rust`` extension and the ``mstar-server`` binary into the
published Linux wheels.

Project metadata lives in pyproject.toml; this file exists only for those two
build steps. configs/ stays the single source of truth (unchanged for checkouts
and in-flight model PRs); nothing is duplicated in git. The sdist carries
configs/ via MANIFEST.in so a wheel built from the sdist also picks them up.
"""

import os
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_ROOT = Path(__file__).parent
_SRC = _ROOT / "configs"
_DST_PKG = "mstar/default_configs"


class build_py_with_configs(build_py):
    def run(self):
        super().run()
        dst = Path(self.build_lib) / _DST_PKG
        dst.mkdir(parents=True, exist_ok=True)
        yamls = sorted(_SRC.glob("*.yaml"))
        if not yamls:
            raise RuntimeError(
                f"no configs/*.yaml found at {_SRC}; the sdist must graft "
                "configs/ (see MANIFEST.in) for a wheel build from it"
            )
        for y in yamls:
            shutil.copy2(y, dst / y.name)


# Only the release's cibuildwheel jobs compile rust/: the mstar_rust extension
# and the mstar-server binary (--rust-frontend), which installs into the
# environment's scripts directory. Every other build is the pure-Python
# package, and a checkout builds both with cargo/maturin
# (docs/installation.rst). setuptools' editable finder cannot import a
# top-level extension built in place, and compiling it would slow every
# `pip install -e .`.
rust_extensions = []
if os.environ.get("CIBUILDWHEEL") == "1":
    from setuptools_rust import Binding, RustBin, RustExtension, Strip

    # --locked: the checked-in Cargo.lock versions, not whatever resolves on
    # the day; the license notices are collected from the same lockfiles.
    # Stripped: the symbols are unused at runtime and only bloat six wheels.
    wheel_build = {"cargo_manifest_args": ["--locked"], "strip": Strip.All}
    rust_extensions += [
        RustExtension("mstar_rust", path="rust/Cargo.toml", binding=Binding.PyO3, **wheel_build),
        RustBin("mstar-server", path="rust/server/Cargo.toml", **wheel_build),
    ]

setup(cmdclass={"build_py": build_py_with_configs}, rust_extensions=rust_extensions)
