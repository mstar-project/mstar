"""Build shim: copy the repo-root configs/ into mstar/default_configs/ so the
default deployment YAMLs ship inside the wheel under the mstar namespace.

Project metadata lives in pyproject.toml; this file exists only to populate
the packaged configs at build time. configs/ stays the single source of truth
(unchanged for checkouts and in-flight model PRs); nothing is duplicated in
git. The sdist carries configs/ via MANIFEST.in so a wheel built from the
sdist also picks them up.
"""

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


setup(cmdclass={"build_py": build_py_with_configs})
