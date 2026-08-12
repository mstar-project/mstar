"""Filesystem layout for generated bundles, worktree seeds, and image tags.

Everything the tool generates lives under ``<repo>/.vibesys/`` (gitignored):

    .vibesys/
    ├── bundles/<task>/<instance>/   # VibeSys input bundle (read-only evaluator)
    └── seed/<task>/<instance>/      # git worktree of mstar (mutable candidate)

The bundle and the seed are deliberately *separate trees*: the seed is the
mutable workspace the agent edits, while the bundle carries the objective,
reference oracle, and correctness checker that must not be editable by the
candidate.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Absolute path of the enclosing git repository (the mstar checkout)."""
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return Path(out.strip())


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def base(self) -> Path:
        return self.root / ".vibesys"

    def bundle_dir(self, task: str, instance: str) -> Path:
        return self.base / "bundles" / task / instance

    def seed_dir(self, task: str, instance: str) -> Path:
        return self.base / "seed" / task / instance

    def image_tag(self, task: str, instance: str) -> str:
        return f"mstar-vibesys/{task}:{instance}"


def default_layout() -> Layout:
    return Layout(root=repo_root())
