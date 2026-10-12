"""Standalone git-clone seed of mstar for the mutable candidate workspace.

A `git worktree` would be lighter, but its `.git` is a *gitlink* pointing back
into the parent repo; when VibeSys copies the seed tree into its isolated
candidate workspace the link breaks ("fatal: not a git repository") and the
agent loses `git diff` on its own edits. A `--local` clone checked out at the
pinned commit is a self-contained repo: hardlinked (fast/small), no gitlink, no
parent-repo worktree bookkeeping, and the agent can diff its work.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], check=check, text=True, capture_output=True)


def resolve_commit(repo: Path, rev: str) -> str:
    return _git(repo, "rev-parse", rev).stdout.strip()


def create_seed(repo: Path, commit: str, dest: Path, *, force: bool = False) -> Path:
    """Create (or reuse) a standalone clone of ``repo`` at ``commit`` under ``dest``."""
    if dest.exists():
        if not force:
            return dest
        remove_seed(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--local", "--no-checkout", str(repo), str(dest)],
        check=True, text=True, capture_output=True,
    )
    _git(dest, "checkout", "--quiet", "--detach", commit)
    return dest


def remove_seed(dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)
