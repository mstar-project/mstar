#!/usr/bin/env python3
"""Archive the current Git worktree for local CI without host build products."""

import argparse
import os
import subprocess
import tarfile
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def is_local_file(path):
    return any(
        part in {".git", ".ci-results", ".venv", "venv", "build", "dist", "target", "__pycache__"}
        or part.startswith(".env")
        for part in path.parts
    )


def snapshot(root, destination):
    root = root.resolve()
    tracked = set()
    for entry in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        if metadata.startswith(b"160000 "):
            raise ValueError(f"Local CI does not yet support submodules: {os.fsdecode(name)}")
        tracked.add(os.fsdecode(name))
    untracked = {
        os.fsdecode(name)
        for name in git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if name
    }
    selected = tracked | {name for name in untracked if not is_local_file(Path(name))}
    with tarfile.open(destination, "w", dereference=False) as archive:
        for name in sorted(selected):
            relative = Path(name)
            # A tracked directory can have been replaced by a symlink locally.
            # Never follow it to copy files from outside the worktree.
            if any((root / parent).is_symlink() for parent in relative.parents):
                raise ValueError(f"Cannot snapshot a file through a directory symlink: {name}")
            source = root / relative
            if not source.exists() and not source.is_symlink():
                continue
            if not source.is_symlink() and source.resolve() == destination.resolve():
                continue
            if not source.is_file() and not source.is_symlink():
                raise ValueError(f"Cannot snapshot a non-file: {name}")
            archive.add(source, arcname=name, recursive=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("output_tar", type=Path)
    args = parser.parse_args()
    try:
        snapshot(args.repo_root, args.output_tar)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"snapshot: {error}\n")
