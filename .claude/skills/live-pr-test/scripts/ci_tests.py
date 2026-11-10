#!/usr/bin/env python3
"""Print CI's test files that reach a PR's change, one per line: what the live test runs on the GPUs.

usage: ci_tests.py <base-branch>      (in the PR's checkout; fetches the base from origin)

CI's tests are the CPU Core job's paths (scripts/ci/select_cpu_tests.py) plus the CUDA cases in
test/rust/test_arena_transport.py. Of those, a file reaches the change when the PR adds or changes
it, or a conftest.py above it, or when it imports, directly or through other modules of the
checkout, a module the PR changes (a change under rust/ is a change to mstar_rust). The imports
are read statically, so one made with importlib, or a path built at run time, isn't seen. No
output: no CI test reaches the change. Stdlib only.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path, PurePosixPath

EXTRA = ("test/rust/test_arena_transport.py",)  # CUDA cases the rust-transport job runs without a GPU
SKIP_DIRS = {".git", ".venv", "node_modules", "target", "__pycache__", "build", "dist"}


def module_of(path: str) -> str | None:
    """The dotted module a repo-relative .py path is, or for a change under rust/, mstar_rust."""
    if path.startswith("rust/"):
        return "mstar_rust"
    if not path.endswith(".py"):
        return None
    parts = PurePosixPath(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports(root: Path, path: str) -> set[str]:
    """What the file at `path` imports, as dotted names: each module, and each name imported from one (which
    may be a submodule). A bare name also as a sibling of the file, as pytest's rootdir insertion finds it."""
    try:
        tree = ast.parse((root / path).read_text(), path)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    package = list(PurePosixPath(path).parent.parts)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join([*package[:len(package) - node.level + 1], *([base] if base else [])])
            found = [base, *(f"{base}.{a.name}" if base else a.name for a in node.names)]
        else:
            continue
        for name in found:
            names.add(name)
            names.add(".".join([*package, name]))
    return names


def select(root: Path, changed_paths: list[str], candidates: list[str]) -> list[str]:
    """Of `candidates` (CI's test files), those that reach `changed_paths` (see the module docstring)."""
    changed_paths = [p for p in changed_paths if p]
    hit = {p for p in candidates if p in changed_paths}
    conftests = [PurePosixPath(p).parent for p in changed_paths if PurePosixPath(p).name == "conftest.py"]
    hit |= {p for p in candidates if any(d in PurePosixPath(p).parents for d in conftests)}
    reached = {m for m in map(module_of, changed_paths) if m}
    if reached:
        # name -> the modules whose import of it runs it: importing a.b.c runs a, a.b and a.b.c
        importers: dict[str, set[str]] = {}
        for f in (p.relative_to(root).as_posix() for p in root.rglob("*.py")):
            if SKIP_DIRS.isdisjoint(PurePosixPath(f).parts):
                for name in imports(root, f):
                    parts = name.split(".")
                    for i in range(1, len(parts) + 1):
                        importers.setdefault(".".join(parts[:i]), set()).add(module_of(f))
        frontier = set(reached)
        while frontier:  # every module that imports a changed one, directly or through others
            frontier = set().union(*(importers.get(m, ()) for m in frontier)) - reached
            reached |= frontier
        hit |= {p for p in candidates if module_of(p) in reached}
    return sorted(hit)


def candidates(root: Path) -> list[str]:
    """CI's test files: test_*.py under the CPU Core job's paths, and EXTRA."""
    sys.path.insert(0, str(root / "scripts/ci"))
    from select_cpu_tests import expand_paths
    files = [p.relative_to(root).as_posix() for d in expand_paths(root) for p in sorted((root / d).rglob("test_*.py"))]
    return sorted({*files, *(e for e in EXTRA if (root / e).exists())})


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               check=True).stdout.strip())
    subprocess.run(["git", "-C", str(root), "fetch", "-q", "origin", sys.argv[1]], check=True)
    changed = subprocess.run(["git", "-C", str(root), "diff", "--name-only", "FETCH_HEAD...HEAD"],
                             capture_output=True, text=True, check=True).stdout.split("\n")
    for path in select(root, changed, candidates(root)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
