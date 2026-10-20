#!/usr/bin/env python3
"""Print the pytest arguments for the Mac CPU smoke suite.

The paths are the ones the GitHub CPU Core job runs (.github/workflows/ci.yml);
there is no list of test files. New modules under these paths are included
automatically, and modules that need a GPU, weights or an optional package
skip themselves. Every other test module in the repository is listed in
NOT_CPU_CORE with the place it runs instead; test/ci/test_cpu_selection.py
fails on a test module that is in neither.
"""

import sys
from pathlib import Path, PurePosixPath

CPU_TEST_PATHS = (
    "test/modular",
    "test/distributed",
    "test/chatterbox",
    "test/kokoro",
    "test/omnivoice",
    "test/sampling_test",
    "mstar/model/*/tests",
    "benchmark/worker_phases",
)
NOT_CPU_CORE = {
    "test/ci": "the build job runs it with unittest, without PyTorch",
    "test/rust": "the rust-transport job runs it against the built mstar_rust extension",
    "test/integration": "needs a GPU and real weights; skips without the checkpoint",
    "test/pi05/batch_test.py": "a load-test script named like a test module; it has no tests",
}
IGNORE = "--ignore="


def read_manifest(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def expand_paths(root, patterns=CPU_TEST_PATHS):
    """Return the CPU test paths with each glob replaced by the directories it matches."""
    paths = []
    for pattern in patterns:
        if "*" not in pattern:
            paths.append(pattern)
            continue
        matches = sorted(p.relative_to(root).as_posix() for p in root.glob(pattern) if p.is_dir())
        if not matches:
            raise ValueError(f"CPU test path matches nothing: {pattern}")
        paths.extend(matches)
    return paths


def is_collected(path, collected_paths):
    path = PurePosixPath(path)
    return any(path == PurePosixPath(c) or PurePosixPath(c) in path.parents for c in collected_paths)


def select_tests(collected_paths, excluded_paths):
    """Return the collected paths without the Mac exclusions.

    ``--ignore`` does not drop a path given on the command line, so an excluded
    file that is itself a collected path is removed instead.
    """
    if len(excluded_paths) != len(set(excluded_paths)):
        raise ValueError("Duplicate paths in exclusion manifest")
    unknown = [path for path in excluded_paths if not is_collected(path, collected_paths)]
    if unknown:
        raise ValueError(f"Excluded paths absent from CPU test paths: {', '.join(sorted(unknown))}")
    excluded = set(excluded_paths)
    selected = [path for path in collected_paths if path not in excluded]
    if not selected:
        raise ValueError("CPU smoke selection is empty")
    return selected + [IGNORE + path for path in excluded_paths if path not in collected_paths]


def main():
    root = Path(__file__).resolve().parents[2]
    try:
        excluded = read_manifest(root / "test/cpu-mac-exclude.txt")
        paths = expand_paths(root)
        selected = select_tests(paths, excluded)
        for path in paths:
            if not (root / path).exists():
                raise ValueError(f"CPU test path does not exist: {path}")
        for path in excluded:
            if not (root / path).is_file():
                raise ValueError(f"Excluded test file does not exist: {path}")
    except (OSError, ValueError) as error:
        print(f"CPU smoke selection: {error}", file=sys.stderr)
        return 1
    print(
        "Mac CPU smoke excludes compiled CPU FlexAttention (requires AVX2) "
        "and video encoding (PyAV ARM crash); see CONTRIBUTING.md:",
        file=sys.stderr,
    )
    for path in excluded:
        print(f"  {path}", file=sys.stderr)
    print(f"Excluded {len(excluded)} modules from the CPU test paths.", file=sys.stderr)
    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
