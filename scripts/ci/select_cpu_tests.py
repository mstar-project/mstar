#!/usr/bin/env python3
"""Select the Mac CPU smoke suite from the shared core manifest."""

import sys
from pathlib import Path


def read_manifest(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def select_tests(core_paths, excluded_paths):
    """Preserve core order and reject stale or ambiguous exclusions."""
    for name, paths in (("core", core_paths), ("exclusion", excluded_paths)):
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate paths in {name} manifest")
    unknown = set(excluded_paths) - set(core_paths)
    if unknown:
        raise ValueError(f"Excluded paths absent from core manifest: {', '.join(sorted(unknown))}")
    excluded = set(excluded_paths)
    selected = [path for path in core_paths if path not in excluded]
    if not selected:
        raise ValueError("CPU smoke selection is empty")
    return selected


def main():
    root = Path(__file__).resolve().parents[2]
    try:
        core = read_manifest(root / "test/cpu-core.txt")
        excluded = read_manifest(root / "test/cpu-mac-exclude.txt")
        selected = select_tests(core, excluded)
        for path in core:
            if not (root / path).is_file():
                raise ValueError(f"Core test file does not exist: {path}")
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
    print(f"Selected {len(selected)} of {len(core)} core modules.", file=sys.stderr)
    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
