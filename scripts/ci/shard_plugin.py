"""A pytest plugin for CI's shards: with MSTAR_CI_SHARD=i/n set, pytest collects only shard i's test files.

CPU Core runs as two shards on two runners and loads this with `-p shard_plugin` (.github/workflows/ci.yml,
PYTHONPATH=scripts/ci). Each test file goes whole to one shard, so its module fixtures run once, and which
shard is a stable hash of its path: no list of files to keep. The salt makes the shards' test seconds match
within 0.2% on four CPU Core runs of 2026-11-10 (from their junit reports); a new file lands on either side at
random. Without the variable, every file is collected. Plain Python, so test/ci checks the split
without pytest.
"""

import os
import zlib

SALT = "522:"
TEST_FILES = ("test_*.py", "*_test.py")  # pytest's default python_files


def shard_of(path, shards):
    """The shard, from 0, of the test file at `path` (relative to the repository root, with "/")."""
    return zlib.crc32((SALT + path).encode()) % shards


def parse_shard(spec):
    """MSTAR_CI_SHARD's "i/n" as (i - 1, n), or None when unset."""
    if not spec:
        return None
    index, _, shards = spec.partition("/")
    index, shards = int(index), int(shards)
    if not 1 <= index <= shards:
        raise ValueError(f"MSTAR_CI_SHARD={spec!r}: want i/n with 1 <= i <= n")
    return index - 1, shards


def pytest_ignore_collect(collection_path, config):
    """Skip a test file of another shard before it is imported; leave everything else to pytest."""
    shard = parse_shard(os.environ.get("MSTAR_CI_SHARD", ""))
    if shard is None or not collection_path.is_file() or not any(
            collection_path.match(pattern) for pattern in TEST_FILES):
        return None
    path = collection_path.relative_to(config.rootpath).as_posix()
    return True if shard_of(path, shard[1]) != shard[0] else None
