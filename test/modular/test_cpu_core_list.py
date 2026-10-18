"""test/cpu-core.txt names every CPU Core module once, and each one exists.

pytest collects a repeated path once, so a duplicate line is harmless at run
time, but it hides an entry a rebase dropped. A path that no longer exists makes
the whole CPU Core job fail at collection.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPU_CORE = ROOT / "test" / "cpu-core.txt"


def _entries() -> list[str]:
    return [line.strip() for line in CPU_CORE.read_text().splitlines() if line.strip()]


def test_cpu_core_entries_are_unique():
    entries = _entries()
    repeated = sorted({e for e in entries if entries.count(e) > 1})
    assert not repeated, f"listed more than once in test/cpu-core.txt: {repeated}"


def test_cpu_core_entries_exist():
    missing = [e for e in _entries() if not (ROOT / e).is_file()]
    assert not missing, f"test/cpu-core.txt names files that don't exist: {missing}"
