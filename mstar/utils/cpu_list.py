"""Linux CPU list strings (``0-31,128-159``), as sysfs and ``taskset`` write them."""
from collections.abc import Iterable


def parse_cpu_list(text: str) -> frozenset[int]:
    """The CPUs in a list such as ``0-3,8,10-11``; empty for an empty string."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return frozenset(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """``cpus`` as ranges, the inverse of ``parse_cpu_list``."""
    ordered = sorted(set(cpus))
    parts = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        parts.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(parts)
