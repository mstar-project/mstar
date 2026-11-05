"""Tuned kernel tables: which implementation runs, with which config, kept as data.

One JSON file per (op, device) holds the tuned shapes of that op on that device::

    mstar/kernels/tables/moe_gemm/NVIDIA_H100_80GB_HBM3.json
    {
      "format": 2,
      "op": "moe_gemm",
      "entries": [
        {
          "dtype": "bfloat16",
          "shape": {"E": 128, "K": 2048, "N": 1536, "top_k": 8},
          "provenance": {"device": "NVIDIA H100 80GB HBM3", "date": "2026-10-30",
                         "harness": "python -m benchmark.kernels.sweep moe_gemm ...", ...},
          "rows": {
            "1": {"impl": "triton", "config": {"BLOCK_SIZE_M": 16, "num_warps": 4, ...}},
            "64": {"impl": "triton", "config": {...}},
            "*": {"impl": "triton", "config": {...}}
          },
          "measured": {"1": {"us": 31.2, "default_us": 34.8}, ...}
        }
      ]
    }

A file is named for what it applies to: one GPU model (its ``torch.cuda.get_device_name``, see
:func:`file_stem`), or one architecture (``sm_90``, ``gfx942``) for configs that hold on every
GPU of it. A lookup searches each directory of :func:`table_dirs` in order, the GPU's file before
its architecture's, and takes the first entry for its dtype and shape.

Each row names an implementation of the op and, optionally, its config; what a config holds is the
implementation's business. Rows are keyed by the op's bucket (a token count): a row takes the
counts up to its key, and the ``"*"`` row, always last, every count above the others (or every
count, for an op without a bucket). An optional ``measured`` object, keyed like the rows, records
what the sweep timed, in microseconds per call.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2
TABLE_DIR = Path(__file__).parent / "tables"
TABLES_ENV = "MSTAR_KERNEL_TABLES"
# Provenance every entry carries, so a reader can tell where a choice came from and redo it.
REQUIRED_PROVENANCE = ("device", "date", "harness")
ANY_BUCKET = "*"


def table_dirs() -> tuple[Path, ...]:
    """Where tables are looked up, first match wins: the directories in
    ``MSTAR_KERNEL_TABLES`` (``os.pathsep``-separated), then the ones shipped in the package."""
    extra = os.environ.get(TABLES_ENV, "")
    return (*(Path(p).expanduser() for p in extra.split(os.pathsep) if p), TABLE_DIR)


def file_stem(name: str) -> str:
    """A GPU or architecture name as a file stem: each run of characters other than letters,
    digits, ``.``, ``-`` and ``_`` becomes one ``_`` (``NVIDIA H100 80GB HBM3`` ->
    ``NVIDIA_H100_80GB_HBM3``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


@dataclass(frozen=True)
class DeviceKey:
    """The device a table applies to: the GPU's name and its architecture."""

    name: str
    arch: str | None = None

    def stems(self) -> tuple[str, ...]:
        """File stems to try, most specific first."""
        stems = [file_stem(self.name)]
        if self.arch:
            stems.append(file_stem(self.arch))
        return tuple(dict.fromkeys(stems))


def device_key(device: torch.device | int | None = None) -> DeviceKey | None:
    """The table key of a CUDA or ROCm device (the current one by default); None elsewhere,
    where no table applies."""
    from mstar.platforms import current_platform

    if isinstance(device, torch.device):
        if device.type != "cuda":
            return None
        device = device.index
    if not (current_platform.is_cuda() or current_platform.is_rocm()):
        return None
    return DeviceKey(torch.cuda.get_device_name(device), current_platform.arch(device))


@dataclass(frozen=True)
class Choice:
    """One row: the implementation that runs and its config (None: the implementation's own)."""

    impl: str
    config: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {"impl": self.impl, **({"config": dict(self.config)} if self.config is not None else {})}

    def describe(self) -> str:
        return self.impl if self.config is None else f"{self.impl} {json.dumps(dict(self.config))}"


Rows = tuple[tuple[int | None, Choice], ...]  # ascending bucket; None is "*", last


@dataclass(frozen=True)
class TableEntry:
    """One tuned shape of one op: its rows and where they came from."""

    op: str
    dtype: str
    shape: tuple[tuple[str, int], ...]
    rows: Rows
    provenance: Mapping[str, Any]
    source: str
    # what the sweep measured per bucket, e.g. {"us": 31.2, "default_us": 34.8}; read by people
    # and later sweeps, not by lookups
    measured: Mapping[int | None, Mapping[str, float]] = field(default_factory=lambda: MappingProxyType({}))

    def impls(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(choice.impl for _, choice in self.rows))

    def buckets(self) -> tuple[int | None, ...]:
        return tuple(key for key, _ in self.rows)

    def to_json(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": dict(self.shape),
            "provenance": dict(self.provenance),
            "rows": {_bucket_key(key): choice.to_json() for key, choice in self.rows},
            **({"measured": {_bucket_key(key): dict(m) for key, m in self.measured.items()}} if self.measured else {}),
        }


def _bucket_key(bucket: int | None) -> str:
    return ANY_BUCKET if bucket is None else str(bucket)


def dtype_name(dtype: torch.dtype | str) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``; a name is checked and returned."""
    name = str(dtype).removeprefix("torch.")
    if not isinstance(getattr(torch, name, None), torch.dtype):
        raise ValueError(f"{dtype!r} is not a torch dtype")
    return name


def _keyed(raw: Any, where: str, what: str) -> list[tuple[int | None, Any]]:
    """An object keyed by bucket (a non-negative integer, or "*" last), ascending."""
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{where}: '{what}' must be a non-empty object")
    items = []
    for key, value in raw.items():
        if key == ANY_BUCKET:
            bucket = None
        elif isinstance(key, str) and key.isdigit():
            bucket = int(key)
        else:
            raise ValueError(f"{where}: {what} key {key!r} is neither a non-negative integer nor '*'")
        items.append((bucket, value))
    return sorted(items, key=lambda item: float("inf") if item[0] is None else item[0])


def _parse_rows(raw: Any, where: str) -> Rows:
    rows = []
    for bucket, row in _keyed(raw, where, "rows"):
        if not isinstance(row, Mapping) or not isinstance(row.get("impl"), str) or not row["impl"]:
            raise ValueError(f"{where}: row {_bucket_key(bucket)!r} must be an object naming its 'impl'")
        unknown = set(row) - {"impl", "config"}
        if unknown:
            raise ValueError(f"{where}: row {_bucket_key(bucket)!r} has unknown keys {sorted(unknown)}")
        config = row.get("config")
        if config is not None and not isinstance(config, Mapping):
            raise ValueError(f"{where}: row {_bucket_key(bucket)!r}'s 'config' must be an object")
        # shared by every lookup: read-only
        rows.append((bucket, Choice(row["impl"], None if config is None else MappingProxyType(dict(config)))))
    return tuple(rows)


def _parse_measured(raw: Any, where: str) -> Mapping[int | None, Mapping[str, float]]:
    measured = {}
    for bucket, times in _keyed(raw, where, "measured"):
        if not isinstance(times, Mapping) or not all(
                isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
                for k, v in times.items()):
            raise ValueError(f"{where}: measured {_bucket_key(bucket)!r} must map names to numbers")
        measured[bucket] = MappingProxyType(dict(times))
    return MappingProxyType(measured)


def parse_table(data: Any, op: str, source: str) -> tuple[TableEntry, ...]:
    """The entries of one table file's JSON, checked."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{source}: a table must be a JSON object")
    if data.get("format") != FORMAT_VERSION:
        raise ValueError(f"{source}: unsupported table format {data.get('format')!r}; expected {FORMAT_VERSION}")
    if data.get("op") != op:
        raise ValueError(f"{source}: table is for op {data.get('op')!r}, found under {op!r}")
    entries, seen = [], set()
    for i, raw in enumerate(data.get("entries") or ()):
        where = f"{source} entry {i}"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: must be an object")
        unknown = set(raw) - {"dtype", "shape", "provenance", "rows", "measured"}
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        shape = raw.get("shape")
        if not isinstance(shape, Mapping) or not all(
                isinstance(k, str) and type(v) is int for k, v in shape.items()):
            raise ValueError(f"{where}: 'shape' must map dimension names to integers")
        provenance = raw.get("provenance")
        missing = [k for k in REQUIRED_PROVENANCE if not isinstance(provenance, Mapping) or not provenance.get(k)]
        if missing:
            raise ValueError(f"{where}: provenance lacks {missing}")
        entry = TableEntry(
            op=op,
            dtype=dtype_name(raw.get("dtype", "")),
            shape=tuple(sorted(shape.items())),
            rows=_parse_rows(raw.get("rows"), where),
            provenance=MappingProxyType(dict(provenance)),
            measured=_parse_measured(raw["measured"], where) if "measured" in raw else MappingProxyType({}),
            source=source,
        )
        key = (entry.dtype, entry.shape)
        if key in seen:
            raise ValueError(f"{where}: a second entry for dtype {entry.dtype}, shape {dict(entry.shape)}")
        seen.add(key)
        entries.append(entry)
    return tuple(entries)


@functools.lru_cache(maxsize=None)
def load_table(path: Path, op: str) -> tuple[TableEntry, ...]:
    """A table file's entries, read once per process. A malformed file raises: a table that
    silently stopped applying would show up only as a slower server."""
    return parse_table(json.loads(path.read_text()), op, str(path))


def candidate_files(op: str, device: DeviceKey, dirs: tuple[Path, ...] | None = None) -> Iterator[Path]:
    """The files a lookup reads for ``op`` on ``device``, in the order it reads them."""
    for directory in table_dirs() if dirs is None else dirs:
        for stem in device.stems():
            yield directory / op / f"{stem}.json"


def find_entries(
    op: str, device: DeviceKey, dtype: str, shape: tuple[tuple[str, int], ...],
    dirs: tuple[Path, ...] | None = None,
) -> Iterator[TableEntry]:
    """Every entry for (dtype, shape), in ``candidate_files`` order; the first applies."""
    for path in candidate_files(op, device, dirs):
        if path.is_file():
            for entry in load_table(path, op):
                if entry.dtype == dtype and entry.shape == shape:
                    yield entry


def write_entry(path: Path, entry: TableEntry) -> None:
    """Put ``entry`` into the table at ``path``, replacing the entry for the same (dtype, shape) if
    there is one. Entries are kept sorted, so a diff shows only what changed."""
    entries = list(parse_table(json.loads(path.read_text()), entry.op, str(path))) if path.is_file() else []
    entries = [e for e in entries if (e.dtype, e.shape) != (entry.dtype, entry.shape)]
    entries.append(entry)
    entries.sort(key=lambda e: (e.dtype, e.shape))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_table(entry.op, entries))
    load_table.cache_clear()


def dump_table(op: str, entries: list[TableEntry]) -> str:
    """A table file's text: indented, with one line per row so a diff shows each bucket."""
    def compact(value) -> str:
        return json.dumps(value, separators=(", ", ": "))

    def lines(rows: dict) -> str:
        return ",\n".join(f"        {json.dumps(key)}: {compact(row)}" for key, row in rows.items())

    blocks = []
    for entry in entries:
        data = entry.to_json()
        rows, measured = lines(data["rows"]), lines(data.get("measured", {}))
        blocks.append(
            "    {\n"
            f'      "dtype": {json.dumps(data["dtype"])},\n'
            f'      "shape": {compact(data["shape"])},\n'
            f'      "provenance": {compact(data["provenance"])},\n'
            f'      "rows": {{\n{rows}\n      }}'
            + (f',\n      "measured": {{\n{measured}\n      }}\n' if "measured" in data else "\n")
            + "    }"
        )
    body = ",\n".join(blocks)
    return (f'{{\n  "format": {FORMAT_VERSION},\n  "op": {json.dumps(op)},\n'
            f'  "entries": [\n{body}\n  ]\n}}\n')
