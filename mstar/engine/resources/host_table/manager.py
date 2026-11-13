"""Tables of fixed-width rows kept in host memory, a step's rows fetched by id: the resource.

A table too large for the device, of which a step reads a few rows, stays where the
checkpoint put it. Each of its parts is a read-only mapping of the part's safetensors bytes,
which every process that maps the file shares through the page cache: the ranks of a
tensor-parallel node read one copy, and each gets whole rows.

A step declares ``HostTableStep()``. Inside its forward the model hands ``fetch`` the row ids,
which copies them to the host, gathers the rows there and copies those into the step's device
buffers; ``rows`` hands them out. The ids are the device's (a decode step's depend on the
token the step before sampled), so ``fetch`` is a host sync inside the forward and waits for
the step in flight. Eager only (``can_capture`` is False).
"""

import json
import logging
import mmap
import struct
import time
import warnings
from collections.abc import Mapping
from pathlib import Path

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.host_table.config import (
    HostTable,
    HostTableConfig,
    HostTablePart,
    HostTableSpec,
    HostTableStep,
)
from mstar.engine.resources.step import StepContext
from mstar.utils.h2d import PinnedStager
from mstar.utils.hf_snapshot import resolve_snapshot_dir
from mstar.utils.profiler import PhaseLaps

logger = logging.getLogger(__name__)

# madvise's MADV_POPULATE_READ (Linux 5.14), which older ``mmap`` modules do not name
_MADV_POPULATE_READ = getattr(mmap, "MADV_POPULATE_READ", 22)
# one madvise call populates this much: the call keeps the GIL
_POPULATE_BYTES = 256 << 20


class _Shards:
    """The safetensors files of a checkpoint directory, by the tensors they hold."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        index = directory / "model.safetensors.index.json"
        # a checkpoint in one file has no index
        self._files: dict[str, str] | None = json.loads(index.read_text())["weight_map"] if index.is_file() else None
        # file -> (where its data starts, tensor name -> dtype, shape and data_offsets)
        self._headers: dict[Path, tuple[int, dict]] = {}

    def locate(self, tensor: str) -> tuple[Path, int, dict]:
        """``tensor``'s file, the byte its data starts at there, and its header entry."""
        name = "model.safetensors" if self._files is None else self._files.get(tensor)
        if name is None:
            raise KeyError(f"the checkpoint in {self._directory} holds no tensor {tensor!r}")
        path = self._directory / name
        if path not in self._headers:
            with open(path, "rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                self._headers[path] = 8 + size, json.loads(f.read(size))
        base, header = self._headers[path]
        if tensor not in header:
            raise KeyError(f"{path} holds no tensor {tensor!r}")
        return path, base + header[tensor]["data_offsets"][0], header[tensor]


def _prefault(mapped: mmap.mmap, start: int, length: int) -> None:
    """Read ``length`` bytes of the mapping from ``start`` into the page cache and this
    process's page tables."""
    end = start + length
    at = start - start % mmap.PAGESIZE
    try:
        while at < end:
            mapped.madvise(_MADV_POPULATE_READ, at, min(_POPULATE_BYTES, end - at))
            at += _POPULATE_BYTES
    except OSError as error:
        logger.warning("host table not prefaulted (%s): its rows fault in as they are first read", error)


class HostTableManager(Resource):
    """One node's tables in host memory (``config.tables``), and the device buffers a step's
    rows of them land in."""

    # holds nothing per request
    prefix_skip_safe = True
    supports_drop = True

    @classmethod
    def build(cls, spec: HostTableSpec, info: EngineResourceInfo):
        return cls(spec.config, info.device)

    def __init__(self, config: HostTableConfig, device: torch.device):
        self.config = config
        self._device = device
        self._maps: dict[Path, mmap.mmap] = {}
        start = time.perf_counter()
        shards = _Shards(resolve_snapshot_dir(config.checkpoint, config.cache_dir))
        # table -> one uint8 ``[rows, width]`` view of the mapping per part
        self._tables: dict[str, list[torch.Tensor]] = {
            name: [self._map_part(name, table, part, shards) for part in table.parts]
            for name, table in config.tables.items()
        }
        # a gather lands in pageable scratch, and the stager copies it to the device from
        # pinned memory it does not rewrite until that copy has run
        self._scratch = torch.empty(config.staging_bytes, dtype=torch.uint8)
        self._stager = PinnedStager(torch.uint8, numel=config.staging_bytes)
        # slot -> table -> a device buffer per part, as wide as the widest step yet
        self._buffers: dict[int, dict[str, list[torch.Tensor]]] = {}
        # the planned step's slot and what it has fetched
        self._slot: int | None = None
        self._rows: dict[str, tuple[torch.Tensor, ...]] = {}
        logger.info(
            "host tables: %.2f GiB of %s mapped%s in %.1f s",
            sum(part.numel() for parts in self._tables.values() for part in parts) / 2**30, sorted(self._tables),
            " and read in" if config.prefault else "", time.perf_counter() - start,
        )

    def _map_part(self, name: str, table: HostTable, part: HostTablePart, shards: _Shards) -> torch.Tensor:
        path, start, meta = shards.locate(part.tensor)
        nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
        if meta["shape"][:1] != [table.rows] or nbytes != table.rows * part.width:
            raise ValueError(
                f"host table {name!r}: {part.tensor} is {meta['dtype']} {meta['shape']} in the checkpoint, "
                f"declared as {table.rows} rows of {part.width} bytes")
        mapped = self._maps.get(path)
        if mapped is None:
            with open(path, "rb") as f:
                mapped = self._maps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if self.config.prefault:
            _prefault(mapped, start, nbytes)
        with warnings.catch_warnings():
            # torch warns that the buffer is read-only; nothing writes through this tensor
            warnings.simplefilter("ignore", UserWarning)
            rows = torch.frombuffer(mapped, dtype=torch.uint8, offset=start, count=nbytes)
        return rows.view(table.rows, part.width)

    def can_capture(self) -> bool:
        # fetch reads the ids on the host
        return False

    @property
    def force_double_buffer(self):
        # a step's rows are copied into reused buffers the step before may still be reading
        return True

    # -- the step ----------------------------------------------------------------------------

    def plan(self, step: HostTableStep, ctx: StepContext) -> None:
        self._slot = ctx.slot
        self._rows = {}

    def commit(self, step: HostTableStep, ctx: StepContext) -> None:
        # a fetch outside a step has no buffers to write
        self._slot = None

    # -- what the model calls ----------------------------------------------------------------

    def fetch(self, ids: Mapping[str, torch.Tensor]) -> None:
        """Bring the rows ``ids`` names (table -> integer row ids, any shape) to the device:
        every table's ids in one copy to the host, a gather per table there, and the rows
        copied into the step's buffers. Call it once a step, for all the tables it reads."""
        if self._slot is None:
            raise RuntimeError("host table: no step is planned; the step declares HostTableStep()")
        if not ids:
            return
        laps = PhaseLaps("res.host_table.")
        flat = [table_ids.reshape(-1) for table_ids in ids.values()]
        # the step's one host sync: it waits for whatever the ids depend on
        host = (flat[0] if len(flat) == 1 else torch.cat(flat)).cpu()
        laps.lap("sync")
        at = 0
        for (name, table_ids), one in zip(ids.items(), flat, strict=True):
            self._rows[name] = self._gather(name, host[at:at + one.numel()], table_ids.shape)
            at += one.numel()
        laps.lap("gather")

    def rows(self, name: str) -> tuple[torch.Tensor, ...]:
        """The rows the step fetched from table ``name``: per part, uint8 ``(*ids.shape,
        width)`` on the device, in the ids' order. They are the step's own: their buffers are
        written again when the step's slot comes round."""
        if name not in self._rows:
            raise KeyError(f"host table {name!r}: this step fetched {sorted(self._rows)}")
        return self._rows[name]

    def _gather(self, name: str, ids: torch.Tensor, shape: torch.Size) -> tuple[torch.Tensor, ...]:
        if name not in self._tables:
            raise KeyError(f"no host table {name!r}; this node has {sorted(self._tables)}")
        n, rows = ids.numel(), self.config.tables[name].rows
        if n:
            low, high = (int(bound) for bound in torch.aminmax(ids))
            if low < 0 or high >= rows:
                raise IndexError(f"host table {name!r} has {rows} rows; asked for ids from {low} to {high}")
        held = self._buffers.setdefault(self._slot, {})
        if name not in held or held[name][0].shape[0] < n:
            held[name] = [
                torch.empty(n, part.shape[1], dtype=torch.uint8, device=self._device)
                for part in self._tables[name]
            ]
        out = []
        for part, buffer in zip(self._tables[name], held[name], strict=True):
            width = part.shape[1]
            step = self._scratch.numel() // width
            for start in range(0, n, step):
                end = min(start + step, n)
                chunk = self._scratch[: (end - start) * width].view(end - start, width)
                torch.index_select(part, 0, ids[start:end], out=chunk)
                self._stager.copy_(buffer[start:end].view(-1), chunk.view(-1).numpy())
            out.append(buffer[:n].view(*shape, width))
        return tuple(out)

    def cleanup(self):
        self._rows = {}
        self._buffers.clear()
        # a mapping closes once the last tensor over it is gone
        self._tables.clear()
        self._maps.clear()
