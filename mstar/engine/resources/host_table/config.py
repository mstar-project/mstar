"""What a model declares for tables of fixed-width rows kept in host memory: config, spec, step.

Kept free of the manager so a submodule can declare a step without importing it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from mstar.engine.resources.spec import NodeResourceSpec, check_sizes
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class HostTablePart(NamedTuple):
    """One checkpoint tensor ``[rows, ...]`` of a table, and the bytes one of its rows takes."""
    tensor: str
    width: int


@dataclass(frozen=True)
class HostTable:
    """A table of ``rows`` rows. A row is one row of each part (quantized values and their
    scales, say), fetched together."""
    rows: int
    parts: tuple[HostTablePart, ...]


@dataclass
class HostTableConfig:
    # the checkpoint whose safetensors files hold the tables: a directory, or a Hub id
    # resolved through ``cache_dir``
    checkpoint: str
    # table name -> its rows
    tables: dict[str, HostTable]
    cache_dir: str | None = None
    # bytes of rows one copy to the device stages; the pinned arena is eight of them
    staging_bytes: int = 4 << 20
    # read and map every page of the tables at build. Without it a row's first read is a
    # page fault, from the disk when the file is not in the page cache.
    prefault: bool = True

    def __post_init__(self):
        if not self.tables:
            raise ValueError("a host table resource must declare a table")
        for name, table in self.tables.items():
            if not table.parts:
                raise ValueError(f"host table {name!r} must declare a part")
            check_sizes({f"tables[{name!r}].rows": table.rows})
            check_sizes({f"tables[{name!r}] part {part.tensor!r} width": part.width for part in table.parts})
        self.check_staging()

    def check_staging(self) -> None:
        """A staged copy must hold at least one row of the widest part."""
        widest = max(part.width for table in self.tables.values() for part in table.parts)
        check_sizes({"staging_bytes": self.staging_bytes}, smallest=widest)


@dataclass
class HostTableSpec(NodeResourceSpec):
    config: HostTableConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.host_table.manager import HostTableManager

        return HostTableManager

    def apply_yaml_overrides(self, staging_bytes: int | None = None, prefault: bool | None = None):
        """What the box decides, not the model: the pinned memory a copy stages, and whether
        the tables are read into the page cache at startup."""
        if staging_bytes is not None:
            self.config.staging_bytes = staging_bytes
            self.config.check_staging()
        if prefault is not None:
            if not isinstance(prefault, bool):
                raise ValueError(f"prefault must be true or false, got {prefault!r}")
            self.config.prefault = prefault


@dataclass(frozen=True)
class HostTableStep(ResourceStep):
    pass
