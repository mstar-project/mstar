"""What a model declares about the state it keeps between steps.

Declarations are values: mappings are read-only, sets are frozensets, and everything hashes.
"""

from collections.abc import Iterator, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Literal, get_args, get_origin

import torch


class FrozenMap(Mapping):
    """A read-only copy of a mapping, in its order, that hashes, which `MappingProxyType` doesn't."""

    __slots__ = ("_items",)

    def __init__(self, items=()):
        self._items = dict(items)

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(frozenset(self._items.items()))

    def __repr__(self) -> str:
        return f"FrozenMap({self._items!r})"


EMPTY = FrozenMap()


def _is_positive(value) -> bool:
    return type(value) is int and value >= 1


def _is_host_type(kind) -> bool:
    if get_origin(kind) is tuple:
        return all(arg is Ellipsis or _is_host_type(arg) for arg in get_args(kind))
    return kind in (int, float, bool, str, bytes)


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]       # global, before sharding
    _: KW_ONLY
    dtype: torch.dtype | None    # math dtype, None is the engine's compute dtype
    shard: int | None = None     # axis split across TP ranks, None is replicated


@dataclass(frozen=True)
class Entry:
    tensors: Mapping[str, TensorType]
    _: KW_ONLY
    host: Mapping[str, type] = EMPTY    # int, float, bool, str, bytes, or tuples of these

    def __post_init__(self):
        object.__setattr__(self, "tensors", FrozenMap(self.tensors))
        object.__setattr__(self, "host", FrozenMap(self.host))


def KVEntry(*, heads: int, head_dim: int) -> Entry:
    """One token's keys and values, each of shape (``heads``, ``head_dim``) and sharded on its heads."""
    t = TensorType((heads, head_dim), dtype=None, shard=0)
    return Entry({"k": t, "v": t})


class Mask:
    """Which entries a read at log position q sees. An entry's index is its log position."""


@dataclass(frozen=True)
class Causal(Mask):
    """Sees {k : k <= q}."""


@dataclass(frozen=True)
class Window(Mask):
    """Sees {k : q - size < k <= q}."""

    size: int


@dataclass(frozen=True)
class Attend:
    q_heads: int    # a multiple of the heads of the entry's k and v


# An update reads and writes one entry field in one kernel.
@dataclass(frozen=True, kw_only=True)
class CausalConv:
    activation: Literal["silu"] | None = "silu"


@dataclass(frozen=True, kw_only=True)
class GatedDeltaNet:
    qk_l2norm: bool = True
    scale: float | None = None    # None: head_k_dim ** -0.5


@dataclass(frozen=True, kw_only=True)
class Read:
    mask: Mask
    op: Attend | None = None
    layers: frozenset[int] | None = None    # None: all of the array's layers
    walks: frozenset[str] | None = None     # None: all walks

    def __post_init__(self):
        for name in ("layers", "walks"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, frozenset(getattr(self, name)))


@dataclass(frozen=True)
class Array:
    """Per request, per label and per declared layer, entries at log positions 0 to n - 1, with n <= ``max_len``.

    An append-only array only grows, and a rewritable one also takes writes in place. All layers of an array have
    the same length at the end of every step. ``nodes`` lists who may touch the array, and the engine places it.
    """

    name: str
    _: KW_ONLY
    nodes: frozenset[str]
    layers: frozenset[int]
    entry: Entry
    max_len: int
    reads: Mapping[str, Read] = EMPTY    # empty: read whole
    updates: Mapping[str, CausalConv | GatedDeltaNet] = EMPTY
    rewritable: bool = False

    def __post_init__(self):
        object.__setattr__(self, "nodes", frozenset(self.nodes))
        object.__setattr__(self, "layers", frozenset(self.layers))
        object.__setattr__(self, "reads", FrozenMap(self.reads))
        object.__setattr__(self, "updates", FrozenMap(self.updates))
        where = f"Array {self.name!r}"
        if not self.nodes:
            raise ValueError(f"{where}.nodes must be non-empty")
        if not self.layers or not all(type(layer) is int and layer >= 0 for layer in self.layers):
            raise ValueError(f"{where}.layers must be non-empty ints >= 0, got {sorted(self.layers, key=repr)}")
        if not _is_positive(self.max_len):
            raise ValueError(f"{where}.max_len must be a positive int, got {self.max_len!r}")
        self._check_entry()
        for key, read in self.reads.items():
            self._check_read(key, read)
        for key, update in self.updates.items():
            self._check_update(key, update)

    def _check_entry(self) -> None:
        for name, tensor in self.entry.tensors.items():
            where = f"Array {self.name!r}.entry.tensors[{name!r}]"
            if not (isinstance(tensor.shape, tuple) and tensor.shape and all(map(_is_positive, tensor.shape))):
                raise ValueError(f"{where}.shape must be a non-empty tuple of positive ints, got {tensor.shape!r}")
            if tensor.shard is not None and not (type(tensor.shard) is int and 0 <= tensor.shard < len(tensor.shape)):
                raise ValueError(f"{where}.shard must be None or an axis of shape {tensor.shape}, got {tensor.shard!r}")
        for name, kind in self.entry.host.items():
            if not _is_host_type(kind):
                raise ValueError(f"Array {self.name!r}.entry.host[{name!r}] must be int, float, bool, str, bytes or a "
                                 f"tuple of these, got {kind!r}")

    def _check_read(self, key: str, read: Read) -> None:
        where = f"Array {self.name!r}.reads[{key!r}]"
        if not isinstance(read, Read):
            raise ValueError(f"{where} must be a Read, got {read!r}")
        for name in ("layers", "walks"):
            if getattr(read, name) is not None and not getattr(read, name):
                raise ValueError(f"{where}.{name} must be None or non-empty")
        if read.layers is not None and not read.layers <= self.layers:
            raise ValueError(f"{where}.layers has {sorted(read.layers - self.layers)}, outside the array's layers")
        if isinstance(read.mask, Window) and not _is_positive(read.mask.size):
            raise ValueError(f"{where}.mask.size must be a positive int, got {read.mask.size!r}")
        if not isinstance(read.op, Attend):
            return
        if not _is_positive(read.op.q_heads):
            raise ValueError(f"{where}.op.q_heads must be a positive int, got {read.op.q_heads!r}")
        k, v = self.entry.tensors.get("k"), self.entry.tensors.get("v")
        if k is None or v is None or len(k.shape) != 2 or len(v.shape) != 2 or k.shape[0] != v.shape[0]:
            raise ValueError(f"{where}.op: Attend needs entry fields k (h, d_k) and v (h, d_v)")
        if k.dtype != v.dtype or k.shard != v.shard or k.shard not in (0, None):
            raise ValueError(f"{where}.op: Attend needs k and v of one dtype and one shard, 0 or None; "
                             f"got {k.dtype}, {v.dtype} and {k.shard}, {v.shard}")
        if read.op.q_heads % k.shape[0]:
            raise ValueError(f"{where}.op: q_heads {read.op.q_heads} is not a multiple of k's {k.shape[0]} heads")

    def _check_update(self, key: str, update: CausalConv | GatedDeltaNet) -> None:
        where = f"Array {self.name!r}.updates[{key!r}]"
        if not is_value(self):
            raise ValueError(f"{where}: only a Value (rewritable, max_len 1) takes an update")
        if isinstance(update, CausalConv):
            field, rank, shape = "conv", 2, "(channels, kernel - 1)"
        elif isinstance(update, GatedDeltaNet):
            field, rank, shape = "state", 3, "(h_v, d_k, d_v)"
        else:
            raise ValueError(f"{where} must be a CausalConv or a GatedDeltaNet, got {update!r}")
        tensor = self.entry.tensors.get(field)
        if tensor is None or len(tensor.shape) != rank:
            raise ValueError(f"{where}: {type(update).__name__} needs entry field {field} of shape {shape}")


def is_value(array: Array) -> bool:
    return array.rewritable and array.max_len == 1


def Value(name, *, nodes, layers, entry, updates=EMPTY) -> Array:
    """A rewritable array of one entry."""
    return Array(name, nodes=nodes, layers=layers, entry=entry, updates=updates, max_len=1, rewritable=True)
