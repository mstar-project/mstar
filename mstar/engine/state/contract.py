"""What a model declares about the state it keeps between steps.

Declarations are values: mappings are read-only, sets are frozensets, and everything hashes.
"""

from collections.abc import Iterator, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar, Literal, NamedTuple, get_args, get_origin

import torch

from mstar.engine.resources.kv.config import Span
from mstar.engine.resources.step import Segment


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

    # whether what a read sees depends on its query's log position, which another array's queries don't carry here
    uses_query_position: ClassVar[bool]
    # whether a read sees a row's adds of each step as a whole, so splitting them across steps changes what it sees
    sees_step_whole: ClassVar[bool]


@dataclass(frozen=True)
class Causal(Mask):
    """Sees {k : k <= q}."""

    uses_query_position = True
    sees_step_whole = False


@dataclass(frozen=True)
class Window(Mask):
    """Sees {k : q - size < k <= q}."""

    uses_query_position = True
    sees_step_whole = False

    size: int


@dataclass(frozen=True)
class Block(Mask):
    """Sees the entries of q's own segment in this step, on an array with ``max_len`` 0."""

    uses_query_position = False
    sees_step_whole = True


@dataclass(frozen=True)
class Bidirectional(Mask):
    """Sees {k : k < n}, where n is the length of q's row on this array after this layer's latest add: every entry the
    row keeps and every entry this step adds for it. For a read with ``queries``, q's row is the row paired with q's
    segment, on this array's label. It uses no query position. On an array that keeps entries; ``Block`` is the
    within-step form."""

    uses_query_position = False
    sees_step_whole = True


class Op:
    """A kernel the engine plans before the step and runs on an array's entries.

    ``reads`` and ``writes`` name the entry fields it reads and overwrites in place, ``adds`` the fields of the entries
    it appends, and ``mutates`` the arguments of its call that it changes in place.
    """

    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ()
    adds: ClassVar[tuple[str, ...]] = ()
    mutates: ClassVar[tuple[str, ...]] = ()

    def check(self, where: str, entry: Entry) -> None:
        """Raise if the kernel can't run as configured on ``entry``, in a message that opens with ``where``. ``entry``
        holds every field the op names."""


@dataclass(frozen=True)
class Attend(Op):
    reads = ("k", "v")
    q_heads: int    # a multiple of the heads of the entry's k and v

    def check(self, where: str, entry: Entry) -> None:
        if not _is_positive(self.q_heads):
            raise ValueError(f"{where}.q_heads must be a positive int, got {self.q_heads!r}")
        k, v = entry.tensors["k"], entry.tensors["v"]
        if len(k.shape) != 2 or len(v.shape) != 2 or k.shape[0] != v.shape[0]:
            raise ValueError(f"{where}: Attend needs entry fields k (h, d_k) and v (h, d_v)")
        if k.dtype != v.dtype or k.shard != v.shard or k.shard not in (0, None):
            raise ValueError(f"{where}: Attend needs k and v of one dtype and one shard, 0 or None; "
                             f"got {k.dtype}, {v.dtype} and {k.shard}, {v.shard}")
        if self.q_heads % k.shape[0]:
            raise ValueError(f"{where}: q_heads {self.q_heads} is not a multiple of k's {k.shape[0]} heads")


@dataclass(frozen=True, kw_only=True)
class CausalConv(Op):
    reads = ("conv",)
    writes = ("conv",)
    activation: Literal["silu"] | None = "silu"

    def check(self, where: str, entry: Entry) -> None:
        if len(entry.tensors["conv"].shape) != 2:
            raise ValueError(f"{where}: CausalConv needs entry field conv of shape (channels, kernel - 1)")


@dataclass(frozen=True, kw_only=True)
class GatedDeltaNet(Op):
    reads = ("state",)
    writes = ("state",)
    qk_l2norm: bool = True
    scale: float | None = None    # None: head_k_dim ** -0.5

    def check(self, where: str, entry: Entry) -> None:
        if len(entry.tensors["state"].shape) != 3:
            raise ValueError(f"{where}: GatedDeltaNet needs entry field state of shape (h_v, d_k, d_v)")


@dataclass(frozen=True, kw_only=True)
class Read:
    """A view of the entries a row has kept. ``mask`` says which entries each query can still reach, so the engine
    can free the rest and tell when two requests can share a prefix. A read leaves the entries as they were."""

    mask: Mask
    op: Op | None = None
    layers: frozenset[int] | None = None    # None: all of the array's layers
    walks: frozenset[str] | None = None     # None: all walks
    queries: str | None = None              # the array whose segments lay out the queries; None: this array
    packed: bool = True                     # queries may come as (tokens, heads, head_dim); False: only as rows
    segmented: bool = False                 # a request's segments lie end to end in one row

    def __post_init__(self):
        if not isinstance(self.mask, Mask):
            raise ValueError(f"Read.mask must be a Mask, got {self.mask!r}")
        unset = [name for name in Mask.__annotations__ if not hasattr(type(self.mask), name)]
        if unset:
            raise ValueError(f"Read.mask: {type(self.mask).__name__} leaves {unset} unset, which every mask declares")
        if not (self.op is None or isinstance(self.op, Op)):
            raise ValueError(f"Read.op must be None or an Op, got {self.op!r}")
        if self.op is not None and (self.op.writes or self.op.adds):
            raise ValueError(f"Read.op: a read changes no field, and {type(self.op).__name__} writes "
                             f"{list(self.op.writes)} and adds {list(self.op.adds)}")
        if self.segmented and (self.packed or self.queries is not None or not isinstance(self.mask, Block)):
            raise ValueError(f"Read.segmented lays a request's segments end to end in one row, which takes Block(), "
                             f"packed=False and no queries, and got mask={self.mask!r}, packed={self.packed}, "
                             f"queries={self.queries!r}")
        for name in ("layers", "walks"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, frozenset(getattr(self, name)))


@dataclass(frozen=True, kw_only=True)
class Rope:
    theta: float = 10000.0
    rotary_dim: int | None = None              # None: head_dim
    interleave: bool = False
    scale: float = 1.0
    # Llama 3.1 scaling: all three or none
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    old_context_len: int | None = None
    dtype: torch.dtype | None = None           # None: q's
    table_dtype: torch.dtype | None = None     # None: angles in the kernel


@dataclass(frozen=True, kw_only=True)
class Positions:
    """Each row's position on the array, which the model rotates or embeds its next tokens at. It reads no entries,
    and it lives on the host and moves when a step is planned: by the row's span, by ``ArrayStep.advance`` or by
    ``ArrayStep.position_ids``. It forks and takes prefix hits with its array."""

    rope: Rope | None = None    # None: a counter only


@dataclass(frozen=True)
class Array:
    """Per request, per label and per declared layer, entries at log positions 0 to n - 1, with n <= ``max_len``.

    An append-only array only grows, and a rewritable one also takes writes in place. All layers of an array have
    the same length at the end of every step. ``nodes`` lists who may touch the array, and the engine places it.
    ``updates`` are kernels that replace a Value's entry with its next version. The version they overwrite is gone,
    so prefix caching and undo save a copy before one runs.
    With ``max_len`` 0 an array keeps nothing between steps, so every add is scratch.
    """

    name: str
    _: KW_ONLY
    nodes: frozenset[str]
    layers: frozenset[int]
    entry: Entry
    max_len: int
    reads: Mapping[str, Read] = EMPTY    # empty: read whole
    updates: Mapping[str, Op] = EMPTY
    rewritable: bool = False
    labels: frozenset[str] = frozenset({"main"})
    positions: Positions | None = None

    def __post_init__(self):
        object.__setattr__(self, "nodes", frozenset(self.nodes))
        object.__setattr__(self, "layers", frozenset(self.layers))
        object.__setattr__(self, "labels", frozenset(self.labels))
        object.__setattr__(self, "reads", FrozenMap(self.reads))
        object.__setattr__(self, "updates", FrozenMap(self.updates))
        where = f"Array {self.name!r}"
        if not self.nodes:
            raise ValueError(f"{where}.nodes must be non-empty")
        if not all(type(layer) is int and layer >= 0 for layer in self.layers):
            raise ValueError(f"{where}.layers must be ints >= 0, got {sorted(self.layers, key=repr)}")
        if not self.layers and not is_value(self):
            raise ValueError(f"{where}.layers is empty, and only a Value may declare no layers")
        if not self.labels:
            raise ValueError(f"{where}.labels must be non-empty")
        if not (type(self.max_len) is int and self.max_len >= 0):
            raise ValueError(f"{where}.max_len must be an int >= 0, got {self.max_len!r}")
        self._check_entry()
        for key, read in self.reads.items():
            self._check_read(key, read)
        for key, update in self.updates.items():
            self._check_update(key, update)

    def _check_entry(self) -> None:
        # a layer reads nothing from an array with no layers and no reads, so an axis may be 0
        least = 1 if self.layers or self.reads else 0
        for name, tensor in self.entry.tensors.items():
            where = f"Array {self.name!r}.entry.tensors[{name!r}]"
            if not (isinstance(tensor.shape, tuple) and tensor.shape and all(
                type(axis) is int and axis >= least for axis in tensor.shape
            )):
                raise ValueError(f"{where}.shape must be a non-empty tuple of ints >= {least}, got {tensor.shape!r}")
            if tensor.shard is not None and not (type(tensor.shard) is int and 0 <= tensor.shard < len(tensor.shape)):
                raise ValueError(f"{where}.shard must be None or an axis of shape {tensor.shape}, got {tensor.shard!r}")
        if self.entry.host and not is_value(self):
            raise ValueError(f"Array {self.name!r}.entry.host: only a Value (rewritable, max_len 1) holds host fields")
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
        if read.queries is not None and read.mask.uses_query_position:
            raise ValueError(f"{where}.queries: a read of another array's queries takes a mask that uses no query "
                             f"position, and {type(read.mask).__name__} does")
        if read.queries is not None and read.op is None:
            raise ValueError(f"{where}.queries: a read with no op lays out no queries")
        if read.queries == self.name:
            raise ValueError(f"{where}.queries names this array, and a read without queries lays out its own queries; "
                             "leave it None")
        if read.op is not None:
            self._check_op(f"{where}.op", read.op)

    def _check_update(self, key: str, update: Op) -> None:
        where = f"Array {self.name!r}.updates[{key!r}]"
        if not is_value(self):
            raise ValueError(f"{where}: only a Value (rewritable, max_len 1) takes an update")
        if not isinstance(update, Op):
            raise ValueError(f"{where} must be an Op, got {update!r}")
        if not update.writes or update.adds:
            raise ValueError(f"{where}: an update overwrites fields in place, and {type(update).__name__} writes "
                             f"{list(update.writes)} and adds {list(update.adds)}")
        self._check_op(where, update)

    def _check_op(self, where: str, op: Op) -> None:
        missing = sorted({*op.reads, *op.writes, *op.adds} - self.entry.tensors.keys())
        if missing:
            raise ValueError(f"{where}: {type(op).__name__} needs entry fields {missing}")
        op.check(where, self.entry)


def is_value(array: Array) -> bool:
    return array.rewritable and array.max_len == 1


def read_layers(array: Array, read: Read) -> frozenset[int]:
    """The layers ``read`` reads of ``array``: its own, or all of the array's."""
    return read.layers or array.layers


def Value(name, *, nodes, layers, entry, updates=EMPTY) -> Array:
    """A rewritable array of one entry."""
    return Array(name, nodes=nodes, layers=layers, entry=entry, updates=updates, max_len=1, rewritable=True)


@dataclass(frozen=True)
class Sampler:
    name: str
    _: KW_ONLY
    nodes: frozenset[str]
    vocab_size: int
    penalty: bool = True
    min_p: bool = False
    eos_ids: tuple[int, ...] = ()    # the ids min_tokens bars
    draws: int = 1                   # calls to sample in each step that lists it

    def __post_init__(self):
        object.__setattr__(self, "nodes", frozenset(self.nodes))
        object.__setattr__(self, "eos_ids", tuple(self.eos_ids))
        if not self.nodes:
            raise ValueError(f"Sampler {self.name!r}.nodes must be non-empty")


@dataclass(frozen=True, kw_only=True)
class SampleSettings:
    """A request's settings for one sampler, keyed by the sampler's name in `get_request_resource_configs`."""

    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    repetition_window: int = 0
    penalize_prompt: bool = True
    min_tokens: int = 0
    ignore_eos: bool = False
    top_p_first: bool = False
    top_p_min_keep: int = 1
    return_logprobs: bool = False
    logprobs_mode: str = "raw"


@dataclass(frozen=True, kw_only=True)
class Fork:
    src: str
    dst: str
    at: Literal["start", "end"]           # before the step plans or writes, or at commit
    rows: frozenset[int] | None = None    # None: every row

    def __post_init__(self):
        if self.at not in ("start", "end"):
            raise ValueError(f"Fork.at must be 'start' or 'end', got {self.at!r}")


@dataclass(frozen=True, kw_only=True)
class ArrayStep:
    """What one step does to one array, under `SubmoduleStep.state`."""

    segments: tuple[Segment, ...] | None = None    # None: the step's segments
    keep: bool = True                              # False: every add this step is scratch
    forks: tuple[Fork, ...] = ()
    placeholders: frozenset[tuple[int, str]] = frozenset()    # (row, label): holds a place, reads and keeps nothing
    eager_add: bool = False                        # this step's adds run outside the captured graph
    fixed_layout: bool = False                     # replays of its capture bucket plan the capture's layout
    advance: tuple[int, ...] | None = None         # positions per row; None: the span
    position_ids: torch.Tensor | None = None       # explicit ids, (tokens,) in segment order
    host: Mapping[tuple[int, str, int], Mapping[str, object]] = EMPTY    # (row, label, layer) -> fields, at commit


@dataclass(frozen=True, kw_only=True)
class SampleStep:
    """What one step does with one sampler, under `SubmoduleStep.state`."""

    apply_penalty: bool = True
    apply_filters: bool = True
    prompt_tokens: Mapping[int, torch.Tensor] = EMPTY    # row -> ids the penalty counts
    kept_rows: frozenset[int] | None = None              # None: every row's sample is kept


@dataclass(frozen=True, kw_only=True)
class Layout:
    """What a request writes to one label of one array, from `process_prompt`'s ``state_layout``."""

    spans: tuple[Span, ...] = ()
    growth: tuple[tuple[str, int], ...] = ()    # (loop, entries per iteration)
    scratch: int = 0                            # most one scratch step writes past the kept length


@dataclass(frozen=True, kw_only=True)
class PagedNHD:
    """A storage format a state op can address: pages of (tokens, heads, head_dim)."""

    dtypes: frozenset[torch.dtype] | None = None    # None: any
    devices: frozenset[str] | None = None           # None: any

    def __post_init__(self):
        for name in ("dtypes", "devices"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, frozenset(getattr(self, name)))


class PagedNHDTarget(NamedTuple):
    """Where a state op writes this step's entries on one layer of a `PagedNHD` array."""

    cache: torch.Tensor
    page_of_token: torch.Tensor
    slot_of_token: torch.Tensor
    count: int    # tokens to write


class Reached(NamedTuple):
    """What a read with no op returns: each field's reached entries packed by row, and where each row starts."""

    fields: Mapping[str, torch.Tensor]
    row_offsets: torch.Tensor
