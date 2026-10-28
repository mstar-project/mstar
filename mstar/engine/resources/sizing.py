"""How big each pool is, what a request holds of it, and how many requests that makes.

A pool is a resource that allocates its storage up front as equal units: a paged
KV cache's pages, a recurrent or bounded pool's slots, a ring's sessions. Its
config says what a unit costs and what one request holds, so the arithmetic runs
on declarations: in the conductor before any worker exists, and in the engine
before anything is allocated.

Kept free of the managers, like the configs that implement it.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import torch

from mstar.engine.resources.spec import NodeResourceSpec

if TYPE_CHECKING:
    from mstar.model.base import Model

logger = logging.getLogger(__name__)


class PoolConfig(ABC):
    """A resource config whose resource allocates its storage as equal units."""

    #: what a unit is called in logs and messages
    pool_unit: ClassVar[str]
    #: the ``resources:`` key that sizes the pool
    size_key: ClassVar[str]
    #: whether the engine sizes the pool for the deployment's requests when neither the
    #: model nor the deployment sized it
    derivable: ClassVar[bool] = True

    @property
    def sized(self) -> bool:
        """Whether the model or the deployment set the size. An unsized pool holds a
        default until the engine sizes it (``size_pool``)."""
        return True

    def size_pool(self, units: int) -> None:
        """Take the engine's size for an unsized pool."""
        raise NotImplementedError(f"{type(self).__name__} is not sized by the engine")

    @property
    @abstractmethod
    def pool_units(self) -> int | None:
        """Units allocated, the reserved ones included. None for a store that
        grows with each request instead of drawing from a pool."""

    @property
    def reserved_units(self) -> int:
        """Units no request is given: a sink page or slot, a padding session."""
        return 0

    @abstractmethod
    def unit_bytes(self, kv_dtype: torch.dtype, num_shards: int = 1) -> int:
        """Device bytes of one unit on one rank of a ``num_shards``-way instance.
        ``kv_dtype`` is the engine's, for whatever the config leaves to it."""

    def units_per_request(self, max_seq_len: int | None) -> int:
        """Units one stream of a request holds at ``max_seq_len`` tokens."""
        del max_seq_len
        return 1

    @property
    def host_units(self) -> int:
        """Units of pinned host memory kept for offloading."""
        return 0


class PoolDemand(NamedTuple):
    """One pool as the sizing arithmetic sees it, on one rank."""

    key: str
    unit: str
    size_key: str
    unit_bytes: int
    # allocated, the reserved ones included; None for a store that grows per request
    units: int | None
    reserved: int
    # units a request holds at most: every stream, at ``max_seq_len``
    per_request: int
    max_seq_len: int | None
    host_units: int = 0
    # the engine sizes it from memory: neither the model nor the deployment did
    unsized: bool = False

    @property
    def requests(self) -> int | None:
        """Requests the pool holds at once at ``per_request`` each."""
        if self.units is None:
            return None
        return max(self.units - self.reserved, 0) // max(self.per_request, 1)

    @property
    def nbytes(self) -> int:
        return (self.units or 0) * self.unit_bytes

    @property
    def request_bytes(self) -> int:
        return self.per_request * self.unit_bytes

    @property
    def can_evict(self) -> bool:
        """Whether a full pool can make room by offloading a request to the host."""
        return self.host_units > 0


def spec_pool(spec: NodeResourceSpec) -> PoolConfig | None:
    config = getattr(spec, "config", None)
    return config if isinstance(config, PoolConfig) else None


def _request_len(spec: NodeResourceSpec, by_key: Mapping[str, NodeResourceSpec]) -> int | None:
    """The longest request ``spec`` holds tokens for: its own ``max_seq_len``, or the first
    of its dependencies' (index keys grow with the cache they index)."""
    own = getattr(spec.config, "max_seq_len", None)
    if own is not None:
        return own
    for key in sorted(spec.depends_on()):
        dep = by_key.get(key)
        length = getattr(getattr(dep, "config", None), "max_seq_len", None)
        if length is not None:
            return length
    return None


def pool_demands(
    specs: Sequence[NodeResourceSpec],
    kv_dtype: torch.dtype,
    num_shards: Callable[[NodeResourceSpec], int] = lambda spec: 1,
    model: "Model | None" = None,
    request_len: int | None = None,
) -> list[PoolDemand]:
    """Every pool among ``specs``, in declaration order.

    A request holds one stream per label the model declares for a pool in
    ``kv_labels`` (CFG's conditional and unconditional ones, say), each at the
    pool's ``max_seq_len``. A model that knows a tighter bound says so in
    ``Model.pool_units_per_request``.

    An unsized pool is sized for the deployment's requests: with ``request_len``
    (the serving config's ``max_seq_len``), a label that grows holds that many
    tokens, and a label that doesn't (an encoder's context) what its pool declares.
    """
    by_key = {spec.resource_key: spec for spec in specs}
    labels = model.kv_labels() if model is not None else {}
    demands = []
    for spec in specs:
        pool = spec_pool(spec)
        if pool is None:
            continue
        length = _request_len(spec, by_key)
        unit_bytes = pool.unit_bytes(kv_dtype, num_shards(spec))
        if pool.pool_units is None and unit_bytes == 0:
            continue  # a growing store that declares no size
        unsized = pool.derivable and not pool.sized
        declared = labels.get(spec.resource_key, {})
        # the length of each stream a request holds: one per declared label (at least one)
        streams = [length] * max(len(declared), 1)
        if unsized and request_len is not None and declared:
            streams = [request_len if label.growth else length for label in declared.values()]
        per_request = model.pool_units_per_request(spec) if model is not None else None
        if per_request is None:
            per_request = sum(pool.units_per_request(n) for n in streams)
        demands.append(PoolDemand(
            key=spec.resource_key,
            unit=pool.pool_unit,
            size_key=pool.size_key,
            unit_bytes=unit_bytes,
            units=pool.pool_units,
            reserved=pool.reserved_units,
            per_request=per_request,
            max_seq_len=max(streams, key=lambda n: n or 0),
            host_units=pool.host_units,
            unsized=unsized,
        ))
    return demands


def concurrency(demands: Sequence[PoolDemand]) -> int | None:
    """Requests every pool holds at once at its per-request bound; None with no pool."""
    held = [d.requests for d in demands if d.units is not None]
    return min(held) if held else None


def requests_within(demands: Sequence[PoolDemand], budget: int) -> int:
    """Requests that ``budget`` bytes hold with every pool sized to the same count.

    The reserved units come off first; a store with no pool still takes its
    per-request bytes.
    """
    fixed = sum(d.reserved * d.unit_bytes for d in demands if d.units is not None)
    per_request = sum(d.request_bytes for d in demands)
    if per_request <= 0:
        return 0
    return max(budget - fixed, 0) // per_request


def requests_for_unsized(demands: Sequence[PoolDemand], budget: int) -> int:
    """Requests the unsized pools could hold, sized alike, in what ``budget`` bytes leave
    after the sized pools. A store with no pool takes its per-request bytes from the same budget."""
    left = budget - sum(d.nbytes for d in demands if d.units is not None and not d.unsized)
    return requests_within([d for d in demands if d.unsized or d.units is None], left)


def default_capture_reserve(total: int) -> int:
    """What pools sized for the deployment's requests must leave of a ``total``-byte device
    for graph capture, the largest eager step and the workspaces built after them, unless
    ``capture_reserve_gib`` says.

    What that takes depends on the model's capture buckets, not on the device: 3 to 15 GiB
    on an H100 for the models measured (Qwen3-TTS's 15 GiB the most), and the startup
    report gives each deployment its own figure."""
    return max(4 * 2**30, total // 4)


def units_for(demand: PoolDemand, requests: int) -> int:
    """Units ``demand``'s pool needs to hold ``requests`` requests."""
    return demand.reserved + requests * demand.per_request


def _size(n: int) -> str:
    for scale, name in ((2**30, "GiB"), (2**20, "MiB"), (2**10, "KiB")):
        if n >= scale:
            return f"{n / scale:.{2 if scale == 2**30 else 1}f} {name}"
    return f"{n} B"


def describe(demand: PoolDemand) -> str:
    """One pool on one line: what it holds, what a request takes, how many fit."""
    unit = demand.unit
    length = "" if demand.max_seq_len is None else f" at max_seq_len {demand.max_seq_len}"
    if demand.units is None:
        return (
            f"{demand.key}: grows per request, {_size(demand.unit_bytes)} per {unit}; "
            f"a request takes {demand.per_request} {unit}s{length} ({_size(demand.request_bytes)})"
        )
    host = f", {demand.host_units} more on the host" if demand.host_units else ""
    return (
        f"{demand.key}: {demand.units} {unit}s x {_size(demand.unit_bytes)} = {_size(demand.nbytes)}"
        f"{host}; a request takes {demand.per_request}{length} -> {demand.requests} at once"
    )


def memory_allows(demands: Sequence[PoolDemand], spare: int) -> str | None:
    """What the pools would hold if ``spare`` bytes went to them too, split so one
    request count fits every pool; None with no pool."""
    pooled = [d for d in demands if d.units is not None]
    if not pooled:
        return None
    n = requests_within(demands, sum(d.nbytes for d in pooled) + max(spare, 0))
    sizes = ", ".join(f"{d.key} {units_for(d, n)} {d.unit}s" for d in pooled)
    return f"{_size(max(spare, 0))} spare would hold {n} at once ({sizes})"
