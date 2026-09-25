"""Tensor uuids as integers.

A tensor uuid used to be ``str(uuid4())``: globally unique by construction,
but 36 bytes on every wire frame, a fresh allocation per output tensor, and a
string hash on every store/lookup. The replacement is one ``u64``::

    (entity_index << 48) | counter

The entity prefix is what keeps it globally unique. A bare per-process counter
would not be: the *receiving* worker keys its own maps
(``TensorStore.per_req_tensors``, ``read_finished``, ``_shm_files``,
``_arena_locs``, ``uuid_to_shard_dim``) by the **sender's** uuid, so two
senders both minting 5 would collide and silently hand back the wrong tensor.

The production entities -- ``worker_{rank}`` and the api-server data worker --
get **stable** indices, because those uuids cross process boundaries and two
peers have to agree. Anything else (tests, fixtures) is assigned from a
process-local pool above the production range: collision-free within a process,
not stable across them, which is fine because a synthetic entity never has a
peer in another process. Deliberately not a hash of the name -- a hash can
collide, and a uuid collision corrupts data silently rather than failing.
"""
import logging
import threading

logger = logging.getLogger(__name__)

ENTITY_BITS = 16
COUNTER_BITS = 48
COUNTER_MASK = (1 << COUNTER_BITS) - 1
MAX_ENTITY_INDEX = (1 << ENTITY_BITS) - 1

#: Entities that mint tensor uuids, by stable index. Part of the wire format
#: only in the sense that two peers must agree on it; adding an entity is safe,
#: renumbering an existing one is not.
_DATA_WORKER = "api_server_preprocess_worker"
_DATA_WORKER_INDEX = 0
_WORKER_PREFIX = "worker_"
_WORKER_BASE = 1

#: Indices at or above this are handed out per process, in first-seen order,
#: for entity ids that are not part of the production set.
_DYNAMIC_BASE = 1 << (ENTITY_BITS - 1)
_dynamic: dict[str, int] = {}
_dynamic_lock = threading.Lock()


def entity_index(entity_id: str) -> int:
    """Stable small integer for an entity that mints tensor uuids."""
    if entity_id == _DATA_WORKER:
        return _DATA_WORKER_INDEX
    if entity_id.startswith(_WORKER_PREFIX):
        rank = entity_id.removeprefix(_WORKER_PREFIX)
        if rank.isdigit():
            index = _WORKER_BASE + int(rank)
            if index >= _DYNAMIC_BASE:
                raise ValueError(
                    f"worker rank {rank} exceeds the stable entity range "
                    f"(0..{_DYNAMIC_BASE - 1}) in a tensor uuid"
                )
            return index
    return _dynamic_index(entity_id)


def _dynamic_index(entity_id: str) -> int:
    """An index for an entity outside the production set.

    Stable for the life of the process and unique within it. NOT stable across
    processes, so an entity whose tensors are read by a peer in another process
    must be in the production set instead.
    """
    index = _dynamic.get(entity_id)
    if index is not None:
        return index
    with _dynamic_lock:
        index = _dynamic.get(entity_id)
        if index is None:
            index = _DYNAMIC_BASE + len(_dynamic)
            if index > MAX_ENTITY_INDEX:
                raise ValueError(
                    f"no entity indices left for {entity_id!r}; "
                    f"{len(_dynamic)} non-production entities already assigned"
                )
            _dynamic[entity_id] = index
            logger.info(
                "tensor uuids for %r use process-local entity index %d; only "
                "the production entity ids are stable across processes",
                entity_id, index,
            )
        return index


class TensorUuidMinter:
    """Hands out this entity's tensor uuids. Not thread-safe by design: the
    call sites are all on the worker's main thread, and a lock here would sit
    on the output path of every forward pass."""

    __slots__ = ("_prefix", "_counter")

    def __init__(self, entity_id: str):
        self._prefix = entity_index(entity_id) << COUNTER_BITS
        self._counter = 0

    def mint(self) -> int:
        self._counter += 1
        if self._counter > COUNTER_MASK:
            raise RuntimeError(
                f"exhausted the {COUNTER_BITS}-bit tensor uuid counter"
            )
        return self._prefix | self._counter


def owner_index(uuid: int) -> int:
    """Which entity minted ``uuid`` — for debugging and assertions."""
    return uuid >> COUNTER_BITS


def counter_of(uuid: int) -> int:
    return uuid & COUNTER_MASK
