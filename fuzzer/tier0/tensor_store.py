"""TensorStore: the per-request tensor refcounts of the transport layer.

Ops: put, incref, release, set_meta, remove, remove_all. The machine mirrors
each op into a shadow model. A release op never drops more references than
the machine took. Thus an unbalanced count comes from the store.

Invariants
----------
store.mirrors_model             the store holds exactly the model's keys
store.refcount_matches          each refcount equals the model's
store.refcount_never_negative   a refcount never falls below zero
store.metadata_matches          persist and mem_registered match the model
store.can_gc_definition         can_gc is refcount <= 0 and not persist
store.is_registered_matches     is_registered matches the stored flag
store.no_empty_request_entries  an emptied request is not left keyed
store.get_all_uuids_matches     get_all_uuids agrees with the model
store.quiesce_is_empty          removing every tensor empties the store

Not covered:

* a caller that releases a reference it did not take
* the contents, the size and the device of a tensor: every tensor here is
  `torch.zeros(1)`
* concurrent access from the background threads of the transport"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from fuzzer.common.case import Op
from fuzzer.common.machine import StateMachine, require
from fuzzer.tier0 import _stubs  # noqa: F401
from mstar.communication.tensors import TensorStore


@dataclass
class _Shadow:
    """What the model believes about one (request, uuid) entry."""

    refs: int = 0
    persist: bool = False
    registered: bool = False


@dataclass
class _Model:
    """The model of the whole store. The store must always agree with it."""

    entries: dict[tuple[str, str], _Shadow] = field(default_factory=dict)


class TensorStoreMachine(StateMachine):
    name = "tensor_store"

    @classmethod
    def gen_config(cls, rng: random.Random) -> dict:
        return {
            "num_requests": rng.randint(1, 3),
            "num_uuids": rng.randint(1, 4),
        }

    def __init__(self, config: dict) -> None:
        self.rids = [f"r{i}" for i in range(config["num_requests"])]
        self.uuids = [f"u{i}" for i in range(config["num_uuids"])]
        self.store = TensorStore()
        self.model = _Model()

    # -- generation ----------------------------------------------------------

    def gen_op(self, rng: random.Random) -> Op:
        rid = rng.randrange(len(self.rids))
        uuid = rng.randrange(len(self.uuids))
        choice = rng.random()
        if choice < 0.22:
            return Op("put", (rid, uuid))
        if choice < 0.45:
            return Op("incref", (rid, uuid, rng.randint(0, 3)))
        if choice < 0.68:
            return Op("release", (rid, uuid, rng.randint(0, 3)))
        if choice < 0.80:
            return Op("set_meta", (rid, uuid, rng.randint(0, 1), rng.randint(0, 1)))
        if choice < 0.94:
            return Op("remove", (rid, uuid))
        return Op("remove_all", (rid,))

    # -- execution -----------------------------------------------------------

    def _names(self, rid_index: int, uuid_index: int) -> tuple[str, str]:
        """Map two indices from an op onto a request ID and a uuid."""
        return (
            self.rids[rid_index % len(self.rids)],
            self.uuids[uuid_index % len(self.uuids)],
        )

    def execute(self, op: Op) -> None:
        if op.kind == "put":
            rid, uuid = self._names(*op.args)
            self.store.put_tensor(rid, uuid, torch.zeros(1))
            # put_tensor replaces the info object, so the model restarts.
            self.model.entries[(rid, uuid)] = _Shadow()

        elif op.kind == "incref":
            rid, uuid = self._names(op.args[0], op.args[1])
            count = op.args[2]
            self.store.increment_ref(rid, uuid, count)
            shadow = self.model.entries.get((rid, uuid))
            if shadow is not None:
                shadow.refs += count

        elif op.kind == "release":
            rid, uuid = self._names(op.args[0], op.args[1])
            shadow = self.model.entries.get((rid, uuid))
            # Never release more references than this machine took.
            count = 0 if shadow is None else min(op.args[2], shadow.refs)
            if count == 0:
                return
            self.store.dereference(rid, uuid, count)
            shadow.refs -= count

        elif op.kind == "set_meta":
            rid, uuid = self._names(op.args[0], op.args[1])
            persist, registered = bool(op.args[2]), bool(op.args[3])
            self.store.set_metadata(
                rid, uuid, persist=persist, mem_registered=registered
            )
            shadow = self.model.entries.get((rid, uuid))
            if shadow is not None:
                shadow.persist = persist
                shadow.registered = registered

        elif op.kind == "remove":
            rid, uuid = self._names(*op.args)
            self.store.remove_tensor(rid, uuid)
            self.model.entries.pop((rid, uuid), None)

        elif op.kind == "remove_all":
            rid = self.rids[op.args[0] % len(self.rids)]
            for uuid in self.store.get_all_uuids(rid):
                self.store.remove_tensor(rid, uuid)
                self.model.entries.pop((rid, uuid), None)

        else:
            raise AssertionError(f"unknown op {op.kind}")

    # -- invariants ----------------------------------------------------------

    def check(self) -> None:
        real = {
            (rid, uuid): info
            for rid, per_uuid in self.store.per_req_tensors.items()
            for uuid, info in per_uuid.items()
        }

        require(
            "store.mirrors_model",
            set(real) == set(self.model.entries),
            f"store holds {sorted(set(real) - set(self.model.entries))} that the "
            f"model does not, and is missing "
            f"{sorted(set(self.model.entries) - set(real))}",
        )

        for key, info in real.items():
            shadow = self.model.entries[key]
            rid, uuid = key
            require(
                "store.refcount_matches",
                info.ref_cnt == shadow.refs,
                f"{key}: store ref_cnt={info.ref_cnt}, model says {shadow.refs}",
            )
            require(
                "store.refcount_never_negative",
                info.ref_cnt >= 0,
                f"{key}: ref_cnt fell to {info.ref_cnt}; the tensor would be "
                "released while a reader still holds it",
            )
            require(
                "store.metadata_matches",
                (info.persist, info.mem_registered)
                == (shadow.persist, shadow.registered),
                f"{key}: store persist/registered="
                f"{(info.persist, info.mem_registered)}, model says "
                f"{(shadow.persist, shadow.registered)}",
            )
            require(
                "store.can_gc_definition",
                self.store.can_gc(rid, uuid)
                == (info.ref_cnt <= 0 and not info.persist),
                f"{key}: can_gc={self.store.can_gc(rid, uuid)} but "
                f"ref_cnt={info.ref_cnt} persist={info.persist}",
            )
            require(
                "store.is_registered_matches",
                self.store.is_registered(rid, uuid) == info.mem_registered,
                f"{key}: is_registered disagrees with the stored flag",
            )

        require(
            "store.no_empty_request_entries",
            all(per_uuid for per_uuid in self.store.per_req_tensors.values()),
            "an emptied request is still keyed in per_req_tensors; the dict "
            "grows without bound across requests",
        )
        for rid in self.rids:
            require(
                "store.get_all_uuids_matches",
                sorted(self.store.get_all_uuids(rid))
                == sorted(u for (r, u) in self.model.entries if r == rid),
                f"get_all_uuids({rid}) disagrees with the model",
            )

    def final_check(self) -> None:
        """Remove every request. The store must then be empty."""
        for rid in self.rids:
            for uuid in self.store.get_all_uuids(rid):
                self.store.remove_tensor(rid, uuid)
        require(
            "store.quiesce_is_empty",
            self.store.per_req_tensors == {},
            f"after removing every tensor the store still holds "
            f"{self.store.per_req_tensors!r}",
        )
