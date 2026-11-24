"""Per-request tensors at sizes each request sets, held from admission to the request's end.

A request declares its sizes when it opens (``SizedValueReqConfig``) and waits
in the engine's admission queue, in arrival order, until ``max_bytes`` has room
for its fields at those sizes (see ``Resource.ask``). The store allocates
nothing: a step's forward makes the tensors, ``put`` keeps them as they are,
and ``take`` hands them back to the next step and forgets them, so a tensor the
step replaces is freed there and not at the step's end.
"""

from __future__ import annotations

import threading

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.sized.config import SizedValueConfig, SizedValueReqConfig, SizedValueSpec
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import ASKS, AdmitOutcome, AdmitRuntimeError


class _Request:
    __slots__ = ("config", "admitted", "nbytes", "tensors", "cleared", "lent")

    def __init__(self, config: SizedValueReqConfig):
        self.config = config
        # Whether admission took its bytes, and the bytes it still holds of them. A request that
        # cleared every field holds none and is still admitted.
        self.admitted = False
        self.nbytes = 0
        # by (label, layer, field)
        self.tensors: dict[tuple[str, int, str], torch.Tensor] = {}
        self.cleared: set[tuple[str, int, str]] = set()
        # the (label, layer) rows a step took and hasn't put back
        self.lent: set[tuple[str, int]] = set()


class SizedValueStore(Resource):
    @classmethod
    def build(cls, spec: SizedValueSpec, info: EngineResourceInfo) -> SizedValueStore:
        if info.joint_comm_group is not None and info.joint_comm_group.world_size > 1:
            raise NotImplementedError("a sized value store does not shard across ranks yet")
        return cls(
            spec.config, info.device, name=spec.resource_key,
            episode_lifetime=spec.episode_lifetime,
        )

    def __init__(
        self, config: SizedValueConfig, device: torch.device, name: str = "sized_value",
        episode_lifetime: bool = False,
    ):
        self.config = config
        # the resource key, which a deployment's ``resources:`` block names
        self.name = name
        self.episode_lifetime = episode_lifetime
        self._episode_owner_by_request: dict[int | str, tuple[str, int]] = {}
        self._device = torch.device(device)
        self._requests: dict[int, _Request] = {}
        self._used = 0
        # Guards the requests and the bytes taken, which intake, removal and the admission
        # calls on the main thread share with a step's take, put and clear on the GPU thread.
        self._lock = threading.Lock()

    def can_capture(self) -> bool:
        return False

    @property
    def deferred_bytes(self) -> int:
        return self.config.budget

    # Request lifecycle

    def ingest_request(self, rid, overrides: ResourceReqConfig | None = None):
        if isinstance(overrides, SizedValueReqConfig):
            with self._lock:
                self._requests[rid] = _Request(overrides)

    def bind_episode(self, rid: int, slot: int, episode_id: str | None, *, cold: bool, reset: bool) -> None:
        if episode_id is None:
            return
        owner = ("episode", slot)
        with self._lock:
            if cold:
                if owner in self._requests:
                    raise ValueError(f"sized value store {self.name!r}: episode slot {slot} already has state")
            else:
                prev = self._requests.pop(owner, None)
                if prev is not None:
                    if reset:
                        self._used -= prev.nbytes
                        prev.tensors.clear()
                        prev.cleared.clear()
                        prev.lent.clear()
                        prev.admitted = False
                        prev.nbytes = 0
                        if rid not in self._requests:
                            self._requests[rid] = prev
                    else:
                        cur = self._requests.get(rid)
                        if not prev.admitted:
                            if cur is None:
                                self._requests[rid] = prev
                        else:
                            if cur is not None and cur.config.sizes != prev.config.sizes:
                                self._requests[owner] = prev
                                raise ValueError(
                                    f"{self.name}: warm episode sizes changed from {prev.config.sizes} "
                                    f"to {cur.config.sizes} without reset"
                                )
                            self._requests[rid] = prev
            self._episode_owner_by_request[rid] = owner

    def end_episode(self, slot: int) -> None:
        owner = ("episode", slot)
        with self._lock:
            if owner in self._episode_owner_by_request.values():
                raise ValueError(f"episode slot {slot} still has an active request")
            req = self._requests.pop(owner, None)
            if req is not None:
                self._used -= req.nbytes
                req.tensors.clear()

    def remove_request(self, rid):
        with self._lock:
            owner = self._episode_owner_by_request.pop(rid, None)
            req = self._requests.pop(rid, None)
            if owner is not None:
                if req is not None:
                    req.lent.clear()
                    self._requests[owner] = req
                return
            if req is not None:
                self._used -= req.nbytes

    def logical_length(self, rid, label: str = "main") -> int:
        with self._lock:
            key = rid if rid in self._requests else self._episode_owner_by_request.get(rid, rid)
            req = self._requests.get(key)
            if req is None:
                return 0
            return 1 if any(lbl == label for (lbl, _, _) in req.tensors) else 0

    def held(self, rid) -> int:
        with self._lock:
            key = rid if rid in self._requests else self._episode_owner_by_request.get(rid, rid)
            req = self._requests.get(key)
            if req is None:
                return 0
            return len({lbl for (lbl, _, _) in req.tensors})

    def _need(self, req: _Request) -> int:
        cleared = sum(self.config.fields[name].bytes_at(req.config.sizes[label]) for label, _, name in req.cleared)
        return req.config.bytes(self.config) - cleared

    def ask(self, rid, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        with self._lock:
            req = self._requests.get(rid)
            if req is None or req.admitted:
                return None
            need = self._need(req)
        if need > self.config.budget:
            return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                f"the request's tensors take {need} bytes; the store holds {self.config.budget} "
                f"(resources: {self.name}: max_bytes)", request_ids=frozenset({rid})))
        return ASKS

    def reserve(self, rid, node_name: str, graph_walk: str, published: None) -> int | None:
        with self._lock:
            req = self._requests.get(rid)
            if req is None or req.admitted:
                return None
            need, room = self._need(req), self.config.budget - self._used
            if need > room:
                return room
            req.admitted, req.nbytes = True, need
            self._used += need
            return None

    def unreserve(self, rid) -> None:
        with self._lock:
            req = self._requests.get(rid)
            if req is not None:
                self._used -= req.nbytes
                req.admitted, req.nbytes = False, 0

    def grew_since(self, mark: int) -> bool:
        with self._lock:
            return self.config.budget - self._used > mark

    # Model-facing, through the array's handle

    def sizes(self, rid, label: str) -> dict[str, int] | None:
        """The sizes ``rid`` declared for ``label``, None where it declared none."""
        with self._lock:
            req = self._requests.get(rid)
            return None if req is None else req.config.sizes.get(label)

    def take(self, rid, label: str, layer: int) -> dict[str, torch.Tensor | None]:
        """Hand over the row's tensors, None for a field that holds none. The store keeps no reference, and
        the row stays lent until a `put` or a `clear` of it."""
        with self._lock:
            req = self._requests[rid]
            if (label, layer) in req.lent:
                raise RuntimeError(
                    f"{self.name}: request {rid}'s tensors on layer {layer} were read by a step that never wrote "
                    "them back, so they are lost")
            req.lent.add((label, layer))
            return {name: req.tensors.pop((label, layer, name), None) for name in self.config.fields}

    def put(self, rid, label: str, layer: int, tensors: dict[str, torch.Tensor]) -> None:
        """Keep ``tensors`` for the row as they are, each within its field's shape at the row's sizes."""
        with self._lock:
            req = self._requests[rid]
            for name, tensor in tensors.items():
                if (label, layer, name) in req.cleared:
                    raise ValueError(f"field {name!r} was cleared for request {rid}, and a cleared field stays unset")
                on = tensor.device
                if on.type != self._device.type or self._device.index not in (None, on.index):
                    raise ValueError(f"field {name!r} is written on {on}, and the store counts its bytes on "
                                     f"{self._device}")
                req.tensors[label, layer, name] = tensor
            req.lent.discard((label, layer))

    def clear(self, rid, label: str, layer: int, names: tuple[str, ...]) -> None:
        """Drop the row's ``names`` for the rest of the request and give their bytes back."""
        with self._lock:
            req = self._requests[rid]
            for name in names:
                if (label, layer, name) in req.cleared:
                    continue
                req.cleared.add((label, layer, name))
                req.tensors.pop((label, layer, name), None)
                if req.admitted:
                    back = self.config.fields[name].bytes_at(req.config.sizes[label])
                    req.nbytes -= back
                    self._used -= back
            req.lent.discard((label, layer))

    def post_warmup_validate(self):
        if self._requests or self._used:
            raise RuntimeError(f"sized value store {self.name} still holds {len(self._requests)} requests after warmup")
