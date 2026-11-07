"""Certify preparation of inputs whose producing capture has not run yet.

Only ordered structural packing is allowed. The result is a list of ordinary
copies from leased outputs and real inputs; a fake tensor is never staged.
This scope deliberately does not enter a torch dispatch mode: independent
operations on real tensors must keep doing real work.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from typing import Any, Collection, Iterator

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

_ACTIVE: ContextVar["ConnectedInputs | None"] = ContextVar("connected_inputs", default=None)


def active_guard() -> "ConnectedInputs | None":
    return _ACTIVE.get()


class ConnectedInputs(FakeTensorMode):
    """One unsubmitted attempt's unknown values and their ordered copy spans."""

    def __init__(self, bindings: dict[Any, torch.Tensor]):
        super().__init__(allow_fallback_kernels=False, allow_non_fake_inputs=True)
        self.bindings = bindings
        self.reason: str | None = None
        # Keep the fake alive too: Python may otherwise reuse its id.
        self._spans: dict[int, tuple[torch.Tensor, tuple[Any, ...]]] = {}

    @contextmanager
    def preparing(self) -> Iterator[None]:
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def refuse(self, reason: str) -> None:
        self.reason = reason
        raise ValueError(reason)

    def unknown(self, key: Any) -> FakeTensor:
        source = self.bindings[key]
        if not source.is_contiguous():
            self.refuse("a future row is not contiguous")
        meta = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype, device="meta")
        value = FakeTensor(self, meta, source.device)
        self._spans[id(value)] = value, (key,)
        return value

    def _parts(self, tensor: torch.Tensor) -> tuple[Any, ...]:
        if isinstance(tensor, FakeTensor):
            entry = self._spans.get(id(tensor))
            if entry is None:
                self.refuse("an unknown value has no certified source")
            return entry[1]
        if not tensor.is_contiguous():
            self.refuse("a packed real input is not contiguous")
        key = ("real", id(tensor))
        self.bindings[key] = tensor
        return (key,)

    def dispatch(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        aliases = {
            "aten.alias.default", "aten.detach.default", "aten.view.default",
            "aten._unsafe_view.default", "aten.unsqueeze.default",
            "aten.squeeze.default", "aten.squeeze.dim", "aten.squeeze.dims",
            "aten.as_strided.default", "prims.view_of.default",
        }
        if name in aliases:
            source = args[0]
            if not source.is_contiguous():
                self.refuse(f"{name} would change future storage order")
            spans = self._parts(source)
        elif name in {"aten.stack.default", "aten.cat.default"}:
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
            if dim != 0:
                self.refuse(f"{name} packs future rows on an unsupported axis")
            spans = tuple(key for tensor in args[0] for key in self._parts(tensor))
        else:
            self.refuse(f"preparation needs future values for {name}")
        value = super().dispatch(func, types, args, kwargs)
        if not isinstance(value, FakeTensor) or not value.is_contiguous():
            self.refuse(f"{name} did not preserve contiguous future rows")
        if name in aliases and value.storage_offset() != source.storage_offset():
            self.refuse(f"{name} shifted a future row")
        if sum(self.bindings[key].numel() for key in spans) != value.numel():
            self.refuse(f"{name} changed the future row span")
        self._spans[id(value)] = value, spans
        return value

    def contains(self, value: Any) -> bool:
        if isinstance(value, FakeTensor):
            return True
        if isinstance(value, dict):
            return any(self.contains(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return any(self.contains(v) for v in value)
        if is_dataclass(value) and not isinstance(value, type):
            return any(self.contains(getattr(value, f.name)) for f in fields(value))
        return False

    def require_real(self, value: Any, where: str) -> None:
        if self.contains(value):
            self.refuse(f"future values escaped into {where}")

    def copies(
        self, value: torch.Tensor, target: torch.Tensor, expected: tuple[Any, ...],
        keys: Collection[Any] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Lower a certified mixed real/future field to existing RunPart copies."""
        spans = self._parts(value)
        future = tuple(key for key in spans if key[0] != "real")
        if future != expected:
            self.refuse("future rows were omitted, duplicated, or reordered")
        if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
            self.refuse("future packing disagrees with its captured destination")
        if not target.is_contiguous():
            self.refuse("a future destination is not contiguous")
        flat, offset, copies = target.view(-1), 0, []
        for key in spans:
            source = self.bindings[key]
            if source.dtype != target.dtype or source.device != target.device:
                self.refuse("future packing casts a source")
            size = source.numel()
            if keys is None or key in keys:
                copies.append((source.view(-1), flat.narrow(0, offset, size)))
            offset += size
        return copies
