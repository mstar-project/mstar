"""Certify preparation of inputs whose producing capture has not run yet.

Only ordered structural packing is allowed. The result is a list of ordinary
copies from leased outputs and real inputs; a fake tensor is never staged.
This scope deliberately does not enter a torch dispatch mode: independent
operations on real tensors must keep doing real work.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from enum import Enum
from types import MappingProxyType, SimpleNamespace
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

    @contextmanager
    def isolated_info(self, batch: Any) -> Iterator[None]:
        """Certify preparation on copied data, never opaque request-owned objects.

        Alternative chunks cannot apply speculative host mutations. Only
        unchanged plain metadata is eligible; owner/generator state is refused
        before calling preparation, without invoking custom copy/equality hooks.
        """
        names = ("per_request_info", "per_request_info_wrapped", "per_request_input_metadata")
        original = tuple(getattr(batch, name) for name in names)

        def copy_data(value, active):
            kind = type(value)
            if (value is None or kind in (bool, int, float, str, bytes, torch.dtype, torch.device)
                    or isinstance(value, Enum)):
                return value, (kind, value)
            if id(value) in active:
                self.refuse("connected preparation metadata contains a cycle")
            active = active | {id(value)}
            if kind in (dict, MappingProxyType):
                entries = [(copy_data(k, active), copy_data(v, active)) for k, v in value.items()]
                copied = {k[0]: v[0] for k, v in entries}
                return (MappingProxyType(copied) if kind is MappingProxyType else copied,
                        (kind, tuple((k[1], v[1]) for k, v in entries)))
            if kind in (list, tuple, set, frozenset) or (isinstance(value, tuple) and hasattr(kind, "_fields")):
                entries = [copy_data(item, active) for item in value]
                copied = ([entry[0] for entry in entries] if kind is list else
                          kind(*(entry[0] for entry in entries)) if hasattr(kind, "_fields") else
                          kind(entry[0] for entry in entries))
                return copied, (kind, tuple(entry[1] for entry in entries))
            if is_dataclass(value) and not isinstance(value, type) or kind is SimpleNamespace:
                entries = [(name, copy_data(getattr(value, name), active)) for name in
                           (vars(value) if kind is SimpleNamespace else (f.name for f in fields(value)))]
                copied = object.__new__(kind) if kind is not SimpleNamespace else SimpleNamespace()
                for name, entry in entries:
                    object.__setattr__(copied, name, entry[0])
                return copied, (kind, tuple((name, entry[1]) for name, entry in entries))
            self.refuse(f"connected preparation cannot isolate metadata of type {kind.__name__}")

        copied, before = copy_data(original, set())
        for name, value in zip(names, copied, strict=True):
            setattr(batch, name, value)
        try:
            yield
            _, after = copy_data(copied, set())
            if after != before:
                self.refuse("connected preparation mutates request metadata")
        finally:
            for name, value in zip(names, original, strict=True):
                setattr(batch, name, value)

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

    def rebind(self, key: Any, source: torch.Tensor) -> None:
        """A certified seed is replaced by the same-shaped captured carry."""
        old = self.bindings[key]
        if old.shape != source.shape or old.dtype != source.dtype or old.device != source.device:
            self.refuse("a Loop carry disagrees with its first iteration's seed")
        self.bindings[key] = source

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
