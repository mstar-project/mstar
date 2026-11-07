"""Certify preparation of inputs whose producing capture has not run yet.

Only reversible structural packing is allowed. It lowers to ordinary copies
from leased outputs and real inputs; a fake tensor is never staged.
This scope deliberately does not enter a torch dispatch mode: independent
operations on real tensors must keep doing real work.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType, SimpleNamespace
from typing import Any, Collection, Iterator

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily

from mstar.streaming.stream_buffer import StreamChunkInfo

_ACTIVE: ContextVar["ConnectedInputs | None"] = ContextVar("connected_inputs", default=None)


def active_guard() -> "ConnectedInputs | None":
    return _ACTIVE.get()


class ConnectedInputs(FakeTensorMode):
    """One unsubmitted attempt's unknown values and their structural copies."""

    def __init__(self, bindings: dict[Any, torch.Tensor]):
        super().__init__(allow_fallback_kernels=False, allow_non_fake_inputs=True)
        self.bindings = bindings
        self.reason: str | None = None
        # Keep the fake alive too: Python may otherwise reuse its id.
        self._recipes: dict[int, tuple[torch.Tensor, tuple]] = {}

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
            # Preparation sees ordinary chunk metadata, never the owner's
            # mutable policy snapshots. The real receipt remains on the batch.
            if isinstance(value, StreamChunkInfo):
                value = value._replace(reservation=None)
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
        self._recipes[id(value)] = value, ("leaf", key, tuple(source.shape), source.dtype, source.device)
        return value

    def _recipe(self, tensor: torch.Tensor) -> tuple:
        if isinstance(tensor, FakeTensor):
            entry = self._recipes.get(id(tensor))
            if entry is None:
                self.refuse("an unknown value has no certified source")
            return entry[1]
        key = ("real", id(tensor))
        self.bindings[key] = tensor
        return "leaf", key, tuple(tensor.shape), tensor.dtype, tensor.device

    def _parts(self, tensor: torch.Tensor) -> tuple[Any, ...]:
        def leaves(recipe):
            if recipe[0] == "leaf":
                return (recipe[1],)
            return tuple(key for child in recipe[2] for key in leaves(child))
        return leaves(self._recipe(tensor))

    def dispatch(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        aliases = {
            "aten.alias.default", "aten.detach.default", "aten.view.default",
            "aten._unsafe_view.default", "aten.unsqueeze.default",
            "aten.squeeze.default", "aten.squeeze.dim", "aten.squeeze.dims",
            "aten.as_strided.default", "prims.view_of.default", "prims.split_dim.default",
            "prims.collapse_view.default", "aten.clone.default",
        }
        source = args[0]
        if name in aliases:
            recipe = ("view", tuple(source.shape), (self._recipe(source),))
        elif name in {"aten.t.default", "aten.transpose.int", "aten.permute.default"}:
            axes = list(range(source.ndim))
            if name == "aten.t.default":
                if source.ndim > 2:
                    self.refuse("transpose needs at most two dimensions")
                axes.reverse()
            elif name == "aten.transpose.int":
                i, j = args[1] % source.ndim, args[2] % source.ndim
                axes[i], axes[j] = axes[j], axes[i]
            else:
                axes = [axis % source.ndim for axis in args[1]]
            recipe = ("permute", tuple(axes.index(i) for i in range(source.ndim)), (self._recipe(source),))
        elif name in {"aten.stack.default", "aten.cat.default"}:
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
            stack = name == "aten.stack.default"
            dim %= source[0].ndim + int(stack)
            recipe = ("stack" if stack else "cat", (dim, tuple(t.shape[dim] for t in source) if not stack else ()),
                      tuple(self._recipe(t) for t in source))
        elif name == "aten.slice.Tensor":
            dim = args[1] if len(args) > 1 else 0
            start = args[2] if len(args) > 2 else 0
            end = args[3] if len(args) > 3 else None
            step = args[4] if len(args) > 4 else 1
            if slice(start, end, step).indices(source.shape[dim]) != (0, source.shape[dim], 1):
                self.refuse("preparation slices a future row")
            recipe = ("view", tuple(source.shape), (self._recipe(source),))
        elif name == "aten.constant_pad_nd.default":
            padding = tuple(args[1])
            if any(pad < 0 for pad in padding):
                self.refuse("preparation crops a future row")
            fill = args[2] if len(args) > 2 else kwargs.get("value", 0)
            recipe = ("pad", padding, (self._recipe(source),))
        else:
            self.refuse(f"preparation needs future values for {name}")
        value = super().dispatch(func, types, args, kwargs)
        if not isinstance(value, FakeTensor):
            self.refuse(f"{name} did not preserve a future tensor")
        if name == "aten.as_strided.default" and (
            not source.is_contiguous() or not value.is_contiguous()
            or source.storage_offset() != value.storage_offset() or source.numel() != value.numel()
        ):
            self.refuse("preparation changes future storage addressing")
        if recipe[0] == "pad":
            # An independent leaf, staged once before the interior writes.
            # Certification itself never writes a leased destination.
            with unset_fake_temporarily():
                constant = torch.full(tuple(value.shape), fill, dtype=value.dtype, device=value.device)
            recipe = ("pad", recipe[1], (self._recipe(constant), *recipe[2]))
        self._recipes[id(value)] = value, recipe
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

    def same(self, left: Any, right: Any) -> bool:
        """Whether two prepared choices have the same host meaning and real values.

        Future tensors are equal only as structural recipes over the same
        bindings. Independent tensors must actually agree, not merely fit the
        same capture. Opaque equality methods are never called speculatively.
        """
        if type(left) is not type(right):
            return False
        if isinstance(left, FakeTensor):
            def equal(a, b):
                if a[0] != b[0]:
                    return False
                if a[0] == "leaf":
                    return a[2:] == b[2:] and (
                        self.same(self.bindings[a[1]], self.bindings[b[1]])
                        if a[1][0] == b[1][0] == "real" else a[1] == b[1]
                    )
                return (a[1] == b[1] and len(a[2]) == len(b[2])
                        and all(equal(x, y) for x, y in zip(a[2], b[2], strict=True)))
            return (left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
                    and equal(self._recipe(left), self._recipe(right)))
        if isinstance(left, torch.Tensor):
            return (left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
                    and torch.equal(left, right))
        if (left is None or type(left) in (bool, int, float, str, bytes, torch.dtype, torch.device)
                or isinstance(left, Enum)):
            return left == right
        if type(left) in (dict, MappingProxyType):
            return left.keys() == right.keys() and all(self.same(left[k], right[k]) for k in left)
        if isinstance(left, (tuple, list)):
            return len(left) == len(right) and all(self.same(a, b) for a, b in zip(left, right, strict=True))
        if is_dataclass(left) and not isinstance(left, type):
            return all(self.same(getattr(left, f.name), getattr(right, f.name)) for f in fields(left))
        if type(left) in (set, frozenset) and all(type(v) in (str, int) for v in left | right):
            return left == right
        self.refuse(f"connected alternatives cannot compare {type(left).__name__}")

    def require_real(self, value: Any, where: str) -> None:
        if self.contains(value):
            self.refuse(f"future values escaped into {where}")

    @staticmethod
    def materialize(value: Any, bindings: dict[Any, torch.Tensor]) -> Any:
        """An admitted ordinary input, now backed by its retained real values.

        Each fake keeps the certificate that prepared it. Real inputs stay
        unchanged; landed future bindings come from their retained owner.
        """
        if isinstance(value, FakeTensor):
            mode = value.fake_mode
            if not isinstance(mode, ConnectedInputs):
                raise ValueError("a future input has no connected preparation certificate")
            target = torch.empty(tuple(value.shape), dtype=value.dtype, device=value.device)
            for _, source, destination in mode._lower(mode._recipe(value), target, bindings):
                destination.copy_(source)
            return target
        if isinstance(value, dict):
            return {key: ConnectedInputs.materialize(item, bindings) for key, item in value.items()}
        if isinstance(value, list):
            return [ConnectedInputs.materialize(item, bindings) for item in value]
        if isinstance(value, tuple):
            items = [ConnectedInputs.materialize(item, bindings) for item in value]
            return type(value)(*items) if hasattr(type(value), "_fields") else tuple(items)
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{field.name: ConnectedInputs.materialize(getattr(value, field.name), bindings)
                                     for field in fields(value) if field.init})
        return value

    def rebind(self, key: Any, source: torch.Tensor) -> None:
        """A certified seed is replaced by the same-shaped captured carry."""
        old = self.bindings[key]
        if old.shape != source.shape or old.dtype != source.dtype or old.device != source.device:
            self.refuse("a Loop carry disagrees with its first iteration's seed")
        self.bindings[key] = source

    def _lower(self, recipe: tuple, target: torch.Tensor, bindings: dict | None = None):
        kind = recipe[0]
        if kind == "leaf":
            _, key, shape, dtype, device = recipe
            source = self.bindings[key] if key[0] == "real" or bindings is None else bindings.get(key)
            if (source is None or tuple(source.shape) != shape or source.dtype != dtype
                    or source.device != device or tuple(target.shape) != shape):
                self.refuse("a landed input disagrees with its certified future geometry")
            return [(key, source, target)]
        args, children = recipe[1:]
        if kind == "view":
            try:
                views = (target.view(args),)
            except RuntimeError:
                self.refuse("future packing cannot invert a reshape without copying")
        elif kind == "permute":
            views = (target.permute(args),)
        elif kind == "stack":
            views = target.unbind(args[0])
        elif kind == "cat":
            views = target.split(args[1], dim=args[0])
        else:
            slices = [slice(None)] * target.ndim
            for i in range(len(args) // 2):
                start, end = args[2 * i:2 * i + 2]
                slices[-1 - i] = slice(start, target.shape[-1 - i] - end)
            views = (target, target[tuple(slices)])
        return [copy for child, view in zip(children, views, strict=True)
                for copy in self._lower(child, view, bindings)]

    def copies(
        self, value: torch.Tensor, target: torch.Tensor, expected: tuple[Any, ...],
        keys: Collection[Any] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pure lowering to existing copies, including strided owner views."""
        future = tuple(key for key in self._parts(value) if key[0] != "real")
        if future != expected:
            self.refuse("future rows were omitted, duplicated, or reordered")
        if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
            self.refuse("future packing disagrees with its captured destination")
        if not target.is_contiguous():
            self.refuse("a future destination is not contiguous")
        return [(source, destination) for key, source, destination in self._lower(self._recipe(value), target)
                if keys is None or key in keys]
