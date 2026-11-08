"""Capture pure tensor branches into the backend's existing operation tree."""
from __future__ import annotations

from types import BuiltinFunctionType, FunctionType
from typing import Any

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from mstar.engine.accelerator_graph_backend import GraphConditional


def _check_targets(graph: torch.fx.GraphModule) -> None:
    """Refuse effects before fake tracing can invoke an opaque implementation."""
    if not isinstance(graph, torch.fx.GraphModule):
        raise ValueError("conditional branches must be traced tensor graphs")
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "get_attr":
            value = getattr(graph, node.target)
            if isinstance(value, torch.fx.GraphModule):
                _check_targets(value)
                continue
            raise ValueError("conditional branch constants must be explicit tensor operands")
        if node.op == "call_method":
            target = getattr(torch.ops.aten, node.target, None)
            if isinstance(target, torch._ops.OpOverloadPacket):
                if all(getattr(target, overload)._schema.returns for overload in target.overloads()):
                    continue
            elif node.target in ("float", "double", "half", "bfloat16", "long", "int", "bool",
                                 "size", "dim", "numel", "stride", "element_size", "is_contiguous"):
                continue
        if node.op == "call_function":
            target = node.target
            if target is torch.ops.higher_order.cond:
                continue
            if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
                overloads = (target,) if isinstance(target, torch._ops.OpOverload) else tuple(
                    getattr(target, name) for name in target.overloads()
                )
                if target.namespace == "aten" and all(op._schema.returns for op in overloads):
                    if torch.Tag.nondeterministic_seeded in getattr(target, "tags", ()):
                        raise ValueError("conditional branches cannot draw random numbers")
                    continue
            elif isinstance(target, FunctionType) and target.__module__ == "torch.nn.functional":
                continue
            elif isinstance(target, BuiltinFunctionType) and (
                target.__module__ in ("torch", "torch._C._nn", "_operator")
            ):
                continue
        raise ValueError(f"conditional branch operation is unsupported: {node.op} {node.target}")


def _check_branches(branches, operands) -> None:
    from torch._higher_order_ops.utils import _maybe_fake_tracing, check_input_alias_and_mutation

    for branch in branches:
        _check_targets(branch)
        materialized = _maybe_fake_tracing(branch, operands, False)
        _check_targets(materialized)
        inputs = [node.meta.get("val") for node in materialized.graph.find_nodes(op="placeholder")]
        _, input_output, output_output, mutated = check_input_alias_and_mutation(materialized, inputs)
        if input_output or output_output or mutated:
            raise ValueError("conditional branches must not mutate inputs or alias their outputs")


class ConditionalWarmup(TorchDispatchMode):
    """Discover and prime branches without executing unselected device work."""

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(self, backend, *, world_size: int = 1):
        super().__init__()
        self.supports_higher_order_operators = True
        self.backend = backend
        self.world_size = world_size
        self.found = False
        self._priming = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is not torch.ops.higher_order.cond:
            return func(*args, **(kwargs or {}))
        pred, yes, no, operands = args
        self.found = True
        if self.world_size != 1:
            raise ValueError("conditional capture currently requires one joint rank")
        if not self._priming:
            _check_branches((yes, no), operands)
            self._prime(yes, no, operands)
            with self:
                return (yes if bool(pred) else no)(*operands)
        # Both branches are only recorded in a discarded graph. They have the
        # same output metadata, so either result can feed the recorded suffix.
        with self:
            out = yes(*operands)
            no(*operands)
        return out

    def _prime(self, yes, no, operands) -> None:
        backend = self.backend
        if backend.device_type != "cuda" or backend.while_refusal() is not None:
            raise ValueError("this backend cannot capture tensor conditionals")
        previous = backend.current_stream()
        pool = backend.graph_pool_handle()
        graph = backend.create_graph(keep_graph=True)
        self._priming = True
        try:
            with torch.cuda.graph(graph, pool=pool.handle, stream=backend.new_stream(),
                                  capture_error_mode="relaxed"), self:
                yes(*operands)
                no(*operands)
        except BaseException:
            pool.on_failure(graph)
            backend.recover_failed_capture(pool, previous)
            raise
        finally:
            self._priming = False
            graph.reset()


class ConditionalCapture(TorchDispatchMode):
    """Straight capture segments and native IFs sharing one capture stream."""

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(self, backend, pool, stream):
        super().__init__()
        self.supports_higher_order_operators = True
        self.backend, self.pool = backend, pool
        self.stream = stream if stream is not None else backend.new_stream()
        self.retained: list[torch.Tensor] = []
        self.owned: list[torch.Tensor] = []
        self._ops: list[Any] = []
        self._context = None
        self._graph = None

    def _begin(self) -> None:
        self._graph = self.backend.create_graph(keep_graph=True)
        self._context = self.backend.capture(self._graph, pool=self.pool, stream=self.stream)
        self._context.__enter__()

    def _end(self, exc=(None, None, None)) -> None:
        context, self._context = self._context, None
        if context is not None:
            context.__exit__(*exc)
            self._ops.append(self._graph)
            self._graph = None

    def _record(self, run):
        enclosing, self._ops = self._ops, []
        try:
            self._begin()
            with self:
                output = run()
            self._end()
            return tuple(self._ops), output
        except BaseException as error:
            self._end((type(error), error, error.__traceback__))
            raise
        finally:
            self._ops = enclosing

    def capture(self, run, *, keep_graph=False):
        ops, output = self._record(run)
        return self.backend.compose(ops, keep_graph=keep_graph, retained=self.retained, owned=self.owned), output

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is not torch.ops.higher_order.cond:
            return func(*args, **(kwargs or {}))
        pred, yes, no, operands = args
        if pred.device != self.backend.device or pred.numel() != 1 or pred.dtype != torch.bool:
            raise ValueError("conditional capture needs a same-device boolean scalar")
        condition, complement = pred.to(torch.int64), torch.logical_not(pred).to(torch.int64)
        self.retained.extend((condition, complement))
        self.retained.extend(x for x in pytree.tree_leaves(operands) if isinstance(x, torch.Tensor))
        self._end()
        true_ops, true_out = self._record(lambda: yes(*operands))
        false_ops, false_out = self._record(lambda: no(*operands))
        left, spec = pytree.tree_flatten(true_out)
        right, other = pytree.tree_flatten(false_out)
        if spec != other or not left or any(
            not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or (a.shape, a.dtype, a.device, a.stride()) != (b.shape, b.dtype, b.device, b.stride())
            for a, b in zip(left, right, strict=True)
        ):
            raise ValueError("conditional outputs must have identical tensor metadata")
        result = [torch.empty_like(value) for value in left]
        self.retained.extend((*left, *right))
        self.owned.extend(result)
        self._ops.extend((
            GraphConditional("if", condition, (*true_ops, *zip(result, left, strict=True))),
            GraphConditional("if", complement, (*false_ops, *zip(result, right, strict=True))),
        ))
        self._begin()
        return pytree.tree_unflatten(result, spec)
