"""Kernels that write state themselves, in one launch with the math before it, in place of a layer's `add`.

An op is a plain call behind a graph break, as `add` is. It names a fused kernel, the formats that kernel can address,
and a reference over the contract's calls that any other format runs, so an unlisted format loses speed and still boots.
The reference is what the fused kernel is held to.
"""

import functools
import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mstar.engine.state.calls import ArrayHandle
from mstar.engine.state.contract import PagedNHD, PagedNHDTarget

# Off by default: each eager call of a fused op copies the whole array to the host to check what it wrote.
_DEBUG_ASSERTS = os.environ.get("MSTAR_KV_DEBUG_ASSERTS", "0") == "1"


@dataclass(frozen=True)
class StateOp:
    name: str
    fused: Callable[..., Any]
    reference: Callable[..., Any]
    adds: tuple[str, ...]
    writes: tuple[str, ...]
    reads: tuple[str, ...]
    mutates: tuple[str, ...]
    formats: tuple[PagedNHD, ...]
    # where a call's arguments sit: the handles, the handle the kernel's target addresses, and the layer
    handle_at: Mapping[str, int]
    target_at: int
    layer_at: int


# every op imported so far, by name, so that a conformance test can run each one on each format it lists
STATE_OPS: dict[str, StateOp] = {}


def state_op(
    *, adds: Sequence[str] = (), writes: Sequence[str] = (), reads: Sequence[str] = (), mutates: Sequence[str] = (),
    formats: Sequence[PagedNHD], reference: Callable[..., Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Make a kernel a state op: ``adds``, ``writes`` and ``reads`` name the handle parameters it touches, and
    ``mutates`` names its own arguments it changes in place. The kernel takes those, a ``layer``, and a keyword-only
    ``target``, and ``reference`` takes the same without ``target``. Raises ``ValueError`` at import on any other shape,
    and on a name another op holds.

    A call passes its arguments by position. It counts as the layer's add on the array in ``adds``.
    """

    def decorate(fused: Callable[..., Any]) -> Callable[..., Any]:
        op = _state_op(fused, tuple(adds), tuple(writes), tuple(reads), tuple(mutates), tuple(formats), reference)

        @torch.compiler.disable
        @functools.wraps(fused)
        def call(*args: Any) -> Any:
            return _call(op, args)

        call.op = op
        STATE_OPS[op.name] = op
        return call

    return decorate


def _state_op(fused, adds, writes, reads, mutates, formats, reference) -> StateOp:
    name = fused.__name__

    def refusal(cause: str) -> ValueError:
        return ValueError(f"state op {name!r}: {cause}")

    if name in STATE_OPS:
        raise refusal(f"the name is taken by the op in {STATE_OPS[name].fused.__module__}")
    parameters = dict(inspect.signature(fused).parameters)
    target = parameters.pop("target", None)
    if target is None or target.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise refusal("its kernel needs a keyword-only parameter target")
    missing = sorted({*adds, *writes, *reads, *mutates, "layer"} - parameters.keys())
    if missing:
        raise refusal(f"parameters {missing} aren't in its signature, {list(parameters)}")
    if len(adds) + len(writes) != 1:
        raise refusal(f"no layout yet: a target addresses one array, and it adds {adds} and writes {writes}")
    if not all(isinstance(fmt, PagedNHD) for fmt in formats):
        raise refusal(f"formats must be PagedNHD, the one storage format a kernel can address; got {formats}")

    def shape(of):
        return [(p.name, p.kind, p.default) for p in of]

    if shape(inspect.signature(reference).parameters.values()) != shape(parameters.values()):
        raise refusal(f"its reference {reference.__name__!r} must take its signature without target")
    at = {param: i for i, param in enumerate(parameters)}
    return StateOp(
        name=name, fused=fused, reference=reference, adds=adds, writes=writes, reads=reads, mutates=mutates,
        formats=formats, handle_at={handle: at[handle] for handle in (*adds, *writes, *reads)},
        target_at=at[(adds or writes)[0]], layer_at=at["layer"],
    )


def bind_state_ops(handles: Mapping[str, Any]) -> None:
    """Cache on each array handle which ops may run their fused kernel on it. A paged handle still runs the reference
    on a step whose reads attend the step's own k and v (``PagedArrayHandle.fused_ops``)."""
    for handle in handles.values():
        if isinstance(handle, ArrayHandle):
            handle.fused_ops = frozenset(name for name, op in STATE_OPS.items() if handle.takes_fused(op))


def _check_call(op: StateOp, args: Sequence[Any]) -> None:
    for at in op.handle_at.values():
        handle: ArrayHandle = args[at]
        handle._check_open(f"state op {op.name!r}")


def _call(op: StateOp, args: Sequence[Any]) -> Any:
    _check_call(op, args)
    handle: ArrayHandle = args[op.target_at]
    if op.name not in handle.fused_ops:
        return op.reference(*args)
    layer = handle._layer_id(args[op.layer_at])
    target = handle.fused_target(layer)
    # A capture records its launches, so the check's host sync can't run in one.
    if not _DEBUG_ASSERTS or handle._capturing:
        return op.fused(*args, target=target)
    array = handle._kv.kv_cache.tensor
    before = array.to("cpu", copy=True)
    result = op.fused(*args, target=target)
    _check_only_slots_changed(op, handle, array.to("cpu"), before, handle._pool_layer[layer], target)
    return result


def _check_only_slots_changed(
    op: StateOp, handle: ArrayHandle, array: torch.Tensor, before: torch.Tensor, layer: int, target: PagedNHDTarget,
) -> None:
    changed = array.ne(before) & ~(array.isnan() & before.isnan())
    changed[layer][target.page_of_token[:target.count].cpu(), :, target.slot_of_token[:target.count].cpu()] = False
    if changed.any():
        raise handle._refusal(f"state op {op.name!r} changed {int(changed.sum())} elements outside this step's slots")


def conformance_problems(call: Callable[..., Any], args: Sequence[Any], tolerance: float) -> list[str]:
    """What a fused op does differently from its reference, on a handle in an open step over a real pool: the mutated
    arguments and the step's slots within ``tolerance``, absolute and relative as rounding is, every other byte of the
    array unchanged, and every tensor argument it doesn't list as mutated unchanged."""
    op: StateOp = call.op
    handle: ArrayHandle = args[op.target_at]
    if not handle.takes_fused(op):
        return [f"the handle takes no fused kernel of {op.name!r}, so a call runs the reference either way"]
    layer = handle._pool_layer[handle._layer_id(args[op.layer_at])]
    names = list(inspect.signature(op.reference).parameters)
    mutated = [names.index(name) for name in op.mutates]
    array = handle._kv.kv_cache.tensor
    start = array.clone()
    others = {
        names[i]: (arg, arg.clone()) for i, arg in enumerate(args) if i not in mutated and isinstance(arg, torch.Tensor)
    }

    def run(fused: bool) -> tuple[list[torch.Tensor], torch.Tensor]:
        array.copy_(start)
        handle._kept.clear()
        handle._read.clear()
        handle.fused_ops = frozenset({op.name}) if fused else frozenset()
        copies = [arg.clone() if i in mutated else arg for i, arg in enumerate(args)]
        call(*copies)
        return [copies[i] for i in mutated], array.clone()

    reference_args, reference_array = run(fused=False)
    fused_args, fused_array = run(fused=True)
    problems: list[str] = []
    for name, expected, actual in zip(op.mutates, reference_args, fused_args, strict=True):
        if not torch.allclose(actual.float(), expected.float(), atol=tolerance, rtol=tolerance):
            problems.append(f"argument {name}: largest difference {(actual.float() - expected.float()).abs().max()}")
    target = PagedNHDTarget(*handle._kv.write_target(layer_idx=layer, label=handle.label))
    pages, slots = target.page_of_token[:target.count], target.slot_of_token[:target.count]
    written, expected = fused_array[layer][pages, :, slots].float(), reference_array[layer][pages, :, slots].float()
    if not torch.allclose(written, expected, atol=tolerance, rtol=tolerance):
        problems.append(f"the step's slots: largest difference {(written - expected).abs().max()}")
    outside = fused_array.clone()
    outside[layer][pages, :, slots] = start[layer][pages, :, slots]
    if not torch.equal(outside, start):
        problems.append("the array outside the step's slots changed")
    problems += [f"argument {name} changed, and the op doesn't list it" for name, (arg, was) in others.items()
                 if not torch.equal(arg, was)]
    return problems
