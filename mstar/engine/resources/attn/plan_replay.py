"""Replaying FlashInfer's ``plan()`` for a wrapper captured at fixed shapes.

Most of a FlashInfer ``plan()`` call is Python around one planner call
(dtype and backend resolution, shape checks, the module lookup), and for a
captured wrapper all of it comes out the same every step but for the step's
host arrays. ``PlanReplay`` learns what a plan does from FlashInfer itself,
rather than restating its internals:

- the planner call: which arguments are the step's arrays, the rest fixed;
- the wrapper buffers it writes, each a copy of one of the arrays;
- the wrapper attributes it sets per step, each a simple function of one.

It learns from synthetic steps, and checks a replay against FlashInfer's own
plan of the same arrays before using it. On real steps it then lets FlashInfer
plan the first few and every few hundredth, recording each, and stops
replaying if a recording disagrees with what the replay would have done.
Anything it cannot learn or check keeps calling ``plan()``: slower, never
different.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Hashable
from typing import Any, NamedTuple

import torch

logger = logging.getLogger(__name__)

# a step's host arrays by name: indptrs, page indices, lengths
Arrays = dict[str, torch.Tensor]

def _host(t: torch.Tensor):
    """A host array as numpy, which is cheaper than torch ops at this size."""
    return t.numpy() if t.device.type == "cpu" else t


# How a plan may derive a per-step attribute from one of the step's arrays.
_DERIVATIONS: dict[str, Callable[[torch.Tensor], Any]] = {
    "is": lambda t: t,
    "last": lambda t: int(_host(t)[-1]),
    "max": lambda t: int(_host(t).max()),
    "max_span": lambda t: int((_host(t)[1:] - _host(t)[:-1]).max()),
}
# buffers above this many elements (FlashInfer's workspaces) are not snapshotted
_MAX_SNAPSHOT = 1 << 24
# synthetic steps per configuration: one to resolve FlashInfer's module, three
# to learn from, two to check a replay with
_N_SYNTHETIC = 6
# real steps FlashInfer plans and the replay is checked against: the first
# ones of a configuration, then one in this many
_CHECK_FIRST = 16
_CHECK_EVERY = 512
# recording wraps a planner module that wrappers share: one recording at a time
_RECORDING = threading.Lock()
_MISSING = object()


def _derive(f: Callable[[torch.Tensor], Any], t: torch.Tensor) -> Any:
    try:
        return f(t)
    except (RuntimeError, IndexError, ValueError):  # e.g. the widest span of one row
        return _MISSING


def _same(a: Any, b: Any) -> bool:
    """Identity for tensors, equality otherwise."""
    if a is b:
        return True
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor) or type(a) is not type(b):
        return False
    try:
        return bool(a == b)
    except Exception:
        return False


def _array_names(value: Any, arrays: Arrays) -> set[str]:
    """The arrays ``value`` is: the tensor itself, or a host copy of it."""
    if not isinstance(value, torch.Tensor):
        return set()
    return {
        name for name, t in arrays.items()
        if value is t or (
            value.device == t.device and value.dtype == t.dtype
            and value.shape == t.shape and torch.equal(value, t)
        )
    }


def _snapshot(w) -> dict[str, torch.Tensor]:
    return {
        k: v.clone() for k, v in vars(w).items()
        if isinstance(v, torch.Tensor) and v.numel() <= _MAX_SNAPSHOT
    }


def _copy_names(before: torch.Tensor, after: torch.Tensor, arrays: Arrays) -> set[str]:
    """The arrays ``after`` holds a copy of at its front, the rest unchanged."""
    names = set()
    for name, t in arrays.items():
        n = t.shape[0]
        if t.dim() != 1 or after.dim() != 1 or n > after.shape[0]:
            continue
        if torch.equal(after[:n].cpu(), t.cpu().to(after.dtype)) and torch.equal(after[n:], before[n:]):
            names.add(name)
    return names


def _one(candidates: list[set]) -> Any | None:
    """The one candidate every record agrees on, else None."""
    common = set.intersection(*candidates) if candidates else set()
    return next(iter(common)) if len(common) == 1 else None


class _Call(NamedTuple):
    """A planner call: the planner, its arguments and its result."""
    fn: Callable
    args: list
    result: Any


def _recorded(w, flashinfer_plan: Callable[[Arrays], None], arrays: Arrays) -> _Call | None:
    """Have FlashInfer plan ``arrays``, and return its planner call, or None
    unless it made exactly one."""
    with _RECORDING:
        module = getattr(w, "_cached_module", None)
        fn = getattr(module, "plan", None)
        if fn is None:
            flashinfer_plan(arrays)
            return None
        me = threading.get_ident()
        calls = []

        def recording(*args):
            result = fn(*args)
            if threading.get_ident() == me:
                calls.append(_Call(fn, list(args), result))
            return result

        module.plan = recording
        try:
            flashinfer_plan(arrays)
        finally:
            module.plan = fn
    return calls[0] if len(calls) == 1 else None


def _args_match(replayed: list, recorded: list, arrays: Arrays) -> bool:
    return len(replayed) == len(recorded) and all(
        _same(x, y) or bool(_array_names(x, arrays) & _array_names(y, arrays))
        for x, y in zip(replayed, recorded, strict=True)
    )


class _Record(NamedTuple):
    """One FlashInfer plan, observed."""
    arrays: Arrays
    fn: Callable
    args: list
    result: Any
    attrs: dict[str, Any]
    before: dict[str, torch.Tensor]
    after: dict[str, torch.Tensor]


class _Replay:
    """One configuration's plan, learned: the planner call with the step's
    arrays in their slots, the buffer copies and the attribute updates."""

    def __init__(self, fn, template: list, params: dict[int, str], copies: dict[str, str],
                 derived: dict[str, tuple[Callable, str]], info_attr: str):
        self.fn = fn
        self.template = template
        self.params = params
        self.copies = copies
        self.derived = derived
        self.info_attr = info_attr
        self.steps = 0

    @classmethod
    def learn(cls, records: list[_Record]) -> tuple[_Replay | None, str]:
        fn, n = records[0].fn, len(records[0].args)
        if any(r.fn is not fn or len(r.args) != n for r in records):
            return None, "the planner call changes"
        template, params = list(records[0].args), {}
        for i in range(n):
            name = _one([_array_names(r.args[i], r.arrays) for r in records])
            if name is not None:
                params[i] = name
            elif not all(_same(r.args[i], template[i]) for r in records):
                return None, f"planner argument {i} varies with no array behind it"
        planner_buffers = {id(a) for a in template if isinstance(a, torch.Tensor)}

        info = [k for k, v in records[0].attrs.items() if v is records[0].result]
        if len(info) != 1 or any(r.attrs.get(info[0]) is not r.result for r in records):
            return None, "no one attribute holds the planner's result"
        info_attr = info[0]

        copies = {}
        for buf in records[0].after:
            if id(records[0].attrs.get(buf)) in planner_buffers:
                continue  # written by the planner itself
            if any(buf not in r.before or buf not in r.after for r in records):
                return None, f"buffer {buf} comes and goes"
            written = [r for r in records if not torch.equal(r.before[buf], r.after[buf])]
            if not written:
                continue
            name = _one([_copy_names(r.before[buf], r.after[buf], r.arrays) for r in written])
            if name is None:
                return None, f"buffer {buf} is written with something other than a step array"
            copies[buf] = name

        derived = {}
        for attr in set().union(*(r.attrs.keys() for r in records)) - {info_attr}:
            values = [r.attrs.get(attr, _MISSING) for r in records]
            if all(_same(v, values[0]) for v in values):
                continue  # the same every plan of this configuration
            how = _one([
                {(d, name) for d, f in _DERIVATIONS.items() for name, t in r.arrays.items()
                 if _same(_derive(f, t), r.attrs.get(attr, _MISSING))}
                for r in records
            ])
            if how is None:
                return None, f"attribute {attr} varies with no simple function of a step array"
            derived[attr] = (_DERIVATIONS[how[0]], how[1])
        return cls(fn, template, params, copies, derived, info_attr), ""

    def fits(self, w, arrays: Arrays) -> bool:
        return all(arrays[name].shape[0] <= getattr(w, buf).shape[0] for buf, name in self.copies.items())

    def args(self, arrays: Arrays) -> list:
        args = self.template.copy()
        for i, name in self.params.items():
            args[i] = arrays[name]
        return args

    def __call__(self, w, arrays: Arrays) -> None:
        for buf, name in self.copies.items():
            src = arrays[name]
            dst = getattr(w, buf)[: src.shape[0]]
            if src.data_ptr() != dst.data_ptr():
                dst.copy_(src, non_blocking=True)
        for attr, (f, name) in self.derived.items():
            setattr(w, attr, f(arrays[name]))
        setattr(w, self.info_attr, self.fn(*self.args(arrays)))

    def disagrees(self, before: dict, after: dict, call: _Call | None, arrays: Arrays) -> str:
        """Why FlashInfer's recorded plan of ``arrays`` (``call``, and the
        wrapper's attributes before and after it) is not what the replay
        does; empty when it is."""
        if call is None or call.fn is not self.fn:
            return "the planner"
        if not _args_match(self.args(arrays), call.args, arrays):
            return "the planner call"
        for attr, value in after.items():
            if attr == self.info_attr:
                continue
            if attr in self.derived:
                f, name = self.derived[attr]
                expected = _derive(f, arrays[name])
            else:
                expected = before.get(attr, _MISSING)
            if not _same(expected, value):
                return f"attribute {attr}"
        return ""


class PlanReplay:
    """Plans a captured FlashInfer wrapper ``w`` by replay where one was
    learned and checked for the configuration, else by its own ``plan()``.

    ``synthetic(n)`` gives ``n`` steps at ``w``'s captured shapes, the first at
    the bucket's full size (FlashInfer fixes a graph wrapper's row count on its
    first plan), each differing from the others in every row count, length
    and span a plan could read. They are planned once per configuration,
    outside graph capture, to learn from.
    """

    def __init__(self, w, synthetic: Callable[[int], list[Arrays]]):
        self._w = w
        self._synthetic = synthetic
        self._replays: dict[Hashable, _Replay | None] = {}
        self._last_key: Hashable = _MISSING

    def plan(self, key: Hashable, arrays: Arrays, flashinfer_plan: Callable[[Arrays], None]) -> None:
        """Plan one step of configuration ``key``. A replay runs only right
        after another plan of the same configuration: the rest of what
        FlashInfer's plan sets is the configuration's, and still in place."""
        w = self._w
        replay = self._replays.get(key, _MISSING)
        if replay is _MISSING and not self._capturing():
            replay = self._replays[key] = self._learn(key, flashinfer_plan)
            self._last_key = key
        if not isinstance(replay, _Replay) or self._last_key != key or not replay.fits(w, arrays):
            flashinfer_plan(arrays)
        elif replay.steps < _CHECK_FIRST or replay.steps % _CHECK_EVERY == 0:
            before = dict(vars(w))
            call = _recorded(w, flashinfer_plan, arrays)
            why = replay.disagrees(before, vars(w), call, arrays)
            if why:
                self._replays[key] = None
                logger.warning(
                    "%s %s: FlashInfer's plan() no longer matches its replay (%s); "
                    "planning with FlashInfer from here on", type(w).__name__, key, why,
                )
        else:
            replay(w, arrays)
        if isinstance(replay, _Replay):
            replay.steps += 1
        self._last_key = key

    def _learn(self, key: Hashable, flashinfer_plan: Callable[[Arrays], None]) -> _Replay | None:
        try:
            steps = self._synthetic(_N_SYNTHETIC)
            flashinfer_plan(steps[0])  # resolves the planner module
            records = [self._record(flashinfer_plan, s) for s in steps[1:4]]
            if any(r is None for r in records):
                replay, why = None, "no single planner call to record"
            else:
                replay, why = _Replay.learn(records)
            if replay is not None:
                why = self._check(replay, flashinfer_plan, steps[4], steps[5])
                replay = None if why else replay
        except Exception as e:  # a FlashInfer this was not written against
            replay, why = None, f"{type(e).__name__}: {e}"
        if replay is None:
            logger.info("%s %s: FlashInfer's plan() is not replayed (%s)", type(self._w).__name__, key, why)
        return replay

    def _on_cuda(self) -> bool:
        device = getattr(self._w, "device", None)
        return device is not None and torch.device(device).type == "cuda"

    def _capturing(self) -> bool:
        return self._on_cuda() and torch.cuda.is_current_stream_capturing()

    def _sync(self) -> None:
        if self._on_cuda():
            torch.cuda.current_stream(self._w.device).synchronize()

    def _record(self, flashinfer_plan, arrays: Arrays) -> _Record | None:
        w = self._w
        self._sync()
        before = _snapshot(w)
        call = _recorded(w, flashinfer_plan, arrays)
        self._sync()
        if call is None:
            return None
        return _Record(arrays, call.fn, call.args, call.result, dict(vars(w)), before, _snapshot(w))

    def _check(self, replay: _Replay, flashinfer_plan, a: Arrays, b: Arrays) -> str:
        """Replay ``a`` over FlashInfer's plan of ``b`` and compare with
        FlashInfer's own plan of ``a``; empty when they agree."""
        w = self._w
        want = self._record(flashinfer_plan, a)
        if want is None or want.fn is not replay.fn:
            return "no planner call to check against"
        if not _args_match(replay.args(a), want.args, a):
            return "the replayed planner call differs"
        flashinfer_plan(b)
        replay(w, a)
        self._sync()
        after = _snapshot(w)
        planner_buffers = {id(x) for x in want.args if isinstance(x, torch.Tensor)}
        for buf, t in want.after.items():
            if id(want.attrs.get(buf)) in planner_buffers:
                continue
            if buf not in after or not torch.equal(t, after[buf]):
                return f"buffer {buf} differs after a replay"
        attrs = vars(w)
        for attr, v in want.attrs.items():
            got = attrs.get(attr, _MISSING)
            if attr == replay.info_attr:
                if list(got) != list(v):
                    return "the planner's result differs"
            elif not _same(got, v):
                return f"attribute {attr} differs after a replay"
        return ""

