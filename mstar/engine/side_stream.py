"""Work of a forward that is independent of its main path, on a second stream beside it.

A model hands the engine a branch that reads only tensors already computed and is not needed until
some later point (a MoE block's shared experts, which read the block's input and join the routed
experts' output at the end): ``fork(fn, *args)`` runs ``fn(*args)`` on the device's side stream after
everything the main stream has queued, and ``join(token)`` makes the main stream wait for it before
the result is read. Inside an accelerator-graph capture the fork and join events become edges of the
graph, so a replay runs the branch's kernels concurrently with the main path's (vLLM overlaps
DeepSeek-style shared experts the same way). Off CUDA, ``fork`` runs ``fn`` inline and ``join`` does
nothing, so the result is the same on every device.

The branch must not communicate (no collective on the side stream: ranks would order them
differently) and its result must be read only after ``join``. Its inputs stay referenced until the
join, and its outputs are recorded on the main stream, so the caching allocator reuses neither while
the other stream may still touch them. One side stream per (device, main stream): a node capturing
on its own stream (``Engine._capture_stream_for``) gets its own.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

# (device index, main stream) -> its side stream
_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}


class _Token:
    __slots__ = ("done", "keep", "main")

    def __init__(self, done, keep, main):
        self.done, self.keep, self.main = done, keep, main


def _tensors(obj) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for o in obj for t in _tensors(o)]
    return []


def _side_stream(device: torch.device, main: torch.cuda.Stream) -> torch.cuda.Stream:
    key = (device.index if device.index is not None else torch.cuda.current_device(), main.cuda_stream)
    stream = _STREAMS.get(key)
    if stream is None:
        stream = _STREAMS[key] = torch.cuda.Stream(device=device)
    return stream


@torch.compiler.disable
def fork(fn: Callable[..., Any], *args: Any) -> tuple[Any, _Token | None]:
    """``(fn(*args), token)``, ``fn`` queued on the side stream behind the main stream's work. Read
    the result only after ``join(token)``."""
    device = next((t.device for t in _tensors(args)), None)
    if device is None or device.type != "cuda":
        return fn(*args), None
    main = torch.cuda.current_stream(device)
    side = _side_stream(device, main)
    start = torch.cuda.Event()
    start.record(main)
    side.wait_event(start)
    with torch.cuda.stream(side):
        out = fn(*args)
        done = torch.cuda.Event()
        done.record(side)
    for t in _tensors(out):
        t.record_stream(main)
    return out, _Token(done, args, main)


@torch.compiler.disable
def join(token: _Token | None) -> None:
    """The main stream waits for the branch ``fork`` queued."""
    if token is None:
        return
    torch.cuda.current_stream(token.main.device).wait_event(token.done)
    token.keep = None
