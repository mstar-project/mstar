"""Two independent branches of a captured step on two streams.

A decode step of a large model is hundreds of small kernels in a row; where a layer has two
branches that do not depend on each other (a router and a projection reading the same input, the
query and key paths of an attention layer), running them on two streams lets their kernels overlap
and takes their latencies off the critical path. ``Fork.run`` does that only while a CUDA graph is
being captured, where the fork and join are recorded as graph dependencies and the graph's memory
pool keeps the branches' tensors alive across streams; eagerly (prefill, CPU, tests) it just calls
the two functions in order, so results are the same either way.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import torch

_enabled = os.environ.get("MSTAR_AUX_STREAM", "0") == "1"  # off until the served windows say it pays


def aux_stream_enabled() -> bool:
    """``MSTAR_AUX_STREAM=1`` turns the two-stream captures on (off by default until measured)."""
    return _enabled


class Fork:
    """One auxiliary stream and the two events that fork from and join the current stream."""

    def __init__(self) -> None:
        self._stream: torch.cuda.Stream | None = None
        self._fork: torch.cuda.Event | None = None
        self._join: torch.cuda.Event | None = None

    def _lazy(self) -> None:
        if self._stream is None:
            self._stream = torch.cuda.Stream()
            self._fork = torch.cuda.Event()
            self._join = torch.cuda.Event()

    def run(self, fn0: Callable[[], Any], fn1: Callable[[], Any]) -> tuple[Any, Any]:
        """``fn0`` on the current stream, ``fn1`` on the auxiliary one, joined before returning,
        while a capture is in progress; otherwise both in order on the current stream."""
        if not (_enabled and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            return fn0(), fn1()
        self._lazy()
        self._fork.record()
        r0 = fn0()
        with torch.cuda.stream(self._stream):
            self._fork.wait()
            r1 = fn1()
            self._join.record()
        self._join.wait()
        return r0, r1
