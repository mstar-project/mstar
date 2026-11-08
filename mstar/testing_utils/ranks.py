"""Start a test's ranks without paying an interpreter's startup per rank.

A "spawn" rank starts a fresh interpreter and re-imports torch and the worker
stack, about 3 CPU-seconds before it runs any test code; a multi-rank test
module pays that once per rank per test. These helpers fork each rank from a
server process that has those imports done. The server forks from its own clean
state, not from the test process, so a rank inherits no test state and no
threads, as with "spawn".

Keep "spawn" for a child whose test depends on a fresh interpreter or on being
the test process's own child (orphan and watchdog tests read ``getppid``).
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable

import torch.multiprocessing as torch_mp

# A module that fails to import here is skipped by the server, and imported by
# whichever rank needs it.
_PRELOAD = ["torch", "torch.distributed", "mstar.worker.worker"]


def rank_context() -> multiprocessing.context.BaseContext:
    """A multiprocessing context whose processes fork from the preloaded server."""
    ctx = multiprocessing.get_context("forkserver")
    # Process-wide, and read only when the server first starts.
    ctx.set_forkserver_preload(_PRELOAD)
    return ctx


def spawn_ranks(fn: Callable[..., object], args: tuple = (), nprocs: int = 1) -> None:
    """``torch.multiprocessing.spawn(fn, args, nprocs)`` from the preloaded server:
    runs ``fn(rank, *args)`` on each rank, joins them, and raises if any rank failed."""
    rank_context()
    torch_mp.start_processes(fn, args=args, nprocs=nprocs, join=True, start_method="forkserver")
