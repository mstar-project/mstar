"""Starting threads, and moving the threads a step passes through onto their CPUs.

A step passes between the worker's main thread, its GPU thread and its plan
thread several times (submit, launch, done, commit, plan ready), and each
handoff blocks one thread and wakes another. On a host with deep idle states a
thread woken on another core first waits for that core to leave C6 (130-175 µs
on the B300 host this was measured on); woken on a core that is already
running, it starts at once. The API server's result threads (the message
thread, and the Rust frontend's bridge) hand every step's tokens on the same
way. ``step_thread_core`` (``mstar.utils.cpus``) pins them.

A new thread starts with its creator's CPU mask, so a thread or pool that a
pinned thread starts would share its CPU. ``Thread`` and ``ThreadPool`` start
theirs on the process mask; ruff refuses a bare ``threading.Thread`` or
``ThreadPoolExecutor`` under ``mstar/`` (``banned-api`` in pyproject.toml).
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor

from mstar.utils.cpus import PINNABLE

# ATen splits an op into chunks of at least this many elements per thread
_INTRA_OP_GRAIN = 32768

# the process mask, recorded by the first pin, for the threads started later
_process_mask: frozenset[int] | None = None


def pin_thread(cpus: Collection[int], process_mask: Collection[int]) -> None:
    """Confine the calling thread to ``cpus``.

    A thread's intra-op (OpenMP) pool is started by its first parallel op
    with the mask the thread has then, and a pool confined to one CPU runs a
    parallel op many times slower. So go back to the process mask first (the
    main thread, already confined, starts the GPU and plan threads), start the
    pool there with one op as wide as the pool, and only then move.
    """
    import torch

    global _process_mask
    _process_mask = frozenset(process_mask)
    os.sched_setaffinity(0, _process_mask)
    torch.empty(torch.get_num_threads() * _INTRA_OP_GRAIN, dtype=torch.uint8).fill_(0)
    os.sched_setaffinity(0, set(cpus))


def _to_process_mask() -> None:
    if PINNABLE and _process_mask is not None:
        os.sched_setaffinity(0, _process_mask)


class Thread(threading.Thread):
    """A ``threading.Thread`` that runs on the process mask, whatever its
    creator's and whatever its ``run`` is: a target, an override that doesn't
    call ``super().run()``, a mixin's. Wrapped at ``start`` so that calling
    ``run`` directly stays on the caller's thread and mask."""

    def start(self) -> None:
        run = self.run

        def on_process_mask() -> None:
            try:
                _to_process_mask()
                run()
            finally:
                # as Thread.run drops its target: no cycle through the bound method
                del self.run

        self.run = on_process_mask
        super().start()


def _on_process_mask(initializer: Callable[..., object] | None) -> Callable[..., None]:
    def init(*args) -> None:
        _to_process_mask()
        if initializer is not None:
            initializer(*args)
    return init


class ThreadPool(ThreadPoolExecutor):
    """A ``ThreadPoolExecutor`` whose threads run on the process mask before
    ``initializer``, whatever the mask of the thread that started them."""

    def __init__(
        self,
        max_workers: int | None = None,
        thread_name_prefix: str = "",
        initializer: Callable[..., object] | None = None,
        initargs: tuple = (),
    ):
        super().__init__(max_workers, thread_name_prefix, _on_process_mask(initializer), initargs)
