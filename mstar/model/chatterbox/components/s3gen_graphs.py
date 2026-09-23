"""CUDA-graph replay of whole flow solves.

The flow-matching estimator is 56 small transformer blocks of width 256 over
a few hundred mel frames. One Euler step launches several hundred kernels of
a few microseconds each, so a solve (10 steps, guidance doubling the rows)
is bound by kernel launches rather than by arithmetic, whatever the batch.
Capturing a whole solve, the estimator calls, the guidance mix and the Euler
updates, as one CUDA graph per shape and replaying it removes that overhead.

A shape is ``(rows, frames, steps)``. Frames are already padded to a bucket
by ``S3Gen.tokens_to_mel_rows``; rows are padded here to the next captured
row count (zero rows behind a zero mask, cut off the result). Graphs are
captured on first use, or ahead of time with ``warmup``, and share one memory
pool: they never run concurrently and every static input and output stays
allocated, so no replay can write into another graph's memory. The least
recently used graph is dropped past ``max_graphs``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch

Solve = Callable[..., torch.Tensor]  # (mu, mask, spks, cond, noise, n_timesteps) -> mel


@dataclass
class _Entry:
    graph: object  # torch.cuda.CUDAGraph (or a stand-in with ``replay``)
    inputs: dict[str, torch.Tensor]  # static, ``rows`` deep
    output: torch.Tensor


class SolveGraphs:
    """Replays ``solve`` from a captured CUDA graph per ``(rows, frames, steps)``.

    ``solve`` takes ``(mu, mask, spks, cond, noise, n_timesteps)`` (batch first,
    mel frames last) and returns the solved mel; it must not draw random
    numbers or synchronise. ``rows`` are the row counts captured; a batch
    beyond the largest runs eagerly. Off the GPU everything runs eagerly.
    """

    INPUTS = ("mu", "mask", "spks", "cond", "noise")

    def __init__(
        self, solve: Solve, *, rows: Sequence[int] = (1, 2, 4, 8), max_graphs: int = 64,
    ):
        if not rows or min(rows) < 1:
            raise ValueError("rows must be positive row counts")
        self._solve = solve
        self.rows = tuple(sorted(set(int(r) for r in rows)))
        self.max_graphs = int(max_graphs)
        self._graphs: OrderedDict[tuple[int, int, int], _Entry] = OrderedDict()
        self._pool = None
        self.captures = 0
        self.replays = 0

    # -- shape bookkeeping ---------------------------------------------------

    def rows_for(self, batch: int) -> int | None:
        """The captured row count a ``batch``-row solve is padded to, or None."""
        for r in self.rows:
            if r >= batch:
                return r
        return None

    def __len__(self) -> int:
        return len(self._graphs)

    def keys(self) -> list[tuple[int, int, int]]:
        return list(self._graphs)

    # -- solving -------------------------------------------------------------

    def __call__(
        self, mu: torch.Tensor, mask: torch.Tensor, spks: torch.Tensor, cond: torch.Tensor,
        noise: torch.Tensor, n_timesteps: int,
    ) -> torch.Tensor:
        batch, _, frames = mu.shape
        rows = self.rows_for(batch)
        if rows is None or not self._capturable(mu):
            return self._solve(mu, mask, spks, cond, noise, n_timesteps)
        key = (rows, int(frames), int(n_timesteps))
        entry = self._graphs.get(key)
        inputs = dict(zip(self.INPUTS, (mu, mask, spks, cond, noise), strict=True))
        if entry is None:
            entry = self._capture(key, inputs)
        else:
            self._graphs.move_to_end(key)
        self._load(entry, inputs, batch)
        entry.graph.replay()
        self.replays += 1
        return entry.output[:batch].clone()

    def warmup(self, frames: Iterable[int], n_timesteps: int, example: dict[str, torch.Tensor]) -> int:
        """Capture every ``(rows, frames)`` shape ahead of time from ``example``
        inputs (one row each, frames ≥ every requested length; the head is used).
        Returns how many graphs were captured."""
        before = self.captures
        for f in sorted(set(int(x) for x in frames)):
            for rows in self.rows:
                key = (rows, f, int(n_timesteps))
                if key in self._graphs:
                    continue
                inputs = {
                    name: (t[:1] if name == "spks" else t[:1, :, :f]) for name, t in example.items()
                }
                self._capture(key, inputs)
        return self.captures - before

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _capturable(mu: torch.Tensor) -> bool:
        return mu.is_cuda

    def _load(self, entry: _Entry, inputs: dict[str, torch.Tensor], batch: int) -> None:
        for name, value in inputs.items():
            static = entry.inputs[name]
            static[:batch].copy_(value)
            if batch < static.shape[0]:
                static[batch:].zero_()

    def _capture(self, key: tuple[int, int, int], inputs: dict[str, torch.Tensor]) -> _Entry:
        rows, frames, n_timesteps = key
        while len(self._graphs) >= self.max_graphs:
            self._graphs.popitem(last=False)
        static = {}
        for name, value in inputs.items():
            shape = (rows, *value.shape[1:])
            buf = torch.zeros(shape, dtype=value.dtype, device=value.device)
            buf[: value.shape[0]].copy_(value)
            static[name] = buf

        def run() -> torch.Tensor:
            return self._solve(
                static["mu"], static["mask"], static["spks"], static["cond"], static["noise"], n_timesteps,
            )

        graph, output = self._record(run)
        entry = _Entry(graph=graph, inputs=static, output=output)
        self._graphs[key] = entry
        self.captures += 1
        return entry

    def _record(self, run: Callable[[], torch.Tensor]):
        """Warm ``run`` up on a side stream (cuBLAS/cuDNN pick their kernels),
        then capture it into the shared pool."""
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            output = run()
        return graph, output
