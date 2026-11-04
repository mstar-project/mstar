"""Timing a kernel launch the way decode runs it: replayed from a CUDA graph."""
from __future__ import annotations

from collections.abc import Callable

import torch


def time_graph(launch: Callable[[], object], replays: int = 10, warmup: int = 3) -> float:
    """Microseconds per replay of a CUDA graph that holds one call of ``launch``.

    ``launch`` runs ``warmup`` times on a side stream first (Triton compiles there, outside the
    capture). A config that fails to compile or to launch raises here; callers skip it."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays
