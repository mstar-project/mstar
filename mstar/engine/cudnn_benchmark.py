"""Per-node convolution algorithm search, applied by the engine around a node's forward.

``torch.backends.cudnn.benchmark`` makes cuDNN (MIOpen on AMD GPUs) time the
candidate algorithms on the first call at each new convolution input size and
cache the fastest, where it would otherwise take a heuristic pick. That pays off
for a node whose convs see a few fixed sizes, such as a vocoder captured per
batch size, and costs a search at every new size for one whose input size
changes per request, such as an image or audio encoder (up to a few hundred ms
per conv and size on MIOpen). The flag is process-wide, so a node that wants the
search declares it (``NodeSubmodule.cudnn_benchmark``) and the engine sets it
around that node's forward launch, capture sizing, warm-up and capture only, as
it does the float32 settings (``mstar.engine.fp32_precision``, whose notes on
threads apply here too). A capture itself runs with the search off
(``capture_graph``) and records the algorithm the warm-up's search cached for
that size.
"""
import contextlib

import torch

_NO_CHANGE = contextlib.nullcontext()


def node_cudnn_benchmark(node_name: str, submodule) -> bool | None:
    """The submodule's declared setting, checked when the engine loads it."""
    benchmark = getattr(submodule, "cudnn_benchmark", None)
    if benchmark is not None and not isinstance(benchmark, bool):
        raise ValueError(f"{node_name}: cudnn_benchmark must be a bool or None, got {benchmark!r}")
    return benchmark


def cudnn_benchmark_scope(benchmark: bool | None):
    """Set ``torch.backends.cudnn.benchmark`` for the block, then restore what
    was there before, also when the block raises. None changes nothing."""
    if benchmark is None:
        return _NO_CHANGE
    return _applied(benchmark)


@contextlib.contextmanager
def _applied(benchmark: bool):
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = benchmark
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous
