"""Kernels as engine-level ops: the registry (``registry``), the parity bar
(``numerics``), tuned configs per architecture (``configs``) and the build of
C++/CUDA/HIP sources (``hip.build``). See docs/kernels.rst."""

from mstar.kernels.numerics import Tolerance
from mstar.kernels.registry import Choice, KernelChoice, KernelOp, ShapePolicy, get, load_all, ops

__all__ = ["Choice", "KernelChoice", "KernelOp", "ShapePolicy", "Tolerance", "get", "load_all", "ops"]
