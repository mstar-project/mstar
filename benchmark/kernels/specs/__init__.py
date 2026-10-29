"""The built-in kernel specs, by name: ``spec(shapes=None) -> KernelSpec``."""

from benchmark.kernels.specs import linear, rms_norm

SPECS = {"rms_norm": rms_norm.spec, "linear": linear.spec}
