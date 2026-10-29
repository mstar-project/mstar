"""The built-in kernel specs, by name: ``spec(shapes=None) -> KernelSpec``."""

from benchmark.kernels.specs import decode_gemv, fused_add_rms_norm, linear, rms_norm

SPECS = {
    "rms_norm": rms_norm.spec, "linear": linear.spec, "fused_add_rms_norm": fused_add_rms_norm.spec,
    "decode_gemv": decode_gemv.spec, "decode_gemv_silu_mul": decode_gemv.silu_mul_spec,
}
