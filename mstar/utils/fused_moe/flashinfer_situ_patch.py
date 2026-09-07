"""Teach FlashInfer's CUTLASS fused-MoE (SM90 mixed-input MXFP4 paths) the SiTU-GLU activation.

FlashInfer 0.6.18 lists ``ActivationType.Situ`` in its Python enum for the trtllm-gen (SM100)
MoE, but the JIT-compiled CUTLASS backend used on Hopper has no such activation: its C++ enum
stops at ``Identity``, so ``Situ`` (10) lands on ``InvalidType`` and the launch throws. This
patch (idempotent, additive) edits the JIT sources shipped inside the installed package:

* ``common.h``: ``Situ`` enum member after ``Identity`` (value 10, matching Python);
* ``moe_gemm_kernels.h``: ``isGatedActivation`` treats ``Situ`` as gated (fc1 is ``[up | gate]``);
* ``moe_gemm_template_dispatch.h``: the bias-act GEMM switch maps ``Situ`` to the identity
  epilogue (the activation runs in the separate activation kernel on Hopper);
* ``cutlass_fused_moe_kernels.cuh``: a ``SituAdaptor`` functor
  ``alpha*tanh(gate/alpha)*sigmoid(gate) * beta*tanh(linear/beta)`` (alpha/beta from the
  per-expert ``swiglu_alpha``/``swiglu_beta`` tensors: 4 and 25 for Kimi K3) and its entries
  in both activation-kernel dispatch tables.

Run ``python -m mstar.utils.fused_moe.flashinfer_situ_patch`` once per environment (it also
clears the cached ``fused_moe_90`` build so the next call recompiles). ``--check`` only reports.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

SITU_ADAPTOR = '''
// SiTU-GLU (Kimi K3): alpha * tanh(gate / alpha) * sigmoid(gate) * beta * tanh(linear / beta).
// alpha and beta arrive through the per-expert swiglu_alpha / swiglu_beta arrays (4 and 25 for
// Kimi K3); a non-positive beta leaves the linear branch unchanged (the Kimi-Linear form).
struct SituAdaptor {
  constexpr static bool IS_GLU = true;
  float alpha = 4.0f;
  float beta = 25.0f;
  float limit = std::numeric_limits<float>::infinity();

  template <class T>
  __device__ T operator()(T const& gate, T const& linear) const {
    cutlass::epilogue::thread::Sigmoid<T> sigmoid{};
    cutlass::epilogue::thread::Tanh<T> tanh{};
    T gate_act = tanh(gate * (1.0f / alpha)) * alpha * sigmoid(gate);
    if (beta > 0.0f) {
      return gate_act * (tanh(linear * (1.0f / beta)) * beta);
    }
    return gate_act * linear;
  }
};
'''

# (relative path under flashinfer/data/csrc, anchor, replacement, marker that proves it is applied)
EDITS = [
    (
        "nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/common.h",
        "  Identity,\n  InvalidType\n",
        "  Identity,\n  Situ,\n  InvalidType\n",
        "  Situ,\n",
    ),
    (
        "nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h",
        "         activation_type == ActivationType::GegluTanh;\n}",
        "         activation_type == ActivationType::GegluTanh ||\n"
        "         activation_type == ActivationType::Situ;\n}",
        "activation_type == ActivationType::Situ;",
    ),
    (
        "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h",
        "    case ActivationType::InvalidType:\n      TLLM_THROW(\"Activation type for fpA_intB must be valid.\");",
        "    case ActivationType::Situ:\n"
        "      // the SiTU activation runs in the activation kernel; GEMM1 keeps the identity epilogue\n"
        "      runGemm<cutlass_extensions::EpilogueOpDefault>(inputs, hopper_inputs);\n"
        "      break;\n"
        "    case ActivationType::InvalidType:\n      TLLM_THROW(\"Activation type for fpA_intB must be valid.\");",
        "case ActivationType::Situ:",
    ),
    (
        "fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
        "// ============================== Gated Activation =================================\n",
        SITU_ADAPTOR + "\n// ============================== Gated Activation =================================\n",
        "struct SituAdaptor {",
    ),
    (
        "fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
        "                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, SwigluStepAdaptor>\n"
        "                 : nullptr;",
        "                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, SwigluStepAdaptor>\n"
        "             : activation_type == ActivationType::Situ\n"
        "                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, SituAdaptor>\n"
        "                 : nullptr;",
        "doGatedActivationKernel<ActivationOutputType, GemmOutputType, SituAdaptor>",
    ),
    (
        "fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
        "                              decltype(nvfp4_4over6_config_tag)>  // Identity\n      };",
        "                              decltype(nvfp4_4over6_config_tag)>,  // Identity\n"
        "          &doActivationKernel<T, GemmOutputType, ScaleBiasType, SituAdaptor,\n"
        "                              decltype(block_scaling_type)::value,\n"
        "                              decltype(disableFP4QuantFastMathTag)::value,\n"
        "                              decltype(nvfp4_4over6_config_tag)>  // Situ\n      };",
        "decltype(nvfp4_4over6_config_tag)>  // Situ",
    ),
]


def csrc_root() -> pathlib.Path:
    import flashinfer

    return pathlib.Path(flashinfer.__file__).parent / "data" / "csrc"


def cached_build_dirs() -> list[pathlib.Path]:
    try:
        from flashinfer.jit.env import FLASHINFER_JIT_DIR

        base = pathlib.Path(FLASHINFER_JIT_DIR)
    except Exception:
        return []
    return [p for p in base.glob("**/cached_ops/fused_moe_*") if p.is_dir()]


def apply(check_only: bool = False) -> int:
    root = csrc_root()
    changed = 0
    for rel, anchor, replacement, marker in EDITS:
        path = root / rel
        text = path.read_text()
        if marker in text:
            print(f"already patched: {rel} ({marker.strip()[:40]})")
            continue
        if anchor not in text:
            print(f"ANCHOR NOT FOUND in {rel}: {anchor[:60]!r}", file=sys.stderr)
            return 2
        if text.count(anchor) != 1:
            print(f"ANCHOR AMBIGUOUS in {rel}", file=sys.stderr)
            return 2
        if check_only:
            print(f"would patch: {rel}")
        else:
            path.write_text(text.replace(anchor, replacement))
            print(f"patched: {rel}")
        changed += 1
    if changed and not check_only:
        for d in cached_build_dirs():
            shutil.rmtree(d, ignore_errors=True)
            print(f"removed cached build: {d}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report without editing")
    sys.exit(apply(check_only=ap.parse_args().check))
