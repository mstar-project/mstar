"""The weight formats a ``quantization:`` block can name, and what this build runs of each.

A format here is a name and its facts; its kernels live with the layers (``LinearFormat`` subclasses in
``mstar/model/components``, the MoE experts' dispatch in ``components/moe.py``). ``linear`` / ``experts``
say whether this build serves the format on that layer kind; the resolver refuses a spec that asks for one
it doesn't, naming the format and the kind, so a format can be catalogued (and its checkpoints detected)
before its kernels land.
"""

from __future__ import annotations

from typing import NamedTuple

AUTO = "auto"
BF16 = "bf16"


class FormatInfo(NamedTuple):
    name: str
    # weight bits and the activation the GEMM takes ("bf16" for weight-only formats)
    weight_bits: int
    activation: str
    # the weight scale's granularity, (rows, cols) of the [N, K] weight one scale covers; None: per tensor
    block: tuple[int, int] | None
    # the oldest CUDA compute capability whose tensor cores run the GEMM
    min_capability: tuple[int, int]
    # served on a linear / on the MoE routed experts by this build
    linear: bool
    experts: bool
    doc: str


_CATALOG = (
    FormatInfo("fp8_tensor", 8, "fp8 per tensor (dynamic)", None, (8, 9), True, False,
               "e4m3, one fp32 scale per weight; torch._scaled_mm TensorWise"),
    FormatInfo("fp8_channel", 8, "fp8 per token (dynamic)", (1, 0), (8, 9), True, False,
               "e4m3, one fp32 scale per output channel; torch._scaled_mm RowWise (CUTLASS on sm90)"),
    FormatInfo("fp8_block", 8, "fp8 per token per 128 (dynamic)", (128, 128), (8, 9), True, True,
               "e4m3, one fp32 scale per 128x128 tile (DeepSeek/Qwen/GLM/MiniMax FP8 checkpoints); dense: the "
               "fp8_linear kernel registry (Triton w8a16, FlashInfer sm90, cuBLAS blockwise), experts: "
               "fused_experts_fp8"),
    FormatInfo("mxfp8", 8, "mxfp8", (1, 32), (10, 0), True, False,
               "e4m3, one e8m0 scale per 32 along K; cuBLASLt block-scaled GEMM"),
    FormatInfo("nvfp4", 4, "nvfp4", (1, 16), (10, 0), True, True,
               "e2m1, one e4m3 scale per 16 along K plus an fp32 tensor scale (modelopt nvidia/*-NVFP4 as stored, or "
               "quantized at load); cuBLASLt FP4 GEMM, experts FlashInfer's TRT-LLM FP4 MoE"),
    FormatInfo("mxfp4", 4, "bf16", (1, 32), (8, 0), True, True,
               "e2m1, one e8m0 scale per 32 along K (compressed-tensors mxfp4-pack-quantized: Kimi K3's routed "
               "experts); H100/H200: Triton, dequantized to bf16 in registers; B200/B300 experts: FlashInfer's "
               "TRT-LLM FP4 MoE"),
    FormatInfo("w4a16", 4, "bf16", (1, 128), (8, 0), True, True,
               "int4, one bf16/fp16 scale (and zero point if asymmetric) per group along K (compressed-tensors "
               "pack-quantized, AutoAWQ, GPTQ without act-order); Triton, dequantized to bf16 in registers"),
)

FORMATS: dict[str, FormatInfo] = {info.name: info for info in _CATALOG}
QUANTIZED = tuple(FORMATS)
# the KV cache's own dtype; only the model's dtype is served today
KV_CACHE_FORMATS: dict[str, str] = {"fp8_e4m3": "e4m3 pages with per-head scales (not served yet)"}


def info(name: str) -> FormatInfo:
    return FORMATS[name]


def served(name: str, kind: str) -> bool:
    """Whether this build serves format ``name`` on a layer of ``kind`` (``linear`` or ``experts``)."""
    if name == BF16:
        return True
    entry = FORMATS[name]
    return entry.linear if kind == "linear" else entry.experts
