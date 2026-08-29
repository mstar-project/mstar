"""FlashInfer helpers used from model code.

The batched paged-attention wrappers live in
``mstar.engine.resources.attn.wrappers``.
"""

import torch


def run_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-06,
    rms_norm_dtype=None
):
    """RMS norm through ``mstar::flashinfer_rmsnorm``.

    Behind an op rather than called directly: FlashInfer's kernel is a TVM-FFI
    call dynamo can't trace, and a break here lands inside every decoder
    layer's ``input_layernorm`` — which makes the layer body a frame dynamo
    recompiles once per layer.
    """
    from mstar.engine.resources.attn.wrappers import flashinfer_rmsnorm

    del flashinfer_rmsnorm  # imported for its registration side effect
    return torch.ops.mstar.flashinfer_rmsnorm(input, weight, eps, rms_norm_dtype)
