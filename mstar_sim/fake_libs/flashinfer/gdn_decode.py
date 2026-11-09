"""FlashInfer's GDN decode (the fp32-state path; there is no bf16-state backend)."""

from mstar_sim.kernels import gated_delta_rule_decode_pretranspose

_GDN_DECODE_BF16_STATE_AVAILABLE = False

__all__ = ["gated_delta_rule_decode_pretranspose"]
