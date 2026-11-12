"""Cacheless self-attention through Dao's FlashAttention-3 (``flash_attn_3``), for
a layout of equal-length segments, on Hopper.

FA3's batched kernel takes the packed ``[n * S, H, D]`` tensors as the
``[n, S, H, D]`` batch they are, without a transpose. On an H200 it runs a DiT's
long self-attention ~2-4% faster than FlashInfer's FA3 backend, which a ragged
schedule drives, and than FA3's own varlen kernel. ``flash_attn_3`` is not an M*
dependency: ``kernels`` reports it unavailable when it doesn't import.
"""

from mstar.engine.resources.attn.ragged.cudnn_wrappers import UniformSegmentsWrapper


class FlashAttn3RaggedPrefillWrapper(UniformSegmentsWrapper):
    """Equal-length segments through ``flash_attn_interface.flash_attn_func``."""

    def _attend(self, q, k, v, causal, return_lse):
        from flash_attn_interface import flash_attn_func

        out = flash_attn_func(q, k, v, softmax_scale=self.sm_scale, causal=causal, return_attn_probs=return_lse)
        return out if return_lse else (out, None)
