"""A stand-in for the parts of FlashInfer mstar calls, for running mstar on the
emulated GPU (``mstar_sim``) where no FlashInfer is installed.

Same module layout, names and signatures as FlashInfer (0.6.18 sources,
mstar's pinned 0.7.2 call shapes); each kernel returns its outputs' shapes
and charges the simulated device once, as one fused kernel
(``mstar_sim.clock``). What mstar references but this does not emulate
raises ``NotImplementedError`` when called.

The version is not one mstar's attention wrappers restate FlashInfer's
planner internals for (``DIRECT_PLAN_FLASHINFER_VERSIONS``), so they plan
through ``plan()``, which is what this emulates.
"""

from . import (
    activation,
    comm,
    decode,
    gdn_decode,
    gdn_prefill,
    mla,
    norm,
    prefill,
    rope,
    sampling,
    utils,
)
from .activation import silu_and_mul
from .decode import BatchDecodeWithPagedKVCacheWrapper
from .gdn_prefill import chunk_gated_delta_rule
from .mla import BatchMLAPagedAttentionWrapper
from .norm import fused_add_rmsnorm, gemma_fused_add_rmsnorm, gemma_rmsnorm, rmsnorm
from .prefill import BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper
from .rope import (
    apply_llama31_rope_pos_ids_inplace,
    apply_rope_pos_ids_inplace,
    apply_rope_with_cos_sin_cache_inplace,
)
from .sampling import (
    sampling_from_probs,
    top_k_renorm_probs,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_probs,
    top_p_sampling_from_probs,
)

__version__ = "0.7.2+mstar_sim"

__all__ = [
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchMLAPagedAttentionWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
    "activation",
    "apply_llama31_rope_pos_ids_inplace",
    "apply_rope_pos_ids_inplace",
    "apply_rope_with_cos_sin_cache_inplace",
    "chunk_gated_delta_rule",
    "comm",
    "decode",
    "fused_add_rmsnorm",
    "gdn_decode",
    "gdn_prefill",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "mla",
    "norm",
    "prefill",
    "rmsnorm",
    "rope",
    "sampling",
    "sampling_from_probs",
    "silu_and_mul",
    "top_k_renorm_probs",
    "top_k_top_p_sampling_from_probs",
    "top_p_renorm_probs",
    "top_p_sampling_from_probs",
    "utils",
]
