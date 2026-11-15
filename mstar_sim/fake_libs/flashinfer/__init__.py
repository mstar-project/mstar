"""A stand-in for the parts of FlashInfer mstar calls, for running mstar on the
emulated GPU (``mstar_sim``) where no FlashInfer is installed.

Same module layout, names and signatures as FlashInfer (0.6.18 sources,
mstar's pinned 0.7.2 call shapes); each kernel returns its outputs' shapes
and charges the simulated device once, as one fused kernel
(``mstar_sim.clock``). What mstar references but this does not emulate
raises ``NotImplementedError`` when called.

It reports mstar's pinned version, so mstar's captured attention wrappers
take their direct planner path (``DIRECT_PLAN_FLASHINFER_VERSIONS``): the
wrappers keep the planner state and modules that path reads, and a direct
planner call prices the step as ``plan()`` does (``_paged.py``).
"""

from . import (
    activation,
    comm,
    decode,
    gdn_decode,
    gdn_prefill,
    gemm,
    mla,
    norm,
    prefill,
    rope,
    sampling,
    topk,
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
from .topk import TopKTieBreak, top_k_page_table_transform

# The emulator's planner version: _paged.py and mla.py port its scheduler.cuh sizing.
# The optional compatibility audit reports drift from mstar's pin. Updating this port,
# not merely mstar's dependency, requires regenerating its golden (python -m mstar_sim.planner_golden).
__version__ = "0.7.2"

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
    "gemm",
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
    "TopKTieBreak",
    "top_k_page_table_transform",
    "topk",
    "top_k_renorm_probs",
    "top_k_top_p_sampling_from_probs",
    "top_p_renorm_probs",
    "top_p_sampling_from_probs",
    "utils",
]
