"""A stand-in for the parts of FlashInfer mstar calls, for running mstar on the
emulated GPU (``mstar_sim``) where no FlashInfer is installed.

Same module layout, names and signatures as FlashInfer (0.6.18 sources,
mstar's pinned 0.7.2 call shapes); each kernel returns its outputs' shapes
and charges the simulated device once, as one fused kernel
(``mstar_sim.clock``). What mstar references but this does not emulate
raises ``NotImplementedError`` when called.

It reports mstar's pinned version (read from ``DIRECT_PLAN_FLASHINFER_VERSIONS``),
so mstar's captured attention wrappers take their direct planner path: the
wrappers keep the planner state and modules that path reads, and a direct
planner call prices the step as ``plan()`` does (``_paged.py``).
"""

import sys

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


def __getattr__(name: str):
    if name == "__version__":
        # mstar's pinned version, the one its direct planner path checks, so the stand-in takes
        # the path a GPU with the pin takes; on a tree without that check, the sources it follows
        try:
            from mstar.engine.resources.attn.wrappers import DIRECT_PLAN_FLASHINFER_VERSIONS
        except ImportError:
            return "0.6.18"
        return max(DIRECT_PLAN_FLASHINFER_VERSIONS, key=lambda v: tuple(int(p) for p in v.split(".") if p.isdigit()))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}", name=name, obj=sys.modules[__name__])


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
