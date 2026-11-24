"""Shared DeepSeek-V4.1 model components."""

from mstar.model.components.deepseek_v41.attention import (
    DeepSeekV41Attention,
    DeepSeekV41Indexer,
    SharedSelection,
    select_candidate_blocks,
)
from mstar.model.components.deepseek_v41.backbone import DeepSeekV41TextBackbone, DeepSeekV41TextCache
from mstar.model.components.deepseek_v41.compressor import DeepSeekV41Compressor
from mstar.model.components.deepseek_v41.compressor_source import (
    DeepSeekV41CompressorSource,
    compressor_partial_value,
    load_compressor_weights,
)
from mstar.model.components.deepseek_v41.config import DeepSeekV41Config
from mstar.model.components.deepseek_v41.engram import DeepSeekV41Engram, DeepSeekV41EngramEmbedding
from mstar.model.components.deepseek_v41.engram_hash import (
    EngramLayout,
    bucket_primes,
    build_compressed_token_map,
    compute_engram_hashes,
    compute_hash_multipliers,
)
from mstar.model.components.deepseek_v41.layer import DeepSeekV41Layer
from mstar.model.components.deepseek_v41.moe import DeepSeekV41MoE, DeepSeekV41MoeConfig
from mstar.model.components.deepseek_v41.quant import ReferenceRMSNorm, RoundedLinear, round_fp4, round_fp8
from mstar.model.components.deepseek_v41.rope import DeepSeekV41Rope, apply_rotary_emb, inv_freq, rotate_tail

__all__ = [
    "DeepSeekV41Attention",
    "DeepSeekV41Compressor",
    "DeepSeekV41CompressorSource",
    "DeepSeekV41Config",
    "DeepSeekV41Engram",
    "DeepSeekV41EngramEmbedding",
    "DeepSeekV41Indexer",
    "DeepSeekV41Layer",
    "DeepSeekV41MoE",
    "DeepSeekV41MoeConfig",
    "DeepSeekV41Rope",
    "DeepSeekV41TextBackbone",
    "DeepSeekV41TextCache",
    "EngramLayout",
    "ReferenceRMSNorm",
    "RoundedLinear",
    "SharedSelection",
    "apply_rotary_emb",
    "bucket_primes",
    "build_compressed_token_map",
    "compressor_partial_value",
    "compute_engram_hashes",
    "compute_hash_multipliers",
    "inv_freq",
    "load_compressor_weights",
    "rotate_tail",
    "round_fp4",
    "round_fp8",
    "select_candidate_blocks",
]
