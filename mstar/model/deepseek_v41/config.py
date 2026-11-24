"""DeepSeek-V4.1 serving configuration and state-contract resource keys."""

from mstar.model.components.deepseek_v41.config import DeepSeekV41Config

WINDOW_STATE = "window_state"
COMPRESS_STATE = "compress_state"
COMPRESSOR_PARTIAL = "compressor_partial"
NGRAM_STATE = "ngram_state"
REQUEST_META = "request_meta"
SAMPLER = "sampler"

__all__ = [
    "COMPRESSOR_PARTIAL",
    "COMPRESS_STATE",
    "DeepSeekV41Config",
    "NGRAM_STATE",
    "REQUEST_META",
    "SAMPLER",
    "WINDOW_STATE",
]
