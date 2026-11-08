"""Resource keys and walk names the shared model scaffolds read.

A model that builds on `SampledTokenDecoder` or `PackedBlockLoopEncoder`
declares these resources under these keys. Kept free of imports, so a model's
config module can name them without loading torch or the engine.
"""

KV_CACHE = "kv_cache"
ROPE = "rope"
SAMPLER = "sampler"
# a vision tower's cacheless attention
VISION_ATTN = "vision_attn"
# text prefill and decode rows batched in one step (`Model.get_combined_graph_walks`)
MIXED = "mixed"
