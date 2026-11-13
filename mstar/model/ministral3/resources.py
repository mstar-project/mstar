"""Declarations for the text node, returned by the Model's resource hook.

Nothing here constructs a resource. The engine builds, sizes and binds them;
deployments tune the standard KV capacity through their resources overrides.
"""

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    SamplerSpec,
)
from mstar.model.keys import KV_CACHE, ROPE, SAMPLER
from mstar.model.ministral3.config import ATTN, LLM_NODE, Ministral3TextConfig


def text_node_resources(
    config: Ministral3TextConfig, *, enable_min_p: bool = False,
) -> list[NodeResourceSpec]:
    return [
        KVSpec(resource_key=KV_CACHE, nodes={LLM_NODE}, config=PagedKVConfig(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            num_qo_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
        )),
        AttentionSpec(resource_key=ATTN, nodes={LLM_NODE}, config=AttentionConfig(
            kv_cache=KV_CACHE,
        )),
        PositionSpec(resource_key=ROPE, nodes={LLM_NODE}, config=PositionConfig(
            kv_cache=KV_CACHE,
            rotary_dim=config.head_dim,
            rope_theta=config.rope_theta,
            yarn=config.yarn,
        )),
        SamplerSpec(
            resource_key=SAMPLER, nodes={LLM_NODE}, vocab_size=config.vocab_size, enable_min_p=enable_min_p,
        ),
    ]
