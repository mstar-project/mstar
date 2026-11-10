"""Ricursive's Ripple-65B: a dense text LLM, one ``LLM`` node on the shared decoder scaffold."""
import logging
from pathlib import Path

import torch

from mstar.engine.precision import NodePrecision
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.model.keys import MIXED
from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards
from mstar.model.multimodal import PromptPart, chat_turns
from mstar.model.multimodal_decoder import MIXED_SPLIT, MultimodalDecoderModel
from mstar.model.rebuild import RebuildPolicy
from mstar.model.ripple.config import ATTN_FULL, ATTN_SLIDING, KV_CACHE, ROPE, RippleConfig
from mstar.model.submodule_base import ChunkingPolicy, NodeSubmodule
from mstar.utils.fast_tokenize import encode_ids

logger = logging.getLogger(__name__)

_STACKED_PARAMS = [
    StackedParamRule(".self_attn.qkg_proj", ".self_attn.q_proj", 0),
    StackedParamRule(".self_attn.qkg_proj", ".self_attn.k_proj", 1),
    StackedParamRule(".self_attn.qkg_proj", ".self_attn.g_proj", 2),
    StackedParamRule(".mlp.gate_up_proj", ".mlp.gate_proj", 0),
    StackedParamRule(".mlp.gate_up_proj", ".mlp.up_proj", 1),
]


class RippleModel(MultimodalDecoderModel):
    NAME = "Ripple-65B"
    # chat_template.jinja is the tools-aware template; tokenizer_config.json holds a simpler one
    METADATA_FILES = (
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    )

    def __init__(self, model_path_hf: str, cache_dir: str | None = None):
        from transformers import AutoTokenizer

        super().__init__(model_path_hf, cache_dir)
        self.config = RippleConfig.from_hf(self.local_dir)
        # the checkpoint ships no generation_config.json; the vendor's server samples at 1.0
        self.sampling_defaults = {"temperature": 1.0, **self.sampling_defaults}
        # config.json's auto_map would otherwise stop a server at a prompt to run the repo's code
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, local_files_only=True, trust_remote_code=False)

    def request_kwargs(self) -> frozenset[str]:
        # chat_template.jinja reads no enable_thinking: the reasoning stays on, so a request asking it off is ignored
        return super().request_kwargs() - {"enable_thinking"}

    def rebuild_policies(self) -> dict[str, RebuildPolicy]:
        # not yet dropped: no drop and rebuild has run through its sliding-window KV pool
        return {}

    def get_node_precision(self, node_name: str) -> NodePrecision:
        """bf16 under bf16 autocast, as the model has served."""
        del node_name
        return NodePrecision(torch.bfloat16, autocast=True)

    def get_node_resources(self) -> list[NodeResourceSpec]:
        cfg = self.config
        # each layer of the stack is the cache's layer at its stack index; the cache keeps only
        # the window of the ones the sliding attention runs
        sliding = tuple(i for i, s in enumerate(cfg.is_sliding) if s)
        return [
            KVSpec(
                resource_key=KV_CACHE, nodes={"LLM"},
                config=PagedKVConfig(
                    num_layers=cfg.num_hidden_layers,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    max_seq_len=cfg.max_position_embeddings,
                    num_qo_heads=cfg.num_attention_heads,
                ),
            ),
            AttentionSpec(
                resource_key=ATTN_SLIDING, nodes={"LLM"},
                config=AttentionConfig(
                    kv_cache=KV_CACHE, sliding_window=cfg.sliding_window, layers=sliding, splits=(MIXED_SPLIT,),
                ),
            ),
            AttentionSpec(
                resource_key=ATTN_FULL, nodes={"LLM"},
                config=AttentionConfig(kv_cache=KV_CACHE, splits=(MIXED_SPLIT,)),
            ),
            PositionSpec(
                resource_key=ROPE, nodes={"LLM"},
                # the reference rounds its cos/sin to bf16 and rotates in fp32
                config=PositionConfig(
                    kv_cache=KV_CACHE, rotary_dim=cfg.head_dim, rope_theta=cfg.rope_theta,
                    rope_table_dtype=torch.bfloat16,
                ),
            ),
            self._sampler_spec(cfg.vocab_size),
        ]

    def get_chunking_policies(self) -> dict[str, tuple[ChunkingPolicy, ...]]:
        from mstar.model.ripple.submodules import RippleLLMSubmodule

        budget = RippleLLMSubmodule.MAX_BATCH_TOKENS
        return {"LLM": (
            ChunkingPolicy(walks=("prefill_text",), chunkable=True, max_batch_tokens=budget),
            # carries the budget for the steps that mix a chunk with decode rows
            ChunkingPolicy(walks=(MIXED,), max_batch_tokens=budget),
        )}

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        # the template joins each message's content as a string
        messages = [
            {"role": turn.role, "content": "".join(part.text or "" for part in turn.parts)}
            for turn in chat_turns(parts)
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return torch.tensor(encode_ids(self.tokenizer, text), dtype=torch.long)

    def _create_submodule(
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule | None:
        from mstar.model.ripple.language_model import RippleForCausalLM
        from mstar.model.ripple.submodules import RippleLLMSubmodule

        if node_name != "LLM":
            return None
        model = self._build(lambda: RippleForCausalLM(self.config, tp_group), dtype, device)
        load_weights_into(
            model, iter_safetensors_shards(Path(self._resolve_weights()), device=device),
            stacked_params=_STACKED_PARAMS, strict=True,
        )
        model.requires_grad_(False).eval()
        logger.info("Loaded Ripple-65B onto %s", device)
        return RippleLLMSubmodule(model, self.config)
