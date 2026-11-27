"""Ricursive's Ripple-65B: a dense text LLM, one ``LLM`` node on the shared decoder scaffold."""
import logging
from pathlib import Path

import torch

from mstar.engine.state import Array, Attend, Causal, KVEntry, Positions, Read, Rope, Sampler, SampleSettings, Window
from mstar.model.keys import MIXED
from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards
from mstar.model.multimodal import PromptPart, chat_turns
from mstar.model.multimodal_decoder import MIXED_SPLIT, SAMPLING_KEYS, MultimodalDecoderModel
from mstar.model.rebuild import RebuildPolicy
from mstar.model.ripple.config import ATTN_FULL, ATTN_SLIDING, KV_CACHE, SAMPLER, RippleConfig
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

    def declare_state(self) -> list[Array | Sampler]:
        cfg = self.config
        every = frozenset(range(cfg.num_hidden_layers))
        # the cache keeps only the window of the layers the sliding read runs
        sliding = frozenset(i for i, s in enumerate(cfg.is_sliding) if s)
        attend = Attend(q_heads=cfg.num_attention_heads)
        # the reference rounds its cos/sin to bf16 and rotates in fp32
        rope = Rope(theta=cfg.rope_theta, rotary_dim=cfg.head_dim, table_dtype=torch.bfloat16)
        return [
            Array(
                KV_CACHE, nodes={"LLM"}, layers=every,
                entry=KVEntry(heads=cfg.num_key_value_heads, head_dim=cfg.head_dim),
                max_len=cfg.max_position_embeddings, positions=Positions(rope=rope),
                reads={
                    ATTN_SLIDING: Read(mask=Window(cfg.sliding_window), op=attend, layers=sliding),
                    ATTN_FULL: Read(mask=Causal(), op=attend, layers=every - sliding),
                },
            ),
            Sampler(
                SAMPLER, nodes={"LLM"}, vocab_size=cfg.vocab_size, penalty=True,
                min_p=self._sampler_spec(cfg.vocab_size).enable_min_p,
            ),
        ]

    def kernel_settings(self) -> dict[str, dict]:
        return {ATTN_SLIDING: {"splits": (MIXED_SPLIT,)}, ATTN_FULL: {"splits": (MIXED_SPLIT,)}}

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, SampleSettings]:
        config = super().get_request_resource_configs(input_modalities, output_modalities, model_kwargs)[SAMPLER]
        return {SAMPLER: SampleSettings(**{k: getattr(config, k) for k in SAMPLING_KEYS})}

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
