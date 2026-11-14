"""Dense Qwen3 (Qwen3-8B and its siblings): text in, text out on one ``LLM``
node of the shared decoder scaffold, on the shared Qwen3 decoder."""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import torch
from jinja2 import TemplateError

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.model.keys import KV_CACHE, ROPE
from mstar.model.multimodal import TEXT, PromptPart, chat_turns
from mstar.model.multimodal_decoder import MIXED_SPLIT, MultimodalDecoderModel
from mstar.model.qwen3.config import ATTN, LLM_NODE, qwen3_lm_config
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.fast_tokenize import encode_ids

logger = logging.getLogger(__name__)


class Qwen3Model(MultimodalDecoderModel):
    NAME = "Qwen3"
    # the tokenizer prefers chat_template.jinja to tokenizer_config.json's template,
    # and some repos keep theirs nowhere else
    METADATA_FILES = (
        "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
    )

    def __init__(self, model_path_hf: str, cache_dir: str | None = None):
        super().__init__(model_path_hf, cache_dir)
        hf = json.loads((Path(self.local_dir) / "config.json").read_text())
        self.config = qwen3_lm_config(hf)
        self.config.stop_token_ids = self._stop_token_ids(hf)

    @functools.cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.local_dir)

    def _stop_token_ids(self, hf: dict) -> frozenset[int]:
        """Every EOS the configs name, and the tokenizer's: a chat turn ends on
        ``<|im_end|>``, which ``config.json`` may not list."""
        local = Path(self.local_dir)
        generation = local / "generation_config.json"
        sources = [hf.get("eos_token_id")]
        if generation.is_file():
            sources.append(json.loads(generation.read_text()).get("eos_token_id"))
        if any((local / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
            sources.append(self.tokenizer.eos_token_id)
        ids = set()
        for source in sources:
            if source is not None:
                ids.update(source if isinstance(source, list) else [source])
        return frozenset(int(i) for i in ids)

    def request_kwargs(self) -> frozenset[str]:
        # /generate may nest enable_thinking as OpenAI clients send it
        return super().request_kwargs() | {"chat_template_kwargs"}

    def get_node_resources(self) -> list[NodeResourceSpec]:
        cfg, nodes = self.config, {LLM_NODE}
        return [
            KVSpec(
                resource_key=KV_CACHE, nodes=nodes,
                config=PagedKVConfig(
                    num_layers=cfg.num_hidden_layers,
                    num_kv_heads=cfg.num_key_value_heads, num_qo_heads=cfg.num_attention_heads,
                    head_dim=cfg.head_dim,
                    max_seq_len=cfg.max_position_embeddings,
                ),
            ),
            AttentionSpec(
                resource_key=ATTN, nodes=nodes, config=AttentionConfig(kv_cache=KV_CACHE, splits=(MIXED_SPLIT,)),
            ),
            PositionSpec(
                resource_key=ROPE, nodes=nodes, config=PositionConfig(kv_cache=KV_CACHE, rope_theta=cfg.rope_theta),
            ),
            self._sampler_spec(cfg.vocab_size),
        ]

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        """The chat through the checkpoint's template, the assistant turn opened.
        ``enable_thinking``, flat or in ``chat_template_kwargs``, reaches the template."""
        messages = [
            {"role": turn.role, "content": "\n".join(p.text or "" for p in turn.parts if p.modality == TEXT)}
            for turn in chat_turns(parts)
        ]
        template_kwargs = kwargs.get("chat_template_kwargs")
        if template_kwargs is not None and not isinstance(template_kwargs, dict):
            # a ValueError is a 400; anything else reaching the data worker is a 500
            raise ValueError("chat_template_kwargs must be an object")
        thinking = kwargs.get("enable_thinking")
        if thinking is None and template_kwargs is not None:
            thinking = template_kwargs.get("enable_thinking")
        if thinking is not None and not isinstance(thinking, bool):
            raise ValueError(f"enable_thinking must be a boolean; got {thinking!r}")
        extra = {} if thinking is None else {"enable_thinking": thinking}
        try:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **extra,
            )
        except TemplateError as exc:
            raise ValueError(f"the chat template refused the chat: {exc}") from exc
        # the rendered text through the Rust backend in parallel chunks: a long prompt's
        # single-threaded encode would hold up the data worker
        return torch.tensor(encode_ids(self.tokenizer, text), dtype=torch.long)

    def _create_submodule(
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule | None:
        if node_name != LLM_NODE:
            return None
        weights = Path(self._resolve_weights())
        if not any(weights.glob("*.safetensors")):
            logger.info("Qwen3: no weights in %s, dummy mode", weights)
            return None
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3.submodules import Qwen3LMSubmodule

        lm = self.build_lm(device, tp_group, dtype)
        lm.load_weights(
            (key.removeprefix("model."), tensor) for key, tensor in iter_safetensors_shards(weights, device=device)
            if key.startswith(("model.", "lm_head."))
        )
        lm.requires_grad_(False).eval()
        return Qwen3LMSubmodule(lm, self.config)

    def build_lm(self, device, tp_group=None, dtype: torch.dtype = torch.bfloat16):
        """The decoder on ``device``, weights unset."""
        from mstar.model.components.qwen3_lm import Qwen3DenseLM

        with torch.device("meta"):
            lm = Qwen3DenseLM(self.config, attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE, comm_group=tp_group)
        return lm.to(dtype).to_empty(device=device)
