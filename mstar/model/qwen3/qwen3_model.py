"""Dense Qwen3 (Qwen3-8B and its siblings): text in, text out on one ``LLM``
node of the shared decoder scaffold, on the shared Qwen3 decoder.

``model_kwargs.draft`` turns on speculative decoding with a DSpark, EAGLE-3 or
draft-model drafter (`mstar.model.components`): the drafter's layers join the
target's KV pool, and every decode step drafts and verifies ``num_draft_tokens``
tokens.
Without the block the model is plain Qwen3.
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import torch
from jinja2 import TemplateError

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.draft.config import DraftConfig
from mstar.engine.resources.draft.drafter import ModelGeometry, drafter_resources
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.model.hf_config import hf_weight_dtype
from mstar.model.keys import KV_CACHE, ROPE
from mstar.model.multimodal import TEXT, PromptPart, chat_turns
from mstar.model.multimodal_decoder import MultimodalDecoderModel
from mstar.model.qwen3.config import ATTN, LLM_NODE, qwen3_lm_config
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.fast_tokenize import encode_ids
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)


class Qwen3Model(MultimodalDecoderModel):
    NAME = "Qwen3"
    # the tokenizer prefers chat_template.jinja to tokenizer_config.json's template,
    # and some repos keep theirs nowhere else
    METADATA_FILES = (
        "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
    )

    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None, draft: dict | None = None,
        accelerator_graph: bool = True,
    ):
        """``draft``: the speculative-decoding block (``method``: ``dspark``,
        ``eagle3`` or ``draft_model``).
        ``accelerator_graph: false`` runs the LLM node uncaptured (no resident
        planning, so a draft block verifies synchronously)."""
        super().__init__(model_path_hf, cache_dir)
        self.accelerator_graph = bool(accelerator_graph)
        self.hf_config = hf = json.loads((Path(self.local_dir) / "config.json").read_text())
        self.config = qwen3_lm_config(hf)
        self.config.stop_token_ids = self._stop_token_ids(hf)
        self.draft = DraftConfig.from_model_kwargs({"draft": draft})
        self.drafter_decl = None if self.draft is None else self._drafter(self.draft)

    def _drafter(self, draft: DraftConfig):
        """The drafter's declaration, from its checkpoint's config alone."""
        from mstar.model.components.dspark import DSpark, DSparkConfig

        if draft.method not in ("dspark", "draft_model", "eagle3"):
            raise ValueError(f"Qwen3 drafts with method dspark, eagle3 or draft_model, not {draft.method!r}")
        directory = self._drafter_dir = resolve_snapshot_dir(
            draft.model, cache_dir=self.cache_dir, allow_patterns=["*.json"],
        )
        if draft.method == "draft_model":
            return self._draft_model(draft, directory)
        if draft.method == "eagle3":
            return self._eagle3(draft, directory)
        dspark = DSparkConfig.from_hf(json.loads((directory / "config.json").read_text()))
        if max(dspark.taps) >= self.config.num_hidden_layers:
            # its checkpoint names the last layer's output, HF's hidden_states[L], which is after
            # the final norm; the tap hook hands over the stream before it
            raise ValueError(
                f"the DSpark drafter reads target layer {max(dspark.taps) - 1}'s output, the last layer's, "
                "which it was trained on after the final norm; the target's taps read the stream before it"
            )
        # its query heads and RoPE base are checked at startup (`geometry_problems`)
        return DSpark(
            dspark, draft.num_draft_tokens, kv_key=KV_CACHE, pos_key=ROPE, checkpoint=draft.model,
            cache_dir=self.cache_dir,
        )

    @functools.cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.local_dir)

    def _draft_model(self, draft: DraftConfig, directory):
        """A separate draft model of the target's family and tokenizer (`DraftModel`)."""
        from mstar.model.components.draft_model import DraftModel, draft_model_refusals

        hf = json.loads((directory / "config.json").read_text())
        problems = draft_model_refusals(hf, self.config.vocab_size)
        mine, theirs = directory / "tokenizer.json", Path(self.local_dir) / "tokenizer.json"
        if mine.is_file() and theirs.is_file() and \
                json.loads(mine.read_text()).get("model") != json.loads(theirs.read_text()).get("model"):
            problems.append("a draft model with another tokenizer is not supported")
        if problems:
            raise ValueError(f"draft model {draft.model!r}: " + "; ".join(problems))
        # its query heads, KV geometry and RoPE base are checked at startup (`geometry_problems`)
        return DraftModel(
            qwen3_lm_config(hf), draft.num_draft_tokens, dtype=hf_weight_dtype(hf), kv_key=KV_CACHE, pos_key=ROPE,
            checkpoint=draft.model, cache_dir=self.cache_dir,
        )

    def _eagle3(self, draft: DraftConfig, directory):
        """EAGLE-3 over this target's taps (`Eagle3`)."""
        from mstar.model.components.eagle3 import Eagle3, Eagle3Config, default_taps

        hf = json.loads((directory / "config.json").read_text())
        try:
            config = Eagle3Config.from_hf(hf)
        except ValueError as exc:
            raise ValueError(f"draft model {draft.model!r}: {exc}") from None
        taps = config.taps or default_taps(self.config.num_hidden_layers)
        # the weights' dtype from one level, the top's if it names one: merged, a top-level torch_dtype and an inner
        # dtype that disagree would be refused though each level agrees with itself
        level = hf if "dtype" in hf or "torch_dtype" in hf else hf.get("transformer_layer_config", {})
        # its query heads, KV geometry, RoPE base and the width it reads are checked at startup (`geometry_problems`)
        return Eagle3(
            config, draft.num_draft_tokens, dtype=hf_weight_dtype(level),
            taps=taps, kv_key=KV_CACHE, pos_key=ROPE, attn_key=ATTN, checkpoint=draft.model, cache_dir=self.cache_dir,
        )

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
        specs = [
            KVSpec(
                resource_key=KV_CACHE, nodes=nodes,
                config=PagedKVConfig(
                    # a drafter's layers join after the target's, and the verify slack past the
                    # context, by `drafter_resources`
                    num_layers=cfg.num_hidden_layers,
                    num_kv_heads=cfg.num_key_value_heads, num_qo_heads=cfg.num_attention_heads,
                    head_dim=cfg.head_dim, max_seq_len=cfg.max_position_embeddings,
                ),
            ),
            AttentionSpec(
                resource_key=ATTN, nodes=nodes, config=AttentionConfig(kv_cache=KV_CACHE, splits=self.mixed_splits()),
            ),
            PositionSpec(
                resource_key=ROPE, nodes=nodes, config=PositionConfig(kv_cache=KV_CACHE, rope_theta=cfg.rope_theta),
            ),
            self._sampler_spec(cfg.vocab_size),
        ]
        if self.drafter_decl is not None:
            specs = drafter_resources(specs, self.drafter_decl, nodes, self.draft)
        return specs

    def llm_node_class(self) -> type:
        from mstar.model.qwen3.submodules import Qwen3LMSubmodule

        return Qwen3LMSubmodule

    def verify_node_fields(self) -> dict:
        cfg = self.config
        return dict(
            input_kinds=frozenset({TEXT}),
            target=ModelGeometry(
                num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim, hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size,
                dtype=hf_weight_dtype(self.hf_config),
            ),
            checkpoint=str(self._drafter_dir),
            prefill_steps=self.drafter_decl.prefill_steps(), verify_steps=self.drafter_decl.verify_steps(),
        )

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

        lm = self.build_lm(device, tp_group, dtype)
        lm.load_weights(
            (key.removeprefix("model."), tensor) for key, tensor in iter_safetensors_shards(weights, device=device)
            if key.startswith(("model.", "lm_head."))
        )
        lm.requires_grad_(False).eval()
        return self.make_submodule(lm, device, tp_group, dtype)

    def build_lm(self, device, tp_group=None, dtype: torch.dtype = torch.bfloat16):
        """The target's module on ``device``, weights unset."""
        from mstar.model.components.qwen3_lm import Qwen3DenseLM

        with torch.device("meta"):
            lm = Qwen3DenseLM(self.config, attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE, comm_group=tp_group)
        return lm.to(dtype).to_empty(device=device)

    def make_submodule(self, lm, device, tp_group=None, dtype: torch.dtype = torch.bfloat16) -> NodeSubmodule:
        """The node around a loaded target, with the drafter the deployment names."""
        from mstar.model.qwen3.submodules import Qwen3LMSubmodule

        drafter = None
        if self.drafter_decl is not None:
            drafter = self.drafter_decl.build(device=device, comm_group=tp_group, dtype=dtype, target=lm)
        return Qwen3LMSubmodule(lm, self.config, drafter=drafter, accelerator_graph=self.accelerator_graph)
