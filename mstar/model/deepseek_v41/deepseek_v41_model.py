"""DeepSeekV41Model: DeepSeek-V4.1-Flash causal text LM on ``mstar.engine.state``."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from jinja2 import TemplateError

from mstar.conductor.request_info import FINAL_PREFILL
from mstar.distributed.base import ShardingConfig
from mstar.engine.state import Array, Entry, Sampler, TensorType, Value
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import KVLabel, Model, ProcessPromptOutput, WalkStep
from mstar.model.components.deepseek_v41.backbone import DeepSeekV41TextBackbone
from mstar.model.components.deepseek_v41.compressor_source import compressor_partial_value
from mstar.model.components.deepseek_v41.engram_hash import build_compressed_token_map
from mstar.model.components.request_kwargs import (
    SAMPLING_KEYS,
    generation_config_dir,
    sample_settings,
    sampling_config,
    sampling_defaults,
)
from mstar.model.deepseek_v41.config import (
    COMPRESS_STATE,
    COMPRESSOR_PARTIAL,
    NGRAM_STATE,
    REQUEST_META,
    SAMPLER,
    WINDOW_STATE,
    DeepSeekV41Config,
)
from mstar.model.deepseek_v41.submodules import DeepSeekV41LLMSubmodule
from mstar.model.deepseek_v41.weight_loader import load_deepseek_v41_weights
from mstar.model.loader import iter_safetensors_shards
from mstar.model.multimodal import TEXT, PromptPart, chat_turns
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import ByteLevelTextOutput, load_snapshot_tokenizer, resolve_checkpoint

logger = logging.getLogger(__name__)

DEFAULT_SAMPLING: Mapping[str, object] = MappingProxyType({"temperature": 1.0, "top_p": 1.0})


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class DeepSeekV41Model(ByteLevelTextOutput, Model):
    """DeepSeek-V4.1-Flash text-only causal LM on ``mstar.engine.state``."""

    DECODE_LOOP = "decode_loop"
    PREPROCESS_ONLY_KWARGS = ("messages", "tools", "chat_template_kwargs")
    sampling_defaults: Mapping[str, object] = DEFAULT_SAMPLING

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs: Any) -> None:
        self.cache_dir = cache_dir
        self.model_path_hf = str(kwargs.get("checkpoint_path") or model_path_hf)
        self._config_variant = kwargs.get("config_variant", "full")
        if self._config_variant == "reduced":
            self.config = DeepSeekV41Config.reduced()
        else:
            local_cfg = Path(self.model_path_hf) / "config.json" if self.model_path_hf else None
            if local_cfg is not None and local_cfg.is_file():
                self.config = DeepSeekV41Config.from_hf(json.loads(local_cfg.read_text()))
            else:
                self.config = DeepSeekV41Config()
        if "max_seq_len" in kwargs:
            self.config = dataclasses.replace(
                self.config,
                max_seq_len=_positive_int("max_seq_len", kwargs["max_seq_len"]),
            )
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
        self._tokenizer = None
        self._token_map = kwargs.get("token_map")
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self.sampling_defaults = sampling_defaults(
            generation_config_dir(self.model_path_hf, cache_dir),
            DEFAULT_SAMPLING,
        )

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_snapshot_tokenizer(self.model_path_hf, self.cache_dir)
        return self._tokenizer

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"messages", "tools", "chat_template_kwargs", "max_output_tokens", *SAMPLING_KEYS})

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    persist=True,
                    output_modality="text",
                    conductor_new_token=True,
                ),
            ],
        )
        decode = Loop(
            name=self.DECODE_LOOP,
            max_iters=self.config.max_seq_len,
            outputs=[],
            section=GraphNode(
                name="LLM",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        conductor_new_token=True,
                    ),
                    GraphEdge(next_node="LLM", name="text_inputs"),
                ],
            ),
        )
        return {"prefill": prefill, "decode": decode}

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        del partition_name, input_modalities, output_modalities, model_kwargs
        if not input_signals.get("text_inputs"):
            raise ValueError("DeepSeek-V4.1 prefill requires text_inputs")
        return [
            WalkStep("prefill", step_metadata={FINAL_PREFILL: True}),
            WalkStep("decode", bind={"text_inputs": "new_token"}),
        ]

    def declare_state(self) -> list[Array | Sampler]:
        c = self.config
        state: list[Array | Sampler] = [
            Value(
                WINDOW_STATE,
                nodes={"LLM"},
                layers=frozenset(range(c.num_hidden_layers)),
                entry=Entry({"kv": TensorType((c.sliding_window, c.head_dim), dtype=None)}),
            ),
            Value(
                REQUEST_META,
                nodes={"LLM"},
                layers=frozenset({0}),
                entry=Entry({}, host={"pos": int}),
            ),
            Sampler(
                SAMPLER,
                nodes={"LLM"},
                vocab_size=c.vocab_size,
                penalty=True,
                min_p=bool(self.sampling_defaults.get("min_p")),
            ),
        ]
        if c.kv_source_layer_ids:
            state.append(
                Value(
                    COMPRESS_STATE,
                    nodes={"LLM"},
                    layers=frozenset(c.kv_source_layer_ids),
                    entry=Entry(
                        {
                            "compress_kv": TensorType((c.max_seq_len, c.head_dim), dtype=None),
                            "index_k": TensorType((c.max_seq_len, c.index_head_dim), dtype=None),
                        }
                    ),
                )
            )
        if c.pooled_source_layer_ids:
            max_ratio = max(c.compress_ratios[i] for i in c.pooled_source_layer_ids)
            state.append(
                compressor_partial_value(
                    COMPRESSOR_PARTIAL,
                    node="LLM",
                    source_layers=frozenset(c.pooled_source_layer_ids),
                    ratio=max_ratio,
                    head_dim=c.head_dim,
                )
            )
        if c.engram_layer_ids:
            tail_len = max(1, c.engram_max_ngram_size - 1)
            state.append(
                Value(
                    NGRAM_STATE,
                    nodes={"LLM"},
                    layers=frozenset({0}),
                    entry=Entry({"tail": TensorType((tail_len,), dtype=torch.long)}),
                )
            )
        return state

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {}

    def get_request_resource_configs(
        self,
        input_modalities: list[str],
        output_modalities: list[str],
        model_kwargs: dict,
    ) -> dict[str, Any]:
        del input_modalities, output_modalities
        return {SAMPLER: sample_settings(sampling_config(self.sampling_defaults, model_kwargs))}

    def get_default_sharding_config(self) -> ShardingConfig:
        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

    def get_loop_caps(self, max_output_tokens: int | None = None) -> dict[str, int]:
        budget = self.config.max_seq_len if max_output_tokens is None else max_output_tokens
        return {self.DECODE_LOOP: max(0, min(int(budget), self.config.max_seq_len) - 1)}

    def _resolve_token_map(self) -> torch.Tensor | None:
        if self._token_map is not None:
            return torch.as_tensor(self._token_map, dtype=torch.long)
        if self._tokenizer_mode == "byte" or not self.config.engram_layer_ids:
            return None
        try:
            tok = self.tokenizer
            return torch.from_numpy(
                build_compressed_token_map(tok, self.config.engram_compressed_vocab_size)
            )
        except Exception:  # noqa: BLE001
            return None

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group: Any = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name != "LLM":
            raise ValueError(f"Unknown DeepSeek-V4.1 node: {node_name!r}")
        if node_name not in self._submodule_cache:
            dtype = autocast_dtype if autocast_dtype is not None else torch.bfloat16
            token_map = self._resolve_token_map()
            checkpoint_dir = resolve_checkpoint(self.model_path_hf, self.cache_dir)
            with torch.device("meta"):
                backbone = DeepSeekV41TextBackbone(
                    self.config,
                    token_map=token_map,
                    comm_group=tp_group,
                ).to(dtype=dtype)
            backbone.to_empty(device=device)
            load_deepseek_v41_weights(
                backbone,
                iter_safetensors_shards(checkpoint_dir, device="cpu"),
                device=device,
            )
            backbone.eval()
            self._submodule_cache[node_name] = DeepSeekV41LLMSubmodule(backbone, self.config).eval()
        return self._submodule_cache[node_name]

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: dict[str, list[torch.Tensor]] | None = None,
        prompt_parts: Any = None,
        **kwargs: Any,
    ) -> ProcessPromptOutput:
        if input_modalities != ["text"] or output_modalities != ["text"]:
            raise ValueError("DeepSeek-V4.1 Phase 1 supports text input and text output only")
        tensors = tensors or {}
        if "text_inputs" in tensors:
            raw = tensors["text_inputs"]
            if len(raw) != 1:
                raise ValueError("Expected one prompt token tensor")
            ids = raw[0].reshape(-1).to(torch.long)
        elif self._tokenizer_mode == "byte":
            if prompt_parts is not None:
                prompt = "".join(part.text or "" for part in prompt_parts)
            if not isinstance(prompt, str):
                raise ValueError("Expected a string prompt")
            raw_bytes = prompt.encode("utf-8")
            ids = torch.tensor(list(raw_bytes) or [self.config.bos_token_id], dtype=torch.long)
        else:
            messages = kwargs.get("messages")
            if messages is None:
                parts = prompt_parts or [PromptPart(modality=TEXT, text=prompt or "", role="user")]
                if prompt_parts is None and not isinstance(prompt, str):
                    raise ValueError("Expected a string prompt or messages")
                messages = [
                    {
                        "role": turn.role,
                        "content": "\n".join(p.text or "" for p in turn.parts if p.modality == TEXT),
                    }
                    for turn in chat_turns(parts)
                ]
            template_kw = dict(kwargs.get("chat_template_kwargs") or {})
            if kwargs.get("tools"):
                template_kw["tools"] = kwargs["tools"]
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False,
                    **template_kw,
                )
            except TemplateError as exc:
                raise ValueError(f"DeepSeek-V4.1 chat template refused the chat: {exc}") from exc
            ids = encoded[0].to(torch.long)

        if ids.numel() == 0 or ids.numel() > self.config.max_seq_len:
            raise ValueError(f"Prompt must contain 1..{self.config.max_seq_len} token IDs, got {ids.numel()}")
        max_output_tokens = self.get_max_output_tokens(**kwargs)
        if ids.numel() + max_output_tokens > self.config.max_seq_len:
            raise ValueError(
                f"prompt ({ids.numel()} tokens) + max_output_tokens ({max_output_tokens}) "
                f"exceeds max_seq_len ({self.config.max_seq_len})"
            )
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("Prompt token IDs must be within [0, vocab_size)")
        return ProcessPromptOutput({"text_inputs": [ids]}, {"state_layout": {}})
