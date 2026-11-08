"""Dense Qwen3 (Qwen3-8B and its siblings): text in, text out on one ``LLM``
node, a packed prefill then a decode loop, on the shared Qwen3 decoder.
"""
from __future__ import annotations

import dataclasses
import json
import logging

import torch
from jinja2 import TemplateError

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import (
    ForwardPassArgs,
    KVLabel,
    LabelLayout,
    Model,
    PrefixKeys,
    ProcessPromptOutput,
    Span,
)
from mstar.model.multimodal import TEXT, PromptPart, chat_turns
from mstar.model.qwen3.config import (
    ATTN,
    DECODE_LOOP,
    DECODE_WALK,
    KV_CACHE,
    LLM_NODE,
    PREFILL_WALK,
    ROPE,
    SAMPLER,
    qwen3_lm_config,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import TextOutput
from mstar.utils.generation_kwargs import normalize_generation_kwargs
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

# sampling knobs the checkpoint's generation_config.json may default
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")
# every sampling knob a request may set
_REQUEST_KNOBS = tuple(f.name for f in dataclasses.fields(SamplingReqConfig) if not f.name.startswith("_"))


class Qwen3Model(Model):
    """``max_seq_len`` bounds prompt plus reply per request (default: the
    checkpoint's context); ``prefill_token_buckets`` replaces the captured
    prefill ladder.
    ``serve_min_tokens`` lets a request set ``min_tokens``: off by default,
    since barring the stop ids takes every sample step off FlashInfer's fused
    sampler."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        max_seq_len: int | None = None,
        accelerator_graph: bool = True,
        prefill_token_buckets: list[int] | None = None,
        serve_min_tokens: bool = False,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = resolve_snapshot_dir(
            model_path_hf, cache_dir=cache_dir, allow_patterns=["*.json", "*.txt", "*.model"],
        )
        self.hf_config = hf = json.loads((self.local_dir / "config.json").read_text())
        self.config = qwen3_lm_config(hf)
        limit = self.config.max_position_embeddings
        self.max_seq_len = limit if max_seq_len is None else int(max_seq_len)
        if not 0 < self.max_seq_len <= limit:
            raise ValueError(f"max_seq_len {self.max_seq_len} must be in 1..{limit}")
        self.accelerator_graph = bool(accelerator_graph)
        self.prefill_token_buckets = prefill_token_buckets
        generation = self.local_dir / "generation_config.json"
        self.generation = json.loads(generation.read_text()) if generation.is_file() else {}
        self.serve_min_tokens = bool(serve_min_tokens)
        self._tokenizer = None
        self._stop_token_ids: frozenset[int] | None = None
        self._submodule: NodeSubmodule | None = None

    # ── tokenizer ──

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
        return self._tokenizer

    @property
    def stop_token_ids(self) -> frozenset[int]:
        """Every EOS the configs name, and the tokenizer's: a chat turn ends on
        ``<|im_end|>``, which ``config.json`` may not list."""
        if self._stop_token_ids is None:
            sources = [self.generation.get("eos_token_id"), self.hf_config.get("eos_token_id")]
            if any((self.local_dir / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
                sources.append(self.tokenizer.eos_token_id)
            ids = set()
            for source in sources:
                if source is not None:
                    ids.update(source if isinstance(source, list) else [source])
            self._stop_token_ids = frozenset(int(i) for i in ids)
        return self._stop_token_ids

    def text_output(self) -> TextOutput:
        return TextOutput(self._chunk_detokenizer(), self.stop_token_ids)

    # ── resources ──

    def get_node_resources(self) -> list[NodeResourceSpec]:
        config, nodes = self.config, {LLM_NODE}
        specs = [
            KVSpec(
                resource_key=KV_CACHE, nodes=nodes,
                config=PagedKVConfig(
                    num_layers=config.num_hidden_layers,
                    num_kv_heads=config.num_key_value_heads, num_qo_heads=config.num_attention_heads,
                    head_dim=config.head_dim,
                    max_seq_len=self.max_seq_len,
                ),
            ),
            AttentionSpec(resource_key=ATTN, nodes=nodes, config=AttentionConfig(kv_cache=KV_CACHE)),
            PositionSpec(
                resource_key=ROPE, nodes=nodes, config=PositionConfig(kv_cache=KV_CACHE, rope_theta=config.rope_theta),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes=nodes, vocab_size=config.vocab_size,
                # opt-in: without stop ids a min_tokens request is a 400
                min_tokens_stop_ids=tuple(sorted(self.stop_token_ids)) if self.serve_min_tokens else (),
            ),
        ]
        return specs

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        # the reply's ids extend the keys (an agent's next turn resends them)
        return {KV_CACHE: {"main": KVLabel(
            growth=((DECODE_WALK, DECODE_LOOP),),
            keys=PrefixKeys("text_inputs", (PREFILL_WALK,), growth=True),
        )}}

    def checkpoint_path(self) -> str:
        return str(self.local_dir)

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs], model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        kw = normalize_generation_kwargs(model_kwargs)
        # every knob the request sets reaches the sampler, which refuses what this
        # deployment can't honour (a 400) rather than it being dropped here
        params = {key: kw[key] for key in _REQUEST_KNOBS if key in kw}
        # the checkpoint's generation_config.json supplies what the request leaves out
        defaults = {"temperature": 0.0} if self.generation.get("do_sample") is False else self.generation
        for key in _SAMPLING_KEYS:
            value = kw.get(key, defaults.get(key))
            if value is not None:
                params[key] = value
        return {SAMPLER: SamplingReqConfig(**params)}

    # ── graph: prefill, then a decode loop ──

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name=LLM_NODE, input_names=["text_inputs"],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True)],
        )
        decode = Loop(
            name=DECODE_LOOP,
            section=GraphNode(
                name=LLM_NODE, input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=LLM_NODE, name="text_inputs"),
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
                ],
            ),
            # a runaway guard; the request's budget is its loop cap
            max_iters=self.max_seq_len,
            outputs=[],
        )
        return {PREFILL_WALK: prefill, DECODE_WALK: decode}

    def get_initial_forward_pass_args(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
        edge.tensor_info = input_signals.get("text_inputs", [])
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(
                input_modalities=input_modalities, output_modalities=output_modalities,
                graph_walk=PREFILL_WALK, is_prefill=True,
            ),
            inputs=[edge], unpersist_tensors=list(edge.tensor_info),
        )

    def get_partition_forward_pass_args(
        self, partition_name: str, partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]], incoming_connections=None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        if not metadata.is_prefill:
            # the decode loop stopped: a stop id or the request's budget
            return ForwardPassArgs(full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True)
        metadata.is_prefill = False
        metadata.graph_walk = DECODE_WALK
        edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(full_metadata=metadata, inputs=[edge], unpersist_tensors=list(edge.tensor_info))

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the prefill samples the first token, so the loop runs one fewer
        return {DECODE_LOOP: max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    # ── prompt ──

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, prompt_parts: list[PromptPart] | None = None, **kwargs,
    ) -> ProcessPromptOutput:
        if output_modalities != ["text"] or set(input_modalities) - {TEXT}:
            raise ValueError("Qwen3 takes text and produces text")
        if tensors and tensors.get("text_inputs"):
            (ids,) = tensors["text_inputs"]
        elif prompt is None and not prompt_parts:
            raise ValueError("the request carries no prompt")
        else:
            ids = self._render_chat(prompt, prompt_parts, kwargs)
        ids = ids.reshape(-1).to(torch.long)
        n, budget = ids.numel(), self.get_max_output_tokens(**kwargs)
        # in the data worker, a ValueError reaches the client as a 400
        if n == 0 or n + budget > self.max_seq_len:
            raise ValueError(f"prompt ({n} tokens) + max_output_tokens ({budget}) must fit max_seq_len "
                             f"({self.max_seq_len})")
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("prompt token ids must be within the vocabulary")
        layout = LabelLayout(
            spans=(Span("ids", PREFILL_WALK, n, n, "text_inputs"),), growth=((DECODE_LOOP, 1),),
        )
        return ProcessPromptOutput({"text_inputs": [ids]}, {"kv_layout": {KV_CACHE: {"main": layout}}})

    def _render_chat(self, prompt: str | None, prompt_parts: list[PromptPart] | None, kwargs: dict) -> torch.Tensor:
        """The chat through the checkpoint's template, the assistant turn opened.
        ``enable_thinking`` (flat or in ``chat_template_kwargs``) reaches the template."""
        parts = prompt_parts or [PromptPart(modality=TEXT, text=prompt, role="user")]
        messages = [
            {"role": turn.role, "content": "\n".join(p.text or "" for p in turn.parts if p.modality == TEXT)}
            for turn in chat_turns(parts)
        ]
        # flat, or as OpenAI clients send it (chat_template_kwargs)
        thinking = kwargs.get("enable_thinking", (kwargs.get("chat_template_kwargs") or {}).get("enable_thinking"))
        extra = {} if thinking is None else {"enable_thinking": bool(thinking)}
        try:
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False, **extra,
            )
        except TemplateError as exc:
            raise ValueError(f"the chat template refused the chat: {exc}") from exc
        return torch.tensor(ids, dtype=torch.long)

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality != "text":
            raise ValueError(f"unsupported output modality {modality!r}")
        return super().postprocess(output, modality, request_kwargs)

    # ── submodule ──

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={LLM_NODE}, shard_dim={})

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None, autocast_dtype: torch.dtype | None = None, **kwargs,
    ) -> NodeSubmodule | None:
        if node_name != LLM_NODE:
            raise ValueError(f"unknown node {node_name!r}")
        if self._submodule is None:
            self._submodule = self._create_submodule(device, tp_group, autocast_dtype)
        return self._submodule

    def _create_submodule(self, device, tp_group, dtype) -> NodeSubmodule | None:
        weights = resolve_snapshot_dir(
            self.model_path_hf, cache_dir=self.cache_dir, allow_patterns=["*.json", "*.safetensors"],
        )
        if not any(weights.glob("*.safetensors")):
            logger.info("Qwen3: no weights in %s, dummy mode", weights)
            return None
        from mstar.model.loader.iterators import iter_safetensors_shards

        lm = self.build_lm(device, tp_group, dtype or torch.bfloat16)
        lm.load_weights(
            (key.removeprefix("model."), tensor) for key, tensor in iter_safetensors_shards(weights, device=device)
            if key.startswith(("model.", "lm_head."))
        )
        lm.requires_grad_(False).eval()
        return self.make_submodule(lm)

    def build_lm(self, device, tp_group=None, dtype: torch.dtype = torch.bfloat16):
        """The decoder on ``device``, weights unset."""
        from mstar.model.components.qwen3_lm import Qwen3DenseLM

        with torch.device("meta"):
            lm = Qwen3DenseLM(self.config, attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE, comm_group=tp_group)
        return lm.to(dtype).to_empty(device=device)

    def make_submodule(self, lm) -> NodeSubmodule:
        """The node around a loaded decoder."""
        from mstar.model.qwen3.submodules import Qwen3LMSubmodule

        return Qwen3LMSubmodule(
            lm, self.stop_token_ids, accelerator_graph=self.accelerator_graph,
            prefill_token_buckets=self.prefill_token_buckets,
        )
