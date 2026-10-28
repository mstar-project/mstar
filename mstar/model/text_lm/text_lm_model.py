"""Text in, text out: a causal LM from a Hugging Face checkpoint on one ``LLM`` node.

The checkpoint's ``model_type`` picks the language-model family (``config.LM_FAMILIES``):
its config, its module and its weight names are all that differ between models here. The
graph (a prefill, then a decode loop), the resources (paged KV, attention, RoPE positions,
sampler), the chat template, stop handling and prefix-cache keys are shared.
"""
from __future__ import annotations

import json
import logging

import torch
from jinja2 import TemplateError

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
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
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.text_lm.config import (
    ATTN,
    DECODE_LOOP,
    DECODE_WALK,
    KV_CACHE,
    LLM_NODE,
    LM_FAMILIES,
    PREFILL_WALK,
    ROPE,
    SAMPLER,
)
from mstar.utils.generation_kwargs import normalize_generation_kwargs
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

# sampling knobs a request may set, and the checkpoint's generation_config.json may default
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")


class TextLMModel(Model):
    """A text-only causal LM. ``max_seq_len`` bounds prompt plus reply (the KV cache's
    per-request bound); ``max_output_tokens`` is the reply budget a request gets when it
    names none."""

    # packed prefill captures, in prompt tokens per step; a deployment overrides them in
    # model_kwargs (an empty list leaves prefill eager)
    PREFILL_TOKEN_BUCKETS = (512, 1024, 2048, 4096, 8192)

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        checkpoint_path: str | None = None,
        max_seq_len: int = 32768,
        max_output_tokens: int = 4096,
        accelerator_graph: bool = True,
        prefill_token_buckets: list[int] | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.local_dir = resolve_snapshot_dir(checkpoint_path or model_path_hf, cache_dir=cache_dir)
        self.hf_config = hf = json.loads((self.local_dir / "config.json").read_text())
        model_type = hf.get("model_type")
        if model_type not in LM_FAMILIES:
            raise ValueError(f"no text LM family serves model_type {model_type!r}; known: {sorted(LM_FAMILIES)}")
        self.family = LM_FAMILIES[model_type]
        self.config = self.family.config(hf)
        if not 0 < max_seq_len <= self.config.max_position_embeddings:
            raise ValueError(
                f"max_seq_len {max_seq_len} must be in 1..{self.config.max_position_embeddings}"
            )
        self.max_seq_len = int(max_seq_len)
        self.default_max_output_tokens = int(max_output_tokens)
        self.accelerator_graph = bool(accelerator_graph)
        buckets = self.PREFILL_TOKEN_BUCKETS if prefill_token_buckets is None else prefill_token_buckets
        # a bucket past the context could never be replayed
        self.prefill_token_buckets = [int(n) for n in buckets if int(n) <= self.max_seq_len]
        generation = self.local_dir / "generation_config.json"
        self.generation = json.loads(generation.read_text()) if generation.is_file() else {}
        self._tokenizer = None
        self._detokenizer = None
        self._stop_token_ids: frozenset[int] | None = None
        self._submodule: NodeSubmodule | None = None

    @property
    def prebuild_fused_moe(self) -> bool:
        """The conductor builds the fused-MoE align op before the workers start, for an MoE."""
        return bool(getattr(self.config, "num_experts", 0))

    # -------------------------------------------------------------------
    # tokenizer
    # -------------------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
        return self._tokenizer

    def stop_token_ids(self) -> frozenset[int]:
        """The tokenizer's EOS and every EOS the configs name: a chat turn ends on the
        tokenizer's (``<|im_end|>``), which ``config.json`` may not list."""
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

    # -------------------------------------------------------------------
    # resources
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        config = self.config
        nodes = {LLM_NODE}
        kv = PagedKVConfig(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            num_qo_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            max_seq_len=self.max_seq_len,
        )
        return [
            KVSpec(resource_key=KV_CACHE, nodes=nodes, config=kv),
            AttentionSpec(resource_key=ATTN, nodes=nodes, config=AttentionConfig(kv_cache=KV_CACHE)),
            PositionSpec(
                resource_key=ROPE, nodes=nodes,
                config=PositionConfig(kv_cache=KV_CACHE, rope_theta=config.rope_theta),
            ),
            SamplerSpec(resource_key=SAMPLER, nodes=nodes, vocab_size=config.vocab_size),
        ]

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        # the reply's ids extend the keys: an agent's next turn resends them in its prompt
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
        params = {"ignore_eos": bool(kw.get("ignore_eos", False))}
        # the checkpoint's generation_config.json supplies what the request leaves out
        defaults = dict(self.generation)
        if defaults.get("do_sample") is False:
            defaults = {"temperature": 0.0}
        for key in _SAMPLING_KEYS:
            value = kw.get(key, defaults.get(key))
            if value is not None:
                params[key] = value
        return {SAMPLER: SamplingReqConfig(**params)}

    # -------------------------------------------------------------------
    # graph: prefill, then a decode loop
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name=LLM_NODE,
            input_names=["text_inputs"],
            outputs=[GraphEdge(
                next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text",
                conductor_new_token=True, persist=True,
            )],
        )
        decode = Loop(
            name=DECODE_LOOP,
            section=GraphNode(
                name=LLM_NODE,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=LLM_NODE, name="text_inputs"),
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
                ],
            ),
            # a runaway guard; the request's own budget is its loop cap
            max_iters=self.max_seq_len - 1,
            outputs=[],
        )
        return {PREFILL_WALK: prefill, DECODE_WALK: decode}

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
        edge.tensor_info = input_signals.get("text_inputs", [])
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(
                input_modalities=input_modalities, output_modalities=output_modalities,
                graph_walk=PREFILL_WALK, is_prefill=True,
            ),
            inputs=[edge],
            unpersist_tensors=list(edge.tensor_info),
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections=None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        if not metadata.is_prefill:
            # the decode loop stopped (a stop token or the request's budget)
            return ForwardPassArgs(full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True)
        metadata.is_prefill = False
        metadata.graph_walk = DECODE_WALK
        edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(full_metadata=metadata, inputs=[edge], unpersist_tensors=list(edge.tensor_info))

    def get_max_output_tokens(self, **model_kwargs) -> int:
        budget = normalize_generation_kwargs(model_kwargs).get("max_output_tokens")
        return self.default_max_output_tokens if budget is None else budget

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the prefill samples the first token, so the loop runs one fewer
        return {DECODE_LOOP: max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    # -------------------------------------------------------------------
    # prompt and reply
    # -------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        if output_modalities != ["text"] or set(input_modalities) - {TEXT}:
            raise ValueError("this model takes text and produces text")
        if tensors and "text_inputs" in tensors:
            (ids,) = tensors["text_inputs"]
        elif prompt is None and not prompt_parts:
            raise ValueError("the request carries no prompt")
        else:
            ids = self._render_chat(prompt, prompt_parts, kwargs)
        ids = ids.reshape(-1).to(torch.long)
        budget = self.get_max_output_tokens(**kwargs)
        # here, in the data worker, a ValueError reaches the client as a 400
        if ids.numel() == 0 or ids.numel() + budget > self.max_seq_len:
            raise ValueError(
                f"prompt ({ids.numel()} tokens) + max_output_tokens ({budget}) must fit "
                f"max_seq_len ({self.max_seq_len})"
            )
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("prompt token ids must be within the vocabulary")
        n = ids.numel()
        layout = LabelLayout(spans=(Span("ids", PREFILL_WALK, n, n, "text_inputs"),), growth=((DECODE_LOOP, 1),))
        return ProcessPromptOutput({"text_inputs": [ids]}, {"kv_layout": {KV_CACHE: {"main": layout}}})

    def _render_chat(self, prompt: str | None, prompt_parts: list[PromptPart] | None, kwargs: dict) -> torch.Tensor:
        """The chat's turns through the checkpoint's template, the assistant turn opened. A
        bare prompt is one user turn. ``enable_thinking`` reaches templates that switch on it."""
        parts = prompt_parts or [PromptPart(modality=TEXT, text=prompt, role="user")]
        messages = [
            {"role": turn.role, "content": "\n".join(p.text or "" for p in turn.parts if p.modality == TEXT)}
            for turn in chat_turns(parts)
        ]
        template_kwargs = {}
        if kwargs.get("enable_thinking") is not None:
            template_kwargs["enable_thinking"] = bool(kwargs["enable_thinking"])
        try:
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False, **template_kwargs,
            )
        except TemplateError as exc:
            raise ValueError(f"the chat template refused the chat: {exc}") from exc
        return torch.tensor(ids, dtype=torch.long)

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality != "text":
            raise ValueError(f"unsupported output modality {modality!r}")
        if self._detokenizer is None:
            from mstar.model.utils import ByteLevelDetokenizer

            # a token's raw bytes: a character split across tokens reaches the client whole
            self._detokenizer = ByteLevelDetokenizer(self.tokenizer)
        return self._detokenizer.to_bytes(output.reshape(-1).tolist())

    # -------------------------------------------------------------------
    # submodule
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={LLM_NODE}, shard_dim={}, ep_enabled_nodes={LLM_NODE})

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None, moe_parallel=None,
    ) -> NodeSubmodule | None:
        if node_name != LLM_NODE:
            raise ValueError(f"unknown node {node_name!r}")
        if sp_group is not None and sp_group.world_size != 1:
            raise ValueError("sequence parallelism is not implemented for text LMs")
        if self._submodule is None:
            self._submodule = self._create_submodule(device, tp_group, autocast_dtype, moe_parallel)
        return self._submodule

    def _create_submodule(self, device, tp_group, autocast_dtype, moe_parallel) -> NodeSubmodule | None:
        if not (self.local_dir / "model.safetensors.index.json").is_file() and not any(
            self.local_dir.glob("*.safetensors")
        ):
            logger.info("%s: no weights in %s, dummy mode", type(self).__name__, self.local_dir)
            return None
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.text_lm.submodules import TextLMSubmodule

        with torch.device("meta"):
            lm = self.family.build(self.config, comm_group=tp_group, moe_parallel=moe_parallel)
        if autocast_dtype is not None:
            lm = lm.to(autocast_dtype)
        lm.to_empty(device=device)
        weights = iter_safetensors_shards(self.local_dir, device=device)
        lm.load_weights(
            (name, tensor) for key, tensor in weights
            if (name := self.family.weight_name(key)) is not None
        )
        lm.eval()
        return TextLMSubmodule(
            lm, stop_token_ids=self.stop_token_ids(), accelerator_graph=self.accelerator_graph,
            prefill_token_buckets=self.prefill_token_buckets,
        )

