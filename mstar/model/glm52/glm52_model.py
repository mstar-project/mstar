"""Glm52Model: Model implementation for GLM-5.2 (text generation).

GLM-5.2 (zai-org/GLM-5.2) is a 753B-total / ~40B-active MoE causal LM with
MLA + DSA sparse attention and 1M context. Unlike the composite models in
the zoo it is a single autoregressive loop, so the graph is the minimal
prefill -> decode shape and all of the integration substance lives in the
engine/submodule layer.

Architecture (1 node, default single partition):
    LLM (ar) - embed + 78 decoder layers (3 dense + 75 MoE, MLA/DSA
               attention) + lm_head, kept as one fat node: everything
               colocates on the same TP group, and splitting it would only
               add IPC overhead.

Scaffold status (bring-up order, per docs/adding_models.rst):
    [x] registry / config / graph walks / conductor state machine (this file)
    [ ] components/ + weight loading    <- next
    [ ] MLA attention + paged latent cache (sequencing vs the Kimi K2.7
        branch users/garv/kimik27-integration, which already carries
        weight-absorbed MLA + FlashInfer MLA + the engine changes)
    [ ] DSA indexer + IndexShare
    [ ] MTP speculation, CUDA graph configs, FP8
"""

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.base import EngineType
from mstar.engine.kv_cache_engine import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.sampling import SamplingConfig

logger = logging.getLogger(__name__)


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    except Exception as e:
        logger.warning("Error downloading from huggingface: %s", str(e))
        return repo_id
    return str(Path(local_dir))


class Glm52Model(Model):
    """GLM-5.2: 753B MoE causal LM (MLA + DSA), text in / text out."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf
        self.config = Glm52ModelConfig()

        # Tokenizer only — weights load lazily in get_submodule so the
        # conductor process never touches the 750 GB checkpoint.
        from transformers import AutoTokenizer

        tokenizer_source = _resolve_local_hf_snapshot(
            model_path_hf, cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, cache_dir=cache_dir,
        )

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -------------------------------------------------------------------
    # Model ABC: KV cache config
    # -------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        # MLA does not fit the num_kv_heads x head_dim layout: the cache
        # holds one latent vector per token per layer (kv_lora_rank + rope
        # dims = 576), shared by all 64 query heads. Declared here as a
        # single-"head" pool of width 576 — the layout vLLM/SGLang use for
        # DeepSeek-family models. The attention math over this pool needs
        # the paged-MLA engine path (see the sequencing note in the module
        # docstring); the geometry below is what that path expects.
        return [KVCacheConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=1,
            head_dim=self.config.cache_latent_dim,
            max_seq_len=self.config.max_seq_len,
            num_qo_heads=self.config.num_attention_heads,
        )]

    # -------------------------------------------------------------------
    # Model ABC: node engine types
    # -------------------------------------------------------------------

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {"LLM": EngineType.KV_CACHE}

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    conductor_new_token=True,
                    persist=True,
                ),
            ],
        )

        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="LLM",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node="LLM",
                        name="text_inputs",
                    ),
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        return dict(prefill=prefill, decode=decode)

    # -------------------------------------------------------------------
    # Model ABC: conductor state machine (prefill -> decode -> done)
    # -------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill",
            is_prefill=True,
        )

        graph_edge = GraphEdge(next_node="LLM", name="text_inputs")
        graph_edge.tensor_info = input_signals.get("text_inputs", [])
        inputs = [graph_edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections=None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False

        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            # The decode Loop ran to EOS (submodule check_stop) or to
            # max_iters; either way the request is finished.
            request_done = True
            metadata.kwargs["decode_finished"] = True

        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        graph_edge = GraphEdge(next_node="LLM", name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [graph_edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": metadata.is_prefill},
        )

    # -------------------------------------------------------------------
    # Model ABC: prompt processing
    # -------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if prompt is None:
            return {}

        # GLM-5.2 chat template (adds [gMASK]<sop> etc. and the assistant
        # turn). TODO: thinking mode / reasoning_effort dial once the
        # OpenAI adapter plumbs it through.
        if getattr(self.tokenizer, "chat_template", None):
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]

        return {"text_inputs": [input_ids.to(torch.long)]}

    def get_sampling_config(
        self, node_name: str,
        model_kwargs: dict | None = None,
    ) -> SamplingConfig | None:
        model_kwargs = model_kwargs or {}
        keys = ["temperature", "top_p", "repetition_penalty", "ignore_eos"]
        params = {
            k: model_kwargs.get(k, getattr(self.config, k))
            for k in keys
        }
        return SamplingConfig(
            vocab_size=self.config.vocab_size,
            **params,
        )

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get("max_output_tokens", self.config.max_output_tokens)

    # -------------------------------------------------------------------
    # Model ABC: postprocess
    # -------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        **kwargs,
    ) -> bytes:
        if modality == "text":
            token_ids = output.flatten().tolist()
            return self.tokenizer.decode(
                token_ids, skip_special_tokens=True,
            ).encode("utf-8")
        raise ValueError(f"Unsupported modality for GLM-5.2: {modality!r}")

    # -------------------------------------------------------------------
    # Model ABC: sharding
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

    # -------------------------------------------------------------------
    # Model ABC: submodule loading
    # -------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        if node_name != "LLM":
            return None
        # Components are the next bring-up step (see module docstring);
        # until they land the model runs in dummy mode only.
        raise NotImplementedError(
            "GLM-5.2 components are not implemented yet — run in dummy mode "
            "(engines without real computation) or wait for "
            "mstar/model/glm52/components/.",
        )
