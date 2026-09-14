"""M* model wrapper for the Kimi-K2.7 text backbone."""
from __future__ import annotations

import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    KVConfig,
    KVLayout,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.kimi_k2_7.config import ATTN, KV_CACHE, ROPE, SAMPLER, KimiK2Config
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)

LLM_NODE = "LLM"
DECODE_LOOP = "decode_loop"


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    from pathlib import Path

    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id, cache_dir=cache_dir, local_files_only=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Error downloading %r from huggingface: %s", repo_id, e)
        return repo_id
    return str(Path(local_dir))


class KimiK2Model(Model):
    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        # A local checkpoint directory or an HF repo id. Deployments point this at
        # their own copy with ``mstar serve --model-path`` / ``mstar-serve
        # --model-path`` rather than hardcoding a path in the config yaml.
        self.model_path_hf = model_path_hf
        self._config_variant = kwargs.get("config_variant", "full")
        if self._config_variant == "reduced":
            self.config = KimiK2Config.reduced()
        elif self._config_variant == "reduced_quantized":
            self.config = KimiK2Config.reduced_quantized()
        elif self._config_variant == "reduced_quantized_inkernel":
            self.config = KimiK2Config.reduced_quantized_inkernel()
        elif self._config_variant == "k27_code":
            self.config = KimiK2Config.k27_code()
        else:
            self.config = KimiK2Config()
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
        self._tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path_hf,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _kv_and_attn_specs(self) -> list[NodeResourceSpec]:
        """The cache's shape and the backend over it, which Kimi picks
        together: absorbed MLA caches one latent per token, naive full K/V."""
        cfg = self.config
        if not cfg.mla_absorb:
            return [
                KVConfig(
                    num_layers=cfg.num_hidden_layers,
                    num_kv_heads=cfg.num_attention_heads,
                    head_dim=cfg.padded_head_dim,
                    max_seq_len=cfg.max_position_embeddings,
                    num_qo_heads=cfg.num_attention_heads,
                ),
                AttentionConfig(kv_cache=KV_CACHE),
            ]

        from mstar.model.kimi_k2_7.components.rope import yarn_get_mscale

        rope = cfg.rope_scaling
        mscale = yarn_get_mscale(rope["factor"], rope.get("mscale_all_dim", 0.0))
        return [
            KVConfig(
                num_layers=cfg.num_hidden_layers,
                num_kv_heads=1,
                head_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
                max_seq_len=cfg.max_position_embeddings,
                num_qo_heads=cfg.num_attention_heads,
                layout=KVLayout.MLA,
            ),
            AttentionConfig(
                kv_cache=KV_CACHE,
                backend=AttnBackend.MLA,
                # MLA scales by the unabsorbed qk_head_dim, which the latent
                # width does not carry
                softmax_scale=cfg.qk_head_dim ** -0.5 * mscale * mscale,
                mla_ckv_dim=cfg.kv_lora_rank,
            ),
        ]

    def get_node_resources(self) -> list[NodeResourceSpec]:
        kv_config, attn_config = self._kv_and_attn_specs()
        return [
            KVSpec(resource_key=KV_CACHE, nodes={LLM_NODE}, config=kv_config),
            AttentionSpec(resource_key=ATTN, nodes={LLM_NODE}, config=attn_config),
            SamplerSpec(
                resource_key=SAMPLER,
                nodes={LLM_NODE},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=True,
            ),
            # Kimi applies its own DeepSeek-yarn rotary in the layer, so this
            # only tracks the position counters; the rope params here are
            # unused (see `KimiMLAAttention.position_ids`).
            PositionSpec(
                resource_key=ROPE,
                nodes={LLM_NODE},
                config=PositionConfig(
                    kv_cache=KV_CACHE,
                    rotary_dim=self.config.qk_rope_head_dim,
                    rope_theta=self.config.rope_theta,
                ),
            ),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        return {
            SAMPLER: SamplingReqConfig(
                **{
                    key: model_kwargs.get(key, getattr(self.config, key))
                    for key in (
                        "temperature", "top_p", "repetition_penalty", "ignore_eos",
                    )
                }
            )
        }

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name=LLM_NODE,
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
            name=DECODE_LOOP,
            section=GraphNode(
                name=LLM_NODE,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        conductor_new_token=True,
                    ),
                    GraphEdge(
                        next_node=LLM_NODE,
                        name="text_inputs",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        return dict(prefill=prefill, decode=decode)

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

        graph_edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
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
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False

        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            request_done = True
            metadata.kwargs["decode_finished"] = True

        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        graph_edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [graph_edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": metadata.is_prefill},
        )

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
        if self._tokenizer_mode == "byte":
            # Reduced serve maps UTF-8 bytes directly to token ids, avoiding HF IO.
            vocab = self.config.vocab_size
            byte_ids = [min(b, vocab - 1) for b in prompt.encode("utf-8")] or [0]
            input_ids = torch.tensor(byte_ids, dtype=torch.long)
            return {"text_inputs": [input_ids]}
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        return {"text_inputs": [input_ids]}

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        request_kwargs: dict | None = None,
    ) -> bytes:
        if modality == "text":
            token_ids = output.tolist() if output.numel() else []
            if self._tokenizer_mode == "byte":
                # Synthetic reduced models emit arbitrary byte ids; return raw bytes.
                return bytes((t & 0xFF) for t in token_ids)
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            return text.encode("utf-8")
        raise ValueError(f"Unsupported modality for Kimi-K2.7: {modality!r}")

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={LLM_NODE}, shard_dim={})

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype,
        )
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self,
        node_name: str,
        device: str,
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name != LLM_NODE:
            return None

        source = self._resolve_checkpoint()
        if source is None:
            logger.info(
                "KimiK2Model: no checkpoint resolved for node %r — dummy mode (None).",
                node_name,
            )
            return None

        self._maybe_apply_checkpoint_quant_config(source)

        from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
        from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule
        from mstar.model.loader import load_weights

        with torch.device("meta"):
            language_model = KimiForCausalLM(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            language_model = language_model.to(autocast_dtype)
        language_model.to_empty(device=device)
        load_weights(language_model, source, device=device)
        from mstar.model.components.quantization import process_weights_after_loading

        process_weights_after_loading(language_model, torch.device(device))
        language_model.eval()

        logger.info("Successfully loaded Kimi-K2.7 submodule for %s", node_name)
        return KimiLLMSubmodule(language_model=language_model, config=self.config)

    def _resolve_checkpoint(self) -> str | None:
        from pathlib import Path

        path = getattr(self, "model_path_hf", None)
        if not path:
            return None
        if Path(path).exists():
            return str(path)
        return _resolve_local_hf_snapshot(path, cache_dir=getattr(self, "cache_dir", None))

    def _maybe_apply_checkpoint_quant_config(self, source: str) -> None:
        import json
        from pathlib import Path

        from mstar.model.components.quantization import CompressedTensorsQuantConfig

        if self.config.quantization_config is not None:
            return
        config_json = Path(source) / "config.json"
        if not config_json.is_file():
            return
        try:
            with open(config_json) as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:  # unreadable / malformed — stay bf16
            logger.warning("KimiK2Model: could not read %s: %s", config_json, e)
            return
        quant_raw = raw.get("quantization_config") or (
            raw.get("text_config") or {}
        ).get("quantization_config")
        quant = CompressedTensorsQuantConfig.from_hf_config_dict(quant_raw)
        if quant is not None:
            logger.info(
                "KimiK2Model: compressed-tensors checkpoint (%d-bit, group_size=%d) "
                "— dequantizing on load.", quant.num_bits, quant.group_size,
            )
            self.config.quantization_config = quant
