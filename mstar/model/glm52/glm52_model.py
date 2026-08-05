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
    [x] components/ + weight loading (fp8-block dequant; experts fp8-resident)
    [x] MLA attention (absorbed default + naive fallback) on the paged latent
        cache path from users/garv/kimik27-integration
    [ ] DSA indexer + IndexShare (Phase C — at ctx <= 2048 dense MLA IS the
        exact DSA computation, so M1 correctness needs no indexer)
    [ ] MTP speculation (Phase D; layer-78 weights consciously skipped)
    [ ] fused fp8 expert kernel (M4 perf debt; reference dispatch until then)
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
        checkpoint_path = kwargs.get("checkpoint_path")
        self.model_path_hf = checkpoint_path or model_path_hf
        self._config_variant = kwargs.get("config_variant", "full")
        if self._config_variant == "reduced":
            self.config = Glm52ModelConfig.reduced()
        elif self._config_variant == "reduced_fp8":
            self.config = Glm52ModelConfig.reduced_fp8()
        else:
            self.config = Glm52ModelConfig()
        # "byte" maps UTF-8 bytes to token ids for reduced serve (no HF IO).
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
        self._tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @property
    def tokenizer(self):
        # Lazy: weights AND tokenizer load on demand so the conductor process
        # never touches the 750 GB checkpoint or HF IO in dummy/byte mode.
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer_source = _resolve_local_hf_snapshot(
                self.model_path_hf, cache_dir=self.cache_dir,
            )
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source, cache_dir=self.cache_dir,
                )
            except ValueError:
                # The checkpoint's tokenizer_config declares transformers-5's
                # TokenizersBackend class, which transformers 4.x cannot
                # construct — but the underlying tokenizer.json is
                # version-independent. Verified on transformers 4.57:
                # template render, roundtrip, and special-token decode all
                # match the checkpoint's declared ids.
                self._tokenizer = self._fast_tokenizer_fallback(tokenizer_source)
        return self._tokenizer

    @staticmethod
    def _fast_tokenizer_fallback(source: str):
        from transformers import PreTrainedTokenizerFast

        snap = Path(source)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(snap / "tokenizer.json"))
        template = snap / "chat_template.jinja"
        if template.is_file():
            tokenizer.chat_template = template.read_text()
        return tokenizer

    # -------------------------------------------------------------------
    # Model ABC: KV cache config
    # -------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        # MLA does not fit the num_kv_heads x head_dim layout: the absorbed
        # cache holds one latent vector per token per layer (kv_lora_rank +
        # rope dims = 576), shared by all 64 query heads — the layout
        # vLLM/SGLang use for DeepSeek-family models, served by the
        # mla_absorb engine path from users/garv/kimik27-integration.
        # No Yarn -> softmax scale is plain qk_head_dim**-0.5.
        if self.config.mla_absorb:
            return [KVCacheConfig(
                num_layers=self.config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=self.config.cache_latent_dim,
                max_seq_len=self.config.max_seq_len,
                num_qo_heads=self.config.num_attention_heads,
                attention_backend="mla_absorb",
                softmax_scale=self.config.qk_head_dim ** -0.5,
                mla_ckv_dim=self.config.kv_lora_rank,
            )]
        # Naive fallback (reduced-test parity): full K/V padded to the
        # FlashInfer head size, one KV head per query head.
        return [KVCacheConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_attention_heads,
            head_dim=self.config.padded_head_dim,
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

        if self._tokenizer_mode == "byte":
            # Reduced serve maps UTF-8 bytes directly to token ids, avoiding HF IO.
            vocab = self.config.vocab_size
            byte_ids = [min(b, vocab - 1) for b in prompt.encode("utf-8")] or [0]
            return {"text_inputs": [torch.tensor(byte_ids, dtype=torch.long)]}

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
            if self._tokenizer_mode == "byte":
                # Synthetic reduced models emit arbitrary byte ids; return raw
                # bytes without ever touching the HF tokenizer.
                return bytes((t & 0xFF) for t in token_ids)
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
        if node_name != "LLM":
            return None

        source = self._resolve_checkpoint()
        if source is None:
            logger.info(
                "Glm52Model: no checkpoint resolved for node %r — dummy mode (None).",
                node_name,
            )
            return None

        self._maybe_apply_checkpoint_quant_config(source)

        from mstar.model.components.quantization import (
            process_weights_after_loading,
        )
        from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
        from mstar.model.glm52.submodules import Glm52LLMSubmodule
        from mstar.model.loader import load_weights

        with torch.device("meta"):
            language_model = Glm52ForCausalLM(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            language_model = language_model.to(autocast_dtype)
        language_model.to_empty(device=device)
        load_weights(language_model, source, device=device)
        process_weights_after_loading(language_model, torch.device(device))
        language_model.eval()

        logger.info("Successfully loaded GLM-5.2 submodule for %s", node_name)
        return Glm52LLMSubmodule(language_model=language_model, config=self.config)

    def _resolve_checkpoint(self) -> str | None:
        path = getattr(self, "model_path_hf", None)
        if not path:
            return None
        if Path(path).exists():
            return str(path)
        return _resolve_local_hf_snapshot(path, cache_dir=self.cache_dir)

    def _maybe_apply_checkpoint_quant_config(self, source: str) -> None:
        import json

        from mstar.model.glm52.quantization import Fp8BlockQuantConfig

        if self.config.quantization_config is not None:
            return
        config_json = Path(source) / "config.json"
        if not config_json.is_file():
            return
        try:
            with open(config_json) as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:  # unreadable / malformed — stay bf16
            logger.warning("Glm52Model: could not read %s: %s", config_json, e)
            return
        quant = Fp8BlockQuantConfig.from_hf_config_dict(
            raw.get("quantization_config"),
        )
        if quant is not None:
            logger.info(
                "Glm52Model: fp8 %s checkpoint (block %s) — dense dequant on "
                "load, routed experts fp8-resident=%s.",
                quant.fmt, quant.weight_block_size, self.config.moe_fp8_resident,
            )
            self.config.quantization_config = quant
