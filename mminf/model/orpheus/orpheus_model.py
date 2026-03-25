"""
OrpheusModel: Model implementation for Orpheus TTS.

Orpheus consists of a Llama 3.2 3B LLM that generates custom audio tokens
and a SNAC decoder that converts those tokens to 24kHz PCM audio. The LLM
generates 7 tokens per audio frame; each group of 7 tokens decomposes into
3 SNAC codebook levels which the SNAC model decodes into a waveform chunk.

Architecture (2 nodes):
    LLM           (ar)          - Llama 3.2 3B with extended vocab for audio tokens
    snac_decoder  (audio_codec) - SNAC 24kHz decoder

Graph walks (2):
    prefill - LLM only: fills KV cache with text prompt tokens
    decode  - Sequential[LLM, snac_decoder]: generate audio token, decode to PCM
"""

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mminf.communication.tensors import NameToTensorList
from mminf.engine.ar_engine import KVCacheConfig
from mminf.engine.base import EngineType
from mminf.graph.base import GraphEdge, GraphNode, GraphSection, Sequential, TensorPointerInfo
from mminf.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mminf.model.base import CurrentForwardMetadata, ForwardPassArgs, Model, NodeSubmodule
from mminf.model.orpheus.config import OrpheusModelConfig

logger = logging.getLogger(__name__)


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        return repo_id
    return str(Path(local_dir))


class OrpheusModel(Model):
    """Orpheus TTS model: Llama 3.2 3B + SNAC 24kHz decoder."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf
        self.config = OrpheusModelConfig()

        tokenizer_source = _resolve_local_hf_snapshot(
            "canopylabs/orpheus-3b-0.1-pretrained",
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            cache_dir=cache_dir,
        )

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -------------------------------------------------------------------
    # Model ABC: KV cache config
    # -------------------------------------------------------------------

    def get_kv_cache_config(self) -> KVCacheConfig:
        return KVCacheConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_attention_heads,
        )

    # -------------------------------------------------------------------
    # Model ABC: node engine types
    # -------------------------------------------------------------------

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "LLM": EngineType.AR,
            "snac_decoder": EngineType.AUDIO_CODEC,
        }

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name="LLM",
            input_ids=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="new_token",
                    is_new_token=True,
                    persist=True,
                ),
            ],
        )

        decode = Sequential(
            [
                GraphNode(
                    name="LLM",
                    input_ids=["text_inputs"],
                    outputs=[
                        GraphEdge(
                            next_node="snac_decoder",
                            name="audio_token",
                        ),
                        GraphEdge(
                            next_node=EMPTY_DESTINATION,
                            name="new_token",
                            is_new_token=True,
                            persist=True,
                        ),
                    ],
                ),
                GraphNode(
                    name="snac_decoder",
                    input_ids=["audio_token"],
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="audio_chunk",
                            output_modality="audio",
                        ),
                    ],
                ),
            ]
        )

        return dict(
            prefill=prefill,
            decode=decode,
        )

    # -------------------------------------------------------------------
    # Model ABC: prompt processing
    # -------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        **kwargs,
    ) -> NameToTensorList:
        if prompt is None:
            return {}

        voice = kwargs.get("voice", "tara")

        # Format: "{voice}: {text}"
        adapted_prompt = f"{voice}: {prompt}" if voice else prompt
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")

        # Wrap with special tokens: [128259, ...tokens..., 128009, 128260, 128261, 128257]
        start_token = torch.tensor([self.config.start_token_id], dtype=torch.long)
        end_tokens = torch.tensor(self.config.end_token_ids, dtype=torch.long)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids[0], end_tokens])

        return {"text_inputs": [all_input_ids]}

    # -------------------------------------------------------------------
    # Model ABC: forward pass args
    # -------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardMetadata(
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

    def get_forward_pass_args(
        self,
        metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
    ) -> ForwardPassArgs:
        request_done = False

        if metadata.is_prefill:
            # Transition from prefill to decode
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            # Check for stop token
            tokens = new_tokens.get("new_token", [])
            if self.config.stop_token_id in tokens:
                request_done = True

        # Build inputs for this forward pass
        if metadata.graph_walk == "prefill":
            graph_edge = GraphEdge(next_node="LLM", name="text_inputs")
            graph_edge.tensor_info = persist_signals.get("text_inputs", [])
            inputs = [graph_edge]
        else:
            # decode: previous token feeds back as text_inputs
            graph_edge = GraphEdge(next_node="LLM", name="text_inputs")
            graph_edge.tensor_info = persist_signals.get("new_token", [])
            inputs = [graph_edge]

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": metadata.is_prefill},
            request_done=request_done,
        )

    # -------------------------------------------------------------------
    # Model ABC: postprocess
    # -------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
    ) -> bytes:
        if modality == "audio":
            if output.numel() == 0:
                return b""
            return output.cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Orpheus: {modality!r}")

    # -------------------------------------------------------------------
    # Model ABC: submodule loading
    # -------------------------------------------------------------------

    def get_submodule(self, node_name: str, device: str = "cpu") -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        logger.info("Successfully loaded Orpheus submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(self, node_name: str, device: str) -> NodeSubmodule | None:
        if node_name == "LLM":
            return self._create_llm_submodule(device)
        elif node_name == "snac_decoder":
            return self._create_snac_submodule(device)
        return None

    def _create_llm_submodule(self, device: str) -> NodeSubmodule:
        from mminf.model.orpheus.components.language_model import OrpheusForCausalLM
        from mminf.model.orpheus.submodules import OrpheusLLMSubmodule
        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        local_dir = _resolve_local_hf_snapshot(
            self.model_path_hf,
            cache_dir=self.cache_dir,
        )

        with torch.device("meta"):
            language_model = OrpheusForCausalLM(self.config)

        load_weights_from_hf_shards(
            repo_dir=local_dir,
            modules=[ModuleAndPrefix(language_model)],
            device=device,
        )

        language_model.eval()
        return OrpheusLLMSubmodule(
            language_model=language_model,
            config=self.config,
        )

    def _create_snac_submodule(self, device: str) -> NodeSubmodule:
        from mminf.model.orpheus.components.snac import SNAC
        from mminf.model.orpheus.submodules import SNACDecoderSubmodule

        snac_source = _resolve_local_hf_snapshot(
            self.config.snac_model_id,
            cache_dir=self.cache_dir,
        )
        snac_model = SNAC.from_pretrained(snac_source).eval().to(device)
        return SNACDecoderSubmodule(
            snac_model=snac_model,
            config=self.config,
        )
