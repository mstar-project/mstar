"""Zonos2 TTS model: LLM (multi-codebook AR decoder) + DAC vocoder.

This is the ``Model`` ABC implementation. It wires Zonos2 into the mstar
serving stack. It mirrors ``mstar/model/orpheus/orpheus_model.py`` in
structure. That model streams tokens from an autoregressive LLM partition to
an audio-codec partition. This one adapts it for Zonos2's multi-codebook
frames and DAC vocoder.

Two async partitions:
  * ``LLM``  (KV-cache engine)  — prefill, then a decode loop. Each step
    samples a frame ``[cb0..cb8, text]`` and streams it to ``DAC``.
  * ``DAC``  (stateless engine) — it accumulates streamed frames, runs
    ``shear_up``, and DAC-decodes to PCM. It emits the PCM to the client.

Graph walks: ``prefill`` -> ``decode`` (Loop) on LLM; ``dac_chunk`` on DAC.
"""
from __future__ import annotations

import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_cache_engine import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.zonos2.config import Zonos2Config
from mstar.model.zonos2.prompt import BYTE_TEXT_VOCAB_SIZE, TTSPromptBuilder
from mstar.model.zonos2.tts_sampling import TTSSamplingParams
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

logger = logging.getLogger(__name__)

_LLM = "LLM"
_DAC_NODE = "dac_decoder"
_LLM_PART = "LLM"
_DAC_PART = "DAC"
_DECODE_LOOP = "decode_loop"


class Zonos2Model(Model):
    """Zonos2 multi-codebook TTS: AR LLM + streaming DAC vocoder."""

    def __init__(
        self,
        model_path_hf: str | None = None,
        cache_dir: str | None = None,
        config: Zonos2Config | None = None,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # Extra kwargs come from the serving YAML's ``model_kwargs``. They win
        # over the checkpoint config. This matches the pi05 pattern.
        self._yaml_overrides = dict(kwargs)
        self.config = config or self._load_config()
        self.sampling_params = TTSSamplingParams()
        self._prompt_builder = TTSPromptBuilder(
            n_codebooks=self.config.n_codebooks,
            audio_pad_id=self.config.audio_pad_id,
            text_vocab=self.config.text_vocab or BYTE_TEXT_VOCAB_SIZE,
        )
        self._submodule_cache: dict[str, torch.nn.Module | None] = {}

    def _load_config(self) -> Zonos2Config:
        """Build the config from the checkpoint's ``params.json``, with YAML
        ``model_kwargs`` overrides. Fall back to defaults if unavailable."""
        if self.model_path_hf:
            try:
                from mstar.model.zonos2.weight_loader import (
                    load_zonos2_config_from_checkpoint,
                    resolve_zonos2_checkpoint,
                )

                ckpt = resolve_zonos2_checkpoint(self.model_path_hf, self.cache_dir)
                cfg = load_zonos2_config_from_checkpoint(ckpt, **self._yaml_overrides)
                if cfg.text_vocab is None:
                    cfg.text_vocab = BYTE_TEXT_VOCAB_SIZE
                logger.info(
                    "Zonos2: config from checkpoint (%d layers, dim %d, %d experts)",
                    cfg.num_layers, cfg.hidden_size, cfg.moe_n_experts,
                )
                return cfg
            except Exception as e:
                logger.warning(
                    "Zonos2: could not load config from %s (%s); using defaults.",
                    self.model_path_hf, e,
                )
        cfg = Zonos2Config()
        for key, value in self._yaml_overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    # ------------------------------------------------------------------
    # Model ABC: engines + KV cache
    # ------------------------------------------------------------------
    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        return [
            KVCacheConfig(
                num_layers=self.config.num_layers,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_position_embeddings,
                num_qo_heads=self.config.num_qo_heads,
            )
        ]

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {_LLM: EngineType.KV_CACHE, _DAC_NODE: EngineType.STATELESS}

    def get_max_output_tokens(self, **model_kwargs) -> int:
        return model_kwargs.get("max_output_tokens", self.sampling_params.max_tokens)

    # ------------------------------------------------------------------
    # Model ABC: graph walks
    # ------------------------------------------------------------------
    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name=_LLM,
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="new_token",
                    conductor_new_token=True,
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node=_DAC_NODE, name="new_token", target_partition=_DAC_PART,
                ),
            ],
        )

        decode = Loop(
            name=_DECODE_LOOP,
            section=GraphNode(
                name=_LLM,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=_LLM, name="text_inputs"),
                    StreamingGraphEdge(
                        next_node=_DAC_NODE, name="new_token", target_partition=_DAC_PART,
                    ),
                ],
            ),
            # Hard safety ceiling only. The graph is built once at init, so this
            # value is baked in and CANNOT see per-request model_kwargs; calling
            # get_max_output_tokens() here silently pinned every request to the
            # global TTSSamplingParams.max_tokens default (1024), truncating any
            # utterance longer than ~1024 frames. The real per-request bound
            # (natural EOS + request max_tokens) is enforced in check_stop, so
            # this only needs to be a ceiling the sequence can never physically
            # exceed: the KV-cache / position capacity.
            max_iters=self.config.max_position_embeddings,
            outputs=[],
        )

        dac_chunk = GraphNode(
            name=_DAC_NODE,
            input_names=["new_token"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT, name="audio_chunk", output_modality="audio",
                ),
            ],
        )

        return dict(prefill=prefill, decode=decode, dac_chunk=dac_chunk)

    # ------------------------------------------------------------------
    # Partition API (LLM + DAC async streaming)
    # ------------------------------------------------------------------
    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(
            partitions=[_LLM_PART, _DAC_PART],
            connections=[
                Connection(
                    from_partition=_LLM_PART,
                    to_partition=_DAC_PART,
                    edge_name="new_token",
                    chunk_policy_factory=lambda: FixedChunkPolicy(
                        chunk_size=self.config.dac_chunk_frames,
                    ),
                ),
            ],
        )

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name=_LLM_PART,
                graph_walks={"prefill", "decode"},
                initial_walk="prefill",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name=_DAC_PART,
                graph_walks={"dac_chunk"},
                initial_walk=None,
                producer_partitions=[_LLM_PART],
            ),
        ]

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        if partition_name == _LLM_PART:
            return self._llm_partition_forward(partition_metadata, persist_signals)
        if partition_name == _DAC_PART:
            partition_metadata.graph_walk = "dac_chunk"
            return ForwardPassArgs(
                full_metadata=partition_metadata, inputs=[], unpersist_tensors=[],
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    def _llm_partition_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> ForwardPassArgs:
        """prefill -> decode loop -> done."""
        request_done = False
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            request_done = True
            metadata.kwargs["decode_finished"] = True

        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True,
            )

        graph_edge = GraphEdge(next_node=_LLM, name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [graph_edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": metadata.is_prefill},
        )

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        if partition_name == _LLM_PART:
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="prefill",
                is_prefill=True,
            )
            graph_edge = GraphEdge(next_node=_LLM, name="text_inputs")
            graph_edge.tensor_info = input_signals.get("text_inputs", [])
            inputs = [graph_edge]
            unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=inputs,
                unpersist_tensors=unpersist_tensors,
                step_metadata={"is_prefill": True},
            )
        if partition_name == _DAC_PART:
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="dac_chunk",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[],
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    # ------------------------------------------------------------------
    # Model ABC: prompt + postprocess
    # ------------------------------------------------------------------
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
        frames = self._prompt_builder.build(prompt)  # (num_frames, n_codebooks + 1)
        return {"text_inputs": [frames]}

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality == "audio":
            if output.numel() == 0:
                return b""
            return output.cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Zonos2: {modality!r}")

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.sample_rate

    # ------------------------------------------------------------------
    # Model ABC: sharding
    # ------------------------------------------------------------------
    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={_LLM}, shard_dim={})

    # ------------------------------------------------------------------
    # Model ABC: submodule construction
    # ------------------------------------------------------------------
    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> torch.nn.Module | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, tp_group, autocast_dtype)
        logger.info("Loaded Zonos2 submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(self, node_name, device, tp_group, autocast_dtype):
        if node_name == _LLM:
            return self._create_llm_submodule(device, tp_group, autocast_dtype)
        if node_name == _DAC_NODE:
            return self._create_dac_submodule(device)
        return None

    def _create_llm_submodule(self, device, tp_group, autocast_dtype):
        from mstar.model.zonos2.components.language_model import Zonos2ForCausalLM
        from mstar.model.zonos2.submodules import Zonos2LLMSubmodule
        from mstar.model.zonos2.weight_loader import (
            load_zonos2_weights,
            resolve_zonos2_checkpoint,
        )

        with torch.device("meta"):
            model = Zonos2ForCausalLM(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            model = model.to(autocast_dtype)
        model.to_empty(device=device)

        if self.model_path_hf:
            try:
                ckpt = resolve_zonos2_checkpoint(self.model_path_hf, self.cache_dir)
                load_zonos2_weights(model, ckpt, device=device)
            except Exception as e:
                logger.warning(
                    "Zonos2: weight loading failed (%s); LLM weights are uninitialized.", e,
                )
        else:
            logger.warning(
                "Zonos2: no model_path_hf given; LLM weights are uninitialized."
            )
        model.eval()

        return Zonos2LLMSubmodule(
            model=model,
            n_codebooks=self.config.n_codebooks,
            text_vocab=self.config.text_vocab,
            eoa_id=self.config.eoa_id,
            params=self.sampling_params,
        )

    def _create_dac_submodule(self, device):
        from mstar.model.zonos2.submodules import Zonos2DACSubmodule
        from mstar.model.zonos2.vocoder import StreamingDacDecoder

        decoder = StreamingDacDecoder(
            n_codebooks=self.config.n_codebooks,
            audio_pad_id=self.config.audio_pad_id,
            codebook_size=self.config.codebook_size,
            sample_rate=self.config.sample_rate,
            model_type=self.config.dac_model_type,
            overlap_frames=self.config.dac_overlap_frames,
            hop_length=self.config.dac_hop_length,
        )
        return Zonos2DACSubmodule(decoder, self.config.n_codebooks).to(device)
