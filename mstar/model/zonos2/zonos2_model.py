"""Zonos2 TTS model: a multi-codebook AR decoder and a DAC vocoder.

This is the ``Model`` ABC implementation that wires Zonos2 into the mstar
serving stack. Its structure follows ``mstar/model/orpheus/orpheus_model.py``,
which streams tokens from an autoregressive LLM partition to an audio-codec
partition. This model adapts that structure to the multi-codebook frames and
the DAC vocoder of Zonos2.

There are two async partitions:

* ``LLM`` (KV-cache engine). It runs a prefill, then a decode loop. Each step
  samples a frame ``[cb0..cb8, text]`` and streams it to ``DAC``.
* ``DAC`` (stateless engine). It collects the streamed frames, runs
  ``shear_up``, decodes to PCM, and emits the PCM to the client.

Voice cloning adds a third node, ``speaker_encoder`` (stateless), in the LLM
partition. It turns a reference clip into one speaker embedding, and the LLM
writes that embedding over the hidden state of a reserved prompt frame. It runs
only on the ``prefill_clone`` walk, so a text-only request never uses it.

Graph walks: ``prefill`` (or ``prefill_clone``), then ``decode`` (a Loop) on
the LLM, and ``dac_chunk`` on the DAC.
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
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
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
_SPK_NODE = "speaker_encoder"
_LLM_PART = "LLM"
_DAC_PART = "DAC"
_DECODE_LOOP = "decode_loop"
_PREFILL = "prefill"
_PREFILL_CLONE = "prefill_clone"
_DECODE = "decode"
_DAC_CHUNK = "dac_chunk"


class Zonos2Model(Model):
    """Zonos2 multi-codebook TTS: an AR LLM and a streaming DAC vocoder."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        config: Zonos2Config | None = None,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        # The extra kwargs come from the ``model_kwargs`` of the serving YAML.
        # They override the checkpoint config. This follows the pi05 pattern.
        self._yaml_overrides = dict(kwargs)
        self.config = config or self._load_config()
        self.sampling_params = TTSSamplingParams()
        self._prompt_builder = TTSPromptBuilder(
            n_codebooks=self.config.n_codebooks,
            audio_pad_id=self.config.audio_pad_id,
            text_vocab=self.config.text_vocab or BYTE_TEXT_VOCAB_SIZE,
            speaking_rate_num_buckets=self.config.speaking_rate_num_buckets,
            quality_bucket_counts=self.config.quality_bucket_counts,
            speaker_background_num_buckets=self.config.speaker_background_num_buckets,
            accurate_mode_num_buckets=self.config.accurate_mode_num_buckets,
        )
        self._submodule_cache: dict[str, torch.nn.Module | None] = {}

    def _load_config(self) -> Zonos2Config:
        """Build the config from the ``params.json`` of the checkpoint.

        The ``model_kwargs`` of the serving YAML override the values. Only
        ``skip_weight_loading``, which the tests use, falls back to the
        defaults. A serving run that cannot read its own ``params.json`` would
        otherwise build the wrong architecture and then serve noise without an
        error.
        """
        if self.skip_weight_loading:
            cfg = Zonos2Config()
            for key, value in self._yaml_overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
            return cfg

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
        # The code declares speaker_encoder always. The serving YAML lists the
        # node without knowledge of the checkpoint, and EngineManager looks up
        # every configured node here. On a checkpoint with no speaker, the
        # submodule factory returns None and no walk reaches the node.
        return {
            _LLM: EngineType.KV_CACHE,
            _DAC_NODE: EngineType.STATELESS,
            _SPK_NODE: EngineType.STATELESS,
        }

    def get_max_output_tokens(self, **model_kwargs) -> int:
        return model_kwargs.get("max_output_tokens", self.sampling_params.max_tokens)

    # ------------------------------------------------------------------
    # Model ABC: graph walks
    # ------------------------------------------------------------------
    def _llm_prefill_node(self) -> GraphNode:
        """Return the prefill node of the LLM, shared by both prefill walks.

        The node always declares ``speaker_embedding``. On the text-only walk
        the conductor gives it an empty tensor list, as it does for the
        ``new_token`` edge on the first decode step. The submodule treats absent
        and empty the same way.
        """
        return GraphNode(
            name=_LLM,
            input_names=["text_inputs", "speaker_embedding"],
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

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = self._llm_prefill_node()

        # Voice cloning: encode the reference clip, then prefill with the
        # embedding. Both nodes are in the LLM partition, so this is an ordinary
        # in-partition sequence and it needs no streaming connection.
        prefill_clone = Sequential([
            GraphNode(
                name=_SPK_NODE,
                input_names=["audio_inputs"],
                outputs=[GraphEdge(next_node=_LLM, name="speaker_embedding")],
            ),
            self._llm_prefill_node(),
        ])

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
            # A hard safety ceiling only. The code builds the graph once at
            # init, so this value cannot see the per-request model_kwargs. Do
            # NOT call get_max_output_tokens() here: it pins every request to
            # the global TTSSamplingParams.max_tokens default of 1024, and it
            # then truncates any utterance longer than about 1024 frames.
            # ``check_stop`` enforces the real per-request bound, which is a
            # natural EOS or the max_tokens of the request. This value therefore
            # only needs to be a ceiling that the sequence can never reach: the
            # KV-cache and position capacity.
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

        walks: dict[str, GraphSection] = {
            _PREFILL: prefill, _DECODE: decode, _DAC_CHUNK: dac_chunk,
        }
        # The clone walk uses the speaker_encoder node. That node exists, and it
        # needs a node_groups entry, only on a speaker-conditioned model.
        if self.config.speaker_enabled:
            walks[_PREFILL_CLONE] = prefill_clone
        return walks

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
        llm_walks = {_PREFILL, _DECODE}
        if self.config.speaker_enabled:
            llm_walks.add(_PREFILL_CLONE)
        return [
            PartitionDefinition(
                name=_LLM_PART,
                graph_walks=llm_walks,
                initial_walk=_PREFILL,
                producer_partitions=[],
            ),
            PartitionDefinition(
                name=_DAC_PART,
                graph_walks={_DAC_CHUNK},
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
            partition_metadata.graph_walk = _DAC_CHUNK
            return ForwardPassArgs(
                full_metadata=partition_metadata, inputs=[], unpersist_tensors=[],
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    def _llm_partition_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> ForwardPassArgs:
        """Advance the LLM partition: prefill, then the decode loop, then done."""
        request_done = False
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = _DECODE
        elif metadata.graph_walk == _DECODE:
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
            # Raw reference audio needs the encoder node. An embedding from the
            # caller (see process_prompt) is already encoded, so it takes the
            # plain prefill walk and carries the speaker_embedding signal.
            needs_encoder = (
                self.config.speaker_enabled and bool(input_signals.get("audio_inputs"))
            )
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk=_PREFILL_CLONE if needs_encoder else _PREFILL,
                is_prefill=True,
            )
            text_edge = GraphEdge(next_node=_LLM, name="text_inputs")
            text_edge.tensor_info = input_signals.get("text_inputs", [])
            inputs = [text_edge]
            if needs_encoder:
                audio_edge = GraphEdge(next_node=_SPK_NODE, name="audio_inputs")
                audio_edge.tensor_info = input_signals.get("audio_inputs", [])
                inputs.append(audio_edge)
            else:
                spk_edge = GraphEdge(next_node=_LLM, name="speaker_embedding")
                spk_edge.tensor_info = input_signals.get("speaker_embedding", [])
                inputs.append(spk_edge)
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
                graph_walk=_DAC_CHUNK,
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[],
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    # ------------------------------------------------------------------
    # Model ABC: prompt + postprocess
    # ------------------------------------------------------------------
    def load_audio(self, filepath: str, device: str):
        """Decode a reference clip at the sample rate of the speaker encoder.

        The base implementation hardcodes 16 kHz, but the encoder needs 24 kHz.
        That combination forces a lossy upsample with no content above 8 kHz.
        The embedding then degrades, but the code does not fail.
        """
        from torchcodec.decoders import AudioDecoder

        from mstar.model.base import TensorAndMetadata

        sample_rate = self.config.speaker_encoder_sample_rate
        decoder = AudioDecoder(filepath, sample_rate=sample_rate, num_channels=1)
        audio = decoder.get_all_samples().data[0]
        return TensorAndMetadata(
            data=audio, metadata=dict(sample_rate=sample_rate, num_channels=1),
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

        speaker_embedding = self._resolve_speaker_embedding(kwargs.get("speaker_embedding"))
        has_reference_audio = bool((tensors or {}).get("audio_inputs"))
        speaker = speaker_embedding is not None or has_reference_audio
        if speaker and not self.config.speaker_enabled:
            raise ValueError(
                "Reference audio / speaker_embedding was supplied, but this Zonos2 "
                "checkpoint is not speaker-conditioned (params.json has "
                "speaker_enabled=false), so it has no speaker projection weights."
            )

        from mstar.model.zonos2.conditioning import (
            resolve_quality_buckets,
            resolve_speaking_rate_bucket,
        )

        frames = self._prompt_builder.build(
            prompt,
            speaker=speaker,
            clean_speaker_background=bool(kwargs.get("clean_speaker_background", True)),
            accurate_mode=bool(kwargs.get("accurate_mode", False)),
            speaking_rate_bucket=resolve_speaking_rate_bucket(
                self.config,
                speaking_rate_bucket=kwargs.get("speaking_rate_bucket"),
                speaking_rate=kwargs.get("speaking_rate"),
                speed=kwargs.get("speed"),
            ),
            quality_buckets=resolve_quality_buckets(
                self.config,
                quality_buckets=kwargs.get("quality_buckets"),
                quality_values=kwargs.get("quality_values"),
            ),
        )  # (num_frames, n_codebooks + 1)

        out: NameToTensorList = {"text_inputs": [frames]}
        if speaker_embedding is not None:
            out["speaker_embedding"] = [speaker_embedding]
        return out

    def _resolve_speaker_embedding(self, value) -> torch.Tensor | None:
        """Normalize an embedding from the caller to ``(1, speaker_embedding_dim)``.

        A client can cache the output of the encoder. Later requests for the
        same voice then skip the encoder node.
        """
        if value is None:
            return None
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        tensor = tensor.to(torch.float32).reshape(-1)
        expected = self.config.speaker_embedding_dim
        if tensor.numel() != expected:
            raise ValueError(
                f"speaker_embedding must have {expected} elements, got {tensor.numel()}."
            )
        return tensor.unsqueeze(0).contiguous()

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
        if node_name == _SPK_NODE:
            return self._create_speaker_encoder_submodule(device)
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

        if self.skip_weight_loading:
            logger.warning("Zonos2: skip_weight_loading set; LLM weights are uninitialized.")
        else:
            ckpt = resolve_zonos2_checkpoint(self.model_path_hf, self.cache_dir)
            load_zonos2_weights(model, ckpt, device=device)
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

    def _create_speaker_encoder_submodule(self, device):
        """Build the voice-clone encoder node.

        Unlike the LLM, this is an off-the-shelf HF model, not a Zonos2
        checkpoint. ``model.pth`` holds only the projections downstream of it.
        """
        from mstar.model.zonos2.speaker_encoder import Qwen3SpeakerEncoder
        from mstar.model.zonos2.submodules import Zonos2SpeakerEncoderSubmodule

        if not self.config.speaker_enabled:
            # This checkpoint has no speaker projection weights, so there is
            # nothing to feed. Skip the encoder download.
            logger.info(
                "Zonos2: checkpoint is not speaker-conditioned; skipping the "
                "voice-clone encoder."
            )
            return None

        if self.skip_weight_loading:
            logger.warning(
                "Zonos2: skip_weight_loading set; no speaker encoder is built, so "
                "voice-clone requests will fail."
            )
            return None

        encoder = Qwen3SpeakerEncoder(
            model_id=self.config.speaker_encoder_model_id,
            embedding_dim=self.config.speaker_embedding_dim,
            cache_dir=self.cache_dir,
            device=device,
        )
        return Zonos2SpeakerEncoderSubmodule(
            encoder=encoder,
            sample_rate=self.config.speaker_encoder_sample_rate,
        ).to(device)
