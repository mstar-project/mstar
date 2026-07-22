"""Qwen3-TTS model contract and two-partition streaming topology.

The 0.6B CustomVoice checkpoint is a text-to-speech model without the large
multimodal Thinker used by Qwen3-Omni. Its autoregressive Talker predicts one
12 Hz codec frame per step: group 0 comes from the Talker language model and
groups 1-15 come from a small depth-wise CodePredictor. The speech-tokenizer
decoder turns those frames into 24 kHz PCM.

Architecture (two asynchronous partitions):
    Talker - text/voice prefill, then autoregressive 16-group codec frames
    Codec  - stateless speech-tokenizer decoder producing PCM chunks

Streaming topology:
    Talker --[codec_tokens, LeftContextChunkPolicy(300, 25)]--> Codec

Request state machine:
    Talker: talker_prefill -> talker_decode loop -> done on EOS/token limit
    Codec:  waits for streamed frames -> codec_chunk -> emits audio -> waits

This class runs in the API/conductor side. It owns request validation, graph
and partition declarations, state-machine transitions, sampling defaults, and
lazy worker-side construction. Heavy weights are not loaded in ``__init__``.
"""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

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
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig
from mstar.model.submodule_base import NodeSubmodule
from mstar.streaming.chunk_policy import LeftContextChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge
from mstar.utils.sampling import SamplingConfig

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _resolve_model_metadata(repo_id: str, cache_dir: str | None) -> str:
    """Resolve only the files needed to construct config and tokenize input.

    The API and conductor processes do not need model tensors. Downloading a
    metadata-only snapshot here keeps them from allocating or transferring the
    multi-gigabyte checkpoint. Workers fetch the complete snapshot lazily from
    ``get_submodule``.
    """
    local_path = Path(repo_id)
    if local_path.is_dir():
        return str(local_path)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "speech_tokenizer/config.json",
        ],
    )


# ---------------------------------------------------------------------------
# Model contract
# ---------------------------------------------------------------------------


class Qwen3TTSModel(Model):
    """Qwen3-TTS 12 Hz CustomVoice model contract.

    GPU computation is split into an autoregressive Talker partition and a
    streaming Codec partition. This class owns only model-level scheduling,
    prompt processing, configuration, and output encoding.
    """

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir

        # The lightweight API-side object needs config and tokenizer only.
        self.local_dir = _resolve_model_metadata(model_path_hf, cache_dir)
        self.config = Qwen3TTSModelConfig.from_pretrained(self.local_dir)
        if self.config.tts_model_type != "custom_voice":
            raise ValueError(
                "The first Qwen3-TTS integration supports only CustomVoice "
                f"checkpoints, got {self.config.tts_model_type!r}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir,
            cache_dir=cache_dir,
            fix_mistral_regex=True,
        )

        # Each worker asks only for nodes assigned to it. Cache the resulting
        # wrappers so Talker and Codec weights are materialized at most once.
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def _ensure_full_snapshot(self) -> str:
        """Make all weight files available immediately before worker loading."""
        if (Path(self.local_dir) / "model.safetensors").is_file():
            return self.local_dir
        if Path(self.model_path_hf).is_dir():
            raise FileNotFoundError(
                f"No model.safetensors found in {self.model_path_hf}"
            )
        from huggingface_hub import snapshot_download

        self.local_dir = snapshot_download(
            repo_id=self.model_path_hf,
            cache_dir=self.cache_dir,
        )
        return self.local_dir

    # -----------------------------------------------------------------------
    # Model ABC: KV cache and engine assignment
    # -----------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        """Declare the paged self-attention cache used only by the Talker."""
        talker = self.config.talker
        return [KVCacheConfig(
            num_layers=talker.num_hidden_layers,
            num_kv_heads=talker.num_key_value_heads,
            head_dim=talker.head_dim,
            max_seq_len=talker.max_position_embeddings,
            num_qo_heads=talker.num_attention_heads,
            nodes=["Talker"],
            # FlashInfer 0.6.x auto-selects an FA3 JIT kernel on Hopper. The
            # supported deployment toolchain uses the compatible FA2 kernel.
            flashinfer_backend="fa2",
        )]

    def get_node_engine_types(self) -> dict[str, EngineType]:
        """Talker keeps cross-step KV state; Codec is a pure frame decoder."""
        return {
            "Talker": EngineType.KV_CACHE,
            "Codec": EngineType.STATELESS,
        }

    # -----------------------------------------------------------------------
    # Model ABC: walk graph declaration
    # -----------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        """Declare prefill, autoregressive decode, and codec chunk walks.

        ``talker_input_embeds`` is the recurrent Talker edge. ``codec_tokens``
        crosses the asynchronous partition boundary and is buffered according
        to ``get_partition_topology`` before Codec is scheduled.
        """
        # Prefill seeds both recurrent paths: the embedding for the next
        # Talker step is persisted, while the first codec frame starts the
        # Talker-to-Codec stream.
        talker_prefill = GraphNode(
            name="Talker",
            input_names=["text_inputs", "speaker_id", "language_id"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="talker_input_embeds",
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="Codec",
                    name="codec_tokens",
                    target_partition="Codec",
                ),
            ],
        )

        # Each loop iteration predicts one complete 16-group codec frame and
        # feeds the summed codec embedding back into the next Talker step.
        talker_decode = Loop(
            name="talker_decode_loop",
            section=GraphNode(
                name="Talker",
                input_names=["talker_input_embeds"],
                outputs=[
                    GraphEdge(
                        next_node="Talker",
                        name="talker_input_embeds",
                    ),
                    StreamingGraphEdge(
                        next_node="Codec",
                        name="codec_tokens",
                        target_partition="Codec",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        # Codec is deliberately a separate walk/engine so waveform decoding
        # can overlap with subsequent Talker steps.
        codec_chunk = GraphNode(
            name="Codec",
            input_names=["codec_tokens"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="audio_chunk",
                    output_modality="audio",
                ),
            ],
        )
        return {
            "talker_prefill": talker_prefill,
            "talker_decode": talker_decode,
            "codec_chunk": codec_chunk,
        }

    # -----------------------------------------------------------------------
    # Asynchronous partitions and stream buffering
    # -----------------------------------------------------------------------

    def get_partitions(self) -> list[PartitionDefinition]:
        """Split autoregressive generation from independently scheduled audio."""
        return [
            PartitionDefinition(
                name="Talker",
                graph_walks={"talker_prefill", "talker_decode"},
                initial_walk="talker_prefill",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name="Codec",
                graph_walks={"codec_chunk"},
                initial_walk=None,
                producer_partitions=["Talker"],
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        """Buffer codec frames with the decoder's required left context.

        The first Codec invocation receives up to ``chunk_frames`` new frames.
        Later invocations prepend ``left_context_frames`` old frames to avoid
        convolution boundary artifacts; ``CodecSubmodule.postprocess`` removes
        the duplicated PCM prefix before emission.
        """
        codec = self.config.codec
        return PartitionTopology(
            partitions=["Talker", "Codec"],
            connections=[
                Connection(
                    from_partition="Talker",
                    to_partition="Codec",
                    edge_name="codec_tokens",
                    chunk_policy_factory=lambda: LeftContextChunkPolicy(
                        chunk=codec.chunk_frames,
                        left_context=codec.left_context_frames,
                    ),
                ),
            ],
        )

    # -----------------------------------------------------------------------
    # API preprocessing
    # -----------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs: Any,
    ) -> NameToTensorList:
        """Validate a CustomVoice request and build Talker input tensors.

        Qwen3-TTS expects an assistant ChatML turn rather than a generic user
        turn. Speaker and language are separate codec-side conditioning IDs;
        ``-1`` means automatic language selection.
        """
        del tensors
        if not prompt:
            raise ValueError("Qwen3-TTS requires a non-empty text prompt")
        if set(input_modalities) != {"text"}:
            raise ValueError(
                "Qwen3-TTS CustomVoice currently supports text input only"
            )
        if set(output_modalities) != {"audio"}:
            raise ValueError("Qwen3-TTS CustomVoice supports audio output only")
        if kwargs.get("instruct"):
            raise ValueError(
                "Qwen3-TTS 0.6B CustomVoice does not support instructions"
            )

        speaker = str(
            kwargs.get("speaker", kwargs.get("voice", self.config.default_speaker))
        ).lower()
        if speaker not in self.config.talker.spk_id:
            supported = ", ".join(sorted(self.config.talker.spk_id))
            raise ValueError(
                f"Unsupported Qwen3-TTS speaker {speaker!r}; supported: {supported}"
            )

        language = str(kwargs.get("language", self.config.default_language)).lower()
        if language != "auto" and language not in self.config.talker.codec_language_id:
            supported = ", ".join(
                ["auto", *sorted(self.config.talker.codec_language_id)]
            )
            raise ValueError(
                f"Unsupported Qwen3-TTS language {language!r}; supported: {supported}"
            )

        dialect = self.config.talker.spk_is_dialect.get(speaker, False)
        if dialect and language in {"auto", "chinese"}:
            language = str(dialect)

        # Match the official processor template exactly. `_build_prefill`
        # relies on the fixed assistant suffix when separating prompt tokens
        # into the initial prefill and per-frame text conditioning stream.
        formatted = (
            f"<|im_start|>assistant\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        encoded = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = encoded["input_ids"]
        if text_inputs.ndim == 2:
            text_inputs = text_inputs[0]

        language_id = self.config.talker.codec_language_id.get(language, -1)
        return {
            "text_inputs": [text_inputs.to(dtype=torch.long)],
            "speaker_id": [torch.tensor(
                [self.config.talker.spk_id[speaker]], dtype=torch.long
            )],
            "language_id": [torch.tensor([language_id], dtype=torch.long)],
        }

    # -----------------------------------------------------------------------
    # Conductor partition state machine
    # -----------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        """Create each partition's initial state.

        Talker starts immediately from API tensors. Codec has no direct API
        inputs and remains dormant until its incoming streaming connection has
        enough frames to schedule ``codec_chunk``.
        """
        model_kwargs = model_kwargs or {}
        if partition_name == "Talker":
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="talker_prefill",
                is_prefill=True,
                kwargs={
                    "talker_max_tokens": self.get_max_output_tokens(**model_kwargs),
                    "subtalker_sampling": self._get_subtalker_sampling(model_kwargs),
                },
            )
            inputs = []
            for name in ("text_inputs", "speaker_id", "language_id"):
                edge = GraphEdge(next_node="Talker", name=name)
                edge.tensor_info = input_signals.get(name, [])
                inputs.append(edge)
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=inputs,
                unpersist_tensors=sum(
                    [edge.tensor_info for edge in inputs], start=[]
                ),
                request_done="audio" not in output_modalities,
                step_metadata=self._talker_step_metadata(metadata),
            )

        if partition_name == "Codec":
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="codec_chunk",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
            )
        raise ValueError(f"Unknown Qwen3-TTS partition: {partition_name!r}")

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Advance Talker prefill/decode and re-arm the streaming Codec.

        Loop iterations are executed inside the ``talker_decode`` graph walk,
        so the conductor sees that walk once after its loop stops. Codec is
        rescheduled by connection readiness rather than an internal walk
        transition.
        """
        del incoming_connections
        if partition_name == "Talker":
            if partition_metadata.graph_walk == "talker_prefill":
                partition_metadata.graph_walk = "talker_decode"
                partition_metadata.is_prefill = False
                edge = GraphEdge(next_node="Talker", name="talker_input_embeds")
                edge.tensor_info = persist_signals.get("talker_input_embeds", [])
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[edge],
                    unpersist_tensors=list(edge.tensor_info),
                    step_metadata=self._talker_step_metadata(partition_metadata),
                )
            if partition_metadata.graph_walk == "talker_decode":
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[],
                    unpersist_tensors=[],
                    request_done=True,
                )
            raise ValueError(
                "Talker entered an unexpected graph walk: "
                f"{partition_metadata.graph_walk!r}"
            )

        if partition_name == "Codec":
            partition_metadata.graph_walk = "codec_chunk"
            return ForwardPassArgs(
                full_metadata=partition_metadata,
                inputs=[],
                unpersist_tensors=[],
                step_metadata={
                    "codec_chunk_frames": self.config.codec.chunk_frames,
                    "codec_left_context_frames": (
                        self.config.codec.left_context_frames
                    ),
                },
            )
        raise ValueError(f"Unknown Qwen3-TTS partition: {partition_name!r}")

    # -----------------------------------------------------------------------
    # Sampling and output encoding
    # -----------------------------------------------------------------------

    def get_sampling_config(
        self,
        node_name: str,
        model_kwargs: dict | None = None,
    ) -> SamplingConfig | None:
        """Configure sampling for codec group 0 predicted by the Talker.

        Residual groups 1-15 use the independent ``subtalker_*`` settings
        carried in step metadata and consumed by ``TalkerSubmodule``.
        """
        if node_name != "Talker":
            return None
        model_kwargs = model_kwargs or {}
        generation = self.config.generation
        do_sample = model_kwargs.get("do_sample", generation.do_sample)
        temperature = model_kwargs.get(
            "temperature",
            model_kwargs.get("talker_temperature", generation.temperature),
        )
        if not do_sample:
            temperature = 0.0
        return SamplingConfig(
            vocab_size=self.config.talker.vocab_size,
            temperature=temperature,
            top_k=model_kwargs.get("top_k", generation.top_k),
            top_p=model_kwargs.get("top_p", generation.top_p),
            repetition_penalty=model_kwargs.get(
                "repetition_penalty", generation.repetition_penalty
            ),
            ignore_eos=model_kwargs.get("ignore_eos", False),
        )

    def get_max_output_tokens(self, **model_kwargs: Any) -> int:
        return model_kwargs.get(
            "max_output_tokens",
            model_kwargs.get("max_new_tokens", self.config.generation.max_new_tokens),
        )

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.codec.output_sample_rate

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        request_kwargs: dict | None = None,
    ) -> bytes:
        """Encode emitted waveform tensors as raw little-endian PCM16 bytes."""
        del request_kwargs
        if modality != "audio":
            raise ValueError(f"Unsupported Qwen3-TTS output modality: {modality!r}")
        if output.numel() == 0:
            return b""
        pcm = output.detach().cpu()
        if pcm.is_floating_point():
            pcm = (pcm.clamp(-1, 1) * 32767).to(torch.int16)
        elif pcm.dtype != torch.int16:
            pcm = pcm.to(torch.int16)
        return pcm.contiguous().numpy().tobytes()

    # -----------------------------------------------------------------------
    # Lazy worker-side submodule and weight loading
    # -----------------------------------------------------------------------

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule | None:
        """Build only the node assigned to this worker and cache the wrapper."""
        del sp_group
        if node_name not in self.get_node_engine_types():
            raise ValueError(f"Unknown Qwen3-TTS node: {node_name!r}")
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]

        self._ensure_full_snapshot()
        if node_name == "Talker":
            submodule = self._create_talker_submodule(
                device=device,
                tp_group=tp_group,
                autocast_dtype=autocast_dtype,
            )
        else:
            submodule = self._create_codec_submodule(device=device)
        self._submodule_cache[node_name] = submodule
        return submodule

    @staticmethod
    def _verify_loaded(
        module: torch.nn.Module,
        loaded: set[str],
        component: str,
    ) -> None:
        """Fail startup if checkpoint filtering left any parameter uninitialized."""
        expected = set(dict(module.named_parameters()))
        missing = sorted(expected - loaded)
        if missing:
            preview = ", ".join(missing[:8])
            raise RuntimeError(
                f"{component} checkpoint did not initialize {len(missing)} "
                f"parameters: {preview}"
            )

    def _create_talker_submodule(
        self,
        device: str,
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        ) -> NodeSubmodule:
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_tts.components.talker import (
            Qwen3TTSCodePredictor,
            Qwen3TTSTalkerModel,
        )
        from mstar.model.qwen3_tts.submodules import TalkerSubmodule

        # Construct on meta first so worker startup never holds both a random
        # initialization and checkpoint tensors in device memory.
        with torch.device("meta"):
            talker = Qwen3TTSTalkerModel(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            talker = talker.to(autocast_dtype)
        talker.to_empty(device=device)

        # The top-level checkpoint interleaves Talker and CodePredictor under
        # ``talker.*``. Stream only the non-CodePredictor keys into this model.
        def talker_weights():
            for name, tensor in iter_safetensors_shards(
                self.local_dir, device=device, prefix="talker."
            ):
                if not name.startswith("talker.code_predictor."):
                    yield name.removeprefix("talker."), tensor

        loaded = load_hf_weights(
            talker,
            talker_weights(),
            stacked_params=LLAMA_STACKED_PARAMS,
        )
        self._verify_loaded(talker, loaded, "Qwen3-TTS Talker")
        talker.eval()

        # CodePredictor is small and depth-wise. It is loaded separately from
        # ``talker.code_predictor.*`` and remains replicated under Talker TP.
        with torch.device("meta"):
            code_predictor = Qwen3TTSCodePredictor(self.config)
        if autocast_dtype is not None:
            code_predictor = code_predictor.to(autocast_dtype)
        code_predictor.to_empty(device=device)
        cp_prefix = "talker.code_predictor."
        cp_weights = (
            (name.removeprefix(cp_prefix), tensor)
            for name, tensor in iter_safetensors_shards(
                self.local_dir, device=device, prefix=cp_prefix
            )
        )
        loaded = load_hf_weights(
            code_predictor,
            cp_weights,
            stacked_params=LLAMA_STACKED_PARAMS,
        )
        self._verify_loaded(
            code_predictor, loaded, "Qwen3-TTS CodePredictor"
        )
        # Piecewise CUDA Graph execution indexes all residual LM heads as one
        # tensor; consolidate after the individual checkpoint heads are loaded.
        code_predictor.consolidate_stacked_weights()
        code_predictor.eval()
        return TalkerSubmodule(talker, code_predictor, self.config)

    def _create_codec_submodule(self, device: str) -> NodeSubmodule:
        """Build the official speech-tokenizer decoder from its sub-checkpoint."""
        try:
            from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2DecoderConfig,
            )
            from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2Decoder,
            )
        except ImportError as exc:
            raise ImportError(
                "Qwen3-TTS Codec requires the 'qwen-tts' package; install "
                "M* with the qwen3_tts optional dependency"
            ) from exc

        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_tts.submodules import CodecSubmodule

        # Reuse the official decoder implementation, but keep graph scheduling,
        # chunk padding, overlap trimming, and output transport in M*.
        decoder_config = Qwen3TTSTokenizerV2DecoderConfig(
            **self.config.codec.decoder_kwargs()
        )
        with torch.device("meta"):
            decoder = Qwen3TTSTokenizerV2Decoder(decoder_config)
        decoder.to_empty(device=device)

        codec_dir = Path(self.local_dir) / "speech_tokenizer"
        prefix = "decoder."
        weights = (
            (name.removeprefix(prefix), tensor)
            for name, tensor in iter_safetensors_shards(
                codec_dir, device=device, prefix=prefix
            )
        )
        loaded = load_hf_weights(decoder, weights)
        self._verify_loaded(decoder, loaded, "Qwen3-TTS Codec")
        decoder.eval()
        return CodecSubmodule(decoder, self.config)

    def _get_subtalker_sampling(self, model_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve sampling controls for residual codec groups 1 through 15."""
        generation = self.config.generation
        do_sample = model_kwargs.get(
            "subtalker_dosample", generation.subtalker_dosample
        )
        return {
            "do_sample": do_sample,
            "temperature": (
                model_kwargs.get(
                    "subtalker_temperature", generation.subtalker_temperature
                )
                if do_sample
                else 0.0
            ),
            "top_k": model_kwargs.get("subtalker_top_k", generation.subtalker_top_k),
            "top_p": model_kwargs.get("subtalker_top_p", generation.subtalker_top_p),
        }

    @staticmethod
    def _talker_step_metadata(
        metadata: CurrentForwardConductorMetadata,
    ) -> dict[str, Any]:
        """Copy conductor-owned generation controls into worker step metadata."""
        return {
            "is_prefill": metadata.is_prefill,
            "talker_max_tokens": metadata.kwargs["talker_max_tokens"],
            "subtalker_sampling": metadata.kwargs["subtalker_sampling"],
        }
