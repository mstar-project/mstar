"""
Qwen3OmniModel: 3-partition streaming model for Qwen3-Omni-Moe.

Qwen3-Omni is a dual-AR multimodal model with a Thinker (30B-A3B MoE)
that reasons over text/audio/vision inputs and a Talker (3B-A0.3B MoE)
that converts Thinker hidden states into streaming codec tokens.  A
Code2Wav vocoder converts codec tokens to 24 kHz PCM audio.

Architecture (3 async partitions):
    Thinker  — multimodal encoder + MoE LLM (text, audio, vision prefill -> decode)
    Talker   — smaller MoE LLM that predicts codec tokens from Thinker hidden states
    Code2Wav — vocoder that converts codec tokens to audio waveform

Streaming topology:
    Thinker --[thinker_states, FixedChunkPolicy(1)]--> Talker
    Talker  --[codec_tokens,  FixedChunkPolicy(25)]--> Code2Wav

Conductor-triggered pipelined prefill (Approach C):
    After each Thinker walk completes (prefill_text, prefill_audio,
    prefill_vision, thinker_decode), the conductor sends a
    ``talker_trigger`` to the Talker partition.  During prefill each
    trigger extends the Talker KV cache with the new Thinker hidden
    states.  The final trigger (when thinker_decode starts) tells the
    Talker to sample its first codec token and transition to decode.

Text-only mode:
    When output_modalities does not include "audio", only the Thinker
    partition runs.  Talker and Code2Wav are idle.
"""

import logging
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mminf.engine.base import EngineType
from mminf.engine.kv_store import KVCacheConfig
from mminf.graph.base import DynamicLoop, GraphEdge, GraphNode, Sequential, TensorPointerInfo
from mminf.graph.special_destinations import EMIT_TO_CLIENT
from mminf.model.base import ForwardPassArgs, MAX_OUTPUT_TOKENS, Model, TensorAndMetadata
from mminf.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
from mminf.model.submodule_base import NodeSubmodule
from mminf.model.utils import Operation, WeightConverter
from mminf.streaming.chunk_policy import FixedChunkPolicy, LeftContextChunkPolicy
from mminf.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge
from mminf.utils.sampling import SamplingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    """Download (or locate) a HuggingFace snapshot and return the local path."""
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    except Exception as e:
        logger.warning("Error downloading from HuggingFace: %s", str(e))
        return repo_id
    return str(Path(local_dir))


# ---------------------------------------------------------------------------
# Qwen3OmniModel
# ---------------------------------------------------------------------------

class Qwen3OmniModel(Model):
    """Qwen3-Omni: Thinker + Talker + Code2Wav 3-partition streaming model."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf

        self.CONVERTER = [
            WeightConverter(
                source_patterns=[
                    "mlp.experts.*.gate_proj.weight",
                    "mlp.experts.*.up_proj.weight",
                ],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[
                    Operation("MergeModulelist",  dim=0),
                    Operation("Concatenate", dim=1)
                ]
            ),
            WeightConverter(
                source_patterns=["mlp.experts.*.down_proj.weight"],
                target_patterns="mlp.experts.down_proj",
                operations=[Operation("MergeModulelist",  dim=0)],
            ),
        ]

        # Load config from pretrained checkpoint
        from mminf.model.qwen3_omni.config import Qwen3OmniModelConfig

        local_dir = _resolve_local_hf_snapshot(model_path_hf, cache_dir=cache_dir)
        self.config = Qwen3OmniModelConfig.from_pretrained(local_dir)
        self.local_dir = local_dir

        # Tokenizer (Thinker uses a Qwen-family tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, cache_dir=cache_dir, trust_remote_code=True,
        )

        # Full multimodal processor: combines tokenizer + image_processor +
        # video_processor + audio feature_extractor + chat template support.
        # Used by process_prompt to build the full ChatML prompt with the
        # correct image_pad / audio_pad / video_pad expansion.
        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                local_dir, cache_dir=cache_dir, trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(
                "Could not load Qwen3-Omni AutoProcessor (%s); "
                "process_prompt will fall back to raw tokenizer.encode.",
                e,
            )
            self._processor = None

        # Lazy submodule cache -- each worker only loads what it needs
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -----------------------------------------------------------------------
    # Model ABC: KV cache config
    # -----------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        """Return separate KV cache configs for Thinker and Talker."""
        thinker_cfg = KVCacheConfig(
            num_layers=self.config.thinker_text.num_hidden_layers,
            num_kv_heads=self.config.thinker_text.num_key_value_heads,
            head_dim=self.config.thinker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.thinker_text.num_attention_heads,
            nodes=["Thinker"]
        )
        talker_cfg = KVCacheConfig(
            num_layers=self.config.talker_text.num_hidden_layers,
            num_kv_heads=self.config.talker_text.num_key_value_heads,
            head_dim=self.config.talker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.talker_text.num_attention_heads,
            nodes=["Talker"]
        )
        return [thinker_cfg, talker_cfg]

    # -----------------------------------------------------------------------
    # Model ABC: node engine types
    # -----------------------------------------------------------------------

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "audio_encoder": EngineType.ENC_DEC,
            "vision_encoder": EngineType.ENC_DEC,
            "Thinker": EngineType.AR,
            "Talker": EngineType.AR,
            "Code2Wav": EngineType.AUDIO_CODEC,
        }
    
    def get_max_talker_output_tokens(self, **model_kwargs):
        return model_kwargs.get("talker_max_output_tokens", MAX_OUTPUT_TOKENS)

    # -----------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -----------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphNode | Sequential]:
        """Define all graph walks for the 3-partition architecture.

        Thinker walks:
            prefill_text   - text token embedding + Thinker prefill
            prefill_audio  - audio feature encoding + Thinker prefill
            prefill_vision - vision feature encoding + Thinker prefill
            thinker_decode - autoregressive text token generation

        Talker walks:
            talker_prefill - prefill Talker KV cache from Thinker states
            talker_decode  - autoregressive codec token generation

        Code2Wav walks:
            code2wav_chunk - vocoder streaming decode
        """
        # -- Thinker prefill walks: process inputs and stream hidden states
        #    to the Talker partition via StreamingGraphEdge --
        prefill_text = GraphNode(
            name="Thinker",
            input_ids=["text_inputs"],
            outputs=[
                GraphEdge( # last prefill samples a token
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_states",
                    target_partition="Talker",
                ),
                # The thinker_mask tensor includes two masks: one for multimodal inputs,
                # and one for text inputs (allowing us to cut out the system prompt and
                # assistant history from the talker input)
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_mask",
                    target_partition="Talker",
                ),
            ],
        )

        prefill_audio = Sequential([
            GraphNode(
                name="audio_encoder",
                # audio_seqlens carries the original (pre-padding) length of
                # each audio clip, used by the encoder to compute attention
                # masks and output position IDs.
                input_ids=["audio_features", "audio_seqlens"],
                outputs=[GraphEdge(next_node="Thinker", name="audio_embeds")],
            ),
            GraphNode(
                name="Thinker",
                input_ids=["audio_embeds"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
        ])

        prefill_vision = Sequential([
            GraphNode(
                name="vision_encoder",
                # image_grid_thw / video_grid_thw carries the (T, H, W) grid
                # dimensions per image/video, used by the encoder to compute
                # spatial position IDs and patch counts.
                input_ids=["pixel_values", "image_grid_thw"],
                outputs=[
                    GraphEdge(next_node="Thinker", name="vision_embeds"),
                    GraphEdge(next_node="Thinker", name="deepstack")
                ],
            ),
            GraphNode(
                name="Thinker",
                input_ids=["vision_embeds", "deepstack", "video_second_per_grid", "image_grid_thw"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
        ])

        # -- Thinker decode: produces new_token (persist) + thinker_states
        #    (streaming to Talker) --
        thinker_decode = DynamicLoop(
            name="thinker_decode_loop",
            section=GraphNode(
                name="Thinker",
                input_ids=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(
                        next_node="Thinker",
                        name="text_inputs",
                        output_modality="text",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        # -- Talker prefill: receives thinker_states + talker_trigger --
        # Dual-input gating: both thinker_states from streaming and
        # talker_trigger from conductor cross-partition trigger must be
        # present for a prefill step.
        talker_prefill = GraphNode(
            name="Talker",
            input_ids=["thinker_states", "thinker_mask", "talker_trigger"],
            outputs=[],
        )

        talker_last_prefill = Sequential(
            sections=[
                GraphNode(
                    name="Talker",
                    input_ids=["thinker_states", "thinker_mask", "talker_trigger"],
                    outputs=[
                        GraphEdge(
                            next_node="Talker",
                            name="talker_input_embeds",
                            persist=True
                        ),
                        StreamingGraphEdge(
                            next_node="Code2Wav",
                            name="codec_tokens",
                            target_partition="Code2Wav",
                        ),
                    ]
                )
            ]
        )

        # -- Talker decode: autoregressive codec token generation --
        talker_decode = DynamicLoop(
            name="talker_decode_loop",
            section=Sequential(
                sections=[
                    GraphNode(
                        name="Talker",
                        input_ids=["thinker_states", "thinker_mask", "talker_input_embeds"],
                        outputs=[
                            GraphEdge(
                                next_node="Talker",
                                name="talker_input_embeds",
                                persist=True
                            ),
                            StreamingGraphEdge(
                                next_node="Code2Wav",
                                name="codec_tokens",
                                target_partition="Code2Wav",
                            ),
                        ]
                    )
                ]
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        # -- Code2Wav chunk: vocoder streaming decode --
        code2wav_chunk = GraphNode(
            name="Code2Wav",
            input_ids=["codec_tokens"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="audio_chunk",
                    output_modality="audio",
                ),
            ],
        )

        return {
            "prefill_text": prefill_text,
            "prefill_audio": prefill_audio,
            "prefill_vision": prefill_vision,
            "thinker_decode": thinker_decode,
            "talker_prefill": talker_prefill,
            "talker_last_prefill": talker_last_prefill,
            "talker_decode": talker_decode,
            "code2wav_chunk": code2wav_chunk,
        }

    # -----------------------------------------------------------------------
    # Partition API: 3-partition streaming topology
    # -----------------------------------------------------------------------

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name="Thinker",
                graph_walks={
                    "prefill_text", "prefill_audio",
                    "prefill_vision", "thinker_decode",
                },
                initial_walk="prefill_text",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name="Talker",
                graph_walks={"talker_prefill", "talker_last_prefill", "talker_decode"},
                initial_walk="talker_prefill",
                producer_partitions=["Thinker"],
            ),
            PartitionDefinition(
                name="Code2Wav",
                graph_walks={"code2wav_chunk"},
                initial_walk="code2wav_chunk",
                producer_partitions=["Talker"],
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(
            partitions=["Thinker", "Talker", "Code2Wav"],
            connections=[
                Connection(
                    from_partition="Thinker",
                    to_partition="Talker",
                    edge_name="thinker_states",
                    chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1, continue_after_done=True),
                ),
                Connection(
                    from_partition="Thinker",
                    to_partition="Talker",
                    edge_name="thinker_mask",
                    chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1, continue_after_done=True),
                ),
                Connection(
                    from_partition="Talker",
                    to_partition="Code2Wav",
                    edge_name="codec_tokens",
                    chunk_policy_factory=lambda: LeftContextChunkPolicy(
                        chunk=self.config.code2wav.codec_chunk_frames,
                        left_context=self.config.code2wav.codec_left_context_frames,
                    ),
                ),
            ],
        )

    # -----------------------------------------------------------------------
    # Model ABC: sampling config
    # -----------------------------------------------------------------------
    def get_sampling_config(
        self, node_name: str,
        model_kwargs: dict | None = None,
    )  -> SamplingConfig | None:
        if model_kwargs is None:
            model_kwargs = {}

        if node_name == "Thinker":
            temperature = model_kwargs.get("thinker_temperature", 0.7)
            top_p = model_kwargs.get("thinker_top_p", 0.9)
            return SamplingConfig(
                temperature=temperature, top_p=top_p
            )
        if node_name == "Talker":
            temperature = model_kwargs.get("talker_temperature", 0.9)
            top_k = model_kwargs.get("talker_top_k", 50)
            top_p = model_kwargs.get("talker_top_p", 1.0)
            repetition_penalty = model_kwargs.get("talker_repetition_penalty", 1.05)
            return SamplingConfig(
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty
            )
        # fallback to default config
        return SamplingConfig()

    # -----------------------------------------------------------------------
    # Model ABC: initial forward pass args
    # -----------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        audio_output = "audio" in output_modalities

        if model_kwargs is None:
            model_kwargs = {}

        if partition_name == "Thinker":
            return self._get_thinker_initial_args(
                input_modalities, output_modalities,
                input_signals, model_kwargs or {},
            )
        elif partition_name == "Talker":
            # Talker starts in prefill mode
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="talker_prefill",
                is_prefill=True,
                kwargs={
                    "audio_output": audio_output,
                    "talker_prefill_done": False,
                    "num_thinker_prefill_steps": len(input_modalities),
                    "prefill_chunks_processed": 0,
                    "voice": model_kwargs.get("voice", "Ethan"),
                    "talker_max_tokens": self.get_max_talker_output_tokens(**model_kwargs),
                },
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[GraphEdge(next_node="Talker", name="talker_trigger")] if audio_output else [],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
                step_metadata={
                    "voice": model_kwargs.get("voice", "Ethan"),
                    "talker_max_tokens": full_metadata.kwargs.get("talker_max_tokens")
                }
            )
        elif partition_name == "Code2Wav":
            # Code2Wav starts with code2wav_chunk walk but no inputs --
            # it self-triggers via StreamBuffer when codec tokens arrive.
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="code2wav_chunk",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    def _get_thinker_initial_args(
        self,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> ForwardPassArgs:
        """Build initial ForwardPassArgs for the Thinker partition.

        Constructs a prefill schedule from the input modalities, then
        begins the first walk in that schedule (always prefill_text).
        """
        audio_output = "audio" in output_modalities

        # Build prefill schedule: list of (graph_walk_name, tensor_info)
        schedule = self._build_thinker_prefill_schedule(
            input_modalities, input_signals,
        )

        first_walk = schedule[0][0] if schedule else "thinker_decode"

        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=first_walk,
            is_prefill=bool(schedule),
            kwargs={
                "prefill_schedule": schedule,
                "prefill_step": 0,
                "audio_output": audio_output,
            },
        )

        # First walk inputs
        inputs = self._get_thinker_prefill_inputs(full_metadata, input_signals)
        unpersist_tensors = sum(
            [inp.tensor_info for inp in inputs], start=[]
        )

        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={
                "is_prefill": True,
                # Tell the Thinker whether to emit thinker_states.  Text only
                # requests skip it to save cross-partition bandwidth.
                "audio_output": audio_output,
                "is_last_prefill": len(schedule) == 1
            },
        )

    def _build_thinker_prefill_schedule(
        self,
        input_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[tuple[str, dict[str, TensorPointerInfo]]]:
        """Build the sequential prefill schedule for the Thinker.

        Order: [prefill_text] + [prefill_audio if audio inputs] + [prefill_vision if vision inputs]

        Each schedule entry is ``(walk_name, {input_name: tensor_info})``,
        capturing all tensors needed by that step's first node.  For audio
        and vision walks, this includes auxiliary tensors like
        ``audio_seqlens`` and ``image_grid_thw`` that the encoder nodes
        require alongside the primary feature tensor.
        """
        schedule: list[tuple[str, dict[str, TensorPointerInfo]]] = []

        texts = input_signals.get("text_inputs", [])
        audio_features = input_signals.get("audio_features", [])
        audio_seqlens = input_signals.get("audio_seqlens", [])
        pixel_values = input_signals.get("pixel_values", [])
        image_grid_thws = input_signals.get("image_grid_thw", [])
        # video uses pixel_values_videos in HF; we accept both keys here
        pixel_values_videos = input_signals.get("pixel_values_videos", [])
        video_grid_thws = input_signals.get("video_grid_thw", [])
        video_second_per_grid = input_signals.get("video_second_per_grid", [])

        text_idx = audio_idx = vision_idx = video_idx = 0
        for mod in input_modalities:
            if mod == "text":
                if text_idx < len(texts):
                    schedule.append((
                        "prefill_text",
                        {"text_inputs": texts[text_idx]},
                    ))
                    text_idx += 1
            elif mod == "audio":
                if audio_idx < len(audio_features):
                    entry: dict[str, TensorPointerInfo] = {
                        "audio_features": audio_features[audio_idx],
                    }
                    if audio_idx < len(audio_seqlens):
                        entry["audio_seqlens"] = audio_seqlens[audio_idx]
                    schedule.append(("prefill_audio", entry))
                    audio_idx += 1
            elif mod == "image":
                if vision_idx < len(pixel_values):
                    entry = {"pixel_values": pixel_values[vision_idx]}
                    if vision_idx < len(image_grid_thws):
                        entry["image_grid_thw"] = image_grid_thws[vision_idx]
                    schedule.append(("prefill_vision", entry))
                    vision_idx += 1
            elif mod == "video":
                # Video uses pixel_values_videos + video_grid_thw, but the
                # graph node still consumes them under the "pixel_values" /
                # "image_grid_thw" input names (the vision encoder is shared).
                if video_idx < len(pixel_values_videos):
                    entry = {"pixel_values": pixel_values_videos[video_idx]}
                    if video_idx < len(video_grid_thws):
                        entry["image_grid_thw"] = video_grid_thws[video_idx]
                    if video_idx < len(video_second_per_grid):
                        entry["video_second_per_grid"] = video_second_per_grid[video_idx]
                    schedule.append(("prefill_vision", entry))
                    video_idx += 1

        return schedule

    def _get_thinker_prefill_inputs(
        self,
        metadata: CurrentForwardConductorMetadata,
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        """Construct input GraphEdges for the current Thinker prefill step.

        Each schedule entry maps an ``(walk_name, {input_name: tensor_info})``.
        We emit one GraphEdge per input so that auxiliary tensors like
        ``audio_seqlens`` and ``image_grid_thw`` reach the encoder node
        alongside the primary feature tensor.
        """
        schedule = metadata.kwargs["prefill_schedule"]
        step = metadata.kwargs["prefill_step"]
        walk_name, tensor_dict = schedule[step]

        # Determine the target node — for audio/vision, the first node in
        # the Sequential walk is the encoder (not the Thinker).
        if walk_name == "prefill_text":
            target_node = "Thinker"
        elif walk_name == "prefill_audio":
            target_node = "audio_encoder"
        elif walk_name == "prefill_vision":
            target_node = "vision_encoder"
        else:
            raise ValueError(f"Unrecognized prefill walk: {walk_name}")

        edges: list[GraphEdge] = []
        for input_name, tensor_info in tensor_dict.items():
            if input_name == "video_second_per_grid":
                continue # goes directly to Thinker
            edge = GraphEdge(next_node=target_node, name=input_name)
            edge.tensor_info = [tensor_info]
            edges.append(edge)

        if walk_name == "prefill_vision":
            for key in ["image_grid_thw", "video_second_per_grid"]:
                edge = GraphEdge(next_node="Thinker", name=key)
                if key in tensor_dict:
                    edge.tensor_info = [tensor_dict[key]]
                edges.append(edge)
        return edges

    # -----------------------------------------------------------------------
    # Model ABC: partition forward pass args (STATE MACHINE)
    # -----------------------------------------------------------------------

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        if partition_name == "Thinker":
            return self._get_thinker_forward(
                partition_metadata, persist_signals, new_tokens,
            )
        elif partition_name == "Talker":
            return self._get_talker_forward(
                partition_metadata, persist_signals, new_tokens,
                incoming_connections,
            )
        elif partition_name == "Code2Wav":
            conn = incoming_connections[0] if incoming_connections else None
            return self._get_code2wav_forward(
                partition_metadata, conn,
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    # -- Thinker state machine ---------------------------------------------

    def _get_thinker_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
    ) -> ForwardPassArgs:
        """Thinker partition state machine.

        1. Build prefill schedule: [prefill_text] + [prefill_audio] + [prefill_vision]
        2. Pop walks from schedule until done
        3. Transition to thinker_decode
        4. Each decode step: check new_token for EOS (im_end_token_id)
        5. On EOS: request_done=True for Thinker
        """

        if metadata.is_prefill:
            # Advance prefill schedule
            step = metadata.kwargs["prefill_step"] + 1
            schedule = metadata.kwargs["prefill_schedule"]

            if step < len(schedule):
                # More prefill steps remaining
                metadata.kwargs["prefill_step"] = step
                metadata.graph_walk = schedule[step][0]
            else:
                # All prefill done -- transition to thinker_decode
                metadata.is_prefill = False
                metadata.graph_walk = "thinker_decode"

        elif metadata.graph_walk == "thinker_decode":
            # if the decode loop returns to conductor, the thinker is fully done
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )


        if metadata.is_prefill:
            # Still in prefill -- delegate to _get_thinker_prefill_inputs
            # which handles the (walk_name, {input_name: tensor_info}) schedule
            # entry format and emits one GraphEdge per input (so auxiliary
            # tensors like image_grid_thw / audio_seqlens reach the encoder
            # alongside the primary feature tensor).
            schedule = metadata.kwargs["prefill_schedule"]
            step = metadata.kwargs["prefill_step"]
            is_last_prefill = (step == len(schedule) - 1)
            inputs = self._get_thinker_prefill_inputs(metadata, persist_signals)
        else:
            # Decode: previous token feeds back as text_inputs
            is_last_prefill = False
            edge = GraphEdge(next_node="Thinker", name="text_inputs")
            edge.tensor_info = persist_signals.get("new_token", [])
            inputs = [edge]

        unpersist_tensors = sum(
            [inp.tensor_info for inp in inputs], start=[]
        )

        step_metadata = {
            "is_prefill": metadata.is_prefill,
            "is_last_prefill": is_last_prefill,
            # Persist the audio_output flag across every Thinker step so
            # the submodule can gate thinker_states emission.  Default True
            # for backwards compatibility with callers that never set it.
            "audio_output": metadata.kwargs.get("audio_output", True),
        }

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=step_metadata,
        )

    # -- Talker state machine ----------------------------------------------

    def _get_talker_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Talker partition state machine.

        1. While prefill: return empty inputs (wait for cross-partition trigger)
           - When trigger arrives with is_last_prefill=False:
             extend KV cache only, no outputs
           - When trigger arrives with is_last_prefill=True:
             sample first codec token, produce all_codes
        2. After last prefill produces all_codes: transition to talker_decode
           - Set graph_walk="talker_decode", is_prefill=False
           - Return all_codes as input edge (conductor-driven)
        3. Each decode step: check all_codes for codec_eos
           - If codec_eos: request_done=True for Talker
           - Else: return all_codes as input again (loop)
        """
        if metadata.graph_walk == "talker_prefill":
            metadata.kwargs["prefill_chunks_processed"] += 1
            is_last_prefill = metadata.kwargs["num_thinker_prefill_steps"] == \
                 metadata.kwargs["prefill_chunks_processed"]
            metadata.graph_walk = "talker_last_prefill" if is_last_prefill else "talker_prefill"
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[GraphEdge(next_node="Talker", name="talker_trigger")],
                unpersist_tensors=[],
                step_metadata={
                    "is_prefill": True,
                    # voice is used for the last prefill
                    "voice": metadata.kwargs.get("voice", "Ethan"),
                    "talker_max_tokens": metadata.kwargs.get("talker_max_tokens")
                },
            )
        elif metadata.graph_walk == "talker_last_prefill":
            metadata.is_prefill = False
            metadata.graph_walk = "talker_decode"
            metadata.kwargs["talker_prefill_done"] = True

            # Feed talker_input_embeds back as input for first decode step
            edge = GraphEdge(next_node="Talker", name="talker_input_embeds")
            edge.tensor_info = persist_signals["talker_input_embeds"]
            inputs = [edge]
            unpersist_tensors = sum(
                [inp.tensor_info for inp in inputs], start=[]
            )

            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=inputs,
                unpersist_tensors=unpersist_tensors,
                step_metadata={
                    "is_prefill": False,
                    "talker_max_tokens": metadata.kwargs.get("talker_max_tokens")
                },
            )

        elif metadata.graph_walk == "talker_decode":
            # If the decode dynamic loop reaches the conductor, we can end the request.
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        raise ValueError(
            f"Talker in unexpected state: walk={metadata.graph_walk!r}, "
            f"is_prefill={metadata.is_prefill}"
        )

    # -- Code2Wav state machine --------------------------------------------

    def _get_code2wav_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        conn: StreamingConnectionState | None,
    ) -> ForwardPassArgs:
        """Code2Wav partition: streaming vocoder, self-triggered by StreamBuffer.

        Same pattern as Orpheus SNAC -- the conductor just tracks whether
        there are more codec tokens to process.
        """
        chunk_size = self.config.code2wav.codec_chunk_frames
        token_count = conn.token_count if conn else 0
        consumed = conn.consumed_count if conn else 0
        producer_done = conn.producer_done if conn else False

        metadata.graph_walk = "code2wav_chunk"

        available = token_count - consumed

        # Nothing left to decode
        if available <= 0 and producer_done:
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        step_metadata = {"consumed_tokens": chunk_size}

        # Check if this is the last chunk
        new_consumed = consumed + chunk_size
        remaining_after = token_count - new_consumed
        is_last = producer_done and remaining_after < chunk_size

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[],
            unpersist_tensors=[],
            step_metadata=step_metadata,
            request_done=is_last,
        )

    # -----------------------------------------------------------------------
    # Model ABC: prompt processing
    # -----------------------------------------------------------------------

    def load_video(
        self, filepath: str, device: str
    ) -> TensorAndMetadata:
        # TODO: support audio in video
        from qwen_omni_utils.v2_5.vision_process import fetch_video
        video_input, video_sample_fps = fetch_video(
            {"video": filepath},
            return_video_sample_fps=True,
            image_patch_size=14,
            return_video_metadata=False
        )
        return TensorAndMetadata(
            data=video_input.to(device),
            metadata=dict(
                video_sample_fps=video_sample_fps
            )
        )

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        input_metadata: dict[str, dict] = {},
        **kwargs,
    ) -> NameToTensorList:
        """Build the full ChatML prompt + derived multimodal tensors.

        Uses HF's full ``AutoProcessor`` (combines tokenizer + image_processor
        + video_processor + feature_extractor + chat template) to:

        1. Build a ChatML-formatted prompt from ``prompt`` and any
           multimodal inputs in ``tensors``.
        2. Apply ``add_generation_prompt=True`` so the model receives the
           ``<|im_start|>assistant\\n`` suffix and knows to start the
           assistant response.
        3. Run the image_processor / feature_extractor on the raw modality
           tensors to produce ``pixel_values`` / ``image_grid_thw`` /
           ``audio_features`` / ``audio_seqlens``.
        4. Expand the single ``<|image_pad|>`` / ``<|audio_pad|>`` /
           ``<|video_pad|>`` placeholder in the tokenized text to N copies
           where N = number of patches after spatial merge (this is what
           ``Qwen3OmniMoeProcessor.replace_multimodal_special_tokens`` does
           internally).

        The result has ``text_inputs`` containing the FULL templated +
        expanded token IDs, plus the per-modality tensor outputs needed by
        the Thinker's prefill walks.
        """
        result: NameToTensorList = {}

        if tensors is None:
            tensors = {}

        # ----- Convert raw modality tensors to PIL/numpy form for HF -----
        raw_image_inputs = tensors.get("image_inputs", [])
        raw_audio_inputs = tensors.get("audio_inputs", [])
        raw_video_inputs = tensors.get("video_inputs", [])

        pil_images: list = []
        for img in raw_image_inputs:
            # data_worker.py provides images as (C, H, W) float32 in [0, 1]
            # on the GPU.  HF processors expect PIL/numpy uint8 (H, W, C)
            # in [0, 255] -- otherwise the default do_rescale=True double-
            # rescales and the model sees a near-zero (essentially black)
            # tensor regardless of the actual image content.
            if img.dtype.is_floating_point:
                img_u8 = (img * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                img_u8 = img
            if img_u8.dim() == 3 and img_u8.shape[0] in (1, 3):
                img_u8 = img_u8.permute(1, 2, 0)  # CHW -> HWC
            pil_images.append(img_u8.cpu().contiguous().numpy())

        np_audios: list = []
        for waveform in raw_audio_inputs:
            np_audios.append(waveform.cpu().numpy())

        # ----- Preferred path: text-only chat template + separate modality processors -----
        #
        # We deliberately DO NOT include image/audio/video content blocks in
        # the messages list passed to apply_chat_template.  HF's chat template
        # would otherwise insert ``<|vision_start|><|image_pad|>...<|vision_end|>``
        # placeholders into text_inputs, which we don't want because:
        #
        #   1. Our prefill_vision / prefill_audio walks already wrap the
        #      modality content in their own start/end tokens before pushing
        #      it into the Thinker's KV cache.  Having the same wrapping in
        #      text_inputs would make the model see each modality twice
        #      (once as actual encoder embeddings via the modality walks,
        #      once as generic token embeddings via prefill_text), which is
        #      noise.
        #
        #   2. Unlike HF's single-shot prefill (which masked-scatter's the
        #      vision embeds INTO the placeholder positions in input_embeds),
        #      our multi-walk prefill builds up the same final KV cache via
        #      sequential walks.  The modality placeholders in text_inputs
        #      would never be replaced by real content in our flow.
        #
        # Functionally, both approaches end up with the same set of
        # embeddings in the KV cache (text + modality content).  Stripping
        # the placeholders avoids noise from the unfilled embeddings.
        # if self._processor is not None:
            # try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Qwen, a virtual human developed by the "
                    "Qwen team, Alibaba Group, capable of perceiving "
                    "auditory and visual inputs, as well as generating "
                    "text and speech."
                ),
            },
        ]
        if prompt is not None:
            messages.append(
                {"role": "user", "content": prompt},
            )

        # apply_chat_template with TEXT-ONLY content -> no modality
        # placeholders are inserted.  add_generation_prompt=True
        # appends the trailing ``<|im_start|>assistant\n`` so the
        # model knows to start the assistant response.
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            text, return_tensors="pt"
        )["input_ids"][0]
        result["text_inputs"] = [input_ids]

        result["pixel_values"] = []
        result["image_grid_thw"] = []
        result["audio_seqlens"] = []
        result["audio_features"] = []
        result["video_second_per_grid"] = []
        result["video_grid_thw"] = []
        result["pixel_values_videos"] = []

        # Run image_processor / feature_extractor SEPARATELY for the
        # modality outputs.  These don't touch text_inputs.
        for img in pil_images:
            img_proc = self._processor.image_processor
            img_out = img_proc(images=[img], return_tensors="pt")
            result["pixel_values"].append(img_out["pixel_values"])
            result["image_grid_thw"] += img_out["image_grid_thw"]

        for audio in np_audios:
            feat_extractor = self._processor.feature_extractor
            sr = getattr(feat_extractor, "sampling_rate", 16000)
            aud_out = feat_extractor(
                audio, sampling_rate=sr,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt"
            )
            aud_out["input_features"] = (
                aud_out["input_features"]
                .permute(0, 2, 1)[aud_out["attention_mask"].bool()]
                .permute(1, 0)
            )
            result["audio_seqlens"].append(
                aud_out["attention_mask"].sum(-1).to(torch.long)
            )
            result["audio_features"].append(
                aud_out["input_features"]
            )

        # Video uses the video_processor; left as TODO since our
        # prefill_vision walk doesn't yet handle video frame stacks.
        for video, meta in zip(raw_video_inputs, input_metadata.get("video_inputs", []), strict=True):
            fps = meta.get(
                "video_sample_fps", 2.0
            )
            vid_out = self._processor.video_processor(
                videos=video,
                size={
                    "shortest_edge": 128 * 32 * 32,
                    "longest_edge": 768 * 32 * 32,
                }
            )
            result["video_second_per_grid"].append(
                torch.tensor([self._processor.video_processor.temporal_patch_size / fps])
            )
            result["video_grid_thw"] += vid_out["video_grid_thw"]
            result["pixel_values_videos"].append(vid_out["pixel_values_videos"])

        return result

    # -----------------------------------------------------------------------
    # Model ABC: postprocess
    # -----------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
    ) -> bytes:
        if modality == "text":
            detok = self.tokenizer.decode(output)
            return detok.encode("utf-8")
        elif modality == "audio":
            if output.numel() == 0:
                return b""
            return output.cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Qwen3-Omni: {modality!r}")

    # -----------------------------------------------------------------------
    # Model ABC: submodule loading
    # -----------------------------------------------------------------------

    def get_submodule(self, node_name: str, device: str = "cpu") -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        logger.info("Successfully loaded Qwen3-Omni submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule

        # W3: If the Thinker was just loaded and the Talker already exists
        # (but TTS embeds were not initialized because Thinker wasn't
        # available at Talker creation time), initialize them now.
        if node_name == "Thinker":
            talker_sub = self._submodule_cache.get("Talker")
            if (
                talker_sub is not None
                and hasattr(talker_sub, '_tts_pad_embed_cached')
                and talker_sub._tts_pad_embed_cached is None
                and hasattr(submodule, 'model')
            ):
                try:
                    talker_sub.init_tts_embeds(submodule.model.embed_tokens)
                except Exception as e:
                    logger.warning(
                        "Deferred TTS embed init failed: %s", e,
                    )

        return submodule

    def _create_submodule(self, node_name: str, device: str) -> NodeSubmodule | None:
        if node_name == "Thinker":
            return self._create_thinker_submodule(device)
        elif node_name == "Talker":
            return self._create_talker_submodule(device)
        elif node_name == "Code2Wav":
            return self._create_code2wav_submodule(device)
        elif node_name == "audio_encoder":
            return self._create_audio_encoder_submodule(device)
        elif node_name == "vision_encoder":
            return self._create_vision_encoder_submodule(device)
        return None

    def _create_thinker_submodule(self, device: str) -> NodeSubmodule:
        from mminf.model.qwen3_omni.components.thinker import Qwen3OmniThinkerModel
        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        with torch.device("meta"):
            thinker_model = Qwen3OmniThinkerModel(self.config)

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[
                ModuleAndPrefix(thinker_model, prefix="thinker"),
            ],
            device=device,
            conv=self.CONVERTER
        )

        thinker_model.eval()
        thinker_model.set_qkv_proj_weights()

        # Return a placeholder -- the actual ThinkerSubmodule class will be
        # implemented in a separate submodules.py file.
        from mminf.model.qwen3_omni.submodules import ThinkerSubmodule
        return ThinkerSubmodule(
            thinker_model=thinker_model,
            config=self.config,
        )

    def _create_talker_submodule(self, device: str) -> NodeSubmodule:
        from mminf.model.qwen3_omni.components.talker import Qwen3OmniTalkerModel
        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        with torch.device("meta"):
            talker_model = Qwen3OmniTalkerModel(self.config)

        # Talker weights: text_projection, hidden_projection, codec_head,
        # codec_embedding, and the transformer layers are all under "talker."
        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[
                ModuleAndPrefix(talker_model, prefix="talker"),
            ],
            device=device,
            conv=self.CONVERTER
        )
        talker_model.eval()

        with torch.device("meta"):
            code_predictor = Qwen3OmniCodePredictor(self.config)
        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[
                ModuleAndPrefix(code_predictor, prefix="talker.code_predictor"),
            ],
            device=device,
        )
        code_predictor.consolidate_stacked_weights()
        code_predictor.set_qkv_proj_weights()
        code_predictor.eval()

        talker_model.set_qkv_proj_weights()

        from mminf.model.qwen3_omni.submodules import TalkerSubmodule
        talker_sub = TalkerSubmodule(
            talker_model=talker_model,
            code_predictor=code_predictor,
            config=self.config,
        )

        # W3: Pre-compute TTS special embeddings using the Thinker's
        # embedding table.  The HF reference computes:
        #   tts_pad_embed = talker.text_projection(thinker.embed_tokens(pad_id))
        #   tts_bos_embed = talker.text_projection(thinker.embed_tokens(bos_id))
        #   tts_eos_embed = talker.text_projection(thinker.embed_tokens(eos_id))
        #
        # Two cases:
        #
        # 1. Colocated (Thinker + Talker on same worker): grab the
        #    Thinker submodule's already-loaded embed_tokens directly.
        #    Zero extra memory.
        #
        # 2. Disaggregated (Talker on a different worker than Thinker):
        #    load JUST the embed_tokens layer from the checkpoint, use
        #    it to compute the 3 projected TTS embeds (~12 KB cached on
        #    the Talker submodule), and immediately discard the
        #    embedding layer to free its memory.
        thinker_sub = self._submodule_cache.get("Thinker")
        if thinker_sub is not None and hasattr(thinker_sub, "model"):
            # Colocated: reuse the already-loaded embed_tokens
            try:
                embed_tokens = thinker_sub.model.model.embed_tokens
                talker_sub.init_tts_embeds(embed_tokens)
            except Exception as e:
                logger.warning(
                    "Could not init TTS embeds from colocated Thinker "
                    "embed_tokens: %s", e,
                )
        else:
            # Disaggregated: load embed_tokens temporarily from the checkpoint
            logger.info(
                "Thinker submodule not loaded on this worker; loading "
                "embed_tokens temporarily to compute Talker TTS special embeds."
            )
            text_config = self.config.thinker_text
            embed_tokens = torch.nn.Embedding(
                text_config.vocab_size, text_config.hidden_size,
            )
            try:
                load_weights_from_hf_shards(
                    repo_dir=self.local_dir,
                    modules=[ModuleAndPrefix(
                        embed_tokens, prefix="thinker.model.embed_tokens"
                    )],
                    device=device,
                )
                talker_sub.init_tts_embeds(embed_tokens)
            except Exception as e:
                logger.warning(
                    "Failed to load Thinker embed_tokens for TTS embeds; "
                    "Talker will use zero fallback (degraded audio quality): %s",
                    e,
                )
            finally:
                # Free the temporary embedding layer (~620 MB).  The 12 KB
                # of cached projected embeds remain on the Talker submodule.
                del embed_tokens

        return talker_sub

    def _create_code2wav_submodule(self, device: str) -> NodeSubmodule:
        # Code2Wav is the vocoder that converts codec tokens to audio waveform.
        # The actual model class will be defined in components.
        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards
        from mminf.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav

        # The vocoder is dominated by Conv1d/ConvTranspose1d at small channel
        # counts where cuDNN's default heuristic picks a sub-optimal algo.
        # benchmark=True autotunes per shape on the warm-up call, before
        # CUDA-graph capture, so the chosen algo is baked into the graph.
        torch.backends.cudnn.benchmark = True

        code2wav_model = Qwen3OmniMoeCode2Wav(self.config.code2wav)
        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[
                ModuleAndPrefix(code2wav_model, prefix="code2wav"),
            ],
            device=device,
        )
        code2wav_model.eval()
        code2wav_model.consolidate()

        from mminf.model.qwen3_omni.submodules import Code2WavSubmodule
        return Code2WavSubmodule(
            code2wav_model=code2wav_model,
            config=self.config,
        )

    def _create_audio_encoder_submodule(self, device: str) -> NodeSubmodule:
        """Load the audio encoder (AuT) from HF weights."""
        # Reuse HF audio encoder directly (Whisper-style, not perf-critical)
        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )

        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        # Load config only (no weights)
        config = AutoConfig.from_pretrained(
            self.local_dir,
            trust_remote_code=True,
        )

        # This should be a Qwen3OmniMoeConfig
        audio_config = config.thinker_config.audio_config

        # Build the audio encoder from config.
        # IMPORTANT: pass attn_implementation="flash_attention_2" so the
        # encoder uses the cu_seqlens FA2 path. With the HF default
        # (which resolves to "sdpa"), Qwen3OmniMoeAudioAttention runs
        # SDPA on the full packed sequence (no per-segment fusion),
        # which is significantly slower than FA2's varlen path.
        audio_encoder = Qwen3OmniMoeAudioEncoder._from_config(
            audio_config, attn_implementation="flash_attention_2"
        )

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[ModuleAndPrefix(audio_encoder, prefix="thinker.audio_tower")],
            device=device,
        )
        audio_encoder.eval()

        from mminf.model.qwen3_omni.submodules import AudioEncoderSubmodule
        return AudioEncoderSubmodule(audio_encoder=audio_encoder, config=self.config)

    def _create_vision_encoder_submodule(self, device: str) -> NodeSubmodule:
        """Load the vision encoder (SigLIP2 ViT) from HF weights."""
        # Reuse HF vision encoder directly
        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeVisionEncoder,
        )

        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        # Load full config (no weights)
        config = AutoConfig.from_pretrained(
            self.local_dir,
            trust_remote_code=True,
        )

        # Extract the vision sub-config
        vision_config = config.thinker_config.vision_config

        # Build the vision encoder.
        # CRITICAL: pass attn_implementation="flash_attention_2". Without
        # this, vision_config._attn_implementation defaults to None and is
        # resolved to "sdpa" at runtime (modeling_utils.py:1889). With
        # "sdpa", Qwen3OmniMoeVisionAttention.forward falls into the
        # per-segment Python loop (modeling_qwen3_omni_moe.py:892-913),
        # which issues N sequential attention calls per layer for an
        # N-frame video. This causes the 10× V2T/V2S TTFT regression vs
        # vllm-omni. With "flash_attention_2", a single varlen FA2 call
        # per layer handles all frames at once via cu_seqlens.
        vision_encoder = Qwen3OmniMoeVisionEncoder._from_config(
            vision_config, attn_implementation="flash_attention_2"
        )

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[ModuleAndPrefix(vision_encoder, prefix="thinker.visual")],
            device=device,
        )
        vision_encoder.eval()

        from mminf.model.qwen3_omni.submodules import VisionEncoderSubmodule
        return VisionEncoderSubmodule(vision_encoder=vision_encoder, config=self.config)
