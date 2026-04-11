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
from mminf.graph.base import GraphEdge, GraphNode, Sequential, TensorPointerInfo
from mminf.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mminf.model.base import ForwardPassArgs, Model, NodeSubmodule
from mminf.model.qwen3_omni.components.code2wav import Qwen3OmniCode2Wav
from mminf.model.utils import Operation, WeightConverter
from mminf.streaming.chunk_policy import FixedChunkPolicy, SlidingWindowChunkPolicy
from mminf.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

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

        # Lazy submodule cache -- each worker only loads what it needs
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -----------------------------------------------------------------------
    # Model ABC: KV cache config
    # -----------------------------------------------------------------------

    def get_kv_cache_config(self) -> dict[str, KVCacheConfig]:
        """Return separate KV cache configs for Thinker and Talker."""
        thinker_cfg = KVCacheConfig(
            num_layers=self.config.thinker_text.num_hidden_layers,
            num_kv_heads=self.config.thinker_text.num_key_value_heads,
            head_dim=self.config.thinker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.thinker_text.num_attention_heads,
        )
        talker_cfg = KVCacheConfig(
            num_layers=self.config.talker_text.num_hidden_layers,
            num_kv_heads=self.config.talker_text.num_key_value_heads,
            head_dim=self.config.talker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.talker_text.num_attention_heads,
        )
        return {
            "Thinker": thinker_cfg,
            "Talker": talker_cfg,
        }

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
                    is_new_token=True,
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_states",
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
                        is_new_token=True,
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
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
                outputs=[GraphEdge(next_node="Thinker", name="vision_embeds")],
            ),
            GraphNode(
                name="Thinker",
                input_ids=["vision_embeds"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        is_new_token=True,
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                ],
            ),
        ])

        # -- Thinker decode: produces new_token (persist) + thinker_states
        #    (streaming to Talker) --
        thinker_decode = GraphNode(
            name="Thinker",
            input_ids=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    is_new_token=True,
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_states",
                    target_partition="Talker",
                ),
            ],
        )

        # -- Talker prefill: receives thinker_states + talker_trigger --
        # Dual-input gating: both thinker_states from streaming and
        # talker_trigger from conductor cross-partition trigger must be
        # present for a prefill step.
        talker_prefill = GraphNode(
            name="Talker",
            input_ids=["thinker_states", "talker_trigger"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="new_token",
                    is_new_token=True,
                    persist=True,
                ),
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="all_codes",
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="Code2Wav",
                    name="codec_tokens",
                    target_partition="Code2Wav",
                ),
            ],
        )

        # -- Talker decode: autoregressive codec token generation --
        talker_decode = GraphNode(
            name="Talker",
            input_ids=["all_codes", "thinker_states"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="all_codes",
                    is_new_token=True,
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="Code2Wav",
                    name="codec_tokens",
                    target_partition="Code2Wav",
                ),
            ],
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
                graph_walks={"talker_prefill", "talker_decode"},
                initial_walk=None,  # triggered by conductor after Thinker walks
                producer_partitions=["Thinker"],
            ),
            PartitionDefinition(
                name="Code2Wav",
                graph_walks={"code2wav_chunk"},
                initial_walk=None,  # self-triggered via StreamBuffer
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
                    from_partition="Talker",
                    to_partition="Code2Wav",
                    edge_name="codec_tokens",
                    chunk_policy_factory=lambda: SlidingWindowChunkPolicy(
                        window=self.config.code2wav.chunk_size + self.config.code2wav.left_context_size,
                        stride=self.config.code2wav.chunk_size
                    ),
                ),
            ],
        )

    # -----------------------------------------------------------------------
    # Conductor-triggered pipelined prefill (Approach C)
    # -----------------------------------------------------------------------

    def get_consumer_partition_triggers(
        self,
        completed_partition: str,
        completed_walk: str,
        all_partition_states: dict,
        persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> dict[str, ForwardPassArgs]:
        """Send talker_trigger to Talker after each Thinker walk completes.

        Called by the conductor after _process_done_forward for the completed
        partition.  While the Talker is still in prefill mode, each Thinker
        walk completion triggers a Talker prefill step:

        - prefill_text / prefill_audio / prefill_vision completions:
          is_last_prefill=False  (extend KV cache, no codec output)
        - thinker_decode start (first decode step completion):
          is_last_prefill=True   (sample first codec token, transition to decode)

        Once the Talker transitions to decode, no more triggers are sent
        (the Talker self-drives via its own decode loop).
        """
        if completed_partition != "Thinker":
            return {}

        # Only trigger Talker while it is still in prefill mode
        talker_state = all_partition_states.get("Talker")
        if talker_state is None or talker_state.is_done:
            return {}

        talker_metadata = talker_state.metadata
        if not talker_metadata.is_prefill:
            # Talker has already transitioned to decode -- no more triggers
            return {}

        # Check if the Talker partition has audio output enabled
        if "audio" not in talker_metadata.output_modalities:
            return {}

        # Race condition guard (Comment 11): once we've already sent the
        # "last prefill" trigger, don't send another.  This handles the
        # case where the Thinker finishes multiple decode steps before the
        # Talker transitions from prefill to decode.  Without this guard,
        # each additional thinker_decode completion would fire another
        # is_last_prefill=True trigger, causing duplicate codec sampling.
        if talker_metadata.kwargs.get("_last_prefill_sent", False):
            return {}

        # Determine if this is the last prefill trigger.
        # The last prefill trigger is sent when the Thinker's first decode
        # step completes (completed_walk == "thinker_decode").
        is_last_prefill = (completed_walk == "thinker_decode")

        # Persist the "last prefill sent" flag into the Talker's own
        # metadata so subsequent triggers skip (handled by the conductor,
        # which updates target_pstate.metadata from trigger_fwd_args).
        new_talker_kwargs = {
            **talker_metadata.kwargs,
            "is_last_prefill": is_last_prefill,
        }
        if is_last_prefill:
            new_talker_kwargs["_last_prefill_sent"] = True

        trigger_metadata = CurrentForwardConductorMetadata(
            input_modalities=talker_metadata.input_modalities,
            output_modalities=talker_metadata.output_modalities,
            graph_walk="talker_prefill",
            is_prefill=True,
            kwargs=new_talker_kwargs,
        )

        trigger_edge = GraphEdge(next_node="Talker", name="talker_trigger")

        # Determine projection walk_name for the Talker (W2):
        # The Talker uses walk_name to decide text_projection vs
        # hidden_projection for the incoming Thinker states.
        #   - "prefill_text"    -> all text tokens    -> text_projection
        #   - "prefill_audio"   -> audio embeddings   -> hidden_projection
        #   - "prefill_vision"  -> vision embeddings   -> hidden_projection
        #   - "thinker_decode"  -> text decode tokens  -> text_projection
        projection_walk_name = completed_walk

        return {
            "Talker": ForwardPassArgs(
                full_metadata=trigger_metadata,
                inputs=[trigger_edge],
                unpersist_tensors=[],
                step_metadata={
                    "is_prefill": True,
                    "is_last_prefill": is_last_prefill,
                    "sample_token": is_last_prefill,
                    "walk_name": projection_walk_name,
                },
            ),
        }

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

        if partition_name == "Thinker":
            return self._get_thinker_initial_args(
                input_modalities, output_modalities,
                input_signals, model_kwargs or {},
            )
        elif partition_name == "Talker":
            # Talker starts in prefill mode, waiting for cross-partition trigger.
            # No initial inputs -- the conductor triggers it after each Thinker walk.
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="talker_prefill",
                is_prefill=True,
                kwargs={
                    "audio_output": audio_output,
                    "talker_prefill_done": False,
                },
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
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
                "temperature": model_kwargs.get("temperature", 0.7),
                "top_p": model_kwargs.get("top_p", 0.9),
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
                "temperature": full_metadata.kwargs["temperature"],
                "top_p": full_metadata.kwargs["top_p"],
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
            edge = GraphEdge(next_node=target_node, name=input_name)
            edge.tensor_info = [tensor_info]
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
        request_done = False

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
            # Check for EOS in newly generated tokens
            tokens = new_tokens.get("new_token", [])
            for t in tokens:
                if t == self.config.im_end_token_id:
                    request_done = True
                    break

        # Build inputs for next step
        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        is_last_prefill = True
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
            edge = GraphEdge(next_node="Thinker", name="text_inputs")
            edge.tensor_info = persist_signals.get("new_token", [])
            inputs = [edge]

        unpersist_tensors = sum(
            [inp.tensor_info for inp in inputs], start=[]
        )

        step_metadata = {
            "is_prefill": metadata.is_prefill,
            "temperature": metadata.kwargs.get("temperature", 0.7),
            "top_p": metadata.kwargs.get("top_p", 0.9),
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
        request_done = False

        if metadata.is_prefill:
            # Talker is in prefill mode, waiting for conductor triggers.
            # The actual work happens when get_consumer_partition_triggers()
            # fires from the Thinker side.  Here we just check if the
            # last prefill has been completed (the trigger sets this flag).
            is_last_prefill = metadata.kwargs.get("is_last_prefill", False)

            if is_last_prefill:
                # Last prefill done -- transition to talker_decode.
                # The talker_prefill walk should have produced all_codes
                # as its first codec token output.
                metadata.is_prefill = False
                metadata.graph_walk = "talker_decode"
                metadata.kwargs["talker_prefill_done"] = True

                # Feed all_codes back as input for first decode step
                edge = GraphEdge(next_node="Talker", name="all_codes")
                edge.tensor_info = persist_signals.get("all_codes", [])
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
                        "is_last_prefill": False,
                    },
                )
            else:
                # Not the last prefill -- just extend KV cache.
                # Return empty inputs; the trigger ForwardPassArgs from
                # get_consumer_partition_triggers() drives the actual work.
                return ForwardPassArgs(
                    full_metadata=metadata,
                    inputs=[],
                    unpersist_tensors=[],
                    step_metadata={
                        "is_prefill": True,
                        "is_last_prefill": False,
                    },
                )

        elif metadata.graph_walk == "talker_decode":
            # Decode loop: check only the layer-0 code (first element) for
            # codec EOS.  Higher codebook layers (1-31) are residual codes
            # and should not be compared against the EOS token ID.
            tokens = new_tokens.get("all_codes", [])
            codec_eos = self.config.talker.codec_eos_token_id
            if tokens and tokens[0] == codec_eos:
                request_done = True

            if request_done:
                return ForwardPassArgs(
                    full_metadata=metadata,
                    inputs=[],
                    unpersist_tensors=[],
                    request_done=True,
                )

            # Feed all_codes back for next decode step
            edge = GraphEdge(next_node="Talker", name="all_codes")
            edge.tensor_info = persist_signals.get("all_codes", [])
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
                },
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
        chunk_size = self.config.code2wav.chunk_size
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

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Tokenize text and compute derived multimodal tensors.

        The data worker loads raw modality tensors (``image_inputs``,
        ``audio_inputs``, ``video_inputs``) from ``file_paths`` and passes
        them in via the ``tensors`` dict.  This method:
          1. Tokenizes ``prompt`` to produce ``text_inputs``.
          2. For each modality present in ``tensors``, computes derived
             tensors needed by the Thinker's preprocess:
             - ``image_inputs`` -> ``pixel_values`` + ``image_grid_thw``
             - ``audio_inputs`` -> ``audio_features`` + ``audio_seqlens``
             - ``video_inputs`` -> ``pixel_values_videos`` + ``video_grid_thw``

        Returns only the NEW keys to merge into the data worker's tensor
        dict (the existing ``*_inputs`` keys are preserved).
        """
        result: NameToTensorList = {}

        # --- Text tokenization ---
        if prompt is not None:
            tokens = self.tokenizer.encode(prompt)
            result["text_inputs"] = [
                torch.tensor(tokens, dtype=torch.long)
            ]

        if tensors is None:
            tensors = {}

        # --- Image processing: pixel_values + image_grid_thw ---
        image_inputs = tensors.get("image_inputs", [])
        if image_inputs:
            # Use HF AutoImageProcessor if available for correct preprocessing
            # (resizing, normalization, patch extraction, grid computation).
            pixel_values_list = []
            grid_thw_list = []
            try:
                processor = getattr(self, "_image_processor", None)
                if processor is None:
                    from transformers import AutoImageProcessor
                    processor = AutoImageProcessor.from_pretrained(
                        self.local_dir, trust_remote_code=True,
                    )
                    self._image_processor = processor
                for img in image_inputs:
                    # data_worker.py provides images as (C, H, W) float32 in
                    # [0, 1] on the GPU.  HF's Qwen2VLImageProcessor (which
                    # Qwen3-Omni inherits) defaults to do_rescale=True with
                    # rescale_factor=1/255, so passing already-normalized
                    # [0, 1] floats causes a SECOND rescale (→[0, 0.004]) and
                    # the model ends up seeing a uniform near-zero tensor —
                    # i.e., "a completely black square".  Convert back to
                    # uint8 [0, 255] HWC numpy (the PIL-equivalent format)
                    # so the processor's full normalization pipeline runs
                    # correctly.
                    if img.dtype.is_floating_point:
                        img_proc = (img * 255.0).clamp(0, 255).to(torch.uint8)
                    else:
                        img_proc = img
                    if img_proc.dim() == 3 and img_proc.shape[0] in (1, 3):
                        # CHW -> HWC for HF processor
                        img_proc = img_proc.permute(1, 2, 0)
                    img_proc = img_proc.cpu().contiguous().numpy()

                    proc_out = processor(images=img_proc, return_tensors="pt")
                    pixel_values_list.append(proc_out["pixel_values"][0])
                    if "image_grid_thw" in proc_out:
                        grid_thw_list.append(proc_out["image_grid_thw"][0])
            except Exception as e:
                logger.warning(
                    "Qwen3-Omni: image processor unavailable (%s); "
                    "passing raw image_inputs through as pixel_values", e,
                )
                pixel_values_list = list(image_inputs)
            if pixel_values_list:
                result["pixel_values"] = pixel_values_list
            if grid_thw_list:
                result["image_grid_thw"] = grid_thw_list

        # --- Audio processing: audio_features + audio_seqlens ---
        audio_inputs = tensors.get("audio_inputs", [])
        if audio_inputs:
            audio_features_list = []
            audio_seqlens_list = []
            try:
                processor = getattr(self, "_audio_processor", None)
                if processor is None:
                    from transformers import AutoFeatureExtractor
                    processor = AutoFeatureExtractor.from_pretrained(
                        self.local_dir, trust_remote_code=True,
                    )
                    self._audio_processor = processor
                for waveform in audio_inputs:
                    # waveform: (channels, time) from data_worker.
                    # Most processors expect mono (time,) or (1, time).
                    wave = waveform
                    if wave.dim() == 2 and wave.shape[0] > 1:
                        wave = wave.mean(dim=0)  # mix channels to mono
                    elif wave.dim() == 2:
                        wave = wave.squeeze(0)
                    sampling_rate = getattr(processor, "sampling_rate", 16000)
                    proc_out = processor(
                        wave.cpu().numpy(),
                        sampling_rate=sampling_rate,
                        return_tensors="pt",
                    )
                    audio_features_list.append(proc_out["input_features"][0])
                    audio_seqlens_list.append(
                        torch.tensor(wave.shape[-1], dtype=torch.long)
                    )
            except Exception as e:
                logger.warning(
                    "Qwen3-Omni: audio processor unavailable (%s); "
                    "passing raw audio_inputs through as audio_features", e,
                )
                audio_features_list = list(audio_inputs)
            if audio_features_list:
                result["audio_features"] = audio_features_list
            if audio_seqlens_list:
                result["audio_seqlens"] = audio_seqlens_list

        # --- Video processing: pixel_values_videos + video_grid_thw ---
        video_inputs = tensors.get("video_inputs", [])
        if video_inputs:
            # TODO: proper video frame extraction + grid computation via
            # AutoVideoProcessor (or AutoImageProcessor on stacked frames).
            # For now, pass raw video tensors through.
            result["pixel_values_videos"] = list(video_inputs)

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

        # Code Predictor is a separate small model loaded under "talker.code_predictor."
        # It is used inside the TalkerSubmodule for multi-step codec prediction.
        from mminf.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor

        with torch.device("meta"):
            code_predictor = Qwen3OmniCodePredictor(self.config)

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[
                ModuleAndPrefix(code_predictor, prefix="talker.code_predictor"),
            ],
            device=device,
        )

        talker_model.eval()
        code_predictor.eval()

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
        #    embedding layer to free its memory.  Peak temp cost is
        #    vocab_size * hidden_size * 2 bytes (~620 MB for the
        #    151k-vocab Thinker), held only briefly during init.
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
        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        # Code2Wav is the vocoder that converts codec tokens to audio waveform.
        # The actual model class will be defined in components.
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

        hf_cfg = self.config.code2wav.get_hf_config()
        code2wav_hf = Qwen3OmniMoeCode2Wav._from_config(hf_cfg)

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[
                ModuleAndPrefix(code2wav_hf, prefix="code2wav"),
            ],
            device=device,
        )

        code2wav_model = Qwen3OmniCode2Wav(hf_cfg, code2wav_hf)
        code2wav_model.eval()

        from mminf.model.qwen3_omni.submodules import Code2WavSubmodule
        return Code2WavSubmodule(
            code2wav_model=code2wav_model,
            config=self.config,
        )

    def _create_audio_encoder_submodule(self, device: str) -> NodeSubmodule:
        """Load the audio encoder (AuT) from HF weights."""
        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        # Reuse HF audio encoder directly (Whisper-style, not perf-critical)
        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )

        # Load config only (no weights)
        config = AutoConfig.from_pretrained(
            self.local_dir,
            trust_remote_code=True,
        )

        # This should be a Qwen3OmniMoeConfig
        audio_config = config.thinker_config.audio_config

        # Build the audio encoder from config
        audio_encoder = Qwen3OmniMoeAudioEncoder._from_config(audio_config)

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
        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        # Reuse HF vision encoder directly
        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeVisionEncoder,
        )

        # Load full config (no weights)
        config = AutoConfig.from_pretrained(
            self.local_dir,
            trust_remote_code=True,
        )

        # Extract the vision sub-config
        vision_config = config.thinker_config.vision_config

        # Build the vision encoder
        vision_encoder = Qwen3OmniMoeVisionEncoder._from_config(vision_config)

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[ModuleAndPrefix(vision_encoder, prefix="thinker.visual")],
            device=device,
        )
        vision_encoder.eval()

        from mminf.model.qwen3_omni.submodules import VisionEncoderSubmodule
        return VisionEncoderSubmodule(vision_encoder=vision_encoder, config=self.config)
