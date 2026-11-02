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
    Talker  --[codec_tokens,  ScheduledLeftContextChunkPolicy((12, 13), 25, 25)]--> Code2Wav

Producer-triggered Talker walks:
    The Thinker assigns its streamed states the Talker walk they run under
    (see ``_talker_walk``).  States from a Thinker prefill walk extend the Talker
    KV cache in ``talker_prefill``; the first ``thinker_decode`` state runs
    ``talker_last_prefill``, which samples the first codec token, and the
    rest feed ``talker_decode``.  The Talker never needs to know how many
    prefill walks the Thinker runs.  A chunked Thinker prefill streams each
    chunk's states as it runs (``per_chunk_outputs``), and the Talker extends
    its KV with each, so it keeps pace with the Thinker's prompt.

Text-only mode:
    When output_modalities does not include "audio", only the Thinker
    partition runs.  Talker and Code2Wav are idle.
"""

import logging
from pathlib import Path
from typing import NamedTuple

import torch
from jinja2 import TemplateError
from transformers import AutoTokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import PartitionDefinition
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import (
    MAX_OUTPUT_TOKENS,
    KVLabel,
    LabelLayout,
    Model,
    PrefixKeys,
    ProcessPromptOutput,
    Span,
    TensorAndMetadata,
    WalkStep,
)
from mstar.model.components import MoeParallelConfig
from mstar.model.media_limits import VideoBudget, probe_video
from mstar.model.multimodal import (
    TEXT,
    PromptPart,
    as_uint8_chw,
    check_attachments,
    check_plan,
    find_media_spans,
    messages_from_parts,
    parts_from_modalities,
    prefill_plan,
    span_name,
    split_around_spans,
    split_spans,
)
from mstar.model.qwen3_omni.components.rope import get_rope_index_vision
from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
from mstar.model.qwen3_omni.config import (
    AUDIO_ATTN,
    CODE_PRED_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    TALKER_SAMPLER,
    THINKER_ATTN,
    THINKER_KV,
    THINKER_MIXED,
    THINKER_POS,
    THINKER_SAMPLER,
    VISION_ATTN,
)
from mstar.model.qwen3_omni.video_reader import header_charged, install_budgeted_readers
from mstar.model.submodule_base import ChunkingPolicy, NodeSubmodule
from mstar.model.utils import Operation, TextOutput, WeightConverter
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import (
    Connection,
    PartitionTopology,
    ProducerWalkCtx,
    StreamingGraphEdge,
)

logger = logging.getLogger(__name__)

# Code2Wav streaming knobs a deployment YAML's ``model_kwargs`` may set; the
# rest of the Code2Wav config comes from the checkpoint.
CODE2WAV_SERVING_OVERRIDES = ("codec_chunk_schedule", "codec_chunk_frames", "codec_left_context_frames")


def apply_code2wav_serving_overrides(code2wav, model_kwargs: dict) -> None:
    """Move the Code2Wav streaming knobs out of ``model_kwargs`` into the
    config, rejecting values that would drop or repeat streamed audio."""
    for name in CODE2WAV_SERVING_OVERRIDES:
        if model_kwargs.get(name) is not None:
            value = model_kwargs.pop(name)
            setattr(code2wav, name, tuple(value) if isinstance(value, list) else value)
    code2wav.check_streaming_knobs()


def _thinker_to_talker_policy() -> FixedChunkPolicy:
    # Past the Thinker's end only the decode loop keeps running; an empty
    # chunk in a prefill walk would run a prefill pass with nothing in it.
    # Partial: the Talker extends its KV with each Thinker prefill chunk.
    return FixedChunkPolicy(
        chunk_size=1, continue_after_done=frozenset({"talker_decode"}),
        allow_partial=True,
    )


def _talker_walk(ctx: ProducerWalkCtx) -> str:
    """The Talker walk a Thinker pass's streamed states run under."""
    if ctx.producer_walk != "thinker_decode":
        return "talker_prefill"
    # The first decode iteration closes the prefill and samples the first codec
    first = ctx.loop_iters.get("thinker_decode_loop", 0) == 0
    return "talker_last_prefill" if first else "talker_decode"

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


class ImageSpan(NamedTuple):
    length: int
    advance: int


def image_span(grid_thw: torch.Tensor, merge_size: int) -> ImageSpan:
    """An image's tokens share one temporal position and sit on the merged
    grid, so it advances the position by the grid's longer side."""
    t, h, w = (int(n) for n in grid_thw)
    h, w = h // merge_size, w // merge_size
    return ImageSpan(t * h * w + 2, max(h, w) + 2)


# ---------------------------------------------------------------------------
# Qwen3OmniModel
# ---------------------------------------------------------------------------

class Qwen3OmniModel(Model):
    """Qwen3-Omni: Thinker + Talker + Code2Wav 3-partition streaming model."""

    LENGTH_KWARGS = ("max_output_tokens", "talker_max_output_tokens")
    PREPROCESS_PARALLEL = True

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
        from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig

        local_dir = _resolve_local_hf_snapshot(model_path_hf, cache_dir=cache_dir)
        self.config = Qwen3OmniModelConfig.from_pretrained(local_dir)
        self.local_dir = local_dir
        apply_code2wav_serving_overrides(self.config.code2wav, kwargs)

        # Tokenizer (Thinker uses a Qwen-family tokenizer). Qwen3-Omni is native
        # in transformers, so no loader here runs code from the checkpoint repo.
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, cache_dir=cache_dir, trust_remote_code=False,
        )

        # Full multimodal processor: combines tokenizer + image_processor +
        # video_processor + audio feature_extractor + chat template support.
        # Used by process_prompt to build the full ChatML prompt with the
        # correct image_pad / audio_pad / video_pad expansion.
        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                local_dir, cache_dir=cache_dir, trust_remote_code=False,
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

    # -------------------------------------------------------------------
    # Model ABC: resources
    # -------------------------------------------------------------------
    def get_node_resources(self) -> list[NodeResourceSpec]:
        thinker_kv = PagedKVConfig(
            num_layers=self.config.thinker_text.num_hidden_layers,
            num_kv_heads=self.config.thinker_text.num_key_value_heads,
            head_dim=self.config.thinker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.thinker_text.num_attention_heads,
        )
        talker_kv = PagedKVConfig(
            num_layers=self.config.talker_text.num_hidden_layers,
            num_kv_heads=self.config.talker_text.num_key_value_heads,
            head_dim=self.config.talker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.talker_text.num_attention_heads,
        )

        return [
            KVSpec(
                resource_key=THINKER_KV,
                nodes={"Thinker"},
                config=thinker_kv
            ),
            KVSpec(
                resource_key=TALKER_KV,
                nodes={"Talker"},
                config=talker_kv
            ),
            AttentionSpec(
                resource_key=THINKER_ATTN,
                nodes={"Thinker"},
                config=AttentionConfig(
                    kv_cache=THINKER_KV
                ),
            ),
            AttentionSpec(
                resource_key=TALKER_ATTN,
                nodes={"Talker"},
                config=AttentionConfig(
                    kv_cache=TALKER_KV
                ),
            ),
            PositionSpec(
                resource_key=THINKER_POS,
                nodes={"Thinker"},
                config=PositionConfig(
                    kv_cache=THINKER_KV
                )
            ),
            PositionSpec(
                resource_key=TALKER_POS,
                nodes={"Talker"},
                config=PositionConfig(
                    kv_cache=TALKER_KV
                )
            ),
            SamplerSpec(
                resource_key=THINKER_SAMPLER,
                nodes={"Thinker"},
                vocab_size=self.config.thinker_text.vocab_size,
                enable_repetion_penalty=True,
            ),
            SamplerSpec(
                resource_key=TALKER_SAMPLER,
                nodes={"Talker"},
                vocab_size=self.config.talker_text.vocab_size,
                enable_repetion_penalty=True,
            ),
            SamplerSpec(
                resource_key=CODE_PRED_SAMPLER,
                nodes={"Talker"},
                vocab_size=self.config.code_predictor.vocab_size,
                enable_repetion_penalty=False,
            ),
            *self._encoder_resources(),
        ]

    def _encoder_resources(self) -> list[NodeResourceSpec]:
        """The encoders' attention: within each 8 s audio window, and within
        each image or video frame group. No cache, so nothing else."""
        from mstar.model.components.aut_submodule import AuTEncoderSubmodule
        from mstar.model.qwen3_omni.submodules import VisionEncoderSubmodule

        audio, vision = self.config.audio_encoder, self.config.vision
        return [
            RaggedAttentionSpec(
                resource_key=AUDIO_ATTN, nodes={"audio_encoder"},
                config=RaggedAttentionConfig(
                    num_qo_heads=audio.encoder_attention_heads,
                    num_kv_heads=audio.encoder_attention_heads,
                    head_dim=audio.head_dim,
                    max_segments_per_request=AuTEncoderSubmodule.max_windows_per_replay(audio),
                ),
            ),
            RaggedAttentionSpec(
                resource_key=VISION_ATTN, nodes={"vision_encoder"},
                config=RaggedAttentionConfig(
                    num_qo_heads=vision.num_heads,
                    num_kv_heads=vision.num_heads,
                    head_dim=vision.head_dim,
                    # Frames of 256 patches or more fill the largest captured
                    # bucket at this many; a request with more frames runs the
                    # block loop eagerly (`VisionEncoderSubmodule._fits_capture`).
                    max_segments_per_request=VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1] // 256,
                ),
            ),
        ]

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, ResourceReqConfig]:
        del input_modalities, output_modalities
        model_kwargs = model_kwargs or {}

        return {
            THINKER_SAMPLER: SamplingReqConfig(
                temperature=model_kwargs.get("thinker_temperature", 0.7),
                top_p=model_kwargs.get("thinker_top_p", 0.9),
                ignore_eos=model_kwargs.get("ignore_eos", False),
                repetition_penalty=model_kwargs.get("thinker_repetition_penalty", 1.0)
            ),
            TALKER_SAMPLER: SamplingReqConfig(
                temperature=model_kwargs.get("talker_temperature", 0.9),
                top_k=model_kwargs.get("talker_top_k", 50),
                top_p=model_kwargs.get("talker_top_p", 1.0),
                repetition_penalty=model_kwargs.get("talker_repetition_penalty", 1.05)
            ),
            CODE_PRED_SAMPLER: SamplingReqConfig(
                temperature=model_kwargs.get("code_predictor_temperature", 1.0),
                top_k=model_kwargs.get("code_predictor_top_k", 50),
                top_p=model_kwargs.get("code_predictor_top_p", 0.8),
            )
        }

    def checkpoint_path(self) -> str:
        return self.local_dir

    def _audio_frames(self, seqlens: torch.Tensor) -> torch.Tensor:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

        return _get_feat_extract_output_lengths(seqlens, n_window=self.config.audio_encoder.n_window)

    def _get_max_talker_output_tokens(self, **model_kwargs):
        return model_kwargs.get("talker_max_output_tokens", MAX_OUTPUT_TOKENS)

    # -----------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -----------------------------------------------------------------------

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        return {"Thinker": {THINKER_MIXED: {
            "prefill_text", "prefill_audio", "prefill_vision", "thinker_decode",
        }}}

    def get_chunking_policies(self) -> dict[str, tuple[ChunkingPolicy, ...]]:
        from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

        budget = ThinkerSubmodule.MAX_BATCH_TOKENS
        return {"Thinker": (
            ChunkingPolicy(
                walks=("prefill_text", "prefill_audio", "prefill_vision"),
                chunkable=True, max_batch_tokens=budget,
                # the Talker extends its KV with each chunk's states as it lands
                per_chunk_outputs=frozenset({"thinker_states", "thinker_mask"}),
            ),
            # carries the budget for the steps that mix a chunk with decode rows
            ChunkingPolicy(walks=(THINKER_MIXED,), max_batch_tokens=budget),
        )}

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
            input_names=["text_inputs"],
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
                input_names=["audio_features", "audio_seqlens"],
                outputs=[GraphEdge(next_node="Thinker", name="audio_embeds")],
            ),
            GraphNode(
                name="Thinker",
                input_names=["audio_embeds"],
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
                input_names=["pixel_values", "image_grid_thw"],
                outputs=[
                    GraphEdge(next_node="Thinker", name="vision_embeds"),
                    GraphEdge(next_node="Thinker", name="deepstack")
                ],
            ),
            GraphNode(
                name="Thinker",
                input_names=["vision_embeds", "deepstack", "video_second_per_grid", "image_grid_thw"],
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
        thinker_decode = Loop(
            name="thinker_decode_loop",
            section=GraphNode(
                name="Thinker",
                input_names=["text_inputs"],
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
            max_iters=self.config.thinker_text.max_position_embeddings,
            outputs=[],
        )

        # -- Talker prefill: extends the KV cache with each streamed state --
        talker_prefill = GraphNode(
            name="Talker",
            input_names=["thinker_states", "thinker_mask"],
            outputs=[],
        )

        talker_last_prefill = Sequential(
            sections=[
                GraphNode(
                    name="Talker",
                    input_names=["thinker_states", "thinker_mask"],
                    outputs=[
                        GraphEdge(
                            next_node=EMPTY_DESTINATION,
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
        talker_decode = Loop(
            name="talker_decode_loop",
            section=Sequential(
                sections=[
                    GraphNode(
                        name="Talker",
                        input_names=["thinker_states", "thinker_mask", "talker_input_embeds"],
                        outputs=[
                            GraphEdge(
                                next_node="Talker",
                                name="talker_input_embeds",
                            ),
                            StreamingGraphEdge(
                                next_node="Code2Wav",
                                name="codec_tokens",
                                target_partition="Code2Wav",
                            ),
                        ],
                    )
                ]
            ),
            max_iters=self.config.thinker_text.max_position_embeddings,
            outputs=[],
        )

        # -- Code2Wav chunk: vocoder streaming decode --
        code2wav_chunk = GraphNode(
            name="Code2Wav",
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
            ),
            PartitionDefinition(
                name="Talker",
                graph_walks={"talker_prefill", "talker_last_prefill", "talker_decode"},
            ),
            PartitionDefinition(
                name="Code2Wav",
                graph_walks={"code2wav_chunk"},
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
                    chunk_policy_factory=_thinker_to_talker_policy,
                    consumer_walk=_talker_walk,
                ),
                Connection(
                    from_partition="Thinker",
                    to_partition="Talker",
                    edge_name="thinker_mask",
                    chunk_policy_factory=_thinker_to_talker_policy,
                    consumer_walk=_talker_walk,
                ),
                Connection(
                    from_partition="Talker",
                    to_partition="Code2Wav",
                    edge_name="codec_tokens",
                    chunk_policy_factory=self.config.code2wav.chunk_policy,
                ),
            ],
        )

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        # Qwen3-Omni's Code2Wav vocoder emits speech at 24 kHz.
        return 24000

    # -----------------------------------------------------------------------
    # Model ABC: walk plan
    # -----------------------------------------------------------------------

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        """The Thinker prefills each span of the prompt where it was written,
        then decodes fed its first token. Every Thinker step says whether to
        emit ``thinker_states``: a text-only request skips them, to save
        cross-partition bandwidth.

        With audio out, the Thinker's stream moves the Talker through its
        prefill walks (``_talker_walk``), and the conductor feeds
        ``talker_decode`` its first input once ``talker_last_prefill`` has
        sampled the first codec token. Code2Wav runs once per chunk of codec
        tokens, until the worker reports the stream's final chunk; a
        count-based end would miss the left-context policy's flush pass.
        """
        audio_output = "audio" in output_modalities
        if partition_name == "Thinker":
            thinker = {"audio_output": audio_output}
            return [
                WalkStep(walk, bind=bind, step_metadata=thinker)
                for walk, bind in self._build_thinker_prefill_schedule(input_modalities, input_signals)
            ] + [WalkStep("thinker_decode", bind={"text_inputs": "new_token"}, step_metadata=thinker)]
        if partition_name not in ("Talker", "Code2Wav"):
            raise ValueError(f"Unknown partition: {partition_name!r}")
        if not audio_output:
            return []
        if partition_name == "Code2Wav":
            return [WalkStep("code2wav_chunk")]
        talker_max_tokens = self._get_max_talker_output_tokens(**model_kwargs)
        prefill = {"voice": model_kwargs.get("voice", "Ethan"), "talker_max_tokens": talker_max_tokens}
        return [
            WalkStep("talker_prefill", step_metadata=prefill),
            WalkStep("talker_last_prefill", step_metadata=prefill),
            WalkStep("talker_decode", step_metadata={"talker_max_tokens": talker_max_tokens}),
        ]

    # Per-modality walk name and the signals it binds: the walk's input names
    # (primary feature tensor first, then that encoder's auxiliary tensors) to
    # the prompt lists whose entries fill them. Video rides the vision encoder,
    # so its tensors arrive under the vision input names.
    _MM_WALKS: dict[str, tuple[str, dict[str, str]]] = {
        "audio": ("prefill_audio", {
            "audio_features": "audio_features",
            "audio_seqlens": "audio_seqlens",
        }),
        "image": ("prefill_vision", {
            "pixel_values": "pixel_values",
            "image_grid_thw": "image_grid_thw",
        }),
        "video": ("prefill_vision", {
            "pixel_values": "pixel_values_videos",
            "image_grid_thw": "video_grid_thw",
            "video_second_per_grid": "video_second_per_grid",
        }),
    }
    # the prompt lists whose entries each prefill in a walk of their own
    SPANS = ("text_inputs", *sorted({key for _, keys in _MM_WALKS.values() for key in keys.values()}))

    def _build_thinker_prefill_schedule(
        self,
        input_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[tuple[str, dict[str, str]]]:
        """The Thinker's prefill walks, each as ``(walk_name, {input_name: span signal})``.

        Walks the prefill plan, so text spans and attachments prefill in the
        order written and each attachment gets its own walk. ``process_prompt``
        split ``text_inputs`` against the same plan, so the nth span belongs to
        the nth text step; each list entry arrives as a signal of its own
        (`split_spans`). A modality walk binds every tensor its walk's nodes
        need: the auxiliary tensors alongside the primary feature tensor.
        """
        schedule: list[tuple[str, dict[str, str]]] = []
        # ``check_plan`` already failed a mismatch at intake, where a 400 can
        # still be returned; skipping here keeps one that slipped through from
        # raising inside the conductor instead.
        for step in prefill_plan(parts_from_modalities(input_modalities)):
            if step.modality == TEXT:
                span = span_name("text_inputs", step.index)
                if span in input_signals:
                    schedule.append(("prefill_text", {"text_inputs": span}))
                continue
            walk, signal_names = self._MM_WALKS[step.modality]
            bind = {
                name: span_name(key, step.index)
                for name, key in signal_names.items()
                if span_name(key, step.index) in input_signals
            }
            if bind:
                schedule.append((walk, bind))
        return schedule

    # -----------------------------------------------------------------------
    # Model ABC: prompt processing
    # -----------------------------------------------------------------------

    def load_video(
        self, filepath: str, device: str
    ) -> TensorAndMetadata:
        # TODO: support audio in video
        from qwen_omni_utils.v2_5.vision_process import VIDEO_READER_BACKENDS, fetch_video, smart_nframes

        install_budgeted_readers(VIDEO_READER_BACKENDS)
        # the sampled readers decode the frames fetch_video samples, at full size before resizing them
        probe = probe_video(filepath)
        if probe is not None:
            total, fps, height, width = probe
            VideoBudget().add(height, width, smart_nframes({"video": filepath}, total_frames=total, video_fps=fps))
        with header_charged(probe is not None):
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

    # The chat template writes each attachment as this triple; the modality
    # walk re-emits the two sentinels around its encoder output, so only the
    # pad interior is dropped from the text spans.
    _PLACEHOLDER_TOKENS: dict[str, tuple[str, str, str]] = {
        "audio": ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"),
        "image": ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"),
        "video": ("<|vision_start|>", "<|video_pad|>", "<|vision_end|>"),
    }

    def _placeholder_specs(self) -> dict[str, tuple[int, int, int]]:
        """``(start, pad, end)`` sentinel ids, read off the tokenizer.

        The tokenizer tokenized the prompt, so it decides these ids.
        ``thinker_config`` carries its own copies, and on the released
        checkpoint they disagree: scanning with those finds nothing.
        """
        specs: dict[str, tuple[int, int, int]] = {}
        for modality, tokens in self._PLACEHOLDER_TOKENS.items():
            ids = tuple(self.tokenizer.convert_tokens_to_ids(t) for t in tokens)
            if any(i is None or i == self.tokenizer.unk_token_id for i in ids):
                continue
            specs[modality] = ids
        return specs

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        input_metadata: dict[str, dict] = {},
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
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

        The result holds the text spans and the per-modality tensors the
        Thinker's prefill walks need, each list entry as a signal of its own
        (`split_spans`: ``text_inputs.N``, ``pixel_values.N``, ...).
        """
        result: NameToTensorList = {}

        attached = tensors is not None
        if tensors is None:
            tensors = {}

        # ----- Convert raw modality tensors to PIL/numpy form for HF -----
        raw_image_inputs = tensors.get("image_inputs", [])
        raw_audio_inputs = tensors.get("audio_inputs", [])
        raw_video_inputs = tensors.get("video_inputs", [])

        # HF processors expect uint8 (H, W, C): their default do_rescale=True
        # would rescale the data worker's [0, 1] floats a second time.
        pil_images = [as_uint8_chw(img, truncate=True).permute(1, 2, 0).numpy() for img in raw_image_inputs]

        np_audios: list = []
        for waveform in raw_audio_inputs:
            np_audios.append(waveform.cpu().numpy())

        # input_modalities is the layout, here and in the schedule builder;
        # prompt_parts only fills its text slots and roles.
        texts, roles = prompt, None
        if prompt_parts is not None:
            texts = [p.text or "" for p in prompt_parts if p.modality == TEXT]
            roles = [p.role for p in prompt_parts]
        parts = parts_from_modalities(input_modalities, texts, roles)
        if attached:
            check_attachments(parts, {
                "image": len(raw_image_inputs),
                "audio": len(raw_audio_inputs),
                "video": len(raw_video_inputs),
            })

        # HF puts modality content INSIDE its message's turn, each attachment
        # rendered as ``<|x_start|><|x_pad|><|x_end|>`` at its own place:
        #
        #   <|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>{prompt}<|im_end|>
        #
        # So writing the parts out in request order and tokenizing once lets
        # the placement be read back off the ids, with no boundary
        # re-tokenized. The walks re-emit the sentinels themselves
        # (``ThinkerSubmodule._wrap_audio_input``) and the encoder embeddings
        # replace the pad tokens, so the Thinker's spans are what lies between
        # the placeholders.
        #
        # A voice persona as the system prompt keeps the Thinker's text plain
        # enough for the Talker to say, so a client that sends its own system
        # message and wants audio puts the persona text in that message.
        messages = messages_from_parts(parts, default_system=(
            "You are Qwen, a virtual human developed by the "
            "Qwen team, Alibaba Group, capable of perceiving "
            "auditory and visual inputs, as well as generating "
            "text and speech."
        ))

        try:
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TemplateError as exc:
            raise ValueError(f"Qwen3-Omni's chat template refused the chat: {exc}") from exc

        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
        spans = find_media_spans(input_ids, self._placeholder_specs())
        segments = split_around_spans(input_ids, spans)
        plan = prefill_plan(parts)
        check_plan(plan, spans, len(segments))
        result["text_inputs"] = segments

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

        if not attached:
            return ProcessPromptOutput(split_spans(result, self.SPANS)[0], {})
        result, kv_layout = split_spans(result, self.SPANS, self._kv_layout(plan, result, "audio" in output_modalities))
        return ProcessPromptOutput(result, {"kv_layout": kv_layout})

    def _kv_layout(self, plan: list[PromptPart], tensors: NameToTensorList, audio_output: bool) -> dict:
        """What each Thinker prefill step writes and, with audio out, what the Talker writes of it.

        The Talker prefills every step's rows except those of a system or earlier
        assistant turn. It then prefills its assistant prefix, and writes a row per decode step.
        """
        from mstar.model.qwen3_omni.submodules import TalkerSubmodule, talker_text_mask

        merge = self.config.vision.spatial_merge_size
        thinker, talker, position = [], [], 0
        for step in plan:
            if step.modality == TEXT:
                ids = tensors["text_inputs"][step.index]
                span = Span("ids", "prefill_text", len(ids), len(ids), "text_inputs")
                kept = int(talker_text_mask(ids, self.config).sum())
            elif step.modality == "audio":
                # the encoder's rows between two sentinels
                rows = int(self._audio_frames(tensors["audio_seqlens"][step.index])) + 2
                span = Span("digest", "prefill_audio", rows, rows, ("audio", step.index))
                kept = rows
            elif step.modality == "image":
                image = image_span(tensors["image_grid_thw"][step.index], merge)
                span = Span("digest", "prefill_vision", image.length, image.advance, ("image", step.index))
                kept = image.length
            else:
                grid = tensors["video_grid_thw"][step.index]
                per_grid = tensors["video_second_per_grid"][step.index].item()
                rows = int(grid.prod()) // merge ** 2 + 2
                # the walk moves the counter one past its end sentinel, which follows the grid's largest position
                positions = get_rope_index_vision(
                    grid, position + 1, self.config.thinker.position_id_per_seconds,
                    spatial_merge_size=merge, seconds_per_grid=per_grid,
                )
                advance = int(positions.max()) + 2 - position
                span = Span("digest", "prefill_vision", rows, advance, ("video", step.index))
                kept = rows
            thinker.append(span)
            talker.append(Span("rows", "talker_prefill", kept, kept, "thinker_states"))
            position += span.advance
        prefix = TalkerSubmodule.TALKER_LAST_PREFILL_TOKENS_PER_REQ
        # A prompt with a video keys only its leading text, since keying past it has never been run or tested.
        video = any(step.modality == "video" for step in plan)
        return {
            THINKER_KV: {"main": LabelLayout(
                spans=tuple(thinker), growth=(("thinker_decode_loop", 1),), keyed_spans=1 if video else None,
            )},
            TALKER_KV: {"main": LabelLayout(
                spans=(*talker, Span("rows", "talker_last_prefill", prefix, prefix, "thinker_states")),
                growth=(("talker_decode_loop", 1),),
            ) if audio_output else LabelLayout()},
        }

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {
            "thinker_decode_loop": self.get_max_output_tokens(**model_kwargs),
            "talker_decode_loop": max(1, self._get_max_talker_output_tokens(**model_kwargs)),
        }

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        """The Thinker keys its prompt by text ids, each image and audio clip by
        file, and its replies as they decode."""
        return {
            THINKER_KV: {"main": KVLabel(
                growth=(("thinker_decode", "thinker_decode_loop"),),
                keys=PrefixKeys("text_inputs", ("prefill_text", "prefill_vision", "prefill_audio"), growth=True),
            )},
            TALKER_KV: {"main": KVLabel(
                growth=(("talker_decode", "talker_decode_loop"),),
            )},
        }

    # -----------------------------------------------------------------------
    # Model ABC: postprocess
    # -----------------------------------------------------------------------

    def text_output(self) -> TextOutput:
        return TextOutput(self._chunk_detokenizer(), (self.config.im_end_token_id,))

    # -----------------------------------------------------------------------
    # Model ABC: sharding
    # -----------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        # Talker LLM (attention + MoE-with-shared-expert) is TP-capable
        # via the same ``ParallelAttention`` / ``ParallelSparseMoeBlock*``
        # parts as the Thinker. The internal CodePredictor is intentionally
        # left at TP=1 (replicated weights, deterministic sampler) — see
        # ``_create_talker_submodule``. ``shard_dim`` stays empty because
        # every cross-edge signal (``thinker_states``, ``thinker_mask``,
        # ``codec_tokens``, ``talker_input_embeds``, ``new_token``) is
        # already replicated by the upstream all-reduce or sampler
        # broadcast before it leaves its producing node.
        # The Thinker's routed experts can also run EP; the Talker's
        # shared-expert block has no EP variant.
        return ShardingConfig(
            groups=[], tp_enabled_nodes={"Thinker", "Talker"}, shard_dim={},
            ep_enabled_nodes={"Thinker"},
        )

    # -----------------------------------------------------------------------
    # Model ABC: submodule loading
    # -----------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype,
            moe_parallel=moe_parallel,
        )
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

    def _create_submodule(
        self, node_name: str, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ) -> NodeSubmodule | None:
        if node_name == "Thinker":
            return self._create_thinker_submodule(
                device, tp_group=tp_group, autocast_dtype=autocast_dtype,
                moe_parallel=moe_parallel,
            )
        elif node_name == "Talker":
            return self._create_talker_submodule(
                device, tp_group=tp_group, autocast_dtype=autocast_dtype,
            )
        elif node_name == "Code2Wav":
            return self._create_code2wav_submodule(device)
        elif node_name == "audio_encoder":
            return self._create_audio_encoder_submodule(device, autocast_dtype)
        elif node_name == "vision_encoder":
            return self._create_vision_encoder_submodule(device, autocast_dtype)
        return None

    @staticmethod
    def _thinker_remap(name: str) -> str | None:
        """Map HF checkpoint keys (after ``thinker.`` prefix strip) to model param paths.

        Handles the ``block_sparse_moe`` → ``mlp`` rename and the per-expert
        weight fusion: ``experts.{N}.{gate,up,down}_proj.weight`` becomes a
        shard_id-carrying key that the MoE weight_loaders consume via
        ``StackedParamRule``.
        """
        import re

        if "rotary_emb" in name:
            return None
        name = name.replace("block_sparse_moe.", "mlp.")
        # Per-expert weights: experts.N.{gate,up,down}_proj.weight
        # → experts.{gate_up_proj,down_proj} with shard_id handled by stacked rules.
        # We rewrite the name so StackedParamRule suffix matching works:
        # "experts.N.gate_proj.weight" → "experts.gate_proj.__N__.weight"
        # The weight_loader on the fused param extracts the expert index from shard_id.
        m = re.match(r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
        if m:
            prefix, expert_idx, proj = m.groups()
            return f"{prefix}.experts.{proj}.__expert{expert_idx}__.weight"
        return name

    # MoE stacked param rules: route per-expert projections into fused params.
    # The shard_id encodes both the projection type AND the expert index.
    # The __expertN__ marker is injected by _thinker_remap; weight_loaders
    # parse it to determine the expert slot.
    _THINKER_STACKED_PARAMS: list = []  # populated lazily below

    def _get_thinker_stacked_params(self):
        from mstar.model.loader.base import StackedParamRule

        if self._THINKER_STACKED_PARAMS:
            return self._THINKER_STACKED_PARAMS
        E = self.config.thinker_text.num_experts
        # MoE expert rules MUST come before dense MLP rules because
        # the dense ".gate_proj" suffix would also match the remapped
        # MoE key "experts.gate_proj.__expertN__.weight". _apply_stacked
        # returns on first match, so longer/more-specific rules go first.
        rules = []
        for i in range(E):
            # source_suffix includes ".weight" so the replacement strips it —
            # the target params (experts.gate_up_proj, experts.down_proj) are
            # bare nn.Parameters, not Linear submodules, so they have no
            # ".weight" suffix in named_parameters().
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.gate_proj.__expert{i}__.weight",
                shard_id=f"gate:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.up_proj.__expert{i}__.weight",
                shard_id=f"up:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj",
                source_suffix=f".experts.down_proj.__expert{i}__.weight",
                shard_id=f"down:{i}",
            ))
        # Dense MLP gate/up fusion and attention qkv fusion.
        rules.append(StackedParamRule(".gate_up_proj", ".gate_proj", 0))
        rules.append(StackedParamRule(".gate_up_proj", ".up_proj", 1))
        rules.append(StackedParamRule(".qkv_proj", ".q_proj", "q"))
        rules.append(StackedParamRule(".qkv_proj", ".k_proj", "k"))
        rules.append(StackedParamRule(".qkv_proj", ".v_proj", "v"))
        self._THINKER_STACKED_PARAMS = rules
        return rules

    def _create_thinker_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_omni.components.thinker import Qwen3OmniThinkerModel

        with torch.device("meta"):
            thinker_model = Qwen3OmniThinkerModel(
                self.config, comm_group=tp_group, moe_parallel=moe_parallel,
            )
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            thinker_model = thinker_model.to(autocast_dtype)
        thinker_model.to_empty(device=device)

        weights = iter_safetensors_shards(
            self.local_dir, device=device,
            prefix="thinker."
        )
        # Strip the "thinker." prefix from checkpoint keys.
        weights = ((k.removeprefix("thinker."), v) for k, v in weights)

        load_hf_weights(
            thinker_model, weights,
            stacked_params=self._get_thinker_stacked_params(),
            name_remapper=self._thinker_remap,
        )
        thinker_model.eval()


        from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
        return ThinkerSubmodule(
            thinker_model=thinker_model,
            config=self.config,
        )

    def _create_talker_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_omni.components.talker import (
            Qwen3OmniTalkerModel,
        )

        with torch.device("meta"):
            # ``tp_group`` shards the Talker LLM's attention + MoE. The
            # CodePredictor stays TP=1 (separate construction below) —
            # its compute is small and the deterministic FlashInfer sampler
            # produces bit-equal codes on every rank, so per-rank
            # replication is cheaper than 150+ NCCL all-reduces per
            # decode step (5 layers x 15 unrolled iterations).
            talker_model = Qwen3OmniTalkerModel(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            talker_model = talker_model.to(autocast_dtype)
        talker_model.to_empty(device=device)

        # Talker and CodePredictor share the "talker." prefix. We stream
        # once and split: code_predictor keys go to the CodePredictor,
        # everything else goes to the TalkerModel.
        talker_weights = []
        code_pred_weights = []
        _CP_PREFIX = "talker.code_predictor."
        _TALKER_PREFIX = "talker."

        initialized_thinker_embed_tokens = False
        text_config = self.config.thinker_text
        embed_tokens = torch.nn.Embedding(
            text_config.vocab_size, text_config.hidden_size,
            device=device
        ).eval()
        for k, v in iter_safetensors_shards(self.local_dir, device=device, prefix=_TALKER_PREFIX):
            if k.startswith(_CP_PREFIX):
                code_pred_weights.append((k.removeprefix(_CP_PREFIX), v))
            else:
                talker_weights.append((k.removeprefix(_TALKER_PREFIX), v))
        for _k, v in iter_safetensors_shards(
            self.local_dir, device=device, prefix="thinker.model.embed_tokens"
        ):
            initialized_thinker_embed_tokens = True
            with torch.no_grad():
                embed_tokens.weight.copy_(v)

        assert initialized_thinker_embed_tokens, \
            "thinker.model.embed_tokens not found to initialize talker TTS embeds"

        stacked = self._get_thinker_stacked_params()
        load_hf_weights(
            talker_model, iter(talker_weights),
            stacked_params=stacked,
            name_remapper=self._thinker_remap,
        )
        talker_model.eval()

        with torch.device("meta"):
            code_predictor = Qwen3OmniCodePredictor(self.config)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            code_predictor = code_predictor.to(autocast_dtype)
        code_predictor.to_empty(device=device)
        load_hf_weights(
            code_predictor, iter(code_pred_weights),
            stacked_params=stacked,
            name_remapper=self._thinker_remap,
        )
        code_predictor.consolidate_stacked_weights()
        code_predictor.eval()

        from mstar.model.qwen3_omni.submodules import TalkerSubmodule
        talker_sub = TalkerSubmodule(
            talker_model=talker_model,
            code_predictor=code_predictor,
            config=self.config,
        )
        talker_sub.init_tts_embeds(embed_tokens)
        del embed_tokens

        return talker_sub

    def _create_code2wav_submodule(self, device: str) -> NodeSubmodule:
        # Code2Wav is the vocoder that converts codec tokens to audio waveform.
        # The actual model class will be defined in components.
        from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

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

        from mstar.model.qwen3_omni.submodules import Code2WavSubmodule
        return Code2WavSubmodule(
            code2wav_model=code2wav_model,
            config=self.config,
        )

    def _create_audio_encoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None,
    ) -> NodeSubmodule:
        """The AuT audio encoder (``thinker.audio_tower``)."""
        from mstar.model.components.aut_encoder import AuTEncoder
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_omni.submodules import AudioEncoderSubmodule

        with torch.device("meta"):
            encoder = AuTEncoder(self.config.audio_encoder, attn_key=AUDIO_ATTN)
        if autocast_dtype is not None:
            encoder = encoder.to(autocast_dtype)
        encoder.to_empty(device=device)
        prefix = "thinker.audio_tower."
        weights = iter_safetensors_shards(self.local_dir, device=device, prefix=prefix)
        encoder.load_weights((k.removeprefix(prefix), v) for k, v in weights)
        return AudioEncoderSubmodule(encoder.eval(), self.config)

    def _create_vision_encoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None,
    ) -> NodeSubmodule:
        """The ViT (``thinker.visual``) with its deepstack mergers."""
        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_omni.components.vision import Qwen3OmniVisionTower
        from mstar.model.qwen3_omni.submodules import VisionEncoderSubmodule
        from mstar.model.utils import check_loaded

        # built on the device, not on meta: the rotary table is a computed buffer
        with torch.device(device):
            tower = Qwen3OmniVisionTower(self.config.vision, attn_key=VISION_ATTN)
        if autocast_dtype is not None:
            tower = tower.to(autocast_dtype)
        prefix = "thinker.visual."
        weights = iter_safetensors_shards(self.local_dir, device=device, prefix=prefix)
        loaded = load_hf_weights(tower, ((k.removeprefix(prefix), v) for k, v in weights))
        check_loaded(tower, loaded, "Qwen3-Omni vision tower")
        return VisionEncoderSubmodule(tower.eval(), self.config)
