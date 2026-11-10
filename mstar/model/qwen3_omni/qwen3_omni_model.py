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
from mstar.conductor.request_info import FINAL_PREFILL, PartitionDefinition
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
    SeqShardSpec,
)
from mstar.engine.resources.lora import LoraSpec
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
from mstar.model.components.code_predictor_kernels import ResidualCodePredictor
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
    THINKER_SHARD,
    THINKER_TEXT_DECODE,
    VISION_ATTN,
)
from mstar.model.qwen3_omni.video_reader import header_charged, install_budgeted_readers
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import ChunkingPolicy, NodeSubmodule
from mstar.model.utils import Operation, TextOutput, WeightConverter, load_generation_defaults
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import (
    Connection,
    PartitionTopology,
    ProducerWalkCtx,
    StreamingGraphEdge,
)

logger = logging.getLogger(__name__)

# In a prefill schedule entry: how many of the step's text tensors lead its attachment
TEXT_BEFORE = "text_before"

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


# per-stage sampler knobs; a plain key targets the Thinker, ``<stage>_<knob>`` its stage
SAMPLING_KNOBS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "penalize_prompt")

# used where the checkpoint's generation_config.json names no value
FALLBACK_GENERATION_DEFAULTS = {
    "thinker_temperature": 0.7,
    "thinker_top_p": 0.9,
    "thinker_repetition_penalty": 1.0,
    "thinker_max_output_tokens": MAX_OUTPUT_TOKENS,
    "talker_temperature": 0.9,
    "talker_top_k": 50,
    "talker_top_p": 1.0,
    "talker_repetition_penalty": 1.05,
    "talker_max_output_tokens": MAX_OUTPUT_TOKENS,
    # no repetition penalty: the CodePredictor sampler lacks the capability
    "code_predictor_temperature": 1.0,
    "code_predictor_top_k": 50,
    "code_predictor_top_p": 0.8,
}

DEFAULT_VOICE = "Ethan"


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


class ThinkerStep(NamedTuple):
    """One Thinker prefill: its walk and the plan parts its row writes, in order."""
    walk: str
    parts: list[PromptPart]


def _thinker_steps(plan: list[PromptPart]) -> list[ThinkerStep]:
    """``plan`` grouped into Thinker prefills: each attachment with the text
    before it, the last one also with the text after it. The prefill schedule
    and the KV layout both read this, so the layout's steps are the ones that run."""
    media = [i for i, part in enumerate(plan) if part.modality != TEXT]
    if not media:
        return [ThinkerStep("prefill_text", [part]) for part in plan]
    steps: list[ThinkerStep] = []
    start = 0
    for at in media:
        end = len(plan) if at == media[-1] else at + 1
        steps.append(ThinkerStep(Qwen3OmniModel._MM_WALKS[plan[at].modality][0], plan[start:end]))
        start = end
    return steps


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
    def _lora_spec(self) -> LoraSpec:
        """Per-request adapters on the Thinker: its attention, any dense MLP layers, the
        LM head (one row per request: the last token's), and the routed experts when the
        YAML's ``targets:`` names them (``experts_gate_up_proj`` / ``experts_down_proj``,
        opt-in). The router takes none, so an adapter touching it is refused with that
        reason; so is one touching the Talker or the encoders, which are other nodes.
        Modules are named as the full model's checkpoint names them (``thinker.*``) or as
        the Thinker alone does."""
        from mstar.model.components.lora.modules import PER_SEGMENT, module_lora_spec
        from mstar.model.qwen3_omni.components.thinker import Qwen3OmniThinkerModel
        return module_lora_spec(
            lambda group: Qwen3OmniThinkerModel(self.config, comm_group=group),
            nodes={"Thinker"}, root_path="model",
            rows={"lm_head": PER_SEGMENT},
        )

    def get_node_resources(self) -> list[NodeResourceSpec]:
        # min_p is on where the checkpoint defaults it on, so that default is not refused
        defaults = self._generation_defaults()
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
            # every walk may split; over KV stored by position only where the deployment sets the
            # shard's exchange to all_gather or ring. A decode step never does.
            SeqShardSpec(resource_key=THINKER_SHARD, nodes={"Thinker"}, labels=("main",), axis=0),
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
                enable_repetition_penalty=True,
                enable_min_p=bool(defaults.get("thinker_min_p")),
            ),
            SamplerSpec(
                resource_key=TALKER_SAMPLER,
                nodes={"Talker"},
                vocab_size=self.config.talker_text.vocab_size,
                enable_repetition_penalty=True,
                enable_min_p=bool(defaults.get("talker_min_p")),
            ),
            SamplerSpec(
                resource_key=CODE_PRED_SAMPLER,
                nodes={"Talker"},
                vocab_size=self.config.code_predictor.vocab_size,
                enable_repetition_penalty=False,
                enable_min_p=bool(defaults.get("code_predictor_min_p")),
            ),
            *self._encoder_resources(),
            self._lora_spec(),
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
                    # The 512 MB default costs 2 GB across both encoders and slots.
                    # FA3 plans take none of it; FA2's captured plans split KV at
                    # the bucket's segment ceiling, ~230 MB here at bs=8.
                    workspace_mb=256,
                ),
            ),
            RaggedAttentionSpec(
                resource_key=VISION_ATTN, nodes={"vision_encoder"},
                config=RaggedAttentionConfig(
                    num_qo_heads=vision.num_heads,
                    num_kv_heads=vision.num_heads,
                    head_dim=vision.head_dim,
                    # Frames of 256 patches or more fill the largest captured
                    # bucket at this many; a step with more frames runs the
                    # block loop eagerly (`PackedBlockLoopEncoder._fits_capture`).
                    max_segments_per_step=VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1] // 256,
                    # FA2's split-KV partials at the 8192-patch bucket: ~95 MB.
                    workspace_mb=128,
                ),
            ),
        ]

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, ResourceReqConfig]:
        """One sampler per stage. Plain keys target the Thinker; a
        ``<stage>_<knob>`` key wins over a plain one for its stage."""
        del input_modalities, output_modalities
        model_kwargs = model_kwargs or {}
        return {
            THINKER_SAMPLER: SamplingReqConfig(
                **self._stage_sampling("thinker_", model_kwargs, primary=True),
                ignore_eos=model_kwargs.get("ignore_eos", False),
            ),
            TALKER_SAMPLER: SamplingReqConfig(
                **self._stage_sampling("talker_", model_kwargs),
            ),
            CODE_PRED_SAMPLER: SamplingReqConfig(
                **self._stage_sampling("code_predictor_", model_kwargs),
            ),
        }

    def _stage_sampling(self, stage: str, model_kwargs: dict, primary: bool = False) -> dict:
        """One stage's sampling knobs: its prefixed key, else the plain key
        (primary stage only), else the checkpoint or fallback default."""
        defaults = self._generation_defaults()
        values = {}
        for knob in SAMPLING_KNOBS:
            key = stage + knob
            if key in model_kwargs:
                values[knob] = model_kwargs[key]
            elif primary and knob in model_kwargs:
                values[knob] = model_kwargs[knob]
            elif key in defaults:
                values[knob] = defaults[key]
        return values

    def _generation_defaults(self) -> dict:
        """Stage-prefixed defaults: the checkpoint's ``generation_config.json``
        over ``FALLBACK_GENERATION_DEFAULTS``."""
        cached = getattr(self, "_generation_defaults_cache", None)
        if cached is None:
            local_dir = getattr(self, "local_dir", None)
            cached = {
                **FALLBACK_GENERATION_DEFAULTS,
                **load_generation_defaults(local_dir, stage="thinker_"),
                **load_generation_defaults(local_dir, prefix="talker_", stage="talker_"),
            }
            self._generation_defaults_cache = cached
        return cached

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get(
            "max_output_tokens", self._generation_defaults()["thinker_max_output_tokens"],
        )

    def _get_max_talker_output_tokens(self, **model_kwargs) -> int:
        value = model_kwargs.get(
            "talker_max_output_tokens", self._generation_defaults()["talker_max_output_tokens"],
        )
        # the conductor refuses a cap past the Talker loop's max_iters
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"talker_max_output_tokens must be an integer >= 1; got {value!r}")
        return value

    def _resolve_voice(self, model_kwargs: dict, audio_output: bool) -> str:
        """The Talker speaker; an explicit voice is checked even for text-only output."""
        if "voice" not in model_kwargs and not audio_output:
            return DEFAULT_VOICE
        voice = model_kwargs.get("voice", DEFAULT_VOICE)
        if not isinstance(voice, str) or not voice:
            raise ValueError(f"voice must be a non-empty name; got {voice!r}")
        voices = self.config.talker.speaker_id or {}
        if voice.lower() not in voices:
            supported = ", ".join(sorted(voices)) or "none"
            raise ValueError(f"unknown voice {voice!r}; supported: {supported}")
        return voice

    def request_kwargs(self) -> frozenset[str]:
        stages = ("", "thinker_", "talker_", "code_predictor_")
        return frozenset(
            {stage + knob for stage in stages for knob in SAMPLING_KNOBS}
            | {"ignore_eos", "voice", "max_output_tokens", "talker_max_output_tokens", "seed"}
        )

    def checkpoint_path(self) -> str:
        return self.local_dir

    # -----------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -----------------------------------------------------------------------

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        return {"Thinker": {THINKER_MIXED: {
            "prefill_text", "prefill_audio", "prefill_vision", "thinker_decode", THINKER_TEXT_DECODE,
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

    def rebuild_policies(self) -> dict[str, RebuildPolicy]:
        # Audio, images and video are rebuilt from host copies of the encoders' outputs, one embedding per slot, and
        # for vision one more per deepstack layer. A prompt with a video keys only its leading text, so it's rebuilt
        # only with the prefix cache off for the request or the server. The Talker's KV is built from the Thinker's
        # states, which no rebuild recomputes.
        vision = self.config.vision
        return {"Thinker": RebuildPolicy(
            walks=frozenset({"prefill_text", "prefill_audio", "prefill_vision"}), ids_input="text_inputs",
            sample_output="new_token", loop="thinker_decode_loop", copy_width={
                "prefill_audio": self.config.thinker_text.hidden_size,
                "prefill_vision": vision.out_hidden_size * (1 + len(vision.deepstack_visual_indexes)),
            },
        )}

    def get_graph_walk_graphs(self) -> dict[str, GraphNode | Sequential]:
        """Define all graph walks for the 3-partition architecture.

        Thinker walks:
            prefill_text   - a prompt with no attachment: one Thinker prefill
            prefill_audio  - audio feature encoding, then one Thinker prefill of
                             the clip spliced between the text around it
            prefill_vision - the same for an image or a video
            thinker_decode - autoregressive text token generation, streaming
                             its states to the Talker
            thinker_text_decode - the same for a request with no audio output

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
                # a clip's unpadded log-mel features: its frame count is the
                # tensor's last dim
                input_names=["audio_features"],
                outputs=[GraphEdge(next_node="Thinker", name="audio_embeds")],
            ),
            GraphNode(
                name="Thinker",
                input_names=["audio_embeds", "text_inputs"],
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
                input_names=["vision_embeds", "deepstack", "video_second_per_grid", "image_grid_thw", "text_inputs"],
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

        # -- Thinker decode with no Talker: the sampled token stays on the
        #    device between steps and reaches the client inline --
        thinker_text_decode = Loop(
            name="thinker_decode_loop",
            section=GraphNode(
                name="Thinker",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node="Thinker", name="text_inputs"),
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="text_inputs",
                        output_modality="text",
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
            THINKER_TEXT_DECODE: thinker_text_decode,
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
                    "prefill_vision", "thinker_decode", THINKER_TEXT_DECODE,
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

        A request with no audio out decodes on ``THINKER_TEXT_DECODE``, whose
        token stays on the device between steps. With audio out, the Thinker's
        stream moves the Talker through its prefill walks (``_talker_walk``),
        and the conductor feeds ``talker_decode`` its first input once
        ``talker_last_prefill`` has sampled the first codec token. Code2Wav runs
        once per chunk of codec tokens, until the worker reports the stream's
        final chunk; a count-based end would miss the left-context policy's
        flush pass.
        """
        audio_output = "audio" in output_modalities
        if partition_name == "Thinker":
            thinker = {"audio_output": audio_output}
            prefills = self._build_thinker_prefill_schedule(input_modalities)
            # the last prefill samples the reply's first token
            return [
                WalkStep(walk, bind=bind, step_metadata={
                    **thinker, TEXT_BEFORE: text_before, FINAL_PREFILL: k == len(prefills) - 1,
                })
                for k, (walk, bind, text_before) in enumerate(prefills)
            ] + [WalkStep(self._decode_walk(audio_output), bind={"text_inputs": "new_token"}, step_metadata=thinker)]
        if partition_name not in ("Talker", "Code2Wav"):
            raise ValueError(f"Unknown partition: {partition_name!r}")
        # an explicit voice is checked even for text-only output: an unknown one is a 400 at admission
        voice = self._resolve_voice(model_kwargs, audio_output)
        if not audio_output:
            return []
        if partition_name == "Code2Wav":
            return [WalkStep("code2wav_chunk")]
        talker_max_tokens = self._get_max_talker_output_tokens(**model_kwargs)
        prefill = {"voice": voice, "talker_max_tokens": talker_max_tokens}
        return [
            WalkStep("talker_prefill", step_metadata=prefill),
            WalkStep("talker_last_prefill", step_metadata=prefill),
            WalkStep("talker_decode", step_metadata={"talker_max_tokens": talker_max_tokens}),
        ]

    @staticmethod
    def _decode_walk(audio_output: bool) -> str:
        return "thinker_decode" if audio_output else THINKER_TEXT_DECODE

    # Per-modality walk name and the signals it binds: the walk's input names
    # (primary feature tensor first, then that encoder's auxiliary tensors) to
    # the prompt lists whose entries fill them. Video rides the vision encoder,
    # so its tensors arrive under the vision input names.
    _MM_WALKS: dict[str, tuple[str, dict[str, str]]] = {
        "audio": ("prefill_audio", {"audio_features": "audio_features"}),
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
    # the attachment lists whose entries each prefill in a walk of their own
    MEDIA_SPANS = tuple(sorted({key for _, keys in _MM_WALKS.values() for key in keys.values()}))

    def _build_thinker_prefill_schedule(self, input_modalities: list[str]) -> list[tuple[str, dict[str, str], int]]:
        """The Thinker's prefill walks, an entry per step of ``_thinker_steps``
        (so a prompt of text, a clip and text is one step), each as
        ``(walk_name, {input_name: signal}, text_before)``.

        Step ``k``'s text arrives as signal ``text_inputs.k``, the spans around
        its attachment, ``text_before`` of them leading it; each attachment's
        tensors arrive as signals of their own (`split_spans`), and a modality
        walk binds every tensor its walk's nodes need: the auxiliary tensors
        alongside the primary feature tensor. ``process_prompt`` split the
        prompt against the same plan, and failed at intake any request whose
        tensors don't fill it.
        """
        schedule: list[tuple[str, dict[str, str], int]] = []
        for k, step in enumerate(_thinker_steps(prefill_plan(parts_from_modalities(input_modalities)))):
            bind = {"text_inputs": span_name("text_inputs", k)}
            if step.walk == "prefill_text":
                schedule.append((step.walk, bind, 0))
                continue
            at = next(i for i, part in enumerate(step.parts) if part.modality != TEXT)
            item = step.parts[at]
            bind |= {name: span_name(key, item.index) for name, key in self._MM_WALKS[item.modality][1].items()}
            schedule.append((step.walk, bind, at))
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
           ``audio_features``.
        4. Expand the single ``<|image_pad|>`` / ``<|audio_pad|>`` /
           ``<|video_pad|>`` placeholder in the tokenized text to N copies
           where N = number of patches after spatial merge (this is what
           ``Qwen3OmniMoeProcessor.replace_multimodal_special_tokens`` does
           internally).

        The result holds the text spans and the per-modality tensors the
        Thinker's prefill walks need (`_split_steps`): each prefill step's text
        as one signal (``text_inputs.N``), each attachment's tensors as signals
        of their own (``pixel_values.N``, ...).
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
        # (``ThinkerSubmodule._audio_row``, ``_wrap_vision_input``) and the encoder embeddings
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
        result["audio_features"] = []
        result["video_second_per_grid"] = []
        result["video_grid_thw"] = []
        result["pixel_values_videos"] = []

        # Run image_processor / feature_extractor SEPARATELY for the
        # modality outputs.  These don't touch text_inputs.
        for img in pil_images:
            img_proc = self._processor.image_processor
            # inferred, a 1, 3 or 4 px tall image would be read as channels first
            img_out = img_proc(images=[img], return_tensors="pt", input_data_format="channels_last")
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
            # the clip's valid frames only, so its frame count is the width
            aud_out["input_features"] = (
                aud_out["input_features"]
                .permute(0, 2, 1)[aud_out["attention_mask"].bool()]
                .permute(1, 0)
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
            return ProcessPromptOutput(self._split_steps(result, plan)[0], {})
        kv_layout = self._kv_layout(plan, result, "audio" in output_modalities)
        result, kv_layout = self._split_steps(result, plan, kv_layout)
        return ProcessPromptOutput(result, {"kv_layout": kv_layout})

    def _split_steps(
        self, tensors: NameToTensorList, plan: list[PromptPart], kv_layout: dict | None = None,
    ) -> tuple[NameToTensorList, dict | None]:
        """``tensors`` as the walk plan binds them: step ``k`` of ``_thinker_steps``
        takes its text spans as one signal, ``text_inputs.k`` (``_kv_layout``
        keys its ids spans under it), and each attachment's tensors are signals
        of their own (`split_spans`)."""
        texts = tensors["text_inputs"]
        out = {**tensors, "text_inputs": []}
        for k, step in enumerate(_thinker_steps(plan)):
            out[span_name("text_inputs", k)] = [texts[part.index] for part in step.parts if part.modality == TEXT]
        return split_spans(out, self.MEDIA_SPANS, kv_layout)

    def prompt_tokens(self, tensors: NameToTensorList, kv_layout: dict[str, dict[str, LabelLayout]]) -> int:
        # text_inputs leaves out each attachment, whose span in the Thinker's layout counts its sentinels and rows
        return sum(span.length for span in kv_layout[THINKER_KV]["main"].spans)

    def _kv_layout(self, plan: list[PromptPart], tensors: NameToTensorList, audio_output: bool) -> dict:
        """What each Thinker prefill step writes and, with audio out, what the Talker writes of it.

        The Talker prefills every step's rows except those of a system or earlier
        assistant turn, one prefill a step. It then prefills its assistant prefix,
        and writes a row per decode step.
        """
        from mstar.model.qwen3_omni.submodules import TalkerSubmodule, talker_text_mask

        merge = self.config.vision.spatial_merge_size
        thinker, talker_rows, position = [], [], 0
        for k, step in enumerate(_thinker_steps(plan)):
            talker_rows.append(0)
            for i, part in enumerate(step.parts):
                if part.modality == TEXT:
                    ids = tensors["text_inputs"][part.index]
                    # the step's text is one signal (`_split_steps`)
                    span = Span("ids", step.walk, len(ids), len(ids), span_name("text_inputs", k))
                    kept = int(talker_text_mask(ids, self.config).sum())
                elif part.modality == "audio":
                    # the encoder's rows between two sentinels
                    frames = tensors["audio_features"][part.index].shape[-1]
                    rows = self.config.audio_encoder.tokens_for_frames(frames) + 2
                    span = Span("digest", step.walk, rows, rows, ("audio", part.index))
                    kept = rows
                elif part.modality == "image":
                    image = image_span(tensors["image_grid_thw"][part.index], merge)
                    span = Span("digest", step.walk, image.length, image.advance, ("image", part.index))
                    kept = image.length
                else:
                    grid = tensors["video_grid_thw"][part.index]
                    per_grid = tensors["video_second_per_grid"][part.index].item()
                    rows = int(grid.prod()) // merge ** 2 + 2
                    # the walk moves the counter one past its end sentinel, which follows the grid's largest position
                    positions = get_rope_index_vision(
                        grid, position + 1, self.config.thinker.position_id_per_seconds,
                        spatial_merge_size=merge, seconds_per_grid=per_grid,
                    )
                    advance = int(positions.max()) + 2 - position
                    span = Span("digest", step.walk, rows, advance, ("video", part.index))
                    kept = rows
                thinker.append(span._replace(opens_step=i == 0))
                talker_rows[-1] += kept
                position += span.advance
        talker = [Span("rows", "talker_prefill", kept, kept, "thinker_states") for kept in talker_rows]
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
        # the Thinker's last prefill emits its first text token and
        # talker_last_prefill its first codec frame, so each loop runs one fewer
        return {
            "thinker_decode_loop": max(1, self.get_max_output_tokens(**model_kwargs) - 1),
            "talker_decode_loop": max(1, self._get_max_talker_output_tokens(**model_kwargs) - 1),
        }

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        """The Thinker keys its prompt by text ids, each image and audio clip by
        file, and its replies as they decode."""
        return {
            THINKER_KV: {"main": KVLabel(
                growth=(("thinker_decode", "thinker_decode_loop"), (THINKER_TEXT_DECODE, "thinker_decode_loop")),
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
        # via the same ``ParallelAttention`` / ``ParallelSparseMoeBlock``
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

    def _create_thinker_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ) -> NodeSubmodule:
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
        rules = thinker_model.checkpoint_rules()
        # the shard selector picked the prefix; the rules apply below it
        weights = ((k.removeprefix(rules.prefix), v) for k, v in weights)
        rules.load(thinker_model, weights)
        thinker_model.eval()


        from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
        return ThinkerSubmodule(thinker_model=thinker_model, config=self.config)

    def _create_talker_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_omni.components.talker import (
            Qwen3OmniTalkerModel,
        )
        from mstar.model.qwen3_omni.components.thinker import thinker_remap, thinker_stacked_params

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

        stacked = list(thinker_stacked_params(self.config.thinker_text.num_experts))
        load_hf_weights(
            talker_model, iter(talker_weights),
            stacked_params=stacked,
            name_remapper=thinker_remap,
        )
        talker_model.eval()

        from mstar.model.qwen3_omni.submodules import TalkerSubmodule
        with torch.device("meta"):
            # its inputs and codec embeddings are in the Talker's width, which
            # Qwen3-Omni's code predictor shares (it has no input projection)
            # the frame's embedding sum accumulates in fp32, rounded once
            code_predictor = ResidualCodePredictor(
                self.config.code_predictor, self.config.code_predictor.hidden_size, TalkerSubmodule.MAX_BATCH_SIZE,
                embed_sum_dtype=torch.float32,
            )
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            code_predictor = code_predictor.to(autocast_dtype)
        code_predictor.to_empty(device=device)
        load_hf_weights(
            code_predictor, iter(code_pred_weights),
            stacked_params=stacked,
            name_remapper=thinker_remap,
        )
        code_predictor.consolidate_stacked_weights()
        code_predictor.eval()

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
