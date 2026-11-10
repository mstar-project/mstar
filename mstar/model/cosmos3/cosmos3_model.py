"""Cosmos3Model: NVIDIA Cosmos3 omni generator on the mstar engine.

Cosmos3 is a text-conditioned diffusion model: a dual-pathway Mixture-of-
Transformers DiT denoises image/video (and optionally sound) latents, which a
Wan VAE decodes to pixels. An optional action head extends the same backbone to
robot-action generation.

Nodes (2 for image generation):
    dit          (kv_cache)  - dual-pathway DiT. The understanding (text)
                               tower prefills the conditioning K/V; the
                               generation tower runs the denoise loop, reading
                               that frozen K/V each step (it is timestep-
                               independent, so caching it once is exact).
    vae_encoder  (stateless) - Wan VAE: conditioning image/video -> clean
                               anchor latents, in parallel with the prefill
                               (conditioned requests only).
    vae_decoder  (stateless) - Wan VAE: final latents -> pixels.

Graph walks (image generation):
    prefill    - the understanding tower runs over the text prompt and writes
                 its per-layer K/V (causal self-attention over text).
    image_gen  - an N-step denoising loop. Each iteration the generation tower
                 attends to [frozen text K/V | current generation tokens],
                 predicts flow velocity, and applies one scheduler step; the
                 final latents go to the VAE decoder, which emits the image.

Edge checkpoints add the reasoner (the understanding tower served as a VLM):
    vision_encoder (stateless) - SigLIP2-style tower + 2x2 patch merger:
                                 packed image/video patches -> text-space tokens.
    reasoner       (kv_cache, sampler, rope) - the same transformer instance
                                 as the DiT (one copy of the text weights, the
                                 same KV pool), run as a causal text model.
    prefill_text / prefill_vision - embed the chat-templated prompt (vision
                 tokens spliced in at the media placeholders), write its K/V,
                 sample the first token.
    decode - one token per iteration of decode_loop until EOS / max tokens.

Streaming rollout (opt-in, ``enable_windowed_video``; ported from #198):
    video_gen_ar - the denoise loop run window by window over a long clip.
                 ``chained`` windows pin the previous window's last frames,
                 re-encoded as a clip of their own, as clean conditioning;
                 ``kv`` windows attend block-causally over the
                 committed K/V of earlier windows (one commit iteration per
                 window appends the finished window's clean K/V; the pool's
                 retention policy releases frames past the context horizon at
                 each commit). Each finished window's latents leave the loop on
                 a streaming edge.
    vae_decoder_ar (partition ``window_decoder``) - decodes each window,
                 continuing the stream's decoder cache, while the loop
                 denoises the next one, and emits the video per window
                 (``stream_video``) or assembled.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import PartitionDefinition
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    KVReqConfig,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec, RaggedAttnBackend
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.windowing import WindowSchedule
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import KVLabel, LabelLayout, Model, ProcessPromptOutput, Span, WalkStep
from mstar.model.components.diffusion.unipc import SOLVER_STATE
from mstar.model.components.mrope import layout_extent
from mstar.model.components.video_encode import encode_mp4
from mstar.model.cosmos3 import constants
from mstar.model.cosmos3.components.packing import (
    ACTION_MODE_FORWARD_DYNAMICS,
    ACTION_MODE_INVERSE_DYNAMICS,
    ACTION_MODE_POLICY,
    ACTION_MODES,
    resolve_action_domain_id,
)
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.submodules import (
    ACTION_GEN_LOOP,
    ACTION_VIDEO_GEN_LOOP,
    ATTN,
    ATTN_GEN,
    COND_LABEL,
    DENOISE_LOOP_FOR_WALK,
    IMAGE_GEN_LOOP,
    KV_CACHE,
    REASONER_DECODE_LOOP,
    REASONER_LABEL,
    SAMPLER,
    UNCOND_LABEL,
    VIDEO_GEN_AR_LOOP,
    VIDEO_GEN_LOOP,
    VIDEO_SOUND_GEN_LOOP,
    Cosmos3AudioDecoderSubmodule,
    Cosmos3DiTSubmodule,
    Cosmos3ReasonerSubmodule,
    Cosmos3VAEDecoderARSubmodule,
    Cosmos3VAEDecoderSubmodule,
    Cosmos3VAEEncoderSubmodule,
    Cosmos3VisionEncoderSubmodule,
    frame_tokens,
    latent_frames,
    sound_frames,
)
from mstar.model.keys import MIXED, ROPE, VISION_ATTN
from mstar.model.media_limits import UndecodableMediaError, VideoBudget
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    TEXT,
    MediaEncoder,
    PromptPart,
    check_attachments,
    media_prefill_step,
    media_walks,
    parts_from_modalities,
)
from mstar.model.submodule_base import ChunkingPolicy
from mstar.model.utils import load_generation_defaults
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

logger = logging.getLogger(__name__)

DIT_NODE = "dit"
VAE_ENCODER_NODE = "vae_encoder"
VAE_DECODER_NODE = "vae_decoder"
AUDIO_DECODER_NODE = "audio_decoder"
VAE_DECODER_AR_NODE = "vae_decoder_ar"
VISION_ENCODER_NODE = "vision_encoder"
# A mixed step's prompt rows, text or media, take the prefill kernels
REASONER_MIXED_SPLIT = CombinedSplit(
    MIXED, prefill_walks=(constants.REASONER_PREFILL_WALK, constants.REASONER_PREFILL_VISION_WALK),
)
REASONER_NODE = "reasoner"

# every request key the model reads; the Cosmos3 adapters map onto these names
_REQUEST_KWARGS = frozenset({
    # reasoner (text output)
    "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "penalize_prompt",
    "ignore_eos", "max_output_tokens", "seed", "enable_thinking", "system_prompt",
    "video_num_frames", "video_fps",
    # generation (image / video / action)
    "size", "width", "height", "num_frames", "fps", "num_inference_steps", "guidance_scale",
    "guidance_interval", "flow_shift", "use_karras_sigmas", "max_sequence_length",
    "negative_prompt", "use_system_prompt", "use_resolution_template", "use_duration_template",
    "condition_frame_indexes_vision", "condition_video_keep",
    "action_mode", "action_chunk_size", "action_fps", "action", "domain_id", "domain_name",
    "raw_action_dim", "robot_state", "generate_sound", "sound_gen", "sound_duration",
    "window_mode", "window_frames", "overlap_frames", "context_frames", "stream_video",
    "session_id", "resume_session", "end_session", "session_timeout_s",
})

# the loops built with max_iters = max_inference_steps, one iteration per step; not the windowed one
DENOISE_LOOPS = (IMAGE_GEN_LOOP, VIDEO_GEN_LOOP, VIDEO_SOUND_GEN_LOOP, ACTION_GEN_LOOP, ACTION_VIDEO_GEN_LOOP)


def _int_kwarg(mk: dict, key: str, default, lo: int | None = None, hi: int | None = None):
    """A request integer (a whole float such as ``16.0`` counts), strictly typed;
    ``ValueError`` outside ``[lo, hi]``."""
    value = mk.get(key)
    if value is None:
        return default
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Cosmos3 {key} must be an integer; got {value!r}.")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        bounds = f"in [{lo}, {hi}]" if hi is not None else f">= {lo}"
        raise ValueError(f"Cosmos3 {key} must be {bounds}; got {value}.")
    return value


def _num_kwarg(mk: dict, key: str, default, positive: bool = False):
    """A request number (int or float), strictly typed; finite, optionally > 0."""
    value = mk.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Cosmos3 {key} must be a finite number; got {value!r}.")
    if positive and not (math.isfinite(value) and value > 0):
        raise ValueError(f"Cosmos3 {key} must be positive and finite; got {value!r}.")
    if not math.isfinite(value):
        raise ValueError(f"Cosmos3 {key} must be a finite number; got {value!r}.")
    return float(value)


def _bool_kwarg(mk: dict, key: str, default=False):
    """A request boolean; strings like ``"false"`` are refused, not truthy."""
    value = mk.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Cosmos3 {key} must be true or false; got {value!r}.")
    return value


def _str_kwarg(mk: dict, key: str, default=None):
    """A request string, strictly typed."""
    value = mk.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"Cosmos3 {key} must be a string; got {value!r}.")
    return value


def _parse_size(size) -> tuple[int, int]:
    """``"WxH"`` with positive integer sides, else ``ValueError``."""
    parts = size.lower().split("x") if isinstance(size, str) else []
    if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
        width, height = int(parts[0]), int(parts[1])
        if width > 0 and height > 0:
            return width, height
    raise ValueError(f"Cosmos3 size must be 'WIDTHxHEIGHT' with positive integers; got {size!r}.")


class Cosmos3Model(Model):
    """NVIDIA Cosmos3 generator implementation."""

    PREPROCESS_ONLY_KWARGS = ("action", "robot_state")
    # Requests prepare at once: the loaders and process_prompt keep no shared
    # state, and their work (image decode, resize, torch ops) releases the GIL.
    # Inline, a burst of image prompts queued ~10 ms each behind one another.
    PREPROCESS_PARALLEL = True
    DECODE_LOOP = REASONER_DECODE_LOOP
    # {denoise loop: iterations}, settled by process_prompt, which alone sees
    # the output modality that picks the loop and its step count
    DENOISE_CAP = "denoise_cap"
    # and the width of a policy request's robot_state, which process_prompt
    # hands the conductor (the values themselves ride the action_inputs edge),
    # and a reasoner prompt's media layout
    INTERNAL_KWARGS = (DENOISE_CAP, "robot_state_dim", "action_shape", MEDIA_LAYOUT)

    PREFILL_WALK = constants.PREFILL_WALK
    PREFILL_COND_WALK = constants.PREFILL_COND_WALK
    PREFILL_COND_VIDEO_WALK = constants.PREFILL_COND_VIDEO_WALK
    IMAGE_GEN_WALK = constants.IMAGE_GEN_WALK
    VIDEO_GEN_WALK = constants.VIDEO_GEN_WALK
    VIDEO_GEN_AR_WALK = constants.VIDEO_GEN_AR_WALK
    VIDEO_DECODE_AR_WALK = constants.VIDEO_DECODE_AR_WALK
    VIDEO_SOUND_GEN_WALK = constants.VIDEO_SOUND_GEN_WALK
    ACTION_GEN_WALK = constants.ACTION_GEN_WALK
    ACTION_VIDEO_GEN_WALK = constants.ACTION_VIDEO_GEN_WALK
    REASONER_PREFILL_WALK = constants.REASONER_PREFILL_WALK
    REASONER_PREFILL_VISION_WALK = constants.REASONER_PREFILL_VISION_WALK
    REASONER_DECODE_WALK = constants.REASONER_DECODE_WALK

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        self._yaml_config_overrides: dict = dict(kwargs)

        self._repo_dir: Path | None = None
        # Byte-faithful streaming detokenizer for the reasoner (built lazily
        # against whichever tokenizer is bound; see ``postprocess``).
        self._detokenizer = None
        self.config: Cosmos3Config = self._load_config()
        self.tokenizer = self._load_tokenizer()
        if self._windowed_serving_enabled() and not self.config.is_distilled:
            logger.warning(
                "Cosmos3 window_mode='kv' is experimental: the checkpoint was not trained "
                "block-causally, and a kv rollout can leave the conditioning image within a "
                "few windows. Prefer window_mode='chained', the default for stream_video."
            )
        # reasoner sampling defaults the checkpoint ships; the config is the fallback
        local = self._repo_dir or Path(model_path_hf)
        self._generation_defaults = (
            load_generation_defaults(local) if self._reasoner_enabled() and local.is_dir() else {}
        )

        self._submodule_cache: dict[str, torch.nn.Module | None] = {}
        # The Wan VAE is shared between the DiT submodule (conditioning encode)
        # and the decoder submodule, so build it once. The transformer is
        # shared between the DiT and reasoner nodes likewise.
        self._vae = None
        self._decoder_only_vaes: dict[str, torch.nn.Module] = {}
        self._encode_vaes: dict[str, torch.nn.Module] = {}
        self._transformer = None

    # ------------------------------------------------------------------
    # Config + tokenizer
    # ------------------------------------------------------------------

    def _ensure_repo(self) -> Path:
        if self._repo_dir is not None:
            return self._repo_dir
        candidate = Path(self.model_path_hf)
        if candidate.exists():
            self._repo_dir = candidate
        else:
            from huggingface_hub import snapshot_download

            self._repo_dir = Path(
                snapshot_download(repo_id=self.model_path_hf, cache_dir=self.cache_dir)
            )
        return self._repo_dir

    def _load_config(self) -> Cosmos3Config:
        if self.skip_weight_loading:
            # Dummy mode still parses a local checkpoint directory's configs
            # (shapes only, no tensors), so structural tests see the served
            # walks; a bare id falls back to the Nano defaults.
            local = Path(self.model_path_hf)
            cfg = Cosmos3Config.from_pretrained(local) if (local / "transformer" / "config.json").exists() \
                else Cosmos3Config()
        else:
            try:
                cfg = Cosmos3Config.from_pretrained(self._ensure_repo())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load Cosmos3 config from %s (%s); using Nano defaults.",
                    self.model_path_hf, exc,
                )
                cfg = Cosmos3Config()

        # Overlay yaml model_kwargs last (so they win over file + defaults).
        if self._yaml_config_overrides:
            valid = {f.name for f in Cosmos3Config.__dataclass_fields__.values()}
            for k, v in self._yaml_config_overrides.items():
                if k in valid:
                    if k in ("image_size_default", "video_size_default") and v is not None:
                        v = tuple(int(x) for x in v)
                    setattr(cfg, k, v)
                else:
                    logger.warning(
                        "Cosmos3Model: yaml model_kwargs key %r is not a Cosmos3Config "
                        "field; ignored.", k,
                    )
        cfg.check_streaming_decode()
        cfg.check_compile()
        return cfg

    def _load_tokenizer(self):
        if self.skip_weight_loading:
            return None
        from transformers import AutoTokenizer

        repo = self._ensure_repo()
        # The published checkpoint ships the text tokenizer under
        # ``text_tokenizer/``; fall back to the repo root for layouts that
        # keep the tokenizer files at the top level (Edge's ``text_tokenizer/``
        # names a tokenizer class the pinned transformers cannot resolve, its
        # root copy resolves to PreTrainedTokenizerFast).
        for sub in (repo / "text_tokenizer", repo):
            try:
                return AutoTokenizer.from_pretrained(str(sub), use_fast=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cosmos3 tokenizer load from %s failed (%s).", sub, exc)
        logger.warning("All Cosmos3 tokenizer sources failed; proceeding without one.")
        return None

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def _reasoner_sampler_spec(self) -> SamplerSpec:
        """The reasoner's sampler; its min-p filter is on when the checkpoint
        defaults min_p, so that default is not refused."""
        return SamplerSpec(
            resource_key=SAMPLER, nodes={REASONER_NODE},
            vocab_size=self.config.vocab_size,
            enable_repetition_penalty=True,
            enable_min_p=bool(getattr(self, "_generation_defaults", {}).get("min_p")),
        )

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The DiT's KV cache and the attention backends over it.

        One cache holds every guidance branch's understanding K/V under its
        own label. Attention is declared twice because the two pathways want
        different backends and the choice cannot be per-layer:

        * ``attn`` (paged FlashInfer) is what the understanding tower's
          prefill writes through — the denoise steps read that K/V back out of
          the pages, so it has to land there — and what a captured denoise
          graph replays against, the dense path being eager-only.
        * ``attn_gen`` (dense FA3) is what an eager denoise step runs: it
          recomputes all of its K/V every step and only reuses the frozen text
          prefix, so the paged path's per-step full-buffer write and
          ``wrapper.plan`` are dead work. ``declare_step`` names one or the
          other per step; both are drop-in for the same declaration.

        ``attention_backend="flashinfer"`` skips the dense spec, which leaves
        every step on the paged path.

        The two specs share one ``PagedKVConfig`` object on purpose: a deployment
        that resizes the cache through ``apply_yaml_overrides`` has to resize
        what the wrappers are planned against too.
        """
        kv_config = PagedKVConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_attention_heads,
        )
        # The reasoner is the same transformer (the DiT's understanding
        # pathway) run as a text model, so it shares the pool and the paged
        # backend; only its sampler is its own.
        kv_nodes = {DIT_NODE, REASONER_NODE} if self._reasoner_enabled() else {DIT_NODE}
        specs: list[NodeResourceSpec] = [
            KVSpec(resource_key=KV_CACHE, nodes=kv_nodes, config=kv_config),
            AttentionSpec(
                resource_key=ATTN,
                nodes=kv_nodes,
                config=AttentionConfig(
                    kv_cache=KV_CACHE, backend=AttnBackend.AUTO,
                    # only the reasoner runs mixed (prompt + decode) steps
                    splits=(REASONER_MIXED_SPLIT,) if self._reasoner_enabled() else (),
                ),
            ),
        ]
        if self._reasoner_enabled():
            specs.append(self._reasoner_sampler_spec())
            # The reasoner's position counter. Its layers apply the 3D mRoPE
            # themselves, from these positions or a media prompt's own.
            specs.append(PositionSpec(
                resource_key=ROPE, nodes={REASONER_NODE}, config=PositionConfig(kv_cache=KV_CACHE),
            ))
            specs.append(self._vision_attention_spec())
        if self.config.attention_backend == "dense_gen":
            specs.append(AttentionSpec(
                resource_key=ATTN_GEN,
                nodes={DIT_NODE},
                config=AttentionConfig(
                    kv_cache=KV_CACHE, backend=AttnBackend.DENSE,
                ),
            ))
        elif self.config.attention_backend != "flashinfer":
            raise ValueError(
                f"Unknown Cosmos3 attention_backend "
                f"{self.config.attention_backend!r} "
                "(expected 'dense_gen' or 'flashinfer')"
            )
        return specs

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        uncond = KVLabel()
        if not self._reasoner_enabled():
            return {KV_CACHE: {COND_LABEL: uncond, UNCOND_LABEL: uncond}}
        main = KVLabel(growth=((self.REASONER_DECODE_WALK, REASONER_DECODE_LOOP),))
        return {KV_CACHE: {COND_LABEL: main, UNCOND_LABEL: uncond}}

    def _vision_attention_spec(self) -> RaggedAttentionSpec:
        """The vision tower's attention, a segment per image or video frame."""
        from mstar.model.cosmos3.submodules import Cosmos3VisionEncoderSubmodule

        reasoner = self.config.reasoner
        vision = reasoner.vision
        # the smallest frame the processors make bounds a replay's segments
        min_patches = min(
            reasoner.image_processor.min_pixels, reasoner.video_processor.min_pixels,
        ) // vision.patch_size ** 2
        return RaggedAttentionSpec(
            resource_key=VISION_ATTN, nodes={VISION_ENCODER_NODE},
            config=RaggedAttentionConfig(
                num_qo_heads=vision.num_attention_heads,
                num_kv_heads=vision.num_attention_heads,
                head_dim=vision.hidden_size // vision.num_attention_heads,
                max_segments_per_step=Cosmos3VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1] // max(min_patches, 1),
                # FA3 varlen takes head_dim 72 as it is; FlashInfer pads it to 128
                # (three pads, a zero fill and a slice a layer). Falls back to
                # FlashInfer where FA3 can't load.
                backend=RaggedAttnBackend.DENSE,
                # the tower keeps its own bf16 and runs outside autocast
                dtype=Cosmos3VisionEncoderSubmodule.BLOCK_LOOP_DTYPE,
            ),
        )

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        # the reasoner's text and media prompt rows share steps with its decode rows
        if not self._reasoner_enabled():
            return {}
        return {REASONER_NODE: {MIXED: set(REASONER_MIXED_SPLIT.prefill_walks) | {self.REASONER_DECODE_WALK}}}

    def get_chunking_policies(self) -> dict[str, tuple[ChunkingPolicy, ...]]:
        # the reasoner's text and media prompts split by its token budget; the
        # mixed walk carries it for the steps a chunk shares with decode rows
        if not self._reasoner_enabled():
            return {}
        from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule

        budget = Cosmos3ReasonerSubmodule.MAX_BATCH_TOKENS
        return {REASONER_NODE: (
            ChunkingPolicy(
                walks=(self.REASONER_PREFILL_WALK, self.REASONER_PREFILL_VISION_WALK),
                chunkable=True, max_batch_tokens=budget,
            ),
            ChunkingPolicy(walks=(MIXED,), max_batch_tokens=budget),
        )}

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {**super().get_loop_caps(**model_kwargs), **(model_kwargs.get(self.DENOISE_CAP) or {})}

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, ResourceReqConfig]:
        """Which cache labels a request is opened with.

        Both guidance branches, unconditionally: a request's guidance regime
        is only settled at prefill (it follows ``guidance_scale``, which
        ``process_prompt`` resolves), and naming a label a request never
        writes costs nothing — labels are created on first write.
        """
        mk = model_kwargs or {}
        # a text request decodes on the reasoner, every other generates media
        if "text" in output_modalities:
            sampling = self.get_sampling_config(REASONER_NODE, mk)
            return {
                KV_CACHE: KVReqConfig(needed_labels=[REASONER_LABEL]),
                SAMPLER: SamplingReqConfig(
                    temperature=sampling.temperature, top_k=sampling.top_k,
                    top_p=sampling.top_p, min_p=sampling.min_p,
                    repetition_penalty=sampling.repetition_penalty,
                    penalize_prompt=sampling.penalize_prompt, ignore_eos=sampling.ignore_eos,
                ),
            }
        return {
            KV_CACHE: KVReqConfig(needed_labels=[COND_LABEL, UNCOND_LABEL]),
        }

    def get_sampling_config(self, node_name: str, model_kwargs: dict | None = None):
        """The reasoner's sampling knobs: OpenAI-standard ``temperature`` /
        ``top_p`` plus ``top_k`` / ``min_p`` / ``repetition_penalty`` /
        ``penalize_prompt`` / ``ignore_eos`` from ``extra_body``. A knob the
        request leaves unset takes the checkpoint's ``generation_config.json``
        value, then the config's. Values pass through unconverted; the
        sampler's ``validate`` refuses a wrong type or range."""
        from mstar.engine.resources.sampler.utils import SamplingConfig

        mk = model_kwargs or {}
        defaults = {
            "temperature": self.config.reasoner_temperature, "top_k": 0, "top_p": 1.0,
            "min_p": 0.0, "repetition_penalty": 1.0,
            **getattr(self, "_generation_defaults", {}),
            "penalize_prompt": True, "ignore_eos": False,
        }

        def pick(key):
            value = mk.get(key)
            return defaults[key] if value is None else value

        return SamplingConfig(
            vocab_size=self.config.vocab_size,
            temperature=pick("temperature"),
            top_k=pick("top_k"),
            top_p=pick("top_p"),
            min_p=pick("min_p"),
            repetition_penalty=pick("repetition_penalty"),
            penalize_prompt=pick("penalize_prompt"),
            ignore_eos=pick("ignore_eos"),
        )

    def request_kwargs(self) -> frozenset[str]:
        """Every request key this model reads, adapter-mapped ones included."""
        return _REQUEST_KWARGS

    def _vision_encoder(self) -> MediaEncoder:
        """The reasoner's one medium: the vision encoder's tokens fill the
        prompt's image and video pads, in prompt order, on the scaffold's
        ``prefill_vision`` walk (beside the DiT's own ``prefill*`` walks)."""
        return MediaEncoder("vision", VISION_ENCODER_NODE, ("pixel_values",), "vision_embeds")

    def _reasoner_enabled(self) -> bool:
        """Whether the reasoner walks (and the vision_encoder + reasoner
        nodes) are served: the checkpoint ships the vision tower, and the
        deployment did not switch it off (``enable_reasoner``)."""
        return bool(self.config.serves_reasoner and self.config.enable_reasoner)

    def _sound_serving_enabled(self) -> bool:
        """Whether the opt-in sound walk (and its audio_decoder node) is served.

        Requires the model capability (``sound_gen``), the serving knob
        (``enable_sound``, yaml-overridable), and — with real weights — the
        checkpoint's ``sound_tokenizer/`` component."""
        if not (self.config.sound_gen and self.config.enable_sound):
            return False
        if self.skip_weight_loading:
            return True
        return (self._ensure_repo() / "sound_tokenizer" / "config.json").exists()

    def _windowed_serving_enabled(self) -> bool:
        """Whether the opt-in windowed-AR video walk (and its vae_decoder_ar
        node + streaming decoder partition) is served."""
        return bool(self.config.enable_windowed_video)

    def get_default_sharding_config(self) -> ShardingConfig:
        # The DiT supports tensor parallelism: per layer the attention heads and
        # the MLP intermediate dim shard across ranks, the residual stream stays
        # full, and the row-parallel out/down projections all-reduce. Signals
        # between nodes stay replicated (empty shard_dim) — the sharding is
        # in-module, Megatron-style. The VAE decoder runs un-sharded on one rank.
        return ShardingConfig(
            groups=[], tp_enabled_nodes={DIT_NODE}, shard_dim={},
            sp_enabled_nodes={DIT_NODE},
        )

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # prefill: the understanding tower runs over the text prompt and writes
        # its conditioning K/V. No graph output — completion notifies the
        # conductor, and the generation loop reads the K/V from the shared cache.
        prefill = GraphNode(
            name=DIT_NODE,
            input_names=["text_inputs", "action_inputs"],
            outputs=[],
        )

        # prefill_cond: the DiT prefills the text prompt while, in parallel, the
        # vae_encoder node encodes the conditioning image into the clean anchor
        # latents that seed the denoise loop. The latents come back to the
        # conductor as a persist signal and enter the generation walk as its
        # ``cond_latents`` input edge.
        def _prefill_cond_walk(cond_input: str) -> Parallel:
            return Parallel(
                [
                    GraphNode(name=DIT_NODE, input_names=["text_inputs", "action_inputs"], outputs=[]),
                    GraphNode(
                        name=VAE_ENCODER_NODE,
                        input_names=[cond_input],
                        outputs=[
                            GraphEdge(
                                next_node=EMPTY_DESTINATION,
                                name="cond_latents",
                                persist=True,
                            ),
                        ],
                    ),
                ]
            )

        prefill_cond = _prefill_cond_walk("image_inputs")
        # prefill_cond_video: video conditioning (action inverse-dynamics /
        # video-to-video) encodes the request video instead.
        prefill_cond_video = _prefill_cond_walk("video_inputs")

        # image_gen: denoising loop -> VAE decode -> emit image. The loop body
        # threads the latents + the scheduler's state back to itself each iteration;
        # on the final iteration the latents route to the decoder. max_iters is an
        # upper bound — each request caps the loop at its own denoise-step count
        # (``_denoise_cap``), so one graph serves image and video (and any
        # per-request num_inference_steps) without a rebuild.
        # image_gen and video_gen are the same denoise loop + VAE decode; they
        # differ only in the emitted modality (one frame vs an encoded clip), so
        # the request's output modality selects between them.
        def _gen_walk(loop_name: str, emit_name: str, modality: str) -> Sequential:
            return Sequential(
                [
                    Loop(
                        name=loop_name,
                        # Async (speculative) scheduling pre-dispatches each
                        # request's next denoise step; its yield-away lets other
                        # ready requests join, so concurrent requests still batch
                        # into one forward (see can_batch/forward_batched).
                        # ``cond_latents`` is the loop's external input (no
                        # loop-back edge produces it): the vae_encoder node's
                        # persist signal enters once at the walk transition —
                        # empty for unconditioned requests — and the loop
                        # re-presents the same worker-local tensor each
                        # iteration (no per-step transfer).
                        section=GraphNode(
                            name=DIT_NODE,
                            input_names=self._denoise_inputs("latents"),
                            outputs=self._denoise_loop_back("latents"),
                            enable_async_scheduling=True,
                        ),
                        max_iters=self.config.max_inference_steps,
                        outputs=[
                            GraphEdge(next_node=VAE_DECODER_NODE, name="latents"),
                        ],
                    ),
                    GraphNode(
                        name=VAE_DECODER_NODE,
                        input_names=["latents"],
                        outputs=[
                            GraphEdge(
                                next_node=EMIT_TO_CLIENT,
                                name=emit_name,
                                output_modality=modality,
                            ),
                        ],
                    ),
                ]
            )

        image_gen = _gen_walk(IMAGE_GEN_LOOP, "image_output", "image")
        video_gen = _gen_walk(VIDEO_GEN_LOOP, "video_output", "video")

        # video_sound_gen (opt-in): the video denoise loop with a jointly
        # denoised sound band threaded alongside the video latents. On the final
        # iteration the video latents route to the Wan VAE and the sound latents
        # to the AVAE audio decoder; the walk emits both a video and an audio
        # output, which the API layer muxes into one file.
        video_sound_gen = Sequential(
            [
                Loop(
                    name=VIDEO_SOUND_GEN_LOOP,
                    section=GraphNode(
                        name=DIT_NODE,
                        input_names=self._denoise_inputs("latents", "sound_latents"),
                        outputs=self._denoise_loop_back("latents", "sound_latents"),
                        enable_async_scheduling=True,
                    ),
                    max_iters=self.config.max_inference_steps,
                    outputs=[
                        GraphEdge(next_node=VAE_DECODER_NODE, name="latents"),
                        GraphEdge(next_node=AUDIO_DECODER_NODE, name="sound_latents"),
                    ],
                ),
                # The two decoders are independent: each runs as soon as its
                # own latents arrive from the loop.
                Parallel(
                    [
                        GraphNode(
                            name=VAE_DECODER_NODE,
                            input_names=["latents"],
                            outputs=[
                                GraphEdge(
                                    next_node=EMIT_TO_CLIENT,
                                    name="video_output",
                                    output_modality="video",
                                ),
                            ],
                        ),
                        GraphNode(
                            name=AUDIO_DECODER_NODE,
                            input_names=["sound_latents"],
                            outputs=[
                                GraphEdge(
                                    next_node=EMIT_TO_CLIENT,
                                    name="audio_output",
                                    output_modality="audio",
                                ),
                            ],
                        ),
                    ]
                ),
            ]
        )

        # action_gen: like image_gen but the loop body jointly denoises the video
        # and action latents (threaded as two self-edges), and the predicted
        # action — not a decoded video — is what the request emits.
        action_gen = Sequential(
            [
                Loop(
                    name=ACTION_GEN_LOOP,
                    section=GraphNode(
                        name=DIT_NODE,
                        input_names=self._denoise_inputs("latents", "action_latents"),
                        outputs=self._denoise_loop_back("latents", "action_latents"),
                        enable_async_scheduling=True,
                    ),
                    max_iters=self.config.max_inference_steps,
                    # The loop's terminal output is matched into the section by
                    # name (Loop.__post_init__ filters to the section's own output
                    # edges), so it must reuse a loop-back name: on the final
                    # iteration the predicted action latents go to the client
                    # instead of back into the loop.
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="action_latents",
                            output_modality="action",
                        ),
                    ],
                ),
            ]
        )

        # action_video_gen (forward dynamics): the same joint video+action denoise,
        # but the action is the clean condition and the predicted video is decoded
        # and emitted. The loop's terminal output reuses the "latents" loop-back
        # name; on the final iteration the video latents route to the VAE decoder
        # instead of back into the loop.
        action_video_gen = Sequential(
            [
                Loop(
                    name=ACTION_VIDEO_GEN_LOOP,
                    section=GraphNode(
                        name=DIT_NODE,
                        input_names=self._denoise_inputs("latents", "action_latents"),
                        outputs=self._denoise_loop_back("latents", "action_latents"),
                        enable_async_scheduling=True,
                    ),
                    max_iters=self.config.max_inference_steps,
                    outputs=[
                        GraphEdge(next_node=VAE_DECODER_NODE, name="latents"),
                    ],
                ),
                GraphNode(
                    name=VAE_DECODER_NODE,
                    input_names=["latents"],
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="video_output",
                            output_modality="video",
                        ),
                    ],
                ),
            ]
        )

        walks = {
            self.PREFILL_WALK: prefill,
            self.PREFILL_COND_WALK: prefill_cond,
            self.PREFILL_COND_VIDEO_WALK: prefill_cond_video,
            self.ACTION_VIDEO_GEN_WALK: action_video_gen,
            self.IMAGE_GEN_WALK: image_gen,
            self.VIDEO_GEN_WALK: video_gen,
            self.ACTION_GEN_WALK: action_gen,
        }
        # The sound walk references the audio_decoder node, which only exists
        # (and only needs a node_groups entry) when sound serving is enabled.
        if self._sound_serving_enabled():
            walks[self.VIDEO_SOUND_GEN_WALK] = video_sound_gen
        if self._reasoner_enabled():
            walks.update(self._reasoner_walks())
        if self._windowed_serving_enabled():
            walks.update(self._windowed_walks())
        return walks

    def _denoise_inputs(self, *bands: str) -> list[str]:
        """A denoise loop body's inputs: what it advances, its scheduler's state and
        the external ``cond_latents``."""
        return [*bands, *self._solver_state(), "cond_latents"]

    def _denoise_loop_back(self, *bands: str) -> list[GraphEdge]:
        return [GraphEdge(next_node=DIT_NODE, name=name) for name in (*bands, *self._solver_state())]

    def _solver_state(self) -> tuple[str, ...]:
        # the distilled SDE sampler carries nothing between steps but the sample
        return () if self.config.distilled_sigmas else SOLVER_STATE

    def _windowed_walks(self) -> dict[str, GraphSection]:
        """Windowed AR video: the same denoise loop, run window by window. On
        each window's last iteration the DiT emits the finished window's
        latents on a streaming edge; the vae_decoder_ar node — its own
        partition, so it decodes window k while the loop denoises window k+1
        — consumes them one window per chunk and emits the video."""
        video_gen_ar = Sequential(
            [
                Loop(
                    name=VIDEO_GEN_AR_LOOP,
                    section=GraphNode(
                        name=DIT_NODE,
                        input_names=self._denoise_inputs("latents"),
                        outputs=[
                            *self._denoise_loop_back("latents"),
                            StreamingGraphEdge(
                                next_node=VAE_DECODER_AR_NODE,
                                name="window_latents",
                                target_partition=constants.WINDOW_DECODER_PARTITION,
                            ),
                        ],
                        enable_async_scheduling=True,
                    ),
                    # kv mode runs one extra (commit) iteration per window.
                    max_iters=self.config.max_windows * (self.config.max_inference_steps + 1),
                    outputs=[],
                ),
            ]
        )
        # The decoder partition's walk must be the bare consumer node (the
        # streaming consumer lookup resolves the edge's node from a top-level
        # GraphNode section).
        video_decode_ar = GraphNode(
            name=VAE_DECODER_AR_NODE,
            input_names=["window_latents"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="video_output",
                    output_modality="video",
                ),
            ],
        )
        return {
            self.VIDEO_GEN_AR_WALK: video_gen_ar,
            self.VIDEO_DECODE_AR_WALK: video_decode_ar,
        }

    def get_partitions(self) -> list[PartitionDefinition]:
        if not self._windowed_serving_enabled():
            return super().get_partitions()
        walks = set(self.get_graph_walk_graphs().keys())
        return [
            PartitionDefinition(
                name="default",
                graph_walks=walks - {self.VIDEO_DECODE_AR_WALK},
            ),
            PartitionDefinition(
                name=constants.WINDOW_DECODER_PARTITION,
                graph_walks={self.VIDEO_DECODE_AR_WALK},
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        if not self._windowed_serving_enabled():
            return super().get_partition_topology()
        return PartitionTopology(
            partitions=["default", constants.WINDOW_DECODER_PARTITION],
            connections=[
                Connection(
                    from_partition="default",
                    to_partition=constants.WINDOW_DECODER_PARTITION,
                    edge_name="window_latents",
                    # One committed window per chunk; the decoder manages its
                    # own left context from the latents it has already seen.
                    chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1),
                ),
            ],
        )

    def _reasoner_walks(self) -> dict[str, GraphSection]:
        """The VLM scaffold's walks over the reasoner: ``prefill_text``, the
        ``prefill_vision`` media walk (the vision encoder, then the whole prompt
        as one row with its vision tokens spliced in), and the ``decode`` loop."""
        return media_walks(
            self.config.reasoner.max_position_embeddings, (self._vision_encoder(),), node=REASONER_NODE,
        )

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        input_metadata: dict | None = None,
        **kwargs,
    ) -> NameToTensorList | ProcessPromptOutput:
        if "text" in (output_modalities or []):
            return self._process_reasoner_prompt(
                prompt, input_modalities, tensors or {}, prompt_parts, input_metadata or {}, kwargs,
            )
        # The DiT prefill waits on text_inputs, so no prompt means the empty
        # prompt (as in the reference), not a missing edge. /generate sends an
        # empty text field as None.
        if prompt is None:
            prompt = ""
        p = self._resolve_gen_params(kwargs, input_modalities, output_modalities)
        cap = {self.DENOISE_CAP: self._denoise_cap(output_modalities, p)}
        if self.tokenizer is None:
            # Tokenizer-less fallback used by structural unit tests.
            return self._action_prompt_output({
                "text_inputs": [
                    torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long)
                ]
            }, kwargs, cap)
        # Both the conditional (positive) and unconditional (negative) prompts are
        # tokenized up front; the denoiser reads the second only when guidance is
        # on. Image/video prompts get the chat template + resolution/duration
        # sentences; action prompts are tokenized raw.
        from mstar.model.cosmos3.components.packing import tokenize_prompt

        negative_prompt = _str_kwarg(kwargs, "negative_prompt")
        video = (tensors or {}).get("video_inputs")
        if p.get("action_mode") == ACTION_MODE_INVERSE_DYNAMICS and video and video[0].shape[0] < p["num_frames"]:
            # the encoder takes the clip's first num_frames frames; a shorter one encodes too few latent frames
            raise ValueError(
                f"Cosmos3 inverse_dynamics needs a video of at least num_frames={p['num_frames']} frames, "
                f"got {video[0].shape[0]}; send a smaller num_frames or action_chunk_size."
            )
        # The chat system prompt and the resolution/duration metadata sentences
        # are opt-in, off by default: the model sees the bare user prompt, which
        # matches the reference serving pipeline (its system-prompt and
        # resolution/duration templates default off too). A request may re-enable
        # any of them. Action prompts never use them — they are just the
        # chat-templated user text plus the end-of-text + start-of-generation
        # markers (matching the NVIDIA action references).
        is_action = "action" in output_modalities
        allow_templates = not is_action
        cond_ids, uncond_ids = tokenize_prompt(
            self.tokenizer, prompt, negative_prompt,
            num_frames=p["num_frames"], height=p["height"], width=p["width"], fps=p["fps"],
            use_system_prompt=allow_templates and _bool_kwarg(kwargs, "use_system_prompt"),
            add_resolution_template=allow_templates and _bool_kwarg(kwargs, "use_resolution_template"),
            add_duration_template=allow_templates and _bool_kwarg(kwargs, "use_duration_template"),
            max_sequence_length=p["max_sequence_length"],
        )
        layout = self._gen_layout(p, cond_ids, uncond_ids, input_modalities, tensors or {})
        return self._action_prompt_output({
            "text_inputs": [
                torch.tensor(cond_ids, dtype=torch.long),
                torch.tensor(uncond_ids, dtype=torch.long),
            ]
        }, kwargs, {"kv_layout": {KV_CACHE: layout}, **cap})

    def _gen_layout(
        self, p: dict, cond_ids: list[int], uncond_ids: list[int], input_modalities: list[str], tensors: dict,
    ) -> dict[str, LabelLayout]:
        """Each guidance branch's writes: its prompt at prefill, one denoise step's
        latents as scratch, and in window mode ``kv`` each window's latents once
        they are clean, under the schedule's retention, with the window's denoise
        steps as that span's scratch."""
        config = self.config
        if "video" in input_modalities and "video_inputs" in tensors:
            prefill = self.PREFILL_COND_VIDEO_WALK
        elif p.get("has_image_condition") and "image_inputs" in tensors:
            prefill = self.PREFILL_COND_WALK
        else:
            prefill = self.PREFILL_WALK
        per_frame = frame_tokens(p["height"], p["width"], config)
        schedule, windows = None, ()
        if p.get("window_mode") is not None:
            schedule = WindowSchedule(
                p["total_latent_units"], p["window_latent_units"],
                context_units=p["context_latent_units"], overlap_units=p["overlap_latent_units"],
            )
            if p["window_mode"] == "kv":
                scratch = 0
                for plan in schedule.windows():
                    commit = (plan.commit_end - plan.commit_start) * per_frame
                    denoise = plan.units * per_frame
                    windows += (Span("rows", self.VIDEO_GEN_AR_WALK, commit, commit, "latents", scratch=denoise),)
            else:
                scratch = max(plan.units for plan in schedule.windows()) * per_frame
        elif p.get("action_mode") is not None:
            # the action band is the chunk plus a policy request's robot_state token
            action_tokens = int(p["action_chunk_size"]) + int(p.get("action_state_rows", 0))
            num_frames = int(p.get("num_frames") or int(p["action_chunk_size"]) + 1)
            scratch = latent_frames(num_frames, config) * per_frame + action_tokens
        else:
            scratch = latent_frames(p["num_frames"], config) * per_frame
            if p.get("generate_sound"):
                scratch += sound_frames(p, config)[1]
        layouts = {UNCOND_LABEL: LabelLayout()}
        branches = [(COND_LABEL, cond_ids), (UNCOND_LABEL, uncond_ids)]
        for label, ids in branches if float(p["guidance_scale"]) != 1.0 else branches[:1]:
            layouts[label] = LabelLayout(
                spans=(Span("ids", prefill, len(ids), len(ids), "text_inputs"), *windows), scratch=scratch,
                retention=schedule.retention(per_frame, len(ids)) if windows else None,
            )
        return layouts

    def _action_prompt_output(self, tensors: NameToTensorList, kwargs: dict, metadata: dict | None = None):
        """Put conditioning values on a tensor edge; only their shape is metadata.
        Forward dynamics sends the action chunk as ``action``; a policy request
        may send the robot's current state as ``robot_state``, which rides the
        same edge. Both go in the checkpoint's convention (``inverted_action_dims``)."""
        metadata = dict(metadata or {})
        mode = str(kwargs.get("action_mode", "")).strip().lower()
        if mode == ACTION_MODE_FORWARD_DYNAMICS and kwargs.get("action") is not None:
            action = torch.as_tensor(kwargs["action"])
            tensors["action_inputs"] = [self._invert_action_dims(action, action.shape[-1])]
            metadata["action_shape"] = list(action.shape)
        state = self._robot_state(kwargs)
        if state is not None:
            tensors["action_inputs"] = [self._invert_action_dims(state.unsqueeze(0), state.numel())]
            metadata["robot_state_dim"] = state.numel()
        return ProcessPromptOutput(tensors, metadata) if metadata else tensors

    @staticmethod
    def _robot_state(mk: dict) -> torch.Tensor | None:
        """A request's ``robot_state`` as a float32 row of finite numbers, or None;
        anything else is a ValueError, which the client gets as a 400. A NaN or
        inf would reach every predicted row, since they all attend to it."""
        raw = mk.get("robot_state")
        if raw is None:
            return None
        values = raw.tolist() if hasattr(raw, "tolist") else raw  # a tensor or a numpy array
        if not isinstance(values, (list, tuple)) or any(isinstance(v, (bool, str, bytes)) for v in values):
            raise ValueError(f"Cosmos3 robot_state must be a list of numbers; got {raw!r}.")
        try:
            state = torch.tensor(values, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError):
            raise ValueError(f"Cosmos3 robot_state must be a list of numbers; got {raw!r}.") from None
        if state.dim() != 1 or state.numel() == 0:
            raise ValueError(f"Cosmos3 robot_state must be one row of numbers; got shape {list(state.shape)}.")
        if not torch.isfinite(state).all():
            raise ValueError("Cosmos3 robot_state values must be finite.")
        return state

    def _robot_state_dim(self, mk: dict) -> int:
        """The width of a request's robot_state, 0 without one: from process_prompt's
        metadata on the conductor (the values stay on the data worker)."""
        if mk.get("robot_state_dim"):
            return int(mk["robot_state_dim"])
        state = self._robot_state(mk)
        return 0 if state is None else state.numel()

    @staticmethod
    def _is_policy(mk: dict) -> bool:
        """A policy request: one that says so, or one that sends a robot_state
        without naming a mode."""
        mode = mk.get("action_mode")
        if mode is None:
            return mk.get("robot_state") is not None or bool(mk.get("robot_state_dim"))
        return str(mode).strip().lower() == ACTION_MODE_POLICY

    def _invert_action_dims(self, rows: torch.Tensor, raw_action_dim: int) -> torch.Tensor:
        """``rows`` with the deployment's ``inverted_action_dims`` mapped to 1 - x, the
        convention its checkpoint was trained in (DROID's 1 - gripper); the map is its
        own inverse, so it serves every action a request sends on the way in and every
        predicted action on the way out. Padding columns past ``raw_action_dim`` stay zero."""
        dims = [d for d in self.config.inverted_action_dims if d < raw_action_dim]
        if not dims:
            return rows
        rows = rows.clone() if rows.is_floating_point() else rows.float()
        rows[..., dims] = 1.0 - rows[..., dims]
        return rows

    def _process_reasoner_prompt(
        self, prompt, input_modalities, tensors, prompt_parts, input_metadata, model_kwargs,
    ) -> ProcessPromptOutput:
        """Render the chat prompt, preprocess its media and expand the
        placeholders — everything the reasoner prefill needs besides the
        encoder pass.

        Returns ``text_inputs`` (the token ids) and, with attachments, the
        packed ``pixel_values`` the vision_encoder node consumes, plus the
        ``MEDIA_LAYOUT`` the vision encoder plans its frames by and the
        reasoner splices the encoder's tokens and places its mRoPE positions
        by."""
        from mstar.model.cosmos3.components.reasoner import (
            IMAGE,
            VIDEO,
            expand_placeholders,
            preprocess_image,
            preprocess_video,
            render_chat,
            vision_layout,
        )

        if not self._reasoner_enabled():
            raise ValueError("This Cosmos3 checkpoint/deployment does not serve the reasoner (text output).")
        if self.tokenizer is None:
            raise ValueError("The Cosmos3 reasoner needs the checkpoint tokenizer.")
        reasoner = self.config.reasoner
        parts = parts_from_modalities(
            input_modalities,
            [p.text or "" for p in prompt_parts if p.modality == TEXT] if prompt_parts is not None else prompt,
            [p.role for p in prompt_parts or ()],
        )
        unsupported = {p.modality for p in parts} - {TEXT, IMAGE, VIDEO}
        if unsupported:
            raise ValueError(
                f"The Cosmos3 reasoner accepts image and video attachments only; got {sorted(unsupported)}."
            )
        check_attachments(parts, {
            IMAGE: len(tensors.get("image_inputs", [])), VIDEO: len(tensors.get("video_inputs", [])),
        })

        image_grids, video_grids, patches = [], [], []
        for image in tensors.get("image_inputs", []):
            pv, grid = preprocess_image(image.cpu(), reasoner.image_processor)
            patches.append(pv)
            image_grids.append(grid)
        video_meta = input_metadata.get("video_inputs", [])
        for i, video in enumerate(tensors.get("video_inputs", [])):
            meta = video_meta[i] if i < len(video_meta) else {}
            source_fps = meta.get("average_fps") or meta.get("fps")
            pv, grid = preprocess_video(
                video.cpu(), reasoner.video_processor, source_fps,
                num_frames=_int_kwarg(model_kwargs, "video_num_frames", None, lo=1),
                fps=_num_kwarg(model_kwargs, "video_fps", None, positive=True),
            )
            patches.append(pv)
            video_grids.append(grid)
        # Media patches are packed in prompt order (images and videos as the
        # parts list them), the order the placeholders are expanded in: the
        # layout's runs, which the vision encoder plans its frames by, follow it.
        ordered_patches: list[torch.Tensor] = []
        img_i = vid_i = 0
        for part in parts:
            if part.modality == IMAGE:
                ordered_patches.append(patches[img_i])
                img_i += 1
            elif part.modality == VIDEO:
                ordered_patches.append(patches[len(image_grids) + vid_i])
                vid_i += 1

        text = render_chat(
            self.tokenizer, parts, reasoner,
            enable_thinking=_bool_kwarg(model_kwargs, "enable_thinking", None),
            system_prompt=_str_kwarg(model_kwargs, "system_prompt"),
        )
        text = expand_placeholders(text, self.tokenizer, reasoner, image_grids, video_grids)
        ids = torch.tensor(self.tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.long)
        max_len = int(reasoner.max_position_embeddings)
        if ids.numel() > max_len:
            raise ValueError(
                f"Cosmos3 reasoner prompt is {ids.numel()} tokens, over the {max_len}-token context."
            )
        # always set, so a layout the client sent never reaches the reasoner
        layout = vision_layout(ids, reasoner, image_grids, video_grids, self._vision_encoder().output)
        out: NameToTensorList = {"text_inputs": [ids]}
        if ordered_patches:
            out["pixel_values"] = [torch.cat(ordered_patches, dim=0)]
        walk = self.REASONER_PREFILL_VISION_WALK if ordered_patches else self.REASONER_PREFILL_WALK
        # a vision run spans its merged grid's longer side in positions, not its tokens
        advance = layout_extent(layout, reasoner.vision.spatial_merge_size, ids.numel())
        main = LabelLayout(
            # media rows fill the expanded placeholders, so the span is rows, not ids
            spans=(Span("rows", walk, ids.numel(), advance, "text_inputs"),),
            growth=((REASONER_DECODE_LOOP, 1),),
        )
        return ProcessPromptOutput(out, {
            MEDIA_LAYOUT: layout,
            "kv_layout": {KV_CACHE: {REASONER_LABEL: main, UNCOND_LABEL: LabelLayout()}},
        })

    def load_video(self, filepath: str, device: str):
        """Decode a conditioning / reasoner video to ``[T, C, H, W]`` in [0, 1].

        torchcodec (the base implementation) needs system FFmpeg shared
        libraries; where they are absent or refuse the file, PyAV — which
        ships its own — decodes the same frames. The metadata carries the
        frame rate under ``average_fps`` either way (the reasoner's frame
        sampling and timestamps read it)."""
        from mstar.model.base import TensorAndMetadata

        try:
            return super().load_video(filepath, device)
        except (ImportError, RuntimeError, OSError, UndecodableMediaError) as exc:
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            logger.warning("torchcodec video decode failed (%s); decoding %s with PyAV.", reason, filepath)
        import av

        budget = VideoBudget()
        frames = []
        with av.open(filepath) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate or stream.base_rate
            fps = float(rate) if rate else None
            for frame in container.decode(stream):
                budget.add(frame.height, frame.width)
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
        if not frames:
            raise ValueError(f"no video frames decoded from {filepath}")
        video = torch.stack(frames).to(device).float().div_(255.0)
        metadata = {
            "num_frames": len(frames), "average_fps": fps,
            "duration_seconds": (len(frames) / fps) if fps else None,
            "height": int(video.shape[-2]), "width": int(video.shape[-1]),
        }
        return TensorAndMetadata(data=video, metadata=metadata)

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality == "text":
            # One sampled token per chunk, emitted as the token's raw bytes
            # (the Edge tokenizer is byte-level BPE): a multi-byte character
            # split across tokens reassembles client-side, where per-token
            # ``decode`` would emit U+FFFD for each fragment. Special tokens
            # (EOS, chat markup) are dropped, like ``skip_special_tokens``.
            from mstar.model.utils import ByteLevelDetokenizer

            if self._detokenizer is None or self._detokenizer.tokenizer is not self.tokenizer:
                self._detokenizer = ByteLevelDetokenizer(self.tokenizer)
            ids = output.reshape(-1).tolist()
            if not (request_kwargs or {}).get("ignore_eos"):
                return self._detokenizer.to_bytes(ids)
            # under ignore_eos the end token ended nothing and stays, one chunk per token
            eos = self.config.reasoner.eos_token_id
            return b"".join(
                self.tokenizer.decode([i]).encode("utf-8") if i == eos else self._detokenizer.to_bytes([i])
                for i in ids
            )
        if modality == "image":
            import io
            import os

            from PIL import Image

            # The decoder emits 8-bit frames [B, C, T, H, W]; take the first one.
            x = output
            if x.ndim == 5:
                x = x[0, :, 0]
            elif x.ndim == 4:
                x = x[0]
            arr = x.permute(1, 2, 0).cpu().numpy()  # H, W, C uint8
            buf = io.BytesIO()
            # PNG is lossless at every compression level, so the level only trades
            # encode time for file size. PIL defaults to 6, which spends ~0.75 s on a
            # 720p frame and dominates the serving latency. Level 0 (no deflate) is
            # the fastest and matches what the OpenAI image endpoint emits at full
            # quality; the decoded pixels are identical regardless. Override with
            # COSMOS3_PNG_COMPRESS for A/B.
            compress_level = int(os.environ.get("COSMOS3_PNG_COMPRESS", "0"))
            Image.fromarray(arr).save(buf, format="PNG", compress_level=compress_level)
            return buf.getvalue()
        if modality == "video":
            import os

            # The decoder emits 8-bit frames [T, C, H, W], torchcodec's layout;
            # encode all of them as an H.264 mp4. The frames already reflect the
            # request fps (it modulates the temporal positions during generation),
            # and the container carries that same rate so playback runs at the
            # requested speed.
            #
            # CRF 18 keeps the H.264 output near-visually-lossless; libx264
            # otherwise defaults to 23, which is visibly lossier. The "ultrafast"
            # preset and multithreading (threads=0) target the same CRF/quality
            # but encode several times faster than libx264's default "medium"
            # preset, which otherwise dominates the serving latency for a
            # many-frame clip. The preset is overridable via COSMOS3_X264_PRESET.
            frames = output.cpu()
            fps = float((request_kwargs or {}).get("fps", self.config.fps))
            preset = os.environ.get("COSMOS3_X264_PRESET", "ultrafast")
            try:
                # Preferred: torchcodec (torchvision >= 0.27 removed write_video).
                from torchcodec.encoders import VideoEncoder

                encoded = VideoEncoder(frames, frame_rate=fps).to_tensor(
                    "mp4",
                    codec="libx264",
                    crf=18,
                    preset=preset,
                    extra_options={"threads": "0"},
                )
                data = encoded.numpy().tobytes()
            except Exception as exc:  # noqa: BLE001 — torchcodec raises RuntimeError/OSError without FFmpeg
                # Fallback for environments without a loadable torchcodec (no
                # FFmpeg shared libraries on the host, or the older
                # decode-only build that lacks VideoEncoder): PyAV, whose
                # wheel bundles FFmpeg with libx264, at the same CRF/preset.
                if not isinstance(exc, ImportError):
                    logger.warning(
                        "Cosmos3 video encode: torchcodec unavailable (%s); encoding with PyAV",
                        str(exc).splitlines()[0][:160],
                    )
                # NTSC rates exactly (encode_mp4's denominator is 1001)
                data = encode_mp4(
                    frames.permute(0, 2, 3, 1).numpy(), fps,
                    options={"crf": "18", "preset": preset, "threads": "0"}, pad_even=True,
                )
            return data
        if modality == "action":
            # The predicted action latents [1, chunk, action_dim] -> [chunk,
            # action_dim] float32 bytes. Columns beyond the request's
            # raw_action_dim are zero padding (the client keeps the first
            # raw_action_dim, the real action width for its embodiment). A
            # policy request's robot_state token is its own input, so it is
            # dropped, and every mode's rows go back to the robot's convention.
            mk = request_kwargs or {}
            x = output[0] if output.ndim == 3 else output
            x = x.detach().to(torch.float32)
            state_dim = self._robot_state_dim(mk) if self._is_policy(mk) else 0
            x = self._invert_action_dims(x[1 if state_dim else 0:], int(mk.get("raw_action_dim") or state_dim))
            return x.cpu().numpy().tobytes()
        if modality == "audio":
            # The audio decoder emits a [channels, samples] waveform in [-1, 1];
            # the serving convention for audio is headerless interleaved 16-bit
            # PCM (the API layer wraps it with the model's sample rate).
            x = output[0] if output.ndim == 3 else output
            pcm = (x.detach().to(torch.float32).clamp(-1, 1) * 32767.0).round().to(torch.int16)
            return pcm.T.contiguous().cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Cosmos3: {modality!r}")

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return int(self.config.sound_sample_rate)

    def get_output_audio_channels(self, modality: str = "audio") -> int:
        return 2

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def _resolve_gen_params(
        self, model_kwargs: dict | None, input_modalities: list[str], output_modalities: list[str],
    ) -> dict:
        """Resolve the per-request generation knobs (size, steps, guidance, …)
        from request ``model_kwargs``, applying defaults. Used by both
        ``process_prompt`` (for resolution-aware tokenization) and the forward-
        pass metadata, so the two stay consistent."""
        mk = model_kwargs or {}
        is_video_request = "video" in (output_modalities or [])
        default_size = self.config.image_size_default
        if is_video_request and self.config.video_size_default is not None:
            default_size = self.config.video_size_default
        width, height = int(default_size[0]), int(default_size[1])
        if mk.get("size") is not None:
            width, height = _parse_size(mk["size"])

        # Action requests resolve their mode up front: it switches the frame,
        # step, guidance and flow-shift defaulting below to the action recipe.
        # A request that sends the robot's state without a mode is a policy request.
        state_dim = self._robot_state_dim(mk)
        action_mode = _str_kwarg(mk, "action_mode")
        if action_mode is None and state_dim and "action" in (output_modalities or []):
            action_mode = ACTION_MODE_POLICY
        if action_mode is not None:
            action_mode = action_mode.strip().lower()
            if action_mode not in ACTION_MODES:
                raise ValueError(
                    f"Unsupported Cosmos3 action_mode={mk.get('action_mode')!r}; "
                    f"expected one of {sorted(ACTION_MODES)}."
                )
        if state_dim and action_mode != ACTION_MODE_POLICY:
            raise ValueError("Cosmos3 robot_state is the input of a policy request (action_mode=policy).")

        if action_mode is not None:
            # An action request predicts one action token per frame, so the
            # frame count and the chunk length are coupled (num_frames = chunk
            # or chunk + 1) and default off each other; when neither is sent,
            # the policy checkpoint's chunk for a policy request, else 16 rows
            # (the reference action default). The chunk is the rows the reply
            # carries: a policy request's robot_state is one more token, clean
            # in front of them at frame 0, so then num_frames = chunk + 1.
            raw_chunk = _int_kwarg(mk, "action_chunk_size", None)
            raw_frames = _int_kwarg(mk, "num_frames", None)
            if raw_chunk is not None:
                action_chunk = raw_chunk
            elif raw_frames is not None:
                action_chunk = raw_frames - 1
            elif action_mode == ACTION_MODE_POLICY and self.config.policy_action_chunk_size:
                action_chunk = int(self.config.policy_action_chunk_size)
            else:
                action_chunk = 16
            if action_chunk <= 0:
                raise ValueError(
                    f"Cosmos3 action_chunk_size must be positive, got {action_chunk}."
                )
            num_frames = raw_frames if raw_frames is not None else action_chunk + 1
            if state_dim and num_frames != action_chunk + 1:
                raise ValueError(
                    "A Cosmos3 policy request with a robot_state requires num_frames to be "
                    f"action_chunk_size + 1 (the state is a token at frame 0); got num_frames={num_frames}, "
                    f"action_chunk_size={action_chunk}."
                )
            if num_frames not in (action_chunk, action_chunk + 1):
                raise ValueError(
                    "Cosmos3 action requests require num_frames to equal "
                    "action_chunk_size or action_chunk_size + 1; got "
                    f"num_frames={num_frames}, action_chunk_size={action_chunk}."
                )
            # A bare action request runs the 480p tier (832x480) — the released
            # policy serving resolution, and the tier whose training shift the
            # action flow-shift default (5.0) matches.
            if "size" not in mk and "width" not in mk and "height" not in mk:
                width, height = 832, 480
        else:
            action_chunk = None
            # A video request without an explicit frame count gets the video
            # default (>1); image requests stay single-frame.
            default_frames = (
                self.config.num_frames_video if "video" in (output_modalities or []) else 1
            )
            num_frames = _int_kwarg(mk, "num_frames", default_frames, lo=1)
        # The cookbook step counts differ per mode (image 50, video 35, action
        # 30 — configs override the action count per checkpoint); default by
        # mode and let the request override. The denoise loop's cap is this
        # many steps (``_denoise_cap``); the conductor refuses one past the
        # loop's static max_iters.
        if action_mode is not None:
            default_steps = self.config.num_inference_steps_action
        elif num_frames > 1:
            default_steps = self.config.num_inference_steps_video
        else:
            default_steps = self.config.num_inference_steps
        steps = _int_kwarg(
            mk, "num_inference_steps", default_steps, lo=1,
        )
        default_guidance = (
            self.config.guidance_scale_action if action_mode is not None else self.config.guidance_scale
        )
        if self.config.distilled_sigmas:
            steps, default_guidance = self._resolve_distilled_params(
                mk, steps, action_mode, input_modalities, output_modalities,
            )
        default_fps = self.config.fps
        if action_mode == ACTION_MODE_POLICY and self.config.policy_conditioning_fps:
            default_fps = self.config.policy_conditioning_fps
        params = {
            "width": _int_kwarg(mk, "width", width, lo=1),
            "height": _int_kwarg(mk, "height", height, lo=1),
            "num_frames": num_frames,
            "fps": _num_kwarg(mk, "fps", float(default_fps), positive=True),
            # sign is not checked; guidance semantics are the recipe's
            "guidance_scale": _num_kwarg(mk, "guidance_scale", float(default_guidance)),
            "num_inference_steps": steps,
            "has_image_condition": "image" in (input_modalities or []),
            "use_karras_sigma": _bool_kwarg(mk, "use_karras_sigmas", None),
            # Prompt-token truncation cap (reference serving default 4096), request-overridable
            "max_sequence_length": _int_kwarg(
                mk, "max_sequence_length", constants.DEFAULT_MAX_SEQUENCE_LENGTH, lo=1,
            ),
        }
        # the packing divides by it, and postprocess muxes at it
        if not math.isfinite(params["fps"]) or params["fps"] <= 0:
            raise ValueError(f"Cosmos3 fps must be positive and finite; got {mk.get('fps')!r}.")
        latent_px = self.config.vae.scale_factor_spatial
        if params["width"] < latent_px or params["height"] < latent_px:
            raise ValueError(
                f"Cosmos3 width and height must be at least {latent_px} px (one latent pixel), "
                f"got {params['width']}x{params['height']}."
            )
        frames_per_latent = self.config.vae.scale_factor_temporal
        if params["has_image_condition"] and action_mode is None and num_frames <= frames_per_latent:
            # the image pins the first latent frame, so one latent frame leaves nothing to denoise
            raise ValueError(
                f"Cosmos3 image conditioning requires a video request (num_frames > {frames_per_latent})."
            )
        # Video-to-video: a non-action video input pins clean conditioning
        # latent frames taken from the request video (reference recipe defaults:
        # indexes (0, 1), keep "first", flow_shift 10.0). Validated here so a
        # malformed request fails at submission rather than mid-denoise.
        has_video_condition = "video" in (input_modalities or []) and action_mode is None
        if has_video_condition:
            from mstar.model.cosmos3.components.packing import normalize_condition_frame_indexes

            if num_frames <= 1:
                raise ValueError("Cosmos3 video conditioning requires a video request (num_frames > 1).")
            indexes = normalize_condition_frame_indexes(
                mk.get("condition_frame_indexes_vision"),
                constants.DEFAULT_CONDITION_FRAME_INDEXES_VISION,
            )
            latent_frames = 1 + (num_frames - 1) // self.config.vae.scale_factor_temporal
            if indexes[-1] >= latent_frames:
                raise ValueError(
                    f"Cosmos3 condition_frame_indexes_vision {indexes} is outside the latent "
                    f"video ({latent_frames} latent frames for num_frames={num_frames})."
                )
            keep = _str_kwarg(mk, "condition_video_keep", constants.DEFAULT_CONDITION_VIDEO_KEEP).strip().lower()
            if keep not in ("first", "last"):
                raise ValueError("Cosmos3 condition_video_keep must be 'first' or 'last'.")
            params["has_video_condition"] = True
            params["condition_frame_indexes_vision"] = indexes
            params["condition_video_keep"] = keep
        # Text-to-image (single frame, no visual conditioning) follows the
        # reference Cosmos3 t2i recipe: classifier-free guidance only on the
        # timestep interval [400, 1000] (outside it the denoise step runs the
        # conditional branch alone) and flow_shift 3.0. Request kwargs override;
        # action modes default to the action flow shift, video-to-video to the
        # reference V2V flow shift; other image-conditioned / video paths keep
        # their own defaults (full CFG, scheduler-config flow_shift).
        is_t2i = num_frames == 1 and not params["has_image_condition"] and action_mode is None
        fs = _num_kwarg(mk, "flow_shift", None, positive=True)
        if fs is None and action_mode is not None:
            fs = self.config.flow_shift_action
        if fs is None and is_t2i:
            fs = self.config.flow_shift_image
        if fs is None and has_video_condition:
            fs = constants.V2V_DEFAULT_FLOW_SHIFT
        if fs is None and num_frames > 1:
            # Plain t2v / i2v: the deployment's video shift (Edge: 12.0), else
            # the checkpoint scheduler's own.
            fs = self.config.flow_shift_video
        if fs is not None:
            params["flow_shift"] = float(fs)
        gi = mk.get("guidance_interval")
        if gi is not None and (
            not isinstance(gi, (list, tuple)) or len(gi) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in gi)
        ):
            raise ValueError(f"Cosmos3 guidance_interval must be two numbers [low, high]; got {gi!r}.")
        if gi is None and is_t2i:
            gi = (400.0, 1000.0)
        if gi is None and action_mode is not None:
            gi = self.config.guidance_interval_action
        if gi is not None:
            params["guidance_interval"] = (float(gi[0]), float(gi[1]))
        # Action requests must name their embodiment explicitly — the domain
        # conditions the action pathway, and a silent default would predict
        # actions for the wrong robot — except a policy request to a policy
        # checkpoint that names the one it was trained for. ``domain_name``
        # resolves through the published embodiment table; a numeric
        # ``domain_id`` wins. The raw action width is likewise required (a
        # robot_state or a forward-dynamics ``action`` array carries it) and
        # bounded by the model's padded action dim.
        if action_mode is not None:
            params["action_mode"] = action_mode
            params["action_chunk_size"] = action_chunk
            domain_name = mk.get("domain_name")
            if domain_name is None and mk.get("domain_id") is None and action_mode == ACTION_MODE_POLICY:
                domain_name = self.config.policy_domain_name
            params["domain_id"] = resolve_action_domain_id(
                mk.get("domain_id"), domain_name, self.config.num_embodiment_domains
            )
            raw_dim = _int_kwarg(mk, "raw_action_dim", None)
            if raw_dim is None and state_dim:
                raw_dim = state_dim
            is_fd = action_mode == ACTION_MODE_FORWARD_DYNAMICS
            fd_shape = None  # forward dynamics' action rows: [steps, width] or [1, steps, width]
            if is_fd and mk.get("action_shape"):
                fd_shape = list(mk["action_shape"])
            elif is_fd and mk.get("action") is not None:
                try:
                    fd_shape = list(torch.as_tensor(mk["action"]).shape)
                except (TypeError, ValueError, RuntimeError):
                    fd_shape = None
            if raw_dim is None and fd_shape:
                raw_dim = int(fd_shape[-1])
            if raw_dim is None:
                raise ValueError(
                    "Cosmos3 action requests require 'raw_action_dim' (it may be left out when "
                    "a robot_state or a forward-dynamics 'action' array is sent, which carries the width)."
                )
            if not 1 <= raw_dim <= self.config.max_action_dim:
                raise ValueError(
                    f"Cosmos3 raw_action_dim must be in [1, {self.config.max_action_dim}], "
                    f"got {raw_dim}."
                )
            params["raw_action_dim"] = raw_dim
            if is_fd and not (
                fd_shape and len(fd_shape) >= 2 and fd_shape[:-2] in ([], [1]) and fd_shape[-2] >= 1
                and fd_shape[-1] >= raw_dim
            ):
                raise ValueError(
                    f"Cosmos3 forward_dynamics requires an 'action' array of shape [steps, {raw_dim}] "
                    f"with at least one step, got {fd_shape}."
                )
            if state_dim:
                if state_dim != raw_dim:
                    raise ValueError(
                        f"A Cosmos3 robot_state must have raw_action_dim={raw_dim} values, got {state_dim}."
                    )
                params["action_state_rows"] = 1
            if mk.get("action_fps") is not None:
                params["action_fps"] = _num_kwarg(mk, "action_fps", None, positive=True)
        # Opt-in sound generation: video-only (image and action requests carry
        # no sound band), and only when the served checkpoint/config enable it.
        if _bool_kwarg(mk, "generate_sound") or _bool_kwarg(mk, "sound_gen"):
            if num_frames <= 1 or action_mode is not None:
                raise ValueError(
                    "Cosmos3 sound generation is supported only for video requests "
                    "(num_frames > 1, no action mode)."
                )
            if not self._sound_serving_enabled():
                raise ValueError(
                    "Cosmos3 sound generation was requested, but sound serving is "
                    "disabled or the checkpoint has no sound_tokenizer/ component."
                )
            params["generate_sound"] = True
            if mk.get("sound_duration") is not None:
                params["sound_duration"] = _num_kwarg(mk, "sound_duration", None, positive=True)
        self._resolve_window_params(mk, params, num_frames, action_mode, has_video_condition)
        return params

    def _resolve_distilled_params(self, mk, steps, action_mode, input_modalities, output_modalities):
        """The 4-step distilled checkpoints fix the sampler: their sigma list
        sets the step count, guidance is baked into the weights (scale 1), and
        the task is the checkpoint's own (t2i / i2v) — no action, sound,
        video-to-video or windowed modes. Mirrors the reference's
        ``Cosmos3DistilledSetTimestepsStep`` checks."""
        fixed = len(self.config.distilled_sigmas)
        if mk.get("num_inference_steps") is not None and mk["num_inference_steps"] != fixed:
            raise ValueError(
                f"This Cosmos3 checkpoint is distilled: num_inference_steps is fixed at {fixed} "
                f"(got {mk['num_inference_steps']}); leave it unset."
            )
        if mk.get("guidance_scale") is not None and mk["guidance_scale"] != 1.0:
            raise ValueError(
                "This Cosmos3 checkpoint is distilled: classifier-free guidance is baked into the "
                f"weights, guidance_scale must be 1.0 (got {mk['guidance_scale']}); leave it unset."
            )
        if (action_mode is not None or mk.get("generate_sound") or mk.get("sound_gen")
                or mk.get("window_mode") or mk.get("stream_video")):
            raise ValueError(
                "This Cosmos3 checkpoint is distilled for text/image-to-video generation; action, "
                "sound and windowed modes are not available on it."
            )
        if "video" in (input_modalities or []):
            raise ValueError("This Cosmos3 checkpoint is distilled; video conditioning is not available on it.")
        return fixed, 1.0

    def _resolve_window_params(self, mk, params, num_frames, action_mode, has_video_condition) -> None:
        """Opt-in windowed AR video: the clip is generated window by window.
        ``chained`` conditions each window on the previous window's tail
        (full bidirectional denoise per window); ``kv`` runs block-causal
        cross-window attention through committed K/V, with frames older than
        the context horizon released from the cache. Frame-count knobs are
        quantized to latent frames here so the whole pipeline agrees on the
        schedule; validation up front so malformed requests fail at
        submission."""
        window_mode = _str_kwarg(mk, "window_mode")
        if window_mode is None:
            if not _bool_kwarg(mk, "stream_video"):
                return
            # The non-windowed walks emit one video at the very end, so a
            # streamed request runs windowed, in the mode that holds the scene.
            window_mode = "chained"
        window_mode = window_mode.strip().lower()
        if window_mode not in ("chained", "kv"):
            raise ValueError(
                f"Cosmos3 window_mode must be 'chained' or 'kv', got {mk.get('window_mode')!r}."
            )
        if not self._windowed_serving_enabled():
            raise ValueError("Cosmos3 windowed video generation is disabled for this deployment.")
        if num_frames <= 1 or action_mode is not None:
            raise ValueError(
                "Cosmos3 windowed generation requires a video request (num_frames > 1, no action mode)."
            )
        if params.get("generate_sound"):
            raise ValueError("Cosmos3 windowed generation does not support sound generation.")
        if has_video_condition:
            raise ValueError("Cosmos3 windowed generation does not support video conditioning.")
        is_kv = window_mode == "kv"
        if not is_kv and mk.get("context_frames") is not None:
            raise ValueError("Cosmos3 context_frames applies to window_mode='kv' only.")
        tf = self.config.vae.scale_factor_temporal
        default_window = self.config.kv_window_frames_default if is_kv else self.config.window_frames_default
        window_frames = _int_kwarg(mk, "window_frames", default_window)
        if window_frames < 1 + tf:
            raise ValueError(f"Cosmos3 window_frames must be at least {1 + tf}, got {window_frames}.")
        window_units = 1 + (window_frames - 1) // tf
        # kv windows advance without re-pinned overlap — cross-window
        # conditioning flows through the committed K/V, and a zero overlap
        # keeps each commit exactly covering the span its denoise steps wrote.
        default_overlap = 0 if is_kv else self.config.overlap_frames_default
        overlap_frames = _int_kwarg(mk, "overlap_frames", default_overlap, lo=0)
        if is_kv and overlap_frames:
            raise ValueError(
                "Cosmos3 window_mode='kv' does not support overlap_frames; "
                "cross-window conditioning comes from the committed context."
            )
        # frames -> latent frames as the reference's pipelines map them: k
        # latent frames pin 1 + (k - 1) * tf pixel frames
        overlap_units = min((overlap_frames - 1) // tf + 1 if overlap_frames > 0 else 0, window_units - 1)
        if overlap_units:
            # Two clean latent frames are the conditioning floor (the V2V
            # recipe's pin count); a single frame visibly degrades the next
            # window.
            overlap_units = max(overlap_units, 2)
        if overlap_units >= window_units:
            raise ValueError(
                f"Cosmos3 windowed request needs window_frames large enough for its overlap "
                f"(window {window_units} vs overlap {overlap_units} latent frames)."
            )
        context_units = 0
        if is_kv:
            context_frames = _int_kwarg(mk, "context_frames", self.config.context_frames_default)
            if context_frames < 0:
                raise ValueError(f"Cosmos3 context_frames must be >= 0, got {context_frames}.")
            # 0 retains all committed frames (no release).
            if context_frames:
                context_units = 1 + (context_frames - 1) // tf
        # rounded up: the decoder trims the assembled video back to num_frames,
        # but cannot add frames a short schedule never generated
        total_units = 1 + -(-(num_frames - 1) // tf)
        # Sessions: a request may name a session (its last window is kept for
        # a follow-up) and resume one — the stored tail then re-pins the head
        # of window 0 as clean conditioning (the chained overlap, at least the
        # two-frame V2V floor), and the schedule grows by those units so
        # ``num_frames`` stays the count of new frames the client receives.
        session_id = _str_kwarg(mk, "session_id")
        resume = _bool_kwarg(mk, "resume_session")
        end_session = _bool_kwarg(mk, "end_session")
        if resume and not session_id:
            raise ValueError("Cosmos3 resume_session requires a session_id.")
        if end_session and not session_id:
            raise ValueError("Cosmos3 end_session requires a session_id.")
        if resume and params.get("has_image_condition"):
            raise ValueError(
                "Cosmos3 resume_session conditions on the session's last frames; "
                "drop the conditioning image."
            )
        if session_id is not None:
            params["session_id"] = session_id
            # Drop the session once this request is done instead of keeping it.
            params["end_session"] = end_session
            timeout = _num_kwarg(mk, "session_timeout_s", None, positive=True)
            if timeout is not None:
                limit = float(self.config.session_timeout_max_s)
                if timeout > limit:
                    raise ValueError(f"Cosmos3 session_timeout_s must be in (0, {limit:g}]; got {timeout:g}.")
                params["session_timeout_s"] = timeout
        resume_units = max(overlap_units, 2) if resume else 0
        if resume_units and resume_units >= window_units:
            raise ValueError(
                f"Cosmos3 resume_session needs window_frames large enough for its "
                f"{resume_units}-frame conditioning head (window {window_units} latent frames)."
            )
        params["resume_latent_units"] = resume_units
        total_units += resume_units
        # Pad the schedule up to whole windows: a short final window can
        # regenerate just a frame or two off almost pure conditioning, which
        # comes out degraded. The decoder trims the assembled video back to
        # the requested frame count.
        stride = window_units - overlap_units
        if total_units > window_units:
            rem = (total_units - window_units) % stride
            total_units += (stride - rem) % stride
        schedule = WindowSchedule(
            total_units, window_units, context_units=context_units, overlap_units=overlap_units,
        )
        if schedule.num_windows > self.config.max_windows:
            raise ValueError(
                f"Cosmos3 windowed request spans {schedule.num_windows} windows, "
                f"over the served limit of {self.config.max_windows}."
            )
        params["window_mode"] = window_mode
        params["window_latent_units"] = window_units
        params["overlap_latent_units"] = overlap_units
        params["context_latent_units"] = context_units
        params["total_latent_units"] = total_units
        params["num_windows"] = schedule.num_windows
        params["stream_video"] = _bool_kwarg(mk, "stream_video")
        # Every window after the first is conditioned generation, which the
        # reference recipe runs at the V2V flow shift; one shift for all
        # windows keeps the per-window schedules consistent. A deployment's
        # video shift (Edge: 12.0) or a request flow_shift wins.
        params.setdefault("flow_shift", constants.V2V_DEFAULT_FLOW_SHIFT)

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        """A media request prefills, then runs its denoise walk; a text request
        runs the reasoner's prefill, then its decode loop fed the first token.
        Every step carries the request's resolved params.

        The windowed decoder partition runs its decode walk once per window the
        stream brings, for every request, text ones included: a walk the
        partition does not serve would never complete there and the request
        would hang after its last token. A non-windowed request leaves it idle
        until the stream's terminal flush, which it skips.
        """
        is_text = "text" in output_modalities
        if partition_name == constants.WINDOW_DECODER_PARTITION:
            params = {} if is_text else self._resolve_gen_params(model_kwargs, input_modalities, output_modalities)
            return [WalkStep(self.VIDEO_DECODE_AR_WALK, step_metadata=params)]
        if is_text:
            if not self._reasoner_enabled():
                raise ValueError("This Cosmos3 deployment does not serve the reasoner (text output).")
            knobs = {
                "max_output_tokens": self.get_max_output_tokens(**model_kwargs),
                "enable_thinking": model_kwargs.get("enable_thinking"),
            }
            # the vision prefill when the prompt's layout places media (the
            # layout says where the encoder's tokens and the positions go), the
            # text-only prefill otherwise
            prefill = media_prefill_step(
                input_signals, (self._vision_encoder(),), model_kwargs.get(MEDIA_LAYOUT),
                model_name="Cosmos3 reasoner",
            )
            prefill = WalkStep(prefill.walk, step_metadata={**knobs, **prefill.step_metadata})
            decode = WalkStep(self.REASONER_DECODE_WALK, bind={"text_inputs": "new_token"}, step_metadata=knobs)
            return [prefill, decode]
        params = self._resolve_gen_params(model_kwargs, input_modalities, output_modalities)
        # Visual conditioning routes through a conditioned prefill that also feeds
        # the DiT the input to VAE-encode: a video (action inverse-dynamics) or an
        # image (image-to-video, action policy/forward-dynamics). Fall back to the
        # text-only prefill if no conditioning signal actually arrived.
        if "video" in input_modalities and "video_inputs" in input_signals:
            prefill = self.PREFILL_COND_VIDEO_WALK
        elif params.get("has_image_condition") and "image_inputs" in input_signals:
            prefill = self.PREFILL_COND_WALK
        else:
            prefill = self.PREFILL_WALK
        denoise = self._denoise_walk(output_modalities, params)
        return [WalkStep(prefill, step_metadata=params), WalkStep(denoise, step_metadata=params)]

    def _denoise_walk(self, output_modalities: list[str], params: dict) -> str:
        """Forward-dynamics runs the joint denoise but decodes the predicted
        video; inverse-dynamics / policy emit the action; image and video share
        the loop but differ in what the VAE node emits."""
        is_video = "video" in output_modalities
        if params.get("action_mode") == "forward_dynamics":
            return self.ACTION_VIDEO_GEN_WALK
        if "action" in output_modalities:
            return self.ACTION_GEN_WALK
        if is_video and params.get("window_mode"):
            return self.VIDEO_GEN_AR_WALK
        if is_video and params.get("generate_sound"):
            return self.VIDEO_SOUND_GEN_WALK
        return self.VIDEO_GEN_WALK if is_video else self.IMAGE_GEN_WALK

    def _denoise_cap(self, output_modalities: list[str], params: dict) -> dict[str, int]:
        """The request's denoise loop and its iterations: one per scheduler step,
        and for a windowed request every window's, plus a commit per kv window."""
        iters = params["num_inference_steps"]
        if params.get("window_mode") is not None:
            iters = params["num_windows"] * (iters + 1 if params["window_mode"] == "kv" else iters)
        walk = self._denoise_walk(output_modalities, params)
        return {DENOISE_LOOP_FOR_WALK.get(walk, IMAGE_GEN_LOOP): iters}

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> torch.nn.Module | None:
        # autocast_dtype is accepted for interface parity (the engine manager
        # passes it to every model). Cosmos3 already casts the meta module to
        # bf16 before to_empty in _build_transformer, so params are allocated
        # directly in the checkpoint dtype and the hint is redundant here.
        # sp_group is the DiT's sequence-parallel comm group (trivial unless the
        # config sets sp_size > 1); it is orthogonal to the tp_group.
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, tp_group, sp_group)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Loaded Cosmos3 submodule for %s", node_name)
        return submodule

    def _create_submodule(self, node_name: str, device: str, tp_group=None, sp_group=None):
        if node_name == DIT_NODE:
            return Cosmos3DiTSubmodule(
                transformer=self._build_transformer(device, tp_group=tp_group, sp_group=sp_group),
                config=self.config,
                scheduler=self._build_scheduler(),
                # Chained windows re-encode the frames they continue from: the
                # decode on the streaming decoder's VAE, the encode on the
                # encoder node's, both shared with those nodes when they are in
                # this process.
                head_decode_vae=self._build_streaming_decode_vae(device) if self._windowed_serving_enabled() else None,
                head_encode_vae=self._build_encode_vae(device) if self._windowed_serving_enabled() else None,
            )
        if node_name == VAE_ENCODER_NODE:
            return Cosmos3VAEEncoderSubmodule(
                vae=self._build_encode_vae(device), config=self.config
            )
        if node_name == VAE_DECODER_NODE:
            # The fast decode converts its VAE to channels-last in place, so it
            # takes the decoder-only copy rather than the shared instance.
            vae = (
                self._build_decoder_only_vae(device)
                if self.config.vae_decode_fast
                else self._build_vae(device)
            )
            return Cosmos3VAEDecoderSubmodule(vae=vae, config=self.config, fast=self.config.vae_decode_fast)
        if node_name == VAE_DECODER_AR_NODE:
            return Cosmos3VAEDecoderARSubmodule(vae=self._build_streaming_decode_vae(device), config=self.config)
        if node_name == AUDIO_DECODER_NODE:
            return Cosmos3AudioDecoderSubmodule(
                sound_tokenizer=self._build_sound_tokenizer(device), config=self.config
            )
        if node_name == REASONER_NODE:
            # The same transformer instance as the DiT node (built once, cached
            # below): one copy of the text weights, shared kv/attn resources.
            return Cosmos3ReasonerSubmodule(
                transformer=self._build_transformer(device, tp_group=tp_group, sp_group=sp_group),
                config=self.config,
                encoders=(self._vision_encoder(),),
            )
        if node_name == VISION_ENCODER_NODE:
            return Cosmos3VisionEncoderSubmodule(
                vision_model=self._build_vision_model(device), config=self.config
            )
        return None

    def _build_scheduler(self):
        if self.skip_weight_loading:
            return None
        return self._scheduler_class().from_pretrained(str(self._ensure_repo() / "scheduler"))

    def _scheduler_class(self):
        """The diffusers scheduler the checkpoint ships: UniPC for the base
        checkpoints, FlowMatchEuler (stochastic, fixed sigmas) for the 4-step
        distilled ones."""
        import diffusers

        name = self.config.scheduler.scheduler_class
        if name == "FlowMatchEulerDiscreteScheduler":
            return diffusers.FlowMatchEulerDiscreteScheduler
        if name == "UniPCMultistepScheduler":
            return diffusers.UniPCMultistepScheduler
        raise ValueError(f"Unsupported Cosmos3 scheduler class {name!r}")

    def _build_transformer(self, device: str, tp_group=None, sp_group=None):
        # Built once per process: the DiT and the reasoner nodes share it.
        if self._transformer is not None:
            return self._transformer
        self._transformer = self._build_transformer_uncached(device, tp_group=tp_group, sp_group=sp_group)
        return self._transformer

    def _build_transformer_uncached(self, device: str, tp_group=None, sp_group=None):
        from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
        from mstar.model.cosmos3.loader import load_transformer_weights

        # Build on the meta device (shapes only, no storage), pin the
        # checkpoint's bf16 dtype, then materialize uninitialized tensors on the
        # target device and overwrite with the checkpoint weights — the same
        # path the other model packages use. bf16 matches the published
        # checkpoint exactly and halves resident weight memory vs the float32
        # meta default; the engine additionally runs the forward under a bf16
        # autocast (a no-op here).
        # Always meta: both branches below call ``to_empty``, which re-allocates
        # uninitialized storage and drops whatever construction produced, so
        # building on CPU only pays for allocation + parameter init that is then
        # thrown away (seconds per billion params on the skip_weight_loading
        # path). Same shape as the orpheus / higgs_audio builders.
        with torch.device("meta"):
            model = Cosmos3OmniTransformer(self.config, comm_group=tp_group, sp_group=sp_group)
        model = model.to(torch.bfloat16)
        if self.skip_weight_loading:
            return model.to_empty(device=device)

        model.to_empty(device=device)
        load_transformer_weights(model, self._ensure_repo(), device=device)
        # Keep the timestep embedder in fp32, like diffusers'
        # ``_keep_in_fp32_modules=["time_embedder"]`` (the upcast is lossless from
        # the bf16 checkpoint and matches diffusers' numerics).
        model.time_embedder.to(torch.float32)
        model.eval()
        return model

    def _build_vae(self, device: str):
        if self.skip_weight_loading:
            return None
        if self._vae is not None:
            return self._vae
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(str(self._ensure_repo() / "vae"))
        self._vae = vae.to(device).eval()
        return self._vae

    def _build_streaming_decode_vae(self, device: str):
        """The VAE the streaming decoder and a chained window's head decode
        on: a decoder-only copy when the decode goes channels-last (the
        conversion is in place), else the shared instance."""
        if self.config.vae_decode_channels_last:
            return self._build_decoder_only_vae(device)
        return self._build_vae(device)

    def _build_decoder_only_vae(self, device: str):
        # A separate instance (the decode needs no encoder) per device, shared
        # by the nodes on it that convert its layout in place, so the shared
        # whole instance stays as loaded.
        if self.skip_weight_loading:
            return None
        if device not in self._decoder_only_vaes:
            from diffusers import AutoencoderKLWan

            vae = AutoencoderKLWan.from_pretrained(str(self._ensure_repo() / "vae"))
            vae.encoder = None
            self._decoder_only_vaes[device] = vae.to(device).eval()
        return self._decoder_only_vaes[device]

    def _build_encode_vae(self, device: str):
        # The encoder node keeps its own instance instead of sharing the
        # decoder's: the encode runs in the checkpoint's bf16, as the reference
        # pipeline's does, while the decode dtype is cuDNN-gated (fp32 before
        # 9.16), so a shared instance would re-cast the full VAE weights on
        # every encode/decode interleave. One per device, shared with a chained
        # window's head when the DiT is on it.
        if self.skip_weight_loading:
            return None
        if device not in self._encode_vaes:
            from diffusers import AutoencoderKLWan

            vae = AutoencoderKLWan.from_pretrained(str(self._ensure_repo() / "vae"), torch_dtype=torch.bfloat16)
            self._encode_vaes[device] = vae.to(device).eval()
        return self._encode_vaes[device]

    def _build_vision_model(self, device: str):
        """The reasoner's vision tower + projector, from ``vision_encoder/``."""
        from mstar.model.cosmos3.components.vision import Cosmos3VisionModel
        from mstar.model.cosmos3.loader import load_vision_encoder_weights

        with torch.device("meta"):
            model = Cosmos3VisionModel(self.config.reasoner)
        model = model.to(torch.bfloat16)
        if self.skip_weight_loading:
            return model.to_empty(device=device)
        model.to_empty(device=device)
        load_vision_encoder_weights(model, self._ensure_repo(), device=device)
        model.eval()
        return model

    def _build_sound_tokenizer(self, device: str):
        if self.skip_weight_loading:
            return None
        from mstar.model.cosmos3.components.sound_tokenizer import Cosmos3SoundTokenizer

        tokenizer = Cosmos3SoundTokenizer.from_pretrained(
            self._ensure_repo(), device=device, dtype=torch.bfloat16
        )
        logger.info(
            "Loaded Cosmos3 sound tokenizer (sr=%d, channels=%d, latent_ch=%d, hop=%d)",
            tokenizer.sample_rate, tokenizer.audio_channels, tokenizer.latent_ch, tokenizer.hop_size,
        )
        return tokenizer
