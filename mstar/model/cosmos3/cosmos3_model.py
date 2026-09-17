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
    reasoner       (kv_cache, sampler) - the same transformer instance as the
                                 DiT (one copy of the text weights, the same KV
                                 pool), run as a causal text model.
    reasoner_prefill / reasoner_prefill_vision - embed the chat-templated
                 prompt (vision tokens scattered over the media placeholders),
                 write its K/V, sample the first token.
    reasoner_decode - one token per loop iteration until EOS / max tokens.

Streaming rollout (opt-in, ``enable_windowed_video``; ported from #198):
    video_gen_ar - the denoise loop run window by window over a long clip.
                 ``chained`` windows re-pin the previous window's tail as clean
                 conditioning; ``kv`` windows attend block-causally over the
                 committed K/V of earlier windows (one commit iteration per
                 window appends the finished window's clean K/V; the pool's
                 retention policy releases frames past the context horizon at
                 each commit). Each finished window's latents leave the loop on
                 a streaming edge.
    vae_decoder_ar (partition ``window_decoder``) - decodes each window behind
                 re-decoded context while the loop denoises the next one, and
                 emits the video per window (``stream_video``) or assembled.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    KVConfig,
    KVReqConfig,
    KVSpec,
    NodeResourceSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
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
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.cosmos3 import constants
from mstar.model.cosmos3.components.packing import ACTION_MODES, resolve_action_domain_id
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.submodules import (
    ACTION_GEN_LOOP,
    ACTION_VIDEO_GEN_LOOP,
    ATTN,
    ATTN_GEN,
    COND_LABEL,
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
)
from mstar.model.multimodal import TEXT, PromptPart, check_attachments, parts_from_modalities
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

logger = logging.getLogger(__name__)

DIT_NODE = "dit"
VAE_ENCODER_NODE = "vae_encoder"
VAE_DECODER_NODE = "vae_decoder"
AUDIO_DECODER_NODE = "audio_decoder"
VAE_DECODER_AR_NODE = "vae_decoder_ar"
VISION_ENCODER_NODE = "vision_encoder"
REASONER_NODE = "reasoner"


class Cosmos3Model(Model):
    """NVIDIA Cosmos3 generator implementation."""

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
        self.config: Cosmos3Config = self._load_config()
        self.tokenizer = self._load_tokenizer()

        self._submodule_cache: dict[str, torch.nn.Module | None] = {}
        # The Wan VAE is shared between the DiT submodule (conditioning encode)
        # and the decoder submodule, so build it once. The transformer is
        # shared between the DiT and reasoner nodes likewise.
        self._vae = None
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

        The two specs share one ``KVConfig`` object on purpose: a deployment
        that resizes the cache through ``apply_yaml_overrides`` has to resize
        what the wrappers are planned against too.
        """
        kv_config = KVConfig(
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
                    kv_cache=KV_CACHE, backend=AttnBackend.FLASHINFER,
                ),
            ),
        ]
        if self._reasoner_enabled():
            specs.append(SamplerSpec(
                resource_key=SAMPLER, nodes={REASONER_NODE},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=True,
            ))
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

    def get_request_resource_configs(
        self,
        partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        """Which cache labels a request is opened with.

        Both guidance branches, unconditionally: a request's guidance regime
        is only settled at prefill (it follows ``guidance_scale``, which
        ``process_prompt`` resolves), and naming a label a request never
        writes costs nothing — labels are created on first write.
        """
        mk = model_kwargs or {}
        if self._is_text_request(partition_fwd_args):
            sampling = self.get_sampling_config(REASONER_NODE, mk)
            return {
                KV_CACHE: KVReqConfig(needed_labels=[REASONER_LABEL]),
                SAMPLER: SamplingReqConfig(
                    temperature=sampling.temperature, top_k=sampling.top_k,
                    top_p=sampling.top_p, repetition_penalty=sampling.repetition_penalty,
                    ignore_eos=sampling.ignore_eos,
                ),
            }
        return {
            KV_CACHE: KVReqConfig(needed_labels=[COND_LABEL, UNCOND_LABEL]),
        }

    def get_sampling_config(self, node_name: str, model_kwargs: dict | None = None):
        """The reasoner's sampling knobs: OpenAI-standard ``temperature`` /
        ``top_p`` plus ``top_k`` / ``repetition_penalty`` / ``ignore_eos``
        from ``extra_body``. Greedy at temperature 0."""
        from mstar.engine.resources.sampler.utils import SamplingConfig

        mk = model_kwargs or {}
        return SamplingConfig(
            vocab_size=self.config.vocab_size,
            temperature=float(mk.get("temperature", self.config.reasoner_temperature)),
            top_k=int(mk.get("top_k", 0)),
            top_p=float(mk.get("top_p", 1.0)),
            repetition_penalty=float(mk.get("repetition_penalty", 1.0)),
            ignore_eos=bool(mk.get("ignore_eos", False)),
        )

    @staticmethod
    def _is_text_request(partition_fwd_args: dict[str, ForwardPassArgs] | None) -> bool:
        """Whether a request decodes text (the reasoner) rather than
        generating media: read off the initial forward-pass args the
        conductor resolved for its partitions."""
        for args in (partition_fwd_args or {}).values():
            md = getattr(args, "full_metadata", None)
            if md is not None and "text" in (md.output_modalities or []):
                return True
        return False

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
            input_names=["text_inputs"],
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
                    GraphNode(name=DIT_NODE, input_names=["text_inputs"], outputs=[]),
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
        # threads the latents + denoise-step index back to itself each iteration;
        # on the final iteration the latents route to the decoder. max_iters is an
        # upper bound — each request stops the loop at its own denoise-step count
        # (Cosmos3DiTSubmodule.check_stop), so one graph serves image and video
        # (and any per-request num_inference_steps) without a rebuild.
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
                            input_names=["latents", "time_index", "cond_latents"],
                            outputs=[
                                GraphEdge(next_node=DIT_NODE, name="latents"),
                                GraphEdge(next_node=DIT_NODE, name="time_index"),
                            ],
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
                        input_names=["latents", "sound_latents", "time_index", "cond_latents"],
                        outputs=[
                            GraphEdge(next_node=DIT_NODE, name="latents"),
                            GraphEdge(next_node=DIT_NODE, name="sound_latents"),
                            GraphEdge(next_node=DIT_NODE, name="time_index"),
                        ],
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
                        input_names=["latents", "action_latents", "time_index", "cond_latents"],
                        outputs=[
                            GraphEdge(next_node=DIT_NODE, name="latents"),
                            GraphEdge(next_node=DIT_NODE, name="action_latents"),
                            GraphEdge(next_node=DIT_NODE, name="time_index"),
                        ],
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
                        input_names=["latents", "action_latents", "time_index", "cond_latents"],
                        outputs=[
                            GraphEdge(next_node=DIT_NODE, name="latents"),
                            GraphEdge(next_node=DIT_NODE, name="action_latents"),
                            GraphEdge(next_node=DIT_NODE, name="time_index"),
                        ],
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
                        input_names=["latents", "time_index", "cond_latents"],
                        outputs=[
                            GraphEdge(next_node=DIT_NODE, name="latents"),
                            GraphEdge(next_node=DIT_NODE, name="time_index"),
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
                initial_walk=None,
                producer_partitions=[],
            ),
            PartitionDefinition(
                name=constants.WINDOW_DECODER_PARTITION,
                graph_walks={self.VIDEO_DECODE_AR_WALK},
                initial_walk=self.VIDEO_DECODE_AR_WALK,
                producer_partitions=["default"],
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
        """The VLM walks. ``reasoner_prefill`` embeds a text-only prompt;
        ``reasoner_prefill_vision`` first runs the vision encoder over the
        request's packed patches and hands the projected tokens to the
        reasoner, which scatters them over the media placeholders. Both
        sample the first token, which persists into the decode loop; each
        decode iteration emits its token and feeds it back."""
        first_token = GraphEdge(
            next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True,
        )
        prefill = GraphNode(
            name=REASONER_NODE,
            input_names=["text_inputs", "position_ids"],
            outputs=[first_token],
        )
        prefill_vision = Sequential(
            [
                GraphNode(
                    name=VISION_ENCODER_NODE,
                    input_names=["pixel_values", "vision_grid_thw"],
                    outputs=[GraphEdge(next_node=REASONER_NODE, name="vision_embeds")],
                ),
                GraphNode(
                    name=REASONER_NODE,
                    input_names=["text_inputs", "position_ids", "vision_embeds"],
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT, name="new_token",
                            output_modality="text", persist=True,
                        ),
                    ],
                ),
            ]
        )
        decode = Loop(
            name=REASONER_DECODE_LOOP,
            section=GraphNode(
                name=REASONER_NODE,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
                    GraphEdge(next_node=REASONER_NODE, name="text_inputs"),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return {
            self.REASONER_PREFILL_WALK: prefill,
            self.REASONER_PREFILL_VISION_WALK: prefill_vision,
            self.REASONER_DECODE_WALK: decode,
        }

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
    ) -> NameToTensorList:
        if "text" in (output_modalities or []):
            return self._process_reasoner_prompt(
                prompt, input_modalities, tensors or {}, prompt_parts, input_metadata or {}, kwargs,
            )
        if prompt is None:
            return {}
        if self.tokenizer is None:
            # Tokenizer-less fallback used by structural unit tests.
            return {
                "text_inputs": [
                    torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long)
                ]
            }
        # Both the conditional (positive) and unconditional (negative) prompts are
        # tokenized up front; the denoiser reads the second only when guidance is
        # on. Image/video prompts get the chat template + resolution/duration
        # sentences; action prompts are tokenized raw.
        from mstar.model.cosmos3.components.packing import tokenize_prompt

        negative_prompt = kwargs.get("negative_prompt")
        p = self._resolve_gen_params(kwargs, input_modalities, output_modalities)
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
            use_system_prompt=allow_templates and bool(kwargs.get("use_system_prompt", False)),
            add_resolution_template=allow_templates and bool(kwargs.get("use_resolution_template", False)),
            add_duration_template=allow_templates and bool(kwargs.get("use_duration_template", False)),
            max_sequence_length=p["max_sequence_length"],
        )
        return {
            "text_inputs": [
                torch.tensor(cond_ids, dtype=torch.long),
                torch.tensor(uncond_ids, dtype=torch.long),
            ]
        }

    def _process_reasoner_prompt(
        self, prompt, input_modalities, tensors, prompt_parts, input_metadata, model_kwargs,
    ) -> NameToTensorList:
        """Render the chat prompt, preprocess its media, expand the
        placeholders and compute the mRoPE positions — everything the
        reasoner prefill needs besides the encoder pass.

        Returns ``text_inputs`` (the token ids), ``position_ids`` ([3, N]) and,
        with attachments, the packed ``pixel_values`` + ``vision_grid_thw``
        the vision_encoder node consumes."""
        from mstar.model.cosmos3.components.reasoner import (
            IMAGE,
            VIDEO,
            expand_placeholders,
            mrope_position_ids,
            preprocess_image,
            preprocess_video,
            render_chat,
        )

        if not self._reasoner_enabled():
            raise ValueError("This Cosmos3 checkpoint/deployment does not serve the reasoner (text output).")
        if self.tokenizer is None:
            raise ValueError("The Cosmos3 reasoner needs the checkpoint tokenizer.")
        reasoner = self.config.reasoner
        parts = parts_from_modalities(
            input_modalities,
            [p.text or "" for p in prompt_parts if p.modality == TEXT] if prompt_parts is not None else prompt,
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
                num_frames=model_kwargs.get("video_num_frames"), fps=model_kwargs.get("video_fps"),
            )
            patches.append(pv)
            video_grids.append(grid)
        # Media patches are packed in prompt order (images, then videos, as
        # the parts list them); the vision encoder returns tokens in that
        # order and the placeholders are expanded in the same order.
        ordered_patches: list[torch.Tensor] = []
        ordered_grids: list[tuple[int, int, int]] = []
        img_i = vid_i = 0
        for part in parts:
            if part.modality == IMAGE:
                ordered_patches.append(patches[img_i])
                ordered_grids.append(image_grids[img_i].thw)
                img_i += 1
            elif part.modality == VIDEO:
                ordered_patches.append(patches[len(image_grids) + vid_i])
                ordered_grids.append(video_grids[vid_i].thw)
                vid_i += 1

        text = render_chat(
            self.tokenizer, parts, reasoner,
            enable_thinking=model_kwargs.get("enable_thinking"),
            system_prompt=model_kwargs.get("system_prompt"),
        )
        text = expand_placeholders(text, self.tokenizer, reasoner, image_grids, video_grids)
        ids = torch.tensor(self.tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.long)
        max_len = int(reasoner.max_position_embeddings)
        if ids.numel() > max_len:
            raise ValueError(
                f"Cosmos3 reasoner prompt is {ids.numel()} tokens, over the {max_len}-token context."
            )
        position_ids, _ = mrope_position_ids(ids, reasoner, image_grids, video_grids)
        out: NameToTensorList = {"text_inputs": [ids], "position_ids": [position_ids]}
        if ordered_patches:
            out["pixel_values"] = [torch.cat(ordered_patches, dim=0)]
            out["vision_grid_thw"] = [torch.tensor(ordered_grids, dtype=torch.long)]
        return out

    def load_video(self, filepath: str, device: str):
        """Decode a conditioning / reasoner video to ``[T, C, H, W]`` in [0, 1].

        torchcodec (the base implementation) needs system FFmpeg shared
        libraries; where they are absent, PyAV — which ships its own — decodes
        the same frames. The metadata carries the frame rate under
        ``average_fps`` either way (the reasoner's frame sampling and
        timestamps read it)."""
        from mstar.model.base import TensorAndMetadata

        try:
            return super().load_video(filepath, device)
        except (ImportError, RuntimeError, OSError) as exc:
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            logger.warning("torchcodec video decode unavailable (%s); decoding %s with PyAV.", reason, filepath)
        import av

        frames = []
        with av.open(filepath) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate or stream.base_rate
            fps = float(rate) if rate else None
            for frame in container.decode(stream):
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
        if not frames:
            raise ValueError(f"no video frames decoded from {filepath}")
        video = torch.stack(frames).to(device).float() / 255.0
        metadata = {
            "num_frames": len(frames), "average_fps": fps,
            "duration_seconds": (len(frames) / fps) if fps else None,
            "height": int(video.shape[-2]), "width": int(video.shape[-1]),
        }
        return TensorAndMetadata(data=video, metadata=metadata)

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality == "text":
            # One sampled token per chunk; the client concatenates the pieces.
            ids = output.reshape(-1).tolist()
            return self.tokenizer.decode(ids, skip_special_tokens=True).encode("utf-8")
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

            # The decoder emits 8-bit frames [B, C, T, H, W]; encode all of them as
            # an H.264 mp4. The frames already reflect the request fps (it modulates
            # the temporal positions during generation), and the container carries
            # that same rate so playback runs at the requested speed.
            #
            # CRF 18 keeps the H.264 output near-visually-lossless; libx264
            # otherwise defaults to 23, which is visibly lossier. The "ultrafast"
            # preset and multithreading (threads=0) target the same CRF/quality
            # but encode several times faster than libx264's default "medium"
            # preset, which otherwise dominates the serving latency for a
            # many-frame clip. Both are overridable via COSMOS3_X264_PRESET.
            x = output[0] if output.ndim == 5 else output  # [C, T, H, W] uint8
            fps = float((request_kwargs or {}).get("fps", self.config.fps))
            preset = os.environ.get("COSMOS3_X264_PRESET", "ultrafast")
            try:
                # Preferred: torchcodec (torchvision >= 0.27 removed write_video).
                from torchcodec.encoders import VideoEncoder

                frames = x.permute(1, 0, 2, 3).contiguous().cpu()  # [T, C, H, W] uint8
                encoded = VideoEncoder(frames, frame_rate=fps).to_tensor(
                    "mp4",
                    codec="libx264",
                    crf=18,
                    preset=preset,
                    extra_options={"threads": "0"},
                )
                data = encoded.numpy().tobytes()
            except ImportError:
                # Fallback for environments without torchcodec (or with the
                # older decode-only torchcodec that lacks VideoEncoder), where
                # torchvision still ships write_video.
                import tempfile

                from torchvision.io import write_video

                frames = x.permute(1, 2, 3, 0).cpu()  # [T, H, W, C] uint8
                fd, path = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)
                try:
                    write_video(
                        path,
                        frames,
                        fps=fps,
                        video_codec="libx264",
                        options={"crf": "18", "preset": preset, "threads": "0"},
                    )
                    with open(path, "rb") as f:
                        data = f.read()
                finally:
                    os.remove(path)
            return data
        if modality == "action":
            # The predicted action latents [1, chunk, action_dim] -> [chunk,
            # action_dim] float32 bytes. Columns beyond the request's
            # raw_action_dim are zero padding (the client keeps the first
            # raw_action_dim, the real action width for its embodiment).
            x = output[0] if output.ndim == 3 else output
            return x.detach().to(torch.float32).cpu().numpy().tobytes()
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
        size = mk.get("size")
        if isinstance(size, str) and "x" in size.lower():
            sw, sh = size.lower().split("x", 1)
            try:
                width, height = int(sw), int(sh)
            except ValueError:
                pass

        # Action requests resolve their mode up front: it switches the frame,
        # step, guidance and flow-shift defaulting below to the action recipe.
        action_mode = mk.get("action_mode")
        if action_mode is not None:
            action_mode = str(action_mode).strip().lower()
            if action_mode not in ACTION_MODES:
                raise ValueError(
                    f"Unsupported Cosmos3 action_mode={mk.get('action_mode')!r}; "
                    f"expected one of {sorted(ACTION_MODES)}."
                )

        if action_mode is not None:
            # An action request predicts one action token per frame, so the
            # frame count and the chunk length are coupled (num_frames = chunk
            # or chunk + 1) and default off each other; chunk 16 when neither
            # is sent (the reference action default).
            raw_chunk = mk.get("action_chunk_size")
            raw_frames = mk.get("num_frames")
            if raw_chunk is not None:
                action_chunk = int(raw_chunk)
            elif raw_frames is not None:
                action_chunk = int(raw_frames) - 1
            else:
                action_chunk = 16
            if action_chunk <= 0:
                raise ValueError(
                    f"Cosmos3 action_chunk_size must be positive, got {action_chunk}."
                )
            num_frames = int(raw_frames) if raw_frames is not None else action_chunk + 1
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
            num_frames = int(mk.get("num_frames", default_frames))
        # The cookbook step counts differ per mode (image 50, video 35, action
        # 30 — configs override the action count per checkpoint); default by
        # mode and let the request override. The denoise loop runs this many
        # steps and stops early (Cosmos3DiTSubmodule.check_stop), so the value
        # is only bounded above by the loop's static max_iters.
        if action_mode is not None:
            default_steps = self.config.num_inference_steps_action
        elif num_frames > 1:
            default_steps = self.config.num_inference_steps_video
        else:
            default_steps = self.config.num_inference_steps
        steps = int(mk.get("num_inference_steps", default_steps))
        steps = max(1, min(steps, self.config.max_inference_steps))
        default_guidance = (
            self.config.guidance_scale_action if action_mode is not None else self.config.guidance_scale
        )
        if self.config.distilled_sigmas:
            steps, default_guidance = self._resolve_distilled_params(
                mk, steps, action_mode, input_modalities, output_modalities,
            )
        params = {
            "width": int(mk.get("width", width)),
            "height": int(mk.get("height", height)),
            "num_frames": num_frames,
            "fps": float(mk.get("fps", self.config.fps)),
            "guidance_scale": float(mk.get("guidance_scale", default_guidance)),
            "num_inference_steps": steps,
            "has_image_condition": "image" in (input_modalities or []),
            "use_karras_sigma": mk.get("use_karras_sigmas"),
            # Prompt-token truncation cap (reference serving default 4096),
            # request-overridable; floor 1 so a bad value can't empty the prompt.
            "max_sequence_length": max(
                1, int(mk.get("max_sequence_length", constants.DEFAULT_MAX_SEQUENCE_LENGTH))
            ),
        }
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
            keep = str(mk.get("condition_video_keep") or constants.DEFAULT_CONDITION_VIDEO_KEEP).strip().lower()
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
        fs = mk.get("flow_shift")
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
        if gi is None and is_t2i:
            gi = (400.0, 1000.0)
        if gi is not None:
            params["guidance_interval"] = (float(gi[0]), float(gi[1]))
        # Action requests must name their embodiment explicitly — the domain
        # conditions the action pathway, and a silent default would predict
        # actions for the wrong robot. ``domain_name`` resolves through the
        # published embodiment table; a numeric ``domain_id`` wins. The raw
        # action width is likewise required (forward-dynamics can infer it
        # from its conditioning ``action`` array) and bounded by the model's
        # padded action dim.
        if action_mode is not None:
            params["action_mode"] = action_mode
            params["action_chunk_size"] = action_chunk
            params["domain_id"] = resolve_action_domain_id(
                mk.get("domain_id"), mk.get("domain_name")
            )
            raw_dim = mk.get("raw_action_dim")
            if raw_dim is None and action_mode == "forward_dynamics" and mk.get("action") is not None:
                try:
                    raw_dim = int(torch.as_tensor(mk["action"]).shape[-1])
                except (TypeError, ValueError, RuntimeError):
                    raw_dim = None
            if raw_dim is None:
                raise ValueError(
                    "Cosmos3 action requests require 'raw_action_dim' "
                    "(forward_dynamics may omit it when the 'action' array carries the width)."
                )
            raw_dim = int(raw_dim)
            if not 1 <= raw_dim <= self.config.max_action_dim:
                raise ValueError(
                    f"Cosmos3 raw_action_dim must be in [1, {self.config.max_action_dim}], "
                    f"got {raw_dim}."
                )
            params["raw_action_dim"] = raw_dim
            for k in ("action_fps", "action"):
                if k in mk:
                    params[k] = mk[k]
        # Opt-in sound generation: video-only (image and action requests carry
        # no sound band), and only when the served checkpoint/config enable it.
        if mk.get("generate_sound") or mk.get("sound_gen"):
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
                params["sound_duration"] = float(mk["sound_duration"])
        self._resolve_window_params(mk, params, num_frames, action_mode, has_video_condition)
        return params

    def _resolve_distilled_params(self, mk, steps, action_mode, input_modalities, output_modalities):
        """The 4-step distilled checkpoints fix the sampler: their sigma list
        sets the step count, guidance is baked into the weights (scale 1), and
        the task is the checkpoint's own (t2i / i2v) — no action, sound,
        video-to-video or windowed modes. Mirrors the reference's
        ``Cosmos3DistilledSetTimestepsStep`` checks."""
        fixed = len(self.config.distilled_sigmas)
        if mk.get("num_inference_steps") is not None and int(mk["num_inference_steps"]) != fixed:
            raise ValueError(
                f"This Cosmos3 checkpoint is distilled: num_inference_steps is fixed at {fixed} "
                f"(got {mk['num_inference_steps']}); leave it unset."
            )
        if mk.get("guidance_scale") is not None and float(mk["guidance_scale"]) != 1.0:
            raise ValueError(
                "This Cosmos3 checkpoint is distilled: classifier-free guidance is baked into the "
                f"weights, guidance_scale must be 1.0 (got {mk['guidance_scale']}); leave it unset."
            )
        if action_mode is not None or mk.get("generate_sound") or mk.get("sound_gen") or mk.get("window_mode"):
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
        window_mode = mk.get("window_mode")
        if window_mode is None:
            if mk.get("stream_video"):
                # The non-windowed walks emit one video at the very end; there
                # is nothing to deliver incrementally.
                raise ValueError(
                    "Cosmos3 stream_video requires a windowed request (set window_mode)."
                )
            return
        window_mode = str(window_mode).strip().lower()
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
        window_frames = int(mk.get("window_frames", self.config.window_frames_default))
        if window_frames < 1 + tf:
            raise ValueError(f"Cosmos3 window_frames must be at least {1 + tf}, got {window_frames}.")
        window_units = 1 + (window_frames - 1) // tf
        # kv windows advance without re-pinned overlap — cross-window
        # conditioning flows through the committed K/V, and a zero overlap
        # keeps each commit exactly covering the span its denoise steps wrote.
        default_overlap = 0 if is_kv else self.config.overlap_frames_default
        overlap_frames = int(mk.get("overlap_frames", default_overlap))
        if is_kv and overlap_frames:
            raise ValueError(
                "Cosmos3 window_mode='kv' does not support overlap_frames; "
                "cross-window conditioning comes from the committed context."
            )
        overlap_units = min(max(round(overlap_frames / tf), 0), window_units - 1)
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
            context_frames = int(mk.get("context_frames", self.config.context_frames_default))
            if context_frames < 0:
                raise ValueError(f"Cosmos3 context_frames must be >= 0, got {context_frames}.")
            # 0 retains all committed frames (no release).
            if context_frames:
                context_units = 1 + (context_frames - 1) // tf
        total_units = 1 + (num_frames - 1) // tf
        # Sessions: a request may name a session (its last window is kept for
        # a follow-up) and resume one — the stored tail then re-pins the head
        # of window 0 as clean conditioning (the chained overlap, at least the
        # two-frame V2V floor), and the schedule grows by those units so
        # ``num_frames`` stays the count of new frames the client receives.
        session_id = mk.get("session_id")
        resume = bool(mk.get("resume_session"))
        if resume and not session_id:
            raise ValueError("Cosmos3 resume_session requires a session_id.")
        if resume and params.get("has_image_condition"):
            raise ValueError(
                "Cosmos3 resume_session conditions on the session's last frames; "
                "drop the conditioning image."
            )
        if session_id is not None:
            params["session_id"] = str(session_id)
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
        params["stream_video"] = bool(mk.get("stream_video"))
        # Every window after the first is conditioned generation, which the
        # reference recipe runs at the V2V flow shift; one shift for all
        # windows keeps the per-window schedules consistent. A deployment's
        # video shift (Edge: 12.0) or a request flow_shift wins.
        params.setdefault("flow_shift", constants.V2V_DEFAULT_FLOW_SHIFT)

    def _step_metadata(self, metadata: CurrentForwardConductorMetadata) -> dict:
        md = {"is_prefill": metadata.is_prefill}
        md.update(metadata.kwargs)
        return md

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        if "text" in (output_modalities or []):
            return self._initial_reasoner_args(input_modalities, output_modalities, input_signals, model_kwargs)
        params = self._resolve_gen_params(model_kwargs, input_modalities, output_modalities)
        # The windowed decoder partition starts idle on its decode walk; the
        # window stream self-triggers its passes, and the resolved params ride
        # along for its per-request window bookkeeping. Non-windowed requests
        # leave it idle until the stream's terminal flush, which it skips.
        if partition_name == constants.WINDOW_DECODER_PARTITION:
            md = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk=self.VIDEO_DECODE_AR_WALK,
                is_prefill=False,
                kwargs=params,
            )
            return ForwardPassArgs(
                full_metadata=md, inputs=[], unpersist_tensors=[],
                step_metadata=self._step_metadata(md),
            )
        # Visual conditioning routes through a conditioned prefill that also feeds
        # the DiT the input to VAE-encode: a video (action inverse-dynamics) or an
        # image (image-to-video, action policy/forward-dynamics). Fall back to the
        # text-only prefill if no conditioning signal actually arrived.
        video_cond = "video" in input_modalities and "video_inputs" in input_signals
        image_cond = params.get("has_image_condition") and "image_inputs" in input_signals
        if video_cond:
            walk = self.PREFILL_COND_VIDEO_WALK
        elif image_cond:
            walk = self.PREFILL_COND_WALK
        else:
            walk = self.PREFILL_WALK
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=walk,
            is_prefill=True,
            kwargs=params,
        )

        inputs: list[GraphEdge] = []
        if "text_inputs" in input_signals:
            edge = GraphEdge(next_node=DIT_NODE, name="text_inputs")
            edge.tensor_info = input_signals["text_inputs"]
            inputs.append(edge)
        cond_signal = "video_inputs" if video_cond else ("image_inputs" if image_cond else None)
        if cond_signal:
            edge = GraphEdge(next_node=VAE_ENCODER_NODE, name=cond_signal)
            edge.tensor_info = input_signals[cond_signal]
            inputs.append(edge)

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._step_metadata(full_metadata),
        )

    def _initial_reasoner_args(
        self, input_modalities, output_modalities, input_signals, model_kwargs,
    ) -> ForwardPassArgs:
        """First walk of a text (reasoner) request: the vision prefill when the
        prompt carried media, the text-only prefill otherwise."""
        if not self._reasoner_enabled():
            raise ValueError("This Cosmos3 deployment does not serve the reasoner (text output).")
        mk = dict(model_kwargs or {})
        has_vision = "pixel_values" in input_signals and "vision_grid_thw" in input_signals
        walk = self.REASONER_PREFILL_VISION_WALK if has_vision else self.REASONER_PREFILL_WALK
        kwargs = {
            "max_output_tokens": self.get_max_output_tokens(**mk),
            "enable_thinking": mk.get("enable_thinking"),
        }
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=walk,
            is_prefill=True,
            kwargs=kwargs,
        )
        inputs: list[GraphEdge] = []
        for name in ("text_inputs", "position_ids"):
            edge = GraphEdge(next_node=REASONER_NODE, name=name)
            edge.tensor_info = input_signals[name]
            inputs.append(edge)
        if has_vision:
            for name in ("pixel_values", "vision_grid_thw"):
                edge = GraphEdge(next_node=VISION_ENCODER_NODE, name=name)
                edge.tensor_info = input_signals[name]
                inputs.append(edge)
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._step_metadata(full_metadata),
        )

    def _reasoner_partition_args(
        self, metadata: CurrentForwardConductorMetadata, persist_signals,
    ) -> ForwardPassArgs:
        """Reasoner transitions: prefill -> decode loop (seeded with the
        persisted first token); the loop's end (EOS / max tokens, decided by
        the submodule's ``check_stop``) finishes the request."""
        request_done = False
        inputs: list[GraphEdge] = []
        if metadata.graph_walk in (self.REASONER_PREFILL_WALK, self.REASONER_PREFILL_VISION_WALK):
            metadata.is_prefill = False
            metadata.graph_walk = self.REASONER_DECODE_WALK
            edge = GraphEdge(next_node=REASONER_NODE, name="text_inputs")
            edge.tensor_info = persist_signals.get("new_token", [])
            inputs.append(edge)
        elif metadata.graph_walk == self.REASONER_DECODE_WALK:
            request_done = True
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._step_metadata(metadata),
            request_done=request_done,
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        if metadata.graph_walk in (
            self.REASONER_PREFILL_WALK, self.REASONER_PREFILL_VISION_WALK, self.REASONER_DECODE_WALK,
        ):
            return self._reasoner_partition_args(metadata, persist_signals)
        request_done = False
        inputs: list[GraphEdge] = []

        # The windowed decoder partition is self-triggered by its stream
        # buffer; the conductor only keeps its walk pinned. Its completion is
        # the stream's final chunk, not a conductor decision.
        if partition_name == constants.WINDOW_DECODER_PARTITION:
            metadata.graph_walk = self.VIDEO_DECODE_AR_WALK
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[],
                step_metadata=self._step_metadata(metadata),
            )

        # Forward-dynamics conditions on a clean action chunk and emits the
        # predicted video; inverse-dynamics / policy emit the action.
        is_fd = metadata.kwargs.get("action_mode") == "forward_dynamics"
        is_action = "action" in metadata.output_modalities
        is_video = "video" in metadata.output_modalities
        joint_action = is_fd or is_action  # walks that also thread action latents
        if metadata.graph_walk in (
            self.PREFILL_WALK, self.PREFILL_COND_WALK, self.PREFILL_COND_VIDEO_WALK
        ):
            metadata.is_prefill = False
            # Pick the denoise walk: forward-dynamics runs the joint denoise but
            # decodes the predicted video; inverse-dynamics / policy emit the
            # action; image and video share the loop but differ in what the VAE
            # node emits.
            if is_fd:
                metadata.graph_walk = self.ACTION_VIDEO_GEN_WALK
            elif is_action:
                metadata.graph_walk = self.ACTION_GEN_WALK
            elif is_video and metadata.kwargs.get("window_mode"):
                metadata.graph_walk = self.VIDEO_GEN_AR_WALK
            elif is_video and metadata.kwargs.get("generate_sound"):
                metadata.graph_walk = self.VIDEO_SOUND_GEN_WALK
            elif is_video:
                metadata.graph_walk = self.VIDEO_GEN_WALK
            else:
                metadata.graph_walk = self.IMAGE_GEN_WALK
            # The first denoise iteration's initial noise + step index are
            # sampled inside the DiT submodule's preprocess. Action and sound
            # walks also thread their extra latents through the loop.
            inputs = [
                GraphEdge(next_node=DIT_NODE, name="latents"),
                GraphEdge(next_node=DIT_NODE, name="time_index"),
            ]
            if joint_action:
                inputs.insert(1, GraphEdge(next_node=DIT_NODE, name="action_latents"))
            elif metadata.graph_walk == self.VIDEO_SOUND_GEN_WALK:
                inputs.insert(1, GraphEdge(next_node=DIT_NODE, name="sound_latents"))
            # The vae_encoder node's clean conditioning latents (persisted at
            # the conductor during the prefill walk) seed the loop's first
            # iteration; unconditioned requests carry an empty edge.
            cond_edge = GraphEdge(next_node=DIT_NODE, name="cond_latents")
            cond_edge.tensor_info = persist_signals.get("cond_latents", [])
            inputs.append(cond_edge)
        elif metadata.graph_walk in (
            self.IMAGE_GEN_WALK, self.VIDEO_GEN_WALK, self.VIDEO_GEN_AR_WALK,
            self.VIDEO_SOUND_GEN_WALK, self.ACTION_GEN_WALK, self.ACTION_VIDEO_GEN_WALK,
        ):
            request_done = True

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._step_metadata(metadata),
            request_done=request_done,
        )

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
            )
        if node_name == VAE_ENCODER_NODE:
            return Cosmos3VAEEncoderSubmodule(
                vae=self._build_encode_vae(device), config=self.config
            )
        if node_name == VAE_DECODER_NODE:
            return Cosmos3VAEDecoderSubmodule(
                vae=self._build_vae(device), config=self.config
            )
        if node_name == VAE_DECODER_AR_NODE:
            # Shares the decoder VAE weights; only the streaming chunk state
            # and compile wrapper are per-node.
            return Cosmos3VAEDecoderARSubmodule(
                vae=self._build_vae(device), config=self.config
            )
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

    def _build_encode_vae(self, device: str):
        # The encoder node keeps its own fp32 instance instead of sharing the
        # decoder's: encode always runs fp32 while the decode dtype is
        # cuDNN-gated (bf16 from 9.16), so a shared instance would re-cast the
        # full VAE weights on every encode/decode interleave.
        if self.skip_weight_loading:
            return None
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(str(self._ensure_repo() / "vae"))
        return vae.float().to(device).eval()

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
