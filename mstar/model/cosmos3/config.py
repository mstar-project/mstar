"""Configuration for the Cosmos3 omni generator.

A single ``Cosmos3Config`` describes every Cosmos3 checkpoint (Nano, Super,
Edge, the Policy-DROID fine-tunes and the Super task variants). The
checkpoints share one dual-pathway MoT architecture; they differ in the
transformer dimensions (``num_hidden_layers`` / ``hidden_size`` /
``num_attention_heads`` / ``intermediate_size``), the two capability flags
(``sound_gen``, ``action_gen``) and, for Edge, the backbone family: a dense
Nemotron text tower (``hidden_act="relu2"``, Nemotron RMSNorm ordering, no
text QK-norm, a ``k_norm_und_for_gen`` on the understanding K the generation
tower reads) instead of Nano's Qwen3-VL one.

Edge checkpoints also carry the reasoner (the understanding tower served as a
VLM): a top-level ``config.json`` with the SigLIP2-style vision tower and
patch-merger projector, plus ``vision_encoder/model.safetensors``. That is
parsed into ``Cosmos3Config.reasoner`` (``None`` for checkpoints without it).

Values load from a local HF checkpoint directory laid out the diffusers way::

    <ckpt>/transformer/config.json   -> the DiT (dual-pathway MoT) dimensions
    <ckpt>/vae/config.json           -> AutoencoderKLWan factors + latent stats
    <ckpt>/scheduler/scheduler_config.json -> UniPC flow scheduler settings (or the
                                        distilled checkpoints' FlowMatchEuler SDE sampler)
    <ckpt>/model_index.json          -> pipeline flags (native flow schedule)
    <ckpt>/modular_model_index.json  -> distilled sampler (is_distilled, distilled_sigmas)
    <ckpt>/config.json               -> reasoner (vision tower + projector), Edge only
    <ckpt>/preprocessor_config.json, video_preprocessor_config.json -> reasoner media processors

Dataclass defaults mirror Cosmos3-Nano so a bare ``Cosmos3Config()`` is a
valid Nano config without any file present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _filtered(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    """Keep only the dict entries that name a field on the dataclass ``cls``."""
    names = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in d.items() if k in names}


@dataclass
class Cosmos3VAEConfig:
    """The Wan2.2-TI2V-5B VAE (``AutoencoderKLWan``) parameters we need at the
    serving layer. The full VAE module loads from the ``vae/`` subfolder via
    diffusers; here we only track the latent geometry and the per-channel
    normalization statistics the pipeline applies to/from latent space.
    """

    z_dim: int = 48
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 4
    # Per-channel latent normalization (length == z_dim). The pipeline maps
    # raw VAE latents x -> (x - mean) / std before denoising and inverts it
    # before decode.
    latents_mean: list[float] = field(default_factory=list)
    latents_std: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cosmos3VAEConfig":
        return cls(**_filtered(cls, d))


@dataclass
class Cosmos3SchedulerConfig:
    """Flow scheduler settings (``scheduler/scheduler_config``).

    The denoise loop drives a diffusers scheduler configured from these fields
    — ``UniPCMultistepScheduler`` for the base checkpoints (we do not
    re-implement the bh2 corrector), ``FlowMatchEulerDiscreteScheduler`` with
    stochastic (SDE) steps over the fixed ``distilled_sigmas`` for the 4-step
    distilled ones; ``scheduler_class`` records which.
    """

    scheduler_class: str = "UniPCMultistepScheduler"
    scheduler_type: str = "unipc"
    prediction_type: str = "flow_prediction"
    predict_x0: bool = True
    solver_order: int = 2
    solver_type: str = "bh2"
    use_flow_sigmas: bool = True
    use_karras_sigmas: bool = True
    final_sigmas_type: str = "zero"
    num_train_timesteps: int = 1000
    flow_shift: float = 1.0
    sigma_min: float = 0.147
    sigma_max: float = 200.0
    # FlowMatchEuler (distilled): re-noise every position each step.
    stochastic_sampling: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cosmos3SchedulerConfig":
        # diffusers stores the flow shift under "flow_shift"; keep the rest by name.
        cfg = cls(**_filtered(cls, d))
        if d.get("_class_name"):
            cfg.scheduler_class = str(d["_class_name"])
        return cfg


@dataclass
class Cosmos3VisionEncoderConfig:
    """The reasoner's packed SigLIP2-style vision tower (``vision_config`` of the
    Edge ``config.json``): patch-embedding linear over ``patch_size**2 * 3``
    pixel patches, a learned square position grid of ``num_patches`` entries
    that is bilinearly resized to each image's patch grid, ``num_hidden_layers``
    pre-LayerNorm encoder blocks attending within one frame, and a post
    LayerNorm. ``spatial_merge_size`` is the projector's 2x2 patch merge."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 16
    num_patches: int = 256
    spatial_merge_size: int = 2
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cosmos3VisionEncoderConfig":
        return cls(**_filtered(cls, d))


@dataclass
class Cosmos3MediaProcessorConfig:
    """Resize/normalize/patchify settings of the reasoner's image and video
    processors (``preprocessor_config.json`` / ``video_preprocessor_config.json``).

    An input is resized (bicubic, antialiased) so both sides are multiples of
    ``patch_size * merge_size`` and the pixel count lands in
    ``[min_pixels, max_pixels]``, normalized with ``image_mean`` /
    ``image_std``, and cut into ``patch_size`` patches in block-major 2x2
    order. Video inputs are first sampled at ``fps`` frames per second, clamped
    to ``[min_frames, max_frames]``."""

    patch_size: int = 16
    merge_size: int = 2
    temporal_patch_size: int = 1
    min_pixels: int = 65536
    max_pixels: int = 16777216
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    # Video sampling (the video processor only).
    fps: float = 2.0
    min_frames: int = 4
    max_frames: int = 768

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cosmos3MediaProcessorConfig":
        kwargs = _filtered(cls, d)
        size = d.get("size") or {}
        if "shortest_edge" in size:
            kwargs["min_pixels"] = int(size["shortest_edge"])
        if "longest_edge" in size:
            kwargs["max_pixels"] = int(size["longest_edge"])
        for key in ("image_mean", "image_std"):
            if key in kwargs:
                kwargs[key] = tuple(float(x) for x in kwargs[key])
        return cls(**kwargs)


@dataclass
class Cosmos3ReasonerConfig:
    """Everything the understanding tower needs beyond the DiT weights to be
    served as a VLM (the Edge ``config.json``): the vision tower, the
    patch-merger projector (``LayerNorm -> 2x2 merge -> Linear -> GELU ->
    Linear`` into the text hidden size), the placeholder token ids, and the
    media processors. The text tower itself is the DiT's understanding
    pathway (``embed_tokens`` / ``layers.N.self_attn.to_*`` / ``mlp`` /
    ``norm`` / ``lm_head``)."""

    vision: Cosmos3VisionEncoderConfig = field(default_factory=Cosmos3VisionEncoderConfig)
    image_processor: Cosmos3MediaProcessorConfig = field(default_factory=Cosmos3MediaProcessorConfig)
    video_processor: Cosmos3MediaProcessorConfig = field(
        default_factory=lambda: Cosmos3MediaProcessorConfig(min_pixels=4096, max_pixels=25165824)
    )
    projector_input_hidden_size: int = 1152
    projector_hidden_size: int = 11520
    projector_out_hidden_size: int = 2048
    use_postshuffle_norm: bool = False
    image_token_id: int = 19
    video_token_id: int = 18
    vision_start_token_id: int = 20
    vision_end_token_id: int = 21
    eos_token_id: int = 11
    max_position_embeddings: int = 131072
    # The chat template thinks by default (``enable_thinking``); a request may
    # turn it off.
    enable_thinking: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cosmos3ReasonerConfig":
        kwargs = _filtered(cls, d)
        vision = d.get("vision_config") or {}
        proj = d.get("projector_config") or {}
        text = d.get("text_config") or {}
        kwargs["vision"] = Cosmos3VisionEncoderConfig.from_dict(
            {**vision, "spatial_merge_size": proj.get("spatial_merge_size", vision.get("spatial_merge_size", 2))}
        )
        if "input_hidden_size" in proj:
            kwargs["projector_input_hidden_size"] = int(proj["input_hidden_size"])
        if "merger_intermediate_size" in proj:
            kwargs["projector_hidden_size"] = int(proj["merger_intermediate_size"])
        elif "projector_hidden_size" in d:
            kwargs["projector_hidden_size"] = int(d["projector_hidden_size"])
        if "out_hidden_size" in proj:
            kwargs["projector_out_hidden_size"] = int(proj["out_hidden_size"])
        elif "hidden_size" in text:
            kwargs["projector_out_hidden_size"] = int(text["hidden_size"])
        if "use_postshuffle_norm" in proj:
            kwargs["use_postshuffle_norm"] = bool(proj["use_postshuffle_norm"])
        eos = text.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0]
        if eos is not None:
            kwargs["eos_token_id"] = int(eos)
        if "max_position_embeddings" in text:
            kwargs["max_position_embeddings"] = int(text["max_position_embeddings"])
        return cls(**kwargs)


# ``model_type`` of a checkpoint whose top-level config.json describes the
# reasoner (vision tower + projector over the DiT's text pathway).
REASONER_MODEL_TYPES: frozenset[str] = frozenset({"cosmos3_edge"})
# transformer/config.json ``backbone_type`` of the Edge checkpoints, and the
# model card's serving recipe for them: 832x480 video (121 frames, 20 UniPC
# steps on the native flow schedule at flow shift 12), 640x640 images,
# cover-scale + center-crop image conditioning (the diffusers 0.40 pipeline
# recipe), action modes at flow shift 10. Applied by ``from_pretrained`` where
# a field still holds the Nano default, so a yaml overrides any of them.
EDGE_BACKBONE_TYPE = "cosmos3_edge_nemotron_dense"
EDGE_RECIPE_DEFAULTS = {
    "conditioning_resize": "aspect_crop",
    "image_size_default": (640, 640),
    "video_size_default": (832, 480),
    "num_frames_video": 121,
    "num_inference_steps_video": 20,
    "flow_shift_video": 12.0,
    "flow_shift_action": 10.0,
}


@dataclass
class Cosmos3Config:
    """Cosmos3 generator configuration (one architecture, swappable weights)."""

    # ----- dual-pathway MoT transformer (the DiT) -----
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    max_position_embeddings: int = 262144

    # ----- 3D interleaved mRoPE -----
    rope_theta: float = 5_000_000.0
    rope_axes_dim: tuple[int, int, int] = (24, 20, 20)  # rope_scaling.mrope_section
    mrope_interleaved: bool = True
    unified_3d_mrope_temporal_modality_margin: int = 15000
    unified_3d_mrope_reset_spatial_ids: bool = True
    base_fps: int = 24
    enable_fps_modulation: bool = True

    # ----- latent geometry / patchify -----
    latent_channel: int = 48
    latent_patch_size: int = 2
    patch_latent_dim: int = 192  # latent_patch_size**2 * latent_channel
    timestep_scale: float = 0.001

    # ----- attention / norm style -----
    joint_attn_implementation: str = "two_way"  # GEN attends [UND|GEN]; UND causal, UND-only
    qk_norm_for_diffusion: bool = True
    qk_norm_for_text: bool = True
    use_moe: bool = True  # MoT two-FFN split (mlp / mlp_moe_gen), NOT sparse experts

    # ----- backbone family -----
    # Nano/Super descend from Qwen3-VL: SwiGLU MLPs (gate/up/down) and the
    # diffusers RMSNorm rounding (normalize, round to bf16, multiply by the
    # weight). Edge descends from a dense Nemotron LM: ``hidden_act="relu2"``
    # (down(relu(up(x))^2), no gate) and the Nemotron RMSNorm ordering (the
    # weight multiplies in fp32, one rounding at the end). Both are read from
    # transformer/config.json; ``hidden_act`` selects the norm family like the
    # diffusers reference does.
    hidden_act: str = "silu"
    backbone_type: str | None = None
    # Edge only: the generation tower attends to a re-normalized view of the
    # understanding K (``layers.N.self_attn.k_norm_und_for_gen``); the text
    # tower's own causal attention keeps the raw K.
    use_und_k_norm_for_gen: bool = False

    # ----- capability flags + modality heads -----
    action_gen: bool = True
    max_action_dim: int = 64
    num_embodiment_domains: int = 32
    sound_gen: bool = True
    sound_dim: int | None = 64
    sound_latent_fps: float = 25.0
    temporal_compression_factor_sound: int = 1
    # Sample rate of the checkpoint's AVAE sound tokenizer. Not in the
    # transformer config; the tokenizer's own config.json is authoritative and
    # cross-checked at load (sound_latent_fps == sample_rate / hop_size).
    sound_sample_rate: int = 48000
    # Serve opt-in sound generation (the video_sound_gen walk + the
    # audio_decoder node with its ~1.9 GB AVAE). Requires the checkpoint to ship
    # sound_tokenizer/; set False to serve video-only and skip loading it.
    enable_sound: bool = True
    # Serve opt-in windowed autoregressive video (the video_gen_ar walk plus
    # the vae_decoder_ar node and its streaming decoder partition). Off by
    # default: the served node set, walks and partitions are unchanged unless
    # a deployment enables it, and requests only run windowed when they ask
    # for a ``window_mode``.
    enable_windowed_video: bool = False
    # Cap on how many windows one request may span (bounds the AR loop's
    # static iteration count together with max_inference_steps).
    max_windows: int = 64
    # Windowed-request defaults, in pixel frames (quantized to latent frames
    # server-side; the Wan VAE downsamples time by scale_factor_temporal).
    # The overlap default matches the V2V recipe's two pinned latent frames —
    # one-frame conditioning visibly degrades the continuation.
    window_frames_default: int = 29
    overlap_frames_default: int = 8
    # kv-mode committed-context horizon (61 px = 16 latent frames): frames
    # older than this behind the commit frontier are released from the cache.
    # A request may pass context_frames=0 to retain everything.
    context_frames_default: int = 61
    # Latent frames of already-generated context re-decoded ahead of each
    # window so the causal VAE's conv stack is warm at the kept frames; the
    # context-derived pixels are trimmed. Raise if window boundaries seam.
    windowed_decode_context_latents: int = 8
    # Sessions (``session_id`` on a windowed request): the DiT node keeps the
    # last window's clean latents and the streaming decoder its decode
    # context per session, so a later request with ``resume_session`` picks
    # the rollout up where the previous one ended. Most-recent sessions kept.
    session_store_size: int = 8
    video_temporal_causal: bool = False
    freeze_und: bool = False

    # ----- default sampling (overridable per request / yaml) -----
    # Number of denoise model evaluations. The per-mode cookbook defaults are
    # t2i 50, t2v/i2v 35, action fd/id 30, DROID policy ~4. ``num_inference_steps``
    # is the image default; ``num_inference_steps_video`` is the video default.
    # A request may override either; the value is clamped to ``max_inference_steps``.
    num_inference_steps: int = 50
    num_inference_steps_video: int = 35
    # Upper bound on the denoise loop's iteration count. The loop is built with
    # this many iterations and each request stops early at its own step count, so
    # one graph serves any per-request step count up to this cap.
    max_inference_steps: int = 100
    # Default frames-per-second for video generation + mp4 playback (overridable
    # per request via ``fps``).
    fps: float = 24.0
    # Default frame count for a video request that doesn't specify ``num_frames``
    # (the Wan VAE downsamples time by 4, so latent frames = 1 + (n - 1) // 4).
    num_frames_video: int = 17
    # Action-request sampling defaults (all three action modes), following the
    # reference action serving recipe: 30-step denoise, guidance 1.0, flow
    # shift 5.0 (the 480p training shift). Checkpoint yamls override — the
    # released DROID policy serves 4 steps at guidance 3.0.
    num_inference_steps_action: int = 30
    guidance_scale_action: float = 1.0
    flow_shift_action: float = 5.0
    # Default output size (width, height) for image and video requests that
    # send no ``size``: Nano/Super serve 1024^2 images and the same square for
    # video unless the deployment says otherwise; Edge is 480p-native (the
    # yaml sets 640x640 images and 832x480 video).
    image_size_default: tuple[int, int] = (1024, 1024)
    video_size_default: tuple[int, int] | None = None
    # Classifier-free guidance defaults for image/video requests (action
    # requests use ``guidance_scale_action``).
    guidance_scale: float = 6.0
    # Flow shifts: text-to-image follows the reference t2i recipe (3.0);
    # video keeps the checkpoint scheduler's shift unless set (Edge: 12.0).
    flow_shift_image: float | None = 3.0
    flow_shift_video: float | None = None
    # How an image-to-video conditioning frame reaches the generation size:
    # "stretch" (plain bilinear resize; the diffusers 0.39 pipeline the Nano
    # checkpoints were validated against) or "aspect_crop" (cover-scale,
    # antialiased resize, center crop, 8-bit rounding; the diffusers 0.40 /
    # vLLM-Omni recipe, set by the Edge yamls). See components/conditioning.py.
    conditioning_resize: str = "stretch"
    # ``model_index.json``: the pipeline sets the UniPC schedule from
    # explicitly linspaced flow sigmas (1 - 1/T ... 0) instead of the
    # scheduler's own timestep spacing. Edge checkpoints set it; the karras
    # transform is off on that path (the reference recipes pass
    # ``use_karras_sigmas=False``) unless a request re-enables it.
    use_native_flow_schedule: bool = False
    # ``modular_model_index.json`` of the 4-step distilled task checkpoints
    # (Super-Text2Image-4Step / Image2Video-4Step): the sampler is a fixed
    # sigma list [1.0, 0.9375, 0.8333, 0.625] driven by a FlowMatchEuler SDE
    # step (``x0 = x - sigma * v``, ``x' = (1 - sigma') x0 + sigma' noise``),
    # classifier-free guidance is baked into the weights (scale forced to 1),
    # and the step count is the list's length.
    is_distilled: bool = False
    distilled_sigmas: tuple[float, ...] | None = None

    # ----- denoise CUDA-graph capture (serving knobs) -----
    # Capture the fixed-shape denoise step as a CUDA graph (the launch-bound-tier
    # accelerator). Set False to serve the denoise loop eagerly. The env var
    # COSMOS3_DISABLE_CUDA_GRAPH, when set, overrides this.
    cuda_graph: bool = True
    # Only capture resolutions whose latent H*W is at or below this; larger tiers
    # (720p+, video) run eager+dense where the graph is net-slower. The env var
    # COSMOS3_GRAPH_MAX_LATENT_AREA overrides this.
    graph_max_latent_area: int = 2000
    # Video denoise steps to capture as CUDA graphs, as (height, width,
    # frames) pixel tiers: a plain t2v/i2v clip length and/or the windowed
    # rollout's window length. Pays only for small, launch-bound tiers: at
    # 832x480 the captured (paged-attention) step measured slower than the
    # eager dense FA3 one, so the Edge yaml leaves this empty.
    # The graph is built with every latent frame declared noisy and carries
    # the clean/noisy layout as a per-token mask input, so one graph per shape
    # serves t2v, i2v (anchor frame) and chained windows (overlap frames);
    # kv-mode windows run eager (their commit iteration is a different step).
    # Empty = no video capture. COSMOS3_GEN_CAPTURE_VIDEO ("480x832x29,...",
    # height x width x frames) overrides.
    gen_capture_video: tuple[tuple[int, int, int], ...] = ()
    # torch.compile the denoise compute (the generation-layer stack around the
    # attention op). Always a win in serving; the parity tests set False to keep
    # their bit-exact bounds on the eager step.
    compile_denoise: bool = True
    # Which attention backends the DiT node declares (see
    # Cosmos3Model.get_node_resources). "dense_gen" (the default) declares the
    # paged FlashInfer backend the understanding prefill and the captured
    # denoise graphs run on, plus a dense FA3 one an eager denoise step runs
    # instead: one varlen pass over [frozen text prefix | fresh gen tokens],
    # skipping the paged path's per-step K/V write and wrapper plan.
    # "flashinfer" declares only the paged backend, so every step uses it.
    attention_backend: str = "dense_gen"

    # ----- sub-configs -----
    vae: Cosmos3VAEConfig = field(default_factory=Cosmos3VAEConfig)
    scheduler: Cosmos3SchedulerConfig = field(default_factory=Cosmos3SchedulerConfig)
    # The understanding tower served as a VLM (vision tower + projector);
    # None for checkpoints that ship no reasoner (Nano/Super generators).
    reasoner: Cosmos3ReasonerConfig | None = None
    # Serve the reasoner walks (and load the vision tower) when the checkpoint
    # has them; a deployment may switch them off to serve the generator alone.
    enable_reasoner: bool = True
    # Default sampling temperature for reasoner requests that send none
    # (0 = greedy). The checkpoint's generation_config samples; the M* default
    # keeps the other chat models' 0.6.
    reasoner_temperature: float = 0.6

    # ----- provenance -----
    local_dir: str = ""

    @property
    def nemotron_norm(self) -> bool:
        """Whether every RMSNorm uses the Nemotron ordering (fp32 weight
        multiply, then one cast) — the dense relu2 backbone family."""
        return self.hidden_act == "relu2"

    @property
    def gated_mlp(self) -> bool:
        """SwiGLU (gate/up/down) MLPs vs the dense two-projection relu2 ones."""
        return self.hidden_act != "relu2"

    @property
    def serves_reasoner(self) -> bool:
        return self.reasoner is not None

    @classmethod
    def from_transformer_dict(cls, d: dict[str, Any]) -> "Cosmos3Config":
        """Build from a diffusers ``transformer/config.json`` dict alone.

        Sub-configs are left at their defaults; use ``from_pretrained`` to also
        populate VAE/scheduler from their sibling folders.
        """
        kwargs = _filtered(cls, d)
        rope = d.get("rope_scaling") or {}
        if "mrope_section" in rope:
            kwargs["rope_axes_dim"] = tuple(rope["mrope_section"])
        if "mrope_interleaved" in rope:
            kwargs["mrope_interleaved"] = bool(rope["mrope_interleaved"])
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "Cosmos3Config":
        """Load from a diffusers-layout checkpoint directory."""
        root = Path(local_dir)
        tcfg_path = root / "transformer" / "config.json"
        if not tcfg_path.exists():
            raise FileNotFoundError(f"transformer/config.json not found under {root}")
        with open(tcfg_path) as f:
            cfg = cls.from_transformer_dict(json.load(f))
        cfg.local_dir = str(root)

        vae_path = root / "vae" / "config.json"
        if vae_path.exists():
            with open(vae_path) as f:
                cfg.vae = Cosmos3VAEConfig.from_dict(json.load(f))

        sched_path = root / "scheduler" / "scheduler_config.json"
        if sched_path.exists():
            with open(sched_path) as f:
                cfg.scheduler = Cosmos3SchedulerConfig.from_dict(json.load(f))

        index_path = root / "model_index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            cfg.use_native_flow_schedule = bool(index.get("use_native_flow_schedule", False))

        if cfg.backbone_type == EDGE_BACKBONE_TYPE:
            defaults = cls()
            for name, value in EDGE_RECIPE_DEFAULTS.items():
                if getattr(cfg, name) == getattr(defaults, name):
                    setattr(cfg, name, value)

        modular_path = root / "modular_model_index.json"
        if modular_path.exists():
            with open(modular_path) as f:
                modular = json.load(f)
            sigmas = modular.get("distilled_sigmas")
            if modular.get("is_distilled") and sigmas:
                cfg.is_distilled = True
                cfg.distilled_sigmas = tuple(float(s) for s in sigmas)

        top_path = root / "config.json"
        if top_path.exists():
            with open(top_path) as f:
                top = json.load(f)
            if top.get("model_type") in REASONER_MODEL_TYPES and (root / "vision_encoder").exists():
                reasoner = Cosmos3ReasonerConfig.from_dict(top)
                for name, attr in (
                    ("preprocessor_config.json", "image_processor"),
                    ("video_preprocessor_config.json", "video_processor"),
                ):
                    proc_path = root / name
                    if proc_path.exists():
                        with open(proc_path) as f:
                            setattr(reasoner, attr, Cosmos3MediaProcessorConfig.from_dict(json.load(f)))
                cfg.reasoner = reasoner

        return cfg
