"""
BagelModel: Model implementation for BAGEL (ByteDance) unified multimodal model.

BAGEL uses a Qwen2 LLM with MoT (Mixture-of-Transformers) architecture,
SigLIP2 ViT for image understanding, and FLUX VAE for image generation.
The LLM itself serves as the denoiser for rectified flow image generation
(no separate diffusion model).

Architecture (4 stages):
    vit_encoder   (enc_dec) - SigLIP2 ViT + connector + pos embed
    vae_encoder   (enc_dec) - VAE encode + patchify + projection
    LLM           (ar)      - Fat stage: embed + Qwen2 + lm_head + CFG + Euler
    vae_decoder   (enc_dec) - VAE decode to pixels

Phases (5):
    prefill_text  - Text token embedding + LLM prefill (causal)
    prefill_vit   - ViT encoding + LLM prefill (bidirectional for images)
    prefill_vae   - VAE encoding + LLM prefill (bidirectional for images)
    decode        - Autoregressive text generation
    image_gen     - Flow matching loop (3-pass CFG + Euler) + VAE decode

The LLM stage absorbs text_emb, lm_head, and flow_proj because they are
always colocated on the same GPU. Keeping them as separate graph stages
would add unnecessary IPC overhead. CFG requires 3 LLM forward passes +
velocity combination, which is easier as one atomic operation.

Output mode is known upfront from the API request's output_modalities
field (no BOI token detection). Prefill is sequential: text tokens are
processed causally, then each image is processed bidirectionally.
"""

import os

import torch
import torch.nn as nn
import yaml

from mminf.communication.tensors import NameToTensorList
from mminf.graph.base import (
    GraphPointer,
    GraphSection,
    GraphStage,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mminf.engine.base import EngineType
from mminf.model.bagel.modeling_utils import PositionEmbedding, TimestepEmbedder
from mminf.model.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
from mminf.model.bagel.tokenization_qwen2 import Qwen2Tokenizer
from mminf.model.bagel.utils import add_special_tokens
from mminf.model.base import STREAM_OUT, CurrentForwardMetadata, Model, StageSubmodule


# ---------------------------------------------------------------------------
# System prompts (used when think_mode=True)
# ---------------------------------------------------------------------------

VLM_THINK_SYSTEM_PROMPT = (
    "You should first think about the reasoning process in the mind "
    "and then provide the user with the answer."
)

GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind "
    "and then generate the image."
)


# ---------------------------------------------------------------------------
# StageSubmodule wrappers
# ---------------------------------------------------------------------------


class ViTEncoderSubmodule(StageSubmodule):
    """SigLIP2 ViT + connector + vit_pos_embed: pixel patches -> ViT features.

    Receives preprocessed inputs containing packed pixel values, position IDs,
    cumulative sequence lengths, and max sequence length. Both vit_encoder and
    vae_encoder receive "image_inputs" as their graph input name; routing is
    handled by the graph pointer's next_stage field.
    """

    def __init__(
        self,
        vit_model: nn.Module,
        connector: nn.Module,
        vit_pos_embed: nn.Module,
    ):
        super().__init__()
        self.vit_model = vit_model
        self.connector = connector
        self.vit_pos_embed = vit_pos_embed

    def preprocess(self, image_inputs: list[torch.Tensor]) -> dict:
        """Convert raw images to packed ViT input format.

        Full implementation should include prepare_vit_images logic from BAGEL:
        - Dynamic resolution computation and SigLIP2 image preprocessing
        - Patch splitting and flattening
        - Position ID computation from image grid
        - Packing multiple images with cu_seqlens for FlashAttention

        Currently assumes single pre-processed image tensor where
        image_inputs[0] has shape [num_tokens, patch_dim].
        """
        # TODO: Full prepare_vit_images integration (SigLIP2 preprocessing)
        pixel_values = image_inputs[0]
        num_tokens = pixel_values.shape[0]
        device = pixel_values.device

        # Compute cu_seqlens for FlashAttention
        vit_token_seqlens = torch.tensor(
            [num_tokens], dtype=torch.int32, device=device
        )
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(vit_token_seqlens, dim=0), (1, 0)
        ).to(torch.int32)
        max_seqlen = int(num_tokens)

        # TODO: compute position_ids from image grid structure
        position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)

        return {
            "packed_pixel_values": pixel_values,
            "packed_position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        **kwargs,
    ) -> NameToTensorList:
        features = self.vit_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        features = self.connector(features)
        pos_emb = self.vit_pos_embed(packed_position_ids)
        features = features + pos_emb
        return {"vit_emb": [features]}


class VAEEncoderSubmodule(StageSubmodule):
    """VAE encode + patchify + vae2llm + time_embedder + latent_pos_embed.

    Encodes an image tensor to VAE latents, patchifies them, and projects
    into the LLM hidden dimension with positional and timestep embeddings.
    """

    def __init__(
        self,
        vae_model: nn.Module,
        vae2llm: nn.Linear,
        time_embedder: nn.Module,
        latent_pos_embed: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
        latent_downsample: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.latent_downsample = latent_downsample

    def preprocess(self, image_inputs: list[torch.Tensor]) -> dict:
        """Convert raw images to VAE encoder input format.

        Computes patchified dimensions as Python ints for CUDA graph
        compatibility (no .item() calls in forward).

        Full implementation should include:
        - Image padding to be divisible by latent_downsample * latent_patch_size
        - VAE position ID computation from latent grid
        - Timestep preparation
        """
        padded_images = image_inputs[0]  # [B, C, H, W]
        device = padded_images.device

        # Compute patchified dimensions as ints (CUDA graph compatible)
        p = self.latent_patch_size
        ds = self.latent_downsample
        _, _, img_h, img_w = padded_images.shape
        h = (img_h // ds) // p
        w = (img_w // ds) // p

        # TODO: proper VAE position IDs based on latent grid structure
        num_patches = h * w
        packed_vae_position_ids = torch.arange(num_patches, device=device)

        # t=0 for initial VAE encoding step
        packed_timesteps = torch.zeros(num_patches, device=device)

        return {
            "padded_images": padded_images,
            "packed_vae_position_ids": packed_vae_position_ids,
            "packed_timesteps": packed_timesteps,
            "h": h,
            "w": w,
        }

    def forward(
        self,
        padded_images: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_timesteps: torch.Tensor,
        h: int,
        w: int,
        **kwargs,
    ) -> NameToTensorList:
        latent = self.vae_model.encode(padded_images)

        p = self.latent_patch_size
        # h, w are already ints from preprocess (CUDA graph compatible)
        packed_latent = []
        for lat in latent:
            lat = lat[:, :h * p, :w * p].reshape(
                self.latent_channel, h, p, w, p
            )
            lat = torch.einsum("chpwq->hwpqc", lat).reshape(
                -1, p * p * self.latent_channel
            )
            packed_latent.append(lat)
        packed_latent = torch.cat(packed_latent, dim=0)

        # Project to hidden dim with timestep and position embeddings
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        return {"vae_emb": [packed_latent]}


class LLMSubmodule(StageSubmodule):
    """Fat LLM wrapper that dispatches based on phase.

    Absorbs text_emb, lm_head, and flow_proj into a single stage to avoid
    unnecessary IPC overhead. Phase-based dispatch handles:

      - prefill_text: embed_tokens -> LLM forward (causal, mode="und")
      - prefill_vit:  BOI + vit_emb + EOI -> LLM forward (bidirectional)
      - prefill_vae:  BOI + vae_emb + EOI -> LLM forward (bidirectional)
      - decode:       embed_tokens -> LLM forward -> lm_head -> argmax
      - image_gen:    3-pass CFG -> llm2vae -> velocity combine -> Euler step

    BOI/EOI tokens (<|vision_start|>, <|vision_end|>) are structural
    delimiters manually inserted around image embeddings during prefill.
    They are NOT predicted by the model (excluded from CE loss during
    training).

    During image_gen, classifier-free guidance requires 3 LLM forward
    passes with different KV caches (main, cfg_text, cfg_img). The
    velocities are combined via:
        v_final = v_cfg_img + img_scale * (
            v_cfg_text + text_scale * (v_main - v_cfg_text) - v_cfg_img
        )
    followed by an Euler step: x_{t+1} = x_t + v_final * dt.

    Multi-cache orchestration is driven by per-request metadata:
      - cache_labels: list of cache labels to iterate over (default ["main"])
      - snapshot_after: (from_label, to_label) to snapshot after this step
    The CacheHandle (provided by AREngine) manages label switching, page
    allocation, and KV data copying.
    """

    def __init__(
        self,
        language_model: nn.Module,
        llm2vae: nn.Linear,
        boi_token_id: int | None = None,
        eoi_token_id: int | None = None,
    ):
        super().__init__()
        self.language_model = language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.llm2vae = llm2vae
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id

    def preprocess(self, **inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Unwrap single-element tensor lists and handle latent initialization.

        For image_gen phase: if "latents" input is empty (first flow matching
        iteration), initializes random noise. The latent shape (latent_seq_len,
        latent_dim) must be provided via per-request metadata since it depends
        on the output image dimensions.

        For all other phases: standard unwrapping of list[Tensor] -> Tensor.
        """
        result = {}
        for k, v in inputs.items():
            if k == "latents" and (not v or v[0].numel() == 0):
                # First flow matching iteration: no latents yet.
                # Latent noise will be initialized in forward using shape
                # info from per-request metadata (latent_seq_len, latent_dim).
                result[k] = None
            elif v:
                result[k] = v[0]
        return result

    def forward(self, phase: str, cache_handle=None, **kwargs) -> NameToTensorList:
        if phase == "prefill_text":
            return self._forward_prefill_text(cache_handle=cache_handle, **kwargs)
        elif phase == "prefill_vit":
            return self._forward_prefill_vit(cache_handle=cache_handle, **kwargs)
        elif phase == "prefill_vae":
            return self._forward_prefill_vae(cache_handle=cache_handle, **kwargs)
        elif phase == "decode":
            return self._forward_decode(cache_handle=cache_handle, **kwargs)
        elif phase == "image_gen":
            return self._forward_image_gen(cache_handle=cache_handle, **kwargs)
        else:
            raise ValueError(f"Unknown LLM phase: {phase!r}")

    def _forward_prefill_text(self, text_inputs: torch.Tensor, cache_handle=None, **kwargs) -> NameToTensorList:
        """embed_tokens -> LLM forward (causal, mode='und') -> KV cache update.

        For generation mode, cache_labels specifies which caches to update
        (e.g., ["main", "cfg_img"] for text prefill).
        """
        emb = self.embed_tokens(text_inputs)
        cache_labels = kwargs.pop("cache_labels", ["main"])
        snapshot_after = kwargs.pop("snapshot_after", None)
        for label in cache_labels:
            if cache_handle is not None:
                cache_handle.set_active_label(label)
            self.language_model(emb, is_causal=True, mode="und",
                                cache_handle=cache_handle, **kwargs)
        if snapshot_after and cache_handle is not None:
            from_label, to_label = snapshot_after
            cache_handle.snapshot(from_label, to_label)
        return {}

    def _forward_prefill_vit(self, vit_emb: torch.Tensor, cache_handle=None, **kwargs) -> NameToTensorList:
        """Wrap vit_emb with BOI/EOI tokens -> LLM forward (bidirectional).

        For generation mode, cache_labels specifies which caches to update.
        snapshot_after triggers a KV cache deepcopy after processing.
        """
        combined = self._wrap_with_boi_eoi(vit_emb)
        cache_labels = kwargs.pop("cache_labels", ["main"])
        snapshot_after = kwargs.pop("snapshot_after", None)
        for label in cache_labels:
            if cache_handle is not None:
                cache_handle.set_active_label(label)
            self.language_model(combined, is_causal=False, mode="und",
                                cache_handle=cache_handle, **kwargs)
        if snapshot_after and cache_handle is not None:
            from_label, to_label = snapshot_after
            cache_handle.snapshot(from_label, to_label)
        return {}

    def _forward_prefill_vae(self, vae_emb: torch.Tensor, cache_handle=None, **kwargs) -> NameToTensorList:
        """Wrap vae_emb with BOI/EOI tokens -> LLM forward (bidirectional).

        For generation mode, cache_labels specifies which caches to update.
        snapshot_after triggers a KV cache deepcopy after processing.
        """
        combined = self._wrap_with_boi_eoi(vae_emb)
        cache_labels = kwargs.pop("cache_labels", ["main"])
        snapshot_after = kwargs.pop("snapshot_after", None)
        for label in cache_labels:
            if cache_handle is not None:
                cache_handle.set_active_label(label)
            self.language_model(combined, is_causal=False, mode="und",
                                cache_handle=cache_handle, **kwargs)
        if snapshot_after and cache_handle is not None:
            from_label, to_label = snapshot_after
            cache_handle.snapshot(from_label, to_label)
        return {}

    def _forward_decode(self, text_inputs: torch.Tensor, cache_handle=None, **kwargs) -> NameToTensorList:
        """embed_tokens -> LLM forward -> lm_head -> argmax."""
        # Remove metadata keys not needed for language_model
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        emb = self.embed_tokens(text_inputs)
        if cache_handle is not None:
            cache_handle.set_active_label("main")
        hidden = self.language_model(emb, is_causal=True, mode="und",
                                     cache_handle=cache_handle, **kwargs)
        logits = self.lm_head(hidden[:, -1:])
        token = torch.argmax(logits, dim=-1)
        return {"new_token": [token]}

    def _forward_image_gen(
        self,
        latents: torch.Tensor | None = None,
        cache_handle=None,
        timestep: torch.Tensor = None,
        next_timestep: torch.Tensor = None,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        latent_seq_len: int | None = None,
        latent_dim: int | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """3-pass CFG -> llm2vae -> velocity combine -> Euler step.

        Uses cache_handle to switch between the 3 frozen KV caches
        (main, cfg_text, cfg_img). write_cache=False since caches are
        frozen during flow matching.

        If latents is None (first iteration), initializes random noise
        using latent_seq_len and latent_dim from per-request metadata.
        """
        # Remove metadata keys not needed for language_model
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)

        # Initialize random noise for first flow matching iteration
        if latents is None:
            if latent_seq_len is None or latent_dim is None:
                raise ValueError(
                    "latent_seq_len and latent_dim must be provided via "
                    "per-request metadata for first image_gen iteration"
                )
            device = next(self.parameters()).device
            latents = torch.randn(latent_seq_len, latent_dim, device=device)
        velocities = {}
        for label in ["main", "cfg_text", "cfg_img"]:
            if cache_handle is not None:
                cache_handle.set_active_label(label)
            hidden = self.language_model(
                latents, is_causal=False, mode="gen",
                cache_handle=cache_handle, write_cache=False, **kwargs,
            )
            velocities[label] = hidden

        # Project to VAE space
        v_main = self.llm2vae(velocities["main"])
        v_cfg_text = self.llm2vae(velocities["cfg_text"])
        v_cfg_img = self.llm2vae(velocities["cfg_img"])

        # CFG velocity combination
        v_final = v_cfg_img + cfg_img_scale * (
            v_cfg_text + cfg_text_scale * (v_main - v_cfg_text) - v_cfg_img
        )

        # Euler step: x_{t+1} = x_t + v * dt
        dt = next_timestep - timestep
        latents = latents + v_final * dt
        return {"latents": [latents]}

    def _wrap_with_boi_eoi(self, emb: torch.Tensor) -> torch.Tensor:
        """Wrap embeddings with <|vision_start|> and <|vision_end|> tokens."""
        if self.boi_token_id is None or self.eoi_token_id is None:
            return emb
        device = emb.device
        boi_ids = torch.tensor([self.boi_token_id], device=device)
        eoi_ids = torch.tensor([self.eoi_token_id], device=device)
        boi_emb = self.embed_tokens(boi_ids)
        eoi_emb = self.embed_tokens(eoi_ids)
        return torch.cat([boi_emb, emb, eoi_emb], dim=0)


class VAEDecoderSubmodule(StageSubmodule):
    """VAE decoder: latent grid -> pixel image."""

    def __init__(
        self,
        vae_model: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
        latent_downsample: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.latent_downsample = latent_downsample

    def preprocess(self, latents: list[torch.Tensor]) -> dict:
        """Prepare VAE decoder inputs.

        Unwraps latents from list. Image dimensions (image_h, image_w)
        are provided via per-request metadata and converted to ints for
        CUDA graph compatibility.
        """
        return {"latents": latents[0]}

    def forward(
        self,
        latents: torch.Tensor,
        image_h: int | torch.Tensor = 0,
        image_w: int | torch.Tensor = 0,
        **kwargs,
    ) -> NameToTensorList:
        # Convert to int if tensor (CUDA graph compatible when passed as int
        # from metadata; tensor fallback for backwards compatibility)
        H = image_h.item() if isinstance(image_h, torch.Tensor) else int(image_h)
        W = image_w.item() if isinstance(image_w, torch.Tensor) else int(image_w)

        p = self.latent_patch_size
        h = H // self.latent_downsample
        w = W // self.latent_downsample

        # Unpatchify: [num_patches, patch_dim] -> [1, C, H_latent, W_latent]
        latent = latents.reshape(1, h, w, p, p, self.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1, self.latent_channel, h * p, w * p
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return {"image_output": [image]}


# ---------------------------------------------------------------------------
# BagelModel
# ---------------------------------------------------------------------------


LLM_CONFIG_FILE = "llm_config.json"
MODEL_CONFIG_FILE = "model_config.yaml"


class BagelModel(Model):
    """
    BAGEL unified multimodal model (ByteDance).

    Architecture: Qwen2 LLM with MoT + SigLIP2 ViT + FLUX VAE.
    The LLM serves as both the autoregressive text model and the denoiser
    for rectified flow image generation (no separate diffusion model).

    Stages (4):
        vit_encoder   (enc_dec) - SigLIP2 ViT + connector + pos embed
        vae_encoder   (enc_dec) - VAE encode + patchify + projection
        LLM           (ar)      - Fat stage: embed + Qwen2 + lm_head + CFG
        vae_decoder   (enc_dec) - VAE decode to pixels

    Phases (5):
        prefill_text  - Text token embedding + LLM prefill (causal)
        prefill_vit   - ViT encoding + LLM prefill (bidirectional)
        prefill_vae   - VAE encoding + LLM prefill (bidirectional)
        decode        - Autoregressive text generation
        image_gen     - Flow matching loop (3-pass CFG + Euler) + VAE decode

    Phase transitions are schedule-driven (no BOI token detection). The
    output mode is known upfront from the API request's output_modalities.
    Prefill steps are constructed as a sequential schedule that walks
    through interleaved text and image inputs.
    """

    def __init__(
        self,
        config_dir: str,
        model_path_hf: str,
    ):
        self.llm_config = Qwen2Config.from_json_file(
            os.path.join(config_dir, LLM_CONFIG_FILE)
        )

        with open(os.path.join(config_dir, MODEL_CONFIG_FILE), "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path_hf = model_path_hf
        self.llm_config.qk_norm = True
        self.llm_config.tie_word_embeddings = False
        self.llm_config.layer_module = "Qwen2MoTDecoderLayer"

        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path_hf)
        self.tokenizer, new_token_ids, _ = add_special_tokens(self.tokenizer)

        self.num_timesteps = self.config["num_timesteps"]
        self.cfg_text_scale = self.config["cfg_text_scale"]
        self.cfg_img_scale = self.config["cfg_img_scale"]
        self.think_mode = self.config["think_mode"]
        self.latent_patch_size = self.config["latent_patch_size"]
        self.latent_channel = self.config["latent_channel"]
        self.max_latent_size = self.config["max_latent_size"]
        self.latent_downsample = self.config["downsample"] * self.latent_patch_size

        self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel

        # Special token IDs
        self.boi_token_id = new_token_ids.get("boi_token_id")   # <|vision_start|>
        self.eoi_token_id = new_token_ids.get("eoi_token_id")   # <|vision_end|>
        self.eos_token_id = new_token_ids.get("eos_token_id")
        self.bos_token_id = new_token_ids.get("bos_token_id")

        # Lazy init cache -- submodules created on first access via
        # get_submodule(). A worker only instantiates the submodules it
        # actually needs (e.g., a worker running only vit_encoder never
        # creates the LLMSubmodule).
        self._submodule_cache: dict[str, StageSubmodule | None] = {}
        self.language_model = None
        self.llm2vae = None
        self.vae_model = None
        self.time_embedder = None
        self.vae2llm = None
        self.latent_pos_embed = None
        self.vae_initialized = False
    
    def _init_language_model_components(self):
        self.language_model = Qwen2ForCausalLM(self.llm_config)
        self.llm2vae = nn.Linear(self.llm_config.hidden_size, self.patch_latent_dim)
        
        # TODO: properly load in weights !!!

    def _init_vae_components(self):
        if self.vae_initialized:
            return
        self.vae_initialized = True
        self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.llm_config.hidden_size)
        self.time_embedder = TimestepEmbedder(self.llm_config.hidden_size)
        self.vae2llm = nn.Linear(self.patch_latent_dim, self.llm_config.hidden_size)
       # TODO: set up VAE model
        # TODO: properly load in weights !!!
    
    def _init_vit_components(self):
        pass # TODO

    # -----------------------------------------------------------------------
    # Lazy submodule initialization
    # -----------------------------------------------------------------------

    def _create_submodule(self, stage_name: str) -> StageSubmodule | None:
        """Create a submodule wrapper on first access. Returns None in dummy mode."""
        # if self.bagel_model is None:
        #     return None

        if stage_name == "LLM":
            self._init_language_model_components()
            return LLMSubmodule(
                language_model=self.language_model,
                llm2vae=self.llm2vae,
                boi_token_id=self.boi_token_id,
                eoi_token_id=self.eoi_token_id,
            )
        elif stage_name == "vit_encoder":
            self._init_vit_components()
            return ViTEncoderSubmodule(
                vit_model=self.vit_model,
                connector=self.connector,
                vit_pos_embed=self.vit_pos_embed,
            )
        elif stage_name == "vae_encoder":
            self._init_vae_components()
            return VAEEncoderSubmodule(
                vae_model=self.vae_model,
                vae2llm=self.vae2llm,
                time_embedder=self.time_embedder,
                latent_pos_embed=self.latent_pos_embed,
                latent_patch_size=self.latent_patch_size,
                latent_channel=self.latent_channel,
                latent_downsample=self.latent_downsample,
            )
        elif stage_name == "vae_decoder":
            self._init_vit_components()
            return VAEDecoderSubmodule(
                vae_model=self.vae_model,
                latent_patch_size=self.latent_patch_size,
                latent_channel=self.latent_channel,
                latent_downsample=self.latent_downsample,
            )
        return None

    # -----------------------------------------------------------------------
    # Model ABC implementation
    # -----------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        **kwargs,
    ) -> NameToTensorList:
        """Tokenize user prompt and system prompt (if think_mode).

        Returns model-specific keys matching get_forward_pass_inputs:
            "text_inputs"    - tokenized user prompt
            "system_prompt"  - tokenized system prompt (think_mode only)
        """
        result: NameToTensorList = {}

        if prompt is not None:
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(prompt)
                result["text_inputs"] = [
                    torch.tensor(tokens, dtype=torch.long)
                ]
            else:
                # Fallback for testing without a tokenizer
                byte_data = prompt.encode("utf-8")
                result["text_inputs"] = [
                    torch.tensor(list(byte_data), dtype=torch.uint8)
                ]

        if self.think_mode and self.tokenizer is not None:
            target_output = output_modalities[0] if output_modalities else "text"
            is_understanding = (target_output == "text")
            sys_prompt = (
                VLM_THINK_SYSTEM_PROMPT if is_understanding
                else GEN_THINK_SYSTEM_PROMPT
            )
            sys_tokens = self.tokenizer.encode(sys_prompt)
            result["system_prompt"] = [
                torch.tensor(sys_tokens, dtype=torch.long)
            ]

        return result

    def get_submodule(self, stage_name: str) -> torch.nn.Module | None:
        if stage_name in self._submodule_cache:
            return self._submodule_cache[stage_name]
        submodule = self._create_submodule(stage_name)
        self._submodule_cache[stage_name] = submodule
        return submodule

    def get_stage_engine_types(self) -> dict[str, EngineType]:
        return {
            "vit_encoder": EngineType.ENC_DEC,
            "vae_encoder": EngineType.ENC_DEC,
            "LLM": EngineType.AR,
            "vae_decoder": EngineType.ENC_DEC,
        }

    def get_phase_graphs(self) -> dict[str, GraphSection]:
        # -- prefill_text: just the LLM stage (text embedding is internal) --
        # No output needed — conductor is notified when the subgraph completes.
        prefill_text = GraphStage(
            name="LLM",
            input_ids=["text_inputs"],
            outputs=[],
        )

        # -- prefill_vit: ViT encoder -> LLM --
        prefill_vit = Sequential([
            GraphStage(
                name="vit_encoder",
                input_ids=["image_inputs"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="vit_emb"),
                ],
            ),
            GraphStage(
                name="LLM",
                input_ids=["vit_emb"],
                outputs=[],
            ),
        ])

        # -- prefill_vae: VAE encoder -> LLM --
        prefill_vae = Sequential([
            GraphStage(
                name="vae_encoder",
                input_ids=["image_inputs"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="vae_emb"),
                ],
            ),
            GraphStage(
                name="LLM",
                input_ids=["vae_emb"],
                outputs=[],
            ),
        ])

        # -- decode: single LLM stage (embed + transformer + lm_head) --
        decode = GraphStage(
            name="LLM",
            input_ids=["text_inputs"],
            outputs=[
                GraphPointer(
                    next_stage=STREAM_OUT,
                    name="new_token",
                    output_modality="text",
                    is_new_token=True,
                    back_to_conductor=True,
                ),
            ],
        )

        # -- image_gen: denoising loop (LLM does CFG+Euler) -> VAE decode --
        # n_iters = num_timesteps - 1 because the loop body performs one
        # Euler step per iteration. With N timestep boundaries (e.g. 50),
        # there are N-1 intervals, so N-1 Euler steps are needed.
        image_gen = Sequential([
            Loop(
                section=GraphStage(
                    name="LLM",
                    input_ids=["latents"],
                    outputs=[
                        GraphPointer(next_stage="LLM", name="latents"),
                    ],
                ),
                n_iters=self.num_timesteps - 1,
                outputs=[
                    GraphPointer(next_stage="vae_decoder", name="latents"),
                ],
            ),
            GraphStage(
                name="vae_decoder",
                input_ids=["latents"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        name="image_output",
                        output_modality="image",
                        back_to_conductor=True,
                    ),
                ],
            ),
        ])

        return dict(
            prefill_text=prefill_text,
            prefill_vit=prefill_vit,
            prefill_vae=prefill_vae,
            decode=decode,
            image_gen=image_gen,
        )

    def get_initial_forward_metadata(
        self,
        input_modalities: list[str],
        output_modalities: list[str],
    ) -> CurrentForwardMetadata:
        target_output = output_modalities[0]  # "text" or "image"
        is_understanding = (target_output == "text")

        # Build prefill schedule: sequential list of (phase_name, step_kwargs)
        schedule: list[tuple[str, dict]] = []

        # 1. System prompt (if think mode enabled)
        if self.think_mode:
            prompt = VLM_THINK_SYSTEM_PROMPT if is_understanding else GEN_THINK_SYSTEM_PROMPT
            schedule.append(("prefill_text", {"prompt": prompt}))

        # 2. Walk through interleaved inputs, building sequential steps
        text_idx, image_idx = 0, 0
        for mod in input_modalities:
            if mod == "text":
                schedule.append(("prefill_text", {"input_idx": text_idx}))
                text_idx += 1
            elif mod == "image":
                if is_understanding:
                    # Understanding: ViT only (no VAE encoding needed)
                    schedule.append(("prefill_vit", {"input_idx": image_idx}))
                else:
                    # Generation/editing: VAE encode the image
                    schedule.append(("prefill_vae", {"input_idx": image_idx}))
                image_idx += 1

        # 3. Annotate schedule with multi-cache metadata for generation mode.
        #    BAGEL's CFG requires 3 caches: main, cfg_img, cfg_text.
        #    - Text prefill: write to main + cfg_img (text-only cache)
        #    - Image prefill (vit/vae): write to main only
        #    - After last image: snapshot main -> cfg_text (system+image cache)
        #    Understanding mode: no annotations needed (default ["main"]).
        if not is_understanding:
            last_image_idx = None
            for i, (phase, _) in enumerate(schedule):
                if phase in ("prefill_vit", "prefill_vae"):
                    last_image_idx = i

            for i, (phase, step_kwargs) in enumerate(schedule):
                if phase == "prefill_text":
                    step_kwargs["cache_labels"] = ["main", "cfg_img"]
                elif phase in ("prefill_vit", "prefill_vae"):
                    step_kwargs["cache_labels"] = ["main"]
                    if i == last_image_idx:
                        step_kwargs["snapshot_after"] = ("main", "cfg_text")

        first_phase = schedule[0][0] if schedule else "decode"

        return CurrentForwardMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            phase=first_phase,
            is_prefill=bool(schedule),
            kwargs={
                "prefill_schedule": schedule,
                "prefill_step": 0,
                "target_output": target_output,
                "num_timesteps": self.num_timesteps,
                "cfg_text_scale": self.cfg_text_scale,
                "cfg_img_scale": self.cfg_img_scale,
            },
        )

    def get_forward_pass_inputs(
        self,
        metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        prev_forward_metadata: CurrentForwardMetadata = None,
    ) -> list[GraphPointer]:
        """Construct the external inputs for the current forward pass.

        The conductor calls this to determine what tensors to send to
        workers at the start of each forward pass. For prefill phases,
        the schedule entry determines which input to route; for decode
        and image_gen, the previous output feeds back in.

        persist_signals key conventions:
            "text_inputs"    - list of per-turn text TensorPointerInfos
            "image_inputs"   - list of per-image TensorPointerInfos
            "system_prompt"  - tokenized system prompt (if think_mode)
            "new_token"      - last generated token (during decode)
            "latents"        - noise latents (for image_gen entry)
        """
        phase = metadata.phase

        if metadata.is_prefill:
            schedule = metadata.kwargs["prefill_schedule"]
            step = metadata.kwargs["prefill_step"]
            _, step_kwargs = schedule[step]

            if phase == "prefill_text":
                ptr = GraphPointer(next_stage="LLM", name="text_inputs")
                if "prompt" in step_kwargs:
                    # System prompt -- tokenized by process_prompt() in data worker
                    ptr.tensor_info = persist_signals.get("system_prompt", [])
                else:
                    idx = step_kwargs["input_idx"]
                    all_text = persist_signals.get("text_inputs", [])
                    ptr.tensor_info = [all_text[idx]] if idx < len(all_text) else []
                return [ptr]

            elif phase == "prefill_vit":
                idx = step_kwargs["input_idx"]
                ptr = GraphPointer(next_stage="vit_encoder", name="image_inputs")
                all_images = persist_signals.get("image_inputs", [])
                ptr.tensor_info = [all_images[idx]] if idx < len(all_images) else []
                return [ptr]

            elif phase == "prefill_vae":
                idx = step_kwargs["input_idx"]
                ptr = GraphPointer(next_stage="vae_encoder", name="image_inputs")
                all_images = persist_signals.get("image_inputs", [])
                ptr.tensor_info = [all_images[idx]] if idx < len(all_images) else []
                return [ptr]

        elif phase == "decode":
            # Previous token feeds back as text_inputs
            ptr = GraphPointer(next_stage="LLM", name="text_inputs")
            ptr.tensor_info = persist_signals.get("new_token", [])
            return [ptr]

        elif phase == "image_gen":
            # Initial noise latents feed the LLM denoising loop.
            # Note: latents are typically initialized by the submodule's
            # preprocess() (random noise), not passed through persist_signals.
            # This lookup handles the case where latents are externally provided.
            ptr = GraphPointer(next_stage="LLM", name="latents")
            ptr.tensor_info = persist_signals.get("latents", [])
            return [ptr]

        return []

    def update_for_next_forward(
        self,
        metadata: CurrentForwardMetadata,
        new_tokens: dict[str, list[int]],
    ) -> CurrentForwardMetadata:
        """Advance phase transitions. Schedule-driven, no BOI detection.

        During prefill, steps through the schedule one entry at a time.
        After all prefill steps, transitions to:
          - decode (text output)
          - decode (image output + think_mode: think first, then generate)
          - image_gen (image output, no think_mode)

        During decode:
          - Text output: EOS marks request complete.
          - Image output + think_mode: EOS transitions to image_gen
            (thinking is done, now generate the image).

        After image_gen, marks request complete (one image per request).
        """
        if metadata.is_prefill:
            step = metadata.kwargs["prefill_step"] + 1
            schedule = metadata.kwargs["prefill_schedule"]

            if step < len(schedule):
                # More prefill steps remaining
                metadata.kwargs["prefill_step"] = step
                metadata.phase = schedule[step][0]
            else:
                # All prefill done -- transition based on target_output
                metadata.is_prefill = False
                target = metadata.kwargs["target_output"]
                if target == "text":
                    metadata.phase = "decode"
                elif target == "image":
                    if self.think_mode:
                        # Think first: decode to generate reasoning, then
                        # EOS triggers transition to image_gen.
                        metadata.phase = "decode"
                    else:
                        metadata.phase = "image_gen"
            return metadata

        if metadata.phase == "decode":
            tokens = new_tokens.get("new_token", [])
            if self.eos_token_id is not None and self.eos_token_id in tokens:
                target = metadata.kwargs["target_output"]
                if self.think_mode and target == "image":
                    # Thinking phase complete — transition to image generation.
                    metadata.phase = "image_gen"
                else:
                    metadata.request_done = True
            # Otherwise stay in decode phase
            return metadata

        if metadata.phase == "image_gen":
            # Image generation complete (one image per request)
            metadata.request_done = True
            return metadata

        return metadata
