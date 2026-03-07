# ---------------------------------------------------------------------------
# StageSubmodule wrappers
# ---------------------------------------------------------------------------


import torch
import torch.nn as nn

from mminf.communication.tensors import NameToTensorList
from mminf.engine.ar_engine import CacheHandle
from mminf.model.bagel.components.language_model import BagelForCausalLM
from mminf.model.bagel.components.modeling_utils import PositionEmbedding, TimestepEmbedder, get_flattened_position_ids_extrapolate
from mminf.model.bagel.config import BagelModelConfig
from mminf.model.base import StageSubmodule


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
        return {"img_emb": [features]}


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
        return {"img_emb": [packed_latent]}


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
        language_model: BagelForCausalLM,
        llm2vae: nn.Linear,
        time_embedder: TimestepEmbedder,
        latent_pos_embed: PositionEmbedding,
        config: BagelModelConfig,
        boi_token_id: int | None = None,
        eoi_token_id: int | None = None,
    ):
        super().__init__()
        self.language_model = language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.llm2vae = llm2vae
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.config = config
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
    
    def _init_latents(
        self,
        device,
        H: int=1024,
        W: int=1024,
    ):
        h, w = (H // self.config.latent_downsample, 
                W // self.config.latent_downsample)
        num_image_tokens = h * w
        return torch.randn(
            num_image_tokens,
            self.config.vae_config.z_channels * self.config.latent_patch_size ** 2,
            device=device
        )

    def preprocess(self, phase: str, **inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Unwrap single-element tensor lists and handle latent initialization.

        For image_gen phase: if "latents" input is empty (first flow matching
        iteration), initializes random noise. The latent shape (latent_seq_len,
        latent_dim) must be provided via per-request metadata since it depends
        on the output image dimensions.

        For all other phases: standard unwrapping of list[Tensor] -> Tensor.
        """
        result = {}
        if phase in ["prefill_text", "decode"]:
            result["text_inputs"] = inputs["text_inputs"][0]
            result["empty_position_ids"] = None
            result["empty_position_ids"] = torch.zeros_like(
                result["text_inputs"], dtype=torch.long,
                device=result["text_inputs"].device
            )

        if phase in ["prefill_vit", "prefill_vae"]:
            img_emb = inputs.get("img_emb")
            result["combined_emb"] = self._wrap_with_boi_eoi(img_emb)
            result["empty_position_ids"] = None
            result["empty_position_ids"] = torch.zeros(
                result["combined_emb"].shape[0], dtype=torch.long,
                device=result["combined_emb"].device
            )
        
        if phase == "prefill_vae":
            result["text_indexes"] = torch.tensor(
                [0, result["combined_emb"].shape[0] - 1], dtype=torch.long,
                device=result["combined_emb"].device
            )
            result["vae_token_indexes"] = torch.arange(
                1, result["combined_emb"].shape[0]-1,
                dtype=torch.long, device=result["combined_emb"].device
            )
        
        if phase == "image_gen":
            H, W = 1024, 1024 # TODO: make this configurable?
            result["vae_position_ids"] = get_flattened_position_ids_extrapolate(
                H, W,
                self.config.latent_downsample, 
                max_num_patches_per_side=self.config.max_latent_size
            )
            if "latents" not in inputs or len(inputs["latents"]) == 0:
                result["latents"] = self._init_latents(
                    device=next(self.parameters()).device,
                    H=H, W=W
                )
                result["time_index"] = 0
            else:
               result["latents"] = inputs["latents"][0]
               result["time_index"] = inputs["time_index"][0].item()
               
            result["empty_combined_emb"] = self._wrap_with_boi_eoi_inplace(
                torch.zeros(
                    (result["latents"].shape[0], self.config.hidden_size),
                    device=result["latents"].device
                )
            )
            result["text_indexes"] = torch.tensor(
                [0, result["empty_combined_emb"].shape[0] - 1], dtype=torch.long,
                device=result["latents"].device
            )
            result["vae_token_indexes"] = torch.arange(
                1, result["empty_combined_emb"].shape[0]-1,
                dtype=torch.long,
                device=result["latents"].device
            )
            result["empty_position_ids"] = torch.zeros(
                result["empty_combined_emb"].shape[0], dtype=torch.long,
                device=result["latents"].device
            )
         
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

    def _forward_prefill_text(
        self, text_inputs: torch.Tensor,
        empty_position_ids: torch.Tensor,
        cache_handle: CacheHandle, **kwargs
    ) -> NameToTensorList:
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
                empty_position_ids = range(
                    cache_handle._get_state().seq_len,
                    cache_handle._get_state().seq_len + text_inputs.shape[0]
                )
            self.language_model(
                emb, empty_position_ids,
                is_causal=True, mode="und",
                cache_handle=cache_handle, **kwargs
            )
        if snapshot_after and cache_handle is not None:
            from_label, to_label = snapshot_after
            cache_handle.snapshot(from_label, to_label)
        return {}

    def _forward_prefill_vit(
        self, combined_emb: torch.Tensor,
        empty_position_ids: torch.Tensor,
        cache_handle: CacheHandle,**kwargs
    ) -> NameToTensorList:
        """Wrap img_emb with BOI/EOI tokens -> LLM forward (bidirectional).

        For generation mode, cache_labels specifies which caches to update.
        snapshot_after triggers a KV cache deepcopy after processing.
        """
        cache_labels = kwargs.pop("cache_labels", ["main"])
        snapshot_after = kwargs.pop("snapshot_after", None)
        for label in cache_labels:
            if cache_handle is not None:
                cache_handle.set_active_label(label)
                empty_position_ids = range(
                    cache_handle._get_state().seq_len,
                    cache_handle._get_state().seq_len + combined_emb.shape[0]
                )
            self.language_model(
                combined_emb, empty_position_ids,
                is_causal=False, mode="und",
                cache_handle=cache_handle, **kwargs
            )
        if snapshot_after and cache_handle is not None:
            from_label, to_label = snapshot_after
            cache_handle.snapshot(from_label, to_label)
        return {}
    
    def _forward_prefill_vae(
        self, combined_emb: torch.Tensor,
        empty_position_ids: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_indexes: torch.Tensor,
        cache_handle: CacheHandle, **kwargs
    ) -> NameToTensorList:
        """Wrap img_emb with BOI/EOI tokens -> LLM forward (bidirectional).

        For generation mode, cache_labels specifies which caches to update.
        snapshot_after triggers a KV cache deepcopy after processing.
        """
        cache_labels = kwargs.pop("cache_labels", ["main"])
        snapshot_after = kwargs.pop("snapshot_after", None)
        for label in cache_labels:
            if cache_handle is not None:
                cache_handle.set_active_label(label)
            self.language_model(
                combined_emb, empty_position_ids,
                is_causal=False, mode="und",
                cache_handle=cache_handle,
                vae_token_indexes=vae_token_indexes,
                text_indexes=text_indexes,
                **kwargs
            )
        if snapshot_after and cache_handle is not None:
            from_label, to_label = snapshot_after
            cache_handle.snapshot(from_label, to_label)
        return {}

    def _forward_decode(
        self, text_inputs: torch.Tensor,
        empty_position_ids: torch.Tensor,
        cache_handle: CacheHandle, **kwargs
    ) -> NameToTensorList:
        """embed_tokens -> LLM forward -> lm_head -> argmax."""
        # Remove metadata keys not needed for language_model
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        emb = self.embed_tokens(text_inputs)
        if cache_handle is not None:
            cache_handle.set_active_label("main")
            empty_position_ids = range(
                cache_handle._get_state().seq_len,
                cache_handle._get_state().seq_len + text_inputs.shape[0]
            )

        hidden = self.language_model(
            emb, empty_position_ids,
            is_causal=True, mode="und",
            cache_handle=cache_handle, **kwargs
        )
        logits = self.lm_head(hidden[-1:])
        token = torch.argmax(logits, dim=-1)
        return {"new_token": [token]}

    def _forward_image_gen(
        self,
        latents: torch.Tensor,
        empty_combined_emb: torch.Tensor,
        empty_position_ids: torch.Tensor,
        vae_position_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        time_index: int,
        cache_handle: CacheHandle,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_renorm_min: float=0.0,
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

        # TODO: support for timestep shift (non-uniform timesteps)
        dt = 1 / self.config.num_timesteps
        timestep = 1 - time_index * dt

        pos_embed = self.latent_pos_embed(vae_position_ids)
        timestep_embeds = self.time_embedder(timestep)
        empty_combined_emb[1:-1] = self.llm2vae(latents) + timestep_embeds \
            + pos_embed

        velocities = {}
        for label in ["main", "cfg_text", "cfg_img"]:
            if cache_handle is not None:
                cache_handle.set_active_label(label)
                empty_position_ids = range(
                    cache_handle._get_state().seq_len,
                    cache_handle._get_state().seq_len + empty_combined_emb.shape[0]
                )
            hidden = self.language_model(
                empty_combined_emb, empty_position_ids,
                is_causal=False, mode="gen",
                cache_handle=cache_handle,
                write_cache=False, 
                vae_token_indexes=vae_token_indexes,
                text_indexes=text_indexes,
                **kwargs,
            )
            velocities[label] = hidden

        # Project to VAE space
        v_main = self.llm2vae(velocities["main"])[1:-1]
        v_cfg_text = self.llm2vae(velocities["cfg_text"])[1:-1]
        v_cfg_img = self.llm2vae(velocities["cfg_img"])[1:-1]

        # CFG velocity combination
        # TODO: non-CFG case
        v_t_ = v_cfg_img + cfg_img_scale * (
            v_cfg_text + cfg_text_scale * (v_main - v_cfg_text) - v_cfg_img
        )
        norm_v_t_ = torch.norm(v_t_)
        norm_v_t = torch.norm(v_main)
        scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        v_final = v_t_ * scale

        # Euler step: x_{t+1} = x_t + v * dt
        latents = latents + v_final * dt
        return {
            "latents": [latents],
            "time_index": [torch.tensor([time_index + 1])]
        }

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

    def _wrap_with_boi_eoi_inplace(self, emb: torch.Tensor) -> torch.Tensor:
        """Wrap embeddings with <|vision_start|> and <|vision_end|> tokens."""
        if self.boi_token_id is None or self.eoi_token_id is None:
            return emb
        device = emb.device
        boi_ids = torch.tensor([self.boi_token_id], device=device)
        eoi_ids = torch.tensor([self.eoi_token_id], device=device)
        boi_emb = self.embed_tokens(boi_ids)
        eoi_emb = self.embed_tokens(eoi_ids)
        emb[0, :] = boi_emb
        emb[-1, :] = eoi_emb
        return emb


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