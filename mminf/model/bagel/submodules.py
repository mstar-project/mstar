# ---------------------------------------------------------------------------
# NodeSubmodule wrappers
# ---------------------------------------------------------------------------


import logging

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.engine.ar_engine import CacheHandle
from mminf.model.bagel.components.language_model import BagelForCausalLM
from mminf.model.bagel.components.modeling_utils import (
    ImageTransform,
    PositionEmbedding,
    TimestepEmbedder,
    get_flattened_position_ids_extrapolate,
    patchify,
)
from mminf.model.bagel.config import BagelModelConfig
from mminf.model.base import NodeSubmodule

logger = logging.getLogger(__name__)


class ViTEncoderSubmodule(NodeSubmodule):
    """SigLIP2 ViT + connector + vit_pos_embed: pixel patches -> ViT features.

    Receives preprocessed inputs containing packed pixel values, position IDs,
    cumulative sequence lengths, and max sequence length. Both vit_encoder and
    vae_encoder receive "image_inputs" as their graph input name; routing is
    handled by the graph edge's next_node field.
    """

    def __init__(
        self,
        vit_model: nn.Module,
        connector: nn.Module,
        vit_pos_embed: nn.Module,
        vit_patch_size: int,
        vit_max_num_patch_per_side: int,
    ):
        super().__init__()
        self.vit_model = vit_model
        self.connector = connector
        self.vit_pos_embed = vit_pos_embed

        self.vit_patch_size = vit_patch_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.transform = ImageTransform(980, 224, 14)
        self.vae_transform = ImageTransform(1024, 512, 16)

    def preprocess(self, graph_walk: str, image_inputs: list[torch.Tensor]) -> dict:
        """Convert raw images to packed ViT input format.

        Full implementation should include prepare_vit_images logic from BAGEL:
        - Dynamic resolution computation and SigLIP2 image preprocessing
        - Patch splitting and flattening
        - Position ID computation from image grid
        - Packing multiple images with cu_seqlens for FlashAttention
        """
        image_tensor = self.vae_transform.resize_transform(image_inputs[0])
        image_tensor = self.transform(image_tensor)

        num_tokens = image_tensor.shape[0]
        device = image_tensor.device

        position_ids = get_flattened_position_ids_extrapolate(
            image_tensor.size(1), image_tensor.size(2),
            self.vit_patch_size,
            max_num_patches_per_side=self.vit_max_num_patch_per_side
        ).to(device)
        pixel_values = patchify(image_tensor, self.vit_patch_size)

        # Compute cu_seqlens for FlashAttention
        vit_token_seqlens = torch.tensor(
            [num_tokens], dtype=torch.int32, device=device
        )
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(vit_token_seqlens, dim=0), (1, 0)
        ).to(torch.int32)
        max_seqlen = int(num_tokens)

        return {
            "packed_pixel_values": pixel_values.to(torch.bfloat16),
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
        logger.debug(
            "Running BAGEL ViT with packed_pixel_values shape=%s, packed_position_ids shape=%s",
            packed_pixel_values.shape, packed_position_ids.shape
        )
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


class VAEEncoderSubmodule(NodeSubmodule):
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
        max_latent_size: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.latent_downsample = latent_downsample
        self.max_latent_size = max_latent_size
        self.transform = ImageTransform(1024, 512, 16)

    def preprocess(self, graph_walk: str, image_inputs: list[torch.Tensor]) -> dict:
        """Convert raw images to VAE encoder input format.

        Computes patchified dimensions as Python ints for CUDA graph
        compatibility (no .item() calls in forward).

        Full implementation should include:
        - Image padding to be divisible by latent_downsample * latent_patch_size
        - VAE position ID computation from latent grid
        - Timestep preparation
        """
        image_tensor: torch.Tensor = self.transform(self.transform.resize_transform(image_inputs[0]))  # [C, H, W]
        device = image_tensor.device

        # Compute patchified dimensions as ints (CUDA graph compatible)
        p = self.latent_patch_size
        ds = self.latent_downsample
        _, img_h, img_w = image_tensor.shape
        h = (img_h // ds)
        w = (img_w // ds)

        packed_vae_position_ids = get_flattened_position_ids_extrapolate(
            img_h, img_w,
            self.latent_downsample,
            max_num_patches_per_side=self.max_latent_size
        )

        return {
            "padded_images": image_tensor.unsqueeze(0).to(torch.bfloat16),
            "packed_vae_position_ids": packed_vae_position_ids,
            "packed_timesteps": torch.tensor([0], device=device),
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
        logger.debug(
            ("Running BAGEL VAE enc with padded_images shape=%s, "
             "packed_vae_position_ids shape=%s, packed_timesteps shape=%s, "
             "h=%d, w=%d"),
            padded_images.shape, packed_vae_position_ids.shape,
            packed_timesteps.shape, h, w
        )

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


class LLMSubmodule(NodeSubmodule):
    """Fat LLM wrapper that dispatches based on graph walk.

    Absorbs text_emb, lm_head, and flow_proj into a single node to avoid
    unnecessary IPC overhead. Graph walk-based dispatch handles:

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

    Multi-cache orchestration is driven by the requires_cfg flag in
    per-request metadata. When True, graph walk methods manage 3 caches:
      - prefill_text: snapshot main->cfg_text, forward [main, cfg_img]
      - prefill_vit/vae: forward [main], snapshot main->cfg_text
      - decode: forward [main, cfg_img]
      - image_gen: 3-pass CFG with conditional skip and renormalization
    The CacheHandle (provided by AREngine) manages label switching, page
    allocation, and KV data copying.
    """

    def __init__(
        self,
        language_model: BagelForCausalLM,
        llm2vae: nn.Linear,
        vae2llm: nn.Linear,
        time_embedder: TimestepEmbedder,
        latent_pos_embed: PositionEmbedding,
        config: BagelModelConfig,
        boi_token_id: int | None = None,
        eoi_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ):
        super().__init__()
        self.language_model = language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.llm2vae = llm2vae
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.config = config
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def _init_latents(
        self,
        device,
        H: int=1024,
        W: int=1024,
    ):
        h, w = (H // self.config.latent_downsample,
                W // self.config.latent_downsample)
        num_image_tokens = h * w
        # torch.random.manual_seed(42)
        return torch.randn(
            num_image_tokens,
            self.config.vae_config.z_channels * self.config.latent_patch_size ** 2,
        ).to(device=device, dtype=torch.bfloat16)

    def preprocess(self, graph_walk: str, **inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Unwrap single-element tensor lists and handle latent initialization.

        For image_gen graph walk: if "latents" input is empty (first flow matching
        iteration), initializes random noise. The latent shape (latent_seq_len,
        latent_dim) must be provided via per-request metadata since it depends
        on the output image dimensions.

        For all other graph walks: standard unwrapping of list[Tensor] -> Tensor.
        """
        device = next(self.parameters()).device
        result = {}
        if graph_walk == "prefill_text":
            # Wrap with BOS EOS
            result["text_inputs"] = inputs["text_inputs"][0].new_zeros(inputs["text_inputs"][0].shape[0] + 2)
            result["text_inputs"][0] = self.bos_token_id
            result["text_inputs"][-1] = self.eos_token_id
            result["text_inputs"][1:-1] = inputs["text_inputs"][0]
        elif graph_walk == "decode" and len(inputs["text_inputs"]) > 0:
            result["text_inputs"] = inputs["text_inputs"][0]
        elif graph_walk == "decode":
            result["text_inputs"] = torch.tensor([self.bos_token_id], device=device)

        if graph_walk in ["prefill_vit", "prefill_vae"]:
            img_emb = inputs["img_emb"][0]
            result["combined_emb"] = self._wrap_with_boi_eoi(img_emb)
            result["empty_pos_ids"] = torch.zeros(
                result["combined_emb"].shape[0], dtype=torch.int32, device=device
            )

        if graph_walk == "prefill_vae":
            text_len = result["combined_emb"].shape[0]
            result["text_indexes"] = torch.zeros(
                text_len, dtype=torch.bool, device=device
            )
            result["text_indexes"][0] = True
            result["text_indexes"][text_len - 1] = True
            result["vae_token_indexes"] = torch.arange(
                1, result["combined_emb"].shape[0]-1,
                dtype=torch.long, device=device
            )

        if graph_walk == "image_gen":
            H, W = 1024, 1024 # TODO: make this configurable?
            result["vae_position_ids"] = get_flattened_position_ids_extrapolate(
                H, W,
                self.config.latent_downsample,
                max_num_patches_per_side=self.config.max_latent_size
            )
            if "latents" not in inputs or len(inputs["latents"]) == 0:
                result["latents"] = self._init_latents(
                    device=device,
                    H=H, W=W
                )
                result["time_index"] = torch.zeros(result["latents"].shape[0], device=device, dtype=torch.bfloat16)
            else:
               result["latents"] = inputs["latents"][0]
               result["time_index"] = inputs["time_index"][0]

            result["empty_combined_emb"] = self._wrap_with_boi_eoi(
                torch.empty(
                    (result["latents"].shape[0], self.config.hidden_size),
                    dtype=torch.bfloat16,
                    device=device
                )
            )
            text_len = result["empty_combined_emb"].shape[0]
            result["text_indexes"] = torch.zeros(
                text_len, dtype=torch.bool, device=device
            )
            result["text_indexes"][0] = True
            result["text_indexes"][text_len - 1] = True
            result["vae_token_indexes"] = torch.arange(
                1, result["empty_combined_emb"].shape[0]-1,
                dtype=torch.long, device=device
            )
            result["empty_pos_ids"] = torch.zeros(
                text_len, dtype=torch.int32, device=device
            )

        return result

    def forward(self, graph_walk: str, cache_handle=None, **kwargs) -> NameToTensorList:
        inputs_and_shapes = ", ".join([
            (f"{key} (shape={val.shape})" if isinstance(val, torch.Tensor) else key) \
                for key, val in kwargs.items()
        ])
        logger.debug("Running BAGEL LLM for graph walk %s and inputs %s", graph_walk, inputs_and_shapes)

        if graph_walk == "prefill_text":
            return self._forward_prefill_text(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "prefill_vit":
            return self._forward_prefill_vit(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "prefill_vae":
            return self._forward_prefill_vae(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "decode":
            return self._forward_decode(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "image_gen":
            return self._forward_image_gen(cache_handle=cache_handle, **kwargs)
        else:
            raise ValueError(f"Unknown LLM graph walk: {graph_walk!r}")

    def _forward_prefill_text(
        self, text_inputs: torch.Tensor,
        cache_handle: CacheHandle, **kwargs
    ) -> NameToTensorList:
        """embed_tokens -> LLM forward (causal, mode='und') -> KV cache update.

        When requires_cfg is True (image generation mode):
        1. Snapshot main -> cfg_text BEFORE forward (cfg_text = context without this text)
        2. Forward for main and cfg_img (both see the text tokens)
        """
        emb = self.embed_tokens(text_inputs)
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)

        if requires_cfg and cache_handle is not None:
            # cfg_text = main before this text (everything except subsequent text)
            cache_handle.snapshot("main", "cfg_text")
            for label in ["main", "cfg_img"]:
                cache_handle.set_active_label(label)
                self.language_model(
                    emb, is_causal=True, mode="und",
                    cache_handle=cache_handle, **kwargs
                )
        else:
            if cache_handle is not None:
                cache_handle.set_active_label("main")
            self.language_model(
                emb, is_causal=True, mode="und",
                cache_handle=cache_handle, **kwargs
            )
        return {}

    def _forward_prefill_vit(
        self, combined_emb: torch.Tensor,
        empty_pos_ids: torch.Tensor,
        cache_handle: CacheHandle, **kwargs
    ) -> NameToTensorList:
        """Wrap img_emb with BOI/EOI tokens -> LLM forward (bidirectional).

        When requires_cfg is True: forward for main only, then snapshot
        main -> cfg_text (cfg_text = context including this image).
        """
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)

        if cache_handle is not None:
            cache_handle.set_active_label("main")
            pos_ids = empty_pos_ids + cache_handle._get_state().position_id_start
        else:
            pos_ids = empty_pos_ids
        self.language_model(
            combined_emb, is_causal=False, mode="und",
            pos_ids=pos_ids,
            custom_advance_pos_id=1,
            cache_handle=cache_handle, **kwargs
        )

        if requires_cfg and cache_handle is not None:
            cache_handle.snapshot("main", "cfg_text")
        return {}

    def _forward_prefill_vae(
        self, combined_emb: torch.Tensor,
        empty_pos_ids: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_indexes: torch.Tensor,
        cache_handle: CacheHandle, **kwargs
    ) -> NameToTensorList:
        """VAE image emb -> LLM forward (bidirectional, gen mode).

        When requires_cfg is True: forward for main only, then snapshot
        main -> cfg_text (cfg_text = context including this image).
        """
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)

        if cache_handle is not None:
            cache_handle.set_active_label("main")
            empty_pos_ids = empty_pos_ids + cache_handle._get_state().position_id_start
        self.language_model(
            combined_emb, is_causal=False, mode="gen",
            cache_handle=cache_handle,
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
            pos_ids=empty_pos_ids,
            custom_advance_pos_id=1,
            **kwargs
        )

        if requires_cfg and cache_handle is not None:
            cache_handle.snapshot("main", "cfg_text")
        return {}

    def _forward_decode(
        self, text_inputs: torch.Tensor,
        cache_handle: CacheHandle, **kwargs
    ) -> NameToTensorList:
        """embed_tokens -> LLM forward -> lm_head -> argmax.

        When requires_cfg is True: also forward for cfg_img to keep its
        KV cache in sync (cfg_img tracks all text, no images).
        """
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)
        emb = self.embed_tokens(text_inputs)

        if cache_handle is not None:
            cache_handle.set_active_label("main")
        hidden = self.language_model(
            emb, is_causal=True, mode="und",
            cache_handle=cache_handle, **kwargs
        )

        if requires_cfg and cache_handle is not None:
            # Keep cfg_img KV cache in sync (discard output)
            cache_handle.set_active_label("cfg_img")
            self.language_model(
                emb, is_causal=True, mode="und",
                cache_handle=cache_handle, **kwargs
            )

        logits = self.lm_head(hidden[-1:])
        token = torch.argmax(logits, dim=-1)
        return {
            "new_token": [token],
        }

    @staticmethod
    def _apply_timestep_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
        """Apply BAGEL's non-linear timestep remapping.

        Maps uniform t in [0,1] to shifted t that spends more time
        at higher noise levels (shift > 1).  shift=1 is identity.
        """
        return shift * t / (1 + (shift - 1) * t)

    def _forward_image_gen(
        self,
        latents: torch.Tensor,
        empty_combined_emb: torch.Tensor,
        vae_position_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        time_index: torch.Tensor,
        empty_pos_ids: torch.Tensor,
        cache_handle: CacheHandle,
        requires_cfg: bool = True,
        **kwargs,
    ) -> NameToTensorList:
        """Flow matching Euler step with optional 3-pass CFG.

        Uses cache_handle to switch between the 3 frozen KV caches
        (main, cfg_text, cfg_img). write_cache=False since caches are
        frozen during flow matching.

        When requires_cfg is False, runs a single forward pass (main only)
        without CFG, saving 2/3 of the compute.

        Renormalization modes (cfg_renorm_type in config):
          - "global": single scalar renorm over all dimensions (default)
          - "channel": per-token renorm (independent scale per latent token)
        """
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)

        N = self.config.num_timesteps
        shift = self.config.timestep_shift

        # Compute shifted timestep and step size for this iteration.
        # time_index goes from 0 to N-2 (N-1 total Euler steps).
        t_uniform = 1.0 - time_index / (N - 1)
        t_uniform_next = 1.0 - (time_index + 1) / (N - 1)
        timestep = self._apply_timestep_shift(t=t_uniform, shift=shift)
        timestep_next = self._apply_timestep_shift(t=t_uniform_next, shift=shift)
        dt = (timestep - timestep_next)[0].item()  # positive step size

        pos_embed = self.latent_pos_embed(vae_position_ids)
        timestep_embeds = self.time_embedder(timestep)
        latents_ = self.vae2llm(latents) + timestep_embeds \
            + pos_embed

        empty_combined_emb[1:-1] = latents_
        logger.debug(f"packed_seq = {empty_combined_emb}")

        if requires_cfg:
            logger.debug("Running 3 fwd passes for cfg gen")
            cfg_text_scale = kwargs.pop("cfg_text_scale", self.config.cfg_text_scale)
            cfg_img_scale = kwargs.pop("cfg_img_scale", self.config.cfg_img_scale)
            renorm_type = kwargs.pop("cfg_renorm_type", self.config.cfg_renorm_type)

            # CFG interval: only apply guidance when timestep is within interval
            cfg_interval = kwargs.pop("cfg_interval", self.config.cfg_interval)
            cfg_lo, cfg_hi = cfg_interval
            t_val = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)
            in_cfg_interval = (t_val > cfg_lo) and (t_val <= cfg_hi)
            effective_text_scale = cfg_text_scale if in_cfg_interval else 1.0
            effective_img_scale = cfg_img_scale if in_cfg_interval else 1.0

            velocities = {}
            for label in ["main", "cfg_text", "cfg_img"]:
                if cache_handle is not None:
                    cache_handle.set_active_label(label)
                    pos_ids = empty_pos_ids + cache_handle._get_state().position_id_start
                else:
                    pos_ids = empty_pos_ids
                hidden = self.language_model(
                    empty_combined_emb, is_causal=False, mode="gen",
                    cache_handle=cache_handle, write_cache=False,
                    vae_token_indexes=vae_token_indexes,
                    pos_ids=pos_ids,
                    text_indexes=text_indexes, **kwargs,
                )
                velocities[label] = hidden

            v_main = self.llm2vae(velocities["main"])[1:-1]
            v_cfg_text = self.llm2vae(velocities["cfg_text"])[1:-1]
            v_cfg_img = self.llm2vae(velocities["cfg_img"])[1:-1]

            # Two-node CFG velocity combination + renormalization
            cfg_renorm_min = kwargs.pop("cfg_renorm_min", self.config.cfg_renorm_min)

            if renorm_type == "text_channel":
                # text_channel: renorm AFTER text CFG, THEN apply image CFG
                v_text_guided = v_cfg_text + effective_text_scale * (v_main - v_cfg_text)
                norm_v = torch.norm(v_main, dim=-1, keepdim=True)
                norm_v_text = torch.norm(v_text_guided, dim=-1, keepdim=True)
                scale = (norm_v / (norm_v_text + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_text_renormed = v_text_guided * scale
                if effective_img_scale > 1.0:
                    v_final = v_cfg_img + effective_img_scale * (v_text_renormed - v_cfg_img)
                else:
                    v_final = v_text_renormed
            else:
                # global / channel: apply both text+image CFG, THEN renorm
                v_text_guided = v_cfg_text + effective_text_scale * (v_main - v_cfg_text)
                if effective_img_scale > 1.0:
                    v_combined = v_cfg_img + effective_img_scale * (v_text_guided - v_cfg_img)
                else:
                    v_combined = v_text_guided

                if renorm_type == "channel":
                    renorm_scale = (
                        v_main.norm(dim=-1, keepdim=True) /
                        (v_combined.norm(dim=-1, keepdim=True) + 1e-8)
                    ).clamp(min=cfg_renorm_min, max=1.0)
                else:
                    renorm_scale = (
                        v_main.norm() / (v_combined.norm() + 1e-8)
                    ).clamp(min=cfg_renorm_min, max=1.0)
                v_final = v_combined * renorm_scale
        else:
            # No CFG: single forward pass
            if cache_handle is not None:
                cache_handle.set_active_label("main")
                pos_ids = empty_pos_ids + cache_handle._get_state().position_id_start
            else:
                pos_ids = empty_pos_ids
            hidden = self.language_model(
                empty_combined_emb, is_causal=False, mode="gen",
                cache_handle=cache_handle, write_cache=False,
                vae_token_indexes=vae_token_indexes,
                pos_ids=pos_ids,
                text_indexes=text_indexes, **kwargs,
            )
            v_final = self.llm2vae(hidden)[1:-1]

        # Euler step: x_{t-dt} = x_t - v * dt  (velocity points data -> noise)
        latents = latents - v_final * dt
        return {
            "latents": [latents],
            "time_index": [time_index + 1]
        }

    def _wrap_with_boi_eoi(self, emb: torch.Tensor) -> torch.Tensor:
        """Wrap embeddings with <|vision_start|> and <|vision_end|> tokens."""
        assert self.boi_token_id is not None and self.eoi_token_id is not None

        device = emb.device
        boi_ids = torch.tensor([self.boi_token_id], device=device)
        eoi_ids = torch.tensor([self.eoi_token_id], device=device)
        with torch.no_grad():
            boi_emb = self.embed_tokens(boi_ids).to(emb.dtype)
            eoi_emb = self.embed_tokens(eoi_ids).to(emb.dtype)
        return torch.cat([boi_emb, emb, eoi_emb], dim=0)

    def _wrap_with_boi_eoi_inplace(self, emb: torch.Tensor) -> torch.Tensor:
        """Wrap embeddings with <|vision_start|> and <|vision_end|> tokens."""
        assert self.boi_token_id is not None and self.eoi_token_id is not None
        device = emb.device
        boi_ids = torch.tensor([self.boi_token_id], device=device)
        eoi_ids = torch.tensor([self.eoi_token_id], device=device)
        with torch.no_grad():
            boi_emb = self.embed_tokens(boi_ids).to(emb.dtype)
            eoi_emb = self.embed_tokens(eoi_ids).to(emb.dtype)
        emb[0, :] = boi_emb
        emb[-1, :] = eoi_emb
        return emb


class VAEDecoderSubmodule(NodeSubmodule):
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

    def preprocess(self, graph_walk: str, latents: list[torch.Tensor]) -> dict:
        """Prepare VAE decoder inputs.

        Unwraps latents from list. Image dimensions (image_h, image_w)
        are provided via per-request metadata and converted to ints for
        CUDA graph compatibility.
        """
        return {"latents": latents[0]}

    def forward(
        self,
        latents: torch.Tensor,
        image_h: int | torch.Tensor = 1024, # BAGEL's default image dim
        image_w: int | torch.Tensor = 1024,
        **kwargs,
    ) -> NameToTensorList:
        logger.debug(
            "Running BAGEL VAE dec with latents shape %s, h %d, w %d",
            str(latents.shape), image_h, image_w
        )
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
