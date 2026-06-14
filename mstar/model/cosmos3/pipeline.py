"""Fused generation pipeline for Cosmos3-Nano (text/image-to-image/video).

Runs the generator in one fused forward per denoising step (text + vision
together), using mstar's DiT forward + packing and the imported diffusers UniPC
scheduler + Wan VAE. Intentionally simple (batch 1, sequential CFG); not the
served path. Produces the same image/video as the diffusers
``Cosmos3OmniPipeline`` on a fixed seed/prompt.

``num_frames == 1`` is text-to-image; ``num_frames > 1`` is text-to-video, and
passing ``image`` anchors frame 0 to a conditioning frame (image-to-video).
"""

from __future__ import annotations

import torch

from mstar.model.cosmos3.packing import build_static_inputs, tokenize_prompt

# Transformer.forward static kwargs produced by build_static_inputs.
_TF_STATIC_FIELDS = (
    "input_ids",
    "text_indexes",
    "position_ids",
    "und_len",
    "sequence_length",
    "vision_token_shapes",
    "vision_sequence_indexes",
    "vision_mse_loss_indexes",
    "vision_noisy_frame_indexes",
)


class Cosmos3Pipeline:
    """Fused t2i / t2v / i2v pipeline for Cosmos3-Nano."""

    def __init__(self, transformer, vae, scheduler, tokenizer, config, device, dtype=torch.bfloat16):
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.dtype = dtype

        self.vae_scale_spatial = int(vae.config.scale_factor_spatial)
        self.vae_scale_temporal = int(vae.config.scale_factor_temporal)
        self._latents_mean = torch.tensor(vae.config.latents_mean, dtype=vae.dtype, device=device)
        self._latents_inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=vae.dtype, device=device)

        # Conditioning-frame preprocessor (PIL / numpy / tensor -> [1,3,H,W] in
        # [-1,1], resized) — the same one the diffusers pipeline uses, for parity.
        from diffusers.video_processor import VideoProcessor

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_spatial, resample="bilinear")

    @classmethod
    def from_model(cls, model, device, dtype=torch.bfloat16):
        """Build from a loaded ``Cosmos3Model`` (DiT + Wan VAE) + imported UniPC."""
        from diffusers import UniPCMultistepScheduler

        transformer = model.get_submodule("dit", device=device).transformer
        vae = model._build_vae(device)
        scheduler = UniPCMultistepScheduler.from_pretrained(str(model._ensure_repo() / "scheduler"))
        return cls(transformer, vae, scheduler, model.tokenizer, model.config, device, dtype)

    def _encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """[1,3,T,H,W] in [-1,1] -> normalized latents [1,C,T_lat,H/16,W/16].

        Takes the distribution mode (``sample_mode="argmax"``) and applies the
        pipeline-side latent normalization, matching the diffusers oracle.
        """
        in_dtype = x.dtype
        dtype = self.vae.dtype
        mean = self._latents_mean.to(device=x.device, dtype=dtype).view(1, -1, 1, 1, 1)
        inv_std = self._latents_inv_std.to(device=x.device, dtype=dtype).view(1, -1, 1, 1, 1)
        raw_mu = self.vae.encode(x.to(dtype)).latent_dist.mode()
        return ((raw_mu - mean) * inv_std).to(in_dtype)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents [1,C,T,H,W] -> pixels [1,3,T,H,W] in [0,1] (un-normalize + Wan VAE)."""
        mean = self._latents_mean.view(1, -1, 1, 1, 1)
        inv_std = self._latents_inv_std.view(1, -1, 1, 1, 1)
        z = latents.to(self.vae.dtype) / inv_std + mean
        decoded = self.vae.decode(z).sample  # [1,3,T,H,W] in [-1,1]
        return (decoded / 2 + 0.5).clamp(0, 1).to(torch.float32)

    def _prepare_latents(self, image, num_frames, height, width, generator, latents, device, dtype):
        """Build the initial vision latents + whether frame 0 is a clean anchor.

        For image-to-video the conditioning frame anchors latent frame 0 (clean,
        VAE-encoded) and the remaining frames start from pure noise; otherwise the
        whole tensor is noise. Mirrors the diffusers ``prepare_latents`` vision path.
        """
        from diffusers.utils.torch_utils import randn_tensor

        is_image = num_frames == 1
        has_image_condition = image is not None and not is_image

        conditioning_frame_2d = None
        if image is not None:
            conditioning_frame_2d = self.video_processor.preprocess(image, height=height, width=width).to(
                device=device, dtype=dtype
            )

        if is_image:
            vision_tensor = (
                conditioning_frame_2d.unsqueeze(2)
                if conditioning_frame_2d is not None
                else torch.zeros(1, 3, 1, height, width, dtype=dtype, device=device)
            )
        else:
            vision_tensor = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            if conditioning_frame_2d is not None:
                vision_tensor[:, :, 0] = conditioning_frame_2d
                if num_frames > 1:
                    vision_tensor[:, :, 1:] = conditioning_frame_2d.unsqueeze(2).expand(
                        -1, -1, num_frames - 1, -1, -1
                    )

        x0 = self._encode_video(vision_tensor).contiguous().float()
        vision_shape = tuple(x0.shape)

        vision_condition_mask = torch.zeros((x0.shape[2], 1, 1), device=device, dtype=dtype)
        if has_image_condition:
            vision_condition_mask[0, 0, 0] = 1.0

        if latents is None:
            pure_noise = randn_tensor(vision_shape, generator=generator, device=device, dtype=dtype)
            latents = (
                vision_condition_mask * x0.to(device=device, dtype=dtype)
                + (1.0 - vision_condition_mask) * pure_noise
            )
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents, has_image_condition

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        image=None,
        num_frames: int = 1,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        fps: float = 24.0,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        decode: bool = True,
    ):
        device, dtype = self.device, self.dtype
        cond_ids, uncond_ids = tokenize_prompt(
            self.tokenizer, prompt, negative_prompt, num_frames=num_frames, height=height, width=width, fps=fps
        )

        latents, has_image_condition = self._prepare_latents(
            image, num_frames, height, width, generator, latents, device, dtype
        )
        latent_shape = tuple(latents.shape)

        cond = build_static_inputs(
            cond_ids, latent_shape, self.config, self.vae_scale_temporal, fps, device,
            has_image_condition=has_image_condition,
        )
        uncond = build_static_inputs(
            uncond_ids, latent_shape, self.config, self.vae_scale_temporal, fps, device,
            has_image_condition=has_image_condition,
        )
        cond_static = {k: cond[k] for k in _TF_STATIC_FIELDS}
        uncond_static = {k: uncond[k] for k in _TF_STATIC_FIELDS}
        num_noisy = cond["num_noisy_vision_tokens"]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.scheduler.timesteps:
            vision_tokens = [latents.to(dtype)]
            vision_timesteps = torch.full((num_noisy,), t.item(), device=device)
            cond_v = self.transformer(
                vision_tokens=vision_tokens, vision_timesteps=vision_timesteps, **cond_static
            )[0][0]
            if guidance_scale != 1.0:
                uncond_v = self.transformer(
                    vision_tokens=vision_tokens, vision_timesteps=vision_timesteps, **uncond_static
                )[0][0]
                velocity = uncond_v + guidance_scale * (cond_v - uncond_v)
            else:
                velocity = cond_v
            latents = self.scheduler.step(
                velocity.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
            )[0].squeeze(0)

        if not decode:
            return latents
        return self._decode(latents)
