"""Text-to-image pipeline for Cosmos3-Nano.

Runs the generator in one fused forward per denoising step (text + vision
together), using mstar's DiT forward + packing and the imported diffusers UniPC
scheduler + Wan VAE. Intentionally simple (batch 1, sequential CFG); not the
served path. Produces the same image as the diffusers ``Cosmos3OmniPipeline`` on
a fixed seed/prompt.
"""

from __future__ import annotations

import torch

from mstar.model.cosmos3.packing import build_t2i_static_inputs, tokenize_t2i_prompt

# Transformer.forward static kwargs produced by build_t2i_static_inputs.
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


class Cosmos3T2IPipeline:
    """Text-to-image pipeline for Cosmos3-Nano."""

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

    @classmethod
    def from_model(cls, model, device, dtype=torch.bfloat16):
        """Build from a loaded ``Cosmos3Model`` (DiT + Wan VAE) + imported UniPC."""
        from diffusers import UniPCMultistepScheduler

        transformer = model.get_submodule("dit", device=device).transformer
        vae = model._build_vae(device)
        scheduler = UniPCMultistepScheduler.from_pretrained(str(model._ensure_repo() / "scheduler"))
        return cls(transformer, vae, scheduler, model.tokenizer, model.config, device, dtype)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents [1,C,T,H,W] -> pixels [1,3,T,H,W] in [0,1] (un-normalize + Wan VAE)."""
        mean = self._latents_mean.view(1, -1, 1, 1, 1)
        inv_std = self._latents_inv_std.view(1, -1, 1, 1, 1)
        z = latents.to(self.vae.dtype) / inv_std + mean
        decoded = self.vae.decode(z).sample  # [1,3,T,H,W] in [-1,1]
        return (decoded / 2 + 0.5).clamp(0, 1).to(torch.float32)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
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
        cond_ids, uncond_ids = tokenize_t2i_prompt(self.tokenizer, prompt, negative_prompt, height, width)

        lat_h = height // self.vae_scale_spatial
        lat_w = width // self.vae_scale_spatial
        shape = (1, self.config.latent_channel, 1, lat_h, lat_w)  # t2i: T_lat = 1
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        cond = build_t2i_static_inputs(cond_ids, shape, self.config, self.vae_scale_temporal, fps, device)
        uncond = build_t2i_static_inputs(uncond_ids, shape, self.config, self.vae_scale_temporal, fps, device)
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
