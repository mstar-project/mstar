"""Ming-flash-omni-2.0 imagegen diffusion pipeline (step 9b).

Native mstar port of vllm-omni's ``pipeline_ming_imagegen.py`` +
``z_image/pipeline_z_image.py`` denoise loop. The upstream pipeline subclasses
``ZImagePipeline`` (diffusers-/vllm_omni-coupled) and reads cross-stage tensors
off a global forward-context. This port:

  * keeps the **denoise loop pure** (``MingImageDenoiser.denoise``) — it takes
    the DiT, scheduler, latents and prompt embeds as plain arguments, so the
    flow-matching + classifier-free-guidance math is unit-testable with stubs
    and has no diffusers dependency;
  * pushes diffusers/transformers loading behind
    :meth:`MingImagePipeline.from_checkpoint` (lazy import) so the module
    imports cleanly even where diffusers is unavailable.

Flow-matching denoise (Z-Image convention):
  - latents start as Gaussian noise; timesteps come from
    FlowMatchEulerDiscreteScheduler with dynamic shifting (``mu`` from
    :func:`calculate_shift`);
  - per step the DiT predicts velocity; CFG combines pos/neg; the prediction is
    negated before ``scheduler.step`` (Z-Image sign convention);
  - final latents are un-shifted/un-scaled and VAE-decoded to ``[-1, 1]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Dynamic-shift ``mu`` for FlowMatchEulerDiscreteScheduler (Z-Image)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


@dataclass
class MingImageGenSamplingParams:
    """Resolved sampling knobs for one image-gen request."""

    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 2.0
    seed: int | None = None
    cfg_truncation: float = 1.0
    cfg_normalization: float = 0.0


def combine_cfg(
    pos: torch.Tensor,
    neg: torch.Tensor,
    guidance_scale: float,
    cfg_normalization: float = 0.0,
) -> torch.Tensor:
    """Classifier-free-guidance combination with optional renormalization.

    ``pred = pos + scale * (pos - neg)``; when ``cfg_normalization > 0`` the
    result is rescaled so its norm does not exceed ``cfg_normalization`` × the
    positive prediction's norm (Z-Image's renorm trick). Operates in fp32.
    """
    pos = pos.float()
    neg = neg.float()
    pred = pos + guidance_scale * (pos - neg)
    if cfg_normalization and float(cfg_normalization) > 0.0:
        ori = torch.linalg.vector_norm(pos)
        new = torch.linalg.vector_norm(pred)
        max_new = ori * float(cfg_normalization)
        scale = torch.where(
            new > max_new,
            (max_new / new.clamp(min=1e-12)).to(pred.dtype),
            pred.new_tensor(1.0),
        )
        pred = pred * scale
    return pred


class MingImageDenoiser:
    """Pure flow-matching + CFG denoise loop (no diffusers coupling).

    Holds references to the DiT transformer and a diffusers-style scheduler
    (anything exposing ``.step(model_output, t, sample) -> (prev_sample, ...)``
    and ``.timesteps``). The loop math mirrors ZImagePipeline.forward steps 6.
    """

    def __init__(self, transformer, scheduler, dtype: torch.dtype = torch.float32) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.dtype = dtype

    def denoise(
        self,
        latents: torch.Tensor,
        timesteps,
        prompt_embeds: list[torch.Tensor],
        negative_prompt_embeds: list[torch.Tensor] | None,
        guidance_scale: float,
        cfg_truncation: float = 1.0,
        cfg_normalization: float = 0.0,
    ) -> torch.Tensor:
        """Run the denoising loop and return the final ``[B, C, H, W]`` latents.

        Args:
            latents: initial noise ``[B, C, H, W]`` (fp32).
            timesteps: iterable of scheduler timesteps (1-D tensor).
            prompt_embeds / negative_prompt_embeds: list[Tensor] one per item.
            guidance_scale: CFG scale; ``> 0`` enables CFG (needs negatives).
            cfg_truncation: disable CFG once normalized time exceeds this.
            cfg_normalization: optional CFG renorm factor (0 = off).
        """
        actual_batch = latents.shape[0]
        do_cfg = guidance_scale > 0 and negative_prompt_embeds is not None

        ts = timesteps if isinstance(timesteps, torch.Tensor) else torch.as_tensor(timesteps)
        norm_ts = ((1000 - ts.float()) / 1000).tolist()

        for i, t in enumerate(timesteps):
            if isinstance(t, torch.Tensor):
                timestep = t.expand(latents.shape[0])
            else:
                timestep = torch.tensor([t] * latents.shape[0])
            timestep = (1000 - timestep) / 1000
            t_norm = norm_ts[i]

            current_scale = guidance_scale
            if do_cfg and cfg_truncation is not None and float(cfg_truncation) <= 1 and t_norm > cfg_truncation:
                current_scale = 0.0
            apply_cfg = do_cfg and current_scale > 0

            latents_typed = latents.to(self.dtype)
            if apply_cfg:
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                embeds_input = prompt_embeds + negative_prompt_embeds
                timestep_input = timestep.repeat(2)
            else:
                latent_model_input = latents_typed
                embeds_input = prompt_embeds
                timestep_input = timestep

            # DiT expects a list of [C, F, H, W] (frame axis inserted at dim 2).
            latent_model_input = latent_model_input.unsqueeze(2)
            model_out = self.transformer(
                list(latent_model_input.unbind(dim=0)),
                timestep_input,
                embeds_input,
            )[0]

            if apply_cfg:
                pos_out = model_out[:actual_batch]
                neg_out = model_out[actual_batch:]
                noise_pred = torch.stack(
                    [
                        combine_cfg(pos_out[j], neg_out[j], current_scale, cfg_normalization)
                        for j in range(actual_batch)
                    ],
                    dim=0,
                )
            else:
                noise_pred = torch.stack([o.float() for o in model_out], dim=0)

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred  # Z-Image sign convention

            latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

        return latents


class MingImagePipeline:
    """Text-to-image / img2img pipeline for Ming-flash-omni-2.0.

    Construct via :meth:`from_checkpoint` (loads VAE / scheduler / DiT /
    condition encoder / optional ByT5 — diffusers + transformers required) or
    inject components directly (used by tests). The conditioning path is Ming's
    own (Qwen2 connector), so there is no Z-Image text encoder / tokenizer.
    """

    def __init__(
        self,
        *,
        transformer,
        scheduler,
        vae,
        condition_encoder,
        image_gen_config,
        byte5=None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.condition_encoder = condition_encoder
        self.image_gen_config = image_gen_config
        self.byte5 = byte5
        self.device = torch.device(device)
        self.dtype = dtype
        self.denoiser = MingImageDenoiser(transformer, scheduler, dtype=dtype)
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8

    @classmethod
    def from_checkpoint(cls, model_path, image_gen_config, *, device="cuda", dtype=torch.bfloat16):
        """Load all components from the checkpoint (lazy diffusers import).

        Kept separate from ``__init__`` so the module imports without diffusers;
        only this path needs it.
        """
        from pathlib import Path

        from diffusers import AutoencoderKL
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        from mstar.model.ming_omni_flash.components.byte5_encoder import MingByT5Encoder
        from mstar.model.ming_omni_flash.components.condition_encoder import MingConditionEncoder
        from mstar.model.ming_omni_flash.components.zimage_transformer import MingZImageTransformer2DModel

        model_path = Path(model_path)
        cfg = image_gen_config

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder=cfg.scheduler_subfolder, local_files_only=True
        )
        scheduler.config["use_dynamic_shifting"] = True

        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder=cfg.vae_subfolder, local_files_only=True, torch_dtype=dtype
        ).to(device).eval()

        transformer = MingZImageTransformer2DModel(
            all_patch_size=tuple(cfg.dit.all_patch_size),
            all_f_patch_size=tuple(cfg.dit.all_f_patch_size),
            dim=cfg.dit.dim,
            n_layers=cfg.dit.n_layers,
            n_refiner_layers=cfg.dit.n_refiner_layers,
            n_heads=cfg.dit.n_heads,
            n_kv_heads=cfg.dit.n_kv_heads,
            in_channels=cfg.dit.in_channels,
            norm_eps=cfg.dit.norm_eps,
            rope_theta=cfg.dit.rope_theta,
            t_scale=cfg.dit.t_scale,
            axes_dims=tuple(cfg.dit.axes_dims),
            axes_lens=tuple(cfg.dit.axes_lens),
            cap_feat_dim=cfg.diffusion_c_input_dim,
        ).to(device, dtype=dtype).eval()

        condition_encoder = MingConditionEncoder(
            cfg, thinker_hidden_size=4096, device=device, dtype=dtype
        )
        condition_encoder.load_from_checkpoint(model_path)

        byte5_dir = model_path / "byt5"
        byte5 = None
        if (byte5_dir / "byt5.json").exists():
            byte5 = MingByT5Encoder.from_checkpoint(byte5_dir, device=torch.device(device), dtype=dtype)

        return cls(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            condition_encoder=condition_encoder,
            image_gen_config=cfg,
            byte5=byte5,
            device=device,
            dtype=dtype,
        )

    def prepare_latents(self, batch_size, height, width, generator=None) -> torch.Tensor:
        """Gaussian init latents ``[B, C, H/vae, W/vae]`` (fp32)."""
        c = self.transformer.in_channels
        vae_scale = self.vae_scale_factor * 2
        shape = (batch_size, c, height // vae_scale, width // vae_scale)
        return torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)

    def build_cap_feats(
        self,
        thinker_hidden_states: torch.Tensor,
        negative_hidden: torch.Tensor | None = None,
        byte5_texts: list[str] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run the condition encoder (+ optional ByT5) → (pos, neg) embed lists.

        Negatives default to zeros (Ming's CFG convention) unless explicit
        negative thinker states are supplied. ByT5 glyph features are appended
        along the sequence dim; the negative side gets zeros for that span so
        CFG doesn't push away from rendered text.
        """
        if thinker_hidden_states.dim() == 2:
            thinker_hidden_states = thinker_hidden_states.unsqueeze(0)
        cap_feats = self.condition_encoder(thinker_hidden_states)

        negative_cap_feats = None
        if negative_hidden is not None:
            if negative_hidden.dim() == 2:
                negative_hidden = negative_hidden.unsqueeze(0)
            negative_cap_feats = self.condition_encoder(negative_hidden)

        if byte5_texts and self.byte5 is not None:
            byte5_feats = self.byte5(byte5_texts).to(cap_feats.dtype)
            cap_feats = torch.cat((cap_feats, byte5_feats), dim=1)
            if negative_cap_feats is not None:
                negative_cap_feats = torch.cat((negative_cap_feats, torch.zeros_like(byte5_feats)), dim=1)

        prompt_embeds = [cap_feats[i] for i in range(cap_feats.shape[0])]
        if negative_cap_feats is not None:
            negative_prompt_embeds = [negative_cap_feats[i] for i in range(negative_cap_feats.shape[0])]
        else:
            negative_prompt_embeds = [self.condition_encoder.zero_negative(e) for e in prompt_embeds]
        return prompt_embeds, negative_prompt_embeds

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Un-shift/un-scale then VAE-decode to a ``[B, 3, H, W]`` image in [-1,1]."""
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(latents, return_dict=False)[0]

    @torch.inference_mode()
    def generate(
        self,
        thinker_hidden_states: torch.Tensor,
        params: MingImageGenSamplingParams,
        *,
        negative_hidden: torch.Tensor | None = None,
        byte5_texts: list[str] | None = None,
    ) -> torch.Tensor:
        """End-to-end text-to-image: condition → denoise → VAE decode."""
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(params.seed))

        prompt_embeds, negative_prompt_embeds = self.build_cap_feats(
            thinker_hidden_states, negative_hidden, byte5_texts
        )
        latents = self.prepare_latents(len(prompt_embeds), params.height, params.width, generator)

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        self.scheduler.set_timesteps(params.num_inference_steps, device=self.device, mu=mu)

        latents = self.denoiser.denoise(
            latents,
            self.scheduler.timesteps,
            prompt_embeds,
            negative_prompt_embeds,
            guidance_scale=params.guidance_scale,
            cfg_truncation=params.cfg_truncation,
            cfg_normalization=params.cfg_normalization,
        )
        return self.decode_latents(latents)


__all__ = [
    "MingImageDenoiser",
    "MingImageGenSamplingParams",
    "MingImagePipeline",
    "calculate_shift",
    "combine_cfg",
]
