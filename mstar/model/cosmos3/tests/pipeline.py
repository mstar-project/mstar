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

from mstar.model.cosmos3.components.packing import (
    action_start_frame_offset,
    build_action_static_inputs,
    build_static_inputs,
    tokenize_prompt,
    vision_condition_frame_indexes,
)

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

# Additional Transformer.forward static kwargs for joint video+action generation.
_TF_ACTION_STATIC_FIELDS = (
    "action_token_shapes",
    "action_sequence_indexes",
    "action_mse_loss_indexes",
    "action_noisy_frame_indexes",
)

# Additional Transformer.forward static kwargs for joint video+sound generation.
_TF_SOUND_STATIC_FIELDS = (
    "sound_token_shapes",
    "sound_sequence_indexes",
    "sound_mse_loss_indexes",
    "sound_noisy_frame_indexes",
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
        generate_sound: bool = False,
        sound_duration: float | None = None,
        condition_video: torch.Tensor | None = None,
        condition_frame_indexes: tuple[int, ...] = (0, 1),
    ):
        """With ``generate_sound`` a jointly denoised AVAE-latent sound band
        rides after the vision tokens (video-mode only); returns
        ``(video_or_latents, sound_latents)`` with the final sound latents
        ``[1, C, T]`` (decode them with the checkpoint's sound tokenizer).

        ``condition_video`` (``[T, C, H, W]`` in [0, 1]) enables video-to-video:
        the latent frames in ``condition_frame_indexes`` are pinned clean from
        the video's VAE-encoded causal prefix and re-injected after every
        scheduler step; the complement is denoised."""
        device, dtype = self.device, self.dtype
        cond_ids, uncond_ids = tokenize_prompt(
            self.tokenizer, prompt, negative_prompt, num_frames=num_frames, height=height, width=width, fps=fps
        )

        latents, has_image_condition = self._prepare_latents(
            image, num_frames, height, width, generator, latents, device, dtype
        )
        latent_shape = tuple(latents.shape)

        vmask = None
        cond_video_latents = None
        noisy_frames = None
        if condition_video is not None:
            # Video-to-video: only the causal prefix feeding the pinned latent
            # frames is encoded (max(indexes)*tcf + 1 pixel frames, padded by
            # repeating the last frame); the complement keeps the noise drawn
            # above (the serving path's RNG order).
            cpf = min(max(condition_frame_indexes) * self.vae_scale_temporal + 1, num_frames)
            clip = condition_video[:cpf]
            if clip.shape[0] < cpf:
                clip = torch.cat([clip, clip[-1:].expand(cpf - clip.shape[0], -1, -1, -1)], dim=0)
            frames = [
                self.video_processor.preprocess(clip[i], height=height, width=width).squeeze(0)
                for i in range(clip.shape[0])
            ]
            prefix = self._encode_video(
                torch.stack(frames, dim=1).unsqueeze(0).to(device=device, dtype=dtype)
            )
            vmask = torch.zeros((1, 1, latent_shape[2], 1, 1), device=device, dtype=dtype)
            cond_video_latents = torch.zeros(latent_shape, device=device, dtype=dtype)
            for f in condition_frame_indexes:
                vmask[:, :, f] = 1.0
                cond_video_latents[:, :, f] = prefix[:, :, f].to(dtype)
            latents = vmask * cond_video_latents + (1.0 - vmask) * latents
            noisy_frames = [f for f in range(latent_shape[2]) if f not in set(condition_frame_indexes)]

        # Sound noise is drawn after the video noise from the same generator
        # (the serving path's RNG order).
        sound_latents = None
        sound_frames = None
        if generate_sound:
            import math

            from diffusers.utils.torch_utils import randn_tensor

            duration = sound_duration if sound_duration is not None else num_frames / fps
            duration = max(float(duration), 1.0 / max(fps, 1.0))
            target = max(1, int(round(duration * self.config.sound_sample_rate)))
            hop = int(round(self.config.sound_sample_rate / self.config.sound_latent_fps))
            sound_frames = max(1, math.ceil(target / hop))
            sound_latents = randn_tensor(
                (1, self.config.sound_dim, sound_frames), generator=generator, device=device, dtype=dtype
            )

        cond = build_static_inputs(
            cond_ids, latent_shape, self.config, self.vae_scale_temporal, fps, device,
            has_image_condition=has_image_condition, sound_latent_frames=sound_frames,
            noisy_frames=noisy_frames,
        )
        uncond = build_static_inputs(
            uncond_ids, latent_shape, self.config, self.vae_scale_temporal, fps, device,
            has_image_condition=has_image_condition, sound_latent_frames=sound_frames,
            noisy_frames=noisy_frames,
        )
        fields = _TF_STATIC_FIELDS + (_TF_SOUND_STATIC_FIELDS if generate_sound else ())
        cond_static = {k: cond[k] for k in fields}
        uncond_static = {k: uncond[k] for k in fields}
        num_noisy = cond["num_noisy_vision_tokens"]

        def _forward(static, vision_tokens, vision_timesteps, t):
            kwargs = dict(vision_tokens=vision_tokens, vision_timesteps=vision_timesteps, **static)
            if generate_sound:
                kwargs["sound_tokens"] = [sound_latents[0].to(dtype)]
                kwargs["sound_timesteps"] = torch.full((sound_frames,), t.item(), device=device)
            preds_vision, preds_sound = self.transformer(**kwargs)
            return preds_vision[0], (preds_sound[0] if generate_sound else None)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.scheduler.timesteps:
            vision_tokens = [latents.to(dtype)]
            vision_timesteps = torch.full((num_noisy,), t.item(), device=device)
            cond_v, cond_s = _forward(cond_static, vision_tokens, vision_timesteps, t)
            if guidance_scale != 1.0:
                uncond_v, uncond_s = _forward(uncond_static, vision_tokens, vision_timesteps, t)
                velocity = uncond_v + guidance_scale * (cond_v - uncond_v)
                sound_v = (uncond_s + guidance_scale * (cond_s - uncond_s)) if generate_sound else None
            else:
                velocity = cond_v
                sound_v = cond_s
            if generate_sound:
                # Joint [video | sound] scheduler step (one multistep history).
                nv = velocity.numel()
                packed = torch.cat([velocity.reshape(1, -1), sound_v.unsqueeze(0).reshape(1, -1)], dim=1)
                packed_lat = torch.cat([latents.reshape(1, -1), sound_latents.reshape(1, -1)], dim=1)
                packed_next = self.scheduler.step(packed, t, packed_lat, return_dict=False)[0]
                latents = packed_next[:, :nv].reshape(latents.shape)
                sound_latents = packed_next[:, nv:].reshape(sound_latents.shape)
            else:
                latents = self.scheduler.step(
                    velocity.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
                )[0].squeeze(0)
            if vmask is not None:
                # Video-to-video: re-inject the clean conditioning frames after
                # the scheduler step, matching the serving path.
                latents = (1.0 - vmask) * latents + vmask * cond_video_latents

        out = latents if not decode else self._decode(latents)
        if generate_sound:
            return out, sound_latents
        return out

    @torch.no_grad()
    def generate_action(
        self,
        *,
        prompt: str,
        mode: str,
        domain_id: int,
        action_chunk_size: int,
        raw_action_dim: int,
        video: torch.Tensor | None = None,
        video_latents: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int = 256,
        width: int = 256,
        fps: float = 24.0,
        action_fps: float | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        flow_shift: float | None = None,
        negative_prompt: str = "",
        generator: torch.Generator | None = None,
        cond_ids: list[int] | None = None,
        uncond_ids: list[int] | None = None,
        return_video: bool = False,
    ):
        """Joint video+action generation (forward_dynamics / inverse_dynamics / policy).

        The conditioning video is VAE-encoded to clean anchor frames per ``mode``
        (all frames for inverse-dynamics; frame 0 for forward-dynamics / policy).
        Action tokens are clean conditioning for forward-dynamics, else noisy and
        predicted. Returns the predicted action ``[1, action_chunk_size,
        raw_action_dim]`` (and the decoded video when ``return_video``).
        """
        from diffusers import UniPCMultistepScheduler
        from diffusers.utils.torch_utils import randn_tensor

        device, dtype = self.device, self.dtype
        action_dim = self.transformer.action_dim
        if num_frames is None:
            num_frames = action_chunk_size + 1
        if action_fps is None:
            action_fps = fps
        action_offset = action_start_frame_offset(action_chunk_size, num_frames)

        if flow_shift is not None:
            scheduler = UniPCMultistepScheduler.from_config(self.scheduler.config, flow_shift=flow_shift)
        else:
            scheduler = UniPCMultistepScheduler.from_config(self.scheduler.config)
        scheduler.set_timesteps(num_inference_steps, device=device)

        if cond_ids is None or uncond_ids is None:
            cond_ids, uncond_ids = tokenize_prompt(
                self.tokenizer, prompt, negative_prompt, num_frames=num_frames,
                height=height, width=width, fps=fps,
            )

        # --- action latents (noise drawn before the video noise, matching the
        # reference ordering so a shared seed reproduces the same sample). ---
        if mode == "forward_dynamics":
            if action is None:
                raise ValueError("Cosmos3 forward_dynamics requires `action`.")
            act = action.to(device=device, dtype=torch.float32)
            if act.ndim == 3:
                act = act.squeeze(0)
            if act.shape[0] < action_chunk_size:
                act = torch.cat([act, act[-1:].repeat(action_chunk_size - act.shape[0], 1)], dim=0)
            elif act.shape[0] > action_chunk_size:
                act = act[:action_chunk_size]
            clean_action = torch.zeros((action_chunk_size, action_dim), dtype=torch.float32)
            clean_action[:, :raw_action_dim] = act[:, :raw_action_dim]
            clean_action = clean_action.to(device=device, dtype=dtype).unsqueeze(0)
            action_clean_mask = torch.ones((1, action_chunk_size, 1), device=device, dtype=dtype)
        else:
            clean_action = torch.zeros((1, action_chunk_size, action_dim), device=device, dtype=dtype)
            action_clean_mask = torch.zeros((1, action_chunk_size, 1), device=device, dtype=dtype)
        a_noise = randn_tensor((1, action_chunk_size, action_dim), generator=generator, device=device, dtype=dtype)
        a_noise[..., raw_action_dim:] = 0
        clean_action[..., raw_action_dim:] = 0
        action_latents = action_clean_mask * clean_action + (1.0 - action_clean_mask) * a_noise
        action_velocity_mask = 1.0 - action_clean_mask

        # --- conditioning video latents (clean per mode) ---
        if video_latents is None:
            if video is None:
                raise ValueError("Cosmos3 action generation requires `video` or `video_latents`.")
            video_latents = self._encode_video(video.to(device=device, dtype=dtype))
        cond_latent = video_latents.to(device=device, dtype=dtype)
        latent_shape = tuple(cond_latent.shape)
        t_lat = latent_shape[2]

        vis_clean = set(vision_condition_frame_indexes(mode, t_lat))
        vmask = torch.zeros((1, 1, t_lat, 1, 1), device=device, dtype=dtype)
        for f in vis_clean:
            vmask[:, :, f] = 1.0
        v_noise = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
        latents = vmask * cond_latent + (1.0 - vmask) * v_noise
        velocity_mask = 1.0 - vmask  # 1 where the video is predicted

        # --- static packing ---
        cond = build_action_static_inputs(
            cond_ids, latent_shape, action_chunk_size, mode, self.config,
            self.vae_scale_temporal, fps, action_fps, action_offset, device,
        )
        do_cfg = guidance_scale != 1.0
        keys = _TF_STATIC_FIELDS + _TF_ACTION_STATIC_FIELDS
        cond_static = {k: cond[k] for k in keys}
        uncond_static = None
        if do_cfg:
            uncond = build_action_static_inputs(
                uncond_ids, latent_shape, action_chunk_size, mode, self.config,
                self.vae_scale_temporal, fps, action_fps, action_offset, device,
            )
            uncond_static = {k: uncond[k] for k in keys}
        num_noisy_v = cond["num_noisy_vision_tokens"]
        num_noisy_a = cond["num_noisy_action_tokens"]
        domain_t = torch.tensor([domain_id], dtype=torch.long, device=device)

        for t in scheduler.timesteps:
            vts = torch.full((num_noisy_v,), t.item(), device=device)
            ats = torch.full((num_noisy_a,), t.item(), device=device)
            step_kwargs = dict(
                vision_tokens=[latents.to(dtype)], vision_timesteps=vts,
                action_tokens=action_latents.to(dtype), action_timesteps=ats, action_domain_id=domain_t,
            )
            v_c, a_c, _ = self.transformer(**cond_static, **step_kwargs)
            if do_cfg:
                v_u, a_u, _ = self.transformer(**uncond_static, **step_kwargs)
                video_v = v_u[0] + guidance_scale * (v_c[0] - v_u[0])
                action_v = a_u + guidance_scale * (a_c - a_u)
            else:
                video_v, action_v = v_c[0], a_c

            video_v = video_v * velocity_mask
            action_v = action_v * action_velocity_mask
            action_v[..., raw_action_dim:] = 0

            nv = video_v.numel()
            packed = torch.cat([video_v.reshape(1, -1), action_v.reshape(1, -1)], dim=1)
            packed_lat = torch.cat([latents.reshape(1, -1), action_latents.reshape(1, -1)], dim=1)
            packed_next = scheduler.step(packed, t, packed_lat, return_dict=False)[0]
            latents = packed_next[:, :nv].reshape(latent_shape)
            action_latents = packed_next[:, nv:].reshape(1, action_chunk_size, action_dim)

            latents = velocity_mask * latents + (1.0 - velocity_mask) * cond_latent
            action_latents = action_velocity_mask * action_latents + (1.0 - action_velocity_mask) * clean_action
            action_latents[..., raw_action_dim:] = 0

        action_out = action_latents[:, :, :raw_action_dim]
        if return_video:
            return action_out, self._decode(latents)
        return action_out

    # ------------------------------------------------------------------
    # Windowed block-causal reference (the kv-mode oracle): a hand-rolled
    # window loop with explicit per-layer context K/V, run directly against
    # the transformer's layers — no engine cache machinery involved.
    # ------------------------------------------------------------------

    def _ref_und_prefill(
        self, branch_ids: list[list[int]], branch_pos: list[torch.Tensor]
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Understanding tower over the guidance branches' text prompts,
        packed [cond | uncond] the way the served prefill runs — projections
        and MLPs over the packed rows, causal attention per branch — so the
        collected per-branch, per-layer rotated (k, v) is bit-identical to
        what the engine caches (a separate per-branch run differs by GEMM
        tiling rounding, which coarse few-step schedules amplify)."""
        tf = self.transformer
        lens = [len(ids) for ids in branch_ids]
        flat = [i for ids in branch_ids for i in ids]
        und_seq = tf.embed_tokens(
            torch.tensor(flat, dtype=torch.long, device=self.device)
        )
        pos = branch_pos[0] if len(branch_pos) == 1 else torch.cat(branch_pos, dim=1)
        cos, sin = tf._rotary(pos, und_seq.device, und_seq.dtype)
        kvs: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in branch_ids]
        for layer in tf.layers:
            attn = layer.self_attn
            h, hkv, d = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
            und_norm = layer.input_layernorm(und_seq)
            q = attn.norm_q(attn.to_q(und_norm).view(-1, h, d))
            k = attn.norm_k(attn.to_k(und_norm).view(-1, hkv, d))
            v = attn.to_v(und_norm).view(-1, hkv, d)
            q = attn._apply_rope(q, cos, sin)
            k = attn._apply_rope(k, cos, sin)
            outs, off = [], 0
            for bi, n in enumerate(lens):
                sl = slice(off, off + n)
                off += n
                kvs[bi].append((k[sl], v[sl]))
                outs.append(attn._attend(q[sl], k[sl], v[sl], is_causal=True))
            out = outs[0] if len(outs) == 1 else torch.cat(outs, 0)
            residual = und_seq + attn.to_out(out)
            und_seq = residual + layer.mlp(layer.post_attention_layernorm(residual))
        return kvs

    def _ref_gen_layers(self, gen_seq, cos, sin, ctx_kv, collect=False):
        """Generation layer stack with explicit context: each layer attends
        its fresh tokens over ``ctx_kv[i]`` (the branch's [text | retained
        frames] K/V) plus itself, non-causally. With ``collect`` the fresh
        rotated (k, v) per layer is returned — what a commit appends."""
        tf = self.transformer
        collected = []
        for i, layer in enumerate(tf.layers):
            attn = layer.self_attn
            h, hkv, d = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
            gen_norm = layer.input_layernorm_moe_gen(gen_seq)
            q = attn.norm_added_q(attn.add_q_proj(gen_norm).view(-1, h, d))
            k = attn.norm_added_k(attn.add_k_proj(gen_norm).view(-1, hkv, d))
            v = attn.add_v_proj(gen_norm).view(-1, hkv, d)
            q = attn._apply_rope(q, cos, sin)
            k = attn._apply_rope(k, cos, sin)
            if collect:
                collected.append((k, v))
            ck, cv = ctx_kv[i]
            out = attn._attend(
                q, torch.cat([ck, k], 0), torch.cat([cv, v], 0), is_causal=False
            )
            residual = gen_seq + attn.to_add_out(out)
            gen_seq = residual + layer.mlp_moe_gen(
                layer.post_attention_layernorm_moe_gen(residual)
            )
        return gen_seq, collected

    @torch.no_grad()
    def windowed_kv(
        self,
        cond_ids: list[int],
        uncond_ids: list[int] | None,
        total_units: int,
        window_units: int,
        context_units: int,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float = 6.0,
        fps: float = 24.0,
        flow_shift: float | None = None,
        generator: torch.Generator | None = None,
        page_size: int = 128,
    ) -> list[torch.Tensor]:
        """Block-causal windowed generation, the served kv mode's oracle.

        Per window: denoise over [text | committed context | window] with a
        fresh scheduler, commit the finished window's clean K/V (no timestep
        embedding), then evict committed frames older than ``context_units``
        behind the frontier under the paged cache's contract — whole pages
        from the first page fully past the text prefix, floor semantics with
        the shortfall re-offered (``context_units`` 0 retains everything).
        Noise draws mirror the serving path: window 0 first, then one draw
        per boundary from the same generator. Returns the per-window clean
        latents (windows carry no overlap in kv mode)."""
        from diffusers import UniPCMultistepScheduler

        device, dtype = self.device, self.dtype
        tf_cfg = self.config
        # The serving schedule pads requests up to whole windows, so the
        # reference takes that as a precondition.
        assert total_units % window_units == 0, "pass a whole-window total"
        num_windows = total_units // window_units
        tokens_per_unit = (height // self.vae_scale_spatial // tf_cfg.latent_patch_size) * (
            width // self.vae_scale_spatial // tf_cfg.latent_patch_size
        )

        branches = [("cond", cond_ids)]
        if uncond_ids is not None and guidance_scale != 1.0:
            branches.append(("uncond", uncond_ids))

        # Per-branch state: text K/V, per-window statics, committed frame K/V
        # ([tokens, heads, dim] per layer, all committed windows concatenated)
        # and the release bookkeeping mirroring the allocator's contract.
        state: dict[str, dict] = {}
        for name, ids in branches:
            statics = []
            for w in range(num_windows):
                start = w * window_units
                s = build_static_inputs(
                    ids,
                    (1, tf_cfg.latent_channel, window_units,
                     height // self.vae_scale_spatial, width // self.vae_scale_spatial),
                    tf_cfg, self.vae_scale_temporal, fps, device,
                    has_image_condition=False, start_frame_offset=start,
                )
                statics.append(s)
            state[name] = {
                "statics": statics,
                "frames": [None] * len(self.transformer.layers),
                "und_len": len(ids),
                "released": 0,
            }
        branch_kvs = self._ref_und_prefill(
            [ids for _, ids in branches],
            [state[name]["statics"][0]["text_mrope_ids"] for name, _ in branches],
        )
        for (name, _), kvs in zip(branches, branch_kvs, strict=True):
            state[name]["text_kv"] = kvs

        def _ctx(branch):
            """Per-layer [text | retained frames] K/V for one branch. The
            eviction hole starts at the first page boundary past the text
            (frame tokens sharing the text's tail page are never released)."""
            st = state[branch]
            keep = -(-st["und_len"] // page_size) * page_size - st["und_len"]
            out = []
            for i, (tk, tv) in enumerate(st["text_kv"]):
                fr = st["frames"][i]
                if fr is None:
                    out.append((tk, tv))
                    continue
                fk, fv = fr
                k = torch.cat([tk, fk[:keep], fk[keep + st["released"]:]], 0)
                v = torch.cat([tv, fv[:keep], fv[keep + st["released"]:]], 0)
                out.append((k, v))
            return out

        def _release(branch, committed_units):
            st = state[branch]
            if context_units == 0:
                return
            target = max(0, committed_units - context_units) * tokens_per_unit
            ask = target - st["released"]
            if ask <= 0:
                return
            stream_len = st["und_len"] + committed_units * tokens_per_unit - st["released"]
            first = -(-st["und_len"] // page_size)
            releasable = stream_len // page_size - first
            k = min(ask // page_size, releasable)
            if k > 0:
                st["released"] += k * page_size

        tf = self.transformer
        gen_latent_shape = (
            1, tf_cfg.latent_channel, window_units,
            height // self.vae_scale_spatial, width // self.vae_scale_spatial,
        )
        latents = torch.randn(gen_latent_shape, generator=generator, device=device, dtype=dtype)
        windows_out: list[torch.Tensor] = []
        for w in range(num_windows):
            if flow_shift is not None:
                scheduler = UniPCMultistepScheduler.from_config(
                    self.scheduler.config, flow_shift=flow_shift
                )
            else:
                scheduler = UniPCMultistepScheduler.from_config(self.scheduler.config)
            scheduler.set_timesteps(num_inference_steps, device=device)
            s0 = state["cond"]["statics"][w]
            num_noisy = s0["num_noisy_vision_tokens"]
            for t in scheduler.timesteps:
                vts = torch.full((num_noisy,), t.item(), device=device)
                vels = {}
                for name, _ in branches:
                    st = state[name]
                    static = st["statics"][w]
                    packed, orig_shapes = tf._patchify_and_pack_latents([latents.to(dtype)])
                    packed = tf.proj_in(packed)
                    ts_embeds = tf.time_embedder(
                        tf.time_proj(vts * tf_cfg.timestep_scale)
                    ).to(packed.dtype)
                    gen_seq = tf._apply_timestep_embeds_to_noisy_tokens(
                        packed_tokens=packed,
                        packed_timestep_embeds=ts_embeds,
                        noisy_frame_indexes=static["vision_noisy_frame_indexes"],
                        token_shapes=static["vision_token_shapes"],
                    )
                    cos, sin = tf._rotary(
                        static["vision_mrope_ids"], gen_seq.device, gen_seq.dtype
                    )
                    gen_seq, _ = self._ref_gen_layers(gen_seq, cos, sin, _ctx(name))
                    gen_out = tf.norm_moe_gen(gen_seq)
                    mse_idx = static["vision_mse_loss_indexes"] - static["und_len"]
                    preds = tf._unpatchify_and_unpack_latents(
                        tf.proj_out(gen_out[mse_idx]),
                        token_shapes_vision=static["vision_token_shapes"],
                        noisy_frame_indexes_vision=static["vision_noisy_frame_indexes"],
                        original_latent_shapes=orig_shapes,
                    )
                    vels[name] = preds[0]
                if len(branches) > 1:
                    velocity = vels["uncond"] + guidance_scale * (vels["cond"] - vels["uncond"])
                else:
                    velocity = vels["cond"]
                latents = scheduler.step(
                    velocity.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
                )[0].squeeze(0)
            windows_out.append(latents.clone())

            # Commit the finished window's clean K/V per branch, then evict.
            for name, _ in branches:
                st = state[name]
                static = st["statics"][w]
                packed, _ = tf._patchify_and_pack_latents([latents.to(dtype)])
                gen_seq = tf.proj_in(packed)
                cos, sin = tf._rotary(
                    static["vision_mrope_ids"], gen_seq.device, gen_seq.dtype
                )
                _, fresh = self._ref_gen_layers(
                    gen_seq, cos, sin, _ctx(name), collect=True
                )
                for i, (k, v) in enumerate(fresh):
                    fr = st["frames"][i]
                    st["frames"][i] = (
                        (k, v) if fr is None
                        else (torch.cat([fr[0], k], 0), torch.cat([fr[1], v], 0))
                    )
                _release(name, (w + 1) * window_units)
            if w + 1 < num_windows:
                latents = torch.randn(
                    gen_latent_shape, generator=generator, device=device, dtype=dtype
                )
        return windows_out
