# Copyright (c) 2026 Robbyant Team.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the causal-fast chunk loop split
# into M* graph-node submodules, one chunk per rollout iteration.
#
# Original file (Robbyant/lingbot-world, wan/image2video_fast.py) was released
# under Apache-2.0, with the full license text available at
# https://github.com/Robbyant/lingbot-world/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

"""LingBot-World-V2 node submodules.

    text_encoder -> wan22's ``Wan22TextEncoderSubmodule`` (UMT5-XXL), reused as is
    vae_encoder  -> LingBotVaeEncoderSubmodule (Wan2.1 VAE encode of the I2V condition)
    dit          -> LingBotDitSubmodule (one chunk per rollout iteration, decode fused)

None declares engine resources yet: the rolling self-attention K/V and the text
K/V live in the request's ``PerRequestState`` (dense, model-owned, freed with the
request), the decoder's causal-conv cache likewise. Moving the K/V onto the
engine's KV pool is the streaming milestone, not this prototype.

Numerics are the model's, not the engine's: ``LingBotWorldModel.get_autocast_dtype``
returns None, the DiT enters the reference's bf16 autocast itself, and the VAE
runs fp32 as the reference does.
"""

import logging

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components import wan_vae
from mstar.model.lingbot_world.components.camera import plucker_embedding
from mstar.model.lingbot_world.components.dit import LingBotCausalDiT
from mstar.model.lingbot_world.components.schedule import add_noise, flow_to_x0, make_fast_schedule
from mstar.model.lingbot_world.config import LingBotWorldConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.model.wan22.submodules import _SingleRequestMixin

logger = logging.getLogger(__name__)

ROLLOUT_LOOP_NAME = "rollout_loop"
ROLLOUT_WALK = "rollout"


def first_frame_mask(num_latent_frames: int, lat_h: int, lat_w: int, device) -> torch.Tensor:
    """The reference's ``[4, lat_f, lat_h, lat_w]`` I2V mask: the first video
    frame is known, repeated over the first latent's four temporal slots."""
    num_frames = (num_latent_frames - 1) * 4 + 1
    msk = torch.ones(1, num_frames, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    return msk.transpose(1, 2)[0]


class LingBotVaeEncoderSubmodule(_SingleRequestMixin, NodeSubmodule):
    """Encodes the conditioning image, followed by ``F - 1`` black frames, with
    the Wan2.1 VAE in fp32 (the reference's ``vae.encode`` of the same clip),
    and prepends the first-frame mask: ``cond_latents`` ``[20, lat_f, h, w]``.

    The image arrives already resized and normalized to [-1, 1] by
    ``LingBotWorldModel.process_prompt``, which runs the reference's CPU bicubic
    resize so the pixels match it exactly.
    """

    disable_torch_compile = True

    def __init__(self, vae: nn.Module, config: LingBotWorldConfig):
        super().__init__()
        self.vae = vae
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info: CurrentForwardPassInfo, inputs, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={"image": inputs["image_inputs"][0]})

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, image: torch.Tensor, **kwargs):
        md = engine_inputs.single_request_info.step_metadata
        lat_f = int(md["num_latent_frames"])
        num_frames = (lat_f - 1) * self.config.vae_stride[0] + 1
        device = self.get_device()
        with torch.amp.autocast("cuda", enabled=False), torch.inference_mode():
            _, h, w = image.shape
            clip = torch.concat([image.float().cpu()[:, None], torch.zeros(3, num_frames - 1, h, w)], dim=1)
            latent = self.vae.encode(clip[None].to(device)).latent_dist.mode().float()
            z = self.config.vae_z_dim
            mean = torch.tensor(self.vae.config.latents_mean).view(1, z, 1, 1, 1).to(latent)
            inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, z, 1, 1, 1).to(latent)
            latent = ((latent - mean) * inv_std)[0]
            msk = first_frame_mask(lat_f, latent.shape[2], latent.shape[3], device)
            return {"cond_latents": [torch.concat([msk, latent])]}


class LingBotDitSubmodule(_SingleRequestMixin, NodeSubmodule):
    """One rollout iteration is one chunk of ``chunk_size`` latent frames:

      * ``len(timesteps_index)`` denoise forwards, each converting the flow
        prediction to x0 and re-noising it to the next timestep from the
        request's generator, then
      * one commit forward over the clean x0 at t=0, which leaves the chunk's
        settled K/V in the rolling cache for the chunks after it, then
      * a decode of the chunk's latents that continues the request's Wan VAE
        decoder cache (#425's ``decode_causal``), so the frames are those of one
        decode of the whole rollout, emitted as ``video_output``.

    At the first iteration the request's state is seeded: the K/V cache, the
    generator (seeded with the request seed), the full rollout's initial noise
    drawn from it in one call, and the Plucker rays for every latent frame. The
    reference draws the initial noise the same way, which is what makes the
    rollout reproducible bit for bit; it requires the rollout length up front.
    """

    disable_torch_compile = True

    def __init__(self, dit: LingBotCausalDiT, vae: nn.Module, config: LingBotWorldConfig):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.config = config
        self.schedule = make_fast_schedule(config.sample_shift, config.timesteps_index, config.num_train_timesteps)
        if vae is not None and config.vae_decode_channels_last:
            wan_vae.to_channels_last(vae)
        self._decode_step = wan_vae.DecoderStep(vae, compile=False) if vae is not None else None
        # Debug hook for the parity harness: keep the last chunk's latents and
        # float frames in the request state.
        self.keep_chunk_outputs = False

    def prepare_inputs(self, graph_walk, fwd_info: CurrentForwardPassInfo, inputs, **kwargs) -> NodeInputs | None:
        state = self.request_state(fwd_info.rid_handle)
        num_chunks = int(fwd_info.step_metadata["num_chunks"])
        chunk = int(state.get("chunk", 0))
        if chunk >= num_chunks:
            logger.info(
                "LingBot dit: skipping async-overshoot chunk %d (request %s runs %d chunks)",
                chunk, fwd_info.request_id, num_chunks,
            )
            return None
        return NodeInputs(tensor_inputs={
            "text_embeds": inputs["text_embeds_pos"][0],
            "cond_latents": inputs["cond_latents"][0],
            "camera_poses": inputs["camera_poses"][0],
            "intrinsics": inputs["intrinsics"][0],
        })

    def _seed_request(self, state, md: dict, seed: int, cond_latents, camera_poses, intrinsics, device):
        cfg = self.config
        lat_f, lat_h, lat_w = cond_latents.shape[1:]
        frame_seqlen = lat_h * lat_w // (cfg.patch_size[1] * cfg.patch_size[2])
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noise = torch.randn(cfg.out_dim, lat_f, lat_h, lat_w, dtype=torch.float32, generator=generator, device=device)
        height, width = lat_h * cfg.vae_stride[1], lat_w * cfg.vae_stride[2]
        plucker = plucker_embedding(
            camera_poses.float().to(device), intrinsics.float(), height, width, lat_h, lat_w,
        ).to(torch.bfloat16)
        state.add_all(
            chunk=0,
            frame_seqlen=frame_seqlen,
            generator=generator,
            noise=noise,
            plucker=plucker,
            kv=self.dit.new_cache(frame_seqlen, lat_f, device),
            feat_cache=wan_vae.new_feat_cache(self.vae) if self.vae is not None else None,
        )

    def forward(
        self, graph_walk, engine_inputs: ModelInputsFromEngine, text_embeds, cond_latents,
        camera_poses, intrinsics, **kwargs,
    ) -> NameToTensorList:
        cfg = self.config
        rid = engine_inputs.request_ids[0]
        info = engine_inputs.per_request_info[rid]
        state = self.request_state(rid)
        device = self.get_device()
        if "kv" not in state:
            self._seed_request(state, info.step_metadata, info.random_seed, cond_latents.to(device),
                               camera_poses, intrinsics, device)
        chunk = int(state["chunk"])
        frame_seqlen = int(state["frame_seqlen"])
        cs = cfg.chunk_size
        frames = slice(chunk * cs, (chunk + 1) * cs)
        latent = state["noise"][:, frames]
        y = cond_latents.to(device)[:, frames]
        plucker = state["plucker"][:, :, frames]
        context = text_embeds.to(device)
        context = context[0] if context.dim() == 3 else context
        current_start = chunk * cs * frame_seqlen
        kv = state["kv"]
        generator = state["generator"]

        with torch.inference_mode():
            steps = self.schedule.step_timesteps
            for i, t in enumerate(steps):
                timestep = torch.stack([t]).to(device)
                flow = self.dit(latent, timestep, context, y, plucker, kv, current_start, frame_seqlen)
                x0 = flow_to_x0(flow, latent, t, self.schedule)
                if i < len(steps) - 1:
                    noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
                    latent = add_noise(x0, noise, steps[i + 1], self.schedule)
            commit_t = torch.stack([steps[-1] * 0.0]).to(device)
            self.dit(x0, commit_t, context, y, plucker, kv, current_start, frame_seqlen)
            pixels = self._decode(x0, state, clip_start=chunk == 0)

        state.add("chunk", chunk + 1)
        if self.keep_chunk_outputs:
            state.add_all(chunk_latents=x0.clone(), chunk_pixels=pixels.clone())
        frames_u8 = (pixels[0] / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
        return {
            "clock": [torch.full((1,), chunk + 1, dtype=torch.int64, device=device)],
            "video_output": [frames_u8],
        }

    def _decode(self, latents: torch.Tensor, state, clip_start: bool) -> torch.Tensor:
        """``[1, 3, frames, H, W]`` in [-1, 1]: the chunk decoded as the
        continuation of the request's stream."""
        cfg = self.config
        dtype = torch.float32 if cfg.vae_decode_fp32 else torch.bfloat16
        z = latents[None].to(dtype)
        mean = torch.tensor(self.vae.config.latents_mean).view(1, cfg.vae_z_dim, 1, 1, 1).to(z)
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, cfg.vae_z_dim, 1, 1, 1).to(z)
        z = z / inv_std + mean
        with torch.amp.autocast("cuda", enabled=False):
            return wan_vae.decode_causal(
                self.vae, z, state["feat_cache"], self._decode_step, clip_start=clip_start, latents_per_call=1,
            )

    def check_stop(self, request_id, request_info: CurrentForwardPassInfo, outputs) -> set[str]:
        done = request_info.dynamic_loop_iter_counts.get(ROLLOUT_LOOP_NAME, 0)
        if done + 1 >= int(request_info.step_metadata.get("num_chunks", 0) or 0):
            return {ROLLOUT_LOOP_NAME}
        return set()
