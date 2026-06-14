"""NodeSubmodule wrappers for the Cosmos3 generator nodes.

Two nodes:
  Cosmos3DiTSubmodule         -- dual-pathway DiT (KV_CACHE). Dispatches by
                                 graph_walk between ``prefill`` (the
                                 understanding tower runs once over the text
                                 prompt and writes its per-layer K/V) and
                                 ``image_gen`` (one denoising step of the
                                 generation tower per loop iteration, attending
                                 to the frozen understanding K/V plus the
                                 current generation tokens, then one scheduler
                                 step). Classifier-free guidance keeps the
                                 conditional and unconditional prompts in two
                                 cache labels and combines their velocities.
  Cosmos3VAEDecoderSubmodule  -- Wan VAE decode (STATELESS): final latents to
                                 pixels.

Because the text tokens never receive a timestep embedding, the understanding
K/V is denoise-step independent, so writing it once and re-reading it every step
matches running the whole transformer each step.
"""

from __future__ import annotations

import logging

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.cosmos3.packing import build_static_inputs
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

PREFILL_WALK = "prefill"
IMAGE_GEN_WALK = "image_gen"

# Conditional prompt K/V lives under the primary label; the unconditional
# (negative) prompt's K/V lives under a second label for classifier-free
# guidance. Both are written once at prefill and read every denoise step.
COND_LABEL = "main"
UNCOND_LABEL = "uncond"


class Cosmos3DiTSubmodule(ARNodeSubmodule):
    """Dual-pathway DiT node (understanding tower + generation denoiser)."""

    def __init__(self, transformer, config, scheduler=None):
        super().__init__()
        self.transformer = transformer
        self.config = config
        # Template scheduler; a fresh instance (with its own multistep state) is
        # built per request from this one's config.
        self._scheduler_template = scheduler
        # Per-request denoising state: packed static inputs (cond/uncond),
        # scheduler, guidance scale, latent shape.
        self._req: dict[str, dict] = {}

    def get_needed_cache_labels(
        self, graph_walk: str, per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str] | None:
        return [COND_LABEL, UNCOND_LABEL]

    # ------------------------------------------------------------------
    # Static packing + scheduler helpers
    # ------------------------------------------------------------------

    def _latent_shape(
        self, height: int, width: int, num_frames: int = 1
    ) -> tuple[int, int, int, int, int]:
        s = self.config.vae.scale_factor_spatial
        t = 1 if num_frames == 1 else 1 + (num_frames - 1) // self.config.vae.scale_factor_temporal
        return (1, self.config.latent_channel, t, height // s, width // s)

    def _build_static(
        self, ids: list[int], height: int, width: int, num_frames: int,
        fps: float, has_image_condition: bool, device,
    ) -> dict:
        static = build_static_inputs(
            list(ids), self._latent_shape(height, width, num_frames), self.config,
            self.config.vae.scale_factor_temporal, fps, device,
            has_image_condition=has_image_condition,
        )
        # proj_out runs on the generation token block, so shift the joint-sequence
        # mse indexes to be relative to the vision tokens.
        static["mse_gen_indexes"] = static["vision_mse_loss_indexes"] - static["und_len"]
        return static

    def _new_scheduler(self, num_inference_steps: int, device):
        from diffusers import UniPCMultistepScheduler

        scheduler = UniPCMultistepScheduler.from_config(self._scheduler_template.config)
        scheduler.set_timesteps(num_inference_steps, device=device)
        return scheduler

    # ------------------------------------------------------------------
    # prepare_inputs
    # ------------------------------------------------------------------

    def prepare_inputs(
        self, graph_walk, fwd_info, inputs, seen_token_mask=None, pos_info={},
    ) -> ARNodeInputs:
        device = self.get_device()
        if graph_walk == PREFILL_WALK:
            return self._prepare_prefill(fwd_info, inputs, device)
        if graph_walk == IMAGE_GEN_WALK:
            return self._prepare_image_gen(fwd_info, inputs, device)
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    def _prepare_prefill(self, fwd_info, inputs, device) -> ARNodeInputs:
        md = fwd_info.step_metadata
        height, width = int(md.get("height", 256)), int(md.get("width", 256))
        num_frames = int(md.get("num_frames", 1))
        fps = float(md.get("fps", 24.0))
        gs = float(md.get("guidance_scale", 6.0))
        steps = int(md.get("num_inference_steps", self.config.num_inference_steps))
        # Image-to-video: latent frame 0 is a clean conditioning anchor supplied
        # in the first denoise step's ``latents``; it stays in the sequence but is
        # not denoised. (Text-to-image / text-to-video have no clean anchor.)
        has_image_condition = bool(md.get("has_image_condition", False))

        cond = self._build_static(
            inputs["text_inputs"][0].tolist(), height, width, num_frames, fps, has_image_condition, device
        )
        uncond = None
        if gs != 1.0:
            uncond = self._build_static(
                inputs["text_inputs"][1].tolist(), height, width, num_frames, fps, has_image_condition, device
            )

        self._req[fwd_info.request_id] = {
            "cond": cond,
            "uncond": uncond,
            "gs": gs,
            "scheduler": self._new_scheduler(steps, device),
            "num_noisy": cond["num_noisy_vision_tokens"],
            "num_vision": cond["num_vision_tokens"],
            "latent_shape": self._latent_shape(height, width, num_frames),
        }
        return ARNodeInputs(input_seq_len=cond["und_len"])

    def _prepare_image_gen(self, fwd_info, inputs, device) -> ARNodeInputs:
        st = self._req[fwd_info.request_id]
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            gen = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
            latents = torch.randn(
                st["latent_shape"], generator=gen, device=device, dtype=self.transformer.proj_in.weight.dtype
            )
            time_index = torch.zeros(1, dtype=torch.long, device=device)
        else:
            latents = inputs["latents"][0]
            time_index = inputs["time_index"][0]
        return ARNodeInputs(
            input_seq_len=st["num_vision"],
            tensor_inputs={"latents": latents, "time_index": time_index},
        )

    # ------------------------------------------------------------------
    # preprocess: plan paged attention for the labels this walk touches.
    # ------------------------------------------------------------------

    def preprocess(self, graph_walk, engine_inputs: ModelInputsFromEngine, inputs) -> dict:
        cm = engine_inputs.cache_manager
        st = self._req[engine_inputs.request_ids[0]]

        if graph_walk == PREFILL_WALK:
            cm.plan_attention(seq_lens=[st["cond"]["und_len"]], is_causal=True, label=COND_LABEL, write_store=False)
            if st["uncond"] is not None:
                cm.plan_attention(
                    seq_lens=[st["uncond"]["und_len"]], is_causal=True, label=UNCOND_LABEL, write_store=False
                )
            return {}

        if graph_walk == IMAGE_GEN_WALK:
            num_vision = st["num_vision"]
            cm.plan_attention(seq_lens=[num_vision], is_causal=False, label=COND_LABEL, write_store=False)
            if st["uncond"] is not None:
                cm.plan_attention(seq_lens=[num_vision], is_causal=False, label=UNCOND_LABEL, write_store=False)
            return {
                "latents": inputs[0].tensor_inputs["latents"],
                "time_index": inputs[0].tensor_inputs["time_index"],
            }
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, **kwargs):
        cm = engine_inputs.cache_manager
        rid = engine_inputs.request_ids[0]
        if graph_walk == PREFILL_WALK:
            return self._forward_prefill(cm, self._req[rid])
        if graph_walk == IMAGE_GEN_WALK:
            return self._forward_image_gen(cm, self._req[rid], **kwargs)
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    def _forward_prefill(self, cm, st) -> dict:
        cond = st["cond"]
        cm.set_active_label(COND_LABEL)
        self.transformer.prefill_und(cond["input_ids"], cond["text_mrope_ids"], cm)
        if st["uncond"] is not None:
            uncond = st["uncond"]
            cm.set_active_label(UNCOND_LABEL)
            self.transformer.prefill_und(uncond["input_ids"], uncond["text_mrope_ids"], cm)
        return {}

    def _denoise(self, cm, static, latents, vision_timesteps):
        return self.transformer.denoise_step(
            latents,
            vision_timesteps,
            static["vision_mrope_ids"],
            static["vision_token_shapes"],
            static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"],
            cm,
        )

    def _forward_image_gen(self, cm, st, latents, time_index, **kwargs) -> dict:
        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        t = scheduler.timesteps[step_index]
        vision_timesteps = torch.full((st["num_noisy"],), t.item(), device=latents.device)

        cm.set_active_label(COND_LABEL)
        cond_v = self._denoise(cm, st["cond"], latents, vision_timesteps)
        if st["uncond"] is not None:
            cm.set_active_label(UNCOND_LABEL)
            uncond_v = self._denoise(cm, st["uncond"], latents, vision_timesteps)
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
        else:
            velocity = cond_v

        new_latents = scheduler.step(
            velocity.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
        )[0].squeeze(0)
        return {"latents": [new_latents], "time_index": [time_index + 1]}

    def cleanup_request(self, request_id: str):
        self._req.pop(request_id, None)


class Cosmos3VAEDecoderSubmodule(NodeSubmodule):
    """Wan VAE decode node: final denoised latents -> pixel frames.

    Applies the pipeline-side latent normalization (the VAE itself returns raw
    latents) before decoding, matching the fused t2i pipeline's decode.
    """

    def __init__(self, vae, config):
        super().__init__()
        self.vae = vae
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={"latents": inputs["latents"][0]})

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, latents, **kwargs):
        vae = self.vae
        mean = torch.tensor(vae.config.latents_mean, dtype=vae.dtype, device=latents.device).view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=vae.dtype, device=latents.device)).view(
            1, -1, 1, 1, 1
        )
        z = latents.to(vae.dtype) / inv_std + mean
        decoded = vae.decode(z).sample  # [1, 3, T, H, W] in [-1, 1]
        image = (decoded / 2 + 0.5).clamp(0, 1).to(torch.float32)
        return {"image_output": [image]}
