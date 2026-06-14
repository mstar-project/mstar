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
from mstar.model.cosmos3.packing import (
    action_start_frame_offset,
    build_action_static_inputs,
    build_static_inputs,
    vision_condition_frame_indexes,
)
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
ACTION_GEN_WALK = "action_gen"

# Conditional prompt K/V lives under the primary label; the unconditional
# (negative) prompt's K/V lives under a second label for classifier-free
# guidance. Both are written once at prefill and read every denoise step.
COND_LABEL = "main"
UNCOND_LABEL = "uncond"

# Combined label for the single FlashInfer plan that runs both guidance branches
# in one forward (see cache_manager.plan_attention_batched_cfg).
CFG_BATCHED_LABEL = "_cfg_batched"


class Cosmos3DiTSubmodule(ARNodeSubmodule):
    """Dual-pathway DiT node (understanding tower + generation denoiser)."""

    # The denoise loop is data-dependent (per-step timestep .item(), scheduler
    # step, classifier-free guidance combine), so torch.compile graph-breaks and
    # buys little; CUDA-graph capture of the fixed-shape step is the accelerator.
    disable_torch_compile = True

    # Run the two classifier-free-guidance branches as a single batched forward
    # per denoise step instead of two sequential forwards. The math is the same;
    # set False to fall back to the sequential path.
    batched_cfg = True

    # Cap on how many requests share one batched denoise step. Concurrent
    # requests at the image-generation walk run their step in a single forward.
    max_gen_batch_size = 8

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

    def _new_scheduler(self, num_inference_steps: int, device, flow_shift=None):
        from diffusers import UniPCMultistepScheduler

        if flow_shift is not None:
            scheduler = UniPCMultistepScheduler.from_config(self._scheduler_template.config, flow_shift=flow_shift)
        else:
            scheduler = UniPCMultistepScheduler.from_config(self._scheduler_template.config)
        scheduler.set_timesteps(num_inference_steps, device=device)
        return scheduler

    def _build_action_static(
        self, ids: list[int], height: int, width: int, num_frames: int, action_chunk: int,
        mode: str, fps: float, action_fps: float, action_offset: int, device,
    ) -> dict:
        static = build_action_static_inputs(
            list(ids), self._latent_shape(height, width, num_frames), action_chunk, mode,
            self.config, self.config.vae.scale_factor_temporal, fps, action_fps, action_offset, device,
        )
        # proj_out runs on the generation token block; shift the joint-sequence
        # mse indexes to be relative to the [vision | action] generation tokens.
        static["mse_gen_indexes"] = static["vision_mse_loss_indexes"] - static["und_len"]
        static["action_mse_gen_indexes"] = static["action_mse_loss_indexes"] - static["und_len"]
        return static

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
        if graph_walk == ACTION_GEN_WALK:
            return self._prepare_action_gen(fwd_info, inputs, device)
        raise ValueError(f"Unknown Cosmos3 DiT graph walk: {graph_walk!r}")

    def _prepare_prefill(self, fwd_info, inputs, device) -> ARNodeInputs:
        md = fwd_info.step_metadata
        height, width = int(md.get("height", 256)), int(md.get("width", 256))
        fps = float(md.get("fps", 24.0))
        gs = float(md.get("guidance_scale", 6.0))
        steps = int(md.get("num_inference_steps", self.config.num_inference_steps))
        cond_ids = inputs["text_inputs"][0].tolist()
        uncond_ids = inputs["text_inputs"][1].tolist() if gs != 1.0 else None

        action_mode = md.get("action_mode")
        if action_mode:
            return self._prepare_action_prefill(
                fwd_info, md, cond_ids, uncond_ids, height, width, fps, gs, steps, device
            )

        num_frames = int(md.get("num_frames", 1))
        # Image-to-video: latent frame 0 is a clean conditioning anchor supplied
        # in the first denoise step's ``latents``; it stays in the sequence but is
        # not denoised. (Text-to-image / text-to-video have no clean anchor.)
        has_image_condition = bool(md.get("has_image_condition", False))

        cond = self._build_static(cond_ids, height, width, num_frames, fps, has_image_condition, device)
        uncond = None
        if uncond_ids is not None:
            uncond = self._build_static(uncond_ids, height, width, num_frames, fps, has_image_condition, device)

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

    def _prepare_action_prefill(
        self, fwd_info, md, cond_ids, uncond_ids, height, width, fps, gs, steps, device,
    ) -> ARNodeInputs:
        mode = md["action_mode"]
        action_chunk = int(md["action_chunk_size"])
        num_frames = int(md.get("num_frames") or action_chunk + 1)
        raw_action_dim = int(md["raw_action_dim"])
        domain_id = int(md.get("domain_id", 0))
        action_fps = float(md.get("action_fps", fps))
        action_offset = action_start_frame_offset(action_chunk, num_frames)

        cond = self._build_action_static(
            cond_ids, height, width, num_frames, action_chunk, mode, fps, action_fps, action_offset, device
        )
        uncond = None
        if uncond_ids is not None:
            uncond = self._build_action_static(
                uncond_ids, height, width, num_frames, action_chunk, mode, fps, action_fps, action_offset, device
            )

        latent_shape = self._latent_shape(height, width, num_frames)
        t_lat = latent_shape[2]
        dtype = self.transformer.proj_in.weight.dtype
        vmask = torch.zeros((1, 1, t_lat, 1, 1), device=device, dtype=dtype)
        for f in vision_condition_frame_indexes(mode, t_lat):
            vmask[:, :, f] = 1.0
        action_clean = torch.zeros((1, action_chunk, 1), device=device, dtype=dtype)
        if mode == "forward_dynamics":
            action_clean[:] = 1.0

        self._req[fwd_info.request_id] = {
            "cond": cond,
            "uncond": uncond,
            "gs": gs,
            "scheduler": self._new_scheduler(steps, device, flow_shift=md.get("flow_shift")),
            "num_noisy": cond["num_noisy_vision_tokens"],
            "num_noisy_action": cond["num_noisy_action_tokens"],
            "num_vision": cond["num_vision_tokens"],
            "num_action": cond["num_action_tokens"],
            "latent_shape": latent_shape,
            "action_mode": mode,
            "action_chunk": action_chunk,
            "action_dim": self.transformer.action_dim,
            "raw_action_dim": raw_action_dim,
            "domain_t": torch.tensor([domain_id], dtype=torch.long, device=device),
            "vmask": vmask,
            "velocity_mask": 1.0 - vmask,
            "action_clean_mask": action_clean,
            "action_velocity_mask": 1.0 - action_clean,
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

    def _prepare_action_gen(self, fwd_info, inputs, device) -> ARNodeInputs:
        st = self._req[fwd_info.request_id]
        # The conditioning video latents and the initial (noisy) action latents
        # are supplied to the first loop iteration; the clean anchors are carried
        # in the looped latents (re-injected each step), like the i2v path.
        latents = inputs["latents"][0]
        action_latents = inputs["action_latents"][0]
        time_index = (
            inputs["time_index"][0]
            if "time_index" in inputs and len(inputs["time_index"])
            else torch.zeros(1, dtype=torch.long, device=device)
        )
        return ARNodeInputs(
            input_seq_len=st["num_vision"] + st["num_action"],
            tensor_inputs={"latents": latents, "action_latents": action_latents, "time_index": time_index},
        )

    # ------------------------------------------------------------------
    # preprocess: plan paged attention for the labels this walk touches.
    # ------------------------------------------------------------------

    def _plan_gen(self, cm, st, num_gen: int) -> None:
        """Plan a denoise step's non-causal attention: one batched plan covering
        both guidance branches when they run together, else a plan per label."""
        if st["uncond"] is None:
            cm.plan_attention(seq_lens=[num_gen], is_causal=False, label=COND_LABEL, write_store=False)
        elif self.batched_cfg:
            cm.plan_attention_batched_cfg(
                labels=[COND_LABEL, UNCOND_LABEL], seq_lens=[num_gen],
                is_causal=False, write_store=False,
            )
        else:
            cm.plan_attention(seq_lens=[num_gen], is_causal=False, label=COND_LABEL, write_store=False)
            cm.plan_attention(seq_lens=[num_gen], is_causal=False, label=UNCOND_LABEL, write_store=False)

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
            rids = engine_inputs.request_ids
            if len(rids) > 1:
                # Cross-request batch: one batched plan over every request's two
                # guidance branches, each with its own page set and token count.
                cm.plan_attention_batched_cfg(
                    labels=[COND_LABEL, UNCOND_LABEL],
                    seq_lens=[self._req[r]["num_vision"] for r in rids],
                    is_causal=False, write_store=False,
                )
                return {
                    "latents": {r: inp.tensor_inputs["latents"] for r, inp in zip(rids, inputs)},
                    "time_index": {r: inp.tensor_inputs["time_index"] for r, inp in zip(rids, inputs)},
                }
            self._plan_gen(cm, st, st["num_vision"])
            return {
                "latents": inputs[0].tensor_inputs["latents"],
                "time_index": inputs[0].tensor_inputs["time_index"],
            }

        if graph_walk == ACTION_GEN_WALK:
            self._plan_gen(cm, st, st["num_vision"] + st["num_action"])
            return {
                "latents": inputs[0].tensor_inputs["latents"],
                "action_latents": inputs[0].tensor_inputs["action_latents"],
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
        if graph_walk == ACTION_GEN_WALK:
            return self._forward_action_gen(cm, self._req[rid], **kwargs)
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

        if st["uncond"] is None:
            cm.set_active_label(COND_LABEL)
            velocity = self._denoise(cm, st["cond"], latents, vision_timesteps)
        elif self.batched_cfg:
            cm.set_active_label(CFG_BATCHED_LABEL)
            cond_v, uncond_v = self.transformer.denoise_step_batched_cfg(
                latents,
                vision_timesteps,
                st["cond"]["vision_mrope_ids"],
                st["uncond"]["vision_mrope_ids"],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
                cm,
            )
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
        else:
            cm.set_active_label(COND_LABEL)
            cond_v = self._denoise(cm, st["cond"], latents, vision_timesteps)
            cm.set_active_label(UNCOND_LABEL)
            uncond_v = self._denoise(cm, st["uncond"], latents, vision_timesteps)
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)

        new_latents = scheduler.step(
            velocity.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
        )[0].squeeze(0)
        return {"latents": [new_latents], "time_index": [time_index + 1]}

    def _denoise_action(self, cm, static, latents, action_latents, vts, ats, domain):
        und_len = static["und_len"]
        return self.transformer.denoise_step(
            latents,
            vts,
            static["position_ids"][:, und_len:],
            static["vision_token_shapes"],
            static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"],
            cm,
            action_latents=action_latents,
            action_token_shapes=static["action_token_shapes"],
            action_noisy_frame_indexes=static["action_noisy_frame_indexes"],
            action_mse_gen_indexes=static["action_mse_gen_indexes"],
            action_timesteps=ats,
            action_domain_id=domain,
        )

    def _forward_action_gen(self, cm, st, latents, action_latents, time_index, **kwargs) -> dict:
        scheduler = st["scheduler"]
        step_index = int(time_index.reshape(-1)[0].item())
        t = scheduler.timesteps[step_index]
        device = latents.device
        vts = torch.full((st["num_noisy"],), t.item(), device=device)
        ats = torch.full((st["num_noisy_action"],), t.item(), device=device)
        domain = st["domain_t"]
        raw, chunk, adim = st["raw_action_dim"], st["action_chunk"], st["action_dim"]
        velocity_mask, vmask = st["velocity_mask"], st["vmask"]
        action_vmask, action_cmask = st["action_velocity_mask"], st["action_clean_mask"]

        if st["uncond"] is None:
            cm.set_active_label(COND_LABEL)
            video_v, action_v = self._denoise_action(cm, st["cond"], latents, action_latents, vts, ats, domain)
        elif self.batched_cfg:
            cm.set_active_label(CFG_BATCHED_LABEL)
            (video_v, action_v), (v_u, a_u) = self.transformer.denoise_step_batched_cfg(
                latents,
                vts,
                st["cond"]["position_ids"][:, st["cond"]["und_len"]:],
                st["uncond"]["position_ids"][:, st["uncond"]["und_len"]:],
                st["cond"]["vision_token_shapes"],
                st["cond"]["vision_noisy_frame_indexes"],
                st["cond"]["mse_gen_indexes"],
                cm,
                action_latents=action_latents,
                action_token_shapes=st["cond"]["action_token_shapes"],
                action_noisy_frame_indexes=st["cond"]["action_noisy_frame_indexes"],
                action_mse_gen_indexes=st["cond"]["action_mse_gen_indexes"],
                action_timesteps=ats,
                action_domain_id=domain,
            )
            video_v = v_u + st["gs"] * (video_v - v_u)
            action_v = a_u + st["gs"] * (action_v - a_u)
        else:
            cm.set_active_label(COND_LABEL)
            video_v, action_v = self._denoise_action(cm, st["cond"], latents, action_latents, vts, ats, domain)
            cm.set_active_label(UNCOND_LABEL)
            v_u, a_u = self._denoise_action(cm, st["uncond"], latents, action_latents, vts, ats, domain)
            video_v = v_u + st["gs"] * (video_v - v_u)
            action_v = a_u + st["gs"] * (action_v - a_u)

        video_v = video_v * velocity_mask
        action_v = action_v * action_vmask
        action_v[..., raw:] = 0

        nv = video_v.numel()
        packed = torch.cat([video_v.reshape(1, -1), action_v.reshape(1, -1)], dim=1)
        packed_lat = torch.cat([latents.reshape(1, -1), action_latents.reshape(1, -1)], dim=1)
        packed_next = scheduler.step(packed, t, packed_lat, return_dict=False)[0]
        new_latents = packed_next[:, :nv].reshape(latents.shape)
        new_action = packed_next[:, nv:].reshape(1, chunk, adim)

        # Re-inject the clean anchors (the conditioning video frames / action
        # tokens stay clean each step; their masked-in values are invariant).
        new_latents = velocity_mask * new_latents + vmask * latents
        new_action = action_vmask * new_action + action_cmask * action_latents
        new_action[..., raw:] = 0
        return {
            "latents": [new_latents],
            "action_latents": [new_action],
            "time_index": [time_index + 1],
            "action_output": [new_action[:, :, :raw]],
        }

    # ------------------------------------------------------------------
    # Cross-request batching: run several requests' denoise step together.
    # ------------------------------------------------------------------

    def can_batch(self, batch, model_inputs) -> bool:
        # Only the image/video denoise step batches across requests, and only
        # when every request is in the two-branch guidance regime (so a single
        # batched plan covers them). One request stays on the simpler path.
        if batch.graph_walk != IMAGE_GEN_WALK or not self.batched_cfg:
            return False
        if len(batch.request_ids) < 2:
            return False
        return all(
            rid in self._req and self._req[rid]["uncond"] is not None
            for rid in batch.request_ids
        )

    def max_batch_size(self, graph_walk: str):
        return self.max_gen_batch_size if graph_walk == IMAGE_GEN_WALK else None

    def forward_batched(self, graph_walk, engine_inputs: ModelInputsFromEngine, latents, time_index, **kwargs):
        if graph_walk != IMAGE_GEN_WALK:
            raise ValueError(f"Cosmos3 batched forward only supports image generation, got {graph_walk!r}")
        cm = engine_inputs.cache_manager
        cm.set_active_label(CFG_BATCHED_LABEL)
        reqs, meta = [], []
        for rid in engine_inputs.request_ids:
            st = self._req[rid]
            lat, ti = latents[rid], time_index[rid]
            t = st["scheduler"].timesteps[int(ti.reshape(-1)[0].item())]
            reqs.append({
                "latents": lat,
                "vision_timesteps": torch.full((st["num_noisy"],), t.item(), device=lat.device),
                "position_ids_cond": st["cond"]["vision_mrope_ids"],
                "position_ids_uncond": st["uncond"]["vision_mrope_ids"],
                "vision_token_shapes": st["cond"]["vision_token_shapes"],
                "vision_noisy_frame_indexes": st["cond"]["vision_noisy_frame_indexes"],
                "vision_mse_loss_indexes": st["cond"]["mse_gen_indexes"],
            })
            meta.append((rid, st, lat, ti, t))

        results = self.transformer.denoise_step_batched(reqs, cm)

        out = {}
        for (rid, st, lat, ti, t), (cond_v, uncond_v) in zip(meta, results):
            velocity = uncond_v + st["gs"] * (cond_v - uncond_v)
            new_latents = st["scheduler"].step(
                velocity.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
            out[rid] = {"latents": [new_latents], "time_index": [ti + 1]}
        return out

    def cleanup_request(self, request_id: str):
        self._req.pop(request_id, None)


class Cosmos3VAEDecoderSubmodule(NodeSubmodule):
    """Wan VAE decode node: final denoised latents -> pixel frames.

    Applies the pipeline-side latent normalization (the VAE itself returns raw
    latents) before decoding, matching the fused t2i pipeline's decode.
    """

    # One-shot decode per request; CUDA-graph capture (not torch.compile) is the
    # speedup path.
    disable_torch_compile = True

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
