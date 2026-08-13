from __future__ import annotations

import base64
import io
import logging
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.lingbot.config import LingBotConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.model.wan22.components.unipc import (
    SOLVER_ORDER,
    UniPCState,
    unipc_convert_model_output,
    unipc_corrector_step,
    unipc_effective_order,
    unipc_predictor_step,
)

logger = logging.getLogger(__name__)
DENOISE_LOOP_NAME = "denoise_loop"


def _no_autocast():
    return torch.amp.autocast("cuda", enabled=False) if torch.cuda.is_available() else nullcontext()


def decode_init_latents(encoded: str) -> torch.Tensor:
    raw = base64.b64decode(encoded)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def make_flow_unipc_tables(num_inference_steps: int, shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas = np.linspace(1, 1 / 1000, num_inference_steps + 1).copy()[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = torch.from_numpy((sigmas * 1000).copy()).to(torch.int64)
    sigmas = torch.from_numpy(np.concatenate([sigmas, [0.0]]).astype(np.float32))
    return sigmas, timesteps


def _module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _transformer_timestep(timestep: torch.Tensor, transformer_dtype: torch.dtype) -> torch.Tensor:
    sigma = timestep.float() / 1000.0
    if transformer_dtype in {torch.bfloat16, torch.float16}:
        sigma = sigma.to(transformer_dtype)
    return (sigma * 1000.0).float()


def _latent_grid(config: LingBotConfig, step_metadata: dict) -> tuple[int, int, int]:
    t_lat = (int(step_metadata["num_frames"]) - 1) // config.vae_scale_factor_temporal + 1
    h_lat = int(step_metadata["height"]) // config.vae_scale_factor_spatial
    w_lat = int(step_metadata["width"]) // config.vae_scale_factor_spatial
    return t_lat, h_lat, w_lat


class LingBotTextEncoderSubmodule(NodeSubmodule):
    disable_torch_compile = True

    def __init__(self, text_encoder: nn.Module, config: LingBotConfig):
        super().__init__()
        self.text_encoder = text_encoder
        self.config = config

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs
    ) -> NodeInputs:
        tensor_inputs = {
            "positive_input_ids": inputs["positive_input_ids"][0],
            "positive_attention_mask": inputs["positive_attention_mask"][0],
            "positive_crop_start": inputs["positive_crop_start"][0],
        }
        if float(fwd_info.step_metadata["guidance_scale"]) > 1.0:
            tensor_inputs.update(
                negative_input_ids=inputs["negative_input_ids"][0],
                negative_attention_mask=inputs["negative_attention_mask"][0],
                negative_crop_start=inputs["negative_crop_start"][0],
            )
        return NodeInputs(tensor_inputs=tensor_inputs)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        positive_crop_start: torch.Tensor,
        negative_input_ids: torch.Tensor | None = None,
        negative_attention_mask: torch.Tensor | None = None,
        negative_crop_start: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        with _no_autocast():
            pos, pos_mask = self._encode_one(positive_input_ids, positive_attention_mask, positive_crop_start)
            if negative_input_ids is not None:
                neg, neg_mask = self._encode_one(negative_input_ids, negative_attention_mask, negative_crop_start)
            else:
                neg = pos.new_zeros(1)
                neg_mask = positive_attention_mask.new_zeros(1)
        return {
            "text_embeds_pos": [pos],
            "text_mask_pos": [pos_mask],
            "text_embeds_neg": [neg],
            "text_mask_neg": [neg_mask],
        }

    def _encode_one(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, crop_start: torch.Tensor):
        device = self.get_device()
        input_ids = input_ids.to(device=device, dtype=torch.long).unsqueeze(0)
        attention_mask = attention_mask.to(device=device, dtype=torch.long).unsqueeze(0)
        outputs = self.text_encoder.encode_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=bool(self.config.hidden_state_skip_layer),
        )
        embeds = outputs.last_hidden_state
        if self.config.hidden_state_skip_layer:
            embeds = outputs.hidden_states[-(self.config.hidden_state_skip_layer + 1)]
        crop = int(crop_start.reshape(-1)[0].item())
        if crop > 0:
            embeds = embeds[:, crop:]
            attention_mask = attention_mask[:, crop:]
        if embeds.shape[0] == 1:
            true_len = int(attention_mask[0].sum().item())
            embeds = embeds[:, :true_len]
            attention_mask = attention_mask[:, :true_len]
        return embeds, attention_mask


class LingBotDitSubmodule(NodeSubmodule):
    _VIDEO_GEN_WALK = "video_gen"
    disable_torch_compile = True

    def __init__(self, transformer: nn.Module, config: LingBotConfig):
        super().__init__()
        self.transformer = transformer
        self.config = config

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs
    ) -> NodeInputs | None:
        tensor_inputs = {
            "text_embeds_pos": inputs["text_embeds_pos"][0],
            "text_mask_pos": inputs["text_mask_pos"][0],
            "text_embeds_neg": inputs["text_embeds_neg"][0],
            "text_mask_neg": inputs["text_mask_neg"][0],
        }
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            device = self.get_device()
            t_lat, h_lat, w_lat = _latent_grid(self.config, fwd_info.step_metadata)
            shape = (1, self.config.in_channels, t_lat, h_lat, w_lat)
            init = inputs.get("init_latents", [])
            if init:
                latents = init[0].to(device=device, dtype=torch.float32)
                if tuple(latents.shape) != shape:
                    raise ValueError(f"init_latents shape must be {shape}, got {tuple(latents.shape)}")
                logger.info("LingBot denoise: using request init_latents with shape %s", tuple(latents.shape))
            else:
                generator = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
                latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            tensor_inputs["latents"] = latents
            tensor_inputs["time_index"] = torch.zeros(1, dtype=torch.int64, device=device)
            tensor_inputs["unipc_model_outputs"] = torch.zeros(
                (SOLVER_ORDER, *shape), dtype=torch.float32, device=device
            )
            tensor_inputs["unipc_last_sample"] = torch.zeros(shape, dtype=torch.float32, device=device)
        else:
            time_index = inputs["time_index"][0]
            k = int(time_index.reshape(-1)[0].item())
            num_steps = int(fwd_info.step_metadata["num_inference_steps"])
            if k >= num_steps:
                return None
            tensor_inputs["latents"] = inputs["latents"][0]
            tensor_inputs["time_index"] = time_index
            tensor_inputs["unipc_model_outputs"] = inputs["unipc_model_outputs"][0]
            tensor_inputs["unipc_last_sample"] = inputs["unipc_last_sample"][0]
        return NodeInputs(tensor_inputs=tensor_inputs)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        latents: torch.Tensor,
        time_index: torch.Tensor,
        unipc_model_outputs: torch.Tensor,
        unipc_last_sample: torch.Tensor,
        text_embeds_pos: torch.Tensor,
        text_mask_pos: torch.Tensor,
        text_embeds_neg: torch.Tensor,
        text_mask_neg: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        step_metadata = engine_inputs.single_request_info.step_metadata
        num_steps = int(step_metadata["num_inference_steps"])
        guidance_scale = float(step_metadata["guidance_scale"])
        shift = float(step_metadata["shift"])
        k = int(time_index.item())
        device = latents.device
        with _no_autocast():
            sigmas, timesteps = make_flow_unipc_tables(num_steps, shift)
            dtype = _module_dtype(self.transformer)
            timestep = _transformer_timestep(timesteps[k], dtype).expand(1).to(device)
            noise_pred = self._noise_prediction(
                latents, timestep, text_embeds_pos, text_mask_pos, text_embeds_neg, text_mask_neg, guidance_scale
            )
            m_k = unipc_convert_model_output(noise_pred, latents, sigmas, k)
            sample = latents
            if k > 0:
                sample = unipc_corrector_step(
                    UniPCState(model_outputs=unipc_model_outputs, last_sample=unipc_last_sample),
                    this_model_output=m_k,
                    this_sample=sample,
                    sigmas=sigmas,
                    step_index=k,
                    order=unipc_effective_order(k - 1, num_steps),
                )
            new_ring = torch.stack([unipc_model_outputs[1], m_k])
            new_latents = unipc_predictor_step(
                UniPCState(model_outputs=new_ring, last_sample=sample),
                sample=sample,
                sigmas=sigmas,
                step_index=k,
                order=unipc_effective_order(k, num_steps),
            )
        return {
            "latents": [new_latents],
            "time_index": [time_index + 1],
            "unipc_model_outputs": [new_ring],
            "unipc_last_sample": [sample],
        }

    def _noise_prediction(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds_pos: torch.Tensor,
        text_mask_pos: torch.Tensor,
        text_embeds_neg: torch.Tensor,
        text_mask_neg: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        dtype = _module_dtype(self.transformer)
        latent_model_input = latents.to(dtype)
        noise_pred = self.transformer(
            latent_model_input,
            timestep,
            text_embeds_pos.to(dtype),
            encoder_attention_mask=text_mask_pos,
            return_dict=False,
        )[0].float()
        if guidance_scale <= 1.0:
            return noise_pred
        noise_uncond = self.transformer(
            latent_model_input,
            timestep,
            text_embeds_neg.to(dtype),
            encoder_attention_mask=text_mask_neg,
            return_dict=False,
        )[0].float()
        return noise_uncond + guidance_scale * (noise_pred - noise_uncond)

    def check_stop(self, request_id: str, request_info: CurrentForwardPassInfo, outputs) -> set[str]:
        iter_idx = request_info.dynamic_loop_iter_counts.get(DENOISE_LOOP_NAME, 0)
        requested = int(request_info.step_metadata.get("num_inference_steps", 0) or 0)
        if requested > 0 and iter_idx + 1 >= requested:
            return {DENOISE_LOOP_NAME}
        return set()


class LingBotVaeDecoderSubmodule(NodeSubmodule):
    disable_torch_compile = True

    def __init__(self, vae: nn.Module, config: LingBotConfig):
        super().__init__()
        self.vae = vae
        self.config = config

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs
    ) -> NodeInputs:
        return NodeInputs(tensor_inputs={"latents": inputs["latents"][0]})

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, latents: torch.Tensor, **kwargs
    ) -> NameToTensorList:
        with _no_autocast():
            device = self.get_device()
            vae = self.vae
            mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
            std_inv = (1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32)).view(
                1, -1, 1, 1, 1
            )
            vae_latents = latents.to(device=device, dtype=torch.float32) / std_inv + mean
            vae_latents = vae_latents.contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                decoded = vae.decode(vae_latents)
            video = decoded[0] if isinstance(decoded, tuple) else decoded.sample
            video = video.float().clamp(-1, 1)
            video = ((video + 1.0) / 2.0).mul(255).clamp(0, 255).to(torch.uint8)
        return {"video_output": [video]}
