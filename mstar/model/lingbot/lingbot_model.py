from __future__ import annotations

import io
import logging
from fractions import Fraction

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.lingbot.config import LingBotConfig
from mstar.model.lingbot.submodules import (
    DENOISE_LOOP_NAME,
    LingBotDitSubmodule,
    LingBotTextEncoderSubmodule,
    LingBotVaeDecoderSubmodule,
    decode_init_latents,
)
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class LingBotModel(Model):
    ENCODE_TEXT_WALK = "encode_text"
    VIDEO_GEN_WALK = "video_gen"

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.config = LingBotConfig()
        self.skip_weight_loading = skip_weight_loading
        self.processor = None
        self._processor_initialized = False
        self._crop_start: int | None = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        return []

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "text_encoder": EngineType.STATELESS,
            "dit": EngineType.STATELESS,
            "vae_decoder": EngineType.STATELESS,
        }

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        encode_text = GraphNode(
            name="text_encoder",
            input_names=[
                "positive_input_ids",
                "positive_attention_mask",
                "positive_crop_start",
                "negative_input_ids",
                "negative_attention_mask",
                "negative_crop_start",
            ],
            outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds_pos", persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_mask_pos", persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds_neg", persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_mask_neg", persist=True),
            ],
        )
        denoise_loop = Loop(
            name=DENOISE_LOOP_NAME,
            section=GraphNode(
                name="dit",
                input_names=[
                    "text_embeds_pos",
                    "text_mask_pos",
                    "text_embeds_neg",
                    "text_mask_neg",
                    "init_latents",
                    "latents",
                    "time_index",
                    "unipc_model_outputs",
                    "unipc_last_sample",
                ],
                outputs=[
                    GraphEdge(next_node="dit", name="latents"),
                    GraphEdge(next_node="dit", name="time_index"),
                    GraphEdge(next_node="dit", name="unipc_model_outputs"),
                    GraphEdge(next_node="dit", name="unipc_last_sample"),
                ],
                enable_async_scheduling=True,
            ),
            max_iters=self.config.max_denoise_steps,
            outputs=[GraphEdge(next_node="vae_decoder", name="latents")],
        )
        vae_decoder = GraphNode(
            name="vae_decoder",
            input_names=["latents"],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="video_output", output_modality="video")],
        )
        return {
            self.ENCODE_TEXT_WALK: encode_text,
            self.VIDEO_GEN_WALK: Sequential([denoise_loop, vae_decoder]),
        }

    def _ensure_processor(self) -> None:
        if self._processor_initialized:
            return
        from transformers import AutoProcessor

        from mstar.model.lingbot.weight_loader import resolve_subfolder_dir

        # Resolve the processor to a LOCAL directory first and load from that path.
        # Passing the repo id makes transformers' tokenizer loader hit the Hub API
        # (is_base_mistral -> model_info), which hangs on a restricted network and
        # errors under HF_HUB_OFFLINE; a local path is treated as offline and skips it.
        processor_dir = resolve_subfolder_dir(self.model_path_hf, "processor", self.cache_dir)
        self.processor = AutoProcessor.from_pretrained(str(processor_dir), trust_remote_code=True)
        self._processor_initialized = True

    def _compute_crop_start(self) -> int:
        if self._crop_start is not None:
            return self._crop_start
        marker = "<|USER_INPUT_MARKER|>"
        marked = PROMPT_TEMPLATE.format(marker)
        marker_pos = marked.find(marker)
        if marker_pos < 0:
            self._crop_start = 0
            return 0
        prefix = self.processor(text=marked[:marker_pos], images=None, videos=None, return_tensors="pt")
        self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    def _prompt_tensors(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_processor()
        text = PROMPT_TEMPLATE.format(prompt)
        inputs = self.processor(
            text=[text],
            images=None,
            videos=None,
            do_resize=False,
            truncation=True,
            max_length=self.config.token_length,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        crop = self._compute_crop_start()
        true_len = int(attention_mask.sum().item())
        return (
            input_ids[:true_len].contiguous(),
            attention_mask[:true_len].contiguous(),
            torch.tensor([crop], dtype=torch.int64),
        )

    def _validate_generation_size(self, model_kwargs: dict) -> None:
        cfg = self.config
        for name, align, default in (
            ("height", cfg.spatial_alignment[0], cfg.default_height),
            ("width", cfg.spatial_alignment[1], cfg.default_width),
        ):
            value = int(model_kwargs.get(name, default))
            if value <= 0 or value % align:
                raise ValueError(f"LingBot {name} must be a positive multiple of {align}; got {value}.")
        frames = int(model_kwargs.get("num_frames", cfg.default_num_frames))
        if frames != 1 and (frames - 1) % cfg.vae_scale_factor_temporal:
            raise ValueError(f"LingBot num_frames must be 1 or 4n+1; got {frames}.")

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if output_modalities and output_modalities[0] != "video":
            raise ValueError(f"LingBot only generates video; got {output_modalities[0]!r}.")
        if input_modalities != ["text"] and set(input_modalities) != {"text"}:
            raise ValueError("LingBot dense text-to-video currently accepts text input only.")
        if prompt is None:
            raise ValueError("LingBot requires a text prompt.")
        self._validate_generation_size(kwargs)
        pos_ids, pos_mask, pos_crop = self._prompt_tensors(prompt)
        neg = kwargs.get("negative_prompt", self.config.default_negative_prompt)
        neg_ids, neg_mask, neg_crop = self._prompt_tensors(neg)
        out: NameToTensorList = {
            "positive_input_ids": [pos_ids],
            "positive_attention_mask": [pos_mask],
            "positive_crop_start": [pos_crop],
            "negative_input_ids": [neg_ids],
            "negative_attention_mask": [neg_mask],
            "negative_crop_start": [neg_crop],
        }
        if kwargs.get("init_latents") is not None:
            out["init_latents"] = [decode_init_latents(kwargs["init_latents"])]
        return out

    def _resolved_kwargs(self, model_kwargs: dict) -> dict:
        requested_steps = int(model_kwargs.get("num_inference_steps", self.config.default_num_inference_steps))
        return {
            "num_inference_steps": max(1, min(requested_steps, self.config.max_denoise_steps)),
            "guidance_scale": float(model_kwargs.get("guidance_scale", self.config.default_guidance_scale)),
            "shift": float(model_kwargs.get("shift", self.config.default_shift)),
            "height": int(model_kwargs.get("height", self.config.default_height)),
            "width": int(model_kwargs.get("width", self.config.default_width)),
            "num_frames": int(model_kwargs.get("num_frames", self.config.default_num_frames)),
        }

    def _get_step_metadata(self, metadata: CurrentForwardConductorMetadata) -> dict:
        return {
            "is_prefill": metadata.is_prefill,
            "num_inference_steps": metadata.kwargs["num_inference_steps"],
            "guidance_scale": metadata.kwargs["guidance_scale"],
            "shift": metadata.kwargs["shift"],
            "height": metadata.kwargs["height"],
            "width": metadata.kwargs["width"],
            "num_frames": metadata.kwargs["num_frames"],
        }

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        kwargs = self._resolved_kwargs(model_kwargs or {})
        kwargs.update(walk_step=0, walk_schedule=[self.ENCODE_TEXT_WALK, self.VIDEO_GEN_WALK])
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=self.ENCODE_TEXT_WALK,
            is_prefill=True,
            kwargs=kwargs,
        )
        inputs = []
        for name in (
            "positive_input_ids",
            "positive_attention_mask",
            "positive_crop_start",
            "negative_input_ids",
            "negative_attention_mask",
            "negative_crop_start",
        ):
            edge = GraphEdge(next_node="text_encoder", name=name)
            edge.tensor_info = input_signals[name]
            inputs.append(edge)
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata=self._get_step_metadata(metadata),
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False
        inputs: list[GraphEdge] = []
        schedule = metadata.kwargs["walk_schedule"]
        step = metadata.kwargs["walk_step"] + 1
        if step < len(schedule):
            metadata.kwargs["walk_step"] = step
            metadata.graph_walk = schedule[step]
            metadata.is_prefill = False
            for name in ("text_embeds_pos", "text_mask_pos", "text_embeds_neg", "text_mask_neg", "init_latents"):
                edge = GraphEdge(next_node="dit", name=name)
                edge.tensor_info = persist_signals.get(name, [])
                inputs.append(edge)
            for name in ("latents", "time_index", "unipc_model_outputs", "unipc_last_sample"):
                inputs.append(GraphEdge(next_node="dit", name=name))
        else:
            request_done = True
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata=self._get_step_metadata(metadata),
            request_done=request_done,
        )

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality != "video":
            raise ValueError(f"Unsupported modality for LingBot: {modality!r}")
        import av

        raw_fps = (request_kwargs or {}).get("fps") or self.config.video_fps
        frames = output[0].permute(1, 2, 3, 0).cpu().numpy()
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=Fraction(float(raw_fps)).limit_denominator(1001))
        stream.height, stream.width = frames.shape[1], frames.shape[2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return buffer.getvalue()

    def get_autocast_dtype(self):
        return None

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None, autocast_dtype=None, sp_group=None
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(self, node_name: str, device: str = "cpu") -> NodeSubmodule | None:
        if self.skip_weight_loading:
            return None
        if node_name == "text_encoder":
            from mstar.model.lingbot.weight_loader import build_lingbot_text_encoder

            text_encoder = build_lingbot_text_encoder(
                self.model_path_hf,
                device=device,
                cache_dir=self.cache_dir,
            )
            return LingBotTextEncoderSubmodule(text_encoder.to(device), self.config)
        if node_name == "dit":
            from mstar.model.lingbot.weight_loader import build_lingbot_transformer

            return LingBotDitSubmodule(
                build_lingbot_transformer(self.model_path_hf, device=device, cache_dir=self.cache_dir),
                self.config,
            )
        if node_name == "vae_decoder":
            from mstar.model.lingbot.weight_loader import build_lingbot_vae_decoder

            vae = build_lingbot_vae_decoder(
                self.model_path_hf,
                device=device,
                cache_dir=self.cache_dir,
            )
            return LingBotVaeDecoderSubmodule(vae, self.config)
        logger.warning("LingBot has no submodule for node %r.", node_name)
        return None
