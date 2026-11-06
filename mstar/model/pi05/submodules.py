"""Vision, prefix and action computations for the Pi0.5 graph."""

import logging
import math
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, Segment, SubmoduleStep
from mstar.model.pi05.components.action_expert import Pi05ActionExpert, Pi05TimeMLP
from mstar.model.pi05.components.flow_matching import sincos_timestep_embedding
from mstar.model.pi05.components.paligemma import Pi05PaliGemmaExpert
from mstar.model.pi05.components.siglip import Pi05SiglipEncoder
from mstar.model.pi05.config import LLM_ATTN, LLM_KV, LLM_POS, Pi05Config
from mstar.model.pi05.kernels.image_normalize import normalize_float_images
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)


class Pi05ViTEncoderSubmodule(NodeSubmodule):
    """SigLIP encoder for camera images.

    Receives one ``(3, H, W)`` image per camera in ``image_inputs``, up to
    ``num_cameras`` of them. Resizes/normalizes each to the SigLIP input format,
    runs SigLIP over ``num_cameras`` slots, and emits ``img_emb`` of shape
    ``(cameras_sent * tokens_per_image, hidden_size)`` for the prefix node.
    """

    disable_autocast = True

    def __init__(self, encoder: Pi05SiglipEncoder, config: Pi05Config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def to(self, *args, **kwargs):
        """Override ``to()`` to always keep the SigLIP vision tower and
        connector in float32, matching lerobot's
        ``to_bfloat16_for_selected_params`` which explicitly preserves
        ``vision_tower`` and ``multi_modal_projector`` in fp32.

        Running SigLIP in bf16 produces ~64 abs delta on the image features
        (27-layer vision transformer accumulates bf16 rounding), which then
        propagates through the prefix KV cache and causes ~0.2 abs delta on
        the final actions — too large for production use.
        """
        # Move to device but FORCE fp32 for all parameters.
        # First apply the standard .to() for device placement:
        result = super().to(*args, **kwargs)
        # Then upcast all parameters back to fp32:
        for param in result.parameters():
            param.data = param.data.to(torch.float32)
        for buf in result.buffers():
            if buf.is_floating_point():
                buf.data = buf.data.to(torch.float32)
        return result

    def _prepare_one(self, images: torch.Tensor) -> torch.Tensor:
        """Resize one request's stack of camera images with aspect-preserving
        letterbox padding.

        Matches openpi's ``image_tools.resize_with_pad_torch`` exactly:
        the longer dimension is scaled to the target size, the shorter
        dimension is scaled proportionally, and the result is padded with
        ``-1`` (the float32 normalized "black" value) to reach the target
        resolution.

        Accepted input encodings (auto-detected by dtype + value range):
          * ``uint8`` in ``[0, 255]`` — what ``Pi05Model.load_image`` hands over
          * ``float`` in ``[0, 1]`` — the base ``Model.load_image`` encoding
          * ``float`` in ``[-1, 1]`` — already-normalized form used by the
            unit/integration tests that bypass the data_worker
        Anything else falls back to a simple float32 cast and assumes the
        caller knows what they're doing.
        """
        if images.dim() == 3:
            # [C, H, W] -- single camera; add a leading camera dim.
            images = images.unsqueeze(0)
        if images.dim() != 4:
            raise ValueError(
                f"Expected images shape [num_cameras, C, H, W], got {tuple(images.shape)}"
            )

        if images.dtype == torch.uint8:
            # uint8 [0, 255] → float32 [-1, 1]
            images = images.to(torch.float32) / 127.5 - 1.0
        else:
            # normalize_float_images detects [0,1] vs [-1,1] and rescales
            # entirely on the GPU — no CPU–GPU sync (replaces the two
            # float(images.min()) / float(images.max()) calls that were here).
            images = normalize_float_images(images.to(torch.float32))

        target_h = target_w = self.config.vit_image_size
        _, _, cur_h, cur_w = images.shape

        if (cur_h, cur_w) == (target_h, target_w):
            return images.clamp(-1.0, 1.0)

        # Aspect-preserving resize: scale by max(cur/target).
        ratio = max(cur_w / target_w, cur_h / target_h)
        resized_h = int(cur_h / ratio)
        resized_w = int(cur_w / ratio)
        resized = nn.functional.interpolate(
            images, size=(resized_h, resized_w), mode="bilinear", align_corners=False
        ).clamp(-1.0, 1.0)

        # Symmetric pad with -1.0 (float32 "black" in the [-1, 1] convention).
        pad_h0, rem_h = divmod(target_h - resized_h, 2)
        pad_h1 = pad_h0 + rem_h
        pad_w0, rem_w = divmod(target_w - resized_w, 2)
        pad_w1 = pad_w0 + rem_w
        return nn.functional.pad(
            resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=-1.0
        )

    def can_batch(
        self,
        batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        """Batch when all requests share the same number of cameras (``prepare_inputs``
        pads every request to ``num_cameras``)."""
        if not model_inputs:
            return False
        first_num_cameras = model_inputs[0].tensor_inputs["pixel_values"].shape[0]
        return all(
            inp.tensor_inputs["pixel_values"].shape[0] == first_num_cameras
            for inp in model_inputs
        )

    def get_accelerator_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list:
        """CUDA graph capture config for the SigLIP encoder.

        Captures the batched encoder forward for bs = 1 during the
        'action_gen' walk. Each capture slot holds one request's pixel_values:
        (num_cameras, 3, H, W). preprocess() stacks them to (bs, num_cameras,
        3, H, W) so shape[0] == bs, satisfying StatelessCudaGraphRunner's
        leading-dim == bs requirement.

        compile=False because warmup() already applies torch.compile to
        forward_batched; _capture_one captures the compiled callable directly.
        """
        H = W = self.config.vit_image_size
        num_cameras = self.config.num_cameras
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="action_gen",
                single_request_inputs=ARNodeInputs(
                    input_seq_len=0,  # not used by StatelessCudaGraphRunner
                    tensor_inputs={
                        "pixel_values": torch.zeros(
                            num_cameras, 3, H, W,
                            device=device, dtype=torch.float32,
                        )
                    },
                ),
                capture_batch_sizes=[1],
                compile=False,
            )
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        # Each camera is letterboxed on its own, since cameras may differ in resolution.
        # Free camera slots get blank frames whose tokens postprocess drops, the same as
        # openpi's image mask (masked tokens get no attention and no positions). More
        # images than slots never get here, max_input_files refuses them.
        pixels = torch.cat([self._prepare_one(image) for image in inputs["image_inputs"]])
        num_images, missing = pixels.shape[0], self.config.num_cameras - pixels.shape[0]
        if missing > 0:
            pixels = torch.cat([pixels, pixels.new_full((missing, *pixels.shape[1:]), -1.0)])
        return NodeInputs(tensor_inputs={"pixel_values": pixels}, kwargs={"num_images": num_images})

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: NodeInputs | None = None,
        **kwargs,
    ):
        """Drop the tokens of the blank frames ``prepare_inputs`` padded the request with,
        so the prefix holds the images the KV layout was counted for."""
        cameras, num_images = self.config.num_cameras, inputs.kwargs["num_images"]
        if num_images < cameras:
            emb = outputs["img_emb"][0]
            outputs["img_emb"] = [emb[: emb.shape[0] // cameras * num_images]]

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # Stack images across requests: (bs, num_cameras, 3, H, W).
        # Leading dim == bs satisfies StatelessCudaGraphRunner's shape validation,
        # and forward_batched flattens it back before the encoder call.
        all_images = [inp.tensor_inputs["pixel_values"] for inp in inputs]
        pixel_values = torch.stack(all_images, dim=0)
        return {"pixel_values": pixel_values}

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        **kwargs # coming from preprocess output
    ) -> NameToTensorList:
        # pixel_values: (1, num_cameras, 3, H, W) from stacked preprocess
        # (sequential path always has bs=1). Flatten to (num_cameras, 3, H, W)
        # before the encoder.
        # Disable autocast so SigLIP runs in fp32, matching lerobot's
        # to_bfloat16_for_selected_params which keeps vision_tower +
        # multi_modal_projector in float32.
        pv = pixel_values.flatten(0, 1)
        features = self.encoder(pv.float())
        # features: [num_cameras, tokens_per_image, hidden]
        # Flatten the camera dimension into the token sequence so the prefix sees
        # a single contiguous sequence of image tokens per request.
        flat = features.reshape(-1, features.shape[-1])
        return {"img_emb": [flat]}

    @torch.amp.autocast("cuda", enabled=False)
    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Batched SigLIP encode for a homogeneous-camera-count batch.

        Args:
            pixel_values: (bs, num_cameras, 3, H, W) — stacked by preprocess().

        Returns:
            Per-request dict: {rid: {"img_emb": [features_for_this_request]}}.
            Each ``img_emb`` tensor has shape (num_cameras * tokens_per_image, hidden_size).
        """
        bs, num_cameras = pixel_values.shape[:2]
        # Flatten to (bs * num_cameras, 3, H, W) for the HF encoder.
        pv = pixel_values.flatten(0, 1)
        features = self.encoder(pv.float())
        # features: (bs * num_cameras, tokens_per_image, hidden_size)
        tokens_per_image, hidden = features.shape[1], features.shape[2]
        # Reshape to (bs, num_cameras * tokens_per_image, hidden_size) so
        # each request gets a single contiguous image-token sequence.
        features = features.reshape(bs, num_cameras * tokens_per_image, hidden)
        return {
            rid: {"img_emb": [features[i]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }


def _keep_norms_float32(module: nn.Module) -> nn.Module:
    # Match lerobot's selective casting: transformer norms stay in fp32.
    for name, param in module.named_parameters():
        if any(part in name for part in ("input_layernorm", "post_attention_layernorm", ".norm.")):
            param.data = param.data.to(torch.float32)
    return module


class Pi05PrefixSubmodule(ARNodeSubmodule):
    """PaliGemma writes the image/text prefix once, then seeds the action Loop."""

    PREFILL_TOKEN_BUCKETS = [512, 1024, 1800]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]

    def __init__(self, embed_tokens: nn.Embedding, paligemma: Pi05PaliGemmaExpert, config: Pi05Config):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.paligemma = paligemma
        self.config = config
        self._text_embed_scale = math.sqrt(config.hidden_size)

    def to(self, *args, **kwargs):
        return _keep_norms_float32(super().to(*args, **kwargs))

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        return [PackedAcceleratorGraphConfig(
            compile_mode="max-autotune-no-cudagraphs",
            capture_graph_walk="action_gen",
            make_node_input=lambda n: ARNodeInputs(
                input_seq_len=n,
                input_embeds=torch.zeros(n, self.config.hidden_size, device=device),
            ),
            capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
            capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            compile=True,
        )]

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        # Robot state is already part of the text prompt. Image tokens are
        # unscaled; PaliGemma's language embeddings are scaled by sqrt(width).
        img_emb = inputs["img_emb"][0]
        text_emb = self.embed_tokens(inputs["text_inputs"][0]) * self._text_embed_scale
        prefix_emb = torch.cat([img_emb, text_emb], dim=0)
        return ARNodeInputs(input_embeds=prefix_emb, input_seq_len=prefix_emb.shape[0])

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs], **kwargs,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[Segment(rid, "main", inp.input_seq_len) for rid, inp in zip(request_ids, inputs, strict=True)],
            steps={LLM_KV: KVStep(), LLM_ATTN: AttentionStep(causal=False), LLM_POS: PositionStep()},
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        return {"prefix_embs": torch.cat([inp.input_embeds for inp in inputs], dim=0)}

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, prefix_embs: torch.Tensor, **kwargs,
    ) -> NameToTensorList:
        self.paligemma(query_sequence=prefix_embs, label="main")
        return {"noisy_actions": [], "timestep_index": []}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, prefix_embs: torch.Tensor, **kwargs,
    ) -> dict[str, NameToTensorList]:
        self.paligemma(query_sequence=prefix_embs, label="main")
        return {rid: {"noisy_actions": [], "timestep_index": []} for rid in engine_inputs.request_ids}


class Pi05ActionSubmodule(ARNodeSubmodule):
    """One Euler step, reading the frozen prefix written by PaliGemma."""

    ACTION_GEN_CAPTURE_BATCH_SIZES = [1, 2, 4]

    def __init__(
        self, action_expert: Pi05ActionExpert, action_in_proj: nn.Linear,
        action_out_proj: nn.Linear, time_mlp: Pi05TimeMLP, config: Pi05Config,
    ):
        super().__init__()
        self.action_expert = action_expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj
        self.time_mlp = time_mlp
        self.config = config
        self._fraction: torch.Tensor | None = None
        self._time_emb_buffer: torch.Tensor | None = None

    def to(self, *args, **kwargs):
        return _keep_norms_float32(super().to(*args, **kwargs))

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        return [BatchedAcceleratorGraphConfig(
            compile_mode="max-autotune-no-cudagraphs",
            capture_graph_walk="action_gen",
            single_request_inputs=ARNodeInputs(
                input_seq_len=self.config.action_horizon,
                tensor_inputs={
                    "noisy_actions": torch.zeros(self.config.action_horizon, self.config.action_dim, device=device),
                    "ts": torch.zeros(1, device=device, dtype=torch.long),
                },
            ),
            capture_batch_sizes=self.ACTION_GEN_CAPTURE_BATCH_SIZES,
        )]

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        horizon, dim = self.config.action_horizon, self.config.action_dim
        if not inputs.get("noisy_actions"):
            device = self.get_device()
            generator = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
            noisy = torch.randn(horizon, dim, device=device, generator=generator)
            ts = torch.zeros(1, device=device, dtype=torch.long)
        else:
            noisy = inputs["noisy_actions"][0].reshape(horizon, dim)
            ts = inputs["timestep_index"][0].reshape(1)
        return ARNodeInputs(input_seq_len=horizon, tensor_inputs={"noisy_actions": noisy, "ts": ts})

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs], **kwargs,
    ) -> SubmoduleStep:
        # Suffix KV is scratch: every Euler step sees the same prefix and positions.
        return SubmoduleStep(
            segments=[Segment(rid, "main", inp.input_seq_len) for rid, inp in zip(request_ids, inputs, strict=True)],
            steps={
                LLM_KV: KVStep(commit=False),
                LLM_ATTN: AttentionStep(causal=False),
                LLM_POS: PositionStep(advance=tuple(0 for _ in request_ids)),
            },
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        return {
            "noisy_actions": torch.cat([inp.tensor_inputs["noisy_actions"] for inp in inputs]),
            "timestep_index": torch.cat([inp.tensor_inputs["ts"] for inp in inputs]),
            "fraction": self._get_timestep_emb_fraction(),
            "time_emb_buffer": self._get_time_emb_buffer(len(inputs)),
        }

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        noisy_actions: torch.Tensor, timestep_index: torch.Tensor,
        fraction: torch.Tensor, time_emb_buffer: torch.Tensor, **kwargs,
    ) -> NameToTensorList:
        actions, index = self._euler_step(
            noisy_actions, timestep_index, fraction=fraction, time_emb_buffer=time_emb_buffer,
        )
        return {"noisy_actions": [actions], "timestep_index": [index]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        noisy_actions: torch.Tensor, timestep_index: torch.Tensor,
        fraction: torch.Tensor, time_emb_buffer: torch.Tensor, **kwargs,
    ) -> BatchedModelOutput:
        actions, index = self._euler_step(
            noisy_actions, timestep_index, fraction=fraction, time_emb_buffer=time_emb_buffer,
        )
        rows = len(timestep_index)
        return BatchedModelOutput(row_outputs={
            "noisy_actions": actions.view(rows, self.config.action_horizon, -1),
            "timestep_index": index.view(rows, 1),
        })

    def _get_timestep_emb_fraction(self) -> torch.Tensor:
        if self._fraction is not None:
            return self._fraction
        device = self.get_device()
        dim = self.config.action_hidden_size
        half = dim // 2
        # Geometric progression of frequencies between min_period and max_period.
        # Use float64 for the frequency computation to match the openpi reference;
        # bf16 has only ~3 digits of precision and rounds higher-frequency
        # components, which compounds through time_mlp -> adaRMS -> 18 layers.
        self._fraction = torch.linspace(
            0.0, 1.0, half, device=device,
            dtype=torch.float64
        )
        return self._fraction


    def _get_time_emb_buffer(self, bs: int) -> torch.Tensor:
        """Return a pre-allocated slice of shape (bs, action_hidden_size).

        Allocated once at the largest capture batch size (float32, matching
        noisy_actions dtype). sincos_timestep_embedding fully overwrites it
        every step, so no zeroing is needed — torch.empty suffices.
        """
        max_bs = max(self.ACTION_GEN_CAPTURE_BATCH_SIZES)
        if self._time_emb_buffer is None:
            self._time_emb_buffer = torch.empty(
                max_bs, self.config.action_hidden_size,
                device=self.get_device(),
                dtype=torch.float32,
            )
        return self._time_emb_buffer[:bs]


    def _euler_step(
        self,
        noisy_actions: torch.Tensor,
        timestep_index: torch.Tensor,
        fraction: torch.Tensor,
        time_emb_buffer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Euler flow-matching step. Shared by sequential and batched paths.

        Args:
            noisy_actions: [horizon, action_dim] (or [total_horizon, action_dim]
                when batched across multiple requests).
            timestep_index: scalar long.

        Returns:
            (next_actions, next_timestep_index) with same shapes as inputs.
        """
        # noisy_actions: [N * horizon, action_dim]
        # timestep_index: [N]

        config = self.config
        num_steps = config.num_flow_steps

        idx = timestep_index.to(noisy_actions.dtype)
        t = 1.0 - idx / num_steps

        time_emb = sincos_timestep_embedding(
            t,
            dim=config.action_hidden_size,
            fraction=fraction,
            output_buffer=time_emb_buffer,
            min_period=config.timestep_min_period,
            max_period=config.timestep_max_period,
        )
        adarms_cond = self.time_mlp(time_emb)

        suffix = self.action_in_proj(noisy_actions)

        suffix_out = self.action_expert(
            query_sequence=suffix,
            adarms_cond=adarms_cond,
            label="main",
        )

        velocity = self.action_out_proj(suffix_out)
        dt = -1.0 / num_steps
        next_actions = noisy_actions + dt * velocity
        next_index = timestep_index + 1
        return next_actions, next_index
