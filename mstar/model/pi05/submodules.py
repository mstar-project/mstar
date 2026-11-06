"""NodeSubmodule wrappers for the Pi0.5 model nodes.

Three submodules:
  Pi05ViTEncoderSubmodule -- SigLIP vision encoder for camera images.
  Pi05LLMSubmodule        -- PaliGemma writes the prefix KV cache.
  Pi05ActionSubmodule     -- the action expert reads that KV cache and runs the
                             whole Euler flow-matching loop.
"""

import logging
import math
from collections.abc import Mapping
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
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, Segment, SlotLease, SubmoduleStep
from mstar.model.pi05.components.action_expert import Pi05ActionExpert, Pi05TimeMLP
from mstar.model.pi05.components.flow_matching import euler_step, sincos_timestep_embedding
from mstar.model.pi05.components.paligemma import Pi05PaliGemmaExpert
from mstar.model.pi05.components.siglip import Pi05SiglipEncoder
from mstar.model.pi05.config import CHUNK_WALK, LLM_ATTN, LLM_KV, LLM_POS, Pi05Config
from mstar.model.pi05.kernels.image_normalize import normalize_float_images
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)


class Pi05ViTEncoderSubmodule(NodeSubmodule):
    """SigLIP encoder for camera images.

    Receives one ``(3, H, W)`` image per camera in ``image_inputs``, up to
    ``num_cameras`` of them. Resizes/normalizes each to the SigLIP input format,
    runs SigLIP over ``num_cameras`` slots, and emits ``img_emb`` of shape
    ``(cameras_sent * tokens_per_image, hidden_size)`` for the LLM node.
    """

    disable_autocast = True

    def __init__(self, encoder: Pi05SiglipEncoder, config: Pi05Config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    # Kept in float32, as openpi keeps them (to_bfloat16_for_selected_params
    # in models_pytorch/gemma_pytorch.py); everything else runs in bfloat16.
    _FLOAT32_PARAM_SELECTORS = ("embeddings.patch_embedding.", "embeddings.position_embedding.")

    def to(self, *args, **kwargs):
        """Apply ``to()``, then set openpi's SigLIP precision: bfloat16 except
        the patch and position embeddings."""
        result = super().to(*args, **kwargs)
        for name, param in result.named_parameters():
            keep = any(sel in name for sel in self._FLOAT32_PARAM_SELECTORS)
            param.data = param.data.to(torch.float32 if keep else torch.bfloat16)
        return result

    def _prepare_one(self, images: torch.Tensor) -> torch.Tensor:
        """Resize one request's stack of camera images with aspect-preserving
        letterbox padding.

        As openpi's ``image_tools.resize_with_pad``: the longer dimension is
        scaled to the target size, the shorter one proportionally, and the
        result is padded with ``-1`` (black once normalized). Its linear
        resize (``jax.image.resize``) antialiases when downscaling, so this
        one does too.

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
            images, size=(resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True,
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

        Captures the batched encoder forward at bs 1 during the action chunk
        walk. Each capture slot holds one request's pixel_values:
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
                capture_graph_walk=CHUNK_WALK,
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
        # before the encoder. Autocast is off: the encoder runs in its
        # parameters' own dtypes (see ``to``).
        pv = pixel_values.flatten(0, 1)
        features = self.encoder(pv)
        # features: [num_cameras, tokens_per_image, hidden]
        # Flatten the camera dimension into the token sequence so the LLM sees
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
        features = self.encoder(pv)
        # features: (bs * num_cameras, tokens_per_image, hidden_size)
        tokens_per_image, hidden = features.shape[1], features.shape[2]
        # Reshape to (bs, num_cameras * tokens_per_image, hidden_size) so
        # each request gets a single contiguous image-token sequence.
        features = features.reshape(bs, num_cameras * tokens_per_image, hidden)
        return {
            rid: {"img_emb": [features[i]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }


class _Pi05ExpertSubmodule(ARNodeSubmodule):
    """What the prefix and action experts share: their norms' precision."""

    # Parameter name fragments whose weights stay in float32 while the rest of
    # the node runs in bf16, as in openpi: its to_bfloat16_for_selected_params
    # keeps the norms (whose loss compounds across the 18 layers), and its
    # time MLP is never cast.
    _FLOAT32_PARAM_SELECTORS = (
        "input_layernorm",
        "post_attention_layernorm",
        ".norm.",   # final RMSNorm / adaRMS norm
        "time_mlp.",  # the adaRMS condition
    )

    def to(self, *args, **kwargs):
        """Apply standard ``to()``, then upcast the ``_FLOAT32_PARAM_SELECTORS``
        parameters (the norms and the time MLP) back to fp32."""
        result = super().to(*args, **kwargs)
        for name, param in result.named_parameters():
            if any(sel in name for sel in self._FLOAT32_PARAM_SELECTORS):
                param.data = param.data.to(torch.float32)
        return result

    def can_batch(
        self,
        batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True


class Pi05LLMSubmodule(_Pi05ExpertSubmodule):
    """PaliGemma over the prefix ``[image_tokens, language_tokens]``: writes the
    KV cache the action expert reads, and tells it so with ``prefix_done``."""

    # For the default image size, three cameras and a simple text prompt, one request is about 910 tokens
    PREFILL_TOKEN_BUCKETS = [512, 1024, 1800] # 2048 was giving OOM
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]

    def __init__(self, embed_tokens: nn.Embedding, paligemma: Pi05PaliGemmaExpert, config: Pi05Config):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.paligemma = paligemma
        self.config = config
        # Prefix scaling as openpi's embed_prefix does it
        # (models_pytorch/pi0_pytorch.py): SigLIP tokens unscaled, language
        # embeddings times sqrt(hidden_size). openpi's Gemma skips its own
        # input normalizer, and ``embed_tokens`` here is a plain nn.Embedding.
        self._text_embed_scale = math.sqrt(config.hidden_size)

    def _embed_tokens_scaled(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed_tokens(ids)
        return emb * self._text_embed_scale

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        del tp_world_size
        return [
            PackedAcceleratorGraphConfig(
                # max-autotune until the default compile level (mstar serve -O) is measured on this model
                compile_mode="max-autotune-no-cudagraphs",
                capture_graph_walk=CHUNK_WALK,
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n,
                    input_embeds=torch.zeros(
                        n, self.config.hidden_size, device=device, dtype=torch.float32
                    ),
                ),
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
                compile=True,
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        # Prefix layout [image_tokens, language_tokens], as openpi's
        # embed_prefix. The robot state is part of the language tokens:
        # ``Pi05Model.process_prompt`` writes it into the prompt.
        # The prefix (and so PaliGemma's residual stream) is float32, the dtype
        # the prefill capture stages it in, whatever dtypes the ViT and the
        # embedding table produce.
        img_emb = inputs["img_emb"][0]
        text_ids = inputs["text_inputs"][0]
        text_emb = self._embed_tokens_scaled(text_ids)
        prefix_emb = torch.cat([img_emb.float(), text_emb.float()], dim=0)
        seq_len = prefix_emb.shape[0]

        return ARNodeInputs(input_embeds=prefix_emb, input_seq_len=seq_len)

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        """Bidirectional (prefix-LM) attention over the packed batch; commits the
        prefix KV and advances positions."""
        segments = [
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        return SubmoduleStep(
            segments=segments,
            steps={
                LLM_KV: KVStep(commit=True),
                LLM_ATTN: AttentionStep(causal=False),
                LLM_POS: PositionStep(),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        del engine_inputs
        return {"prefix_embs": torch.cat([inp.input_embeds for inp in inputs], dim=0)}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        prefix_embs: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        del engine_inputs
        self.paligemma(query_sequence=prefix_embs, label="main")
        return {"prefix_done": [prefix_embs.new_zeros(1)]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        prefix_embs: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """One PaliGemma forward over the concatenated prefixes."""
        self.paligemma(query_sequence=prefix_embs, label="main")
        request_ids = engine_inputs.request_ids
        done = prefix_embs.new_zeros(len(request_ids))
        return {rid: {"prefix_done": [done[i:i + 1]]} for i, rid in enumerate(request_ids)}


class Pi05ActionSubmodule(_Pi05ExpertSubmodule):
    """The action expert: runs every Euler step of flow-matching denoising in one
    step, attending to the prefix KV the LLM node committed, and emits ``actions``."""

    ACTION_GEN_CAPTURE_BATCH_SIZES = [1, 2, 4]

    def __init__(
        self,
        action_expert: Pi05ActionExpert,
        action_in_proj: nn.Linear,
        action_out_proj: nn.Linear,
        time_mlp: Pi05TimeMLP,
        config: Pi05Config,
    ):
        super().__init__()
        self.action_expert = action_expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj
        self.time_mlp = time_mlp
        self.config = config
        self._fraction: torch.Tensor | None = None

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

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        del tp_world_size
        return [
            # Action generation always has latents of the same size, so it is a
            # similar paradigm to AR decode and can use the batched cuda graphs.
            BatchedAcceleratorGraphConfig(
                # max-autotune until the default compile level (mstar serve -O) is measured on this model
                compile_mode="max-autotune-no-cudagraphs",
                capture_graph_walk=CHUNK_WALK,
                single_request_inputs=ARNodeInputs(
                    input_seq_len=self.config.action_horizon,
                    tensor_inputs={
                        "noisy_actions": torch.zeros(
                            self.config.action_horizon, self.config.action_dim, device=device
                        ),
                    }
                ),
                capture_batch_sizes=self.ACTION_GEN_CAPTURE_BATCH_SIZES
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        del inputs  # ``prefix_done`` only orders this step after the prefix's KV write
        device = self.get_device()
        action_horizon = self.config.action_horizon
        generator = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
        noisy = torch.randn(
            action_horizon, self.config.action_dim, device=device, generator=generator
        )
        return ARNodeInputs(input_seq_len=action_horizon, tensor_inputs={"noisy_actions": noisy})

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        """Reads the frozen prefix: the action tokens are written to scratch pages
        and attended, but neither the stored length nor the position counter
        advances, so every Euler step uses the same prefix and action positions."""
        segments = [
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        return SubmoduleStep(
            segments=segments,
            steps={
                LLM_KV: KVStep(commit=False),
                LLM_ATTN: AttentionStep(causal=False),
                LLM_POS: PositionStep(advance=tuple(0 for _ in request_ids)),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        del engine_inputs
        return {
            "noisy_actions": torch.cat([inp.tensor_inputs["noisy_actions"] for inp in inputs], dim=0),
            "fraction": self._get_timestep_emb_fraction(),
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        noisy_actions: torch.Tensor,
        fraction: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        del engine_inputs
        return {"actions": [self._denoise(noisy_actions, fraction=fraction)]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        noisy_actions: torch.Tensor,
        fraction: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """The denoising loop over the concatenated rows, then split per request."""
        horizon = self.config.action_horizon
        actions = self._denoise(noisy_actions, fraction=fraction)
        return {
            rid: {"actions": [actions[i * horizon:(i + 1) * horizon]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def _denoise(self, noisy_actions: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
        """Run all ``num_flow_steps`` Euler steps from t=1 to t=0.

        The adaRMS conditioning depends only on the step, so the time
        embedding, ``time_mlp`` and every norm's modulation are computed for
        all steps at once, in float32 as openpi computes them, and shared by
        every request in the batch.

        Args:
            noisy_actions: [N * horizon, action_dim], the initial noise.

        Returns:
            The denoised actions, same shape as ``noisy_actions``.
        """
        config = self.config
        num_steps = config.num_flow_steps
        dt = -1.0 / num_steps
        with torch.autocast(noisy_actions.device.type, enabled=False):
            steps = torch.arange(num_steps, device=noisy_actions.device, dtype=torch.float32)
            time_emb = sincos_timestep_embedding(
                1.0 - steps / num_steps,
                dim=config.action_hidden_size,
                fraction=fraction,
                output_buffer=steps.new_empty(num_steps, config.action_hidden_size),
                min_period=config.timestep_min_period,
                max_period=config.timestep_max_period,
            )
            modulations = self.action_expert.modulations(self.time_mlp(time_emb))
        actions = noisy_actions
        for step in range(num_steps):
            suffix_out = self.action_expert(
                query_sequence=self.action_in_proj(actions),
                layer_modulations=[mod[step] for mod in modulations.per_layer],
                final_modulation=modulations.final[step],
                label="main",
            )
            actions = euler_step(actions, self.action_out_proj(suffix_out), dt)
        return actions
