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
from mstar.engine.state import ArrayStep, Segment, SlotLease, SubmoduleStep
from mstar.model.pi05.components.action_expert import Pi05ActionExpert, Pi05TimeMLP
from mstar.model.pi05.components.flow_matching import (
    euler_step,
    flow_sde_sigmas,
    flow_sde_timesteps,
    gaussian_logprob,
    sample_mean_var,
    sincos_timestep_embedding,
)
from mstar.model.pi05.components.paligemma import Pi05PaliGemmaExpert
from mstar.model.pi05.components.siglip import Pi05SiglipEncoder
from mstar.model.pi05.config import CHUNK_WALK, LLM_KV, ROLLOUT_WALK, Pi05Config
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
                replay_graph_walks=[CHUNK_WALK, ROLLOUT_WALK],
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
        # Free or masked camera slots get blank frames whose tokens postprocess drops,
        # matching openpi's image mask. More images than slots never get here,
        # max_input_files refuses them.
        camera_mask = kwargs.get("camera_mask")
        if camera_mask is None:
            camera_mask = (getattr(fwd_info, "step_metadata", None) or {}).get("camera_mask")
        pixels = torch.cat([self._prepare_one(image) for image in inputs["image_inputs"]])
        num_images = pixels.shape[0]
        missing = self.config.num_cameras - num_images
        if camera_mask is not None:
            if len(camera_mask) != num_images:
                raise ValueError(f"camera_mask length {len(camera_mask)} != num_images {num_images}")
            if not any(camera_mask):
                raise ValueError("camera_mask must enable at least one camera")
            for i, keep in enumerate(camera_mask):
                if not keep:
                    pixels[i].fill_(-1.0)
        if missing > 0:
            pixels = torch.cat([pixels, pixels.new_full((missing, *pixels.shape[1:]), -1.0)])
        kw: dict[str, Any] = {"num_images": num_images}
        if camera_mask is not None:
            kw["camera_mask"] = tuple(bool(m) for m in camera_mask) + (False,) * missing
        return NodeInputs(tensor_inputs={"pixel_values": pixels}, kwargs=kw)

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: NodeInputs | None = None,
        **kwargs,
    ):
        """Drop the tokens of masked or blank frames so the prefix holds only active cameras."""
        cameras = self.config.num_cameras
        if "camera_mask" in inputs.kwargs:
            mask = tuple(bool(m) for m in inputs.kwargs["camera_mask"])
            if len(mask) < cameras:
                mask = mask + (False,) * (cameras - len(mask))
            if not all(mask):
                emb = outputs["img_emb"][0]
                tokens_per_image = emb.shape[0] // cameras
                active = sum(mask)
                if mask == (True,) * active + (False,) * (cameras - active):
                    outputs["img_emb"] = [emb[: tokens_per_image * active]]
                else:
                    outputs["img_emb"] = [
                        torch.cat(
                            [
                                emb[i * tokens_per_image : (i + 1) * tokens_per_image]
                                for i, keep in enumerate(mask)
                                if keep
                            ],
                            dim=0,
                        )
                    ]
            return
        num_images = inputs.kwargs["num_images"]
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


class _PrefillCaptureConfig(PackedAcceleratorGraphConfig):
    """Leaves out buckets longer than a batch of that size can fill, which would
    otherwise be captured and never replayed."""

    def __init__(self, longest_prefix: int, **kwargs):
        super().__init__(**kwargs)
        self.longest_prefix = longest_prefix

    def get_total_tokens(self, bs: int) -> list[int]:
        return [n for n in super().get_total_tokens(bs) if n <= bs * self.longest_prefix]


class Pi05LLMSubmodule(_Pi05ExpertSubmodule):
    """PaliGemma over the prefix ``[image_tokens, language_tokens]``: writes the
    KV cache the action expert reads, and tells it so with ``prefix_done``."""

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
        cfg = self.config
        longest = cfg.num_cameras * cfg.tokens_per_image + cfg.max_lang_tokens
        return [
            _PrefillCaptureConfig(
                longest_prefix=longest,
                # max-autotune until the default compile level (mstar serve -O) is measured on this model
                compile_mode="max-autotune-no-cudagraphs",
                capture_graph_walk=CHUNK_WALK,
                replay_graph_walks=[CHUNK_WALK, ROLLOUT_WALK],
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n,
                    input_embeds=torch.zeros(
                        n, self.config.hidden_size, device=device, dtype=torch.float32
                    ),
                ),
                # Whole requests' longest prefixes, so a step that fits runs captured: an
                # eager prefill costs tens of ms of host time, a replay about 1 ms.
                capture_token_lengths=sorted({512, *(n * longest for n in self.PREFILL_CAPTURE_BATCH_SIZES)}),
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
        """Prefix-LM attention over the packed batch, which keeps the prefix."""
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            state={LLM_KV: ArrayStep()},
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
        self.paligemma(query_sequence=prefix_embs)
        return {"prefix_done": [prefix_embs.new_zeros(1)]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        prefix_embs: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """One PaliGemma forward over the concatenated prefixes."""
        self.paligemma(query_sequence=prefix_embs)
        request_ids = engine_inputs.request_ids
        done = prefix_embs.new_zeros(len(request_ids))
        return {rid: {"prefix_done": [done[i:i + 1]]} for i, rid in enumerate(request_ids)}


class Pi05ActionSubmodule(_Pi05ExpertSubmodule):
    """The action expert: runs every Euler or Flow-SDE step of flow-matching denoising
    in one step, attending to the prefix KV the LLM node committed, and emits ``actions``."""

    ACTION_GEN_CAPTURE_BATCH_SIZES = [1, 2, 4]
    _ROLLOUT_MAGIC = b"MSTARR01"

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

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        return graph_walk == CHUNK_WALK

    def cg_key_info(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
        per_request_input_metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del per_request_input_metadata, kwargs
        if graph_walk == CHUNK_WALK and any(
            (getattr(info, "step_metadata", None) or {}).get("rollout") is not None
            for info in per_request_info.values()
        ):
            return "rollout"
        return None

    def can_batch(
        self,
        batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        del batch
        return bool(model_inputs) and len({bool(inp.kwargs.get("rollout")) for inp in model_inputs}) == 1

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
        rollout = (getattr(fwd_info, "step_metadata", None) or {}).get("rollout")
        if rollout is None:
            return ARNodeInputs(input_seq_len=action_horizon, tensor_inputs={"noisy_actions": noisy})

        num_steps = self.config.num_flow_steps
        sde_step = rollout["sde_step"]
        if rollout["sde_type"] == "flow_ode":
            denoise_ind = int(sde_step) if isinstance(sde_step, int) else 0
        elif isinstance(sde_step, int):
            denoise_ind = int(sde_step)
        else:
            high = (num_steps - 1) if (rollout.get("ignore_last") and num_steps > 1) else num_steps
            cpu_gen = torch.Generator(device="cpu").manual_seed(fwd_info.random_seed)
            denoise_ind = int(torch.randint(0, high, (1,), generator=cpu_gen).item())

        step_noise = torch.randn(
            action_horizon, self.config.action_dim, device=device, dtype=torch.float32, generator=generator
        )
        return ARNodeInputs(
            input_seq_len=action_horizon,
            tensor_inputs={"noisy_actions": noisy, "step_noise": step_noise},
            kwargs={
                "rollout": {
                    **rollout,
                    "denoise_ind": denoise_ind,
                    "seed": int(fwd_info.random_seed),
                }
            },
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        """Adds the action tokens as scratch, so every Euler step reads the same prefix at the same positions."""
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            state={LLM_KV: ArrayStep(keep=False)},
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        del engine_inputs
        out: dict[str, torch.Tensor | Any] = {
            "noisy_actions": torch.cat([inp.tensor_inputs["noisy_actions"] for inp in inputs], dim=0),
            "fraction": self._get_timestep_emb_fraction(),
        }
        if any("rollout" in inp.kwargs for inp in inputs):
            out["step_noise"] = torch.cat([inp.tensor_inputs["step_noise"] for inp in inputs], dim=0)
            out["rollout_configs"] = [inp.kwargs["rollout"] for inp in inputs]
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        noisy_actions: torch.Tensor,
        fraction: torch.Tensor,
        step_noise: torch.Tensor | None = None,
        rollout_configs: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> NameToTensorList:
        del engine_inputs
        if rollout_configs is None:
            return {"actions": [self._denoise(noisy_actions, fraction=fraction)]}
        actions, chain, logprobs, logprobs_mask = self._denoise_rollout(
            noisy_actions, fraction=fraction, step_noise=step_noise, rollout_configs=rollout_configs
        )
        out: NameToTensorList = {"actions": [actions]}
        if rollout_configs[0]["return_state"]:
            out["rollout"] = [self._pack_rollout_state(chain[0], logprobs[0], logprobs_mask[0], rollout_configs[0])]
        return out

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        noisy_actions: torch.Tensor,
        fraction: torch.Tensor,
        step_noise: torch.Tensor | None = None,
        rollout_configs: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """The denoising loop over the concatenated rows, then split per request."""
        horizon = self.config.action_horizon
        if rollout_configs is None:
            actions = self._denoise(noisy_actions, fraction=fraction)
            return {
                rid: {"actions": [actions[i * horizon:(i + 1) * horizon]]}
                for i, rid in enumerate(engine_inputs.request_ids)
            }
        actions, chain, logprobs, logprobs_mask = self._denoise_rollout(
            noisy_actions, fraction=fraction, step_noise=step_noise, rollout_configs=rollout_configs
        )
        out: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(engine_inputs.request_ids):
            entry: NameToTensorList = {"actions": [actions[i * horizon:(i + 1) * horizon]]}
            if rollout_configs[i]["return_state"]:
                entry["rollout"] = [
                    self._pack_rollout_state(chain[i], logprobs[i], logprobs_mask[i], rollout_configs[i])
                ]
            out[rid] = entry
        return out

    def _denoise_rollout(
        self,
        noisy_actions: torch.Tensor,
        fraction: torch.Tensor,
        step_noise: torch.Tensor,
        rollout_configs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ``num_flow_steps`` in fp32 step math with opt-in Flow-SDE and log-prob scoring."""
        config = self.config
        num_steps = config.num_flow_steps
        horizon = config.action_horizon
        action_dim = config.action_dim
        batch_size = len(rollout_configs)
        device = noisy_actions.device

        with torch.autocast(device.type, enabled=False):
            timesteps = flow_sde_timesteps(num_steps, device=device)
            time_emb = sincos_timestep_embedding(
                timesteps[:num_steps],
                dim=config.action_hidden_size,
                fraction=fraction,
                output_buffer=timesteps.new_empty(num_steps, config.action_hidden_size),
                min_period=config.timestep_min_period,
                max_period=config.timestep_max_period,
            )
            modulations = self.action_expert.modulations(self.time_mlp(time_emb))
            noise_levels = torch.tensor(
                [r["noise_level"] for r in rollout_configs], device=device, dtype=torch.float32
            ).view(batch_size, 1)
            is_sde_req = torch.tensor(
                [r["sde_type"] == "flow_sde" for r in rollout_configs], device=device, dtype=torch.bool
            ).view(batch_size, 1, 1)
            denoise_inds = torch.tensor(
                [r["denoise_ind"] for r in rollout_configs], device=device, dtype=torch.long
            ).view(batch_size, 1, 1)
            sigmas = flow_sde_sigmas(timesteps, noise_levels)

        x_t = noisy_actions.to(torch.float32).view(batch_size, horizon, action_dim)
        eps = step_noise.to(torch.float32).view(batch_size, horizon, action_dim)
        chain_steps = [x_t]
        logprobs = torch.zeros(batch_size, horizon, action_dim, device=device, dtype=torch.float32)

        in_weight = getattr(self.action_in_proj, "weight", None)
        out_weight = getattr(self.action_out_proj, "weight", None)

        for step in range(num_steps):
            x_in = x_t.reshape(batch_size * horizon, action_dim)
            if in_weight is not None and x_in.dtype != in_weight.dtype:
                x_in = x_in.to(in_weight.dtype)
            suffix_out = self.action_expert(
                query_sequence=self.action_in_proj(x_in),
                layer_modulations=[mod[step] for mod in modulations.per_layer],
                final_modulation=modulations.final[step],
            )
            if out_weight is not None and suffix_out.dtype != out_weight.dtype:
                suffix_out = suffix_out.to(out_weight.dtype)
            v_t = self.action_out_proj(suffix_out).to(torch.float32).view(batch_size, horizon, action_dim)

            with torch.autocast(device.type, enabled=False):
                t = timesteps[step]
                t_next = timesteps[step + 1]
                is_sde_step = is_sde_req & (denoise_inds == step)
                sigma_i = sigmas[:, step].view(batch_size, 1, 1)
                x_mean, x_std = sample_mean_var(x_t, v_t, t, t_next, sigma_i, is_sde_step)
                x_next = x_mean + x_std * eps
                step_lp = gaussian_logprob(x_next, x_mean, x_std)
                logprobs = logprobs + step_lp
                x_t = x_next
                chain_steps.append(x_t)

        chain = torch.stack(chain_steps, dim=1)
        logprobs_mask = (is_sde_req & (noise_levels.view(batch_size, 1, 1) > 0.0)).expand(
            batch_size, horizon, action_dim
        )
        return x_t.reshape(batch_size * horizon, action_dim), chain, logprobs, logprobs_mask

    def _pack_rollout_state(
        self,
        chain: torch.Tensor,
        logprobs: torch.Tensor,
        logprobs_mask: torch.Tensor,
        rollout_cfg: dict[str, Any],
    ) -> torch.Tensor:
        device = chain.device
        seed_u64 = int(rollout_cfg["seed"]) & ((1 << 64) - 1)
        seed_i64 = seed_u64 - (1 << 64) if seed_u64 >= (1 << 63) else seed_u64
        magic = torch.tensor(list(self._ROLLOUT_MAGIC), dtype=torch.uint8, device=device)
        meta = torch.tensor(
            [
                self.config.num_flow_steps,
                self.config.action_horizon,
                self.config.action_dim,
                int(rollout_cfg["denoise_ind"]),
                seed_i64,
            ],
            dtype=torch.int64,
            device=device,
        ).view(torch.uint8)
        return torch.cat(
            [
                magic,
                meta,
                chain.contiguous().view(torch.uint8).reshape(-1),
                logprobs.contiguous().view(torch.uint8).reshape(-1),
                logprobs_mask.to(torch.uint8).contiguous().view(torch.uint8).reshape(-1),
            ]
        )

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
            )
            actions = euler_step(actions, self.action_out_proj(suffix_out), dt)
        return actions
