"""NodeSubmodule wrappers for the Pi0.5 model nodes.

Three submodules:
  Pi05ViTEncoderSubmodule    -- SigLIP vision encoder for camera images.
  Pi05PaligemmaSubmodule     -- PaliGemma prefix expert; prefills and writes the
                                prefix KV cache.
  Pi05ActionExpertSubmodule  -- action expert; reads the frozen prefix KV cache
                                and runs the Euler flow-matching denoising loop.
"""

import logging
import math
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.engine.cuda_graph_config import BasicBatchedCudaGraphConfig, FlashInferPackedCudaGraphConfig
from mstar.model.pi05.components.action_expert import Pi05ActionExpert, Pi05TimeMLP
from mstar.model.pi05.components.flow_matching import sincos_timestep_embedding
from mstar.model.pi05.components.paligemma import Pi05PaliGemmaExpert
from mstar.model.pi05.components.siglip import Pi05SiglipEncoder
from mstar.model.pi05.config import Pi05Config
from mstar.model.pi05.kernels.image_normalize import normalize_float_images
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)


class Pi05ViTEncoderSubmodule(NodeSubmodule):
    """SigLIP encoder for camera images.

    Receives raw image tensors of shape ``(num_cameras, 3, H, W)`` per request
    in ``image_inputs``. Resizes/normalizes them to the SigLIP input format,
    runs SigLIP, and emits ``img_emb`` of shape
    ``(num_cameras * tokens_per_image, hidden_size)`` for the LLM node.
    """

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
          * ``uint8`` in ``[0, 255]`` — typical raw decoded image
          * ``float`` in ``[0, 1]`` — what mstar's ``data_worker`` hands over
            after dividing decoded uint8 frames by 255
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
        batch: NodeBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        """Batch when all requests share the same number of cameras.

        If camera counts differ, fall back to sequential execution.
        """
        if not model_inputs:
            return False
        first_num_cameras = model_inputs[0].tensor_inputs["pixel_values"].shape[0]
        return all(
            inp.tensor_inputs["pixel_values"].shape[0] == first_num_cameras
            for inp in model_inputs
        )

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list:
        """CUDA graph capture config for the SigLIP encoder.

        Captures the batched encoder forward for bs ∈ [1, 2, 4] during the
        'prefill' walk. Each capture slot holds one request's pixel_values:
        (num_cameras, 3, H, W). preprocess() stacks them to (bs, num_cameras,
        3, H, W) so shape[0] == bs, satisfying StatelessCudaGraphRunner's
        leading-dim == bs requirement.
        """
        from mstar.engine.cuda_graph_config import BasicBatchedCudaGraphConfig
        H = W = self.config.vit_image_size
        num_cameras = self.config.num_cameras
        return [
            BasicBatchedCudaGraphConfig(
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
                compile=False, # empircally does better than compile=True for now
            )
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        return NodeInputs(tensor_inputs={"pixel_values": self._prepare_one(
            inputs["image_inputs"][0]
        )})

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


class Pi05PaligemmaSubmodule(ARNodeSubmodule):
    """PaliGemma prefix expert: forwards over the prefix
    ``[image_tokens, language_tokens]`` and writes the KV cache that the
    action expert later reads."""

    # Parameter name fragments whose weights must stay in float32 even when
    # the rest of the model is bf16. Matches lerobot's
    # ``to_bfloat16_for_selected_params`` — keeping norms in fp32 prevents
    # the per-layer precision loss that otherwise compounds across 18 layers
    # and causes ~0.2 abs delta on the final actions.
    _FLOAT32_PARAM_SELECTORS = (
        "input_layernorm",
        "post_attention_layernorm",
        ".norm.",   # final RMSNorm / adaRMS norm
    )

    # For the default image size and a simple text prompt, one request is about 400 tokens
    PREFILL_TOKEN_BUCKETS = [512, 1024, 1800] # 2048 was giving OOM
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]

    def __init__(
        self,
        embed_tokens: nn.Embedding,
        paligemma: Pi05PaliGemmaExpert,
        config: Pi05Config,
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.paligemma = paligemma
        self.config = config
        # lerobot scales images by sqrt(H) but text by H: its
        # embed_language_tokens routes through HF Gemma's
        # GemmaTextScaledWordEmbedding, which already bakes in a sqrt(H) factor,
        # so the effective text scale is sqrt(H)*sqrt(H) = H. Our plain
        # nn.Embedding has no internal scale, so we apply the full H here.
        # Mismatching makes the text prefix ~45x too small and corrupts context.
        self._image_embed_scale = math.sqrt(config.hidden_size)
        self._text_embed_scale = float(config.hidden_size)

    def to(self, *args, **kwargs):
        """Apply standard ``to()`` then upcast norm parameters back to fp32.

        Matches lerobot's ``to_bfloat16_for_selected_params`` which keeps
        ``input_layernorm``, ``post_attention_layernorm``, and ``model.norm``
        in float32 while the rest of the transformer runs in bfloat16.
        """
        result = super().to(*args, **kwargs)
        for name, param in result.named_parameters():
            if any(sel in name for sel in self._FLOAT32_PARAM_SELECTORS):
                param.data = param.data.to(torch.float32)
        return result

    def can_batch(
        self,
        batch: NodeBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str] | None:
        return ["main"]

    def _embed_tokens_scaled(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed_tokens(ids)
        return emb * self._text_embed_scale

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[BasicBatchedCudaGraphConfig | FlashInferPackedCudaGraphConfig]:
        prefill_packed = {
            num_tokens: {
                "prefix_embs": torch.zeros(num_tokens, self.config.hidden_size, device=device)
            }
            for num_tokens in self.PREFILL_TOKEN_BUCKETS
        }
        return [
            FlashInferPackedCudaGraphConfig(
                capture_graph_walk="action_gen",
                packed_seq_len_to_inputs=prefill_packed,
                requires_cfg=False,
                labels=["main"],
                compile=True,
                causal_attention=False,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        return self._prepare_inputs_prefill(
            inputs=inputs,
            fwd_info=fwd_info,
        )

    def _prepare_inputs_prefill(
        self,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        # Prefix layout [image_tokens, language_tokens]. Robot state is not a
        # separate stream — process_prompt already appended it as a decimal
        # suffix on the prompt. Image and text embeds use different scales (see
        # __init__); applying them here is load-bearing.
        img_emb = inputs["img_emb"][0] * self._image_embed_scale
        text_ids = inputs["text_inputs"][0]
        text_emb = self._embed_tokens_scaled(text_ids)
        prefix_emb = torch.cat([img_emb, text_emb], dim=0)
        seq_len = prefix_emb.shape[0]

        return ARNodeInputs(input_embeds=prefix_emb, input_seq_len=seq_len)


    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:

        return self._preprocess_prefill(
            inputs=inputs,
            cache_manager=engine_inputs.cache_manager,
        )

    def _preprocess_prefill(
        self,
        inputs: list[ARNodeInputs],
        cache_manager: BatchedCacheManager,
    ) -> dict[str, torch.Tensor | Any]:
        per_request_seqs = [inp.input_embeds for inp in inputs]
        prefix_embs = torch.cat(per_request_seqs, dim=0)
        seq_lens = [inp.input_seq_len for inp in inputs]

        # Bidirectional attention over the prefix; PaliGemma is a prefix-LM.
        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=False, label="main", dtype=torch.bfloat16
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")
        return {"prefix_embs": prefix_embs}

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs # coming from preprocess output
    ) -> NameToTensorList:
        cache_handle=engine_inputs.cache_manager

        return self._forward_prefill(cache_handle=cache_handle, **kwargs)

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs, # coming from preprocess output
    )  -> dict[str, NameToTensorList]:
        """Batched forward: process all requests in a single transformer pass.

        Called by ``KVCacheEngine._execute_batched`` when ``can_batch()`` returns
        True. ``packed_inputs`` comes from ``preprocess()`` which already
        concatenated per-request tensors and planned attention/RoPE for the
        full batch.
        """

        return self._forward_prefill_batched(
            cache_manager=engine_inputs.cache_manager,
            request_ids=engine_inputs.request_ids,
            **kwargs,
        )


    def _forward_prefill_batched(
        self,
        cache_manager: BatchedCacheManager,
        request_ids: list[str],
        prefix_embs: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Batched prefill: single PaliGemma forward over concatenated prefixes."""
        cache_manager.set_active_label("main")
        self.paligemma(
            query_sequence=prefix_embs,
            cache_handle=cache_manager,
            write_cache=True,
        )
        # Prefill produces no graph-edge outputs.
        return {rid: {} for rid in request_ids}

    def _forward_prefill(
        self,
        prefix_embs: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> NameToTensorList:
        if cache_handle is not None:
            cache_handle.set_active_label("main")
        self.paligemma(
            query_sequence=prefix_embs,
            cache_handle=cache_handle,
            write_cache=True,
        )
        return {}

    def postprocess(self, request_id, request_info, outputs, **kwargs):
        outputs["action_expert_trigger"] = []


class Pi05ActionExpertSubmodule(ARNodeSubmodule):
    """Action expert flow-matching loop.

    Runs all ``num_flow_steps`` Euler denoising steps over the action suffix
    in a single forward, attending to the frozen prefix KV cache that the
    PaliGemma submodule wrote, then emits the denoised action tensor.
    """

    # Parameter name fragments whose weights must stay in float32 even when
    # the rest of the model is bf16. Matches lerobot's
    # ``to_bfloat16_for_selected_params`` — keeping norms in fp32 prevents
    # the per-layer precision loss that otherwise compounds across 18 layers
    # and causes ~0.2 abs delta on the final actions.
    _FLOAT32_PARAM_SELECTORS = (
        "input_layernorm",
        "post_attention_layernorm",
        ".norm.",   # final RMSNorm / adaRMS norm
    )

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

        # Lazily allocated on first action Euler step, sized for the largest
        # captured batch. sincos_timestep_embedding fully overwrites this buffer
        # every step, so torch.empty suffices (no zeroing needed).
        self._fraction: torch.Tensor | None = None
        self._time_emb_buffer: torch.Tensor | None = None

    def to(self, *args, **kwargs):
        """Apply standard ``to()`` then upcast norm parameters back to fp32.

        Matches lerobot's ``to_bfloat16_for_selected_params`` which keeps
        ``input_layernorm``, ``post_attention_layernorm``, and ``model.norm``
        in float32 while the rest of the transformer runs in bfloat16.
        """
        result = super().to(*args, **kwargs)
        for name, param in result.named_parameters():
            if any(sel in name for sel in self._FLOAT32_PARAM_SELECTORS):
                param.data = param.data.to(torch.float32)
        return result

    def can_batch(
        self,
        batch: NodeBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str] | None:
        return ["main"]

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

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[BasicBatchedCudaGraphConfig | FlashInferPackedCudaGraphConfig]:
        # Visibility check: log the shape that's about to be captured so it's
        # easy to confirm yaml-level Pi05Config overrides (e.g. action_horizon
        # for the DROID variant) flowed all the way through. The values here
        # are read directly from self.config — same source as the nn.Linear
        # weight shapes — so they're guaranteed consistent.
        logger.info(
            "Pi05ActionExpertSubmodule.get_cuda_graph_configs: capturing 'action_gen' "
            "graph with input_seq_len=%d, noisy_actions=(%d, %d), batch_sizes=[1,2,4] "
            "(num_flow_steps=%d denoising iters runs INSIDE this captured graph; "
            "denoising count is independent of horizon)",
            self.config.action_horizon,
            self.config.action_horizon, self.config.action_dim,
            self.config.num_flow_steps,
        )
        return [
            # Action generation always has latents of the same size, so it is a similar
            # paradigm to AR decode and can use the batched cuda graphs
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="action_gen", requires_cfg=False, labels=["main"],
                single_request_inputs=ARNodeInputs(
                    input_seq_len=self.config.action_horizon,
                    tensor_inputs={
                        "noisy_actions": torch.zeros(
                            self.config.action_horizon, self.config.action_dim, device=device
                        ),
                        "ts": torch.zeros(1, device=device, dtype=torch.long)
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
        return self._prepare_inputs_action_gen(
            inputs=inputs,
            fwd_info=fwd_info,
        )

    def _prepare_inputs_action_gen(
        self,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        device = self.get_device()
        action_horizon = self.config.action_horizon
        action_dim = self.config.action_dim

        generator = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
        noisy = torch.randn(
            action_horizon, action_dim, device=device, generator=generator
        )
        ts = torch.zeros(1, device=device, dtype=torch.long)

        seq_len = action_horizon
        return ARNodeInputs(
            input_seq_len=seq_len,
            tensor_inputs={
                "noisy_actions": noisy,
                "ts": ts
            }
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        return self._preprocess_action_gen(
            inputs=inputs,
            cache_manager=engine_inputs.cache_manager,
        )

    def _preprocess_action_gen(
        self,
        inputs: list[ARNodeInputs],
        cache_manager: BatchedCacheManager,
    ) -> dict[str, torch.Tensor | Any]:
        seq_lens = [inp.input_seq_len for inp in inputs]

        # The action suffix attends to the frozen prefix KV cache. We pass
        # write_store=False so the cache is read-only during all 10 iterations.
        cache_manager.plan_attention(
            seq_lens=seq_lens,
            is_causal=False,
            label="main",
            write_store=False,
            dtype=torch.bfloat16
        )
        cache_manager.plan_rope(
            seq_lens=seq_lens, pos_ids=None, label="main"
        )

        # Concatenate noisy_actions across requests for a single forward.
        cat_noisy = torch.cat(
            [inp.tensor_inputs["noisy_actions"] for inp in inputs],
            dim=0
        ) # [N * horizon, action_dim]

        all_ts = torch.cat(
            [inp.tensor_inputs["ts"] for inp in inputs],
            dim=0
        )

        return {
            "noisy_actions": cat_noisy,
            "timestep_index": all_ts,
            "seq_lens": seq_lens,
            "fraction": self._get_timestep_emb_fraction(),
            "time_emb_buffer": self._get_time_emb_buffer(len(inputs))
        }

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs # coming from preprocess output
    ) -> NameToTensorList:
        cache_handle=engine_inputs.cache_manager
        return self._forward_action_gen(cache_handle=cache_handle, **kwargs)

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs, # coming from preprocess output
    )  -> dict[str, NameToTensorList]:
        return self._forward_action_gen_batched(
            cache_manager=engine_inputs.cache_manager,
            request_ids=engine_inputs.request_ids,
            **kwargs,
        )

    def _forward_action_gen_batched(
        self,
        cache_manager: BatchedCacheManager,
        request_ids: list[str],
        noisy_actions: torch.Tensor,
        timestep_index: torch.Tensor,
        fraction: torch.Tensor,
        time_emb_buffer: torch.Tensor,
        **kwargs
    ) -> dict[str, NameToTensorList]:
        """Batched action_gen: single action expert forward, then split per-request."""

        horizon = self.config.action_horizon

        for _ in range(self.config.num_flow_steps):
            noisy_actions, timestep_index = self._euler_step(
                noisy_actions, timestep_index,
                fraction=fraction,
                time_emb_buffer=time_emb_buffer,
                cache_handle=cache_manager
            )

        # Split back per-request by horizon.
        result: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            start = i * horizon
            end = start + horizon
            result[rid] = {
                "actions": [noisy_actions[start:end]],
            }
        return result

    def _forward_action_gen(
        self,
        noisy_actions,
        timestep_index,
        fraction,
        time_emb_buffer,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> NameToTensorList:
        """Single-request action_gen forward (called from _execute_sequential).

        ``noisy_actions`` and ``timestep_index`` arrive as single-element
        lists from preprocess (to keep the data structure uniform with the
        batched path). We unpack the first element, run the full
        ``num_flow_steps`` Euler denoising loop, and return the denoised action
        tensor.
        """
        # Unpack from list form (preprocess always returns lists now).
        if isinstance(noisy_actions, list):
            noisy_actions = noisy_actions[0]
        if isinstance(timestep_index, list):
            timestep_index = timestep_index[0]

        for _ in range(self.config.num_flow_steps):
            noisy_actions, timestep_index = self._euler_step(
                noisy_actions, timestep_index,
                fraction=fraction,
                time_emb_buffer=time_emb_buffer,
                cache_handle=cache_handle
            )
        return {
            "actions": [noisy_actions],
        }

    def _euler_step(
        self,
        noisy_actions: torch.Tensor,
        timestep_index: torch.Tensor,
        fraction: torch.Tensor,
        time_emb_buffer: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Euler flow-matching step. Shared by sequential and batched paths.

        Args:
            noisy_actions: [horizon, action_dim] (or [total_horizon, action_dim]
                when batched across multiple requests).
            timestep_index: scalar long.
            cache_handle: BatchedCacheManager with attention already planned.

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

        if cache_handle is not None:
            cache_handle.set_active_label("main")
        suffix_out = self.action_expert(
            query_sequence=suffix,
            cache_handle=cache_handle,
            adarms_cond=adarms_cond,
        )

        velocity = self.action_out_proj(suffix_out)
        dt = -1.0 / num_steps
        next_actions = noisy_actions + dt * velocity
        next_index = timestep_index + 1
        return next_actions, next_index
