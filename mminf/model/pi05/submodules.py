"""NodeSubmodule wrappers for the Pi0.5 model nodes.

Two submodules:
  Pi05ViTEncoderSubmodule -- SigLIP vision encoder for camera images.
  Pi05LLMSubmodule        -- combined PaliGemma + action expert. Dispatches by
                             graph_walk between prefill (PaliGemma writes the
                             prefix KV cache) and action_gen (action expert
                             reads the frozen prefix KV cache and runs one
                             Euler flow-matching step).
"""

import logging
import math

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.base import NodeSubmodule
from mminf.model.pi05.components.action_expert import Pi05ActionExpert, Pi05AdaLNMLP
from mminf.model.pi05.components.flow_matching import sincos_timestep_embedding
from mminf.model.pi05.components.paligemma import Pi05PaliGemmaExpert
from mminf.model.pi05.components.siglip import Pi05SiglipEncoder
from mminf.model.pi05.config import Pi05Config

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

    def _preprocess_one(self, images: torch.Tensor) -> torch.Tensor:
        """Resize / normalize one request's stack of camera images.

        Pi0.5 expects 224x224 images normalized to [-1, 1]. We do a simple
        resize without aspect-ratio preservation; the openpi reference uses
        zero-padding letterboxing, but for inference both work as long as the
        client preprocesses consistently.
        """
        if images.dim() == 3:
            # [C, H, W] -- single camera; add a leading camera dim.
            images = images.unsqueeze(0)
        if images.dim() != 4:
            raise ValueError(
                f"Expected images shape [num_cameras, C, H, W], got {tuple(images.shape)}"
            )
        target = self.config.vit_image_size
        if images.shape[-2] != target or images.shape[-1] != target:
            images = nn.functional.interpolate(
                images.float(), size=(target, target), mode="bilinear", align_corners=False
            )
        else:
            images = images.float()
        # Normalize uint8/[0,255] or [0,1] to [-1, 1]. Detect range heuristically.
        if images.max() > 1.5:
            images = images / 127.5 - 1.0
        else:
            images = images * 2.0 - 1.0
        return images

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
        cache_manager: BatchedCacheManager | None = None,
    ) -> dict[str, torch.Tensor]:
        # Stack images across requests into a single batch dimension.
        all_images = []
        for inp in per_request_inputs:
            images = inp["image_inputs"][0]
            all_images.append(self._preprocess_one(images))
        pixel_values = torch.cat(all_images, dim=0)
        return {"pixel_values": pixel_values}

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        features = self.encoder(pixel_values)
        # features: [num_cameras_total, tokens_per_image, hidden]
        # Flatten the camera dimension into the token sequence so the LLM sees
        # a single contiguous sequence of image tokens per request.
        flat = features.reshape(-1, features.shape[-1])
        return {"img_emb": [flat]}


class Pi05LLMSubmodule(NodeSubmodule):
    """Combined PaliGemma prefix expert + action expert.

    Dispatches by graph_walk:
      - ``prefill``:    PaliGemma forwards over the prefix
                        ``[image_tokens, language_tokens, state_tokens]`` and
                        writes the KV cache.
      - ``action_gen``: action expert runs one Euler step of flow-matching
                        denoising over the action suffix, attending to the
                        frozen prefix KV cache. The current ``noisy_actions``
                        and ``timestep_index`` cycle through the loop via
                        loop-back graph edges; on the final iteration the
                        denoised action tensor is emitted as ``action_output``.
    """

    def __init__(
        self,
        embed_tokens: nn.Embedding,
        paligemma: Pi05PaliGemmaExpert,
        action_expert: Pi05ActionExpert,
        action_in_proj: nn.Linear,
        action_out_proj: nn.Linear,
        adaln_mlp: Pi05AdaLNMLP,
        config: Pi05Config,
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.paligemma = paligemma
        self.action_expert = action_expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj
        self.adaln_mlp = adaln_mlp
        self.config = config
        self._embed_scale = math.sqrt(config.hidden_size)

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str] | None:
        return ["main"]

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
        cache_manager: BatchedCacheManager,
    ) -> dict[str, torch.Tensor]:
        if graph_walk == "prefill":
            return self._preprocess_prefill(
                per_request_inputs=per_request_inputs,
                request_ids=request_ids,
                cache_manager=cache_manager,
            )
        if graph_walk == "action_gen":
            return self._preprocess_action_gen(
                per_request_inputs=per_request_inputs,
                request_ids=request_ids,
                per_request_info=per_request_info,
                cache_manager=cache_manager,
            )
        raise ValueError(f"Unknown Pi0.5 LLM graph walk: {graph_walk!r}")

    def _embed_tokens_scaled(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed_tokens(ids)
        return emb * self._embed_scale

    def _preprocess_prefill(
        self,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
    ) -> dict[str, torch.Tensor]:
        per_request_seqs = []
        for inp in per_request_inputs:
            img_emb = inp["img_emb"][0]
            text_ids = inp["text_inputs"][0]
            state_ids = inp["state_inputs"][0]
            text_emb = self._embed_tokens_scaled(text_ids)
            state_emb = self._embed_tokens_scaled(state_ids)
            per_request_seqs.append(torch.cat([img_emb, text_emb, state_emb], dim=0))

        seq_lens = [seq.shape[0] for seq in per_request_seqs]
        prefix_embs = torch.cat(per_request_seqs, dim=0)

        # Bidirectional attention over the prefix; PaliGemma is a prefix-LM.
        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=False, label="main"
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        return {"prefix_embs": prefix_embs, "seq_lens": seq_lens}

    def _preprocess_action_gen(
        self,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
        cache_manager: BatchedCacheManager,
    ) -> dict[str, torch.Tensor]:
        if len(per_request_inputs) != 1:
            raise NotImplementedError("Pi0.5 action_gen does not yet support batching")

        request_id = request_ids[0]
        inputs = per_request_inputs[0]
        info = per_request_info[request_id]

        device = next(self.parameters()).device
        action_horizon = self.config.action_horizon
        action_dim = self.config.action_dim

        if "noisy_actions" not in inputs or len(inputs["noisy_actions"]) == 0:
            generator = torch.Generator(device=device).manual_seed(info.random_seed)
            noisy_actions = torch.randn(
                action_horizon, action_dim, device=device, generator=generator
            )
            timestep_index = torch.zeros((), device=device, dtype=torch.long)
        else:
            noisy_actions = inputs["noisy_actions"][0]
            timestep_index = inputs["timestep_index"][0]

        # The action suffix attends to the frozen prefix KV cache. We pass
        # write_store=False so the cache is read-only during all 10 iterations.
        cache_manager.plan_attention(
            seq_lens=[action_horizon],
            is_causal=False,
            label="main",
            write_store=False,
        )
        cache_manager.plan_rope(
            seq_lens=[action_horizon], pos_ids=None, label="main"
        )

        return {
            "noisy_actions": noisy_actions,
            "timestep_index": timestep_index,
        }

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        graph_walk: str,
        request_info: CurrentForwardPassInfo,
        cache_handle: BatchedCacheManager | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if graph_walk == "prefill":
            return self._forward_prefill(cache_handle=cache_handle, **kwargs)
        if graph_walk == "action_gen":
            return self._forward_action_gen(cache_handle=cache_handle, **kwargs)
        raise ValueError(f"Unknown Pi0.5 LLM graph walk: {graph_walk!r}")

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

    def _forward_action_gen(
        self,
        noisy_actions: torch.Tensor,
        timestep_index: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> NameToTensorList:
        config = self.config
        num_steps = config.num_flow_steps

        # Map iteration index -> timestep value in [1, 0].
        idx = timestep_index.to(noisy_actions.dtype)
        t = 1.0 - idx / num_steps

        time_emb = sincos_timestep_embedding(
            t,
            dim=config.hidden_size,
            min_period=config.timestep_min_period,
            max_period=config.timestep_max_period,
        ).squeeze(0)
        adaln_params = self.adaln_mlp(time_emb)  # [num_layers, 6, hidden]

        suffix = self.action_in_proj(noisy_actions)  # [horizon, hidden]

        if cache_handle is not None:
            cache_handle.set_active_label("main")
        suffix_out = self.action_expert(
            query_sequence=suffix,
            cache_handle=cache_handle,
            adaln_params=adaln_params,
        )

        velocity = self.action_out_proj(suffix_out)  # [horizon, action_dim]
        dt = -1.0 / num_steps
        next_actions = noisy_actions + dt * velocity
        next_index = timestep_index + 1

        if int(next_index.item()) >= num_steps:
            return {"action_output": [next_actions]}
        return {
            "noisy_actions": [next_actions],
            "timestep_index": [next_index],
        }
