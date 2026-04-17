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
from mminf.model.pi05.components.action_expert import Pi05ActionExpert, Pi05TimeMLP
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

    def _preprocess_one(self, images: torch.Tensor) -> torch.Tensor:
        """Resize one request's stack of camera images with aspect-preserving
        letterbox padding.

        Matches openpi's ``image_tools.resize_with_pad_torch`` exactly:
        the longer dimension is scaled to the target size, the shorter
        dimension is scaled proportionally, and the result is padded with
        ``-1`` (the float32 normalized "black" value) to reach the target
        resolution.

        Accepted input encodings (auto-detected by dtype + value range):
          * ``uint8`` in ``[0, 255]`` — typical raw decoded image
          * ``float`` in ``[0, 1]`` — what mminf's ``data_worker`` hands over
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
            images = images.to(torch.float32)
            # The data_worker passes images as float32 in [0, 1] (after
            # dividing decoded uint8 frames by 255). Detect that case and
            # remap to [-1, 1] so SigLIP sees the openpi-normalized range.
            # If the image is already in [-1, 1] (e.g. when test code feeds
            # the submodule directly), the min will be < 0 and we leave it
            # alone.
            if images.numel() > 0:
                img_min = float(images.min())
                img_max = float(images.max())
                if img_min >= -1e-4 and img_max <= 1.0 + 1e-4:
                    images = images * 2.0 - 1.0

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

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        # Disable autocast so SigLIP runs in fp32, matching lerobot's
        # to_bfloat16_for_selected_params which keeps vision_tower +
        # multi_modal_projector in float32.
        features = self.encoder(pixel_values.float())
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

    def __init__(
        self,
        embed_tokens: nn.Embedding,
        paligemma: Pi05PaliGemmaExpert,
        action_expert: Pi05ActionExpert,
        action_in_proj: nn.Linear,
        action_out_proj: nn.Linear,
        time_mlp: Pi05TimeMLP,
        config: Pi05Config,
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.paligemma = paligemma
        self.action_expert = action_expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj
        self.time_mlp = time_mlp
        self.config = config
        # Image features and language token embeddings use DIFFERENT scaling
        # factors in lerobot's reference, even though both end up calling it
        # ``sqrt(hidden_size)``:
        #
        #   * Images: ``embed_image`` returns
        #     ``connector(siglip_features) * sqrt(hidden_size)``  -> scale = sqrt(H).
        #
        #   * Text: lerobot's ``lang_embed_func`` does
        #     ``embed_language_tokens(tokens) * sqrt(hidden_size)``, but
        #     ``embed_language_tokens`` calls HF Gemma's
        #     ``GemmaTextScaledWordEmbedding`` whose ``forward`` already
        #     multiplies the raw lookup by an internal ``embed_scale =
        #     sqrt(hidden_size)``. So the EFFECTIVE text scale is
        #     ``sqrt(H) * sqrt(H) = H``, not ``sqrt(H)``.
        #
        # We use a plain ``nn.Embedding`` for ``embed_tokens`` (no internal
        # scale), so we have to apply the full ``H`` factor manually here.
        # Mismatching this produces a ~45x undersized text prefix and the
        # action expert sees a wildly wrong context.
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

    def can_batch(self, batch) -> bool:
        """Pi0.5 supports batched execution for both graph walks.

        - ``prefill``: prefix embeddings are concatenated across requests and
          processed in a single PaliGemma forward with batched FlashInfer
          attention. Each request can have a different prefix length (different
          text prompt lengths).
        - ``action_gen``: all requests in a batch are at the same Euler
          iteration (guaranteed by the Loop primitive), so their suffix tokens
          can be concatenated and processed in a single action expert forward.

        NOTE: Batching is disabled for now pending investigation of a
        deadlock in the batched prefill path. The forward_batched()
        implementation is ready and tested in-process; the issue is in
        the interaction with the AREngine's _execute_batched flow.
        """
        # TODO: re-enable once the deadlock is resolved.
        return False

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
        return emb * self._text_embed_scale

    def _preprocess_prefill(
        self,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
    ) -> dict[str, torch.Tensor]:
        # Pi0.5 prefix layout (matches lerobot's embed_prefix):
        #   [image_tokens, language_tokens]
        # The robot state is *not* a separate token stream — it has already
        # been formatted as a decimal-string suffix on the language prompt
        # by ``Pi05Model.process_prompt``, then tokenized by the PaliGemma
        # tokenizer. So the LLM only consumes ``img_emb`` + ``text_inputs``.
        #
        # IMPORTANT: lerobot's ``embed_prefix`` scales BOTH the image features
        # (after the multi_modal_projector) and the language token embeddings
        # by ``sqrt(hidden_size)``. We mirror that here. Without the image
        # scaling the SigLIP tokens come in ~sqrt(2048)≈45x too small relative
        # to the language tokens and the action expert sees a wildly wrong
        # prefix. (The standalone test_pi05_model_loaded_via_remapper_matches_
        # lerobot integration test missed this because it bypasses
        # _preprocess_prefill and feeds in lerobot's pre-scaled embed_prefix
        # output directly.)
        per_request_seqs = []
        for inp in per_request_inputs:
            img_emb = inp["img_emb"][0] * self._image_embed_scale
            text_ids = inp["text_inputs"][0]
            text_emb = self._embed_tokens_scaled(text_ids)
            per_request_seqs.append(torch.cat([img_emb, text_emb], dim=0))

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
        device = next(self.parameters()).device
        action_horizon = self.config.action_horizon
        action_dim = self.config.action_dim

        all_noisy = []
        all_ts = []
        for rid, inputs in zip(request_ids, per_request_inputs, strict=True):
            info = per_request_info[rid]
            if "noisy_actions" not in inputs or len(inputs["noisy_actions"]) == 0:
                generator = torch.Generator(device=device).manual_seed(info.random_seed)
                noisy = torch.randn(
                    action_horizon, action_dim, device=device, generator=generator
                )
                ts = torch.zeros((), device=device, dtype=torch.long)
            else:
                noisy = inputs["noisy_actions"][0]
                ts = inputs["timestep_index"][0]
            all_noisy.append(noisy)
            all_ts.append(ts)

        seq_lens = [action_horizon] * len(request_ids)

        # The action suffix attends to the frozen prefix KV cache. We pass
        # write_store=False so the cache is read-only during all 10 iterations.
        cache_manager.plan_attention(
            seq_lens=seq_lens,
            is_causal=False,
            label="main",
            write_store=False,
        )
        cache_manager.plan_rope(
            seq_lens=seq_lens, pos_ids=None, label="main"
        )

        return {
            "noisy_actions": all_noisy,
            "timestep_index": all_ts,
            "seq_lens": seq_lens,
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

    def forward_batched(
        self,
        graph_walk: str,
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
        packed_inputs: dict[str, torch.Tensor],
        per_request_info: dict[str, CurrentForwardPassInfo] | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Batched forward: process all requests in a single transformer pass.

        Called by ``AREngine._execute_batched`` when ``can_batch()`` returns
        True. ``packed_inputs`` comes from ``preprocess()`` which already
        concatenated per-request tensors and planned attention/RoPE for the
        full batch.
        """
        if graph_walk == "prefill":
            return self._forward_prefill_batched(
                cache_manager=cache_manager,
                request_ids=request_ids,
                packed_inputs=packed_inputs,
            )
        if graph_walk == "action_gen":
            return self._forward_action_gen_batched(
                cache_manager=cache_manager,
                request_ids=request_ids,
                packed_inputs=packed_inputs,
            )
        raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")

    def _forward_prefill_batched(
        self,
        cache_manager: BatchedCacheManager,
        request_ids: list[str],
        packed_inputs: dict[str, torch.Tensor],
    ) -> dict[str, NameToTensorList]:
        """Batched prefill: single PaliGemma forward over concatenated prefixes."""
        prefix_embs = packed_inputs["prefix_embs"]
        cache_manager.set_active_label("main")
        self.paligemma(
            query_sequence=prefix_embs,
            cache_handle=cache_manager,
            write_cache=True,
        )
        # Prefill produces no graph-edge outputs.
        return {rid: {} for rid in request_ids}

    def _forward_action_gen_batched(
        self,
        cache_manager: BatchedCacheManager,
        request_ids: list[str],
        packed_inputs: dict[str, torch.Tensor],
    ) -> dict[str, NameToTensorList]:
        """Batched action_gen: single action expert forward, then split per-request."""
        all_noisy: list[torch.Tensor] = packed_inputs["noisy_actions"]
        all_ts: list[torch.Tensor] = packed_inputs["timestep_index"]
        horizon = self.config.action_horizon

        # All requests share the same timestep (Loop iterates in lockstep).
        # Concatenate noisy_actions across requests for a single forward.
        cat_noisy = torch.cat(all_noisy, dim=0)  # [N * horizon, action_dim]
        timestep_index = all_ts[0]  # scalar, same for all

        next_actions, next_index = self._euler_step(
            cat_noisy, timestep_index, cache_manager
        )

        # Split back per-request by horizon.
        result: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            start = i * horizon
            end = start + horizon
            result[rid] = {
                "noisy_actions": [next_actions[start:end]],
                "timestep_index": [next_index],
            }
        return result

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
        noisy_actions,
        timestep_index,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> NameToTensorList:
        """Single-request action_gen forward (called from _execute_sequential).

        ``noisy_actions`` and ``timestep_index`` arrive as single-element
        lists from preprocess (to keep the data structure uniform with the
        batched path). We unpack the first element, run one Euler step, and
        return the loop-back edges.
        """
        # Unpack from list form (preprocess always returns lists now).
        if isinstance(noisy_actions, list):
            noisy_actions = noisy_actions[0]
        if isinstance(timestep_index, list):
            timestep_index = timestep_index[0]

        next_actions, next_index = self._euler_step(
            noisy_actions, timestep_index, cache_handle
        )
        # We ALWAYS return both loop-back edges, even on the final iteration.
        # The Loop primitive (mminf/graph/base.py:Loop) handles the final-iter
        # swap automatically: it matches the section's output ``noisy_actions``
        # to the Loop's terminal output (also named ``noisy_actions``, but
        # routed to EMIT_TO_CLIENT with ``output_modality="action"``) and
        # filters out the ``timestep_index`` loop-back edge. Same convention
        # BAGEL's image_gen uses for ``latents`` / ``time_index``.
        return {
            "noisy_actions": [next_actions],
            "timestep_index": [next_index],
        }

    def _euler_step(
        self,
        noisy_actions: torch.Tensor,
        timestep_index: torch.Tensor,
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
        config = self.config
        num_steps = config.num_flow_steps

        idx = timestep_index.to(noisy_actions.dtype)
        t = 1.0 - idx / num_steps

        time_emb = sincos_timestep_embedding(
            t,
            dim=config.action_hidden_size,
            min_period=config.timestep_min_period,
            max_period=config.timestep_max_period,
        ).squeeze(0)
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
