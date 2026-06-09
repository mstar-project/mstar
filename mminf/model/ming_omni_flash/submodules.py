"""mminf engine submodules for Ming-flash-omni-2.0.

Three submodules covering the multimodal-understanding side of the model:

  * ``VisionEncoderSubmodule`` (enc_dec / stateless) — runs Ming's
    Qwen3MoeVisionTransformer + MingVisionProjector, returns
    LLM-space vision embeddings for the Thinker to splice in.

  * ``AudioEncoderSubmodule`` (enc_dec / stateless) — runs
    MingAudioEncoder + MingAudioProjector, returns LLM-space audio
    embeddings (packed across clips).

  * ``BailingMoeV2ThinkerSubmodule`` (AR / KV-cache) — the Ling-2.0
    MoE LLM. Text-only paths are wired today (step 3d–3f); the
    vision/audio prefill paths grow in via this submodule's
    ``prepare_inputs`` dispatch in step 5b.

Reference: mminf's :class:`OrpheusLLMSubmodule`
(`mminf/model/orpheus/submodules.py:20-176`) is the cleanest text-LLM
template; Qwen3-Omni's submodules
(`mminf/model/qwen3_omni/submodules.py`) show the multimodal extensions
and graph-walk dispatch we mirror here.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.kv_store import PositionInfo
from mminf.model.ming_omni_flash.components.model import LingMoeModel
from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig
from mminf.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 1. VisionEncoderSubmodule (stateless enc_dec engine)
# ===================================================================


class VisionEncoderSubmodule(NodeSubmodule):
    """Wraps Ming's Qwen3MoeVisionTransformer + MingVisionProjector.

    Runs once per request (stateless), consumes ``(pixel_values,
    image_grid_thw)`` and produces ``vision_embeds`` already projected
    into the Thinker's hidden space (no further linear on the LLM
    side — Ming applies the projector + L2 norm before splicing,
    mirroring ``modeling_bailingmm2.extract_image_feature``).

    ``deepstack`` features are deliberately NOT plumbed in step 5a:
    the released ckpt sets ``use_deepstack=False`` and the deepstack
    list is not consumed by ``modeling_bailingmm2``'s text-out path.
    If/when we enable deepstack splicing, ``build_vision_encoder``
    grows a ``use_deepstack=True`` flag and this submodule's forward
    will return both tensors.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_projector: nn.Module,
        config: MingFlashOmniModelConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.vision_projector = vision_projector
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        """Pull pixel_values + grid_thw off the conductor's input bundle.

        ``image_grid_thw`` is produced by ``process_prompt`` from the
        HF image processor; for the test path (no processor) a 1-D
        ``[T, H, W]`` tensor also works (we promote it to ``(1, 3)``).
        """
        if "pixel_values" not in inputs or not inputs["pixel_values"]:
            raise ValueError(
                "VisionEncoderSubmodule: missing 'pixel_values' input. "
                "process_prompt must produce this from the image processor."
            )
        pixel_values = inputs["pixel_values"][0]
        grid_thw = inputs.get(
            "image_grid_thw", inputs.get("grid_thw", [None])
        )[0]
        if grid_thw is None:
            raise ValueError(
                "VisionEncoderSubmodule: 'image_grid_thw' is None. "
                "Make sure process_prompt forwarded image_grid_thw from "
                "the HF image processor (BailingMM2Processor)."
            )
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)  # promote to (1, 3)

        return NodeInputs(
            tensor_inputs={
                "pixel_values": pixel_values,
                "grid_thw": grid_thw,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        """Run encoder → projector → L2-norm.

        Ming applies ``F.normalize(image_embeds, dim=-1)`` after the
        projector (``modeling_bailingmm2.extract_image_feature:101``).
        We mirror that so the Thinker sees the same numeric distribution
        the source model produced during training.
        """
        device = pixel_values.device
        logger.debug(
            "VisionEncoder: pixel_values=%s grid_thw=%s",
            tuple(pixel_values.shape), tuple(grid_thw.shape),
        )
        # The Ming encoder accepts a single torch.Tensor of stacked
        # patches; grid_thw selects which positions / images they belong
        # to. ``use_deepstack=False`` so encoder returns a single tensor.
        with torch.no_grad():
            vision_embeds = self.vision_encoder(
                pixel_values.to(device), grid_thw=grid_thw.to(device),
            )
            if isinstance(vision_embeds, tuple):
                # Defensive: if the encoder was built with
                # ``use_deepstack=True``, drop the deepstack list.
                vision_embeds = vision_embeds[0]
            projected = self.vision_projector(vision_embeds)
            projected = torch.nn.functional.normalize(projected, dim=-1)
        return {"vision_embeds": [projected]}


# ===================================================================
# 2. AudioEncoderSubmodule (stateless enc_dec engine)
# ===================================================================


class AudioEncoderSubmodule(NodeSubmodule):
    """Wraps MingAudioEncoder + MingAudioProjector.

    Consumes a list of variable-length mel spectrograms (one per
    audio clip) and produces packed ``audio_embeds`` ready for the
    Thinker to splice. The packed-sequence forward matches the upstream
    encoder ABI (returns ``(packed, cu_seqlens)``); we drop
    ``cu_seqlens`` after the projector chunks the per-clip lengths
    back via ``MingAudioProjector.compute_output_length`` if needed.

    For step 5a the submodule assumes a single audio clip per request
    (the common case for Q&A / TTS / S2S); multi-clip batched audio
    folds in alongside Thinker batching in a later step.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        audio_projector: nn.Module,
        config: MingFlashOmniModelConfig,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_projector = audio_projector
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        """Pull mel features + (optional) per-clip lengths.

        ``audio_features`` is the only required input today. It's
        either ``(n_mels, T)`` for a single clip or ``(B, n_mels, T)``
        for already-batched input. ``audio_seqlens`` (the original
        unpadded length per clip) is optional — when present the
        encoder uses it to skip the padded tail.
        """
        if "audio_features" not in inputs or not inputs["audio_features"]:
            raise ValueError(
                "AudioEncoderSubmodule: missing 'audio_features' input. "
                "process_prompt must produce this from the audio processor."
            )
        audio_features = inputs["audio_features"][0]
        audio_seqlens = inputs.get("audio_seqlens", [None])[0]
        return NodeInputs(
            tensor_inputs={
                "audio_features": audio_features,
                "audio_seqlens": audio_seqlens,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        audio_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Encoder → projector → L2-norm (if config.audio_encoder.norm_query_embeds).

        Mirrors ``modeling_bailingmm2.extract_audio_feature``:
        L2-normalize along the last dim when ``norm_query_embeds`` is
        set in the audio config (true on the released ckpt).
        """
        device = audio_features.device
        # Accept (n_mels, T) for a single clip or (B, n_mels, T) batched.
        if audio_features.dim() == 2:
            mel_list: list[torch.Tensor] = [audio_features.to(device)]
        elif audio_features.dim() == 3:
            mel_list = [audio_features[i].to(device) for i in range(audio_features.shape[0])]
        else:
            raise ValueError(
                f"AudioEncoderSubmodule: expected audio_features of rank 2 or 3, "
                f"got rank {audio_features.dim()} with shape {tuple(audio_features.shape)}"
            )
        # If audio_seqlens is provided, trim the padded tail of each clip
        # before sending it to the encoder so positional embeddings line up.
        if audio_seqlens is not None:
            mel_list = [
                m[..., : int(audio_seqlens[i].item())]
                for i, m in enumerate(mel_list)
            ]

        logger.debug(
            "AudioEncoder: %d clip(s), per-clip mel T=%s",
            len(mel_list), [int(m.shape[-1]) for m in mel_list],
        )
        with torch.no_grad():
            # Packed encoder returns (total_T', n_state), cu_seqlens int32.
            packed, cu_seqlens = self.audio_encoder(mel_list)
            # Projector expects (B, T, audio_dim) shape — feed one clip
            # at a time when there are multiple, then concat.
            projected_chunks: list[torch.Tensor] = []
            seg_starts = cu_seqlens.tolist()
            for i in range(len(seg_starts) - 1):
                seg = packed[seg_starts[i]:seg_starts[i + 1]].unsqueeze(0)  # (1, T_i, n_state)
                # Projector returns (B, llm_dim, T'_i); transpose to (T'_i, llm_dim).
                projected = self.audio_projector(seg).squeeze(0).transpose(0, 1)
                projected_chunks.append(projected)
            audio_embeds = torch.cat(projected_chunks, dim=0)  # (sum T'_i, llm_dim)

            if self.config.audio_encoder.norm_query_embeds:
                audio_embeds = torch.nn.functional.normalize(audio_embeds, dim=-1)

        return {"audio_embeds": [audio_embeds.to(audio_features.dtype)]}


class BailingMoeV2ThinkerSubmodule(ARNodeSubmodule):
    """Text-only thinker submodule for Ming-flash-omni-2.0.

    Two graph walks:
      * ``prefill``: embed text token ids, fill KV cache, sample first
        token's logits.
      * ``decode``: embed the previous token, single-step forward,
        sample next-token logits.

    The submodule does NOT use ``cache_handle.apply_rope`` — Ling-2.0's
    partial 3D ``video_rope`` is applied inline by
    :class:`LingAttention` using the explicit ``position_ids`` argument.
    """

    def __init__(self, model: LingMoeModel, eos_token_id: int = 156895) -> None:
        super().__init__()
        self.model = model
        self.eos_token_id = eos_token_id
        # Stash the embed_tokens / lm_head as direct attributes so the
        # engine's CUDA-graph captures don't reach through .model.
        self.embed_tokens = model.embed_tokens
        self.lm_head = model.lm_head

    # ------------------------------------------------------------------
    # ARNodeSubmodule contract
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},
    ) -> ARNodeInputs:
        """Build per-request ARNodeInputs from the engine-provided tensors.

        ``inputs["text_inputs"]`` is the token-id tensor — either the
        full prompt (prefill) or the single previous token (decode).
        Mirrors :class:`OrpheusLLMSubmodule.prepare_inputs` since the
        Ling thinker also takes packed token ids.
        """
        token_ids = inputs["text_inputs"][0]
        return ARNodeInputs(
            input_ids=token_ids,
            input_seq_len=token_ids.shape[0],
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        """Plan attention for the engine; pack token ids for forward.

        Single-request only in step 3d; batched preprocess folds in
        step 3e+ via ``can_batch`` + ``forward_batched``.
        """
        if len(inputs) > 1:
            raise NotImplementedError(
                f"BailingMoeV2ThinkerSubmodule: multi-request batching is "
                f"step-3e scope; got {len(inputs)} requests"
            )
        cache_manager = engine_inputs.cache_manager
        seq_lens = [inp.input_seq_len for inp in inputs]

        cache_manager.set_active_label("main")
        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main",
        )
        # We don't call ``cache_manager.apply_rope`` in attention (we
        # have our own partial 3D rope), but mminf's plan_rope also
        # advances internal position-id state used by ``advance_seq_lens``
        # — keep this call for parity with Orpheus.
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        return {
            "text_inputs": torch.cat([inp.input_ids for inp in inputs]),
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        cache_handle = engine_inputs.cache_manager
        # Resolve position_ids from per-request position state. For
        # text-only the rope only needs 1D positions: a contiguous span
        # starting at ``position_id_start``.
        request_info = engine_inputs.single_request_info
        start_pos = 0
        try:
            start_pos = (
                request_info.position_info.get("main", PositionInfo())
                .position_id_start
            )
        except AttributeError:
            # ARNodeSubmodule contract may not always provide
            # position_info; fall back to 0 for prefill, 1 + len for decode.
            pass

        num_tokens = text_inputs.shape[0]
        position_ids = torch.arange(
            start_pos, start_pos + num_tokens,
            dtype=torch.long, device=text_inputs.device,
        )

        # Embed + transformer + lm_head. The LingMoeModel forward calls
        # cache_handle.set_layer_idx per layer + cache_handle.run_attention
        # inside LingAttention.
        logits = self.model(
            cache_handle,
            input_ids=text_inputs,
            position_ids=position_ids,
        )

        # Advance the cache's sequence lengths so the next decode step
        # knows where to read/write. This is the standard post-forward
        # call that mminf's KV cache uses to track positions.
        cache_handle.advance_seq_lens()

        # Sample only the last position's logits (next-token sampling).
        # Engine expects "new_token" downstream, but for prefill we
        # also publish logits so the engine's sampling layer can run.
        last_logits = logits[-1:, :]
        return {"logits": [last_logits]}

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ) -> None:
        """Rebind ``new_token`` → ``text_inputs`` for the decode loop.

        The decode walk's output edge is named ``text_inputs`` so the loop
        feeds the previous sampled token back into the next iteration.
        ``submodule.forward`` returns ``{"logits": [...]}``; the KV-cache
        engine samples that into ``{"new_token": [...]}``; this hook then
        publishes the same tensor under the ``text_inputs`` key so the
        graph router finds an output to attach to the loop edge.

        Mirrors :meth:`OrpheusLLMSubmodule.postprocess`.
        """
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    # ------------------------------------------------------------------
    # Stop conditions
    # ------------------------------------------------------------------

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """Stop the ``decode_loop`` when the sampled token is the EOS
        (``<|role_end|>`` for Ming, token id 156895)."""
        new_tokens = outputs.get("new_token") or []
        if not new_tokens:
            return set()
        last = new_tokens[-1]
        if isinstance(last, torch.Tensor):
            tok = int(last.flatten()[0].item())
        else:
            tok = int(last)
        if tok == self.eos_token_id:
            return {"decode_loop"}
        return set()

    def can_batch(self, batch, model_inputs) -> bool:
        # Step 3d is single-request; step 3e adds batching.
        return False
