# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Higgs-Audio STT
# ---------------------------------------------------------------------------
#
# Two submodules:
#   1. HiggsAudioEncoderSubmodule  (enc_dec engine) — checkpoint's Whisper-style
#      audio_tower + audio_encoder_proj MLP projector, one-shot at prefill
#   2. HiggsAudioLLMSubmodule      (ar engine)      — dense Qwen3 LLM
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.kv_store import PositionInfo
from mstar.model.higgs_audio.config import HiggsAudioModelConfig
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.utils.sampling import SeenTokenMask

logger = logging.getLogger(__name__)


# ===================================================================
# 1. HiggsAudioEncoderSubmodule (enc_dec engine)
# ===================================================================


class HiggsAudioEncoderSubmodule(NodeSubmodule):
    """Audio tower + projector.

    Consumes per-chunk mel spectrograms from ``process_prompt``
    (``(num_chunks, num_mel_bins, T)`` padded to the longest chunk, plus
    per-chunk mel frame counts) and emits the concatenated LLM-space
    audio embeddings ``(total_audio_tokens, hidden_size)``.

    Mirrors the reference pipeline: batch-encode padded chunks, run the
    projector on the padded batch, then slice each chunk to its valid
    (downsampled) length and concatenate in order.
    """

    def __init__(
        self,
        audio_tower: nn.Module,
        projector: nn.Module,
        config: HiggsAudioModelConfig,
    ):
        super().__init__()
        self.audio_tower = audio_tower
        self.projector = projector
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={
                "audio_features": inputs["audio_features"][0],
                "audio_feature_lens": inputs["audio_feature_lens"][0],
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        audio_feature_lens: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        device = self.get_device()
        dtype = next(self.audio_tower.parameters()).dtype
        feats = audio_features.to(device=device, dtype=dtype)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)  # (1, num_mel_bins, T)

        encoded = self.audio_tower(feats)         # (num_chunks, T_out, 1280)
        projected = self.projector(encoded)       # (num_chunks, T', hidden)

        chunks = []
        for i, mel_len in enumerate(audio_feature_lens.tolist()):
            valid = self.config.encoder_output_length(int(mel_len))
            chunks.append(projected[i, :valid])
        audio_embeds = torch.cat(chunks, dim=0)   # (total_audio_tokens, hidden)

        return {"audio_embeds": [audio_embeds]}


# ===================================================================
# 2. HiggsAudioLLMSubmodule (ar engine)
# ===================================================================


class HiggsAudioLLMSubmodule(ARNodeSubmodule):
    """Dense Qwen3 LLM.

    Dispatches on graph_walk:
      - prefill_text:  embed a text span, extend the KV cache. The last
        prefill step (``is_last_prefill``) also emits logits so the
        engine samples the first transcript token.
      - prefill_audio: splice the encoder's audio embeddings, extend the
        KV cache.
      - decode:        embed the previous token, single-step decode.
    """

    def __init__(self, llm: nn.Module, config: HiggsAudioModelConfig):
        super().__init__()
        self.model = llm
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        seen_token_mask: SeenTokenMask = None,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs,
    ) -> ARNodeInputs:
        device = self.get_device()

        if graph_walk == "prefill_audio":
            audio_embeds = inputs["audio_embeds"][0].to(device)
            return ARNodeInputs(
                input_seq_len=audio_embeds.shape[0],
                input_embeds=audio_embeds,
            )

        token_ids = inputs["text_inputs"][0].to(device).reshape(-1)
        if graph_walk == "prefill_text" and seen_token_mask is not None:
            seen_token_mask.add_tokens(token_ids)
        embeds = self.model.embed_tokens(token_ids)
        return ARNodeInputs(
            input_seq_len=token_ids.shape[0],
            input_embeds=embeds,
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        cache_manager = engine_inputs.cache_manager
        seq_lens = [inp.input_seq_len for inp in inputs]
        cache_manager.set_active_label("main")
        cache_manager.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")
        return {
            "input_embeds": torch.cat([inp.input_embeds for inp in inputs], dim=0),
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        hidden = self.model(
            input_embeds=input_embeds,
            cache_handle=engine_inputs.cache_manager,
        )

        request_info = engine_inputs.single_request_info
        emit_logits = graph_walk == "decode" or \
            request_info.step_metadata.get("is_last_prefill", False)
        if not emit_logits:
            return {}

        logits = self.model.lm_head(hidden[-1:, :])
        return {"logits": [logits]}

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return False

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.sampling_config["LLM"].ignore_eos
        decoded_tokens = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
        if (not ignore_eos and token in self.config.stop_token_ids) or \
                decoded_tokens >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
