"""Higgs-Audio STT's node submodules.

- ``HiggsAudioEncoderSubmodule``: the checkpoint's Whisper-style audio tower
  and its MLP projector, once per request, stateless.
- ``HiggsAudioLLMSubmodule``: the dense Qwen3 LLM, on the shared
  `SampledTokenDecoder`; the prompt prefills as one row with the encoder's
  output spliced in at its audio pads.
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.higgs_audio.config import ATTN, HiggsAudioModelConfig
from mstar.model.keys import KV_CACHE
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule


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


class HiggsAudioLLMSubmodule(SampledTokenDecoder):
    ATTN_KEY = ATTN
    # ~12.5 tokens per second of audio: eight 30 s clips and their
    # instructions fill the top media bucket
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}
