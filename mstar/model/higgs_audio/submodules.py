# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Higgs-Audio STT
# ---------------------------------------------------------------------------
#
# Two submodules:
#   1. HiggsAudioEncoderSubmodule  (stateless) — checkpoint's Whisper-style
#      audio_tower + audio_encoder_proj MLP projector, one-shot at prefill
#   2. HiggsAudioLLMSubmodule      (KV/attention/positions/sampler) — dense Qwen3 LLM
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig, PackedCudaGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.higgs_audio.config import ATTN, KV_CACHE, ROPE, SAMPLER, HiggsAudioModelConfig
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 1. HiggsAudioEncoderSubmodule (stateless)
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
# 2. HiggsAudioLLMSubmodule (KV cache + attention + positions + sampler)
# ===================================================================


class HiggsAudioLLMSubmodule(ARNodeSubmodule):
    """Dense Qwen3 LLM.

    Dispatches on graph_walk:
      - prefill_text:  embed a text span, extend the KV cache.
      - prefill_audio: splice the encoder's audio embeddings, extend the
        KV cache.
      - decode:        embed the previous token, single-step decode.

    Every walk samples a token; ``postprocess`` drops it for the prefill
    steps that are not the last one, so the captured graph's output keys
    stay the same whatever the step metadata says. The two prefill walks
    share one packed capture — both hand the forward ``input_embeds`` —
    while decode carries bare ``input_ids`` and embeds inside the graph.
    """

    PREFILL_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(self, llm: nn.Module, config: HiggsAudioModelConfig):
        super().__init__()
        self.model = llm
        self.config = config

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedCudaGraphConfig(
                capture_graph_walk="prefill_text",
                replay_graph_walks=["prefill_text", "prefill_audio"],
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n,
                    input_embeds=torch.zeros(
                        (n, self.config.hidden_size),
                        device=device, dtype=torch.bfloat16,
                    ),
                ),
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
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
        if graph_walk == "decode":
            # embedding happens inside the captured graph; a decode row is
            # one id, so staging it costs 8 bytes instead of a hidden vector
            return ARNodeInputs(input_seq_len=1, input_ids=token_ids)

        # prefill_text shares prefill_audio's capture, which is keyed on
        # embeddings, so the text span is embedded here
        return ARNodeInputs(
            input_seq_len=token_ids.shape[0],
            input_ids=token_ids,
            input_embeds=self.model.embed_tokens(token_ids),
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
        return SubmoduleStep(
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                ) for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                SAMPLER: SamplerStep(apply_penalty=False),
                ROPE: PositionStep(),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if graph_walk == "decode":
            return {
                "input_ids": torch.cat([inp.input_ids for inp in inputs]),
            }
        return {
            "input_embeds": torch.cat(
                [inp.input_embeds for inp in inputs], dim=0
            ),
        }

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None,
        input_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the backbone and sample one token per request."""
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        attn: AttentionManager = engine_inputs.resources[ATTN]

        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)
        hidden = self.model(input_embeds=input_embeds, label="main")

        if graph_walk != "decode":
            # packed prefill: one hidden per request, at its last token
            hidden = attn.select_last_hidden(hidden)

        logits = self.model.lm_head(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {
            "new_token": [self._forward(
                graph_walk=graph_walk,
                engine_inputs=engine_inputs,
                input_ids=input_ids,
                input_embeds=input_embeds,
            )]
        }

    def can_batch(
        self, batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            input_embeds=input_embeds,
        )
        return {
            rid: {"new_token": [new_tokens[i:i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Prefill samples on every step so the captured graph's outputs are
        # fixed; only the last prefill's token is a real one.
        if request_info.graph_walk != "decode" and \
                not request_info.step_metadata.get("is_last_prefill", False):
            outputs.pop("new_token", None)
            return
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs or request_info.graph_walk != "decode":
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        decoded_tokens = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
        if (not ignore_eos and token in self.config.stop_token_ids) or \
                decoded_tokens >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
