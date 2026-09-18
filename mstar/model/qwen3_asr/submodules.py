# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Qwen3-ASR
# ---------------------------------------------------------------------------
#
# Two submodules:
#   1. Qwen3ASREncoderSubmodule  (ragged attention) — packed AuT encoder over
#      every request's audio, windows declared as segments
#   2. Qwen3ASRLLMSubmodule      (KV/attention/positions/sampler) — dense Qwen3
#      with the audio embeddings spliced into the prompt at prefill
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig, PackedCudaGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.aut_encoder import AuTEncoder
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.qwen3_asr.config import (
    ATTN,
    AUT_ATTN,
    DECODE_LOOP,
    DECODE_WALK,
    KV_CACHE,
    PREFILL_WALK,
    ROPE,
    SAMPLER,
    Qwen3ASRModelConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

WINDOW_LABEL = "main"


# ===================================================================
# 1. Qwen3ASREncoderSubmodule
# ===================================================================


class Qwen3ASREncoderSubmodule(NodeSubmodule):
    """Packed AuT encoder.

    Consumes one unpadded log-mel clip per request (``(num_mel_bins, T)``
    from ``process_prompt``) and emits ``audio_embeds`` of shape
    ``(num_audio_tokens, hidden)`` in LLM space. Requests are packed into
    one forward; every 8 s attention window of every request is one
    segment of the node's ragged-attention step, so the runner plans the
    kernel's layout before the forward and nothing is recomputed inside.
    """

    MAX_BATCH_SIZE = 8

    def __init__(self, encoder: AuTEncoder, config: Qwen3ASRModelConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def _param_dtype(self) -> torch.dtype:
        return self.encoder.conv2d1.weight.dtype

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        feats = inputs["audio_features"][0]
        if feats.dim() == 3:
            feats = feats.squeeze(0)
        num_frames = int(feats.shape[-1])
        feats = feats.to(device=self.get_device(), dtype=self._param_dtype())
        return NodeInputs(
            tensor_inputs={"audio_features": feats},
            kwargs={"num_frames": num_frames},
            input_seq_len=self.encoder.config.tokens_for_frames(num_frames),
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        # one segment per attention window, in the packed order the forward
        # lays the requests out in
        segments = [
            Segment(request_id=rid, label=WINDOW_LABEL, span=window)
            for rid, inp in zip(request_ids, inputs, strict=True)
            for window in self.encoder.config.window_lengths(inp.input_seq_len)
        ]
        return SubmoduleStep(segments=segments, steps={AUT_ATTN: AttentionStep(causal=False)})

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        lens = [inp.kwargs["num_frames"] for inp in inputs]
        longest = max(lens)
        padded = torch.stack([
            F.pad(inp.tensor_inputs["audio_features"], (0, longest - n))
            for inp, n in zip(inputs, lens, strict=True)
        ])
        return {"audio_features": padded, "feature_lens": lens}

    def _encode(self, audio_features: torch.Tensor, feature_lens: list[int]) -> list[torch.Tensor]:
        embeds, layout = self.encoder(audio_features, feature_lens)
        return list(embeds.split(layout.tokens_per_request, dim=0))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        feature_lens: list[int],
        **kwargs,
    ) -> NameToTensorList:
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        return {"audio_embeds": [self._encode(audio_features, feature_lens)[0]]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        feature_lens: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        per_request = self._encode(audio_features, feature_lens)
        return {
            rid: {"audio_embeds": [emb]}
            for rid, emb in zip(engine_inputs.request_ids, per_request, strict=True)
        }


# ===================================================================
# 2. Qwen3ASRLLMSubmodule
# ===================================================================


class Qwen3ASRLLMSubmodule(ARNodeSubmodule):
    """Dense Qwen3 decoder.

    Dispatches on graph_walk:
      - prefill: embed the ChatML prompt, scatter the encoder's audio
        embeddings over its ``<|audio_pad|>`` placeholders, extend the KV
        cache, sample the first token.
      - decode: embed the previous token, single-step decode.

    Prefill is captured on packed token buckets keyed on embeddings, decode
    on batch size with bare ids embedded inside the graph.
    """

    PREFILL_TOKEN_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self, llm: Qwen3DenseLM, config: Qwen3ASRModelConfig):
        super().__init__()
        self.model = llm
        self.config = config

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        hidden = self.config.text.hidden_size
        dtype = self.model.embed_tokens.weight.dtype
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk=DECODE_WALK,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedCudaGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n,
                    input_embeds=torch.zeros((n, hidden), device=device, dtype=dtype),
                ),
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

    def splice_audio(self, input_ids: torch.Tensor, audio_embeds: torch.Tensor) -> torch.Tensor:
        """Prompt embeddings with the audio embeddings in place of the
        ``<|audio_pad|>`` placeholders; the counts must agree."""
        embeds = self.model.embed(input_ids)
        slots = input_ids == self.config.audio_token_id
        n_slots = int(slots.sum())
        if n_slots != audio_embeds.shape[0]:
            raise ValueError(
                f"prompt holds {n_slots} audio placeholders but the encoder produced "
                f"{audio_embeds.shape[0]} audio tokens"
            )
        embeds[slots] = audio_embeds.to(embeds.dtype)
        return embeds

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        device = self.get_device()
        token_ids = inputs["text_inputs"][0].to(device).reshape(-1)
        if graph_walk == DECODE_WALK:
            return ARNodeInputs(input_seq_len=1, input_ids=token_ids)
        audio_embeds = inputs["audio_embeds"][0].to(device)
        return ARNodeInputs(
            input_seq_len=token_ids.shape[0],
            input_ids=token_ids,
            input_embeds=self.splice_audio(token_ids, audio_embeds),
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
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
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
        if graph_walk == DECODE_WALK:
            return {"input_ids": torch.cat([inp.input_ids for inp in inputs])}
        return {"input_embeds": torch.cat([inp.input_embeds for inp in inputs], dim=0)}

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None,
        input_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        attn: AttentionManager = engine_inputs.resources[ATTN]
        if input_embeds is None:
            input_embeds = self.model.embed(input_ids)
        hidden = self.model(input_embeds, label="main")
        if graph_walk != DECODE_WALK:
            hidden = attn.select_last_hidden(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=self.model.logits(hidden))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._forward(graph_walk, engine_inputs, input_ids, input_embeds)]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids, input_embeds)
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
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if request_info.graph_walk != DECODE_WALK or "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        decoded = request_info.dynamic_loop_iter_counts.get(DECODE_LOOP, 0) + 1
        if (not ignore_eos and token in self.config.stop_token_ids) or decoded >= request_info.max_tokens:
            return {DECODE_LOOP}
        return set()
