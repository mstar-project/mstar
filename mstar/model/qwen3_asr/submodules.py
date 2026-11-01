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

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.audio_features import LogMelSpectrogram
from mstar.model.components.aut_encoder import AuTEncoder
from mstar.model.components.aut_submodule import AuTEncoderSubmodule
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.components.token_step import TokenStepNode
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
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
    TokenStop,
    device_loopback_enabled,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 1. Qwen3ASREncoderSubmodule
# ===================================================================


class Qwen3ASREncoderSubmodule(AuTEncoderSubmodule):
    """Packed AuT encoder (see ``AuTEncoderSubmodule``).

    Consumes one clip of samples per request (from ``process_prompt``) and
    turns it into unpadded log-mel features on the GPU (``prepare_inputs``).
    """

    def __init__(self, encoder: AuTEncoder, config: Qwen3ASRModelConfig):
        super().__init__(encoder, attn_key=AUT_ATTN)
        self.config = config
        # built here, after the encoder was materialized, so its filter bank
        # and window are real tensors on the encoder's device
        self.log_mel = LogMelSpectrogram(
            num_mel_bins=config.num_mel_bins,
            sampling_rate=config.sampling_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        ).to(device=encoder.conv2d1.weight.device, dtype=torch.float32)  # float32 even under a bf16 default dtype

    def _mel(self, audio: torch.Tensor) -> torch.Tensor:
        """The spectrogram in float32 whatever dtype the engine cast this
        module to (it casts submodules to the compute dtype after they are
        built; a bf16 STFT would not match the reference features)."""
        if self.log_mel.window.dtype != torch.float32:
            self.log_mel.float()
        return self.log_mel(audio)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        audio = inputs["audio"][0].reshape(-1).to(device=self.get_device(), dtype=torch.float32)
        return self.feature_inputs(self._mel(audio).to(self._param_dtype()))  # (num_mel_bins, T)


# ===================================================================
# 2. Qwen3ASRLLMSubmodule
# ===================================================================


class Qwen3ASRLLMSubmodule(TokenStepNode, ARNodeSubmodule):
    """Dense Qwen3 decoder.

    Dispatches on graph_walk:
      - prefill: embed the ChatML prompt, scatter the encoder's audio
        embeddings over its ``<|audio_pad|>`` placeholders, extend the KV
        cache, sample the first token.
      - decode: embed the previous token, single-step decode.

    Prefill is captured on packed token buckets keyed on embeddings, decode
    on batch size with bare ids embedded inside the graph.
    """

    DECODE_WALK = DECODE_WALK
    DECODE_LOOP = DECODE_LOOP
    SAMPLER = SAMPLER
    # prefill's batch is capped by its captures (and uncapped eager)
    MAX_PREFILL_BATCH_SIZE = None
    PREFILL_TOKEN_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    # Prefill and decode are graph replays. The compiled path only served the
    # uncaptured shapes and recompiled per request id (see the encoder).
    disable_torch_compile = True
    # The decode walk reads its last token off the sampler's slot masters and
    # its client token rides inline (see ``device_loopback_signals``).
    reads_device_loopback = True

    def __init__(self, llm: Qwen3DenseLM, config: Qwen3ASRModelConfig):
        super().__init__()
        self.model = llm
        self.config = config
        # only a decode step's token ends the loop: prefill runs outside it
        self.token_stop = TokenStop(
            loop=self.DECODE_LOOP, buffer="new_token", stop_ids=config.stop_token_ids,
            sampler=self.SAMPLER, walk=DECODE_WALK,
        )

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        hidden = self.config.text.hidden_size
        dtype = self.model.embed_tokens.weight.dtype
        return [
            *super().get_accelerator_graph_configs(device, tp_world_size),
            PackedAcceleratorGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n,
                    input_embeds=torch.zeros((n, hidden), device=device, dtype=dtype),
                ),
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

    def splice_audio(
        self, input_ids: torch.Tensor, audio_embeds: torch.Tensor, slot: int, n_slots: int,
    ) -> torch.Tensor:
        """Prompt embeddings with the audio embeddings in place of the
        ``<|audio_pad|>`` run that ``process_prompt`` put at ``slot``. The
        counts are compared on the host: finding the placeholders in
        ``input_ids`` would wait on the device."""
        if n_slots != audio_embeds.shape[0]:
            raise ValueError(
                f"prompt holds {n_slots} audio placeholders but the encoder produced "
                f"{audio_embeds.shape[0]} audio tokens"
            )
        embeds = self.model.embed(input_ids)
        embeds[slot:slot + n_slots] = audio_embeds.to(embeds.dtype)
        return embeds

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        if graph_walk == DECODE_WALK and kwargs.get("has_loopback") and self._device_loopback():
            # one token per row, read off the sampler's slot master in
            # `preprocess`; the routed signal is empty here
            return ARNodeInputs(input_seq_len=1)
        device = self.get_device()
        token_ids = inputs["text_inputs"][0].to(device).reshape(-1)
        if graph_walk == DECODE_WALK:
            return ARNodeInputs(input_seq_len=1, input_ids=token_ids)
        audio_embeds = inputs["audio_embeds"][0].to(device)
        meta = fwd_info.step_metadata
        return ARNodeInputs(
            input_seq_len=token_ids.shape[0],
            input_ids=token_ids,
            input_embeds=self.splice_audio(token_ids, audio_embeds, meta["audio_slot"], meta["audio_tokens"]),
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

    def uniform_row_inputs(self, graph_walk: str, has_loopback: bool = False) -> ARNodeInputs | None:
        # the loop-back decode row above, built once for the step's rows
        if graph_walk == DECODE_WALK and has_loopback and self._device_loopback():
            return ARNodeInputs(input_seq_len=1)
        return None

    def _device_loopback(self) -> bool:
        return device_loopback_enabled() and self.node_resources[SAMPLER].has_slot_masters

    def device_loopback_signals(self, graph_walk: str) -> frozenset[str]:
        if graph_walk == DECODE_WALK and self._device_loopback():
            return frozenset({"text_inputs"})
        return frozenset()

    def inline_client_signals(self, graph_walk: str) -> dict[str, str]:
        # the decode loop's client edge carries the token the stop check
        # already has on the host; prefill's new_token edge is persisted
        return {"text_inputs": "new_token"} if graph_walk == DECODE_WALK else {}

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if graph_walk == DECODE_WALK:
            if any(inp.input_ids is None for inp in inputs):
                # device loop-back: every row is its request's last sampled
                # token, read by slot (padding rows read slot 0)
                sampler: SamplerResource = engine_inputs.resources[SAMPLER]
                ids = sampler.loopback_tokens(engine_inputs.request_ids)
                # rows whose last step ran elsewhere brought their token
                routed = [i for i, inp in enumerate(inputs) if inp.input_ids is not None]
                if routed:
                    sampler.overwrite_rows(routed, (ids, torch.cat([inputs[i].input_ids for i in routed])))
                return {"input_ids": ids}
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

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids, input_embeds)
        # Row i is request i: one clone and one D2H copy per step instead of
        # one per request. A decode step's token is its loop's one signal
        # (`text_inputs`); prefill's is the persisted first token.
        signal = "text_inputs" if graph_walk == DECODE_WALK else "new_token"
        return BatchedModelOutput(
            row_outputs={signal: new_tokens},
            check_stop_buffers={"new_token": new_tokens},
        )
