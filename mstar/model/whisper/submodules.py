# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Whisper ASR
# ---------------------------------------------------------------------------
#
# Two submodules covering the encoder-decoder pipeline:
#   1. WhisperEncoderSubmodule  (stateless)  — HF WhisperEncoder, one-shot
#   2. WhisperDecoderSubmodule  (KV/attention/cross-attention/positions/sampler)
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.encoder_decoder import EncoderDecoderNode
from mstar.model.submodule_base import (
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.whisper.components.decoder import WhisperDecoderModel
from mstar.model.whisper.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    POS,
    SAMPLER,
    WhisperModelConfig,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 1. WhisperEncoderSubmodule (stateless)
# ===================================================================


class WhisperEncoderSubmodule(NodeSubmodule):
    """Thin wrapper around the HF Whisper audio encoder.

    Consumes the log-mel spectrogram produced by ``process_prompt``
    (a fixed 30 s window: ``(num_mel_bins, 3000)``) and emits
    ``encoder_states`` of shape ``(max_source_positions, d_model)``
    for the decoder's cross-attention. Runs once per request.
    """

    def __init__(self, audio_encoder: nn.Module, config: WhisperModelConfig):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={"audio_features": inputs["audio_features"][0]}
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        device = self.get_device()
        dtype = next(self.audio_encoder.parameters()).dtype
        feats = audio_features.to(device=device, dtype=dtype)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)  # (1, num_mel_bins, 3000)
        encoder_states = self.audio_encoder(feats).last_hidden_state.squeeze(0)
        return {"encoder_states": [encoder_states]}


# ===================================================================
# 2. WhisperDecoderSubmodule
# ===================================================================


class WhisperDecoderSubmodule(EncoderDecoderNode, ARNodeSubmodule):
    """Autoregressive Whisper decoder: the forced prompt
    (``<|startoftranscript|><|lang|><|task|><|notimestamps|>``) at prefill,
    then one token per step, with HF generate's token suppression.

    Only decode is captured: every prefill is the same shape (a 4-token
    prompt over a ``max_source_positions`` encoder window), dominated by the
    1500-token cross-K/V projection, which has nothing to gain from a graph.
    """

    KV_CACHE = KV_CACHE
    ATTN = ATTN
    POS = POS
    SAMPLER = SAMPLER
    CROSS_KV_CACHE = CROSS_KV_CACHE
    CROSS_ATTN = CROSS_ATTN
    CONTEXT_LABEL = CONTEXT_LABEL
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # Prefill isn't captured, so no capture bucket bounds its batch. A set too
    # big for the context cache can't be admitted; with offload configured it
    # evicts and retries, without it the batch reforms identically.
    MAX_PREFILL_BATCH_SIZE = 16

    def __init__(self, decoder: WhisperDecoderModel, config: WhisperModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self.stop_token_ids = frozenset({config.eos_token_id})
        self._suppress_ids: torch.Tensor | None = None
        self._begin_suppress_ids: torch.Tensor | None = None

    def process_logits(self, logits: torch.Tensor, graph_walk: str) -> torch.Tensor:
        """HF generate parity: mask the always-suppressed token set, plus
        the begin-suppressed set for the first generated token."""
        device = logits.device
        if self._suppress_ids is None:
            self._suppress_ids = torch.tensor(self.config.suppress_tokens, dtype=torch.long, device=device)
            self._begin_suppress_ids = torch.tensor(self.config.begin_suppress_tokens, dtype=torch.long, device=device)
        if self._suppress_ids.numel():
            logits.index_fill_(-1, self._suppress_ids, float("-inf"))
        if graph_walk == self.PREFILL_WALK and self._begin_suppress_ids.numel():
            logits.index_fill_(-1, self._begin_suppress_ids, float("-inf"))
        return logits
