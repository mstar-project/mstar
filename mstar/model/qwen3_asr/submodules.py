# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Qwen3-ASR
# ---------------------------------------------------------------------------
#
# Two submodules:
#   1. Qwen3ASREncoderSubmodule  (ragged attention) — packed AuT encoder over
#      every request's audio, windows declared as segments
#   2. Qwen3ASRLLMSubmodule      (KV/attention/positions/sampler) — dense Qwen3
#      on the shared SampledTokenDecoder; the prompt prefills as one row with
#      the audio embeddings spliced in at its pads
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.audio_features import LogMelSpectrogram
from mstar.model.components.aut_encoder import AuTEncoder
from mstar.model.components.aut_submodule import AuTEncoderSubmodule
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.qwen3_asr.config import ATTN, AUT_ATTN, KV_CACHE, Qwen3ASRModelConfig
from mstar.model.submodule_base import NodeInputs

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


class Qwen3ASRLLMSubmodule(SampledTokenDecoder):
    """Dense Qwen3 decoder. Prefill is captured on packed token buckets, decode
    on batch size."""

    ATTN_KEY = ATTN
    # the ladders it captured before moving onto the scaffold
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    PREFILL_MEDIA_TOKEN_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    # Prefill and decode are graph replays. The compiled path only served the
    # uncaptured shapes and recompiled per request id (see the encoder).
    disable_torch_compile = True

    def max_batch_size(self, graph_walk: str) -> int | None:
        # a captured prefill is capped by its largest capture, an eager one not
        # at all, as before the scaffold
        return None

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}
