# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Whisper ASR
# ---------------------------------------------------------------------------
#
# Two submodules covering the encoder-decoder pipeline:
#   1. WhisperEncoderSubmodule  (enc_dec engine)  — HF WhisperEncoder, one-shot
#   2. WhisperDecoderSubmodule  (ar engine)       — mstar-native decoder with
#      paged self-attn KV cache + per-request static cross-attn K/V
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
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.whisper.components.decoder import WhisperDecoderModel
from mstar.model.whisper.config import WhisperModelConfig
from mstar.utils.sampling import SeenTokenMask

logger = logging.getLogger(__name__)


# ===================================================================
# 1. WhisperEncoderSubmodule (enc_dec engine)
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
# 2. WhisperDecoderSubmodule (ar engine)
# ===================================================================


class WhisperDecoderSubmodule(ARNodeSubmodule):
    """Autoregressive Whisper decoder.

    Dispatches on graph_walk:
      - prefill: embed the forced decoder prompt
        (``<|startoftranscript|><|lang|><|task|><|notimestamps|>``),
        compute per-layer cross-attn K/V from ``encoder_states`` and
        write them into the engine's cross-attention context pool (they
        are static for the whole generation), fill the self-attn KV
        cache, and emit logits for the first token.
      - decode: embed the previous token, single-step decode.

    Cross-attn K/V live in the engine's per-source context pool (see
    ``KVCacheConfig.cross_attn`` / issue #160); the write + plan happen
    in ``preprocess`` so ``forward`` is pure planned compute. Requests
    still run one per step (``can_batch`` False) — batching is now an
    engine-side follow-up rather than blocked on submodule state.
    """

    def __init__(self, decoder: WhisperDecoderModel, config: WhisperModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self._suppress_ids: torch.Tensor | None = None
        self._begin_suppress_ids: torch.Tensor | None = None

    def _apply_suppress(self, logits: torch.Tensor, is_first_token: bool) -> torch.Tensor:
        """HF generate parity: mask the always-suppressed token set, plus
        the begin-suppressed set for the first generated token."""
        device = logits.device
        if self._suppress_ids is None:
            self._suppress_ids = torch.tensor(
                self.config.suppress_tokens, dtype=torch.long, device=device,
            )
            self._begin_suppress_ids = torch.tensor(
                self.config.begin_suppress_tokens, dtype=torch.long, device=device,
            )
        if self._suppress_ids.numel():
            logits.index_fill_(-1, self._suppress_ids, float("-inf"))
        if is_first_token and self._begin_suppress_ids.numel():
            logits.index_fill_(-1, self._begin_suppress_ids, float("-inf"))
        return logits

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        seen_token_mask: SeenTokenMask = None,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs,
    ) -> ARNodeInputs:
        device = self.decoder.embed_tokens.weight.device
        start_pos = pos_info.get("main", PositionInfo()).position_id_start

        token_ids = inputs["text_inputs"][0].to(device).reshape(-1)
        embeds = self.decoder.embed(token_ids, start_pos)

        tensor_inputs = {}
        if graph_walk == "prefill":
            tensor_inputs["encoder_states"] = inputs["encoder_states"][0]

        return ARNodeInputs(
            input_seq_len=token_ids.shape[0],
            input_embeds=embeds,
            tensor_inputs=tensor_inputs,
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        assert len(inputs) == 1, (
            "WhisperDecoderSubmodule runs one request per step."
        )
        cache_manager = engine_inputs.cache_manager
        seq_lens = [inputs[0].input_seq_len]
        cache_manager.set_active_label("main")
        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main",
        )
        # No plan_rope: Whisper has no RoPE (learned absolute positions).

        # Prefill: compute per-layer cross-attn K/V from the encoder output
        # and write them into the engine's context pool — once per request;
        # they are static for the whole generation.
        encoder_states = inputs[0].tensor_inputs.get("encoder_states")
        if encoder_states is not None:
            device = self.decoder.embed_tokens.weight.device
            dtype = inputs[0].input_embeds.dtype
            encoder_states = encoder_states.to(device=device, dtype=dtype)
            with torch.no_grad():
                cross_kvs = self.decoder.compute_cross_kv(encoder_states)
            for layer_idx, (k, v) in enumerate(cross_kvs):
                cache_manager.add_cross_attn_kv(
                    engine_inputs.request_ids, k, v,
                    layer_idx=layer_idx, label="main",
                )

        # Both prefill and decode attend to the full (fixed) context.
        cache_manager.plan_cross_attention(q_seq_lens=seq_lens, label="main")

        return {"input_embeds": inputs[0].input_embeds}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        hidden = self.decoder(
            input_embeds=input_embeds,
            cache_handle=engine_inputs.cache_manager,
        )
        logits = self.decoder.lm_head(hidden[-1:])
        # The prefill step samples the first generated token.
        logits = self._apply_suppress(logits, is_first_token=graph_walk == "prefill")
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
        # Metadata-only: rebind output name so the decode loop feeds the
        # sampled token back in as the next step's text_inputs.
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
        ignore_eos = request_info.sampling_config["decoder"].ignore_eos
        decoded_tokens = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
        if (not ignore_eos and token == self.config.eos_token_id) or \
                decoded_tokens >= request_info.max_tokens:
            return {"decode_loop"}
        return set()

