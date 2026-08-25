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
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.step import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SubmoduleStep
from mstar.engine.resources.attn.manager import AttentionManager
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.position.manager import PositionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.submodule_base import (
    ARNodeInputs,
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


class WhisperDecoderSubmodule(ARNodeSubmodule):
    """Autoregressive Whisper decoder.

    Dispatches on graph_walk:
      - prefill: embed the forced decoder prompt
        (``<|startoftranscript|><|lang|><|task|><|notimestamps|>``),
        project ``encoder_states`` to per-layer cross-attention K/V and
        write them into the context stream, fill the self-attention KV
        cache, and sample the first transcript token.
      - decode: embed the previous token, single-step decode.

    Both cache streams belong to the step: ``main`` grows by the token
    count, ``context`` grows by the encoder output at prefill and is
    declared zero-span (read-only) thereafter. Only decode is captured —
    prefill runs once per request and is dominated by the 1500-token
    cross-K/V projection, which has nothing to gain from a graph.
    """

    # Every prefill is the same shape (a 4-token forced prompt over a
    # max_source_positions encoder window), so decode is the only walk whose
    # capture buys anything.
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

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
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        device = self.get_device()
        token_ids = inputs["text_inputs"][0].to(device).reshape(-1)

        tensor_inputs = {}
        if graph_walk == "prefill":
            tensor_inputs["encoder_states"] = inputs["encoder_states"][0].to(device)

        return ARNodeInputs(
            input_seq_len=token_ids.shape[0],
            input_ids=token_ids,
            tensor_inputs=tensor_inputs,
        )

    @staticmethod
    def _context_span(inp: ARNodeInputs) -> int:
        """How far this step grows the request's encoder-context stream.

        Non-zero only on the prefill that writes it; every later step reads
        the same pages without extending them.
        """
        encoder_states = inp.tensor_inputs.get("encoder_states")
        return 0 if encoder_states is None else encoder_states.shape[0]

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
    ) -> SubmoduleStep:
        context_segments = tuple(
            Segment(
                request_id=rid,
                label=CONTEXT_LABEL,
                span=self._context_span(inp),
            ) for rid, inp in zip(request_ids, inputs, strict=True)
        )
        return SubmoduleStep(
            # the default, for the resources over the self-attention cache
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
                # The context stream: a real span on the prefill that writes
                # it, zero afterwards. `commit` is what turns the prefill's
                # reservation into resident pages the later steps read.
                CROSS_KV_CACHE: KVStep(segments=context_segments),
                CROSS_ATTN: AttentionStep(
                    segments=context_segments, causal=False,
                ),
                SAMPLER: SamplerStep(apply_penalty=False),
                POS: PositionStep(),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        preprocessed: dict[str, torch.Tensor | Any] = {
            "input_ids": torch.cat([inp.input_ids for inp in inputs]),
        }
        if graph_walk == "prefill":
            # Concatenated in segment order, which is how the context plan
            # laid the requests' pages out.
            preprocessed["encoder_states"] = torch.cat(
                [inp.tensor_inputs["encoder_states"] for inp in inputs], dim=0,
            )
        return preprocessed

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the decoder and sample one token per request."""
        attn: AttentionManager = engine_inputs.resources[ATTN]
        pos: PositionManager = engine_inputs.resources[POS]
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]

        if encoder_states is not None:
            # prefill only: fills the context pages this step reserved, before
            # the layers read them back
            self.decoder.write_cross_kv(encoder_states)

        input_embeds = self.decoder.embed(input_ids, pos.pos_ids("main"))
        hidden = self.decoder(input_embeds=input_embeds, label="main")

        if graph_walk == "prefill":
            # packed prefill: one hidden per request, at its last token
            hidden = attn.select_last_hidden(hidden)

        logits = self.decoder.lm_head(hidden)
        logits = self._apply_suppress(
            logits, is_first_token=graph_walk == "prefill",
        )
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {
            "new_token": [self._forward(
                graph_walk=graph_walk,
                engine_inputs=engine_inputs,
                input_ids=input_ids,
                encoder_states=encoder_states,
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
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            encoder_states=encoder_states,
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
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        decoded_tokens = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
        if (not ignore_eos and token == self.config.eos_token_id) or \
                decoded_tokens >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
