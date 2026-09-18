# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Whisper ASR
# ---------------------------------------------------------------------------
#
# Two submodules covering the encoder-decoder pipeline:
#   1. WhisperEncoderSubmodule  (no resources) — native encoder, batched over
#      requests and captured as one CUDA graph per batch size
#   2. WhisperDecoderSubmodule  (KV/attention/cross-attention/positions/sampler)
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.base import AttentionManager
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
from mstar.model.whisper.components.encoder import WhisperEncoderModel
from mstar.model.whisper.components.timestamps import TimestampRules, inactive_state, rule_state
from mstar.model.whisper.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    DECODE_LOOP,
    DECODE_WALK,
    DETECT_LANGUAGE_WALK,
    KV_CACHE,
    POS,
    PREFILL_PROMPT_WALK,
    PREFILL_WALK,
    SAMPLER,
    WhisperModelConfig,
)

logger = logging.getLogger(__name__)

# Walks that run the encoder and hand the decoder its output.
ENCODER_WALKS = (PREFILL_WALK, DETECT_LANGUAGE_WALK)
# Walks whose sampled token is the first transcript token, subject to the
# begin-suppression rules.
FIRST_TOKEN_WALKS = (PREFILL_WALK, PREFILL_PROMPT_WALK)


# ===================================================================
# 1. WhisperEncoderSubmodule
# ===================================================================


class WhisperEncoderSubmodule(NodeSubmodule):
    """Batched, graph-captured Whisper audio encoder.

    Consumes one log-mel window per request (``(num_mel_bins, 3000)``, from
    ``process_prompt``) and emits ``encoder_states`` of shape
    ``(max_source_positions, d_model)`` for the decoder's cross-attention.
    Every window is the same shape, so a batch is a dense
    ``[bs, num_mel_bins, 3000]`` tensor and the forward is captured once per
    batch size (``BatchedCudaGraphConfig``): the runner pads a smaller batch
    with zero windows and replays the nearest bucket. The node holds no
    resources, so ``declare_step`` stays at the base class's ``None``.
    """

    ENCODER_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    # Graph capture alone removes the per-layer launch gaps; inductor's
    # fusions are a second-order gain on these GEMM-bound shapes and cost one
    # compile per bucket, so they are off until measured to win.
    ENCODER_COMPILE = False

    def __init__(self, encoder: WhisperEncoderModel, config: WhisperModelConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def _param_dtype(self) -> torch.dtype:
        return self.encoder.conv1.weight.dtype

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        del tp_world_size
        window = torch.zeros(
            (self.config.num_mel_bins, self.config.num_frames),
            dtype=self._param_dtype(), device=device,
        )
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                replay_graph_walks=list(ENCODER_WALKS),
                # one window per row; the token count is a row count here
                single_request_inputs=NodeInputs(
                    tensor_inputs={"audio_features": window}, input_seq_len=1,
                ),
                capture_batch_sizes=self.ENCODER_CAPTURE_BATCH_SIZES,
                compile=self.ENCODER_COMPILE,
            ),
        ]

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
        feats = feats.to(device=self.get_device(), dtype=self._param_dtype())
        return NodeInputs(tensor_inputs={"audio_features": feats}, input_seq_len=1)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        return {
            "audio_features": torch.stack(
                [inp.tensor_inputs["audio_features"] for inp in inputs], dim=0,
            ),
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        encoder_states = self.encoder(audio_features)
        return {"encoder_states": [encoder_states[0]]}

    def can_batch(
        self, batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        encoder_states = self.encoder(audio_features)
        return {
            rid: {"encoder_states": [encoder_states[i]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }


# ===================================================================
# 2. WhisperDecoderSubmodule
# ===================================================================


class WhisperDecoderSubmodule(ARNodeSubmodule):
    """Autoregressive Whisper decoder.

    Dispatches on graph_walk:
      - prefill: embed the forced decoder prompt
        (``[<|startofprev|> prev...] <|startoftranscript|><|lang|><|task|>
        [<|notimestamps|>]``), project ``encoder_states`` to per-layer
        cross-attention K/V and write them into the context stream, fill
        the self-attention KV cache, and sample the first transcript token.
      - detect_language: the same with a prompt that ends at
        ``<|startoftranscript|>``; sampling is restricted to language tokens.
      - prefill_prompt: append ``<|lang|><|task|>[<|notimestamps|>]`` after a
        detected language token, over the already written context.
      - decode: embed the previous token, single-step decode.

    Both cache streams belong to the step: ``main`` grows by the token
    count, ``context`` grows by the encoder output at the walk that writes it
    and is declared zero-span (read-only) thereafter. Only decode is captured
    — a prefill is a handful of tokens over a 1500-token cross-K/V projection
    and gains nothing from a graph.
    """

    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # Prefill isn't captured, so the engine's capture cap doesn't bound it and
    # `can_batch` would take the whole ready set. A set too big for the context
    # cache can't be admitted; with offload configured it evicts and retries,
    # without it there is nothing to evict and the batch reforms identically.
    MAX_PREFILL_BATCH_SIZE = 32

    def __init__(self, decoder: WhisperDecoderModel, config: WhisperModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self._suppress_ids: torch.Tensor | None = None
        self._begin_suppress_ids: torch.Tensor | None = None
        self._language_mask: torch.Tensor | None = None
        self.timestamp_rules = TimestampRules(config)

    # -- logit rules ------------------------------------------------------

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

    def _restrict_to_languages(self, logits: torch.Tensor) -> torch.Tensor:
        """Language detection: only the ``<|xx|>`` tokens may be sampled."""
        if self._language_mask is None:
            mask = torch.full((self.config.vocab_size,), float("-inf"), device=logits.device)
            mask[torch.tensor(self.config.language_token_ids, device=logits.device)] = 0.0
            self._language_mask = mask
        return logits + self._language_mask.to(logits.dtype)

    # -- capture ------------------------------------------------------------

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk=DECODE_WALK,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                    # the timestamp rule state rides along as a staged row
                    tensor_inputs={"ts_rules": torch.tensor(inactive_state(), device=device)},
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
        ]

    # -- per-step contract ------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        device = self.get_device()
        token_ids = inputs["text_inputs"][0].to(device).reshape(-1)
        seq_len = token_ids.shape[0]

        tensor_inputs = {}
        if graph_walk in ENCODER_WALKS:
            tensor_inputs["encoder_states"] = inputs["encoder_states"][0].to(device)
        if graph_walk == PREFILL_PROMPT_WALK:
            # the detected language token, followed by the rest of the prompt
            tail = inputs["prompt_tail"][0].to(device).reshape(-1)
            tensor_inputs["prompt_tail"] = tail
            seq_len += tail.shape[0]

        state = self.request_state(fwd_info.request_id)
        if graph_walk != DECODE_WALK:
            # The learned position table caps prompt + transcript at
            # max_target_positions; check_stop reads this back.
            state.add("prompt_len", state.get("prompt_len", 0) + seq_len)
            # Timestamps are on when the prompt omits <|notimestamps|>; the
            # detection walk's prompt ends at <|sot|> and says nothing yet.
            prompt = token_ids.tolist() + tensor_inputs.get("prompt_tail", token_ids[:0]).tolist()
            if graph_walk != DETECT_LANGUAGE_WALK:
                state.add("timestamps", self.config.no_timestamps_token_id not in prompt)
                state.add("generated", [])
        tensor_inputs["ts_rules"] = self._rules_row(graph_walk, state, device)

        return ARNodeInputs(
            input_seq_len=seq_len,
            input_ids=token_ids,
            tensor_inputs=tensor_inputs,
        )

    def _rules_row(self, graph_walk: str, state, device) -> torch.Tensor:
        """This step's timestamp rule for one request (see ``timestamps.py``)."""
        if graph_walk == DETECT_LANGUAGE_WALK or not state.get("timestamps", False):
            row = inactive_state()
        else:
            row = rule_state(state.get("generated", []), self.config)
        return torch.tensor(row, dtype=torch.long, device=device)

    @staticmethod
    def _context_span(inp: ARNodeInputs) -> int:
        """How far this step grows the request's encoder-context stream.

        Non-zero only on the walk that writes it; every later step reads
        the same pages without extending them.
        """
        encoder_states = inp.tensor_inputs.get("encoder_states")
        return 0 if encoder_states is None else encoder_states.shape[0]

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
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

    @staticmethod
    def _row_ids(inp: ARNodeInputs) -> torch.Tensor:
        tail = inp.tensor_inputs.get("prompt_tail")
        if tail is None:
            return inp.input_ids
        return torch.cat([inp.input_ids, tail])

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        preprocessed: dict[str, torch.Tensor | Any] = {
            "input_ids": torch.cat([self._row_ids(inp) for inp in inputs]),
            "ts_rules": torch.stack([inp.tensor_inputs["ts_rules"] for inp in inputs]),
        }
        if graph_walk in ENCODER_WALKS:
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
        ts_rules: torch.Tensor | None = None,
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

        if graph_walk != DECODE_WALK:
            # packed prefill: one hidden per request, at its last token
            hidden = attn.select_last_hidden(hidden)

        logits = self.decoder.lm_head(hidden)
        if graph_walk == DETECT_LANGUAGE_WALK:
            logits = self._restrict_to_languages(logits)
        else:
            logits = self._apply_suppress(
                logits, is_first_token=graph_walk in FIRST_TOKEN_WALKS,
            )
            if ts_rules is not None:
                logits = self.timestamp_rules.apply(logits, ts_rules)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        ts_rules: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {
            "new_token": [self._forward(
                graph_walk=graph_walk,
                engine_inputs=engine_inputs,
                input_ids=input_ids,
                encoder_states=encoder_states,
                ts_rules=ts_rules,
            )]
        }

    def can_batch(
        self, batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        if graph_walk == DECODE_WALK:
            return None  # the decode capture sizes cap it
        return self.MAX_PREFILL_BATCH_SIZE

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        ts_rules: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            encoder_states=encoder_states,
            ts_rules=ts_rules,
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
        if "new_token" not in outputs or request_info.graph_walk == DETECT_LANGUAGE_WALK:
            return set()
        token = outputs["new_token"][0].item()
        state = self.request_state(request_id)
        if state.get("timestamps", False):
            # the rule for the next step reads the history back on the host
            state.get("generated").append(token)
        if request_info.graph_walk != DECODE_WALK:
            return set()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        decoded_tokens = request_info.dynamic_loop_iter_counts.get(DECODE_LOOP, 0) + 1
        # prompt + first token + decoded tokens must fit the position table
        prompt_len = self.request_state(request_id).get("prompt_len", 0)
        positions_left = self.config.max_target_positions - prompt_len - 1 - decoded_tokens
        if (not ignore_eos and token == self.config.eos_token_id) or \
                decoded_tokens >= request_info.max_tokens or positions_left <= 0:
            return {DECODE_LOOP}
        return set()
