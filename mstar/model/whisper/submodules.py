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
from mstar.model.components.audio_features import LogMelSpectrogram
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.whisper.components.alignment import word_timings
from mstar.model.whisper.components.decoder import WhisperDecoderModel
from mstar.model.whisper.components.encoder import WhisperEncoderModel
from mstar.model.whisper.components.timestamps import TimestampRules, inactive_state, rule_state
from mstar.model.whisper.config import (
    ALIGN_WALK,
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

    Consumes one window of samples per request (at most 30 s, from
    ``process_prompt``), pads it to the window, turns the batch into log-mel
    features with one STFT on the GPU (``preprocess``) and emits
    ``encoder_states`` of shape ``(max_source_positions, d_model)`` for the
    decoder's cross-attention. Every window is the same shape, so a batch is
    a dense ``[bs, num_mel_bins, 3000]`` tensor and the forward is captured
    once per batch size (``BatchedCudaGraphConfig``): the runner pads a
    smaller batch with zero windows and replays the nearest bucket. The node
    holds no resources, so ``declare_step`` stays at the base class's
    ``None``.
    """

    ENCODER_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    # Graph capture alone removes the per-layer launch gaps; inductor's
    # fusions are a second-order gain on these GEMM-bound shapes and cost one
    # compile per bucket, so they are off until measured to win.
    ENCODER_COMPILE = False
    # The engine's blanket torch.compile of ``forward_batched`` traces the
    # per-request Python around the tensor work and guards on request ids
    # and batch composition, so every new batch recompiled (1-2 s stalls,
    # up to the recompile limit per frame) until it was measured and
    # switched off. Every window is graph-replayed anyway.
    disable_torch_compile = True

    def __init__(self, encoder: WhisperEncoderModel, config: WhisperModelConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config
        # built here, after the encoder was materialized, so its filter bank
        # and window are real tensors on the encoder's device
        self.log_mel = LogMelSpectrogram(
            num_mel_bins=config.num_mel_bins,
            sampling_rate=config.sampling_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            chunk_length=config.chunk_length,
        ).to(device=encoder.conv1.weight.device, dtype=torch.float32)  # float32 even under a bf16 default dtype

    def _mel(self, audio: torch.Tensor) -> torch.Tensor:
        """The spectrogram in float32 whatever dtype the engine cast this
        module to (it casts submodules to the compute dtype after they are
        built; a bf16 STFT would not match the reference features)."""
        if self.log_mel.window.dtype != torch.float32:
            self.log_mel.float()
        return self.log_mel(audio)

    def _param_dtype(self) -> torch.dtype:
        return self.encoder.conv1.weight.dtype

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        del tp_world_size
        window = torch.zeros(self.config.n_samples, dtype=torch.float32, device=device)
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                replay_graph_walks=list(ENCODER_WALKS),
                # one window of samples per row; the token count is a row
                # count here. ``preprocess`` turns the rows into the log-mel
                # batch the captured forward reads.
                single_request_inputs=NodeInputs(
                    tensor_inputs={"audio": window}, input_seq_len=1,
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
        audio = inputs["audio"][0].reshape(-1).to(device=self.get_device(), dtype=torch.float32)
        audio = self.log_mel.pad_or_trim(audio)  # zero-pad to the 30 s window
        return NodeInputs(tensor_inputs={"audio": audio}, input_seq_len=1)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # one STFT for the whole batch; the runner copies the result into the
        # captured forward's static input
        audio = torch.stack([inp.tensor_inputs["audio"] for inp in inputs], dim=0)
        return {"audio_features": self._mel(audio).to(self._param_dtype())}

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
      - align: after the loop, when word timestamps were asked for: one
        teacher-forced pass over ``<|sot|><|lang|><|task|><|notimestamps|>
        transcript <|eot|>`` outside the caches; the alignment heads'
        cross-attention is turned into word start/end times, emitted as a
        token sequence in Whisper's own timestamp vocabulary
        (``<|startoflm|> <|s0|> word <|e0|><|s1|> word ...``) that the
        detokenizer renders and the serving layer parses.

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
    # See the encoder: dynamo recompiled the prefill walks per request id
    # (a 1-2 s stall each, measured at RTFx 158 -> the fix's number at c=32);
    # decode steps are graph replays and the prefill is four eager layers.
    disable_torch_compile = True

    def __init__(self, decoder: WhisperDecoderModel, config: WhisperModelConfig, tokenizer=None):
        super().__init__()
        self.decoder = decoder
        self.config = config
        # the align walk groups tokens into words and re-encodes them
        self.tokenizer = tokenizer
        self._startoflm_id = tokenizer.convert_tokens_to_ids("<|startoflm|>") if tokenizer is not None else None
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
        if graph_walk == ALIGN_WALK:
            return self._prepare_alignment(fwd_info, inputs)
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

        if graph_walk == DECODE_WALK:
            # The rule state arrives with the token it was advanced by (see
            # ``timestamps.py``): the worker launches this step before the
            # previous token is readable on the host, so it cannot be rebuilt
            # from a Python history here.
            routed = inputs.get("ts_rules")
            if routed:
                tensor_inputs["ts_rules"] = routed[0].to(device).reshape(-1)
            else:
                tensor_inputs["ts_rules"] = torch.tensor(inactive_state(), dtype=torch.long, device=device)
        else:
            state = self.request_state(fwd_info.request_id)
            # The learned position table caps prompt + transcript at
            # max_target_positions; check_stop reads this back.
            state.add("prompt_len", state.get("prompt_len", 0) + seq_len)
            # Timestamps are on when the prompt omits <|notimestamps|>; the
            # detection walk's prompt ends at <|sot|> and says nothing yet.
            prompt = token_ids.tolist() + tensor_inputs.get("prompt_tail", token_ids[:0]).tolist()
            timestamps = graph_walk != DETECT_LANGUAGE_WALK and self.config.no_timestamps_token_id not in prompt
            # the align walk rebuilds the forced prompt from these
            for tok in prompt:
                if self.config.language_of(tok) is not None:
                    state.add("language", tok)
                elif tok in self.config.task_to_id.values():
                    state.add("task", tok)
            state.add("timestamps", timestamps)
            row = rule_state([], self.config) if timestamps else inactive_state()
            tensor_inputs["ts_rules"] = torch.tensor(row, dtype=torch.long, device=device)

        return ARNodeInputs(
            input_seq_len=seq_len,
            input_ids=token_ids,
            tensor_inputs=tensor_inputs,
        )

    # -- word timestamps ----------------------------------------------------

    def _prepare_alignment(self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList) -> ARNodeInputs:
        """The teacher-forcing sequence for the align walk: the forced prompt
        with ``<|notimestamps|>`` (word timing never uses timestamp tokens),
        the transcript's text tokens, end-of-text."""
        device = self.get_device()
        state = self.request_state(fwd_info.request_id)
        generated = [int(t) for part in inputs["transcript"] for t in part.reshape(-1).tolist()]
        language = state.get("language")
        if generated and self.config.language_of(generated[0]) is not None:
            language, generated = generated[0], generated[1:]  # detected, not forced
        task = state.get("task", self.config.task_token("transcribe"))
        text = [t for t in generated if t < self.config.eos_token_id]
        if language is None:
            language = self.config.language_token("en")
        seq = [self.config.decoder_start_token_id, language, task, self.config.no_timestamps_token_id]
        seq += text + [self.config.eos_token_id]
        return ARNodeInputs(
            input_seq_len=len(seq),
            input_ids=torch.tensor(seq, dtype=torch.long, device=device),
            tensor_inputs={
                "encoder_states": inputs["encoder_states"][0].to(device),
                "audio_frames": inputs["audio_frames"][0].to(device).reshape(-1),
            },
        )

    def _align(self, input_ids: torch.Tensor, encoder_states: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Word timings for one request as a token sequence in the timestamp
        vocabulary: ``<|startoflm|>`` then ``<|start|> word <|end|>`` per word,
        each word re-encoded so the detokenizer renders it as text."""
        heads = [(int(layer), int(head)) for layer, head in self.config.alignment_heads]
        weights = self.decoder.cross_attention_weights(input_ids, encoder_states, heads)
        prompt_len = 3  # <|sot|><|lang|><|task|>; the <|notimestamps|> row predicts the first text token
        rows = weights[:, prompt_len:-1]
        text = input_ids[prompt_len + 1:-1].tolist()
        words = word_timings(rows, text, self.tokenizer.decode, num_frames)
        tb, top = self.config.timestamp_begin, self.config.vocab_size - 1

        def stamp(seconds: float) -> int:
            return min(top, tb + int(round(seconds / self.config.timestamp_precision)))

        ids = [self._startoflm_id]
        for w in words:
            ids += [stamp(w["start"])] + self.tokenizer.encode(w["word"], add_special_tokens=False) + [stamp(w["end"])]
        return torch.tensor(ids, dtype=torch.long, device=input_ids.device)

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
    ) -> SubmoduleStep | None:
        if graph_walk == ALIGN_WALK:
            return None  # eager pass over its own inputs; touches no cache
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
        if graph_walk == ALIGN_WALK:
            (inp,) = inputs  # one request per align step
            return {
                "input_ids": inp.input_ids,
                "encoder_states": inp.tensor_inputs["encoder_states"],
                "audio_frames": inp.tensor_inputs["audio_frames"],
            }
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the decoder and sample one token per request; also the
        timestamp rule state each row carries into its next step."""
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
        new_tokens = sampler.sample(engine_inputs.request_ids, logits=logits)
        if ts_rules is None:
            return new_tokens, None
        return new_tokens, self.timestamp_rules.advance(ts_rules, new_tokens)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        ts_rules: torch.Tensor | None = None,
        audio_frames: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if graph_walk == ALIGN_WALK:
            return {"word_tokens": [self._align(input_ids, encoder_states, int(audio_frames.reshape(-1)[0]))]}
        new_tokens, next_rules = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            encoder_states=encoder_states,
            ts_rules=ts_rules,
        )
        out: NameToTensorList = {"new_token": [new_tokens]}
        if next_rules is not None:
            out["ts_rules"] = [next_rules[0]]
        return out

    def can_batch(
        self, batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        if graph_walk == DECODE_WALK:
            return None  # the decode capture sizes cap it
        if graph_walk == ALIGN_WALK:
            return 1
        return self.MAX_PREFILL_BATCH_SIZE

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        ts_rules: torch.Tensor | None = None,
        audio_frames: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        if graph_walk == ALIGN_WALK:
            (rid,) = engine_inputs.request_ids
            return {rid: {"word_tokens": [self._align(input_ids, encoder_states, int(audio_frames.reshape(-1)[0]))]}}
        new_tokens, next_rules = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            encoder_states=encoder_states,
            ts_rules=ts_rules,
        )
        out: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(engine_inputs.request_ids):
            out[rid] = {"new_token": [new_tokens[i:i + 1]]}
            if next_rules is not None:
                out[rid]["ts_rules"] = [next_rules[i]]
        return out

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Metadata-only: rebind output name so the decode loop feeds the
        # sampled token back in as the next step's text_inputs. ``ts_rules``
        # keeps its name; the loop routes it back under it.
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
