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
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import SlotLease, SubmoduleStep
from mstar.engine.resources.position.manager import PositionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.audio_features import LogMelSpectrogram
from mstar.model.components.encoder_decoder import EncoderDecoderNode
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.whisper.components import alignment
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
from mstar.utils.h2d import PinnedStager, to_device_nonblocking

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
    once per batch size (``BatchedAcceleratorGraphConfig``): the runner pads a
    smaller batch with zero windows and replays the nearest bucket. The node
    holds no resources, so ``declare_step`` stays at the base class's
    ``None``.
    """

    ENCODER_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    # Graph capture alone removes the per-layer launch gaps. Inductor's
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

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        del tp_world_size
        window = torch.zeros(self.config.n_samples, dtype=torch.float32, device=device)
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=PREFILL_WALK,
                replay_graph_walks=list(ENCODER_WALKS),
                # one window of samples per row, and the token count is a row
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
        # one STFT for the whole batch. The runner copies the result into the
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


class WhisperDecoderSubmodule(EncoderDecoderNode, ARNodeSubmodule):
    """Autoregressive Whisper decoder on the shared encoder-decoder node.

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

    The context stream, the step declaration, the token feedback and the stop
    checks are the shared node's. Only decode is captured — a prefill is a
    handful of tokens over a 1500-token cross-K/V projection and gains
    nothing from a graph.
    """

    KV_CACHE = KV_CACHE
    ATTN = ATTN
    POS = POS
    SAMPLER = SAMPLER
    CROSS_KV_CACHE = CROSS_KV_CACHE
    CROSS_ATTN = CROSS_ATTN
    CONTEXT_LABEL = CONTEXT_LABEL
    PREFILL_WALK = PREFILL_WALK
    DECODE_WALK = DECODE_WALK
    DECODE_LOOP = DECODE_LOOP
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # Prefill isn't captured, so no capture bucket bounds its batch. A set too
    # big for the context cache can't be admitted; with offload configured it
    # evicts and retries, without it the batch reforms identically.
    MAX_PREFILL_BATCH_SIZE = 32
    # See the encoder: dynamo recompiled the prefill walks per request id
    # (a 1-2 s stall each, RTFx 158 at c=32 before the fix). Decode steps are
    # graph replays and the prefill is four eager layers.
    disable_torch_compile = True

    def __init__(self, decoder: WhisperDecoderModel, config: WhisperModelConfig, tokenizer=None):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self.stop_token_ids = frozenset({config.eos_token_id})
        # the align walk groups tokens into words and re-encodes them
        self.tokenizer = tokenizer
        self._startoflm_id = tokenizer.convert_tokens_to_ids("<|startoflm|>") if tokenizer is not None else None
        self._suppress_ids: torch.Tensor | None = None
        self._begin_suppress_ids: torch.Tensor | None = None
        self._language_mask: torch.Tensor | None = None
        self.timestamp_rules = TimestampRules(config)
        self._stager = PinnedStager(torch.long)
        self._inactive: torch.Tensor | None = None
        alignment.warm_up()

    def _inactive_rules(self, device: torch.device) -> torch.Tensor:
        """The rules-off state, built once: rows are only read (stacked)."""
        if self._inactive is None or self._inactive.device != device:
            self._inactive = torch.tensor(inactive_state(), dtype=torch.long, device=device)
        return self._inactive

    def _decode_raw(self, ids: list[int]) -> str:
        """Word grouping's decode: no clean-up, which rewrites text across token
        boundaries (openai-whisper's tiktoken decode has none)."""
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    # -- logit rules ------------------------------------------------------

    def process_logits(self, logits: torch.Tensor, graph_walk: str) -> torch.Tensor:
        """HF generate parity: mask the always-suppressed token set, plus
        the begin-suppressed set for the first transcript token."""
        device = logits.device
        if self._suppress_ids is None:
            self._suppress_ids = torch.tensor(self.config.suppress_tokens, dtype=torch.long, device=device)
            self._begin_suppress_ids = torch.tensor(self.config.begin_suppress_tokens, dtype=torch.long, device=device)
        if self._suppress_ids.numel():
            logits.index_fill_(-1, self._suppress_ids, float("-inf"))
        if graph_walk in FIRST_TOKEN_WALKS and self._begin_suppress_ids.numel():
            logits.index_fill_(-1, self._begin_suppress_ids, float("-inf"))
        return logits

    def _detect_language(self, logits: torch.Tensor) -> torch.Tensor:
        """The most likely language token per row. Detection is a
        classification, so it takes the argmax whatever temperature the
        request decodes with, as HF and openai-whisper do."""
        return self._restrict_to_languages(logits).argmax(dim=-1)

    def _restrict_to_languages(self, logits: torch.Tensor) -> torch.Tensor:
        """Language detection: only the ``<|xx|>`` tokens may be sampled."""
        if self._language_mask is None:
            mask = torch.full((self.config.vocab_size,), float("-inf"), device=logits.device)
            mask[torch.tensor(self.config.language_token_ids, device=logits.device)] = 0.0
            self._language_mask = mask
        return logits + self._language_mask.to(logits.dtype)

    # -- capture ------------------------------------------------------------

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        return [
            BatchedAcceleratorGraphConfig(
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
        token_ids = to_device_nonblocking(inputs["text_inputs"][0], device).reshape(-1)
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
                tensor_inputs["ts_rules"] = self._inactive_rules(device)
        else:
            state = self.request_state(fwd_info.rid_handle)
            # The route resolved these from the request's kwargs, which is
            # where they came from in the first place: scanning the prompt
            # tensor for them meant a ``.tolist()`` here, and a D2H on the GPU
            # thread waits out everything queued ahead of it — at prefill, the
            # encoder's whole 30 s window.
            meta = fwd_info.step_metadata
            # The detection walk's prompt ends at <|sot|> and has settled
            # nothing yet, so it decodes with the rules off either way.
            timestamps = (
                graph_walk != DETECT_LANGUAGE_WALK and bool(meta.get("timestamps"))
            )
            # the align walk rebuilds the forced prompt from these. The
            # language is None exactly when it is being detected; the sampled
            # token then leads the prompt walk's input, kept on the device.
            language = meta.get("language")
            if language is None and graph_walk == PREFILL_PROMPT_WALK:
                language = token_ids[:1].clone()
            if language is not None:
                state.add("language", language)
            task = meta.get("task")
            if task is not None:
                state.add("task", task)
            row = rule_state([], self.config) if timestamps else inactive_state()
            # on the host: ``preprocess`` stages the batch's rows in one copy
            tensor_inputs["ts_rules"] = torch.tensor(row, dtype=torch.long)

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
        state = self.request_state(fwd_info.rid_handle)
        generated = [int(t) for part in inputs["transcript"] for t in part.reshape(-1).tolist()]
        language = state.get("language")
        if isinstance(language, torch.Tensor):
            language = int(language.item())
        task = state.get("task", self.config.task_token("transcribe"))
        # the 4 prompt tokens and end-of-text stay inside the position table
        text = [t for t in generated if t < self.config.eos_token_id][: self.config.max_target_positions - 5]
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
        ids = input_ids.tolist()
        language = self.config.language_of(ids[1])
        words = word_timings(rows, ids[prompt_len + 1:-1], self._decode_raw, num_frames, language)
        tb, top = self.config.timestamp_begin, self.config.vocab_size - 1

        def stamp(seconds: float) -> int:
            return min(top, tb + int(round(seconds / self.config.timestamp_precision)))

        ids = [self._startoflm_id]
        for w in words:
            ids += [stamp(w["start"])] + self.tokenizer.encode(w["word"], add_special_tokens=False) + [stamp(w["end"])]
        return torch.tensor(ids, dtype=torch.long, device=input_ids.device)

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
        return super().declare_step(graph_walk, request_ids, inputs, slot_lease, piecewise_leases, **kwargs)

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
        rows = [self._row_ids(inp) for inp in inputs]
        device = rows[0].device
        rules = torch.stack([inp.tensor_inputs["ts_rules"] for inp in inputs])
        if rules.device != device:
            # Host values (prefill's rule state) go over in one pinned,
            # non-blocking copy: a torch.tensor(..., device=) from a list is a
            # blocking copy that waits out the stream (at prefill, the
            # encoder's window). Rows already on the device stay as they are.
            staged = torch.empty(rules.shape, dtype=torch.long, device=device)
            self._stager.copy_(staged.view(-1), rules.view(-1).numpy())
            rules = staged
        preprocessed: dict[str, torch.Tensor | Any] = {"input_ids": torch.cat(rows), "ts_rules": rules}
        if graph_walk != DECODE_WALK:
            # packed prefill: where each request's last token sits
            preprocessed["row_lens"] = torch.empty(len(rows), dtype=torch.long, device=device)
            self._stager.copy_(preprocessed["row_lens"], [r.shape[0] for r in rows])
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
        row_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the decoder and sample one token per request; also the
        timestamp rule state each row carries into its next step."""
        pos: PositionManager = engine_inputs.resources[POS]
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]

        if encoder_states is not None:
            # prefill only: fills the context pages this step reserved, before
            # the layers read them back
            self.decoder.write_cross_kv(encoder_states)

        input_embeds = self.decoder.embed(input_ids, pos.pos_ids("main"))
        hidden = self.decoder(input_embeds=input_embeds, label="main")

        if graph_walk != DECODE_WALK:
            # packed prefill: one hidden per request, at its last token.
            # Indexed from the rows themselves: a one-token prompt (language
            # detection) is planned as a decode step, whose attention wrapper
            # keeps no packed offsets to select by.
            hidden = self._last_rows(hidden, row_lens)

        logits = self.decoder.lm_head(hidden)
        if graph_walk == DETECT_LANGUAGE_WALK:
            new_tokens = self._detect_language(logits)
        else:
            logits = self.process_logits(logits, graph_walk)
            if ts_rules is not None:
                logits = self.timestamp_rules.apply(logits, ts_rules)
            new_tokens = sampler.sample(engine_inputs.request_ids, logits=logits)
        if ts_rules is None:
            return new_tokens, None
        return new_tokens, self.timestamp_rules.advance(ts_rules, new_tokens)

    @staticmethod
    def _last_rows(hidden: torch.Tensor, row_lens: torch.Tensor) -> torch.Tensor:
        """The last hidden state of each packed request."""
        last = row_lens.to(hidden.device).cumsum(0) - 1
        return hidden.index_select(0, last)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        ts_rules: torch.Tensor | None = None,
        audio_frames: torch.Tensor | None = None,
        row_lens: torch.Tensor | None = None,
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
            row_lens=row_lens,
        )
        out: NameToTensorList = {"new_token": [new_tokens]}
        if next_rules is not None:
            out["ts_rules"] = [next_rules[0]]
        return out

    def max_batch_size(self, graph_walk: str) -> int | None:
        if graph_walk == ALIGN_WALK:
            return 1
        return super().max_batch_size(graph_walk)

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        ts_rules: torch.Tensor | None = None,
        audio_frames: torch.Tensor | None = None,
        row_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput | dict[str, NameToTensorList]:
        if graph_walk == ALIGN_WALK:
            (rid,) = engine_inputs.request_ids
            return {rid: {"word_tokens": [self._align(input_ids, encoder_states, int(audio_frames.reshape(-1)[0]))]}}
        new_tokens, next_rules = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            encoder_states=encoder_states,
            ts_rules=ts_rules,
            row_lens=row_lens,
        )
        # row i is request i: one clone and one D2H copy per step, not per request
        rows = {"new_token": new_tokens}
        if next_rules is not None:
            rows["ts_rules"] = next_rules
        return BatchedModelOutput(row_outputs=rows, check_stop_buffers={"new_token": new_tokens})

    def _stops(self, token: int, request_info: CurrentForwardPassInfo) -> bool:
        # only a decode step's token ends the loop: the prefill walks run
        # outside it, and detection samples a language, not a transcript token
        return request_info.graph_walk == DECODE_WALK and super()._stops(token, request_info)
