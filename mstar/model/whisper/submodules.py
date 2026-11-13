# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Whisper ASR
# ---------------------------------------------------------------------------
#
# Two submodules covering the encoder-decoder pipeline:
#   1. WhisperEncoderSubmodule  (no resources) — native encoder, batched over
#      requests and captured as one CUDA graph per batch size
#   2. WhisperDecoderSubmodule  (two KV arrays, the decoder state and a sampler)
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
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    at_last_token,
    last_sampled,
    position,
    positions,
    sample,
)
from mstar.model.components.audio_features import LogMelSpectrogram
from mstar.model.components.token_step import TokenStepNode
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    TokenStop,
    splice_loopback_ids,
)
from mstar.model.whisper.components import alignment
from mstar.model.whisper.components.decoder import WhisperDecoderModel
from mstar.model.whisper.components.encoder import WhisperEncoderModel
from mstar.model.whisper.components.timestamps import TimestampRules, inactive_state, rule_state
from mstar.model.whisper.config import (
    ALIGN_WALK,
    CONTEXT_LABEL,
    CROSS_KV_CACHE,
    DECODE_LOOP,
    DECODE_WALK,
    DECODER_STATE,
    DETECT_LANGUAGE_WALK,
    KV_CACHE,
    PREFILL_PROMPT_WALK,
    PREFILL_WALK,
    SAMPLER,
    WhisperModelConfig,
)
from mstar.utils.h2d import PinnedStager, to_device_nonblocking

logger = logging.getLogger(__name__)


class _PromptPackedConfig(PackedAcceleratorGraphConfig):
    """Packed prompt buckets, kept only where a bucket gives every row 3 to 8
    tokens: Whisper prompts are 3 or 4 tokens a request, so a sparser or denser
    bucket would never be chosen, and fewer than one token a row can't be
    captured at all."""

    def get_total_tokens(self, bs: int) -> list[int]:
        return [n for n in super().get_total_tokens(bs) if 3 * bs <= n <= 8 * bs]

# Walks that run the encoder and hand the decoder its output.
ENCODER_WALKS = (PREFILL_WALK, DETECT_LANGUAGE_WALK)
# Walks whose sampled token is the first transcript token, subject to the
# begin-suppression rules; a stop id there ends the request.
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

    # Every size up to 8: an encoder step pads to the next captured size, and at
    # serving concurrency its batches are small (2-3 typical), so a 3 that ran as
    # a 4 spent a quarter of the step's GPU time on a padding row.
    ENCODER_CAPTURE_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]
    # Compiled before capture, like the engine's other captured forwards: at
    # serving concurrency the encoder is most of the GPU time (~6 ms a request
    # on an H200). Turbo, split+MPS, one H200: c=32 +9%, c=64 +20%, measured
    # when the default compile was max-autotune (now MSTAR_COMPILE_LEVEL=2);
    # not yet measured at the default level 1.
    ENCODER_COMPILE = True
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


class WhisperDecoderSubmodule(TokenStepNode, ARNodeSubmodule):
    """Autoregressive Whisper decoder.

    Dispatches on graph_walk:
      - prefill: embed the forced decoder prompt
        (``[<|startofprev|> prev...] <|startoftranscript|><|lang|><|task|>
        [<|notimestamps|>]``), project ``encoder_states`` to per-layer
        cross-attention K/V and add them to the context array, fill
        the self-attention KV cache, and sample the first transcript token.
      - detect_language: the same with a prompt that ends at
        ``<|startoftranscript|>``; sampling is restricted to language tokens.
      - prefill_prompt: append ``<|lang|><|task|>[<|notimestamps|>]`` after a
        detected language token, over the already added context.
      - decode: embed the previous token, single-step decode.
      - align: after the loop, when word timestamps were asked for: one
        teacher-forced pass over ``<|sot|><|lang|><|task|><|notimestamps|>
        transcript <|eot|>`` outside the caches; emits the alignment heads'
        smoothed cross-attention with the token ids
        (``alignment.alignment_output``). ``WhisperModel.postprocess``
        turns it into word timings in the API server.

    Both KV arrays belong to the step: the self-attention's grows by the
    token count, the context by the encoder output at the walk that adds it,
    and it is declared zero-span (read-only) thereafter.

    Not an ``EncoderDecoderNode`` like Canary's decoder, though both add the
    context eagerly in ``preprocess`` so the prefill walk replays a graph sized
    by the prompt alone. The token feedback and batch caps are
    ``TokenStepNode``'s; the base stop condition is its ``token_stop``.
    """

    SAMPLER = SAMPLER
    DECODE_WALK = DECODE_WALK
    DECODE_LOOP = DECODE_LOOP
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # `can_batch` would otherwise take the whole ready set. A set too big for
    # the context cache can't be admitted. With offload configured it evicts and
    # retries, without it there is nothing to evict and the batch reforms
    # identically.
    MAX_PREFILL_BATCH_SIZE = 32
    # The prefill walk replays a graph over its prompt tokens (4 a request);
    # these are packed prompt-token totals, kept where a bucket holds 3 to 8
    # tokens a row (see `_PromptPackedConfig`).
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    PREFILL_TOKEN_BUCKETS = [4, 8, 16, 32, 64, 128]
    # See the encoder: dynamo recompiled the prefill walks per request id
    # (a 1-2 s stall each, RTFx 158 at c=32 before the fix). Decode and
    # prefill steps are graph replays; the walks they don't cover run eagerly.
    disable_torch_compile = True

    def __init__(self, decoder: WhisperDecoderModel, config: WhisperModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        # a stop on the first transcript token ends the partition (`_token_stops`)
        self.token_stop = TokenStop(
            loop=self.DECODE_LOOP, buffer="new_token",
            stop_ids=frozenset({config.eos_token_id}), sampler=self.SAMPLER,
            prefill_walks=frozenset(FIRST_TOKEN_WALKS),
        )
        # The decode walk reads its last token off the sampler's slot masters.
        self.device_loopback = DeviceLoopback(
            walk=DECODE_WALK, signal="text_inputs", sampler=self.SAMPLER, buffer="new_token",
        )
        self._suppress_ids: torch.Tensor | None = None
        self._begin_suppress_ids: torch.Tensor | None = None
        self._language_mask: torch.Tensor | None = None
        self.timestamp_rules = TimestampRules(config)
        # sized for the align walk's longest sequence, so staging it never
        # grows the ring (which waits out the copies in flight)
        self._stager = PinnedStager(torch.long, numel=config.max_target_positions)

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv, self.state, self.sampler = handles[KV_CACHE], handles[DECODER_STATE], handles[SAMPLER]

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
            *super().get_accelerator_graph_configs(device, tp_world_size),
            # The prefill's prompt pass. Its encoder context (1500 tokens a
            # request) is added eagerly in `preprocess` (`ArrayStep.eager_add`),
            # so the graph is sized by the prompt alone. Eager, the prefill was
            # four layers of launches plus the logit rules: ~15% of turbo's
            # decoder steps at c=8 and a quarter of their host time.
            _PromptPackedConfig(
                capture_graph_walk=PREFILL_WALK,
                make_node_input=lambda n: ARNodeInputs(
                    input_ids=torch.zeros(n, dtype=torch.long, device=device),
                    input_seq_len=n,
                    tensor_inputs={
                        # only its length is read (the context span); one
                        # broadcast row, not 1500, per cached template row
                        "encoder_states": torch.zeros(
                            1, self.config.d_model,
                            dtype=next(self.parameters()).dtype, device=device,
                        ).expand(self.config.max_source_positions, -1),
                        # on the host like a real prefill row's: padding rows
                        # are stacked with those in `preprocess`
                        "ts_rules": torch.tensor(inactive_state(), dtype=torch.long),
                    },
                ),
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
                # per-request rows, not tokens; and capture alone is the win
                input_seq_dims={"ts_rules": 0, "row_lens": 0},
                compile=False,
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
        if (row := self.uniform_row_inputs(graph_walk, kwargs.get("has_loopback", False))) is not None:
            # the token is read by slot in `preprocess`
            return row
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

        # A decode step continues the rule state its previous step left in the
        # decoder state: the worker launches it before the previous token is
        # readable on the host, so it cannot be rebuilt from a Python history.
        if graph_walk != DECODE_WALK:
            # From the route's metadata, not the prompt tensor: scanning that
            # meant a ``.tolist()`` here, and a D2H on the GPU thread waits out
            # everything queued ahead of it (at prefill, the encoder's window).
            # The detection walk's prompt ends at <|sot|> and has settled
            # nothing yet, so it decodes with the rules off either way.
            timestamps = (
                graph_walk != DETECT_LANGUAGE_WALK and bool(fwd_info.step_metadata.get("timestamps"))
            )
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
        """What `preprocess` builds the align walk's teacher-forcing sequence
        from, once it has read the transcript off the decoder state."""
        return ARNodeInputs(
            # The decoder state's one entry: the sequence's length is known
            # only in `preprocess`. Nothing budgets on it, since no chunking
            # policy splits the walk and it runs one request a step.
            input_seq_len=1,
            tensor_inputs={"encoder_states": inputs["encoder_states"][0].to(self.get_device())},
            kwargs={
                **fwd_info.step_metadata,
                "ignore_eos": fwd_info.resource_configs[SAMPLER].ignore_eos,
            },
        )

    def _alignment_inputs(self, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs]) -> dict[str, Any]:
        """The teacher-forcing sequence for the align walk: the forced prompt
        with ``<|notimestamps|>`` (word timing never uses timestamp tokens),
        the transcript's text tokens, end-of-text. The transcript comes to the
        host in one read on the GPU thread, which waits out the stream once per
        word-timestamp request."""
        (inp,), (rid,) = inputs, engine_inputs.request_ids  # one request per align step
        meta = inp.kwargs
        eos = self.config.eos_token_id
        fields = self.state.read(0).fields
        # through the next position, which holds the last sampled token
        row = fields["transcript"][0, : position(self.kv, rid, "main") + 1].tolist()
        language = row[meta["language_slot"]] if meta["language"] is None else meta["language"]
        ids = row[meta["transcript_start"]:]
        # An end-of-text from a decode step stopped the loop, and a speculative
        # step may have written past it. One from the prompt walk ended the
        # request before this step, and under ``ignore_eos`` none stops it.
        if not meta["ignore_eos"] and eos in ids[1:]:
            ids = ids[: ids.index(eos, 1)]
        # the 4 prompt tokens and end-of-text stay inside the position table
        text = [t for t in ids if t < eos][: self.config.max_target_positions - 5]
        seq = [self.config.decoder_start_token_id, language, meta["task"], self.config.no_timestamps_token_id]
        seq += text + [eos]
        # back unchanged: a kept step writes every field of its rows
        self.state.write(0, **fields)
        staged = torch.empty(len(seq), dtype=torch.long, device=self.get_device())
        self._stager.copy_(staged, seq)
        return {
            "input_ids": staged,
            "encoder_states": inp.tensor_inputs["encoder_states"],
            "audio_frames": meta["audio_frames"],
        }

    def _align(self, input_ids: torch.Tensor, encoder_states: torch.Tensor, audio_frames: int) -> torch.Tensor:
        """The alignment heads' matrix and the token ids for one request
        (``alignment.alignment_output``); words are timed in postprocess."""
        heads = [(int(layer), int(head)) for layer, head in self.config.alignment_heads]
        weights = self.decoder.cross_attention_weights(input_ids, encoder_states, heads)
        return alignment.alignment_output(weights, input_ids, audio_frames)

    @staticmethod
    def _context_span(inp: ARNodeInputs) -> int:
        """How far this step grows the request's context stream: the encoder
        output on the walk that writes it, zero after."""
        encoder_states = inp.tensor_inputs.get("encoder_states")
        return 0 if encoder_states is None else encoder_states.shape[0]

    def resident_walks(self) -> frozenset[str]:
        # one token a row in the main stream; the context is read, not written,
        # and `preprocess` reads no planned position or KV location
        return frozenset({DECODE_WALK})

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep | None:
        # the default, for the self-attention array and the decoder state
        segments = [
            Segment(
                request_id=rid,
                label="main",
                span=inp.input_seq_len,
            ) for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        if graph_walk == ALIGN_WALK:
            # An eager pass outside the caches that reads the transcript. Its
            # segments are the decoder state's entries rather than the rows the
            # decoder's projections see, which a LoRA would need. Whisper
            # serves no LoRA.
            return SubmoduleStep(segments=segments, state={DECODER_STATE: ArrayStep()})
        context_segments = tuple(
            Segment(
                request_id=rid,
                label=CONTEXT_LABEL,
                span=self._context_span(inp),
            ) for rid, inp in zip(request_ids, inputs, strict=True)
        )
        return SubmoduleStep(
            segments=segments,
            state={
                KV_CACHE: ArrayStep(),
                # The context: a real span on the prefill that adds it, zero
                # afterwards. The commit is what turns the prefill's
                # reservation into resident pages the later steps read. The
                # add happens in `preprocess`, outside any graph.
                CROSS_KV_CACHE: ArrayStep(segments=context_segments, eager_add=graph_walk in ENCODER_WALKS),
                DECODER_STATE: ArrayStep(),
                # detection takes an argmax and keeps no sample
                SAMPLER: SampleStep(
                    apply_penalty=False, kept_rows=frozenset() if graph_walk == DETECT_LANGUAGE_WALK else None,
                ),
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
            return self._alignment_inputs(engine_inputs, inputs)
        if graph_walk == DECODE_WALK:
            if all(inp.input_ids is not None for inp in inputs):
                return {"input_ids": torch.cat([inp.input_ids for inp in inputs])}
            # Device loop-back reads each row's last sampled token by slot. A
            # row whose last step ran elsewhere brought its token.
            ids = last_sampled(self.sampler)
            if any(inp.input_ids is not None and inp.input_seq_len for inp in inputs):
                ids = splice_loopback_ids(ids, inputs, decode=True)
            return {"input_ids": ids}
        rows = [self._row_ids(inp) for inp in inputs]
        device = rows[0].device
        # Host values (the prompt walks' rule state) go over in one pinned,
        # non-blocking copy: a torch.tensor(..., device=) from a list is a
        # blocking copy that waits out the stream (at prefill, the encoder's
        # window).
        host_rules = torch.stack([inp.tensor_inputs["ts_rules"] for inp in inputs])
        rules = torch.empty(host_rules.shape, dtype=torch.long, device=device)
        self._stager.copy_(rules.view(-1), host_rules.view(-1).numpy())
        # packed prefill: where each request's last token sits
        row_lens = torch.empty(len(rows), dtype=torch.long, device=device)
        self._stager.copy_(row_lens, [r.shape[0] for r in rows])
        if graph_walk in ENCODER_WALKS:
            # Fill the context pages this step reserved, here rather than in the
            # forward: the prefill graph is sized by the prompt, not by 1500
            # encoder tokens a request (`ArrayStep.eager_add`). Concatenated in
            # segment order, which is how the context plan laid the pages out;
            # a replay's padding rows come after and are not written.
            self.decoder.add_context(torch.cat(
                [inp.tensor_inputs["encoder_states"] for inp in inputs], dim=0,
            ))
        return {"input_ids": torch.cat(rows), "ts_rules": rules, "row_lens": row_lens}

    def _forward(
        self,
        graph_walk: str,
        input_ids: torch.Tensor,
        ts_rules: torch.Tensor | None,
        row_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the decoder and sample one token per request, and keep the
        token and the timestamp rule state it leads to in the decoder state."""
        state = self.state.read(0).fields
        pos = positions(self.kv)[: input_ids.shape[0]]
        hidden = self.decoder(self.decoder.embed(input_ids, pos))

        if graph_walk == PREFILL_WALK:
            # packed prefill: one hidden per request, at its last token, from
            # the plan's offsets (a replay's row_lens buffer holds stale rows
            # past the real ones). The prompt is several tokens a request.
            hidden, last = at_last_token(self.kv, hidden), at_last_token(self.kv, pos)
        elif graph_walk != DECODE_WALK:
            # Indexed from the rows themselves: a one-token prompt (language
            # detection) is planned as a decode step, whose attention wrapper
            # keeps no packed offsets to select by.
            hidden, last = self._last_rows(hidden, row_lens), self._last_rows(pos, row_lens)
        else:
            last = pos

        rules = state["ts_rules"] if graph_walk == DECODE_WALK else ts_rules
        logits = self.decoder.lm_head(hidden)
        if graph_walk == DETECT_LANGUAGE_WALK:
            new_tokens = self._detect_language(logits)
        else:
            logits = self.timestamp_rules.apply(self.process_logits(logits, graph_walk), rules)
            new_tokens = sample(self.sampler, logits)
        # An eager draw's tokens are int32.
        transcript = state["transcript"].scatter_(1, (last + 1)[:, None], new_tokens[:, None].long())
        self.state.write(0, ts_rules=self.timestamp_rules.advance(rules, new_tokens), transcript=transcript)
        return new_tokens

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
        audio_frames: int | None = None,
        row_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if graph_walk == ALIGN_WALK:
            return {"alignment": [self._align(input_ids, encoder_states, audio_frames)]}
        return {"new_token": [self._forward(graph_walk, input_ids, ts_rules, row_lens)]}

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
        audio_frames: int | None = None,
        row_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList] | BatchedModelOutput:
        if graph_walk == ALIGN_WALK:
            (rid,) = engine_inputs.request_ids
            return BatchedModelOutput(
                per_rid_outputs={rid: {"alignment": [self._align(input_ids, encoder_states, audio_frames)]}},
                # nothing to stop on: no host copy of the matrix for the stop check
                check_stop_buffers={},
            )
        new_tokens = self._forward(graph_walk, input_ids, ts_rules, row_lens)
        # Row i is request i: the engine clones each batch tensor once and hands
        # out row views, and the stop check reads the tokens off one D2H copy,
        # instead of a clone and a copy per request per step. A decode step's
        # token is its loop's `text_inputs`; the prefill's is the persisted
        # first token.
        signal = "text_inputs" if graph_walk == DECODE_WALK else "new_token"
        return BatchedModelOutput(row_outputs={signal: new_tokens}, check_stop_buffers={"new_token": new_tokens})

    def _token_stops(self, request_id: str, token: int, info: CurrentForwardPassInfo) -> bool:
        """A stop id on a decode row's token, or on the first transcript token,
        where it ends the request (an empty transcript has no words to align).
        No budget check at the prompt walk: the loop cap ends decode, and the
        walk plan skips decode for a one-token budget, so a word-timestamp
        request still aligns that token."""
        walk = info.graph_walk
        if walk != DECODE_WALK:
            return (
                walk in FIRST_TOKEN_WALKS and token in self.token_stop.stop_ids
                and not info.resource_configs[SAMPLER].ignore_eos
            )
        return super()._token_stops(request_id, token, info)
