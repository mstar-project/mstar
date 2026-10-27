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
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.position.manager import PositionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.audio_features import LogMelSpectrogram
from mstar.model.components.token_step import TokenStepNode
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    device_loopback_enabled,
)
from mstar.model.whisper.components import alignment
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
    TS_RULES,
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

    # Every size up to 8: an encoder step pads to the next captured size, and at
    # serving concurrency its batches are small (2-3 typical), so a 3 that ran as
    # a 4 spent a quarter of the step's GPU time on a padding row.
    ENCODER_CAPTURE_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]
    # Compiled before capture, like the engine's other captured forwards: at
    # serving concurrency the encoder is most of the GPU time (~6 ms a request
    # on an H200), and Inductor's kernels cut it. Turbo, split+MPS, one H200:
    # c=32 +9%, c=64 +20%. The first start on a machine autotunes every bucket
    # (minutes); the Inductor cache makes later starts ~1 min.
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


_DECODE_LOOPBACK_WRITES = frozenset({TS_RULES})


class WhisperDecoderSubmodule(TokenStepNode, ARNodeSubmodule):
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
        transcript <|eot|>`` outside the caches; emits the alignment heads'
        smoothed cross-attention with the token ids
        (``alignment.alignment_output``). ``WhisperModel.postprocess``
        turns it into word timings in the API server.

    Both cache streams belong to the step: ``main`` grows by the token
    count, ``context`` grows by the encoder output at the walk that writes it
    and is declared zero-span (read-only) thereafter.

    Not an ``EncoderDecoderNode`` like Canary's decoder: that node writes the
    context inside the prefill forward and captures only decode, while this
    one writes it eagerly in ``preprocess`` so the prefill walk can replay a
    graph sized by the prompt alone. The token feedback, batch caps and
    base stop condition are ``TokenStepNode``'s.
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
    # The decode walk reads its last token off the sampler's slot masters and
    # its timestamp rule state from the sampler's declared loop-back state
    # beside it (`device_loopback_signals`).
    reads_device_loopback = True

    def __init__(self, decoder: WhisperDecoderModel, config: WhisperModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self.stop_token_ids = frozenset({config.eos_token_id})
        self._suppress_ids: torch.Tensor | None = None
        self._begin_suppress_ids: torch.Tensor | None = None
        self._language_mask: torch.Tensor | None = None
        self.timestamp_rules = TimestampRules(config)
        # sized for the align walk's longest sequence, so staging it never
        # grows the ring (which waits out the copies in flight)
        self._stager = PinnedStager(torch.long, numel=config.max_target_positions)
        self._inactive: torch.Tensor | None = None

    def _inactive_rules(self, device: torch.device) -> torch.Tensor:
        """The rules-off state, built once: rows are only read (stacked)."""
        if self._inactive is None or self._inactive.device != device:
            self._inactive = torch.tensor(inactive_state(), dtype=torch.long, device=device)
        return self._inactive

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
                    # the timestamp rule state rides along as a staged row,
                    # on the device like a real decode row's: padding rows
                    # are stacked with those in `preprocess`
                    tensor_inputs={"ts_rules": self._inactive_rules(device)},
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            # The prefill's prompt pass. Its encoder context (1500 tokens a
            # request) is written eagerly in `preprocess` (`KVStep.eager_write`),
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
        if graph_walk == DECODE_WALK and kwargs.get("has_loopback") and self._device_loopback():
            # the token and its rule state are read by slot in `preprocess`
            return ARNodeInputs(input_seq_len=1)
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
        """The teacher-forcing sequence for the align walk: the forced prompt
        with ``<|notimestamps|>`` (word timing never uses timestamp tokens),
        the transcript's text tokens, end-of-text. Every value is on the host
        already, so the sequence goes over in ``preprocess``'s pinned copy."""
        meta = fwd_info.step_metadata
        state = self.request_state(fwd_info.rid_handle)
        # The prompt walks' stop check recorded the transcript's first token
        # and a detected language, on this worker: every decoder walk runs on
        # the same ranks (`WhisperModel.validate_config_yaml`).
        decoded = state.get("decoded", None)
        language = meta.get("language")
        if language is None:
            language = state.get("language", None)
        if decoded is None:
            raise RuntimeError(
                f"request {fwd_info.request_id!r}: the align walk found no transcript recorded; the "
                f"{PREFILL_WALK!r} or {PREFILL_PROMPT_WALK!r} walk's stop check records it, on the worker "
                "that ran it"
            )
        if language is None:
            raise RuntimeError(
                f"request {fwd_info.request_id!r}: the align walk found no language; the "
                f"{DETECT_LANGUAGE_WALK!r} walk's stop check records the detected one, on the worker "
                "that ran it"
            )
        # the 4 prompt tokens and end-of-text stay inside the position table
        text = [t for t in decoded if t < self.config.eos_token_id][: self.config.max_target_positions - 5]
        seq = [self.config.decoder_start_token_id, language, meta["task"], self.config.no_timestamps_token_id]
        seq += text + [self.config.eos_token_id]
        return ARNodeInputs(
            input_seq_len=len(seq),
            input_ids=torch.tensor(seq, dtype=torch.long),
            tensor_inputs={"encoder_states": inputs["encoder_states"][0].to(self.get_device())},
            kwargs={"audio_frames": meta["audio_frames"]},
        )

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
                # reservation into resident pages the later steps read. The
                # write happens in `preprocess`, outside any graph.
                CROSS_KV_CACHE: KVStep(
                    segments=context_segments, eager_write=graph_walk in ENCODER_WALKS,
                ),
                CROSS_ATTN: AttentionStep(
                    segments=context_segments, causal=False,
                ),
                SAMPLER: SamplerStep(
                    apply_penalty=False,
                    # the decode forward stores its rows' advanced rule state
                    # (persisted only on the device loop-back path)
                    loopback_writes=_DECODE_LOOPBACK_WRITES if graph_walk == DECODE_WALK else frozenset(),
                ),
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
            ids = torch.empty(inp.input_seq_len, dtype=torch.long, device=self.get_device())
            self._stager.copy_(ids, inp.input_ids.numpy())
            return {
                "input_ids": ids,
                "encoder_states": inp.tensor_inputs["encoder_states"],
                "audio_frames": inp.kwargs["audio_frames"],
            }
        if graph_walk == DECODE_WALK and any(inp.input_ids is None for inp in inputs):
            return self._loopback_inputs(engine_inputs, inputs)
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
            # Fill the context pages this step reserved, here rather than in the
            # forward: the prefill graph is sized by the prompt, not by 1500
            # encoder tokens a request (`KVStep.eager_write`). Concatenated in
            # segment order, which is how the context plan laid the pages out;
            # a replay's padding rows come after and are not written.
            self._write_context(torch.cat(
                [inp.tensor_inputs["encoder_states"] for inp in inputs], dim=0,
            ))
        return preprocessed

    # -- device loop-back --------------------------------------------------

    def _device_loopback(self) -> bool:
        return device_loopback_enabled() and self.node_resources[SAMPLER].has_slot_masters

    def device_loopback_signals(self, graph_walk: str) -> frozenset[str]:
        if graph_walk == DECODE_WALK and self._device_loopback():
            return frozenset({"text_inputs", "ts_rules"})
        return frozenset()

    def uniform_row_inputs(self, graph_walk: str, has_loopback: bool = False) -> ARNodeInputs | None:
        if graph_walk == DECODE_WALK and has_loopback and self._device_loopback():
            return ARNodeInputs(input_seq_len=1)
        return None

    def inline_client_signals(self, graph_walk: str) -> dict[str, str]:
        # the decode loop's client edge carries the token the stop check
        # already has on the host; the prefill's new_token edge is persisted
        return {"text_inputs": "new_token"} if graph_walk == DECODE_WALK else {}

    def _loopback_inputs(
        self, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        """A decode step's token and rule state for every row, read by sampler
        slot (padding rows read the trash row); a row whose last step did not
        keep them on the device (the first decode step, after the prefill)
        brings them routed and they override its slot's."""
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        ids = sampler.loopback_tokens(engine_inputs.request_ids)
        rules = sampler.loopback_state(TS_RULES, engine_inputs.request_ids)
        routed = [i for i, inp in enumerate(inputs) if inp.input_ids is not None]
        if routed:
            sampler.overwrite_rows(
                routed,
                (ids, torch.cat([inputs[i].input_ids for i in routed])),
                (rules, torch.stack([inputs[i].tensor_inputs["ts_rules"] for i in routed])),
            )
        return {"input_ids": ids, "ts_rules": rules}

    def _write_context(self, encoder_states: torch.Tensor) -> None:
        self.decoder.write_cross_kv(encoder_states)

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

        # the prefill's context pages were filled in `preprocess`
        del encoder_states
        input_embeds = self.decoder.embed(input_ids, pos.pos_ids("main"))
        hidden = self.decoder(input_embeds=input_embeds, label="main")

        if graph_walk == PREFILL_WALK:
            # packed prefill: one hidden per request, at its last token, from
            # the plan's offsets (a replay's row_lens buffer holds stale rows
            # past the real ones). The prompt is several tokens a request.
            hidden = engine_inputs.resources[ATTN].select_last_hidden(hidden)
        elif graph_walk != DECODE_WALK:
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
        audio_frames: int | None = None,
        row_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if graph_walk == ALIGN_WALK:
            return {"alignment": [self._align(input_ids, encoder_states, audio_frames)]}
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
        new_tokens, next_rules = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            encoder_states=encoder_states,
            ts_rules=ts_rules,
            row_lens=row_lens,
        )
        # Row i is request i: the engine clones each batch tensor once and hands
        # out row views, and the stop check reads the tokens off one D2H copy,
        # instead of a clone and a copy per request per step. A ``ts_rules``
        # row arrives as [1, k]; ``prepare_inputs`` flattens it. A decode
        # step's token is its loop's `text_inputs`; the prefill's is the
        # persisted first token.
        if graph_walk == DECODE_WALK:
            rows = {"text_inputs": new_tokens}
            if next_rules is not None and self._device_loopback():
                engine_inputs.resources[SAMPLER].store_loopback_state(
                    TS_RULES, engine_inputs.request_ids, next_rules,
                )
        else:
            rows = {"new_token": new_tokens}
        if next_rules is not None:
            rows["ts_rules"] = next_rules
        return BatchedModelOutput(row_outputs=rows, check_stop_buffers={"new_token": new_tokens})

    def _stops(self, token: int, request_info: CurrentForwardPassInfo) -> bool:
        """The shared stop condition (EOS or the token cap) on a decode step's
        token; a prompt walk's token never stops the loop. Either way the token
        is kept for the align walk first."""
        if request_info.graph_walk != DECODE_WALK:
            self._record_prompt_token(request_info, token)
            return False
        self._record(request_info, token)
        if super()._stops(token, request_info):
            self.request_state(request_info.rid_handle).add("stopped", True)
            return True
        return False

    def _record_prompt_token(self, info: CurrentForwardPassInfo, token: int) -> None:
        """For a word-timestamp request, keep what the align walk needs off a
        prompt walk's host token: the detected language, or the first
        transcript token, which starts the transcript ``_record`` extends."""
        # Interim: the transcript lives in request state until the decode
        # loop's outputs reach the align walk directly.
        if not info.step_metadata.get("word_timestamps"):
            return
        state = self.request_state(info.rid_handle)
        if info.graph_walk == DETECT_LANGUAGE_WALK:
            state.add("language", token)
        else:
            state.add("decoded", [token])

    def _record(self, info: CurrentForwardPassInfo, token: int) -> None:
        """Keep a decode step's token for the align walk (the loop no longer
        persists them), if the request has a transcript started. Nothing after
        the step that stopped the loop: a speculative step past it must not
        extend the transcript. Once per step: when the batched stop check
        raises partway, the engine re-runs it per request, and a step whose
        token is already in (after the prompt's, one per earlier iteration) is
        not appended again."""
        state = self.request_state(info.rid_handle)
        decoded = state.get("decoded", None)
        if (
            decoded is not None and not state.get("stopped", False)
            and len(decoded) == 1 + info.dynamic_loop_iter_counts.get(DECODE_LOOP, 0)
        ):
            decoded.append(token)

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
        step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        """``check_stop`` for every request off one ``tolist``."""
        if step_walk is not None and step_walk != DECODE_WALK:
            # as check_stop: a prompt walk's token never stops the loop, and
            # only a word-timestamp request keeps it
            request_ids = [rid for rid in request_ids if request_infos[rid].step_metadata.get("word_timestamps")]
            if not request_ids:
                return {}
        pairs = host_rows.first_values("new_token", request_ids)
        if pairs is None:
            return None
        return {rid: {DECODE_LOOP} for rid, token in pairs if self._stops(token, request_infos[rid])}
