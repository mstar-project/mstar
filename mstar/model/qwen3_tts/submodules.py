# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Qwen3-TTS
# ---------------------------------------------------------------------------
#
# Two submodules cover the complete text-to-speech streaming pipeline:
#   1. TalkerSubmodule (KV_CACHE engine)
#      - Builds the official prefill sequence for every 12 Hz variant:
#        optional instruction turn, ChatML assistant role, codec language /
#        speaker tags, then the text either whole (non-streaming layout) or
#        one token per generated frame (streaming layout).
#      - Maintains the Talker paged KV cache across 12 Hz decode steps.
#      - Predicts codec group 0 with the Talker and groups 1-15 with the
#        depth-wise CodePredictor.
#      - Supports continuous batching and whole-forward decode CUDA Graphs
#        (the CodePredictor depth loop is captured inside them), plus a
#        piecewise graph covering that loop on eager paths like prefill.
#   2. CodecSubmodule (STATELESS engine)
#      - Receives buffered codec frames from the Talker partition.
#      - Pads each window of the chunk ramp to its CUDA Graph capture bucket.
#      - Runs the official speech-tokenizer decoder and trims overlap (and, for
#        voice clones, the reference frames) before emitting 24 kHz PCM.
#   3. RefEncoderSubmodule (no resources; Base voice clone only)
#      - Turns one reference clip into the ECAPA x-vector that replaces the
#        speaker tag and, for in-context cloning, into its codec frames.
#
# Engine-facing lifecycle:
#   prepare_inputs -> preprocess -> forward/forward_batched
#                  -> postprocess -> check_stop (Talker only)
#
# Streaming topology:
#   Talker --[codec_tokens, ScheduledLeftContextChunkPolicy((1, 3, 8, 16), 25, 25)]--> Codec
# ---------------------------------------------------------------------------

from __future__ import annotations

import itertools
import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.audio_watermark import AudioSealWatermarker, WatermarkStream
from mstar.model.qwen3_tts.components.conv_stack import causal_receptive_field_frames, conv_stack
from mstar.model.qwen3_tts.components.snake import fuse_snake_activations
from mstar.model.qwen3_tts.components.speaker_encoder import (
    Qwen3TTSMelFrontEnd,
    Qwen3TTSSpeakerEncoder,
)
from mstar.model.qwen3_tts.components.talker import (
    Qwen3TTSCodePredictor,
    Qwen3TTSTalkerModel,
)
from mstar.model.qwen3_tts.config import (
    CHATML_ASSISTANT_PREFIX_TOKEN_IDS,
    CHATML_ASSISTANT_SUFFIX_TOKEN_IDS,
    CODE_PRED_SAMPLER,
    LANGUAGE_ID_KEY,
    PROMPT_LAYOUT_KEY,
    REF_CLIP_KEY,
    SPEAKER_ID_KEY,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    TALKER_SAMPLER,
    Qwen3TTSModelConfig,
)
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    StopCheck,
)
from mstar.streaming.stream_buffer import StreamChunkInfo
from mstar.utils.h2d import PinnedStager

# the CodePredictor depth loop, as a piecewise capture region
DEPTH_LOOP_REGION = "code_predictor_loop"
# what the first depth pass feeds: the Talker hidden state and the group-0 embedding
FIRST_PASS_TOKENS = 2


class DepthAddressing(NamedTuple):
    """Each depth token's position and batch row, for one batch size. The tensors
    outlive a capture, so a replay spends no launches rebuilding them."""

    first_positions: torch.Tensor  # [batch * FIRST_PASS_TOKENS], the first pass
    first_rows: torch.Tensor  # [batch * FIRST_PASS_TOKENS]
    positions: torch.Tensor  # [groups, batch], the single-token passes
    rows: torch.Tensor  # [batch]


def prefill_rows(prompt_layout: Sequence[int], language_id: int, speaker: bool) -> int:
    """The rows `TalkerSubmodule._build_prefill` writes, where ``speaker`` is True when a
    speaker tag or x-vector takes a row."""
    instruct_len, text_len, stream_text, ref_text_len, ref_frames = prompt_layout
    prefix = len(CHATML_ASSISTANT_PREFIX_TOKEN_IDS)
    # the role, then every codec tag but codec BOS
    rows = instruct_len + prefix + (3 if language_id < 0 else 4) + int(speaker) + 1
    if ref_frames:
        # codec BOS and the reference frames. When the text isn't streamed, both transcripts and TTS EOS come first
        text = ref_text_len - prefix - 2 + text_len + 1
        return rows + 1 + ref_frames + (0 if stream_text else text)
    # a streamed prompt adds its first text token over codec BOS. Any other adds its text and TTS EOS,
    # then TTS PAD over codec BOS
    return rows + (1 if stream_text else text_len + 2)


# ===========================================================================
# 1. TalkerSubmodule - autoregressive 12 Hz codec-frame generation
# ===========================================================================

logger = logging.getLogger(__name__)


class TalkerSubmodule(ARNodeSubmodule):
    """Run text/voice prefill and produce one 16-code frame per AR step.

    Codec group 0 is sampled from the main Talker head. The residual
    CodePredictor then walks groups 1-15 within the same frame. The sum of all
    16 codec embeddings becomes the recurrent input for the next Talker step;
    the complete code vector is streamed independently to ``CodecSubmodule``.
    """

    # Compiling the entire engine-facing forward traces request sampling,
    # residual-code control flow, and recurrent routing in addition to the
    # transformer. It introduces graph breaks and showed no steady-state win
    # for this path. The whole decode forward (CodePredictor loop included) is
    # still CUDA-graph captured.
    disable_torch_compile = True
    MAX_BATCH_SIZE = 32
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    # Prefill prompts are short (a sentence plus the ChatML frame, an optional
    # instruction and speaker slot): a few token buckets cover them, and the
    # eager alternative costs ~20 ms of launch overhead per request.
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    CHATML_ASSISTANT_PREFIX_TOKEN_IDS = CHATML_ASSISTANT_PREFIX_TOKEN_IDS
    CHATML_ASSISTANT_SUFFIX_TOKEN_IDS = CHATML_ASSISTANT_SUFFIX_TOKEN_IDS

    def __init__(
        self,
        talker_model: Qwen3TTSTalkerModel,
        code_predictor: Qwen3TTSCodePredictor,
        config: Qwen3TTSModelConfig,
    ) -> None:
        super().__init__()
        self.model = talker_model
        self.code_predictor = code_predictor
        self.config = config
        self.talker_config = config.talker
        self.cp_config = config.talker.code_predictor
        self.num_codes = config.talker.num_code_groups
        self._suppress_mask: torch.Tensor | None = None
        self._decode_last_token_indices: torch.Tensor | None = None
        self._flag_stager = PinnedStager(torch.bool, numel=self.MAX_BATCH_SIZE)
        self._index_stager = PinnedStager(torch.long, numel=self.MAX_BATCH_SIZE)
        self._prefill_ids: dict[tuple[int, ...], torch.Tensor] = {}

        # Lazy frame-local CodePredictor KV scratch, overwritten every step.
        self._cp_kv_cache: torch.Tensor | None = None
        self._depth_addressing: DepthAddressing | None = None

    def _get_suppress_mask(self) -> torch.Tensor:
        """Cache the checkpoint's static invalid-token mask on the worker GPU."""
        if self._suppress_mask is None:
            vocab_size = self.talker_config.vocab_size
            mask = torch.zeros(
                vocab_size, dtype=torch.bool, device=self.get_device()
            )
            mask[max(0, vocab_size - 1024):] = True
            eos = self.talker_config.codec_eos_token_id
            if 0 <= eos < vocab_size:
                mask[eos] = False
            self._suppress_mask = mask
        return self._suppress_mask

    def _get_batch_suppress_eos(
        self, inputs: list[ARNodeInputs]
    ) -> torch.Tensor:
        """Pack the input-carried minimum-length EOS flags for this batch.

        CUDA Graph replay calls ``preprocess`` with capture-slot dummy request
        ids, so looking up request state here would permanently observe a new
        request at frame zero. ``prepare_inputs`` runs with the real request and
        carries the dynamic flag as a host bool, staged here in one copy.
        """
        flags = torch.empty(len(inputs), dtype=torch.bool, device=self.get_device())
        self._flag_stager.copy_(flags, [item.kwargs["suppress_eos"] for item in inputs])
        return flags

    def _get_decode_last_token_indices(self, batch_size: int) -> torch.Tensor:
        """Return cached packed-sequence indices for one-token decode rows."""
        if self._decode_last_token_indices is None:
            self._decode_last_token_indices = torch.arange(
                self.MAX_BATCH_SIZE,
                dtype=torch.long,
                device=self.get_device(),
            )
        return self._decode_last_token_indices[:batch_size]

    def _mask_invalid_logits(
        self, logits: torch.Tensor, suppress_eos: torch.Tensor
    ) -> torch.Tensor:
        """Apply the static codec mask and the per-request EOS decision."""
        logits = logits.masked_fill(
            self._get_suppress_mask().unsqueeze(0), float("-inf")
        )
        eos = self.talker_config.codec_eos_token_id
        if 0 <= eos < logits.shape[1]:
            logits[:, eos].masked_fill_(suppress_eos, float("-inf"))
        return logits

    def _get_cp_kv_cache(self, batch_size: int) -> torch.Tensor:
        """Frame-local CodePredictor scratch cache, sliced to this micro-batch.

        TODO: make this a resource. The fixed maximum-batch tensor is
        overwritten every Talker step and gives the captured decode graph
        stable addresses.
        """
        if self._cp_kv_cache is None:
            cp = self.cp_config
            self._cp_kv_cache = torch.zeros(
                (cp.num_hidden_layers, self.MAX_BATCH_SIZE, 2,
                 self.num_codes, cp.num_key_value_heads, cp.head_dim),
                dtype=self.model.model.codec_embedding.weight.dtype,
                device=self.get_device(),
            )
        return self._cp_kv_cache[:, :batch_size]

    def _get_depth_addressing(self, batch_size: int) -> DepthAddressing:
        """The depth loop's addressing, cut to this micro-batch from tables built
        once on the first (uncaptured) forward."""
        full = self._depth_addressing
        if full is None:
            device = self.get_device()
            rows = torch.arange(self.MAX_BATCH_SIZE, dtype=torch.long, device=device)
            # int32 is what the eager RoPE reads, so it casts nothing per layer
            first = torch.arange(FIRST_PASS_TOKENS, dtype=torch.int32, device=device)
            full = self._depth_addressing = DepthAddressing(
                first_positions=first.repeat(self.MAX_BATCH_SIZE),
                first_rows=rows.repeat_interleave(FIRST_PASS_TOKENS),
                positions=torch.arange(self.num_codes, dtype=torch.int32, device=device)
                .unsqueeze(1).repeat(1, self.MAX_BATCH_SIZE),
                rows=rows,
            )
        tokens = batch_size * FIRST_PASS_TOKENS
        return DepthAddressing(
            first_positions=full.first_positions[:tokens],
            first_rows=full.first_rows[:tokens],
            positions=full.positions[:, :batch_size],
            rows=full.rows[:batch_size],
        )

    def _project_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map tokenizer embeddings into the Talker hidden width."""
        text_hidden = self.model.model.text_embedding(token_ids)
        projection_dtype = self.model.text_projection.linear_fc1.weight.dtype
        return self.model.text_projection(text_hidden.to(projection_dtype))

    def _device_ids(self, ids: tuple[int, ...]) -> torch.Tensor:
        """``[1, len(ids)]`` token ids on the worker GPU, built once per distinct tuple.

        The prefill's fixed ids (TTS BOS/EOS/PAD, the codec tags of one
        language and speaker) take few distinct values, so no request pays a
        host-to-device copy for them in ``prepare_inputs``.
        """
        tensor = self._prefill_ids.get(ids)
        if tensor is None:
            tensor = torch.tensor([ids], dtype=torch.long, device=self.get_device())
            self._prefill_ids[ids] = tensor
        return tensor

    def _special_text_embeds(
        self, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project TTS BOS/EOS/PAD once for official sequence construction."""
        token_ids = self._device_ids((
            self.config.tts_bos_token_id,
            self.config.tts_eos_token_id,
            self.config.tts_pad_token_id,
        ))
        bos, eos, pad = self._project_text(token_ids).to(dtype).chunk(3, dim=1)
        return bos, eos, pad

    def _frame_embeds(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum of the 16 codec embeddings per frame, in the Talker width: ``[frames, D]``."""
        embeds = self.model.model.codec_embedding(codes[:, 0])
        tables = self.code_predictor.model.codec_embedding
        for group in range(1, codes.shape[1]):
            embeds = embeds + tables[group - 1](codes[:, group])
        return embeds

    def _build_prefill(
        self,
        request_id: str,
        text_ids: torch.Tensor,
        prompt_layout: Sequence[int],
        speaker_id: int,
        language_id: int,
        speaker_embed: torch.Tensor | None = None,
        ref_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the official mixed text/codec prefill embedding sequence.

        ``text_ids`` concatenates an optional instruction turn, the assistant
        turn and (voice clone) the reference transcript turn; ``prompt_layout``
        is ``[instruct_len, text_len, stream_text, ref_text_len, ref_frames]``
        and says where the splits are and how the text is fed. The layout
        mirrors ``Qwen3TTSForConditionalGeneration.generate``:

        * instruction (``<|im_start|>user ... <|im_end|>``) as plain projected
          text embeddings (VoiceDesign, 1.7B CustomVoice style control);
        * the assistant role, then the codec think/language tags, the speaker
          (a built-in tag when ``speaker_id >= 0``, the reference x-vector
          when ``speaker_embed`` is given, nothing for VoiceDesign) and the
          codec PAD, all summed with TTS PAD / BOS text embeddings;
        * ``ref_frames > 0`` (in-context clone): reference transcript + text
          + TTS EOS over codec BOS + the reference frames' codec embeddings,
          laid out per ``generate_icl_prompt``;
        * otherwise ``stream_text == 0`` (the reference default for
          CustomVoice and VoiceDesign): every text token plus TTS EOS enters
          the prefill over codec PADs, closed by TTS PAD + codec BOS; decode
          then adds TTS PAD to each frame;
        * ``stream_text == 1`` (the reference default for Base): only the first
          text token enters the prefill over codec BOS; the remaining tokens
          plus TTS EOS are kept in per-request state and added one per frame.
        """
        text_ids = text_ids.to(device=self.get_device(), dtype=torch.long).view(1, -1)
        layout = [int(v) for v in prompt_layout]
        instruct_len, text_len, stream_text = layout[:3]
        ref_text_len, ref_frames = (layout[3], layout[4]) if len(layout) >= 5 else (0, 0)
        instruct_ids = text_ids[:, :instruct_len]
        assistant_end = text_ids.shape[1] - ref_text_len
        assistant_ids = text_ids[:, instruct_len:assistant_end]
        ref_text_ids = text_ids[:, assistant_end:]
        # ``process_prompt`` checked the wrapper's token ids on the host.
        prefix_len = len(self.CHATML_ASSISTANT_PREFIX_TOKEN_IDS)
        suffix_len = len(self.CHATML_ASSISTANT_SUFFIX_TOKEN_IDS)
        if text_len < 1 or assistant_ids.shape[1] != prefix_len + text_len + suffix_len:
            raise ValueError(
                "Qwen3-TTS prompt layout disagrees with the token stream: "
                f"expected {prefix_len + text_len + suffix_len} assistant tokens, "
                f"got {assistant_ids.shape[1]}"
            )
        if ref_frames > 0 and (ref_codes is None or ref_codes.shape[0] < ref_frames):
            raise ValueError(
                f"Qwen3-TTS in-context clone needs {ref_frames} reference frames, got "
                f"{0 if ref_codes is None else ref_codes.shape[0]}"
            )
        text_tokens = assistant_ids[:, prefix_len:prefix_len + text_len]

        codec = self.talker_config
        codec_prefix = (
            [codec.codec_nothink_id, codec.codec_think_bos_id, codec.codec_think_eos_id]
            if language_id < 0
            else [
                codec.codec_think_id,
                codec.codec_think_bos_id,
                language_id,
                codec.codec_think_eos_id,
            ]
        )
        # The speaker slot sits between the language tags and the codec PAD:
        # a built-in speaker's codec embedding, or the reference x-vector.
        speaker_tag = [speaker_id] if speaker_id >= 0 else []
        codec_ids = self._device_ids((*codec_prefix, *speaker_tag, codec.codec_pad_id, codec.codec_bos_id))
        codec_embeds = self.model.model.codec_embedding(codec_ids)
        dtype = codec_embeds.dtype
        if not speaker_tag and speaker_embed is not None:
            speaker_vector = speaker_embed.to(device=self.get_device(), dtype=dtype).reshape(1, 1, -1)
            split = len(codec_prefix)
            codec_embeds = torch.cat(
                [codec_embeds[:, :split], speaker_vector, codec_embeds[:, split:]], dim=1
            )
        bos_embed, eos_embed, pad_embed = self._special_text_embeds(dtype)

        def project(ids: torch.Tensor) -> torch.Tensor:
            return self._project_text(ids).to(dtype)

        # Tags: TTS PAD over every codec tag but the last, TTS BOS over the
        # codec PAD; the codec BOS pairs with text (or the reference) below.
        role_embed = project(assistant_ids[:, :prefix_len])
        tag_text = torch.cat([
            pad_embed.expand(-1, codec_embeds.shape[1] - 2, -1),
            bos_embed,
        ], dim=1)
        pieces = [role_embed, tag_text + codec_embeds[:, :-1]]
        if instruct_len:
            pieces.insert(0, project(instruct_ids))
        empty_trailing = eos_embed[:, :0]

        if ref_frames > 0:
            # ``<|im_start|>assistant\n{ref}<|im_end|>\n`` -> transcript tokens only.
            ref_tokens = ref_text_ids[:, prefix_len:ref_text_ids.shape[1] - 2]
            text_embed = torch.cat([project(torch.cat([ref_tokens, text_tokens], dim=1)), eos_embed], dim=1)
            reference = ref_codes.to(device=self.get_device(), dtype=torch.long)[:ref_frames]
            codec_embed = torch.cat(
                [codec_embeds[:, -1:], self._frame_embeds(reference).to(dtype).unsqueeze(0)], dim=1
            )
            if not stream_text:
                codec_pads = self.model.model.codec_embedding(
                    codec_ids.new_full((1, text_embed.shape[1]), codec.codec_pad_id)
                )
                pieces += [text_embed + codec_pads, codec_embed + pad_embed]
                trailing = empty_trailing
            elif text_embed.shape[1] > codec_embed.shape[1]:
                pieces.append(text_embed[:, :codec_embed.shape[1]] + codec_embed)
                trailing = text_embed[:, codec_embed.shape[1]:]
            else:
                padded = torch.cat([
                    text_embed,
                    pad_embed.expand(-1, codec_embed.shape[1] - text_embed.shape[1], -1),
                ], dim=1)
                pieces.append(padded + codec_embed)
                trailing = empty_trailing
        elif stream_text:
            pieces.append(project(text_tokens[:, :1]) + codec_embeds[:, -1:])
            trailing = torch.cat([project(text_tokens[:, 1:]), eos_embed], dim=1)
        else:
            text_embed = torch.cat([project(text_tokens), eos_embed], dim=1)
            codec_pads = self.model.model.codec_embedding(
                codec_ids.new_full((1, text_embed.shape[1]), codec.codec_pad_id)
            )
            pieces.append(text_embed + codec_pads)
            pieces.append(pad_embed + codec_embeds[:, -1:])
            # Nothing left to feed: every frame adds TTS PAD (empty stream).
            trailing = empty_trailing
        prefill = torch.cat(pieces, dim=1)

        state = self.request_state(request_id)
        state.add_all(
            trailing_text_hidden=trailing.squeeze(0),
            tts_pad_embed=pad_embed[0, 0],
            generation_step=0,
            generated_frames=0,
        )
        if ref_frames > 0:
            # The codec warms up on the tail of the reference clip ahead of
            # the generated frames (their audio is trimmed); see
            # ``_codec_stream_items``.
            state.add("reference_frames", reference)
        return prefill.squeeze(0)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> ARNodeInputs:
        """Convert routed request tensors into one Talker sequence fragment.

        Prefill creates the mixed conditioning sequence. Decode combines the
        previous frame's summed codec embedding with the next projected text
        condition. This method performs no transformer compute; the engine can
        therefore prepare requests before admitting them to a micro-batch.
        """
        del kwargs
        if graph_walk in ("talker_prefill", "talker_prefill_clone"):
            step_metadata = fwd_info.step_metadata
            input_embeds = self._build_prefill(
                fwd_info.rid_handle,
                inputs["text_inputs"][0],
                step_metadata[PROMPT_LAYOUT_KEY],
                int(step_metadata[SPEAKER_ID_KEY]),
                int(step_metadata[LANGUAGE_ID_KEY]),
                speaker_embed=inputs["speaker_embed"][0] if "speaker_embed" in inputs else None,
                ref_codes=inputs["ref_codes"][0] if "ref_codes" in inputs else None,
            )
            speaker = int(step_metadata[SPEAKER_ID_KEY]) >= 0 or "speaker_embed" in inputs
            assert input_embeds.shape[0] == prefill_rows(
                step_metadata[PROMPT_LAYOUT_KEY], int(step_metadata[LANGUAGE_ID_KEY]), speaker,
            ), f"the prefill built {input_embeds.shape[0]} rows, not the count `prefill_rows` lays out"
            state = self.request_state(fwd_info.rid_handle)
        elif graph_walk == "talker_decode":
            state = self.request_state(fwd_info.rid_handle)
            step = int(state["generation_step"])
            trailing = state["trailing_text_hidden"]
            text_condition = (
                trailing[step]
                if step < trailing.shape[0]
                else state["tts_pad_embed"]
            )
            # ``talker_input_embeds`` is the recurrent graph edge emitted by
            # the previous frame, not a token ID that needs another lookup.
            input_embeds = inputs["talker_input_embeds"][0].to(
                device=self.get_device(),
                dtype=text_condition.dtype,
            ).reshape(1, -1)
            input_embeds = input_embeds + text_condition
            state.add("generation_step", step + 1)
        else:
            raise ValueError(f"Unknown Qwen3-TTS Talker walk: {graph_walk!r}")

        suppress_eos = (
            int(state.get("generated_frames", 0))
            < self.config.generation.min_new_tokens
        )
        return ARNodeInputs(
            input_embeds=input_embeds,
            input_seq_len=input_embeds.shape[0],
            kwargs={"suppress_eos": suppress_eos},
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        """Pack a continuous batch and plan request-specific paged attention.

        Prefill sequences may have different lengths, so embeddings are
        concatenated into one packed tensor. ``last_token_indices`` maps each
        request back to the hidden state used for codec-group-0 prediction.
        FlashInfer receives separate sequence lengths and KV page tables.
        """

        del engine_inputs
        seq_lens = [item.input_seq_len for item in inputs]
        input_embeds = [
            item.input_embeds for item in inputs if item.input_embeds is not None
        ]
        if graph_walk == "talker_decode":
            if any(seq_len != 1 for seq_len in seq_lens):
                raise ValueError("Qwen3-TTS decode expects one token per request")
            # one token per request, so the indices are a fixed arange and a
            # single-request batch is already packed
            packed_embeds = (
                input_embeds[0]
                if len(input_embeds) == 1
                else torch.cat(input_embeds, dim=0)
            )
            last_token_indices = self._get_decode_last_token_indices(len(inputs))
        else:
            packed_embeds = torch.cat(input_embeds, dim=0)
            last_token_indices = torch.empty(len(inputs), dtype=torch.long, device=self.get_device())
            self._index_stager.copy_(last_token_indices, [end - 1 for end in itertools.accumulate(seq_lens)])

        # Materialize the persistent base mask before CUDA Graph capture. Only
        # the per-request EOS flags need to cross the replay input boundary.
        self._get_suppress_mask()
        return {
            "input_embeds": packed_embeds,
            "last_token_indices": last_token_indices,
            "suppress_eos": self._get_batch_suppress_eos(inputs),
        }

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        """One causal step over the packed batch; both walks sample group 0.

        CODE_PRED_SAMPLER belongs to whoever runs the depth loop. The region
        declares it whenever it holds a slot for this step, so declaring it
        here too would plan and commit it twice; when the region has no slot
        the loop runs inline and this step owns it. Keyed off the lease rather
        than the walk, which disagreed with ``_run_frame`` — that picks inline
        vs piecewise from what the runner can actually serve.
        """
        del slot_lease
        steps = {
            TALKER_KV: KVStep(),
            TALKER_ATTN: AttentionStep(causal=True),
            TALKER_POS: PositionStep(),
            TALKER_SAMPLER: SamplerStep(apply_penalty=True),
        }
        if not (piecewise_leases or {}).get(DEPTH_LOOP_REGION):
            steps[CODE_PRED_SAMPLER] = SamplerStep(apply_penalty=False)
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
        )

    def _run_frame(
        self,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Produce one complete codec frame for every request in the batch.

        The main Talker advances the engine-managed KV cache and predicts group
        0; the CodePredictor's unrolled depth loop fills groups 1..N-1 through
        the ``code_predictor`` aux sampler. Both read their params from
        engine-owned static buffers, so this single path serves eager execution
        and CUDA-graph capture alike (mirrors the Qwen3-Omni Talker).
        """
        hidden = self.model(input_embeds, label="main")
        last_hidden = hidden.index_select(0, last_token_indices)
        logits = self.model.codec_head(last_hidden)
        logits = self._mask_invalid_logits(logits, suppress_eos)

        talker_sampler: SamplerResource = engine_inputs.resources[TALKER_SAMPLER]
        code_pred_sampler: SamplerResource = engine_inputs.resources[CODE_PRED_SAMPLER]
        request_ids = engine_inputs.request_ids

        # The repetition penalty applies to group 0 only (its SamplerStep sets
        # apply_penalty); the depth loop's sampler declares no penalty.
        layer0_codes = talker_sampler.sample(request_ids, logits)

        piecewise = self._run_depth_loop_piecewise(
            engine_inputs, last_hidden, layer0_codes
        )
        if piecewise is not None:
            all_codes, codec_embed_sum = piecewise
        else:
            all_codes, codec_embed_sum = self._depth_loop(
                last_hidden, layer0_codes,
                lambda cp_logits: code_pred_sampler.sample(request_ids, cp_logits),
            )

        return {
            "talker_input_embeds": codec_embed_sum,
            "codec_tokens": all_codes,
            "new_token": layer0_codes,
        }

    def _depth_loop(
        self,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
        sample_codes: "Callable[[torch.Tensor], torch.Tensor]",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Walk residual groups 1..N-1, returning (all_codes, codec_embed_sum).

        ``sample_codes`` maps one group's logits to codes. Every op here is
        graph-safe, so this body is captured verbatim — inside the whole-walk
        decode graph, or on its own by the piecewise runner.
        """
        batch_size = layer0_codes.shape[0]
        device = layer0_codes.device
        all_codes = torch.empty(
            batch_size, self.num_codes, dtype=torch.long, device=device
        )
        all_codes[:, 0] = layer0_codes
        codec_embed = self.model.model.codec_embedding(layer0_codes)
        codec_embed_sum = codec_embed.clone()

        cp_cache = self._get_cp_kv_cache(batch_size)
        addressing = self._get_depth_addressing(batch_size)
        # The first pass runs positions 0 and 1 as two tokens of one pass: the
        # Talker hidden state conditions the decoder, and the group-0 embedding
        # predicts group 1. Each later position consumes the previous group's
        # embedding.
        cp_hidden = self.code_predictor.forward_depth_unrolled(
            torch.stack((last_hidden, codec_embed), dim=1),
            addressing.first_positions, addressing.first_rows, cp_cache, cache_pos=0,
        )
        for group_idx in range(1, self.num_codes):
            if group_idx > 1:
                cp_hidden = self.code_predictor.forward_depth_unrolled(
                    codec_embed.unsqueeze(1),
                    addressing.positions[group_idx], addressing.rows, cp_cache, cache_pos=group_idx,
                )
            cp_logits = torch.matmul(
                cp_hidden[:, -1],
                self.code_predictor.lm_head_weight[group_idx - 1].t(),
            )
            codes = sample_codes(cp_logits)
            all_codes[:, group_idx] = codes
            codec_embed = self.code_predictor.model.codec_embedding[
                group_idx - 1
            ](codes)
            codec_embed_sum.add_(codec_embed)
        return all_codes, codec_embed_sum

    def _code_predictor_piecewise_capture(
        self, call: PiecewiseCallInputs,
    ) -> dict[str, torch.Tensor]:
        """Capture entry point: the depth loop *including* its sampling.

        The CODE_PRED_SAMPLER resource reads its params from static buffers the
        region's ``declare_step`` plans, so nothing about sampling has to be
        hoisted out as a static input.
        """
        code_pred_sampler: SamplerResource = call.resources[CODE_PRED_SAMPLER]
        request_ids = call.engine_inputs.request_ids
        all_codes, codec_embed_sum = self._depth_loop(
            call.static_inputs["last_hidden"], call.static_inputs["layer0_codes"],
            lambda cp_logits: code_pred_sampler.sample(request_ids, cp_logits),
        )
        return {"all_codes": all_codes, "codec_embed_sum": codec_embed_sum}

    def _run_depth_loop_piecewise(
        self,
        engine_inputs: ModelInputsFromEngine,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Replay the captured depth loop, or None to run it inline.

        Used on paths the whole-walk graph doesn't cover — above all
        ``talker_prefill``, which is eager because it is variable-length. Left
        eager, this loop costs ~4x its captured time.
        """
        runner = engine_inputs.piecewise_runners.get(DEPTH_LOOP_REGION)
        batch_size = layer0_codes.shape[0]
        if runner is None or not runner.can_run(batch_size=batch_size):
            return None

        output = runner.run(
            static_inputs={
                "last_hidden": last_hidden,
                "layer0_codes": layer0_codes,
            },
            request_ids=engine_inputs.request_ids,
            real_bs=batch_size,
        )
        return output["all_codes"], output["codec_embed_sum"]

    def _codec_stream_items(self, graph_walk: str, request_id: str, frame: torch.Tensor) -> list[torch.Tensor]:
        """Frames this step pushes into the codec stream, one item per frame.

        The clone prefill leads with the tail of the reference clip's frames
        so the codec warms up on the voice being cloned: only the last
        ``left_context_frames`` matter, since that is all the context a
        window ever carries, and every reference frame the stream carries
        costs a codec pass whose audio ``CodecSubmodule`` then trims. Decode
        steps (the captured path) always push exactly one frame.
        """
        if graph_walk != "talker_prefill_clone":
            return [frame]
        reference = self.request_state(request_id).get("reference_frames")
        if reference is None:
            return [frame]
        return [*reference[-self.config.codec.left_context_frames:].unbind(0), frame]

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del kwargs
        output = self._run_frame(
            engine_inputs, input_embeds, last_token_indices, suppress_eos
        )
        request_id = engine_inputs.request_ids[0]
        return {
            "talker_input_embeds": [output["talker_input_embeds"]],
            "codec_tokens": self._codec_stream_items(graph_walk, request_id, output["codec_tokens"][0]),
            "new_token": [output["new_token"]],
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del kwargs
        output = self._run_frame(
            engine_inputs, input_embeds, last_token_indices, suppress_eos
        )
        return {
            request_id: {
                "talker_input_embeds": [output["talker_input_embeds"][i:i + 1]],
                "codec_tokens": self._codec_stream_items(graph_walk, request_id, output["codec_tokens"][i]),
                "new_token": [output["new_token"][i]],
            }
            for i, request_id in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        """Expose the stop token and advance metadata without a GPU sync.

        Whole-graph replay and eager execution normally retain ``new_token``.
        ``codec_tokens`` is the routed output, though, so use its first codebook
        as a fallback.  This keeps EOS detection correct even if an execution
        path filters the sampler-only output before slow-path ``check_stop``.
        """
        del request_info, kwargs
        if "new_token" in outputs:
            outputs["layer0_codes"] = outputs.pop("new_token")
        elif "layer0_codes" not in outputs and "codec_tokens" in outputs:
            codec_tokens = outputs["codec_tokens"][-1]
            outputs["layer0_codes"] = [codec_tokens.reshape(-1)[0]]
        if "layer0_codes" in outputs:
            state = self.request_state(request_id)
            state.add(
                "generated_frames", int(state.get("generated_frames", 0)) + 1
            )

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str] | StopCheck:
        """Stop the graph loop on codec EOS or the request frame limit.

        This callback runs off the GPU execution thread, so reading the sampled
        token with ``item`` is allowed here rather than in ``postprocess``.
        An EOS frame stays out of the codec stream, ``ignore_eos`` or not: EOS
        is loop control and not a codebook index, and only this host copy
        tells it from audio.
        """
        if "layer0_codes" not in outputs:
            return set()
        token = int(outputs["layer0_codes"][0].item())
        generated = int(
            self.request_state(request_id).get("generated_frames", 0)
        )
        max_tokens = request_info.max_tokens
        sampling_config = request_info.resource_configs.get(TALKER_SAMPLER)
        ignore_eos = bool(getattr(sampling_config, "ignore_eos", False))
        is_eos = token == self.talker_config.codec_eos_token_id
        reached_eos = is_eos and not ignore_eos
        stops = set()
        if reached_eos or generated >= max_tokens:
            logger.debug("talker %s: stop after %d frames (eos=%s)", request_id, generated, reached_eos)
            stops = {"talker_decode_loop"}
        return StopCheck(stops, withheld=("codec_tokens",)) if is_eos else stops

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        """Admit compatible prefill/decode requests to continuous batching.

        Both the Talker and the CodePredictor sample per-request off their own
        sampler buffers, so requests with differing sampling settings batch
        together freely.
        """
        return (
            batch.graph_walk in {"talker_prefill", "talker_prefill_clone", "talker_decode"}
            and bool(model_inputs)
            and len(model_inputs) <= self.MAX_BATCH_SIZE
        )

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        """Capture fixed one-token Talker decode batches and packed prefills,
        sampling included.

        Dynamic EOS suppression is carried by ``ARNodeInputs`` and packed into
        a graph input, so replay never consults capture-slot dummy request state.
        Prefill replays a packed capture of the smallest token bucket that
        holds the batch; the clone prefill stays eager because it also pushes
        the reference clip's frames into the codec stream.
        """
        del tp_world_size
        dtype = self.model.model.codec_embedding.weight.dtype
        hidden = self.talker_config.hidden_size

        def prefill_input(num_tokens: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_embeds=torch.zeros(num_tokens, hidden, dtype=dtype, device=device),
                input_seq_len=num_tokens,
                kwargs={"suppress_eos": True},
            )

        prefill = PackedAcceleratorGraphConfig(
            capture_graph_walk="talker_prefill",
            capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
            make_node_input=prefill_input,
            capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            compile=False,
        )
        return [prefill, BatchedAcceleratorGraphConfig(
            capture_graph_walk="talker_decode",
            single_request_inputs=ARNodeInputs(
                input_embeds=torch.zeros(
                    1,
                    self.talker_config.hidden_size,
                    dtype=dtype,
                    device=device,
                ),
                input_seq_len=1,
                # Padding slots clone this template, so every replay input must
                # declare the same dynamic key as a real prepared request.
                kwargs={"suppress_eos": True},
            ),
            capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            compile=True,
        )]

    def get_piecewise_accelerator_graph_configs(
        self,
        device: torch.device,
        autocast_dtype: torch.dtype,
        tp_world_size: int = 1,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """Capture the CodePredictor depth loop, sampling included.

        The whole-walk decode graph already covers this loop; this runner serves
        the paths it can't — chiefly ``talker_prefill``, which stays eager
        because it is variable-length and runs once per request. It uses no
        paged KV cache: its frame-local cache is the engine's fixed-shape
        scratch pool.
        """
        del tp_world_size
        hidden_size = self.talker_config.hidden_size
        capture_dtype = autocast_dtype or self.model.model.codec_embedding.weight.dtype

        def make_static_inputs(
            shape: PiecewiseCaptureShape,
        ) -> dict[str, torch.Tensor]:
            return {
                "last_hidden": torch.zeros(
                    shape.bs, hidden_size, dtype=capture_dtype, device=device,
                ),
                "layer0_codes": torch.zeros(
                    shape.bs, dtype=torch.long, device=device
                ),
            }

        def declare_step(
            request_ids: list[str], seq_lens: list[int],
        ) -> SubmoduleStep:
            # Only the CodePredictor sampler runs in this region (no paged KV);
            # its step sets up the cg-sampler buffers the capture reads from.
            del seq_lens
            return SubmoduleStep(
                segments=[
                    Segment(request_id=rid, label="main", span=1)
                    for rid in request_ids
                ],
                steps={CODE_PRED_SAMPLER: SamplerStep(apply_penalty=False)},
            )

        return {
            DEPTH_LOOP_REGION: PiecewiseBatchedConfig(
                capture_fn=self._code_predictor_piecewise_capture,
                make_static_inputs=make_static_inputs,
                declare_step=declare_step,
                # take the slot before the outer declaration, so it can leave
                # CODE_PRED_SAMPLER to this region
                lease_before_step=True,
                seq_len=1,
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                compile=False,
            )
        }

    def can_use_accelerator_graphs(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs]
    ) -> bool:
        """Replay the whole decode graph or a packed prefill; sampling params
        are read from buffers, so no request's settings can disqualify it."""
        if batch.graph_walk not in ("talker_decode", "talker_prefill") or not self.can_batch(
            batch, model_inputs
        ):
            return False
        return super().can_use_accelerator_graphs(batch, model_inputs)


# ===========================================================================
# 2. CodecSubmodule - fixed-shape streaming waveform decode
# ===========================================================================


class CodecSubmodule(ARNodeSubmodule):
    """Convert streamed 16-code frames into overlap-trimmed PCM chunks.

    The node runs on a stateless engine, but ``ARNodeInputs`` is reused as the
    typed container for fixed-length codec tensors. Each pass carries its
    window's geometry (real frames, repeated context); the neural decoder
    itself has no cross-call state.

    Windows follow the model's ``ScheduledLeftContextChunkPolicy``: a ramp of
    small chunks, then ``chunk_frames`` new frames behind ``left_context_frames``
    of context (a voice clone's reference tail is the first window's context).
    Each distinct window size is a CUDA-graph bucket; a shorter window is
    zero-padded up to the next bucket and trimmed after.

    The decoder's transformer needs the whole window, but the conv stack after
    it (nearly all of the cost) is causal with a short receptive field, so it
    decodes only the new frames behind its receptive field (``conv_context``)
    of the window.
    """

    # fp32, uncompiled: what ``get_stateless_flavor`` used to buy on the old
    # stateless engine, stated directly now that the v1 engine reads these.
    disable_torch_compile = True
    disable_autocast = True

    # The conv stack decodes at most chunk + conv_context frames (35 for the
    # 12 Hz decoder, 2.8 s of audio) whatever the window, so its activations stay small
    # enough to capture batches of 32 next to the Talker; ``can_batch`` keeps
    # the ceiling.
    MAX_BATCH_SIZE = 32
    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(
        self,
        decoder: torch.nn.Module,
        config: Qwen3TTSModelConfig,
        watermarker: AudioSealWatermarker | None = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.config = config
        # Opt-in (``audio_watermark`` in model_kwargs): marks each request's
        # trimmed PCM in ``batch_postprocess``; None leaves it untouched.
        self.watermarker = watermarker
        # Base streams may lead with a full reference tail: capture those windows too.
        leads = (0, config.codec.left_context_frames) if config.supports_reference_audio else (0,)
        self.windows = config.codec.codec_windows(leads)
        self.max_window = self.windows[-1]
        self.total_upsample = 1
        for factor in (
            *config.codec.upsample_rates,
            *config.codec.upsampling_ratios,
        ):
            self.total_upsample *= factor
        fuse_snake_activations(decoder)
        self.conv_context = self._split_conv_context(decoder, config) if config.codec.split_decode else None
        self.split_decode = self.conv_context is not None
        self._conv_start_stager = PinnedStager(torch.long, numel=self.MAX_BATCH_SIZE)

    @staticmethod
    def _split_conv_context(decoder: torch.nn.Module, config: Qwen3TTSModelConfig) -> int | None:
        """Conv-stack context for split decoding, or None (with a warning) to decode whole windows."""
        try:
            receptive_field = causal_receptive_field_frames(decoder)
        except (AttributeError, TypeError) as exc:
            logger.warning(
                "Qwen3-TTS codec: cannot derive the conv stack's receptive field (%s); "
                "decoding whole windows, which is slower", exc,
            )
            return None
        configured = config.codec.conv_context_frames
        if configured is not None and configured < receptive_field:
            logger.warning(
                "Qwen3-TTS codec: conv_context_frames=%d is below the conv stack's %d-frame "
                "receptive field, so split decoding would change the audio; decoding whole "
                "windows, which is slower", configured, receptive_field,
            )
            return None
        return receptive_field if configured is None else configured

    def conv_frames(self, window: int) -> int:
        """Frames the conv stack decodes for a window: its new frames plus ``conv_context``."""
        codec = self.config.codec
        return min(window, self.conv_context + max(*codec.chunk_schedule, codec.chunk_frames))

    def _bucket(self, frames: int) -> int:
        """Smallest captured window that holds ``frames`` (the terminal flush is shorter)."""
        for window in self.windows:
            if frames <= window:
                return window
        raise ValueError(
            f"Codec chunk has {frames} frames, maximum is {self.max_window}"
        )

    @staticmethod
    def _codec_chunk(input_metadata: InputMetadata | None) -> StreamChunkInfo | None:
        """This window's geometry from the stream buffer (offset, context, items, final)."""
        return input_metadata.stream_chunks.get("codec_tokens") if input_metadata else None

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        input_metadata: InputMetadata = EMPTY_INPUT_METADATA,
        **kwargs: Any,
    ) -> ARNodeInputs:
        """Pad one stream window to its capture bucket.

        Input arrives as ``[frames, code_groups]``. The official decoder wants
        ``[quantizers, frames]``; every request is padded to the smallest
        captured window that fits so the ramp's windows and the terminal tail
        all replay CUDA graphs. Every item is a frame: the Talker withholds
        its EOS frame from the stream (``TalkerSubmodule.check_stop``).
        """
        del graph_walk, kwargs
        chunk = self._codec_chunk(input_metadata)
        state = self.request_state(fwd_info.rid_handle)
        received = inputs.get("codec_tokens")
        if received:
            codes = received[0].to(device=self.get_device(), dtype=torch.long)
        else:
            # a stream whose first frame was EOS ends with an empty chunk
            codes = torch.zeros(0, self.config.num_code_groups, dtype=torch.long, device=self.get_device())
        if codes.ndim == 1:
            codes = codes.view(-1, self.config.num_code_groups)
        if codes.ndim != 2:
            raise ValueError(
                f"Expected codec tokens with shape (frames, groups), got {codes.shape}"
            )
        codes = codes[:, :self.config.codec.num_quantizers]
        frames = codes.shape[0]
        context = chunk.context_items if chunk is not None else 0
        # First window frame the conv stack decodes: the new frames' conv context.
        conv_start = context - min(context, self.conv_context) if self.split_decode else 0
        bucket = self._bucket(max(frames, 1))
        if frames < bucket:
            codes = torch.nn.functional.pad(codes, (0, 0, 0, bucket - frames))
        logger.debug(
            "codec %s: window context=%d frames=%d bucket=%d final=%s",
            fwd_info.request_id, context, frames, bucket, chunk is not None and chunk.is_final,
        )
        state.add("codec_bucket", bucket)   # graph-key fallback when a caller has no stream chunk info
        # The window's geometry rides with this pass: under speculative
        # scheduling the next window of the same request is prepared before
        # this one is postprocessed, so request state would be overwritten.
        return ARNodeInputs(
            tensor_inputs={"codec_tokens": codes.t().contiguous()},
            kwargs={
                "frames": frames, "context": context, "conv_start": conv_start,
                "is_final": chunk is not None and chunk.is_final,
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        """Stack the batch's windows, padded to its widest bucket.

        Requests reach the codec at different points of the chunk ramp, so one
        batch mixes windows. The decoder is causal, so trailing padding never
        changes the frames in front of it; each request's ``postprocess`` keeps
        only its own frames.
        """
        del graph_walk, engine_inputs
        windows = [item.tensor_inputs["codec_tokens"] for item in inputs]
        width = max(window.shape[-1] for window in windows)
        codec_tokens = torch.stack([
            torch.nn.functional.pad(window, (0, width - window.shape[-1]))
            for window in windows
        ])
        conv_start = torch.empty(len(inputs), dtype=torch.long, device=codec_tokens.device)
        # capture-time inputs carry no geometry: 0 is a valid start
        self._conv_start_stager.copy_(conv_start, [item.kwargs.get("conv_start", 0) for item in inputs])
        return {"codec_tokens": codec_tokens, "conv_start": conv_start}

    def _decode(self, codec_tokens: torch.Tensor, conv_start: torch.Tensor) -> torch.Tensor:
        """Decode windows to PCM16: per request, the audio of window frames
        ``conv_start`` .. ``conv_start + conv_frames(width)`` (all of it when
        the decoder is not split)."""
        if not self.split_decode:
            wav = self.decoder(codec_tokens)
        else:
            decoder = self.decoder
            hidden = decoder.quantizer.decode(codec_tokens)
            hidden = decoder.pre_conv(hidden).transpose(1, 2)
            hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state.permute(0, 2, 1)
            width = hidden.shape[-1]
            length = self.conv_frames(width)
            # Always from each request's conv_start, even when length == width:
            # postprocess trims relative to it. Past a request's own frames the
            # slice reads padding (or the clamp's repeat); causal convs keep
            # that out of its audio.
            index = (conv_start.view(-1, 1) + torch.arange(length, device=hidden.device)).clamp_(max=width - 1)
            hidden = hidden.gather(2, index.unsqueeze(1).expand(-1, hidden.shape[1], -1))
            for module in conv_stack(decoder):
                hidden = module(hidden)
            wav = hidden
        return (wav.clamp(-1, 1) * 32767).to(torch.int16).squeeze(1)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        conv_start: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        """One request's window (``[1, quantizers, frames]``) -> its ``[samples]`` audio."""
        del graph_walk, engine_inputs, kwargs
        return {"audio_chunk": [self._decode(codec_tokens, conv_start)[0]]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        conv_start: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        wavs = self._decode(codec_tokens, conv_start)
        return {
            request_id: {"audio_chunk": [wavs[i]]}
            for i, request_id in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs: Any,
    ) -> None:
        """Drop padding and repeated-context audio (a clone's reference tail included) before emission.

        ``inputs`` is this pass's ``prepare_inputs`` result (the engine hands it
        back), which carries the window's frame and context counts and the
        window frame the decoded audio starts at.
        """
        del request_info, kwargs
        if "audio_chunk" not in outputs:
            return
        if inputs is None:
            raise ValueError("codec postprocess needs the pass's inputs for its window geometry")
        frames = int(inputs.kwargs["frames"])
        context = int(inputs.kwargs["context"])
        conv_start = int(inputs.kwargs.get("conv_start", 0))
        start = (context - conv_start) * self.total_upsample
        end = (frames - conv_start) * self.total_upsample
        audio = outputs["audio_chunk"][0].reshape(-1)   # one request's samples, whatever the batch shape
        if audio.numel() < end:
            raise ValueError(
                f"codec produced {audio.numel()} samples for a window of {frames} frames "
                f"({end} expected after frame {conv_start})"
            )
        outputs["audio_chunk"][0] = audio[start:end]
        logger.debug(
            "codec %s: emit frames %d..%d (%d samples)", request_id, context, frames, max(end - start, 0)
        )

    def batch_postprocess(
        self,
        outputs: BatchedModelOutput,
        request_infos: dict[str, CurrentForwardPassInfo],
        inputs: list[ARNodeInputs],
        piecewise_runners: Mapping[str, Any] | None = None,
    ) -> None:
        """Watermark the step's trimmed chunks, one AudioSeal call per chunk
        length, each a replay of its capture (``get_piecewise_accelerator_graph_configs``)
        where there is one. Each request's stream state lives in its request
        state, so it goes when the request does."""
        if self.watermarker is None:
            return
        rids, chunks, streams, finals = [], [], [], []
        for request_id, item in zip(request_infos, inputs, strict=True):
            out = outputs.get(request_id, {})
            if "audio_chunk" not in out:
                continue
            state = self.request_state(request_id)
            stream = state.get("watermark_stream")
            if stream is None:
                stream = WatermarkStream()
                state.add("watermark_stream", stream)
            rids.append(request_id)
            chunks.append(out["audio_chunk"][0])
            streams.append(stream)
            finals.append(bool(item.kwargs.get("is_final", False)))
        if not rids:
            return
        marked = self.watermarker.apply_pcm16(chunks, streams, finals, runners=piecewise_runners)
        for request_id, chunk in zip(rids, marked, strict=True):
            outputs.get(request_id)["audio_chunk"][0] = chunk

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """With the watermark on, its generator call per chunk length of the
        schedule (a shorter final chunk pads into the next) and batch bucket.
        Only lengths of whole watermark frames; other chunks run eagerly."""
        del device, autocast_dtype, tp_world_size
        if self.watermarker is None:
            return {}
        codec = self.config.codec
        lengths = [
            frames * self.total_upsample for frames in {*codec.chunk_schedule, codec.chunk_frames}
            if frames * self.total_upsample % self.watermarker.hop == 0
        ]
        if not lengths:
            return {}
        return self.watermarker.piecewise_configs(lengths, self.CAPTURE_BATCH_SIZES)

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        """Any mix of windows batches (``preprocess`` pads to the widest)."""
        del batch
        return 0 < len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def cg_key_info(
        self,
        graph_walk: str,
        per_request_info: Mapping[str, CurrentForwardPassInfo],
        per_request_input_metadata: Mapping[str, InputMetadata] | None = None,
        **kwargs: Any,
    ) -> Any:
        """The widest window bucket in this batch, which ``preprocess`` pads to.

        Derived from the batch's stream chunk info (``per_request_input_metadata``,
        available before ``prepare_inputs`` runs, so a pre-planned lease can
        find its capture); the state written by ``prepare_inputs`` is the
        fallback for callers without it.
        """
        del graph_walk, kwargs
        per_request_input_metadata = per_request_input_metadata or {}
        buckets = set()
        for request_id in per_request_info:
            chunk = self._codec_chunk(per_request_input_metadata.get(request_id))
            if chunk is not None:
                buckets.add(self._bucket(max(chunk.num_items, 1)))
            else:
                buckets.add(self.request_state(request_id).get("codec_bucket"))
        buckets.discard(None)
        return max(buckets) if buckets else None

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        """Capture every window of the chunk schedule for all batch buckets."""
        del tp_world_size
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="codec_chunk",
                replay_graph_walks=["codec_chunk"],
                single_request_inputs=ARNodeInputs(
                    # 1, not the window: batched buckets match on bs, and this
                    # keeps the intern seq_len from aliasing the trailing dims.
                    input_seq_len=1,
                    tensor_inputs={
                        "codec_tokens": torch.zeros(
                            self.config.codec.num_quantizers,
                            window,
                            dtype=torch.long,
                            device=device,
                        )
                    },
                ),
                additional_key_info=window,
                capture_batch_sizes=self.CAPTURE_BATCH_SIZES,
                compile=False,
            )
            for window in self.windows
        ]

    def can_use_accelerator_graphs(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs]
    ) -> bool:
        return (
            batch.graph_walk == "codec_chunk"
            and self.can_batch(batch, model_inputs)
            and all(
                item.tensor_inputs["codec_tokens"].shape
                in {(self.config.codec.num_quantizers, window) for window in self.windows}
                for item in model_inputs
            )
            and super().can_use_accelerator_graphs(batch, model_inputs)
        )


# ===========================================================================
# 3. RefEncoderSubmodule - reference audio -> speaker conditioning (Base)
# ===========================================================================


class RefEncoderSubmodule(NodeSubmodule):
    """Encode one reference clip into the Talker's voice conditioning.

    Runs once per request, before the clone prefill, and owns no resources.
    The x-vector comes from the ECAPA-TDNN encoder over a log-mel spectrogram
    of the 24 kHz clip; for in-context cloning the codec encoder also turns
    the clip into ``ref_frames`` 16-group frames. The node keeps the dtypes it
    was built with (``disable_autocast``): the mel front end and the codec
    encoder stay in float32, the speaker encoder runs in the Talker's dtype
    as the reference does.

    The same clip recurs: a voice is reused across a session, and every
    sentence chunk of a long clone request carries it. The conditioning is
    memoised by clip content, so the encoders run once per distinct clip and a
    repeat costs the prefill alone.
    """

    disable_torch_compile = True
    disable_autocast = True
    CONDITIONING_CACHE_SIZE = 64

    def __init__(
        self,
        speaker_encoder: Qwen3TTSSpeakerEncoder,
        mel_front_end: Qwen3TTSMelFrontEnd,
        codec_encoder: torch.nn.Module,
        config: Qwen3TTSModelConfig,
    ) -> None:
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.mel_front_end = mel_front_end
        self.codec_encoder = codec_encoder
        self.config = config
        # (clip digest, ref_frames) -> (speaker_embed, ref_codes), oldest first.
        self._conditioning: OrderedDict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        del graph_walk, kwargs
        waveform = inputs["audio_inputs"][0].to(device=self.get_device(), dtype=torch.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0) if waveform.shape[0] < waveform.shape[-1] else waveform.mean(dim=-1)
        # ``process_prompt`` settled the layout and hashed the clip on the host.
        layout = fwd_info.step_metadata[PROMPT_LAYOUT_KEY]
        ref_frames = int(layout[4]) if len(layout) >= 5 else 0
        return NodeInputs(
            tensor_inputs={"waveform": waveform.reshape(-1)},
            kwargs={"ref_frames": ref_frames, "clip_key": (fwd_info.step_metadata[REF_CLIP_KEY], ref_frames)},
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveform: torch.Tensor,
        ref_frames: int = 0,
        clip_key: tuple[str, int] | None = None,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        if clip_key is not None and clip_key in self._conditioning:
            self._conditioning.move_to_end(clip_key)
            speaker_embed, codes = self._conditioning[clip_key]
            return {"speaker_embed": [speaker_embed], "ref_codes": [codes]}
        device_type = waveform.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            mels = self.mel_front_end(waveform.unsqueeze(0))
        encoder_dtype = next(self.speaker_encoder.parameters()).dtype
        speaker_embed = self.speaker_encoder(mels.to(encoder_dtype))[0]

        num_quantizers = self.config.codec.num_quantizers
        if ref_frames > 0:
            with torch.autocast(device_type=device_type, enabled=False):
                encoded = self.codec_encoder.encode(
                    input_values=waveform.view(1, 1, -1).float(), return_dict=True
                )
            codes = encoded.audio_codes[0, :num_quantizers, :ref_frames].t().contiguous().long()
            if codes.shape[0] < ref_frames:
                raise ValueError(
                    f"codec encoder produced {codes.shape[0]} frames for a clip declared as {ref_frames}"
                )
        else:
            # x-vector-only clone: no in-context frames. The edge still needs
            # a tensor; the Talker ignores it because the layout says 0 frames.
            codes = torch.zeros(1, num_quantizers, dtype=torch.long, device=waveform.device)
        if clip_key is not None:
            self._conditioning[clip_key] = (speaker_embed, codes)
            while len(self._conditioning) > self.CONDITIONING_CACHE_SIZE:
                self._conditioning.popitem(last=False)
        return {"speaker_embed": [speaker_embed], "ref_codes": [codes]}
