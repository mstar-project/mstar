# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# Copyright (c) 2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
# Copyright (c) 2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the reference front end and
# inference loop split into M* graph-node submodules with continuous batching.
#
# Original files (FunAudioLLM/CosyVoice, cosyvoice/cli/frontend.py,
# cosyvoice/cli/model.py and cosyvoice/llm/llm.py) were released under
# Apache-2.0, with the full license text available at
# https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""Node submodules for CosyVoice3: the compute behind each graph node.

``SpeakerEncoderSubmodule``  reference clip -> speech tokens for the LM prompt
``LMSubmodule``              autoregressive speech tokens (paged KV, continuous
                             batching, repetition-aware sampling, captured decode)
``FlowSubmodule``            speech-token stream -> settled mel frames
``HiFTSubmodule``            mel frames -> waveform chunks

Reference behaviour these mirror: ``cosyvoice/cli/frontend.py``
(``frontend_zero_shot``), ``cosyvoice/cli/model.py`` (``CosyVoice3Model``),
``cosyvoice/llm/llm.py`` (``inference``).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.model.chatterbox.components.audio_frontend import kaldi_fbank_80, resample
from mstar.model.chatterbox.submodules import VoiceCache
from mstar.model.components.audio_features import MatchaLogMel
from mstar.model.cosyvoice3.components.flow import CosyVoice3FlowRow, ReferenceVoice
from mstar.model.cosyvoice3.components.hift import HiftStreamState, stream_vocode
from mstar.model.cosyvoice3.components.sampling import ras_sample
from mstar.model.cosyvoice3.config import (
    LM_ATTN,
    LM_KV,
    LM_LABEL,
    LM_POS,
    SAMPLE_RATE,
    TOKENIZER_SR,
    CosyVoice3Config,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    NodeInputs,
    NodeSubmodule,
)
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)

# Edge names (kept in sync with cosyvoice3_model.py)
TEXT_INPUTS = "text_inputs"  # prompt text + text ids
REF_AUDIO = "ref_audio"  # 24 kHz mono reference clip
PROMPT_TOKENS = "prompt_tokens"  # speech tokens of the clip (LM prompt)
SPEECH_TOKENS = "speech_tokens"  # LM output, streamed to the flow
PREV_TOKEN = "prev_token"
MEL = "mel"  # newly settled mel frames
MEL_FINAL = "mel_final"  # [1] bool: the stream has ended
AUDIO_CHUNK = "audio_chunk"
DECODE_LOOP = "decode_loop"
STEP_INDEX = "lm_step"  # [1] the LM step that produced the token, for check_stop

# Request facts process_prompt settles, carried in step_metadata
MIN_SPEECH_TOKENS = "min_speech_tokens"
MAX_SPEECH_TOKENS = "max_speech_tokens"
VOICE_KEY = "voice_key"  # content hash of the clip

PREFILL_WALKS = ("prefill", "prefill_voice")


def _rid(fwd_info: CurrentForwardPassInfo) -> int:
    if fwd_info.rid_handle == -1:
        raise ValueError(f"request {fwd_info.request_id!r} reached the worker without a rid handle")
    return fwd_info.rid_handle


def _voice_key(fwd_info: CurrentForwardPassInfo) -> int:
    return int((fwd_info.step_metadata or {})[VOICE_KEY])


def reference_tokens(samples: int, config: CosyVoice3Config) -> int:
    """How many prompt tokens `ReferenceEncoder.tokens` makes of a ``samples``-long
    24 kHz clip. `SpeakerEncoderSubmodule` asserts the same count."""
    from mstar.model.chatterbox.components.s3_tokenizer import AudioEncoderV2

    wav16 = -(-samples * TOKENIZER_SR // SAMPLE_RATE)
    frames = wav16 // config.tokenizer.hop_size
    tokens = AudioEncoderV2._conv_out_len(AudioEncoderV2._conv_out_len(frames, 2), 2)
    mel = config.mel
    mel_frames = (samples + 2 * ((mel.n_fft - mel.hop_size) // 2) - mel.n_fft) // mel.hop_size + 1
    return min(tokens, mel_frames // 2)


class ReferenceEncoder:
    """Reference clip -> prompt tokens, prompt mel and x-vector, as
    ``frontend_zero_shot`` computes them: tokens from the 16 kHz clip, mel
    from the 24 kHz one, both cut so the mel has exactly two frames a token.
    Shared by the speaker-encoder and flow nodes when they are colocated."""

    def __init__(self, tokenizer: nn.Module, mel: MatchaLogMel, campplus: nn.Module | None):
        self.tokenizer = tokenizer
        self.mel = mel
        self.campplus = campplus

    @torch.no_grad()
    def tokens(self, wav24: torch.Tensor) -> torch.Tensor:
        wav16 = resample(wav24.float(), SAMPLE_RATE, TOKENIZER_SR)
        tokens, lens = self.tokenizer([wav16])
        tokens = tokens[0, : int(lens[0])]
        frames = self.mel.num_frames(wav24.numel())
        return tokens[: min(int(tokens.numel()), frames // 2)]

    @torch.no_grad()
    def voice(self, wav24: torch.Tensor) -> ReferenceVoice:
        tokens = self.tokens(wav24)
        feat = self.mel(wav24.float()[None])[0].transpose(0, 1)  # [F, 80]
        feat = feat[: 2 * tokens.numel()]
        wav16 = resample(wav24.float(), SAMPLE_RATE, TOKENIZER_SR)
        fbank = kaldi_fbank_80(wav16)
        embedding = self.campplus(fbank[None])[0]
        return ReferenceVoice(prompt_tokens=tokens, prompt_feat=feat, embedding=embedding)


# ===========================================================================
# 1. Speaker encoder: reference clip -> LM prompt tokens
# ===========================================================================


class SpeakerEncoderSubmodule(NodeSubmodule):
    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, encoder: ReferenceEncoder, config: CosyVoice3Config):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = encoder.tokenizer  # registered, so get_device finds its parameters
        self.config = config
        self.cache = VoiceCache(config.voice_cache_size)

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        del graph_walk, kwargs
        wav = inputs[REF_AUDIO][0].to(self.get_device(), torch.float32).reshape(-1)
        return NodeInputs(tensor_inputs={REF_AUDIO: wav}, kwargs={VOICE_KEY: _voice_key(fwd_info)})

    def forward(self, graph_walk, engine_inputs, ref_audio: torch.Tensor, voice_key: int, **kwargs) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        tokens = self.cache.get(voice_key)
        if tokens is None:
            # a new clip only: the token count is data-dependent, so the
            # tokenizer reads its lengths back to the host
            tokens = self.encoder.tokens(ref_audio)
            assert tokens.numel() == reference_tokens(ref_audio.numel(), self.config), (
                f"the speech tokenizer made {tokens.numel()} prompt tokens of a {ref_audio.numel()}-sample clip, "
                "not the count `reference_tokens` gives the LM's kv_layout"
            )
            self.cache.put(voice_key, tokens)
        return {PROMPT_TOKENS: [tokens]}


# ===========================================================================
# 2. LM: text (+ prompt speech) -> speech tokens
# ===========================================================================

HISTORY = "history"
KNOBS = "knobs"
STEP = "step"
EOS_FLAG = "eos_flag"


@dataclass(frozen=True, slots=True)
class LMRequestKnobs:
    """A request's sampling settings, resolved once at prefill."""

    seed: int
    temperature: float
    top_p: float
    top_k: int
    window: int  # ras_window, at most the deployment's (the history width)
    threshold: float  # ras_window * ras_tau; 0 disables the redraw
    ignore_eos: bool
    min_len: int
    max_len: int


# The decode capture's template row
CAPTURE_KNOBS = LMRequestKnobs(
    seed=0, temperature=1.0, top_p=0.8, top_k=25, window=1, threshold=1.0, ignore_eos=False, min_len=0, max_len=1
)


class LMSubmodule(ARNodeSubmodule):
    """One LM step for a continuous batch: prefill packs
    ``[sos | prompt text + text | task_id | prompt speech]`` per request, decode
    embeds the previous speech token. Sampling is ``ras_sample`` on float32
    logits, keyed by the request seed and step, so a decode batch replays one
    captured graph per batch size.

    A row carries its sampling settings as host values; ``preprocess`` sends a
    batch's columns to the device in one pinned copy per dtype."""

    disable_torch_compile = True
    MAX_BATCH_SIZE = 32
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(self, model: nn.Module, config: CosyVoice3Config):
        super().__init__()
        self.model = model
        self.config = config
        self.lm = config.lm
        self.window = int(config.sampling.ras_window)
        self._int_stager = PinnedStager(torch.long, numel=4 * self.MAX_BATCH_SIZE)
        self._float_stager = PinnedStager(torch.float32, numel=3 * self.MAX_BATCH_SIZE)
        self._bool_stager = PinnedStager(torch.bool, numel=2 * self.MAX_BATCH_SIZE)
        self._eos_flags: dict[bool, torch.Tensor] = {}

    # -- request knobs -------------------------------------------------------

    def _knob(self, step_metadata: Mapping[str, Any] | None, key: str):
        meta = step_metadata or {}
        return meta[key] if key in meta else getattr(self.config.sampling, key)

    def _resolve_knobs(self, fwd_info: CurrentForwardPassInfo) -> LMRequestKnobs:
        meta = fwd_info.step_metadata or {}
        window = int(self._knob(meta, "ras_window"))
        return LMRequestKnobs(
            seed=int(fwd_info.random_seed) & (2**63 - 1),
            temperature=float(self._knob(meta, "temperature")),
            top_p=float(self._knob(meta, "top_p")),
            top_k=int(self._knob(meta, "top_k")),
            window=window,
            threshold=float(window) * float(self._knob(meta, "ras_tau")) if window > 0 else 0.0,
            ignore_eos=bool(meta.get("ignore_eos", False)),
            min_len=int(meta[MIN_SPEECH_TOKENS]),
            max_len=int(meta[MAX_SPEECH_TOKENS]),
        )

    def _eos_flag(self, value: bool) -> torch.Tensor:
        """A shared ``[1]`` bool device tensor holding ``value``."""
        flag = self._eos_flags.get(value)
        if flag is None:
            flag = self._eos_flags[value] = torch.full((1,), value, dtype=torch.bool, device=self.get_device())
        return flag

    @torch.no_grad()
    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> ARNodeInputs:
        del kwargs
        device = self.get_device()
        state = self.request_state(_rid(fwd_info))
        if graph_walk in PREFILL_WALKS:
            text_ids = inputs[TEXT_INPUTS][0].to(device, torch.long).reshape(-1)
            prompt = (
                inputs[PROMPT_TOKENS][0].to(device, torch.long).reshape(-1)
                if inputs.get(PROMPT_TOKENS)
                else torch.zeros(0, dtype=torch.long, device=device)
            )
            embeds = self.model.build_prefill_embeds(text_ids, prompt)
            state.add_all(
                generated=0,
                knobs=self._resolve_knobs(fwd_info),
                history=torch.full((max(self.window, 1),), -1, dtype=torch.long, device=device),
            )
            state.remove(EOS_FLAG)
            history = state[HISTORY]
            eos_flag = None
        elif graph_walk == "decode":
            token = inputs[PREV_TOKEN][0].to(device, torch.long).reshape(-1)[:1]
            embeds = self.model.embed_speech(token)
            # committed in postprocess, so a re-driven step does not add the token twice
            history = torch.cat([state[HISTORY][1:], token])
            eos_flag = self._ended_at_prefill(state, token)
        else:
            raise ValueError(f"Unknown LM walk {graph_walk!r}")
        return ARNodeInputs(
            input_embeds=embeds,
            input_seq_len=embeds.shape[0],
            tensor_inputs={HISTORY: history},
            kwargs={KNOBS: state[KNOBS], STEP: int(state["generated"]), EOS_FLAG: eos_flag},
        )

    def _ended_at_prefill(self, state, token: torch.Tensor) -> torch.Tensor | None:
        """Whether the prefill token already ended the request: a control id,
        or the whole token budget. The prefill walk has no decode loop for
        ``check_stop`` to end, so every decode step of such a request is made
        to sample EOS, which the flow drops and which ends the loop on the
        first one. Decided on the first decode step, whose input is the prefill
        token, and kept, so a re-prepared or speculated step agrees. A ``[1]``
        bool on the device (the token is never read back), or None for never."""
        if EOS_FLAG not in state:
            knobs: LMRequestKnobs = state[KNOBS]
            if int(state["generated"]) >= knobs.max_len:
                flag = self._eos_flag(True)
            elif knobs.ignore_eos:
                flag = None
            else:
                flag = token >= self.lm.speech_token_size
            state.add(EOS_FLAG, flag)
        return state[EOS_FLAG]

    def _stage_columns(self, inputs: list[ARNodeInputs]) -> dict[str, torch.Tensor]:
        """The batch's per-row sampling inputs, ``[B, 1]`` each, from three
        non-blocking pinned copies (int64, float32, bool)."""
        device = self.get_device()
        n = len(inputs)
        knobs: list[LMRequestKnobs] = [inp.kwargs[KNOBS] for inp in inputs]
        steps = [inp.kwargs[STEP] for inp in inputs]
        ints = np.array(
            [[k.seed for k in knobs], steps, [k.top_k for k in knobs], [k.window for k in knobs]], dtype=np.int64
        )
        floats = np.array(
            [[k.temperature for k in knobs], [k.top_p for k in knobs], [k.threshold for k in knobs]], dtype=np.float32
        )
        # the reference's minimum length masks only id ``speech_token_size``
        # (its ``sos``), so the other control ids can still end the decode
        flags = np.array(
            [[s < k.min_len or k.ignore_eos for k, s in zip(knobs, steps, strict=True)], [k.ignore_eos for k in knobs]],
            dtype=np.bool_,
        )
        columns = {}
        for names, host, stager in (
            (("seed", "step", "top_k", "window"), ints, self._int_stager),
            (("temperature", "top_p", "threshold"), floats, self._float_stager),
            (("block_sos", "block_stop"), flags, self._bool_stager),
        ):
            dev = torch.empty(host.shape, dtype=stager.dtype, device=device)
            stager.copy_(dev, host.reshape(-1))
            columns.update({name: dev[i].view(n, 1) for i, name in enumerate(names)})
        return columns

    def preprocess(self, graph_walk, engine_inputs, inputs: list[ARNodeInputs]) -> dict[str, Any]:
        del graph_walk, engine_inputs
        never = self._eos_flag(False)
        flags = [inp.kwargs[EOS_FLAG] for inp in inputs]
        packed: dict[str, Any] = {
            "input_embeds": torch.cat([inp.input_embeds for inp in inputs], dim=0),
            HISTORY: torch.stack([inp.tensor_inputs[HISTORY] for inp in inputs]),
            "force_eos": torch.stack([never if flag is None else flag for flag in flags]),
        }
        packed.update(self._stage_columns(inputs))
        return packed

    def declare_step(
        self,
        graph_walk,
        request_ids,
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases=None,
        **kwargs,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, piecewise_leases, kwargs
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=LM_LABEL, span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={LM_KV: KVStep(), LM_ATTN: AttentionStep(causal=True), LM_POS: PositionStep()},
        )

    def _forward(
        self,
        graph_walk,
        engine_inputs,
        input_embeds,
        history,
        seed,
        step,
        temperature,
        top_p,
        top_k,
        window,
        threshold,
        block_sos,
        block_stop,
        force_eos,
    ) -> torch.Tensor:
        hidden = self.model.hidden(input_embeds, label=LM_LABEL)
        if graph_walk in PREFILL_WALKS:
            hidden = engine_inputs.resources[LM_ATTN].select_last_hidden(hidden, label=LM_LABEL)
        logits = self.model.logits(hidden)
        ids = torch.arange(logits.shape[-1], device=logits.device)[None, :]
        blocked = (block_sos & (ids == self.lm.sos_token)) | (block_stop & (ids >= self.lm.speech_token_size))
        blocked = torch.where(force_eos, ids != self.lm.eos_token, blocked)
        # Not the engine sampler: RAS's draws are hashed from (seed, step, id) so
        # a row's tokens do not depend on its batch, and it redraws a token
        # repeated in the last W from the full distribution under a second noise
        # stream, with per-row blocked ids. SamplerResource draws from a Philox
        # stream per request and has no window or blocked ids.
        return ras_sample(
            logits,
            seed=seed.reshape(-1),
            step=step.reshape(-1),
            history=history,
            temperature=temperature.reshape(-1),
            top_p=top_p.reshape(-1),
            top_k=top_k.reshape(-1),
            window=window.reshape(-1),
            repeat_threshold=threshold.reshape(-1),
            blocked=blocked,
        )

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        tokens = self._forward(graph_walk, engine_inputs, **kwargs)
        return {SPEECH_TOKENS: [tokens.reshape(-1)[:1]], STEP_INDEX: [kwargs["step"].reshape(-1)[:1].clone()]}

    def forward_batched(self, graph_walk, engine_inputs, **kwargs) -> dict[str, NameToTensorList]:
        tokens = self._forward(graph_walk, engine_inputs, **kwargs).reshape(-1)
        # a copy: under graph replay ``step`` is the static input the next step overwrites
        steps = kwargs["step"].reshape(-1).clone()
        return {
            rid: {SPEECH_TOKENS: [tokens[i : i + 1]], STEP_INDEX: [steps[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def postprocess(self, request_id, request_info, outputs, inputs: NodeInputs | None = None, **kwargs) -> None:
        del request_info, kwargs
        if SPEECH_TOKENS not in outputs:
            return
        outputs[PREV_TOKEN] = outputs[SPEECH_TOKENS]
        state = self.request_state(request_id)
        state.add("generated", int(state.get("generated", 0)) + 1)
        if inputs is not None and HISTORY in inputs.tensor_inputs:
            state.add(HISTORY, inputs.tensor_inputs[HISTORY])

    def check_stop(self, request_id, request_info, outputs) -> set[str]:
        del request_info
        if SPEECH_TOKENS not in outputs:
            return set()
        token = int(outputs[SPEECH_TOKENS][0].reshape(-1)[0].item())
        # this step's own index, not ``generated``: a speculative next step may
        # already have been postprocessed
        generated = int(outputs[STEP_INDEX][0].reshape(-1)[0].item()) + 1
        knobs: LMRequestKnobs = self.request_state(request_id)[KNOBS]
        stopped = not knobs.ignore_eos and token >= self.lm.speech_token_size
        if stopped or generated >= knobs.max_len:
            return {DECODE_LOOP}
        return set()

    # -- batching and capture ---------------------------------------------------

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return batch.graph_walk in (*PREFILL_WALKS, "decode") and 0 < len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        del tp_world_size
        dtype = self.model.speech_embedding.weight.dtype
        width = max(self.window, 1)
        tensor_inputs = {HISTORY: torch.full((width,), -1, dtype=torch.long, device=device)}
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_embeds=torch.zeros(1, self.lm.hidden_size, dtype=dtype, device=device),
                    input_seq_len=1,
                    tensor_inputs=tensor_inputs,
                    kwargs={KNOBS: CAPTURE_KNOBS, STEP: 0, EOS_FLAG: None},
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                compile=False,
            )
        ]

    def can_use_accelerator_graphs(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return self.can_batch(batch, model_inputs) and super().can_use_accelerator_graphs(batch, model_inputs)


# ===========================================================================
# 3. Flow: speech-token stream -> settled mel frames
# ===========================================================================


def filter_tokens(raw: torch.Tensor, config: CosyVoice3Config) -> torch.Tensor:
    """The tokens the reference hands the flow: speech ids only (control ids
    end the stream), and no run of more than ``max_silent_tokens`` silent
    tokens (the LM still saw them). ``raw`` is a host tensor."""
    tokens = raw[raw < config.lm.speech_token_size]
    if tokens.numel() == 0:
        return tokens
    silent = torch.isin(tokens, torch.tensor(config.silent_tokens, device=tokens.device))
    keep = torch.ones_like(silent)
    run = 0
    for i, is_silent in enumerate(silent.tolist()):
        run = run + 1 if is_silent else 0
        if run > config.max_silent_tokens:
            keep[i] = False
    return tokens[keep]


@contextlib.contextmanager
def _without_tf32():
    """float32 matmuls and convolutions at full precision. The engine enables
    TF32 process-wide, and the guided flow solve amplifies its rounding far
    beyond float32's."""
    matmul, cudnn = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul)
        torch.backends.cudnn.allow_tf32 = cudnn


class FlowSubmodule(NodeSubmodule):
    """Settles the stream's tokens in whole attention chunks and emits their mel.

    Every chunk of the stream carries the whole token history (the edge's
    policy keeps all of it as left context), so the node holds only how far it
    has settled and where its hop schedule is. A request's reference voice is
    conditioned from the clip that rides with its first chunk (per-voice cache)
    and kept in its state for the chunks that follow.
    """

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, flow: nn.Module, encoder: ReferenceEncoder, config: CosyVoice3Config):
        super().__init__()
        self.flow = flow
        self.encoder = encoder
        self.config = config
        self.cache = VoiceCache(config.voice_cache_size)
        self.ratio = config.flow.token_mel_ratio
        self.lookahead = config.flow.pre_lookahead_len
        self.chunk_tokens = config.flow.chunk_tokens
        # one copy per row: a ring of two batches never waits on a copy still in flight
        self._token_stager = PinnedStager(
            torch.long, numel=config.max_speech_tokens, depth=2 * config.flow_max_batch_size
        )

    def _voice(self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, state) -> ReferenceVoice:
        if inputs.get(REF_AUDIO):
            key = _voice_key(fwd_info)
            voice = self.cache.get(key)
            if voice is None:
                wav = inputs[REF_AUDIO][0].to(self.get_device(), torch.float32).reshape(-1)
                voice = self.encoder.voice(wav)
                self.cache.put(key, voice)
            if state.get("voice") is None:
                state.add("voice", voice)
            return voice
        if state.get("voice") is None:
            raise ValueError("CosyVoice3 needs a reference voice; none reached the flow node")
        return state["voice"]

    def hop_size(self, k: int, prompt_tokens: int) -> int:
        """Tokens hop ``k`` settles: 25, 50, 100, 100, ... (reference
        ``CosyVoice2Model.tts``), the first one stretched so that the prompt plus
        the hop ends on a 25-token attention chunk."""
        cfg = self.config
        cap = max(cfg.stream_max_hop_tokens, cfg.stream_hop_tokens)
        hop = int(min(cfg.stream_hop_tokens * cfg.stream_hop_growth**k, cap))
        if k == 0:
            hop += -(-prompt_tokens // self.chunk_tokens) * self.chunk_tokens - prompt_tokens
        return hop

    def plan(self, state, voice: ReferenceVoice, tokens: torch.Tensor, final: bool) -> CosyVoice3FlowRow | None:
        """The row this request needs solved now, or None (nothing settles yet).

        A stream settles every hop whose end and three look-ahead tokens have
        arrived, in one solve; the final chunk settles the rest."""
        n = int(tokens.numel())
        offset = int(state.get("settled", 0))
        streaming = self.config.stream_hop_tokens > 0
        if final:
            if n == 0 or n <= offset:
                return None
            return CosyVoice3FlowRow(voice=voice, tokens=tokens, settle=n, final=True, streaming=streaming)
        if not streaming:
            return None
        target, hops = offset, int(state.get("hops", 0))
        while (nxt := target + self.hop_size(hops, voice.num_prompt_tokens)) + self.lookahead <= n:
            target, hops = nxt, hops + 1
        state.add("hops", hops)
        if target == offset:
            return None
        return CosyVoice3FlowRow(voice=voice, tokens=tokens, settle=target, final=False, streaming=True)

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        del graph_walk
        device = self.get_device()
        rid = _rid(fwd_info)
        state = self.request_state(rid)
        chunks = inputs.get(SPEECH_TOKENS) or []
        raw = chunks[0] if chunks and chunks[0] is not None else None
        if raw is None or raw.numel() == 0:
            raw = state.get("raw", torch.zeros(0, dtype=torch.long))
        else:
            # The one host sync of a flow step: the stream delivers the tokens
            # on the device, and how many settle depends on their values.
            raw = raw.reshape(-1).to("cpu", torch.long)
        state.add("raw", raw)
        voice = self._voice(fwd_info, inputs, state)
        final = bool(kwargs.get("is_final_stream_chunk", False))
        kept = filter_tokens(raw, self.config)
        tokens = torch.empty(kept.shape, dtype=torch.long, device=device)
        self._token_stager.copy_(tokens, kept.numpy())
        meta = fwd_info.step_metadata or {}
        return NodeInputs(
            tensor_inputs={SPEECH_TOKENS: tokens},
            kwargs={
                "rid": rid,
                "voice": voice,
                "final": final,
                "n_timesteps": int(meta.get("n_cfm_timesteps", self.config.flow.cfm.n_timesteps)),
            },
        )

    def _run(self, rids, tokens_list, voices, finals, n_timesteps) -> list[NameToTensorList]:
        rows: list[tuple[int, CosyVoice3FlowRow]] = []
        for i, (rid, tokens, voice, final) in enumerate(zip(rids, tokens_list, voices, finals, strict=True)):
            state = self.request_state(rid)
            row = None if state.get("done") else self.plan(state, voice, tokens, final)
            if row is not None:
                rows.append((i, row))
        mels: dict[int, torch.Tensor] = {}
        with _without_tf32():
            for steps in sorted({n_timesteps[i] for i, _ in rows}):
                group = [(i, row) for i, row in rows if n_timesteps[i] == steps]
                for (i, _), mel in zip(group, self.flow.solve_rows([row for _, row in group], steps), strict=True):
                    mels[i] = mel
        out = []
        n_mel = self.config.flow.output_size
        for i, (rid, final) in enumerate(zip(rids, finals, strict=True)):
            state = self.request_state(rid)
            new = torch.zeros(n_mel, 0, device=self.get_device())
            if i in mels:
                offset = int(state.get("settled", 0))
                new = mels[i][:, offset * self.ratio :].float()
                state.add("settled", dict(rows)[i].settle)
            if final:
                state.add("done", True)
            # on the host: HiFT reads it to plan its emit, and a colocated HiFT
            # gets this very tensor
            out.append({MEL: [new], MEL_FINAL: [torch.tensor([final])]})
        return out

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        del batch
        return 1 < len(model_inputs) <= self.config.flow_max_batch_size

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.config.flow_max_batch_size

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]):
        del graph_walk, engine_inputs
        packed: dict[str, Any] = {SPEECH_TOKENS: [inp.tensor_inputs[SPEECH_TOKENS] for inp in inputs]}
        packed.update({key: [inp.kwargs[key] for inp in inputs] for key in inputs[0].kwargs})
        return packed

    def forward(self, graph_walk, engine_inputs, speech_tokens, rid, voice, final, n_timesteps, **kwargs):
        del graph_walk, engine_inputs, kwargs
        return self._run(rid, speech_tokens, voice, final, n_timesteps)[0]

    def forward_batched(self, graph_walk, engine_inputs, speech_tokens, rid, voice, final, n_timesteps, **kwargs):
        del graph_walk, engine_inputs, kwargs
        return dict(zip(rid, self._run(rid, speech_tokens, voice, final, n_timesteps), strict=True))


# ===========================================================================
# 4. HiFT: mel frames -> waveform
# ===========================================================================


class HiFTSubmodule(NodeSubmodule):
    """Vocodes each request's new mel frames behind a bounded window of its
    history (``hift_context_frames``) and emits only samples no later frame can
    change; the final call flushes the rest, so a stream has exactly the length
    of a whole-utterance call."""

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, hift: nn.Module, config: CosyVoice3Config):
        super().__init__()
        self.hift = hift
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        del graph_walk, kwargs
        device = self.get_device()
        mel = inputs[MEL][0].to(device, self.hift.conv_pre.weight.dtype)
        # a host tensor when the flow node shares this worker; a device read
        # only when the two are placed apart
        final = bool(inputs[MEL_FINAL][0].reshape(-1)[0])
        return NodeInputs(tensor_inputs={MEL: mel}, kwargs={"rid": _rid(fwd_info), "final": final})

    def _vocode(self, rid: int, mel: torch.Tensor, final: bool) -> torch.Tensor:
        state = self.request_state(rid)
        stream = state.get("stream")
        if stream is None:
            stream = HiftStreamState(mel=mel.new_zeros(mel.shape[0], 0))
            state.add("stream", stream)
        if mel.shape[-1] == 0 and not final:
            wav = mel.new_zeros(0)
        else:
            wav = stream_vocode(self.hift, stream, mel, final, self.config.hift_context_frames)
        return (wav.clamp(-1, 1) * 32767).to(torch.int16)

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        del batch
        return 1 < len(model_inputs) <= self.config.hift_max_batch_size

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.config.hift_max_batch_size

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]):
        del graph_walk, engine_inputs
        if len(inputs) == 1:
            return {**inputs[0].tensor_inputs, **inputs[0].kwargs}
        return {
            MEL: [inp.tensor_inputs[MEL] for inp in inputs],
            "rid": [inp.kwargs["rid"] for inp in inputs],
            "final": [inp.kwargs["final"] for inp in inputs],
        }

    @torch.no_grad()
    def forward(self, graph_walk, engine_inputs, mel, rid, final, **kwargs) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        return {AUDIO_CHUNK: [self._vocode(rid, mel, final)]}

    @torch.no_grad()
    def forward_batched(self, graph_walk, engine_inputs, mel, rid, final, **kwargs) -> dict[int, NameToTensorList]:
        del graph_walk, engine_inputs, kwargs
        return {r: {AUDIO_CHUNK: [self._vocode(r, m, f)]} for r, m, f in zip(rid, mel, final, strict=True)}
