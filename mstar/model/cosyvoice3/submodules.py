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
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.model.chatterbox.components.audio import kaldi_fbank_80
from mstar.model.chatterbox.components.audio_frontend import resample
from mstar.model.chatterbox.submodules import VoiceCache
from mstar.model.cosyvoice3.components.flow import FlowRow, ReferenceVoice
from mstar.model.cosyvoice3.components.hift import StreamState, stream_vocode
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

logger = logging.getLogger(__name__)

# Edge names (kept in sync with cosyvoice3_model.py)
TEXT_INPUTS = "text_inputs"        # prompt text + text ids
TOKEN_LIMITS = "token_limits"      # [min, max] speech tokens for this text
REF_AUDIO = "ref_audio"            # 24 kHz mono reference clip
VOICE_KEY = "voice_key"            # content hash of the clip
PROMPT_TOKENS = "prompt_tokens"    # speech tokens of the clip (LM prompt)
SPEECH_TOKENS = "speech_tokens"    # LM output, streamed to the flow
PREV_TOKEN = "prev_token"
MEL = "mel"                        # newly settled mel frames
MEL_FINAL = "mel_final"            # [1] bool: the stream has ended
AUDIO_CHUNK = "audio_chunk"
DECODE_LOOP = "decode_loop"

PREFILL_WALKS = ("prefill", "prefill_voice")


def _rid(fwd_info: CurrentForwardPassInfo) -> int:
    if fwd_info.rid_handle == -1:
        raise ValueError(f"request {fwd_info.request_id!r} reached the worker without a rid handle")
    return fwd_info.rid_handle


class ReferenceEncoder:
    """Reference clip -> prompt tokens, prompt mel and x-vector, as
    ``frontend_zero_shot`` computes them: tokens from the 16 kHz clip, mel
    from the 24 kHz one, both cut so the mel has exactly two frames a token.
    Shared by the speaker-encoder and flow nodes when they are colocated."""

    def __init__(self, tokenizer: nn.Module, mel: nn.Module | None, campplus: nn.Module | None):
        self.tokenizer = tokenizer
        self.mel = mel
        self.campplus = campplus

    @torch.no_grad()
    def tokens(self, wav24: torch.Tensor) -> torch.Tensor:
        wav16 = resample(wav24.float(), SAMPLE_RATE, TOKENIZER_SR)
        tokens, lens = self.tokenizer([wav16])
        tokens = tokens[0, : int(lens[0])]
        frames = self._mel_frames(wav24.numel())
        return tokens[: min(int(tokens.numel()), frames // 2)]

    def _mel_frames(self, num_samples: int) -> int:
        from mstar.model.chatterbox.components.audio import MelSpectrogram24k

        if isinstance(self.mel, MelSpectrogram24k):
            return self.mel.num_frames(num_samples)
        return num_samples // 480

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
        del graph_walk, fwd_info, kwargs
        wav = inputs[REF_AUDIO][0].to(self.get_device(), torch.float32).reshape(-1)
        key = int(inputs[VOICE_KEY][0].reshape(-1)[0].item())
        return NodeInputs(tensor_inputs={REF_AUDIO: wav}, kwargs={VOICE_KEY: key})

    def forward(self, graph_walk, engine_inputs, ref_audio: torch.Tensor, voice_key: int, **kwargs) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        tokens = self.cache.get(voice_key)
        if tokens is None:
            tokens = self.encoder.tokens(ref_audio)
            self.cache.put(voice_key, tokens)
        return {PROMPT_TOKENS: [tokens]}


# ===========================================================================
# 2. LM: text (+ prompt speech) -> speech tokens
# ===========================================================================

HISTORY = "history"


class LMSubmodule(ARNodeSubmodule):
    """One LM step for a continuous batch: prefill packs
    ``[sos | prompt text + text | task_id | prompt speech]`` per request, decode
    embeds the previous speech token. Sampling is ``ras_sample`` on float32
    logits, keyed by the request seed and step, so a decode batch replays one
    captured graph per batch size."""

    disable_torch_compile = True
    MAX_BATCH_SIZE = 32
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(self, model: nn.Module, config: CosyVoice3Config):
        super().__init__()
        self.model = model
        self.config = config
        self.lm = config.lm
        self.window = int(config.sampling.ras_window)

    # -- request knobs -------------------------------------------------------

    def _knob(self, step_metadata: Mapping[str, Any] | None, key: str):
        meta = step_metadata or {}
        return meta[key] if key in meta else getattr(self.config.sampling, key)

    def _row_tensors(self, fwd_info: CurrentForwardPassInfo, state, history: torch.Tensor) -> dict[str, torch.Tensor]:
        device = self.get_device()
        meta = fwd_info.step_metadata
        step = int(state.get("generated", 0))
        window = int(self._knob(meta, "ras_window"))
        threshold = float(window) * float(self._knob(meta, "ras_tau")) if window > 0 else 0.0
        ignore_eos = bool((meta or {}).get("ignore_eos", False))
        return {
            HISTORY: history,
            "seed": torch.tensor([int(state["seed"])], dtype=torch.long, device=device),
            "step": torch.tensor([step], dtype=torch.long, device=device),
            "temperature": torch.tensor([float(self._knob(meta, "temperature"))], device=device),
            "top_p": torch.tensor([float(self._knob(meta, "top_p"))], device=device),
            "top_k": torch.tensor([int(self._knob(meta, "top_k"))], dtype=torch.long, device=device),
            "threshold": torch.tensor([threshold], device=device),
            # the reference's minimum length masks only id ``speech_token_size``
            # (its ``sos``), so the other control ids can still end the decode
            "block_sos": torch.tensor([step < int(state["min_len"]) or ignore_eos], device=device),
            "block_stop": torch.tensor([ignore_eos], device=device),
        }

    @torch.no_grad()
    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> ARNodeInputs:
        del kwargs
        device = self.get_device()
        state = self.request_state(_rid(fwd_info))
        if graph_walk in PREFILL_WALKS:
            text_ids = inputs[TEXT_INPUTS][0].to(device, torch.long).reshape(-1)
            limits = inputs[TOKEN_LIMITS][0].reshape(-1).tolist()
            prompt = (
                inputs[PROMPT_TOKENS][0].to(device, torch.long).reshape(-1)
                if inputs.get(PROMPT_TOKENS) else torch.zeros(0, dtype=torch.long, device=device)
            )
            embeds = self.model.build_prefill_embeds(text_ids, prompt)
            state.add_all(
                generated=0, min_len=int(limits[0]), max_len=int(limits[1]),
                seed=int(fwd_info.random_seed) & (2**63 - 1),
                history=torch.full((max(self.window, 1),), -1, dtype=torch.long, device=device),
            )
            history = state[HISTORY]
        elif graph_walk == "decode":
            token = inputs[PREV_TOKEN][0].to(device, torch.long).reshape(-1)[:1]
            embeds = self.model.embed_speech(token)
            # committed in postprocess, so a re-driven step does not add the token twice
            history = torch.cat([state[HISTORY][1:], token])
        else:
            raise ValueError(f"Unknown LM walk {graph_walk!r}")
        return ARNodeInputs(
            input_embeds=embeds, input_seq_len=embeds.shape[0],
            tensor_inputs=self._row_tensors(fwd_info, state, history),
        )

    def preprocess(self, graph_walk, engine_inputs, inputs: list[ARNodeInputs]) -> dict[str, Any]:
        del graph_walk, engine_inputs
        packed: dict[str, Any] = {"input_embeds": torch.cat([inp.input_embeds for inp in inputs], dim=0)}
        for key in inputs[0].tensor_inputs:
            packed[key] = torch.stack([inp.tensor_inputs[key] for inp in inputs]).reshape(len(inputs), -1)
        return packed

    def declare_step(
        self, graph_walk, request_ids, inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None, piecewise_leases=None, **kwargs,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, piecewise_leases, kwargs
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=LM_LABEL, span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={LM_KV: KVStep(), LM_ATTN: AttentionStep(causal=True), LM_POS: PositionStep()},
        )

    def _forward(self, graph_walk, engine_inputs, input_embeds, history, seed, step, temperature, top_p,
                 top_k, threshold, block_sos, block_stop) -> torch.Tensor:
        hidden = self.model.hidden(input_embeds, label=LM_LABEL)
        if graph_walk in PREFILL_WALKS:
            hidden = engine_inputs.resources[LM_ATTN].select_last_hidden(hidden, label=LM_LABEL)
        logits = self.model.logits(hidden)
        ids = torch.arange(logits.shape[-1], device=logits.device)[None, :]
        blocked = (block_sos & (ids == self.lm.sos_token)) | (block_stop & (ids >= self.lm.speech_token_size))
        return ras_sample(
            logits, seed=seed.reshape(-1), step=step.reshape(-1), history=history,
            temperature=temperature.reshape(-1), top_p=top_p.reshape(-1), top_k=top_k.reshape(-1),
            repeat_threshold=threshold.reshape(-1), blocked=blocked,
        )

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        tokens = self._forward(graph_walk, engine_inputs, **kwargs)
        return {SPEECH_TOKENS: [tokens.reshape(-1)[:1]]}

    def forward_batched(self, graph_walk, engine_inputs, **kwargs) -> dict[str, NameToTensorList]:
        tokens = self._forward(graph_walk, engine_inputs, **kwargs).reshape(-1)
        return {rid: {SPEECH_TOKENS: [tokens[i : i + 1]]} for i, rid in enumerate(engine_inputs.request_ids)}

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
        if SPEECH_TOKENS not in outputs:
            return set()
        token = int(outputs[SPEECH_TOKENS][0].reshape(-1)[0].item())
        state = self.request_state(request_id)
        ignore_eos = bool((request_info.step_metadata or {}).get("ignore_eos", False))
        stopped = not ignore_eos and token >= self.lm.speech_token_size
        if stopped or int(state.get("generated", 0)) >= int(state["max_len"]):
            return {DECODE_LOOP}
        return set()

    # -- batching and capture ---------------------------------------------------

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return batch.graph_walk in (*PREFILL_WALKS, "decode") and 0 < len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list[CudaGraphConfig]:
        del tp_world_size
        dtype = self.model.speech_embedding.weight.dtype
        width = max(self.window, 1)
        tensor_inputs = {
            HISTORY: torch.full((width,), -1, dtype=torch.long, device=device),
            "seed": torch.zeros(1, dtype=torch.long, device=device),
            "step": torch.zeros(1, dtype=torch.long, device=device),
            "temperature": torch.ones(1, device=device),
            "top_p": torch.full((1,), 0.8, device=device),
            "top_k": torch.full((1,), 25, dtype=torch.long, device=device),
            "threshold": torch.ones(1, device=device),
            "block_sos": torch.zeros(1, dtype=torch.bool, device=device),
            "block_stop": torch.zeros(1, dtype=torch.bool, device=device),
        }
        return [BatchedCudaGraphConfig(
            capture_graph_walk="decode",
            single_request_inputs=ARNodeInputs(
                input_embeds=torch.zeros(1, self.lm.hidden_size, dtype=dtype, device=device),
                input_seq_len=1, tensor_inputs=tensor_inputs,
            ),
            capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            compile=False,
        )]

    def can_use_cuda_graphs(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return self.can_batch(batch, model_inputs) and super().can_use_cuda_graphs(batch, model_inputs)


# ===========================================================================
# 3. Flow: speech-token stream -> settled mel frames
# ===========================================================================


def filter_tokens(raw: torch.Tensor, config: CosyVoice3Config) -> torch.Tensor:
    """The tokens the reference hands the flow: speech ids only (control ids
    end the stream), and no run of more than ``max_silent_tokens`` silent
    tokens (the LM still saw them)."""
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

    def _voice(self, inputs: NameToTensorList, state) -> ReferenceVoice:
        if inputs.get(REF_AUDIO):
            key = int(inputs[VOICE_KEY][0].reshape(-1)[0].item())
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
        hop = int(min(cfg.stream_hop_tokens * cfg.stream_hop_growth ** k, cap))
        if k == 0:
            hop += -(-prompt_tokens // self.chunk_tokens) * self.chunk_tokens - prompt_tokens
        return hop

    def plan(self, state, voice: ReferenceVoice, tokens: torch.Tensor, final: bool) -> FlowRow | None:
        """The row this request needs solved now, or None (nothing settles yet).

        A stream settles every hop whose end and three look-ahead tokens have
        arrived, in one solve; the final chunk settles the rest."""
        n = int(tokens.numel())
        offset = int(state.get("settled", 0))
        streaming = self.config.stream_hop_tokens > 0
        if final:
            if n == 0 or n <= offset:
                return None
            return FlowRow(voice=voice, tokens=tokens, settle=n, final=True, streaming=streaming)
        if not streaming:
            return None
        target, hops = offset, int(state.get("hops", 0))
        while (nxt := target + self.hop_size(hops, voice.num_prompt_tokens)) + self.lookahead <= n:
            target, hops = nxt, hops + 1
        state.add("hops", hops)
        if target == offset:
            return None
        return FlowRow(voice=voice, tokens=tokens, settle=target, final=False, streaming=True)

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        del graph_walk
        device = self.get_device()
        rid = _rid(fwd_info)
        state = self.request_state(rid)
        chunks = inputs.get(SPEECH_TOKENS) or []
        raw = chunks[0].to(device, torch.long).reshape(-1) if chunks and chunks[0] is not None else None
        if raw is None or raw.numel() == 0:
            raw = state.get("raw", torch.zeros(0, dtype=torch.long, device=device))
        state.add("raw", raw)
        voice = self._voice(inputs, state)
        final = bool(kwargs.get("is_final_stream_chunk", False))
        tokens = filter_tokens(raw, self.config)
        meta = fwd_info.step_metadata or {}
        return NodeInputs(
            tensor_inputs={SPEECH_TOKENS: tokens},
            kwargs={
                "rid": rid, "voice": voice, "final": final,
                "n_timesteps": int(meta.get("n_cfm_timesteps", self.config.flow.cfm.n_timesteps)),
            },
        )

    def _run(self, rids, tokens_list, voices, finals, n_timesteps) -> list[NameToTensorList]:
        rows: list[tuple[int, FlowRow]] = []
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
            out.append({MEL: [new], MEL_FINAL: [torch.tensor([final], device=new.device)]})
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
        final = bool(inputs[MEL_FINAL][0].reshape(-1)[0].item())
        return NodeInputs(tensor_inputs={MEL: mel}, kwargs={"rid": _rid(fwd_info), "final": final})

    def _vocode(self, rid: int, mel: torch.Tensor, final: bool) -> torch.Tensor:
        state = self.request_state(rid)
        stream = state.get("stream")
        if stream is None:
            stream = StreamState(mel=mel.new_zeros(mel.shape[0], 0))
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
