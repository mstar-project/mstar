"""NodeSubmodules for NVIDIA NemotronLabs VoiceChat-11B (full duplex).

Nodes, and the resources each declares (``NemotronDuplexModel.get_node_resources``):

    conformer_encoder  (none)                          16 kHz speech -> per-frame LLM embeds (+ RNN-T)
    nano_llm           nano_kv, nano_attn, nano_sampler Nemotron-H hybrid Mamba-2 / attention / MLP (9B)
    eartts_talker      (none yet)                      Gemma3 talker -> 31 RVQ codes per frame
    audio_codec        (none)                          RVQ codes -> 22.05 kHz PCM, per-request left context

All four nodes run eager today (``disable_torch_compile``): the Mamba conv/SSM
state and the talker's KV live in ``PerRequestState`` (a dict keyed by request
id), which is neither compile- nor capture-safe. Moving that state onto engine
resources (the recurrent-state pool for Mamba, the KV pool for the talker) is
what unlocks CUDA graphs.
"""
from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.model.nemotron_duplex.components.nemotron_h import MambaStateAccessor
from mstar.model.nemotron_duplex.config import NANO_ATTN, NANO_KV, NANO_SAMPLER, NemotronDuplexConfig
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

# Walks in which the nano LLM (re)starts its recurrent state.
PREFILL_WALKS = {"prefill_text", "prefill_audio"}

# How ``prepare_inputs`` classified a request's step; ``preprocess`` fuses accordingly.
_MODE_FRAME = "frame"      # one streamed audio frame + fed-back prev_text / prev_func
_MODE_PROMPT = "prompt"    # system-prompt token ids (priming)
_MODE_EMBEDS = "embeds"    # pre-fused embeddings


class NemotronHLLMSubmodule(ARNodeSubmodule):
    """Nemotron-H backbone node.

    ``prefill_text`` primes the request with the system prompt exactly like the
    reference: every prompt token is fused with the agent channel (BOS on the
    first token, PAD afterwards) and a PAD function token, nothing is sampled,
    and the fed-back ``prev_text`` / ``prev_func`` leave the prompt region at
    PAD. ``decode`` is frame-synchronous: each step fuses one streamed audio
    frame with the previous agent-text and function tokens (AddFusion), then
    samples the agent text (sampler resource) and the tool-call token (greedy).
    """

    # Mamba state is a Python-dict lookup keyed by request id -- not
    # compile-/capture-safe. Eager until the state moves onto the recurrent pool.
    disable_torch_compile = True

    def __init__(self, language_model: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.language_model = language_model
        self.embeddings = language_model.embeddings
        self.lm_head = language_model.lm_head
        self.config = config

    # -- resources -------------------------------------------------------

    def _mamba_state(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, seq_lens: list[int] | None = None,
    ) -> MambaStateAccessor:
        return MambaStateAccessor(
            request_states=engine_inputs.per_request_states or {},
            request_ids=list(engine_inputs.request_ids),
            is_prefill=graph_walk in PREFILL_WALKS,
            seq_lens=seq_lens,
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        steps = {NANO_KV: KVStep(), NANO_ATTN: AttentionStep(causal=True)}
        if graph_walk == "decode":
            # The prompt region is not sampled (held at PAD), so only decode
            # steps the sampler; the reference's repetition penalty likewise
            # only ever sees generated tokens.
            steps[NANO_SAMPLER] = SamplerStep(apply_penalty=True)
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
        )

    # -- inputs ----------------------------------------------------------

    @staticmethod
    def _tok(inputs: NameToTensorList, name: str, default: int) -> torch.Tensor:
        """The fed-back token under ``name`` as a ``(1,)`` long tensor, or ``default``."""
        if inputs.get(name):
            return inputs[name][0].reshape(-1)[:1].to(torch.long)
        return torch.tensor([default], dtype=torch.long)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        """Host-side bookkeeping only; the embedding lookups and AddFusion run in
        ``preprocess`` on the GPU thread."""
        cfg = self.config
        if "audio_frame" in inputs:
            tensors = {
                "prev_text": self._tok(inputs, "prev_text", cfg.text_bos_id),
                "prev_func": self._tok(inputs, "prev_func", cfg.text_pad_id),
            }
            # The terminal stream chunk (producer_done race) can arrive with the
            # key present but no tensor: run a no-audio step, the loop stops
            # this iteration on the final-chunk signal.
            frames = inputs.get("audio_frame") or []
            if frames:
                tensors["audio_frame"] = frames[0].reshape(1, -1)
            return ARNodeInputs(input_seq_len=1, tensor_inputs=tensors, kwargs={"mode": _MODE_FRAME})
        if "combined_embeds" in inputs:
            emb = inputs["combined_embeds"][0]
            return ARNodeInputs(input_embeds=emb, input_seq_len=emb.shape[0], kwargs={"mode": _MODE_EMBEDS})
        ids = inputs["text_inputs"][0].reshape(-1).to(torch.long)
        return ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0], kwargs={"mode": _MODE_PROMPT})

    def _fuse(self, inp: ARNodeInputs, device: torch.device) -> torch.Tensor:
        """One request's fused input embeddings ``(L, H)`` (AddFusion)."""
        cfg = self.config
        mode = inp.kwargs.get("mode", _MODE_EMBEDS)
        if mode == _MODE_EMBEDS:
            return inp.input_embeds.to(device)
        func_pad = self.embeddings(torch.tensor([cfg.text_pad_id], device=device))
        if mode == _MODE_PROMPT:
            ids = inp.input_ids.to(device)
            agent = torch.full_like(ids, cfg.text_pad_id)
            agent[0] = cfg.text_bos_id
            fused = self.embeddings(agent) * cfg.agent_text_weight + self.embeddings(ids) * cfg.user_audio_weight
            if cfg.use_function_head:
                fused = fused + func_pad * cfg.function_weight
            return fused
        t = inp.tensor_inputs
        fused = self.embeddings(t["prev_text"].to(device)) * cfg.agent_text_weight
        if "audio_frame" in t:
            fused = fused + t["audio_frame"].to(device=device, dtype=fused.dtype) * cfg.user_audio_weight
        if cfg.use_function_head:
            fused = fused + self.embeddings(t["prev_func"].to(device)) * cfg.function_weight
        return fused

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        device = self.embeddings.weight.device
        return {
            "input_embeds": torch.cat([self._fuse(inp, device) for inp in inputs], dim=0),
            "seq_lens": [inp.input_seq_len for inp in inputs],
        }

    # -- forward ---------------------------------------------------------

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        seq_lens: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """One fused forward over the packed batch. Attention runs over the
        planned paged KV; Mamba conv/SSM state is stepped per request from the
        packed rows (``Mamba2Mixer.forward`` / ``MambaStateAccessor``)."""
        cfg = self.config
        rids = list(engine_inputs.request_ids)
        hidden = self.language_model(
            input_embeds, mamba_state=self._mamba_state(graph_walk, engine_inputs, seq_lens), label="main",
        )
        if graph_walk != "decode":
            # Prompt priming: no sampling, the region is held at PAD (reference
            # ``_prime_prompt``), so the first audio frame follows the last PAD.
            pad = torch.full((1,), cfg.text_pad_id, dtype=torch.long, device=hidden.device)
            return {rid: {"prev_text": [pad], "prev_func": [pad]} for rid in rids}

        ends = list(itertools.accumulate(seq_lens))
        last = hidden.index_select(0, torch.tensor([e - 1 for e in ends], device=hidden.device))  # (B, H)
        new_token = engine_inputs.resources[NANO_SAMPLER].sample(rids, logits=self.lm_head(last))
        new_func = None
        if cfg.use_function_head:
            # Tool-call channel: plain argmax, as in the reference.
            new_func = self.language_model.function_head(last).argmax(dim=-1)
        out: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(rids):
            o: NameToTensorList = {"new_token": [new_token[i : i + 1]]}
            if new_func is not None:
                o["new_func"] = [new_func[i : i + 1]]
            out[rid] = o
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        seq_lens: list[int],
        **kwargs,
    ) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, input_embeds, seq_lens)[engine_inputs.request_ids[0]]

    # -- after the forward -------------------------------------------------

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Feed the sampled agent-text + function tokens back into the next
        # frame's AddFusion (the decode loop's prev_text / prev_func edges).
        if "new_token" in outputs:
            outputs["prev_text"] = outputs["new_token"]
        if "new_func" in outputs:
            outputs["prev_func"] = outputs["new_func"]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        # Frame-synchronous duplex: exactly one agent-text token per audio
        # frame, and text EOS is a NORMAL per-frame token (silence / turn
        # boundary), not end-of-generation. The decode loop ends when the
        # audio_frame stream is exhausted (final chunk); only the max-tokens
        # backstop stops it here.
        at_max = (
            request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
            >= request_info.max_tokens
        )
        return {"decode_loop"} if at_max else set()


# ---------------------------------------------------------------------------
# Audio stages
# ---------------------------------------------------------------------------

class ConformerEncoderSubmodule(NodeSubmodule):
    """Fast-Conformer streaming STT encoder: raw 16 kHz audio -> per-frame
    LLM-space embeddings, emitted one frame per stream item so the LLM's
    ``FixedChunkPolicy(1)`` paces the frame-synchronous decode loop.

    Holds the perception stack plus the RNN-T decoder/joint (loaded; the
    user-transcript channel is not emitted yet). Declares no resources.
    """

    # Variable-length per-utterance audio: torch.compile would recompile per
    # shape (and inductor raised on some shapes). Runs once per request, so
    # eager is fine; revisit with the batched-encoder perf pass.
    disable_torch_compile = True

    def __init__(self, perception: nn.Module, rnnt_decoder: nn.Module, rnnt_joint: nn.Module,
                 config: NemotronDuplexConfig):
        super().__init__()
        self.perception = perception
        self.rnnt_decoder = rnnt_decoder
        self.rnnt_joint = rnnt_joint
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        # ``audio_features`` is the raw 16 kHz mono waveform; the perception
        # stack does mel + subsampling + encoder internally.
        wav = inputs["audio_features"][0]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)                                     # (1, N)
        if "audio_seqlens" in inputs:
            lens = inputs["audio_seqlens"][0].to(wav.device)
        else:
            lens = torch.tensor([wav.shape[-1]], device=wav.device)
        return NodeInputs(tensor_inputs={"wav": wav, "lens": lens})

    def forward(self, graph_walk, engine_inputs, wav=None, lens=None, **kwargs) -> NameToTensorList:
        audio_embeds, _ = self.perception(wav, lens)                  # (1, T, H)
        # One stream item per frame: a single (T, H) tensor would be ONE item
        # (the whole utterance), which the chunk policy could not slice.
        return {"audio_frame": list(audio_embeds[0])}                # T x (H,)


class EarTTSTalkerSubmodule(ARNodeSubmodule):
    """Gemma3 talker: one agent text token -> ``num_quantizers`` RVQ codes.

    The MoG / MaskGIT / CFG sampling is internal (not the engine's categorical
    sampler), and the talker keeps its own KV + CFG-unconditional state per
    request in ``PerRequestState``. Declares no resources yet; putting its KV on
    the engine's KV pool is the next step.
    """

    disable_torch_compile = True

    def __init__(self, talker: nn.Module, config: NemotronDuplexConfig,
                 subword_to_char: dict | None = None, char_pad_idx: int | None = None):
        super().__init__()
        self.talker = talker
        self.config = config
        self._s2c = subword_to_char
        self._char_pad = char_pad_idx

    def _step(self, state, text_tok: int, device, generator):
        """One talker frame: agent text token -> ``num_quantizers`` RVQ codes.
        ``state`` is the per-request talker state (``None`` warms up from the
        speaker prompt). Returns ``(codes (num_q,), new_state)``; same math as
        ``DuplexStream._talker_codec_frame`` (the verified streaming path)."""
        cfg, talker = self.config, self.talker
        if state is None:
            state = talker.init_state(
                1, speaker="Aria", device=device,
                subword_id_to_char_ids=self._s2c, char_pad_idx=self._char_pad,
                text_pad_id=cfg.text_pad_id, text_eos_id=cfg.text_eos_id,
                speech_pad_id=cfg.eartts.codebook_size,
            )
            prev = state.get("prev_codes")
            state["_prev_codes"] = prev if prev is not None else talker.initial_prev_codes(1, device=device)
        cur = torch.tensor([[text_tok]], device=device)
        cond = talker.text_conditioning(cur, torch.ones_like(cur, dtype=torch.bool), self._s2c, self._char_pad)
        codes, new_state = talker.infer_codes_one_step(
            state, cur, cur, state["_prev_codes"], cond=cond, text_eos_id=cfg.text_eos_id,
            num_iter=cfg.eartts.inference_num_iter, guidance_scale=cfg.eartts.inference_guidance_scale,
            noise_scale=cfg.eartts.inference_noise_scale, top_p=cfg.eartts.inference_top_p,
            generator=generator,
        )
        new_state["_prev_codes"] = codes
        return codes.squeeze(0), new_state

    def _request_step(self, st, text_tok: int, device) -> torch.Tensor:
        """Advance one request (state carried in its PerRequestState) -> codes."""
        state = st.get("talker_state")
        gen = st.get("talker_gen")
        if gen is None:
            gen = torch.Generator(device=device).manual_seed(self.config.eartts.inference_seed)
            st.add("talker_gen", gen)
        codes, state = self._step(state, text_tok, device, gen)
        st.add("talker_state", state)
        return codes

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> ARNodeInputs:
        # The LLM streams the sampled agent text token under "new_token".
        tok = inputs["new_token"][0]
        return ARNodeInputs(input_ids=tok.reshape(1), input_seq_len=1)

    def preprocess(self, graph_walk, engine_inputs, inputs):
        # Host read of the token ids: the talker's text conditioning is built
        # per token on the host. Eager-only; goes away with the batched talker.
        return {"tokens": [int(inp.input_ids.reshape(-1)[0].item()) for inp in inputs]}

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def forward(self, graph_walk, engine_inputs, tokens=None, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, tokens=tokens)[engine_inputs.request_ids[0]]

    def forward_batched(self, graph_walk, engine_inputs, tokens=None, **kwargs) -> dict[str, NameToTensorList]:
        dev = self.talker.embed_code.weight.device
        out: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(engine_inputs.request_ids):
            codes = self._request_step(engine_inputs.per_request_states[rid], tokens[i], dev)
            out[rid] = {"codec_tokens": [codes]}
        return out


class AudioCodecDecoderSubmodule(NodeSubmodule):
    """RVQ codec decoder: talker codes -> 22.05 kHz int16 PCM (streaming).

    The codec is causal with a multi-frame receptive field, so decoding each
    chunk in isolation clicks at the boundaries. Each request keeps a rolling
    context of its previous ``codec_left_context_frames`` code frames (in its
    ``PerRequestState``); a chunk of NEW frames is decoded as ``context + new``
    and only the new frames' samples are emitted. Same "decode with left
    context, emit the tail" math as the verified standalone path, but O(1) per
    chunk instead of re-decoding the whole history. Declares no resources.
    """

    # Per-request context is keyed by request id -> eager; fp32 codec.
    disable_torch_compile = True
    disable_autocast = True

    CONTEXT_KEY = "codec_ctx"

    def __init__(self, codec: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.codec = codec
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        # ``codec_tokens`` is this chunk's NEW RVQ frames (T_new, num_q); the
        # Talker->Codec connection is a non-overlapping FixedChunkPolicy, so the
        # left context comes from per-request state, not the stream.
        return NodeInputs(tensor_inputs={"codes": inputs["codec_tokens"][0]})

    def forward(self, graph_walk, engine_inputs, codes=None, **kwargs) -> NameToTensorList:
        st = self.request_state(engine_inputs.request_ids[0])
        lc = self.config.eartts.codec_left_context_frames
        if codes.dim() == 3:
            codes = codes[0]                                            # (T_new, num_q)
        elif codes.dim() == 1:
            codes = codes.unsqueeze(0)                                  # single frame -> (1, num_q)
        prev = st.get(self.CONTEXT_KEY)                                 # (n_ctx, num_q) or None
        n_ctx = 0 if prev is None else prev.shape[0]
        full = codes if prev is None else torch.cat([prev, codes], dim=0)
        Tf = full.shape[0]
        code_len = torch.tensor([Tf], device=full.device)
        audio, _ = self.codec.decode(full.long().unsqueeze(0), code_len)  # (1, 1, samples)
        wav = audio.squeeze(1)[0]                                       # (samples,)
        spf = wav.shape[0] // Tf                                        # samples per frame
        new_wav = wav[n_ctx * spf:]                                     # emit only the new frames
        st.add(self.CONTEXT_KEY, full[-lc:].detach())                  # roll the context forward
        return {"audio_chunk": [(new_wav.clamp(-1, 1) * 32767).to(torch.int16)]}
