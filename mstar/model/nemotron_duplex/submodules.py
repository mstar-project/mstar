"""NodeSubmodules for NVIDIA NemotronLabs VoiceChat-11B (duplex).

Nodes:
    nano_llm         (KV_CACHE)  — Nemotron-H hybrid backbone      [implemented]
    conformer_encoder(STATELESS) — Fast-Conformer STT + RNN-T head [Phase 4 stub]
    eartts_talker    (KV_CACHE)  — Gemma3 talker → RVQ codes        [Phase 5 stub]
    audio_codec      (STATELESS) — RVQ codec → 22.05 kHz waveform   [Phase 5 stub]

Per the agreed phasing, the ``nano_llm`` text path is drafted first; the audio
stages carry correct interfaces + TODOs and are only wired once the LLM matches
the HF reference numerically.

Correctness note for ``nano_llm``: it runs eager (``disable_torch_compile``)
because Option-A Mamba state lives in ``per_request_states`` (None during
CUDA-graph capture). Batching + CUDA graphs come with the Option-B SSM pool.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.kv_store import PositionInfo
from mstar.model.nemotron_duplex.components.nemotron_h import MambaStateAccessor
from mstar.model.nemotron_duplex.config import NemotronDuplexConfig
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

# Walks in which the nano LLM is (re)starting its recurrent state.
PREFILL_WALKS = {"prefill_text", "prefill_audio"}


class NemotronHLLMSubmodule(ARNodeSubmodule):
    """Nemotron-H backbone node.

    ``prefill_*`` embeds the prompt (or consumes fused ``combined_embeds`` from
    the encoder) and fills the attention KV cache + seeds Mamba state; ``decode``
    single-steps. Sampling is the engine's; EOS handling is in ``check_stop``.
    """

    # Option-A Mamba state is a Python-dict lookup keyed by request id — not
    # compile-/capture-safe. Keep this node eager until the Option-B pool lands.
    disable_torch_compile = True

    def __init__(self, language_model: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.language_model = language_model
        self.embeddings = language_model.embeddings
        self.lm_head = language_model.lm_head
        self.config = config

    def _mamba_state(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, seq_lens: list | None = None,
    ) -> MambaStateAccessor:
        return MambaStateAccessor(
            request_states=engine_inputs.per_request_states or {},
            request_ids=list(engine_inputs.request_ids),
            is_prefill=graph_walk in PREFILL_WALKS,
            seq_lens=seq_lens,
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},  # noqa: B006 - matches base signature
        **kwargs,
    ) -> ARNodeInputs:
        logger.warning("NDTRACE nano.prepare_inputs walk=%s keys=%s", graph_walk, list(inputs.keys()))
        # Frame-synchronous decode: AddFusion of one streamed audio frame with the
        # fed-back previous agent-text + function tokens (the duplex input per frame).
        if "audio_frame" in inputs:
            return self._fuse_frame(inputs)
        # ``combined_embeds`` (pre-fused) or ``text_inputs`` (system-prompt prefill).
        if "combined_embeds" in inputs:
            emb = inputs["combined_embeds"][0]
            return ARNodeInputs(input_embeds=emb, input_seq_len=emb.shape[0])
        ids = inputs["text_inputs"][0]
        return ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])

    def _fuse_frame(self, inputs: NameToTensorList) -> ARNodeInputs:
        cfg = self.config
        audio = inputs["audio_frame"][0]
        audio = audio.unsqueeze(0) if audio.dim() == 1 else audio          # (1, H)
        dev = audio.device

        def _tok(name, default):
            if name in inputs and inputs[name]:
                return inputs[name][0].reshape(-1)[:1].to(dev)
            return torch.tensor([default], device=dev, dtype=torch.long)

        prev_text = _tok("prev_text", cfg.text_bos_id)                     # carry-in BOS
        prev_func = _tok("prev_func", cfg.text_pad_id)
        fused = self.embeddings(prev_text) * cfg.agent_text_weight + audio * cfg.user_audio_weight
        if cfg.use_function_head:
            fused = fused + self.embeddings(prev_func) * cfg.function_weight
        return ARNodeInputs(input_embeds=fused, input_seq_len=1)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        cache_manager = engine_inputs.cache_manager
        seq_lens = [inp.input_seq_len for inp in inputs]
        cache_manager.set_active_label("main")
        # NoPE: plan attention only — no plan_rope (Nemotron-H attention layers
        # apply no positional encoding).
        cache_manager.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")

        out: dict[str, torch.Tensor | Any] = {"seq_lens": seq_lens}
        if inputs[0].input_embeds is not None:
            out["input_embeds"] = torch.cat([inp.input_embeds for inp in inputs], dim=0)
        else:
            out["input_ids"] = torch.cat([inp.input_ids for inp in inputs], dim=0)
        return out

    def can_batch(self, batch, model_inputs) -> bool:
        # Eager (disable_torch_compile) batched decode/prefill: attention batches via
        # the paged-KV pool, Mamba conv/ssm state via the per-request MambaStateAccessor.
        return len(model_inputs) > 1

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        seq_lens: list | None = None,
        **kwargs,
    ) -> NameToTensorList:
        cache_handle = engine_inputs.cache_manager
        if input_embeds is None:
            input_embeds = self.embeddings(input_ids)
        hidden = self.language_model(
            input_embeds,
            cache_handle=cache_handle,
            mamba_state=self._mamba_state(graph_walk, engine_inputs, seq_lens),
        )
        return self._token_outputs(graph_walk, hidden[-1:], engine_inputs, engine_inputs.request_ids[0])

    def _token_outputs(self, graph_walk, last_hidden, engine_inputs, rid) -> NameToTensorList:
        """Text logits (engine samples -> new_token) plus, on the frame-synchronous
        decode walk, the greedily-sampled function token (aux channel) that is fed
        back into the next frame's AddFusion."""
        out: NameToTensorList = {"logits": [self.lm_head(last_hidden)]}
        if graph_walk == "decode" and self.config.use_function_head:
            func_logits = self.language_model.function_head(last_hidden)
            out["new_func"] = [engine_inputs.sampler.sample_aux("function", [rid], func_logits)]
        return out

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        seq_lens: list | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """One fused forward advancing every request in the batch. Attention uses the
        planned paged-KV layout; Mamba conv/ssm state is stepped per request from the
        packed rows (see ``Mamba2Mixer.forward`` / ``MambaStateAccessor``). Per-request
        next-token logits are read at each request's last packed position."""
        import itertools

        cache_handle = engine_inputs.cache_manager
        if input_embeds is None:
            input_embeds = self.embeddings(input_ids)
        hidden = self.language_model(
            input_embeds,
            cache_handle=cache_handle,
            mamba_state=self._mamba_state(graph_walk, engine_inputs, seq_lens),
        )
        ends = list(itertools.accumulate(seq_lens))          # last index+1 per request
        result: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(engine_inputs.request_ids):
            last = hidden[ends[i] - 1: ends[i]]              # (1, H) request i's last token
            result[rid] = self._token_outputs(graph_walk, last, engine_inputs, rid)
        return result

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Feed the sampled agent-text + function tokens back into the next frame's
        # AddFusion (the decode-loop loop-back edges prev_text / prev_func).
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
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.sampling_config["nano_llm"].ignore_eos
        at_max = (
            request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
            >= request_info.max_tokens
        )
        if (not ignore_eos and token == self.config.eos_token_id) or at_max:
            return {"decode_loop"}
        return set()

    def cleanup_request(self, request_id: str):
        # Drop this request's Mamba conv/ssm state (base clears request_states).
        super().cleanup_request(request_id)


# ---------------------------------------------------------------------------
# Audio stages — Phase 4/5 stubs (interfaces fixed; internals TODO)
# ---------------------------------------------------------------------------

class ConformerEncoderSubmodule(NodeSubmodule):
    """Fast-Conformer streaming STT encoder (audio in → fused embeds + transcript).

    STATELESS node. Holds the perception stack plus the RNN-T decoder/joint —
    all three checkpoint subtrees are loaded (weight loading verified). Produces
    ``combined_embeds`` for the nano LLM and a streaming ``transcript_token``.
    Forward is Phase 4 (audio-in).
    """

    def __init__(self, perception: nn.Module, rnnt_decoder: nn.Module, rnnt_joint: nn.Module,
                 config: NemotronDuplexConfig):
        super().__init__()
        self.perception = perception
        self.rnnt_decoder = rnnt_decoder
        self.rnnt_joint = rnnt_joint
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        # ``audio_features`` is the raw 16 kHz mono waveform for this chunk; the
        # Fast-Conformer perception does mel + subsampling + encoder internally.
        wav = inputs["audio_features"][0]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)                                     # (1, N)
        if "audio_seqlens" in inputs:
            lens = inputs["audio_seqlens"][0].to(wav.device)
        else:
            lens = torch.tensor([wav.shape[-1]], device=wav.device)
        return NodeInputs(tensor_inputs={"wav": wav, "lens": lens})

    def forward(self, graph_walk, engine_inputs, wav=None, lens=None, **kwargs) -> NameToTensorList:
        # audio -> per-frame LLM-space embeddings (T, H). These are the *user-audio*
        # channel; streamed as ``audio_frame`` (the LLM decode edge name) and sliced
        # one frame per nano step by the FixedChunkPolicy, where the nano node fuses
        # each with its fed-back previous text/function tokens (AddFusion).
        audio_embeds, _ = self.perception(wav, lens)                  # (1, T, H)
        logger.warning("NDTRACE encoder.forward: produced audio_frame T=%d", audio_embeds.shape[1])
        return {"audio_frame": [audio_embeds[0]]}                     # (T, H)


class EarTTSTalkerSubmodule(ARNodeSubmodule):
    """Gemma3-text talker: nano hidden/text → RVQ codec codes (autoregressive).

    KV_CACHE node. Emits ``num_quantizers`` codebooks per frame; the extra
    codebooks sample from a ``code_predictor`` aux config (see
    Model.get_aux_sampling_configs), mirroring Qwen3-Omni's Talker. Weights
    loaded (verified); forward is Phase 5 (audio-out).
    """

    # Eager: the MoG/MaskGIT/CFG sampling is internal (not the engine's categorical
    # sampler), and the talker keeps its own KV + CFG-uncond state per request in
    # PerRequestState — not compile/capture-safe, like the nano's Mamba state.
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
        ``state`` is the per-request talker state (``None`` warms up from the speaker
        prompt). Returns ``(codes (num_q,), new_state)``. Same math as
        DuplexStream._talker_codec_frame (the verified streaming path)."""
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

    def prepare_inputs(self, graph_walk, fwd_info, inputs, seen_token_mask=None, pos_info={}, **kwargs):  # noqa: B006
        tok = inputs["agent_token"][0]
        return ARNodeInputs(input_ids=tok.reshape(1), input_seq_len=1)

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {"tokens": [int(inp.input_ids.reshape(-1)[0].item()) for inp in inputs]}

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1

    def forward(self, graph_walk, engine_inputs, tokens=None, **kwargs) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        st = engine_inputs.per_request_states[rid]
        codes = self._request_step(st, tokens[0], self.talker.embed_code.weight.device)
        return {"codec_tokens": [codes]}

    def forward_batched(self, graph_walk, engine_inputs, tokens=None, **kwargs) -> dict[str, NameToTensorList]:
        dev = self.talker.embed_code.weight.device
        out: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(engine_inputs.request_ids):
            codes = self._request_step(engine_inputs.per_request_states[rid], tokens[i], dev)
            out[rid] = {"codec_tokens": [codes]}
        return out


class AudioCodecDecoderSubmodule(NodeSubmodule):
    """RVQ codec decoder: talker codes → 22.05 kHz PCM (sliding-window stream).

    Weights loaded (verified); forward is Phase 5 (audio-out).
    """

    def __init__(self, codec: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.codec = codec
        self.config = config

    def get_stateless_flavor(self) -> str:
        return "audio_codec"

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        # ``codec_tokens`` is the streamed RVQ code window (T, num_q) — under a
        # LeftContextChunkPolicy it carries `left_context` prior frames + the new
        # `chunk` frames; we decode the whole window and emit only the new tail so
        # the causal codec has enough left context (mirrors the standalone
        # DuplexStream "decode history, emit tail" and Orpheus/Qwen streaming codec).
        codes = inputs["codec_tokens"][0]
        n_new = int(inputs["codec_tokens"][1].item()) if len(inputs["codec_tokens"]) > 1 else codes.shape[-2]
        return NodeInputs(tensor_inputs={"codes": codes}, kwargs={"n_new": n_new})

    def forward(self, graph_walk, engine_inputs, codes=None, n_new=None, **kwargs) -> NameToTensorList:
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)                                  # (1, T, num_q)
        T = codes.shape[1]
        code_len = torch.tensor([T], device=codes.device)
        with torch.autocast(device_type="cuda", enabled=False):
            audio, _ = self.codec.decode(codes.long(), code_len)        # (1, 1, samples)
        wav = audio.squeeze(1)[0]                                       # (samples,)
        if n_new is not None and n_new < T:                            # emit only the new tail
            spf = wav.shape[0] // T
            wav = wav[-n_new * spf:]
        return {"audio_chunk": [(wav.clamp(-1, 1) * 32767).to(torch.int16)]}
