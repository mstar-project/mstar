"""NemotronDuplexModel — NVIDIA NemotronLabs VoiceChat-11B in M*.

A full-duplex speech-to-speech model with three stages:

    conformer_encoder (STATELESS) — 16 kHz speech → fused embeds + RNN-T transcript
    nano_llm          (KV_CACHE)  — Nemotron-H hybrid Mamba-2/attn/MLP backbone (9B)
    eartts_talker     (KV_CACHE)  — Gemma3 talker → 31-codebook RVQ codes
    audio_codec       (STATELESS) — RVQ codes → 22.05 kHz PCM

Full-duplex: the model consumes user speech continuously and emits agent text +
agent speech as it generates, frame-synchronously in one backbone, so it handles
natural turn-taking and barge-in/interruptions.

Input:  optional system prompt + user speech.  Output: agent text + agent speech
(+ streaming user transcription, RNN-T channel implemented but not yet wired).

INFERENCE PATHS (all live, verified vs the NeMo reference):
    * ``offline_inference`` — batched offline S2S (frame-synchronous perception →
      AddFusion → nano no-cache re-prefill → sample → talker RVQ → codec).
    * ``create_stream`` / ``DuplexStream`` — online streaming duplex with a
      persistent nano cache (attn KV + Mamba conv/ssm state) and per-frame talker;
      matches the offline text path and vocalizes with CFG + noise sampling.
    * ``realtime_api`` — OpenAI Realtime-API WebSocket server over ``DuplexStream``.

The M*-serving-engine graph path (``prefill_text`` → ``decode`` walk graphs, async
partitions with ``StreamingGraphEdge`` fan-out of transcript / text / audio) is
sketched in ``_duplex_design`` below and is the remaining engine-integration work;
the standalone paths above are what run today.

Contract methods crib from ``mstar/model/orpheus/orpheus_model.py`` (single
streaming LLM+codec) and ``mstar/model/qwen3_omni/qwen3_omni_model.py`` (omni,
multi-KV, aux sampling, async partitions).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.base import EngineType
from mstar.engine.kv_cache_engine import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.nemotron_duplex.config import NemotronDuplexConfig
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.sampling import SamplingConfig

logger = logging.getLogger(__name__)


def _sample_text_token(
    logits: torch.Tensor,          # (B, vocab)
    generated: torch.Tensor,       # (B, T) tokens so far
    step: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    special_ids: set | None = None,
) -> torch.Tensor:
    """Text-channel sampler — exact port of the reference ``_sample_text_token``.

    Greedy when ``temperature==0``. Order: repetition/presence penalty ->
    temperature -> top-p -> softmax + multinomial. If a row's argmax is a
    special id, that row stays greedy. Function/ASR channels use plain argmax.
    """
    special_ids = special_ids or set()
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    sampled = logits.argmax(dim=-1).clone()
    for b in range(logits.shape[0]):
        if logits[b].argmax().item() in special_ids:
            continue
        lg = logits[b].clone()
        if (repetition_penalty != 1.0 or presence_penalty != 0.0) and step > 0:
            prev = generated[b, :step].unique()
            if special_ids:
                st = torch.tensor(list(special_ids), device=prev.device)
                prev = prev[~torch.isin(prev, st)]
            for tok in prev:
                tid = tok.item()
                if repetition_penalty != 1.0:
                    lg[tid] = lg[tid] / repetition_penalty if lg[tid] > 0 else lg[tid] * repetition_penalty
                if presence_penalty != 0.0:
                    lg[tid] -= presence_penalty
        if temperature != 1.0:
            lg = lg / temperature
        if top_p < 1.0:
            s_lg, s_idx = torch.sort(lg, descending=True)
            cum = torch.cumsum(torch.softmax(s_lg, dim=-1), dim=-1)
            rm = cum > top_p
            rm[1:] = rm[:-1].clone()
            rm[0] = False
            lg[s_idx[rm]] = float("-inf")
        sampled[b] = torch.multinomial(torch.softmax(lg, dim=-1), num_samples=1).item()
    return sampled


class _StreamOutput:
    """One streaming step's agent output: an int16 PCM audio chunk and/or a text delta."""

    __slots__ = ("audio", "text")

    def __init__(self, audio: bytes | None = None, text: str | None = None):
        self.audio = audio
        self.text = text


class DuplexStream:
    """Online (streaming) duplex inference session — the Phase-B/C path.

    Audio is fed in chunks via ``append_audio``; each new ~80 ms perception frame
    runs one duplex step (fuse -> nano cached-decode step -> sample text/function
    -> talker frame -> codec frame) and yields agent audio + text deltas. The
    nano attention-KV + Mamba conv/ssm state and the talker KV state persist
    across the whole stream (O(1) per frame). Perception is re-encoded over a
    growing buffer each chunk — correct because the Conformer is causal
    (att_context [70,0]); a true streaming Conformer cache is a later perf item.

    Produces the SAME result as ``offline_inference`` on the concatenated audio.
    """

    def __init__(self, model, device: str, voice: str, temperature: float, top_p: float,
                 repetition_penalty: float, prompt_tokens: torch.Tensor | None = None):
        self.m = model
        self.cfg = model.config
        self.device = device
        self.voice = voice
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.special_ids = {self.cfg.text_pad_id, self.cfg.text_bos_id, self.cfg.text_eos_id}
        # System-prompt tokens (fed via the audio channel, gen_text held at PAD,
        # exactly as offline_inference prepends the prompt). Primes the nano cache.
        self.prompt_tokens = None if prompt_tokens is None else prompt_tokens.to(device).view(-1)

        self.perception = model.get_submodule("conformer_encoder", device=device).perception
        self.nano = model.get_submodule("nano_llm", device=device).language_model
        self.talker = model.get_submodule("eartts_talker", device=device).talker
        self.codec = model.get_submodule("audio_codec", device=device).codec

        # talker char-vocab conditioning maps (built once)
        tok = model._talker_tokenizer()
        from mstar.model.nemotron_duplex.components.eartts_talker import build_char_vocab, subword_to_char_ids
        cv = build_char_vocab(tok)
        self._s2c, _ = subword_to_char_ids(tok, cv)
        self._char_pad = len(cv)

        self.reset()

    def reset(self):
        self._wav = torch.zeros(0)
        self._processed = 0                    # perception frames already stepped
        self._nano_cache = None                # nano decode cache (B.1)
        self._gen_text = []                    # sampled agent text tokens (all frames)
        self._prev_func = self.cfg.text_pad_id
        self._talker_state = None
        self._prev_codes = None
        self._codes_hist = None                # (1, n, num_q) accumulated RVQ codes
        self._emitted = 0                      # waveform samples already emitted
        self._t = 0
        # Dedicated seeded RNG for the talker's stochastic MoG sampling — keeps the
        # audio reproducible and off the rare collapse-to-constant-code states an
        # uncontrolled global RNG hits (mirrors the reference's manual_seed).
        self._talker_gen = torch.Generator(device=self.device).manual_seed(self.cfg.eartts.inference_seed)
        self._prime_prompt()                   # feed the system prompt (if any) first

    @torch.no_grad()
    def _prime_prompt(self):
        """Prime the nano cache with the system prompt, mirroring how
        ``_infer_one`` prepends ``prompt_emb`` to the audio channel and holds
        ``gen_text`` at PAD across the prompt region (no sampling, no talker).
        Frame 0's agent channel is BOS; later prompt frames use PAD."""
        if self.prompt_tokens is None or self.prompt_tokens.numel() == 0:
            return
        nano, cfg = self.nano, self.cfg
        embed_tokens = nano.embed_tokens
        prompt_emb = embed_tokens(self.prompt_tokens)          # (P, H)
        for i in range(prompt_emb.shape[0]):
            prev_text = cfg.text_bos_id if i == 0 else cfg.text_pad_id
            agent_emb = embed_tokens(torch.tensor([prev_text], device=self.device))
            func_emb = embed_tokens(torch.tensor([cfg.text_pad_id], device=self.device))
            fused = agent_emb * cfg.agent_text_weight + prompt_emb[i].unsqueeze(0) * cfg.user_audio_weight
            if cfg.use_function_head:
                fused = fused + func_emb * cfg.function_weight
            self._nano_hidden(fused)                           # primes cache; no sampling
            self._gen_text.append(cfg.text_pad_id)             # prompt region held at PAD

    def append_audio(self, pcm_int16: bytes) -> list[_StreamOutput]:
        chunk = torch.frombuffer(bytearray(pcm_int16), dtype=torch.int16).float() / 32768.0
        self._wav = torch.cat([self._wav, chunk])
        return self._step_new_frames()

    def flush(self) -> list[_StreamOutput]:
        # End of input: no more audio will arrive, so the held-back trailing frames
        # are now final — consume them too (matches offline's zero-padded tail).
        return self._step_new_frames(final=True)

    @torch.no_grad()
    def _step_new_frames(self, final: bool = False) -> list[_StreamOutput]:
        if self._wav.numel() < self.cfg.stt.hop_length:
            return []
        sig = self._wav.unsqueeze(0).to(self.device)
        lens = torch.tensor([self._wav.shape[0]], device=self.device)
        audio_embeds, _ = self.perception(sig, lens)   # (1, T, H) — causal, re-encoded
        T = audio_embeds.shape[1]
        nano, cfg = self.nano, self.cfg
        embed_tokens = nano.embed_tokens
        outs: list[_StreamOutput] = []

        # Hold back the encoder's right-context lookahead frames: the newest
        # `streaming_lookahead_frames` are not yet final (they change as more audio
        # arrives), so consuming them now would permanently diverge from offline.
        # At `final` (flush) there is no future audio, so consume everything.
        limit = T if final else max(self._processed, T - cfg.stt.streaming_lookahead_frames)
        for t in range(self._processed, limit):
            # Global position — with a primed system prompt, gen_text already holds
            # `prompt_len` PADs, so the first audio frame's prev is the last prompt
            # PAD (not BOS). BOS applies only at true global frame 0 (no prompt).
            prev_text = cfg.text_bos_id if len(self._gen_text) == 0 else self._gen_text[-1]
            agent_emb = embed_tokens(torch.tensor([prev_text], device=self.device))
            func_emb = embed_tokens(torch.tensor([self._prev_func], device=self.device))
            fused = agent_emb * cfg.agent_text_weight + audio_embeds[0, t].unsqueeze(0) * cfg.user_audio_weight
            if cfg.use_function_head:
                fused = fused + func_emb * cfg.function_weight   # (1, H) == (L=1, H)

            hidden = self._nano_hidden(fused)           # (1, H)
            text_logits = nano.lm_head(hidden)          # (1, vocab)
            func_logits = nano.function_head(hidden)
            gt = torch.tensor(self._gen_text, device=self.device).view(1, -1)
            text_tok = int(_sample_text_token(
                text_logits, gt, len(self._gen_text), self.temperature, self.top_p,
                self.repetition_penalty, special_ids=self.special_ids,
            )[0])
            self._gen_text.append(text_tok)
            self._prev_func = int(func_logits.argmax(-1)[0])

            outs.append(self._talker_codec_frame(text_tok))
            self._t += 1
        self._processed = limit   # held-back frames stay unconsumed until finalized
        return outs

    def _nano_hidden(self, fused: torch.Tensor) -> torch.Tensor:
        """One nano step with persistent cache (B.1 cached decode; re-prefill fallback)."""
        if hasattr(self.nano.llm, "decode_step") and hasattr(self.nano.llm, "prefill"):
            if self._nano_cache is None:
                h, self._nano_cache = self.nano.llm.prefill(fused)
                return h[-1:]
            return self.nano.llm.decode_step(fused, self._nano_cache)
        # Fallback (pre-B.1): accumulate fused embeds and re-prefill (O(T^2)).
        self._fused_hist = fused if self._nano_cache is None else torch.cat([self._nano_cache, fused], 0)
        self._nano_cache = self._fused_hist
        return self.nano.llm.offline_forward(self._fused_hist)[-1:]

    def _talker_codec_frame(self, text_tok: int) -> _StreamOutput:
        talker, cfg = self.talker, self.cfg
        if self._talker_state is None:
            self._talker_state = talker.init_state(
                1, speaker=self.voice, device=self.device,
                subword_id_to_char_ids=self._s2c, char_pad_idx=self._char_pad,
                text_pad_id=cfg.text_pad_id, text_eos_id=cfg.text_eos_id,
                speech_pad_id=cfg.eartts.codebook_size,
            )
            self._prev_codes = self._talker_state.get("prev_codes")
            if self._prev_codes is None:
                self._prev_codes = talker.initial_prev_codes(1, device=self.device)
        cur = torch.tensor([[text_tok]], device=self.device)
        mask = torch.ones_like(cur, dtype=torch.bool)
        cond = talker.text_conditioning(cur, mask, self._s2c, self._char_pad)
        codes, self._talker_state = talker.infer_codes_one_step(
            self._talker_state, cur, cur, self._prev_codes, cond=cond,
            text_eos_id=cfg.text_eos_id, temperature=self.temperature,
            num_iter=cfg.eartts.inference_num_iter,
            guidance_scale=cfg.eartts.inference_guidance_scale,
            noise_scale=cfg.eartts.inference_noise_scale,
            top_p=cfg.eartts.inference_top_p,
            generator=self._talker_gen,
        )
        self._prev_codes = codes
        # Codec has a multi-frame (causal) receptive field, so decode the full code
        # history and emit only the newly-produced tail (matches offline decode).
        frame = codes.unsqueeze(1)                                      # (1, 1, num_q)
        self._codes_hist = frame if self._codes_hist is None else torch.cat([self._codes_hist, frame], dim=1)
        with torch.autocast(device_type="cuda", enabled=False):
            audio, _ = self.codec.decode(
                self._codes_hist, torch.tensor([self._codes_hist.shape[1]], device=self.device),
            )                                                          # (1, 1, samples)
        wav = audio.squeeze(1)[0]                                       # (samples,)
        new = wav[self._emitted:]
        self._emitted = wav.shape[0]
        pcm = (new.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy().tobytes()
        return _StreamOutput(audio=pcm, text=self.m.tokenizer.decode([text_tok]))


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)
    except Exception as e:  # noqa: BLE001 - fall back to the raw id, mirrors Orpheus
        logger.warning("Error downloading %s from huggingface: %s", repo_id, e)
        return repo_id
    return str(Path(local_dir))


class NemotronDuplexModel(Model):
    """NVIDIA NemotronLabs VoiceChat-11B (duplex S2S)."""

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.config = NemotronDuplexConfig()
        self._tokenizer = None  # lazy — not needed in dummy mode
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @property
    def tokenizer(self):
        # The text tokenizer lives in the Spark repo's ``nano/`` folder (the base
        # repo ships only ``rnnt_tokenizer/``). 131072-token vocab; 256 single-
        # char tokens -> the talker's char vocab. Loaded lazily.
        if self._tokenizer is None:
            import os

            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer

            repo = "pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark"
            local_dir = None
            for f in ("nano/tokenizer.json", "nano/tokenizer_config.json", "nano/special_tokens_map.json"):
                local_dir = os.path.dirname(hf_hub_download(repo, f, cache_dir=self.cache_dir))
            self._tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        return self._tokenizer

    def _talker_tokenizer(self):
        """Adapter exposing the NeMo-tokenizer interface (``.tokenizer.vocab`` /
        ``.vocab``) the talker's char-vocab builder expects, over the HF tokenizer."""
        import types

        v = self.tokenizer.get_vocab()
        return types.SimpleNamespace(tokenizer=types.SimpleNamespace(vocab=v), vocab=v)

    # -------------------------------------------------------------------
    # KV cache config — nano_llm (attention layers only) + eartts_talker
    # -------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        nano = self.config.nano
        eartts = self.config.eartts
        return [
            KVCacheConfig(
                # Only the ``*`` (attention) layers hold a KV cache; the Mamba
                # and MLP layers do not. The submodule maps global layer -> this
                # dense attention index.
                num_layers=nano.num_attention_layers,
                num_kv_heads=nano.num_key_value_heads,
                head_dim=nano.head_dim,
                max_seq_len=nano.max_position_embeddings,
                num_qo_heads=nano.num_attention_heads,
                nodes=["nano_llm"],
            ),
            # Phase 5 — declared now so layout is stable; only used once the
            # talker submodule is implemented.
            KVCacheConfig(
                num_layers=eartts.num_hidden_layers,
                num_kv_heads=eartts.num_key_value_heads,
                head_dim=eartts.head_dim,
                max_seq_len=nano.max_position_embeddings,
                num_qo_heads=eartts.num_attention_heads,
                nodes=["eartts_talker"],
            ),
        ]

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "conformer_encoder": EngineType.STATELESS,  # Phase 4
            "nano_llm": EngineType.KV_CACHE,            # implemented (text path)
            "eartts_talker": EngineType.KV_CACHE,        # Phase 5
            "audio_codec": EngineType.STATELESS,        # Phase 5
        }

    # -------------------------------------------------------------------
    # Graph walks. Text path is live; audio walks are Phase 6 (see design note).
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill_text = GraphNode(
            name="nano_llm",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="new_token",
                    conductor_new_token=True,
                    persist=True,
                ),
            ],
        )
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="nano_llm",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node="nano_llm", name="text_inputs"),
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        conductor_new_token=True,
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return dict(prefill_text=prefill_text, decode=decode)

    @staticmethod
    def _duplex_design() -> str:
        """Phase 6 target (documentation only).

        Nodes/partitions:
            ENC  partition: conformer_encoder  (prefill_audio) — emits
                 ``combined_embeds`` -> nano_llm and streaming ``transcript_token``
                 -> EMIT_TO_CLIENT (modality "text").
            LLM  partition: nano_llm (prefill_text | prefill_audio | decode) —
                 streams ``new_token`` (agent text) to the client and a talker
                 conditioning signal to the TTS partition via StreamingGraphEdge.
            TTS  partition: eartts_talker (tts_decode Loop) -> audio_codec
                 (codec_chunk) -> EMIT_TO_CLIENT (modality "audio"), with a
                 SlidingWindowChunkPolicy on the talker->codec connection
                 (Orpheus pattern) for low-latency 22.05 kHz output.

        Requires overriding get_partition_topology / get_partitions and routing
        cross-partition tensors with StreamingGraphEdge (target_partition=...).
        """
        return "see docstring"

    # -------------------------------------------------------------------
    # Prompt processing
    # -------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        # Phase 4: when audio is in ``input_modalities``, run the conformer
        # encoder path to derive ``combined_embeds`` from ``audio_inputs``.
        if "audio" in input_modalities:
            raise NotImplementedError("Audio input (conformer encoder) — Phase 4.")
        if prompt is None:
            return {}
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(torch.long)
        return {"text_inputs": [ids]}

    # -------------------------------------------------------------------
    # Forward-pass-args state machine (single "default" partition, text path)
    # -------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill_text",
            is_prefill=True,
        )
        graph_edge = GraphEdge(next_node="nano_llm", name="text_inputs")
        graph_edge.tensor_info = input_signals.get("text_inputs", [])
        inputs = [graph_edge]
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections=None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            request_done = True
            metadata.kwargs["decode_finished"] = True

        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True,
            )

        graph_edge = GraphEdge(next_node="nano_llm", name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [graph_edge]
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata={"is_prefill": metadata.is_prefill},
        )

    # -------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------

    def get_sampling_config(self, node_name: str, model_kwargs: dict | None = None) -> SamplingConfig | None:
        if node_name != "nano_llm":
            return None  # eartts_talker sampling: Phase 5 (+ get_aux_sampling_configs)
        model_kwargs = model_kwargs or {}
        keys = ["temperature", "top_p", "repetition_penalty", "ignore_eos"]
        params = {k: model_kwargs.get(k, getattr(self.config, k)) for k in keys}
        return SamplingConfig(vocab_size=self.config.vocab_size, **params)

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.eartts.sample_rate

    # -------------------------------------------------------------------
    # Online (streaming) duplex inference — Phase B/C.
    # -------------------------------------------------------------------

    def prepare_streaming(self, device: str = "cuda"):
        """Warm (build + load) all four submodules so streaming starts promptly."""
        for node in ("conformer_encoder", "nano_llm", "eartts_talker", "audio_codec"):
            self.get_submodule(node, device=device)

    def encode_prompt(self, system_prompt: str, device: str = "cuda") -> torch.Tensor | None:
        """Tokenize a system prompt to nano text-token ids: ``[bos] + ids + [eos]``
        (mirrors the reference ``encode_system_prompt``). Returns ``(1, P)`` or None."""
        if not system_prompt or not system_prompt.strip():
            return None
        tok = self.tokenizer
        ids = tok.encode(system_prompt, add_special_tokens=False)
        ids = [tok.bos_token_id] + ids + [tok.eos_token_id]
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def create_stream(
        self, device: str = "cuda", voice: str = "Aria",
        temperature: float = 0.0, top_p: float = 1.0, repetition_penalty: float = 1.0,
        prompt_tokens: torch.Tensor | None = None, system_prompt: str | None = None,
    ) -> DuplexStream:
        """Open an online duplex streaming session (see ``DuplexStream``).

        A system prompt may be passed either pre-tokenized (``prompt_tokens``,
        ``(1, P)`` nano text ids) or as raw text (``system_prompt``); it primes
        the nano cache before any audio, exactly as offline prepends it."""
        if prompt_tokens is None and system_prompt is not None:
            prompt_tokens = self.encode_prompt(system_prompt, device=device)
        return DuplexStream(self, device, voice, temperature, top_p, repetition_penalty, prompt_tokens)

    # -------------------------------------------------------------------
    # Standalone offline inference (Phase A — mirrors the reference
    # NemotronVoiceChat.offline_inference; bypasses the M* serving engine).
    # -------------------------------------------------------------------

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,       # (B, num_samples) 16 kHz mono
        input_signal_lens: torch.Tensor,  # (B,)
        prompt_tokens: torch.Tensor | None = None,      # (B, P) text token ids, or None
        prompt_token_lens: torch.Tensor | None = None,  # (B,)
        device: str = "cuda",
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        decode_audio: bool = True,
    ) -> dict:
        """Duplex speech-to-speech offline inference (fp32).

        Frame-synchronous loop mirroring the reference: perception -> per-frame
        [AddFusion(prev agent text, audio, prev function) -> nano no-cache re-prefill
        step -> sample text/function] -> talker RVQ codes -> codec -> waveform.

        Batched via a per-utterance loop — correct results for B>1; true continuous
        batching is the Phase-B engine path. Greedy when temperature==0, else
        temperature/top-p/repetition-penalty on the text channel (function/ASR
        channels are always greedy). Returns
        ``{tokens_text, tokens_function, tokens_audio, audio, audio_len, text}``.
        """
        B = input_signal.shape[0]
        outs = []
        for b in range(B):
            p = None
            if prompt_tokens is not None:
                pl = int(prompt_token_lens[b]) if prompt_token_lens is not None else prompt_tokens.shape[1]
                p = prompt_tokens[b : b + 1, :pl]
            outs.append(self._infer_one(
                input_signal[b : b + 1], input_signal_lens[b : b + 1], p, device,
                temperature, top_p, repetition_penalty, decode_audio,
            ))
        return self._collate(outs, decode_audio)

    @torch.no_grad()
    def _infer_one(
        self, input_signal, input_signal_lens, prompt_tokens, device,
        temperature, top_p, repetition_penalty, decode_audio,
    ) -> dict:
        """Single-utterance (B==1) duplex inference; see ``offline_inference``."""
        cfg = self.config
        special_ids = {cfg.text_pad_id, cfg.text_bos_id, cfg.text_eos_id}
        # --- build the component modules (loads real weights) ---
        conf_sub = self.get_submodule("conformer_encoder", device=device)
        nano_sub = self.get_submodule("nano_llm", device=device)
        perception = conf_sub.perception
        nano = nano_sub.language_model
        embed_tokens = nano.embed_tokens

        # --- perception: audio -> per-frame embeddings (B, T, H) ---
        audio_embeds, lengths = perception(input_signal.to(device), input_signal_lens.to(device))
        B, T, H = audio_embeds.shape

        # --- prepend the system prompt embeddings, if any ---
        prompt_len = 0
        if prompt_tokens is not None and prompt_tokens.numel() > 0:
            prompt_len = prompt_tokens.shape[1]
            prompt_emb = embed_tokens(prompt_tokens.to(device))            # (1, P, H)
            audio_embeds = torch.cat([prompt_emb, audio_embeds], dim=1)    # (1, P+T, H)
            T = audio_embeds.shape[1]

        gen_text = torch.full((B, T), cfg.text_pad_id, device=device, dtype=torch.long)
        gen_function = torch.full((B, T), cfg.text_pad_id, device=device, dtype=torch.long)
        input_embeds = torch.zeros_like(audio_embeds)

        def fuse(agent_emb, audio_emb, function_emb):
            out = agent_emb * cfg.agent_text_weight + audio_emb * cfg.user_audio_weight
            if cfg.use_function_head:
                out = out + function_emb * cfg.function_weight
            return out

        def nano_step(seq_embeds):
            # seq_embeds: (L, H) fused embeddings for frames [0:t+1]; returns the
            # last position's text + function logits (non-cache re-prefill).
            hidden = nano.llm.offline_forward(seq_embeds)      # (L, H)
            last = hidden[-1:]                                  # (1, H)
            return nano.lm_head(last), nano.function_head(last)  # (1, vocab) each

        def sample_text(text_logits, step):
            return _sample_text_token(
                text_logits, gen_text, step, temperature, top_p, repetition_penalty, special_ids=special_ids,
            )

        # frame 0: agent channel is BOS
        bos_emb = embed_tokens(torch.tensor([cfg.text_bos_id], device=device))  # (1, H)
        input_embeds[:, 0] = fuse(bos_emb, audio_embeds[:, 0], embed_tokens(gen_function[:, 0]))
        text_logits, function_logits = nano_step(input_embeds[0, :1])
        if prompt_len == 0:
            gen_text[:, 0] = sample_text(text_logits, 0)
            gen_function[:, 0] = function_logits.argmax(-1)

        for t in range(1, T):
            agent_emb = embed_tokens(gen_text[:, t - 1])
            function_emb = embed_tokens(gen_function[:, t - 1])
            input_embeds[:, t] = fuse(agent_emb, audio_embeds[:, t], function_emb)
            if t < prompt_len:
                continue  # prompt region: teacher-forced, no sampling
            text_logits, function_logits = nano_step(input_embeds[0, : t + 1])
            gen_text[:, t] = sample_text(text_logits, t)
            gen_function[:, t] = function_logits.argmax(-1)

        result = {
            "tokens_text": gen_text,
            "tokens_function": gen_function,
            "prompt_len": prompt_len,
        }
        if decode_audio:
            result.update(self._decode_audio_path(gen_text, prompt_len, device, temperature))
        return result

    def _collate(self, outs: list[dict], decode_audio: bool) -> dict:
        """Pad + stack per-utterance results into batched tensors."""
        pad_id = self.config.text_pad_id

        def pad_last(x, n, value=0):
            return F.pad(x, (0, n - x.shape[1]), value=value) if x.shape[1] < n else x

        tmax = max(o["tokens_text"].shape[1] for o in outs)
        res = {
            "tokens_text": torch.cat([pad_last(o["tokens_text"], tmax, pad_id) for o in outs], dim=0),
            "tokens_function": torch.cat([pad_last(o["tokens_function"], tmax, pad_id) for o in outs], dim=0),
            "text": [self.tokenizer.decode(o["tokens_text"][0].tolist()) for o in outs],
        }
        if decode_audio and outs and "audio" in outs[0]:
            cmax = max(o["tokens_audio"].shape[1] for o in outs)
            res["tokens_audio"] = torch.cat(
                [F.pad(o["tokens_audio"], (0, 0, 0, cmax - o["tokens_audio"].shape[1])) for o in outs], dim=0,
            )
            amax = max(o["audio"].shape[1] for o in outs)
            res["audio"] = torch.cat([pad_last(o["audio"], amax) for o in outs], dim=0)
            res["audio_len"] = torch.cat([o["audio_len"] for o in outs], dim=0)
        return res

    def _decode_audio_path(self, gen_text, prompt_len, device, temperature):
        """Talker (autoregressive RVQ codes with char-vocab text conditioning) ->
        codec (waveform)."""
        talker = self.get_submodule("eartts_talker", device=device).talker
        codec = self.get_submodule("audio_codec", device=device).codec
        text_stream = gen_text[:, prompt_len:]                 # (1, T') agent text per frame

        gen = torch.Generator(device=device).manual_seed(self.config.eartts.inference_seed)
        gen_codes = talker.generate_codes(
            text_stream, self._talker_tokenizer(), speaker="Aria", temperature=temperature,
            text_eos_id=self.config.text_eos_id, text_pad_id=self.config.text_pad_id,
            speech_pad_id=self.config.eartts.codebook_size,   # speech-pad frame == all codebook_size
            generator=gen,
        )                                                       # (1, T', num_q)
        code_len = torch.tensor([gen_codes.shape[1]], device=device)
        with torch.autocast(device_type="cuda", enabled=False):
            audio, audio_len = codec.decode(gen_codes, code_len)
        return {"tokens_audio": gen_codes, "audio": audio.squeeze(1), "audio_len": audio_len}

    # -------------------------------------------------------------------
    # Postprocess
    # -------------------------------------------------------------------

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality == "text":
            token_ids = output.tolist() if output.dim() else [int(output)]
            return self.tokenizer.decode(token_ids).encode("utf-8")
        if modality == "audio":
            if output.numel() == 0:
                return b""
            return output.cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for NemotronDuplex: {modality!r}")

    # -------------------------------------------------------------------
    # Sharding — nano_llm attention/MLP are TP-eligible (Phase 10).
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        # TODO(Phase 10): Mamba layers need head-sharded state; declare TP only
        # after the components are built from the distributed linears.
        return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})

    # -------------------------------------------------------------------
    # Submodule loading
    # -------------------------------------------------------------------

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, autocast_dtype=autocast_dtype)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self, node_name: str, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == "nano_llm":
            return self._create_nano_submodule(device, autocast_dtype=autocast_dtype)
        if node_name == "conformer_encoder":
            return self._create_conformer_submodule(device, autocast_dtype=autocast_dtype)
        if node_name == "eartts_talker":
            return self._create_talker_submodule(device, autocast_dtype=autocast_dtype)
        if node_name == "audio_codec":
            return self._create_codec_submodule(device, autocast_dtype=autocast_dtype)
        return None

    def _json_config(self, local_dir: str) -> NemotronDuplexConfig:
        """Config populated from the checkpoint's ``config.json`` (cached).

        Falls back to dataclass defaults if the file is absent (e.g. when
        ``local_dir`` is an unresolved repo id).
        """
        if getattr(self, "_json_config_cache", None) is None:
            try:
                self._json_config_cache = NemotronDuplexConfig.from_pretrained(local_dir)
            except (FileNotFoundError, KeyError) as e:
                logger.warning("Could not read config.json (%s); using config defaults.", e)
                self._json_config_cache = self.config
        return self._json_config_cache

    def _build_and_load(self, module: torch.nn.Module, device: str, autocast_dtype, local_dir: str):
        """meta-build -> cast -> materialize -> load the module's checkpoint subtree.

        Each component's ``load_weights`` streams the single composite
        ``model.safetensors`` and keeps only its own subtree (verified: every
        checkpoint tensor maps to exactly one component param).
        """
        from mstar.model.loader import load_weights

        if autocast_dtype is not None:
            module = module.to(autocast_dtype)
        module.to_empty(device=device)
        load_weights(module, local_dir, device=device)
        return module.eval()

    def _create_nano_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.nemotron_duplex.components.nemotron_h import NemotronHForCausalLM
        from mstar.model.nemotron_duplex.submodules import NemotronHLLMSubmodule

        local_dir = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
        with torch.device("meta"):
            language_model = NemotronHForCausalLM(self.config.nano)
        language_model = self._build_and_load(language_model, device, autocast_dtype, local_dir)
        return NemotronHLLMSubmodule(language_model=language_model, config=self.config)

    def _create_conformer_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.nemotron_duplex.components.conformer import Perception
        from mstar.model.nemotron_duplex.components.rnnt import RnntDecoder, RnntJoint
        from mstar.model.nemotron_duplex.submodules import ConformerEncoderSubmodule

        local_dir = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
        cfg = self._json_config(local_dir)
        with torch.device("meta"):
            perception = Perception(cfg.stt)
            rnnt_dec, rnnt_joint = RnntDecoder(cfg.rnnt), RnntJoint(cfg.rnnt)
        perception = self._build_and_load(perception, device, autocast_dtype, local_dir)
        rnnt_dec = self._build_and_load(rnnt_dec, device, autocast_dtype, local_dir)
        rnnt_joint = self._build_and_load(rnnt_joint, device, autocast_dtype, local_dir)
        return ConformerEncoderSubmodule(perception, rnnt_dec, rnnt_joint, config=self.config)

    def _create_talker_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.nemotron_duplex.components.eartts_talker import EarTTSTalker
        from mstar.model.nemotron_duplex.submodules import EarTTSTalkerSubmodule

        local_dir = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
        cfg = self._json_config(local_dir)
        with torch.device("meta"):
            talker = EarTTSTalker(cfg.eartts)
        talker = self._build_and_load(talker, device, autocast_dtype, local_dir)
        from mstar.model.nemotron_duplex.components.eartts_talker import build_char_vocab, subword_to_char_ids
        tok = self._talker_tokenizer()
        cv = build_char_vocab(tok)
        s2c, _ = subword_to_char_ids(tok, cv)
        return EarTTSTalkerSubmodule(
            talker=talker, config=self.config, subword_to_char=s2c, char_pad_idx=len(cv),
        )

    def _create_codec_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.nemotron_duplex.components.audio_codec import AudioCodec
        from mstar.model.nemotron_duplex.submodules import AudioCodecDecoderSubmodule

        local_dir = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
        cfg = self._json_config(local_dir)
        with torch.device("meta"):
            codec = AudioCodec(cfg.codec)
        codec = self._build_and_load(codec, device, autocast_dtype, local_dir)
        return AudioCodecDecoderSubmodule(codec=codec, config=self.config)
