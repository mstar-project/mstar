"""NemotronDuplexModel — NVIDIA NemotronLabs VoiceChat-11B in M*.

A full-duplex speech-to-speech model with three nodes:

    conformer_encoder — 16 kHz speech → per-frame LLM embeds (+ RNN-T transcript)
    nano_llm          — Nemotron-H hybrid Mamba-2/attn/MLP backbone (9B);
                        resources nano_kv / nano_attn (4 attention layers),
                        mamba_state / mamba (27 Mamba-2 layers), nano_sampler
    eartts_talker     — Gemma3 talker → 31-codebook RVQ codes, decoded in the
                        same step by the RVQ codec → 22.05 kHz PCM; resources
                        talker_kv / talker_attn / talker_pos (two KV labels per
                        session: the text-conditioned and the null-conditioned
                        stream of classifier-free guidance), codec_state (each
                        call's codec stream state)

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

The M* serving path is the walk graphs below: three async partitions
(Encoder → LLM → Talker) joined by ``StreamingGraphEdge``s, with the nano
decode loop consuming one encoder frame per step. The user's audio is a client
stream into the Encoder, so a call can stream live over ``/generate/ws``; a
one-shot upload is a stream of one chunk. ``get_node_resources`` declares
the encoder's bounded KV (each call's attention window) and two recurrent-state
pools (its conv tails, and its audio / mel tails), the nano's paged KV cache, attention, recurrent-state pool (Mamba-2
conv + SSM slots), the Mamba-2 resource and the sampler, and the talker's paged
KV cache, attention, positions and codec stream state. Both decode steps are
accelerator graphs (the talker's with its codec decode), and so is the encoder's live
step (piecewise).

Contract methods crib from ``mstar/model/orpheus/orpheus_model.py`` (single
streaming LLM+codec) and ``mstar/model/qwen3_omni/qwen3_omni_model.py`` (omni,
multi-KV, aux sampling, async partitions).
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, PartitionDefinition
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVReqConfig,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
    apply_yaml_overrides,
)
from mstar.engine.resources.kv.bounded import BoundedKVSpec
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant
from mstar.engine.resources.recurrent import Mamba2Geometry, RecurrentStateConfig, RecurrentStateSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ClientChunk, ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, Span
from mstar.model.multimodal import PromptPart
from mstar.model.nemotron_duplex.components.audio_codec import codec_state_blocks
from mstar.model.nemotron_duplex.components.conformer import (
    encoder_kv_config,
    encoder_state_blocks,
    front_state_blocks,
)
from mstar.model.nemotron_duplex.config import (
    AUDIO_IN,
    CODEC_STATE,
    ENCODER_FRONT,
    ENCODER_KV,
    ENCODER_STATE,
    MAMBA,
    MAMBA_STATE,
    NANO_ATTN,
    NANO_KV,
    NANO_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    NemotronDuplexConfig,
    session_too_long,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import (
    CLIENT,
    ClientStreamLimits,
    Connection,
    PartitionTopology,
    StreamingGraphEdge,
)

logger = logging.getLogger(__name__)


def _sample_text_token(
    logits: torch.Tensor,          # (B, vocab)
    generated: torch.Tensor,       # (B, T) tokens so far
    step: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
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
        self.codec = model.get_submodule("eartts_talker", device=device).codec

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
            num_iter=talker.config.inference_num_iter,
            guidance_scale=talker.config.inference_guidance_scale,
            noise_scale=talker.config.inference_noise_scale,
            top_p=talker.config.inference_top_p,
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

    def __init__(self, model_path_hf: str, cache_dir: str | None = None,
                 max_session_s: float | None = None, max_prompt_tokens: int | None = None, **kwargs):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.config = NemotronDuplexConfig()
        if max_session_s:
            self.max_session_s = max_session_s
        if max_prompt_tokens:
            self.max_prompt_tokens = max_prompt_tokens
        self._tokenizer = None  # lazy — not needed in dummy mode
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @property
    def tokenizer(self):
        # The text tokenizer lives in the Spark repo's ``nano/`` folder (the base
        # repo ships only ``rnnt_tokenizer/``). 131072-token vocab; 256 single-
        # char tokens -> the talker's char vocab. Loaded lazily. AutoTokenizer
        # also reads ``nano/config.json`` when the snapshot has it, whose
        # ``auto_map`` names repo code; transformers' native Nemotron-H config
        # serves instead.
        if self._tokenizer is None:
            import os

            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer

            repo = "pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark"
            local_dir = None
            for f in ("nano/tokenizer.json", "nano/tokenizer_config.json", "nano/special_tokens_map.json"):
                local_dir = os.path.dirname(hf_hub_download(repo, f, cache_dir=self.cache_dir))
            self._tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=False)
        return self._tokenizer

    def _talker_tokenizer(self):
        """Adapter exposing the NeMo-tokenizer interface (``.tokenizer.vocab`` /
        ``.vocab``) the talker's char-vocab builder expects, over the HF tokenizer."""
        import types

        v = self.tokenizer.get_vocab()
        return types.SimpleNamespace(tokenizer=types.SimpleNamespace(vocab=v), vocab=v)

    # -------------------------------------------------------------------
    # Resources. nano_llm: the paged KV (attention layers only) and attention
    # plan, the recurrent-state pool holding every Mamba-2 layer's conv window
    # and SSM state (one slot per session) with the Mamba-2 resource planned
    # against it, and the agent-text sampler (NoPE: no position resource).
    # eartts_talker: a paged KV over its 28 Gemma3 layers with attention and
    # positions; each session holds two streams (labels ``main`` / ``uncond``)
    # and RoPE runs over the full head dim with a per-layer theta the layer
    # passes at call time. conformer_encoder: a bounded KV, each call's
    # attention window, and two recurrent-state pools, its conv tails and its
    # front end's audio / mel tails (one slot per session each). The talker also
    # holds each call's codec stream state in a recurrent-state pool, codec_state
    # (one slot per session).
    # -------------------------------------------------------------------

    # Sessions resident at once (a slot each in mamba_state, ~137 MB in fp32 for
    # Nemotron-H, and in encoder_state / encoder_front / codec_state / encoder_kv;
    # pages in nano_kv and talker_kv). A deployment raising it raises all seven
    # (see validate_config_yaml). 32 is the most one-shot calls that keep up with real
    # time on one H100 (a tick takes ~62 ms at 32, and 40+ miss the 80 ms budget).
    # Live calls paced over /generate/ws cost more per tick: 16 keep up on one
    # H200, and 32 sit at the edge (all finish, but audio can fall behind).
    DEFAULT_SESSIONS = 32           # padding rows take no slot (#258)
    # Longest session served (``model_kwargs: {max_session_s: ...}``). The
    # talker KV pool is sized so DEFAULT_SESSIONS sessions this long fit at
    # once: a full pool stalls every session, not just the one that outgrew it.
    # A page is 128 positions: 28 layers x 128 x 16 heads x 128 (the padded
    # head dim) x K,V x bf16 = 28 MiB. 160 s = 2001 frames, so 37 warm-up + 2001
    # + 1 final PAD frame fit 16 pages a stream; 32 sessions x 2 streams x 16 =
    # 1024 pages (+ the sink) = 28 GiB.
    max_session_s = 160.0
    # Most channels an upload may have; the model takes 16 kHz mono speech.
    max_upload_channels = 2
    # Longest system prompt (``model_kwargs: {max_prompt_tokens: ...}``). The
    # prompt prefills in one step batched with every live session's frame, so
    # it stalls them all for its prefill time
    # TODO(prefill): drop this cap once chunked prefill keeps the prompt out of the frame batch.
    max_prompt_tokens = 512
    TALKER_WARMUP_POSITIONS = 37      # the speaker warm-up each talker stream prefills
    # One encoder step per stream item (an uploaded file or an append), however
    # many: the loop ends with the stream, never at this limit (the largest the
    # Rust runtime's u32 counter takes with room to spare).
    ENCODE_LOOP_ITERS = 2**31 - 1
    # Headroom over max_session_frames for the loops and the conductor's token
    # cap; a cap reached mid-stream hangs the session instead of ending it.
    LOOP_ITERS_MARGIN = 16

    @property
    def max_session_samples(self) -> int:
        return int(self.max_session_s * self.config.stt.sample_rate)

    @property
    def max_session_frames(self) -> int:
        return self.config.stt.num_frames(self.max_session_samples)

    def talker_kv_pages(self, page_size: int, sessions: int | None = None) -> int:
        """Pages for ``sessions`` (DEFAULT_SESSIONS) sessions of max_session_frames, both CFG streams."""
        return self._session_kv_pages(self.TALKER_WARMUP_POSITIONS, 2, page_size, sessions)

    def nano_kv_pages(self, page_size: int, sessions: int | None = None) -> int:
        """Pages for ``sessions`` (DEFAULT_SESSIONS) sessions of the longest prompt and max_session_frames."""
        return self._session_kv_pages(self.max_prompt_tokens, 1, page_size, sessions)

    def _session_kv_pages(self, prefix: int, streams: int, page_size: int, sessions: int | None) -> int:
        sessions = self.DEFAULT_SESSIONS if sessions is None else sessions
        positions = prefix + self.max_session_frames + 1
        # + 1 for the pool's sink page
        return sessions * streams * -(-positions // page_size) + 1

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse a session cap that a per-session resource cannot hold.

        A session holds a slot in each recurrent pool (``mamba_state``,
        ``encoder_state``, ``encoder_front``, ``codec_state``) and in
        ``encoder_kv``, and its nano and talker pages from its first step to its
        last, so a session past any of them waits on admit until
        another ends, and its frames with it. So ``max_concurrent_requests``
        sessions must fit each; the conductor queues the rest.
        """
        cap = config.get("max_concurrent_requests")
        if cap is None:
            logger.warning(
                "%s sets no max_concurrent_requests: a session past the per-session "
                "resources waits on admit until another ends", config_path,
            )
            return
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError(f"{config_path}: max_concurrent_requests must be a positive int; got {cap!r}")
        specs = self.get_node_resources()
        apply_yaml_overrides(specs, config)
        for spec in specs:
            if isinstance(spec, RecurrentStateSpec):
                held, knob = spec.config.usable_slots, "max_slots"
            elif isinstance(spec, BoundedKVSpec):
                held, knob = spec.config.max_slots, "max_slots"
            elif spec.resource_key in (TALKER_KV, NANO_KV):
                kv = spec.config
                pages = self.talker_kv_pages if spec.resource_key == TALKER_KV else self.nano_kv_pages
                per_session = pages(kv.page_size, sessions=1) - 1
                held, knob = (kv.max_num_pages - 1) // per_session, "max_num_pages"
            else:
                continue
            if held < cap:
                raise ValueError(
                    f"{config_path}: {cap} concurrent sessions (max_concurrent_requests) need as "
                    f"many in resources.{spec.resource_key}, which holds {held}; raise its {knob} "
                    "or lower max_concurrent_requests"
                )

    def get_max_output_tokens(self, **model_kwargs) -> int:
        # One text token per audio frame, so the audio fixes the length, within
        # the session cap (audio_in's max_items, in samples). A client cap would
        # cut the text loop short of its stream's end and hang the session.
        del model_kwargs
        return self.max_session_frames + self.LOOP_ITERS_MARGIN

    def get_node_resources(self) -> list[NodeResourceSpec]:
        nano = self.config.nano
        eartts = self.config.eartts
        kv_config = PagedKVConfig(
            # Only the ``*`` (attention) layers hold a KV cache; the Mamba and
            # MLP layers do not. The backbone maps global layer -> dense
            # attention index (``NemotronHLLM._attn_cache_idx``).
            num_layers=nano.num_attention_layers,
            num_kv_heads=nano.num_key_value_heads,
            head_dim=nano.head_dim,
            max_seq_len=nano.max_position_embeddings,
            num_qo_heads=nano.num_attention_heads,
            # a session holds its prompt and every frame for life (no eviction)
            max_num_pages=self.nano_kv_pages(PagedKVConfig.page_size),
        )
        return [
            RecurrentStateSpec(
                resource_key=ENCODER_STATE, nodes={"conformer_encoder"},
                config=RecurrentStateConfig(
                    num_layers=self.config.stt.num_layers,
                    blocks=encoder_state_blocks(self.config.stt),       # in the engine's compute dtype
                    max_slots=self.DEFAULT_SESSIONS + 1,
                ),
            ),
            RecurrentStateSpec(
                resource_key=ENCODER_FRONT, nodes={"conformer_encoder"},
                config=RecurrentStateConfig(
                    num_layers=1, blocks=front_state_blocks(self.config.stt),
                    max_slots=self.DEFAULT_SESSIONS + 1,
                ),
            ),
            # no sink slot: padding rows read and write nothing in it
            BoundedKVSpec(
                resource_key=ENCODER_KV, nodes={"conformer_encoder"},
                config=encoder_kv_config(self.config.stt, max_slots=self.DEFAULT_SESSIONS),
            ),
            KVSpec(resource_key=NANO_KV, nodes={"nano_llm"}, config=kv_config),
            AttentionSpec(
                resource_key=NANO_ATTN, nodes={"nano_llm"},
                config=AttentionConfig(kv_cache=NANO_KV),
            ),
            RecurrentStateSpec(
                resource_key=MAMBA_STATE, nodes={"nano_llm"},
                config=RecurrentStateConfig(
                    num_layers=nano.num_mamba_layers,
                    blocks=Mamba2Geometry(
                        num_heads=nano.mamba_num_heads,
                        head_dim=nano.mamba_head_dim,
                        state_size=nano.ssm_state_size,
                        n_groups=nano.n_groups,
                        conv_kernel_size=nano.conv_kernel,
                    ).to_blocks(),
                    # + 1 for the pool's sink slot, so DEFAULT_SESSIONS sessions fit
                    max_slots=self.DEFAULT_SESSIONS + 1,
                ),
            ),
            LinearAttnSpec(
                resource_key=MAMBA, nodes={"nano_llm"},
                config=LinearAttnConfig(
                    recurrent_state=MAMBA_STATE,
                    variant=LinearAttnVariant.MAMBA2,
                    time_step_limit=nano.time_step_limit,
                ),
            ),
            SamplerSpec(
                resource_key=NANO_SAMPLER, nodes={"nano_llm"},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=True,
            ),
            RecurrentStateSpec(
                resource_key=CODEC_STATE, nodes={"eartts_talker"},
                config=RecurrentStateConfig(
                    num_layers=1, blocks=codec_state_blocks(self.config.codec),
                    max_slots=self.DEFAULT_SESSIONS + 1,
                ),
            ),
            KVSpec(
                resource_key=TALKER_KV, nodes={"eartts_talker"},
                config=PagedKVConfig(
                    num_layers=eartts.num_hidden_layers,
                    num_kv_heads=eartts.num_key_value_heads,
                    head_dim=eartts.kv_head_dim,    # 72 zero-padded to a FlashInfer head dim
                    # the sliding window plus the speaker warm-up, rounded up
                    max_seq_len=eartts.sliding_window + 512,
                    num_qo_heads=eartts.num_attention_heads,
                    max_num_pages=self.talker_kv_pages(PagedKVConfig.page_size),
                ),
            ),
            AttentionSpec(
                resource_key=TALKER_ATTN, nodes={"eartts_talker"},
                config=AttentionConfig(kv_cache=TALKER_KV),
            ),
            PositionSpec(
                resource_key=TALKER_POS, nodes={"eartts_talker"},
                config=PositionConfig(
                    kv_cache=TALKER_KV,
                    rotary_dim=eartts.head_dim,     # Gemma3: RoPE over the full head
                    interleave=False,
                    rope_theta=eartts.rope_theta_local,   # default; full-attention layers pass 1e6
                ),
            ),
        ]

    def get_request_resource_configs(
        self,
        partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        keys = ["temperature", "top_p", "repetition_penalty", "ignore_eos"]
        return {
            NANO_SAMPLER: SamplingReqConfig(
                **{k: model_kwargs.get(k, getattr(self.config, k)) for k in keys}
            ),
            # every talker session reads both of its streams
            TALKER_KV: KVReqConfig(needed_labels=["main", "uncond"]),
        }

    # -------------------------------------------------------------------
    # Graph walks. Text path is live; audio walks are Phase 6 (see design note).
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        """Full-duplex S2S pipeline as three async partitions (frame-synchronous):

            client  --[audio_in,    FixedChunk(1)]--> Encoder
            Encoder --[audio_frame, FixedChunk(1)]--> LLM
            LLM     --[new_token,   FixedChunk(1)]--> Talker --[audio_chunk]--> client

        The LLM (nano) is a frame-driven decode loop: each iteration fuses one
        streamed audio frame with the fed-back previous agent-text/function tokens
        (AddFusion), samples the next agent text (main) + function (aux) token, emits
        the text to the client, streams it to the talker, and feeds both back. An
        optional system-prompt ``prefill_text`` primes the nano cache first.
        """
        # Encoder: one step per chunk of the client's audio stream, batched over
        # the calls that have one; emits the frames that chunk made final,
        # streamed one frame per LLM step. Ends with the client's stream. It is
        # never a speculation target (its input is the client's, not its own
        # output); async scheduling lets another node's step overlap it.
        encode = Loop(
            name="encode_loop",
            section=GraphNode(
                name="conformer_encoder",
                input_names=[AUDIO_IN],
                consumes_stream=True,
                outputs=[StreamingGraphEdge(next_node="nano_llm", name="audio_frame", target_partition="LLM")],
            ),
            max_iters=self.ENCODE_LOOP_ITERS,
            outputs=[],
        )
        # LLM: the frame-synchronous decode loop. A system prompt is not a
        # separate walk: the reference feeds the prompt tokens as leading
        # frames of the same stream, so here they are the leading tokens of
        # the session's first step (prompt + first audio frame, sampled at the
        # last position only). ``text_inputs`` rides the loop as a fed-back
        # token: the request seeds it with the prompt ids (or a -1 sentinel
        # without a prompt) and the node hands back the sentinel once it has
        # consumed the prompt, so a re-injected copy can never win over it.
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="nano_llm",
                input_names=["audio_frame", "prev_text", "prev_func", "text_inputs"],
                consumes_stream=True,
                outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token",
                              output_modality="text", conductor_new_token=True),
                    GraphEdge(next_node="nano_llm", name="prev_text"),   # fed-back agent text
                    GraphEdge(next_node="nano_llm", name="prev_func"),   # fed-back function
                    GraphEdge(next_node="nano_llm", name="text_inputs"), # prompt, consumed once
                    StreamingGraphEdge(next_node="eartts_talker", name="new_token",
                                       target_partition="Talker"),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        # Talker: each streamed agent token -> RVQ codes -> that frame's PCM, to
        # the client. The codec decodes in the talker's step (no stream hop, no
        # step of its own), against each call's stream state in codec_state.
        # No speculative scheduling here. The talker keeps the previous frame's
        # codes in per-request state written by postprocess, so a step built
        # while the previous one is still running would read the wrong frame.
        talker_decode = Loop(
            name="talker_decode_loop",
            section=GraphNode(
                name="eartts_talker",
                input_names=["new_token"],
                consumes_stream=True,
                enable_async_scheduling=False,
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk", output_modality="audio")],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return dict(encode=encode, decode=decode, talker_decode=talker_decode)

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name="Encoder", graph_walks={"encode"}, initial_walk="encode",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name="LLM", graph_walks={"decode"}, initial_walk="decode",
                producer_partitions=["Encoder"],
            ),
            PartitionDefinition(
                name="Talker", graph_walks={"talker_decode"}, initial_walk="talker_decode",
                producer_partitions=["LLM"],
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(partitions=["Encoder", "LLM", "Talker"], connections=[
            # the user's audio, one item per uploaded file or append, at most a
            # session's samples: past them the talker KV and the decode loops run out
            Connection(from_partition=CLIENT, to_partition="Encoder", edge_name=AUDIO_IN,
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1),
                       limits=ClientStreamLimits(max_items=self.max_session_samples,
                                                 item_size=torch.numel, unit="samples")),
            # one audio frame per nano decode step. continue_after_done=False:
            # the duplex nano text loop is 1:1 with audio frames, so when the
            # encoder's frames are exhausted the decode loop must terminate
            # (the final empty chunk carries is_final). continue_after_done is
            # only for the LLM→Talker leg, where the talker keeps going past
            # the last text token.
            Connection(from_partition="Encoder", to_partition="LLM", edge_name="audio_frame",
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1, continue_after_done=False)),
            # one agent token per talker step. continue_after_done=False: the
            # duplex talker emits exactly one RVQ frame per agent-text token
            # (T tokens → T frames), so when the LLM's token stream ends the
            # talker loop terminates via its final chunk (unlike qwen3_omni,
            # whose talker outlives the thinker and self-stops on codec EOS).
            Connection(from_partition="LLM", to_partition="Talker", edge_name="new_token",
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1, continue_after_done=False)),
        ])

    # -------------------------------------------------------------------
    # Prompt processing
    # -------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        # Duplex request: an optional system-prompt text that primes the nano
        # cache, and the user's audio, which streams into the Encoder. An
        # uploaded clip is the stream's first chunk; a live call appends more.
        out: NameToTensorList = {}
        # The agent always speaks: text-only would need the LLM to stop feeding the talker stream
        if "audio" not in output_modalities or set(output_modalities) - {"text", "audio"}:
            raise ValueError(
                f"output modalities {sorted(output_modalities)} are not supported; use audio or text,audio"
            )
        if prompt:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(torch.long)
            # TODO(prefill): stopgap for prompt decode stalls; can be removed once we get prefill
            if ids.numel() > self.max_prompt_tokens:
                raise ValueError(
                    f"system prompt is {ids.numel()} tokens; this server "
                    f"takes prompts up to {self.max_prompt_tokens} tokens"
                )
            out["text_inputs"] = [ids]
        # The data worker loads audio files under ``audio_inputs`` (f"{modality}_inputs").
        wav = (tensors or {}).get("audio_inputs")
        if wav:
            # an upload is one stretch of the call: it may be scaled, as one
            wav = [self._checked_audio(w) for w in wav]
            peak = max((w.abs().max() for w in wav if w.numel()), default=0.0)
            if peak > 1.0:      # float files can exceed full scale, and loud input silences the agent
                wav = [w / peak for w in wav]
            # The session cap is audio_in's max_items, which the data worker
            # charges these samples to as the stream's first chunk.
            out[AUDIO_IN] = wav
        # Seed the decode loop's iteration-0 fed-back tokens. The frame-
        # synchronous nano step lists prev_text / prev_func as inputs, so
        # without these the readiness gate never fires on the first frame (no
        # loop-back exists yet) and the decode loop hangs. Later iterations
        # overwrite them via the loop-back edges. The agent channel starts at
        # BOS only at true global frame 0; after a system prompt the reference
        # holds it at PAD across the prompt, so the first audio frame's prev is
        # the last prompt PAD (this seed is what the first frame sees, ahead of
        # the prefill node's own persisted PAD).
        first_prev = self.config.text_pad_id if prompt else self.config.text_bos_id
        out["prev_text"] = [torch.tensor([first_prev], dtype=torch.long)]
        out["prev_func"] = [torch.tensor([self.config.text_pad_id], dtype=torch.long)]
        # The prompt rides the decode loop as a fed-back token; -1 = no prompt.
        # ``has_prompt`` lets the routing know before any tensor is readable.
        if prompt:
            out["has_prompt"] = [torch.tensor([1], dtype=torch.long)]
        else:
            out["text_inputs"] = [torch.tensor([-1], dtype=torch.long)]
        # a session's first frame writes the prompt along with its own row, and the talker's first frame writes
        # its speaker warm-up
        prompt_len = ids.numel() if prompt else 0
        nano = LabelLayout(
            spans=(Span("ids", "decode", prompt_len, prompt_len, "text_inputs"),) if prompt else (),
            growth=(("decode_loop", 1),),
        )
        warmup = self.TALKER_WARMUP_POSITIONS
        talker = LabelLayout(
            spans=(Span("rows", "talker_decode", warmup, warmup, "audio_prompt_latents"),),
            growth=(("talker_decode_loop", 1),),
        )
        return ProcessPromptOutput(out, {"kv_layout": {
            NANO_KV: {"main": nano}, TALKER_KV: {"main": talker, "uncond": talker},
        }})

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        talker = KVLabel(growth=(("talker_decode", "talker_decode_loop"),))
        return {
            NANO_KV: {"main": KVLabel(growth=(("decode", "decode_loop"),))},
            TALKER_KV: {"main": talker, "uncond": talker},
        }

    def process_stream_chunk(self, chunk: ClientChunk) -> NameToTensorList:
        # Each appended file is one stream item, in order; audio_in's max_items
        # caps the call's samples as its audio grows. A piece of a stream can't
        # be resampled or scaled alone without a seam at every append. The one
        # client stream is audio_in, so a file's named stream changes nothing.
        if chunk.text:
            raise ValueError("this model takes audio appends only; got text")
        for key, items in chunk.tensors.items():
            if items and key != "audio_inputs":
                raise ValueError(f"this model takes audio appends only; got {key.removesuffix('_inputs')}")
        rate = self.config.stt.sample_rate
        for meta in chunk.input_metadata.get("audio_inputs") or []:
            if meta.get("source_sample_rate", rate) != rate:
                raise ValueError(f"appends must be {rate} Hz audio; got {meta['source_sample_rate']} Hz")
        return {AUDIO_IN: [self._checked_audio(w).clamp(-1.0, 1.0) for w in chunk.tensors.get("audio_inputs") or []]}

    @staticmethod
    def _checked_audio(wav: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(wav).all():
            raise ValueError("audio contains NaN or inf samples")
        return wav

    def load_audio(self, filepath: str, device: str):
        """Decode an uploaded audio file to 16 kHz mono via soundfile (the base uses
        torchcodec, whose native libs aren't always available). Returns the same
        ``TensorAndMetadata`` shape the data worker expects."""
        import soundfile as sf

        from mstar.api_server.media_io import AudioTooLongError, check_resample_rate, read_soundfile_capped
        from mstar.model.base import TensorAndMetadata

        try:
            # refused before the decode, which reads every channel of each block
            info = sf.info(filepath)
            check_resample_rate(info.samplerate, 16000)
            if info.channels > self.max_upload_channels:
                raise ValueError(
                    f"audio must have at most {self.max_upload_channels} channels; got {info.channels}"
                )
            data, sr = read_soundfile_capped(filepath, self.max_session_s)   # mono
        except AudioTooLongError:
            raise session_too_long(self.max_session_s) from None
        except sf.SoundFileError as exc:   # garbage or truncated upload
            raise ValueError(f"could not decode the audio file: {exc}") from exc
        audio = torch.from_numpy(data)
        if sr != 16000:
            import torchaudio.functional as AF
            audio = AF.resample(audio, sr, 16000)
        return TensorAndMetadata(data=audio, metadata=dict(sample_rate=16000, num_channels=1,
                                                           source_sample_rate=sr))

    # -------------------------------------------------------------------
    # Forward-pass-args state machines — one per async partition
    # (Encoder -> LLM -> Talker), mirroring qwen3_omni.
    # -------------------------------------------------------------------

    @staticmethod
    def _has_prompt(input_signals, model_kwargs=None) -> bool:
        """Whether the request seeded ``text_inputs`` with prompt ids rather
        than the no-prompt sentinel: ``process_prompt`` adds a ``has_prompt``
        signal alongside (the tensors themselves are not readable here)."""
        return bool(input_signals.get("has_prompt"))

    def _meta(self, imod, omod, walk, is_prefill):
        return CurrentForwardConductorMetadata(
            input_modalities=imod, output_modalities=omod, graph_walk=walk, is_prefill=is_prefill,
        )

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        if partition_name == "Encoder":                    # self-triggered by the client's audio
            return ForwardPassArgs(
                full_metadata=self._meta(input_modalities, output_modalities, "encode", True),
                inputs=[], unpersist_tensors=[], step_metadata={},
            )
        if partition_name == "LLM":
            # Straight into the frame-synchronous decode loop, self-triggered
            # by the audio stream. Seed iteration 0's fed-back tokens (agent
            # BOS / PAD, function PAD) and the prompt ids (a -1 sentinel when
            # there is no prompt) so the node is ready before any loop-back
            # exists. ``prompt_pending`` tells the worker that this request's
            # first step is prompt + frame, which no fixed-shape capture fits.
            has_prompt = self._has_prompt(input_signals, model_kwargs)
            edges = []
            for name in ("prev_text", "prev_func", "text_inputs"):
                e = GraphEdge(next_node="nano_llm", name=name)
                e.tensor_info = input_signals.get(name, [])
                edges.append(e)
            return ForwardPassArgs(
                full_metadata=self._meta(input_modalities, output_modalities, "decode", has_prompt),
                inputs=edges, unpersist_tensors=sum([e.tensor_info for e in edges], start=[]),
                step_metadata={"prompt_pending": has_prompt},
            )
        if partition_name == "Talker":
            return ForwardPassArgs(
                full_metadata=self._meta(input_modalities, output_modalities, "talker_decode", False),
                inputs=[], unpersist_tensors=[], request_done=False,
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections=None,
    ) -> ForwardPassArgs:
        m = partition_metadata

        # All three partitions are stream-terminated: each ends when its node
        # consumes the final chunk of its upstream stream, which the worker
        # reports and the conductor turns into the partition's request_done
        # (and producer_done for the next partition). Nothing here guesses the
        # end from the connection counters: ``consumed_count`` counts chunks
        # popped for execution, so a counter-based end can fire while the last
        # step is still running and its output would arrive for a finished
        # request (observed as sessions losing their final audio frame).
        if partition_name == "LLM":
            m.is_prefill = False                            # the prompt rode the first step
            edges = []
            for name in ("prev_text", "prev_func"):        # fed-back tokens for AddFusion
                e = GraphEdge(next_node="nano_llm", name=name)
                e.tensor_info = persist_signals.get(name, [])
                edges.append(e)
            return ForwardPassArgs(
                full_metadata=m, inputs=edges,
                unpersist_tensors=sum([e.tensor_info for e in edges], start=[]),
                request_done=False, step_metadata={"prompt_pending": False},
            )

        if partition_name in ("Encoder", "Talker"):  # self-triggered by the upstream stream
            return ForwardPassArgs(full_metadata=m, inputs=[], unpersist_tensors=[], request_done=False)
        raise ValueError(f"Unknown partition: {partition_name!r}")

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.eartts.sample_rate

    # -------------------------------------------------------------------
    # Online (streaming) duplex inference — Phase B/C.
    # -------------------------------------------------------------------

    def prepare_streaming(self, device: str = "cuda"):
        """Warm (build + load) all three submodules so streaming starts promptly."""
        for node in ("conformer_encoder", "nano_llm", "eartts_talker"):
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
        temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.0,
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
        temperature: float = 0.7,
        top_p: float = 0.9,
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
        codec = self.get_submodule("eartts_talker", device=device).codec
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
            # PAD fills every silent frame; it and BOS/EOS are not reply text
            return self.tokenizer.decode(token_ids, skip_special_tokens=True).encode("utf-8")
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
        return None

    def _json_config(self, local_dir: str) -> NemotronDuplexConfig:
        """Config populated from the checkpoint's ``config.json`` (cached).

        Falls back to dataclass defaults if the file is absent (e.g. when
        ``local_dir`` is an unresolved repo id).
        """
        if getattr(self, "_json_config_cache", None) is None:
            try:
                cfg = NemotronDuplexConfig.from_pretrained(local_dir)
            except (FileNotFoundError, KeyError) as e:
                logger.warning("Could not read config.json (%s); using config defaults.", e)
                cfg = self.config
            self._refuse_resources_sized_otherwise(cfg)
            self._json_config_cache = cfg
        return self._json_config_cache

    def _refuse_resources_sized_otherwise(self, cfg: NemotronDuplexConfig) -> None:
        """The resources are declared from the dataclass defaults, before the
        checkpoint is read: refuse a config.json that sizes one otherwise,
        rather than run its modules against pools of another shape."""
        loaded = copy.copy(self)
        loaded.config = cfg
        for declared, needed in zip(self.get_node_resources(), loaded.get_node_resources(), strict=True):
            if declared != needed:
                raise ValueError(
                    f"the checkpoint's config.json sizes resources.{declared.resource_key} as {needed}, "
                    f"not as declared ({declared})"
                )

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
        from mstar.model.nemotron_duplex.components.audio_codec import AudioCodec
        from mstar.model.nemotron_duplex.components.eartts_talker import EarTTSTalker
        from mstar.model.nemotron_duplex.submodules import EarTTSTalkerSubmodule

        local_dir = _resolve_local_hf_snapshot(self.model_path_hf, cache_dir=self.cache_dir)
        cfg = self._json_config(local_dir)
        with torch.device("meta"):
            talker = EarTTSTalker(cfg.eartts)
            codec = AudioCodec(cfg.codec)
        talker = self._build_and_load(talker, device, autocast_dtype, local_dir)
        # the codec decodes in the talker's step, in the engine's dtype like the
        # talker (it turns autocast off itself)
        codec = self._build_and_load(codec, device, autocast_dtype, local_dir)
        from mstar.model.nemotron_duplex.components.eartts_talker import build_char_vocab, subword_to_char_ids
        tok = self._talker_tokenizer()
        cv = build_char_vocab(tok)
        s2c, _ = subword_to_char_ids(tok, cv)
        # checkpoint config, so the talker samples with its inference_* knobs
        submodule = EarTTSTalkerSubmodule(
            talker=talker, config=cfg, codec=codec, subword_to_char=s2c, char_pad_idx=len(cv),
        )
        assert submodule.warmup_len == self.TALKER_WARMUP_POSITIONS, (
            f"the checkpoint's speaker warm-up is {submodule.warmup_len} positions, "
            f"where the talker's layout counts {self.TALKER_WARMUP_POSITIONS}"
        )
        return submodule
