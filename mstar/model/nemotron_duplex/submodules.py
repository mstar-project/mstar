"""NodeSubmodules for NVIDIA NemotronLabs VoiceChat-11B (full duplex).

Nodes, and the resources each declares (``NemotronDuplexModel.get_node_resources``):

    conformer_encoder  (none)                    16 kHz speech -> per-frame LLM embeds (+ RNN-T)
    nano_llm           nano_kv, nano_attn,        Nemotron-H hybrid Mamba-2 / attention / MLP (9B):
                       mamba_state, mamba,        paged KV for the 4 attention layers, recurrent-pool
                       nano_sampler               slots for the 27 Mamba-2 layers, the text sampler
    eartts_talker      talker_kv, talker_attn,    Gemma3 talker -> 31 RVQ codes per frame; two KV
                       talker_pos                 labels per session (cond / uncond) in one plan
    audio_codec        (none)                    RVQ codes -> 22.05 kHz PCM, per-request left context

Both autoregressive nodes keep their whole state in engine resources and capture
their decode step as a CUDA graph; the encoder and the codec run eager.
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
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent import RecurrentStep
from mstar.model.nemotron_duplex.config import (
    MAMBA,
    MAMBA_STATE,
    NANO_ATTN,
    NANO_KV,
    NANO_SAMPLER,
    TALKER_ATTN,
    TALKER_CFG_LABEL,
    TALKER_KV,
    TALKER_POS,
    NemotronDuplexConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

# How ``prepare_inputs`` classified a request's step; ``preprocess`` fuses accordingly.
_MODE_FRAME = "frame"      # one streamed audio frame + fed-back prev_text / prev_func
_MODE_PROMPT_FRAME = "prompt_frame"   # a session's first step: the system prompt's tokens, then its first frame
NO_PROMPT = -1             # the ``text_inputs`` loop-back value once the prompt has been consumed (or never existed)
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

    # No torch.compile: the per-layer resource cursors are compile-disabled
    # and the decode step is a CUDA graph replay, which is where the time goes.
    disable_torch_compile = True

    # Decode batch sizes captured as CUDA graphs: one row per live session per
    # 80 ms tick. The recurrent pool holds a slot per row (padding rows take
    # one transiently), so the model's pool sizing covers the largest bucket.
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    DECODE_KEY = "decode"            # the steady-state capture's bucket key (cg_key_info)

    def __init__(self, language_model: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.language_model = language_model
        self.embeddings = language_model.embeddings
        self.lm_head = language_model.lm_head
        self.config = config

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        """Capture the frame-synchronous decode step.

        ``prepare_inputs`` is host-only and ``preprocess`` (embedding + AddFusion)
        runs eagerly before the graph on both capture and replay; the captured
        region is ``forward_batched``: the 56-layer backbone over the paged KV
        and the recurrent pool, the two heads and the sampler. The dummy row is
        one fused frame: a zero audio frame plus the BOS / PAD carry-ins.
        """
        cfg = self.config
        frame = ARNodeInputs(
            input_seq_len=1,
            tensor_inputs={
                "prev_text": torch.tensor([cfg.text_bos_id], dtype=torch.long),
                "prev_func": torch.tensor([cfg.text_pad_id], dtype=torch.long),
                "audio_frame": torch.zeros(1, cfg.nano.hidden_size, device=device),
            },
            kwargs={"mode": _MODE_FRAME},
        )
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=frame,
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                additional_key_info=self.DECODE_KEY,
                compile=False,
            ),
        ]

    # -- resources -------------------------------------------------------

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        steps = {
            NANO_KV: KVStep(),
            NANO_ATTN: AttentionStep(causal=True),
            # one slot per request in the recurrent pool; the Mamba-2 resource
            # picks the single-step kernels when every row spans one token
            MAMBA_STATE: RecurrentStep(),
            MAMBA: LinearAttnStep(),
        }
        if graph_walk == "decode":
            # The prompt region is not sampled (held at PAD), so only decode
            # steps the sampler; the reference's repetition penalty likewise
            # only ever sees generated tokens.
            steps[NANO_SAMPLER] = SamplerStep(apply_penalty=True)
        # a session's prompt + first-frame step has more than one token per
        # request, which no fixed-shape decode capture fits (same answer as
        # cg_key_info, which the engine asks before the inputs exist)
        first = any(inp.kwargs.get("mode") == _MODE_PROMPT_FRAME for inp in inputs)
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            cg_key_info=None if (first or graph_walk != "decode") else self.DECODE_KEY,
            steps=steps,
        )

    def cg_key_info(self, graph_walk: str, per_request_info: Mapping[str, Any]) -> str | None:
        """The decode capture serves a batch unless a request in it still has
        its system prompt pending (its first step is prompt + frame, so it
        runs eager); the routing marks that in ``step_metadata``."""
        if graph_walk != "decode":
            return None
        for info in per_request_info.values():
            if (getattr(info, "step_metadata", None) or {}).get("prompt_pending"):
                return None
        return self.DECODE_KEY

    # -- inputs ----------------------------------------------------------

    @staticmethod
    def _prompt_ids(inputs: NameToTensorList) -> torch.Tensor | None:
        """The pending system-prompt ids from the ``text_inputs`` loop-back, or
        None when it carries the NO_PROMPT sentinel (or is absent)."""
        if not inputs.get("text_inputs"):
            return None
        ids = inputs["text_inputs"][0].reshape(-1).to(torch.long)
        ids = ids[ids >= 0]
        return ids if ids.numel() > 0 else None

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
            # The system prompt rides the loop as ``text_inputs``: real ids on
            # the session's first step, the NO_PROMPT sentinel afterwards.
            prompt = self._prompt_ids(inputs)
            if prompt is not None:
                tensors["prompt_ids"] = prompt
                return ARNodeInputs(input_seq_len=int(prompt.numel()) + 1, tensor_inputs=tensors,
                                    kwargs={"mode": _MODE_PROMPT_FRAME})
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
        if mode == _MODE_PROMPT_FRAME:
            # the reference's ``_prime_prompt``: prompt tokens on the audio
            # channel, the agent channel BOS then PAD, the function channel
            # PAD, all ahead of the first frame in the same stream
            ids = t["prompt_ids"].to(device)
            agent = torch.full_like(ids, cfg.text_pad_id)
            agent[0] = cfg.text_bos_id
            prompt = self.embeddings(agent) * cfg.agent_text_weight + self.embeddings(ids) * cfg.user_audio_weight
            if cfg.use_function_head:
                prompt = prompt + func_pad * cfg.function_weight
            fused = torch.cat([prompt, fused])
        return fused

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        device = self.embeddings.weight.device
        seq_lens = [inp.input_seq_len for inp in inputs]
        if all(inp.kwargs.get("mode") == _MODE_FRAME for inp in inputs):
            # the steady state: one frame per session, fused for the whole
            # batch in a handful of launches instead of a handful per session
            # (at 64 sessions the per-request loop cost ~28 ms of host time
            # per step, more than the forward itself)
            return {"input_embeds": self._fuse_frames([inp.tensor_inputs for inp in inputs], device),
                    "seq_lens": seq_lens}
        return {
            "input_embeds": torch.cat([self._fuse(inp, device) for inp in inputs], dim=0),
            "seq_lens": seq_lens,
        }

    def _fuse_frames(self, tensors: list[NameToTensorList | dict], device: torch.device) -> torch.Tensor:
        """``_fuse`` for a batch of frame steps at once: ``[N, H]``, row ``i`` equal
        to ``_fuse(inputs[i])`` (same weights, same terms)."""
        cfg = self.config
        # rows arrive on either device (a session's first prev_text is built on
        # the host, later ones are the fed-back GPU token), so move each first
        prev_text = torch.cat([t["prev_text"].reshape(-1).to(device) for t in tensors])          # [N]
        fused = self.embeddings(prev_text) * cfg.agent_text_weight                                # [N, H]
        frames = [t.get("audio_frame") for t in tensors]
        present = [f for f in frames if f is not None]
        if present:
            # a row past the end of its audio (the stream closed, the loop
            # still runs on the fed-back text) has no frame: its audio term is 0
            zero = torch.zeros(1, fused.shape[1], device=device, dtype=fused.dtype)
            audio = torch.cat([zero if f is None else f.reshape(1, -1).to(device=device, dtype=fused.dtype)
                               for f in frames])                                                  # [N, H]
            fused = fused + audio * cfg.user_audio_weight
        if cfg.use_function_head:
            prev_func = torch.cat([t["prev_func"].reshape(-1).to(device) for t in tensors])
            fused = fused + self.embeddings(prev_func) * cfg.function_weight
        return fused

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
        """One fused forward over the packed batch: attention over the planned
        paged KV, Mamba-2 over each request's slot in the recurrent pool."""
        cfg = self.config
        rids = list(engine_inputs.request_ids)
        hidden = self.language_model(input_embeds, label="main")
        if graph_walk != "decode":
            # Prompt priming: no sampling, the region is held at PAD (reference
            # ``_prime_prompt``), so the first audio frame follows the last PAD.
            pad = torch.full((1,), cfg.text_pad_id, dtype=torch.long, device=hidden.device)
            return {rid: {"prev_text": [pad], "prev_func": [pad]} for rid in rids}

        if all(n == 1 for n in seq_lens):
            last = hidden                                   # decode: one row per request (captured shape)
        else:
            ends = list(itertools.accumulate(seq_lens))
            last = hidden.index_select(0, torch.tensor([e - 1 for e in ends], device=hidden.device))
        new_token = engine_inputs.resources[NANO_SAMPLER].sample(rids, logits=self.lm_head(last))
        new_func = None
        if cfg.use_function_head:
            # Tool-call channel: plain argmax, as in the reference.
            new_func = self.language_model.function_head(last).argmax(dim=-1)
        # the prompt (if any) has been consumed by this step: the loop-back
        # carries the sentinel from here on
        no_prompt = torch.full((1,), NO_PROMPT, dtype=torch.long, device=hidden.device)
        out: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(rids):
            o: NameToTensorList = {"new_token": [new_token[i : i + 1]], "text_inputs": [no_prompt]}
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
    """Gemma3 talker: one agent text token -> ``num_quantizers`` RVQ codes per
    session per frame, all live sessions in one backbone pass.

    The talker's KV lives in the engine's paged KV pool under two labels per
    request: ``main`` (text-conditioned stream) and ``uncond`` (null-conditioned
    stream for classifier-free guidance), combined into one attention plan
    whose rows are ``[cond rows; uncond rows]`` -- the order ``mog_head.infer``
    expects. A session's first step prefills the 37 speaker warm-up positions
    together with its first frame (38 tokens per label); every later step is
    one token per label. The MoG / MaskGIT sampling is internal (not the
    engine's categorical sampler); its noise is drawn per row, with the row's
    own seeded generator, in ``preprocess`` -- so a session's speech does not
    depend on its batch mates and the whole step is a CUDA graph replay for
    all-decode batches.
    """

    disable_torch_compile = True

    PREV_CODES_KEY = "talker_prev_codes"
    GEN_KEY = "talker_gen"
    DECODE_KEY = "decode"            # the steady-state capture's bucket key (cg_key_info)
    SPEAKER = "Aria"
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self, talker: nn.Module, config: NemotronDuplexConfig,
                 subword_to_char: dict | None = None, char_pad_idx: int | None = None):
        super().__init__()
        self.talker = talker
        self.config = config
        self._s2c = subword_to_char
        self._char_pad = char_pad_idx
        self._warmup: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    # -- per-speaker constants -------------------------------------------

    def warmup(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(cond [P, H], uncond [P, H], prev_codes [Q])`` of the speaker warm-up.
        Computed once per speaker, under the serving autocast (bf16 on CUDA) no
        matter who asks first: the CUDA-graph capture's first preprocess runs
        outside the engine's autocast scope, and a warm-up computed there
        differs in the last bits from one computed under it -- enough to move
        a session's first frame (the talker's frame-0 decision is a knife-edge)."""
        if self._warmup is None:
            cfg = self.config
            dev = self.talker.embed_code.weight.device
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                self._warmup = self.talker.warmup_inputs(
                    self.SPEAKER, self._s2c, self._char_pad,
                    text_pad_id=cfg.text_pad_id, text_eos_id=cfg.text_eos_id, speech_pad_id=cfg.eartts.codebook_size,
                )
        return self._warmup

    @property
    def warmup_len(self) -> int:
        return self.talker.audio_prompt_latents[self.SPEAKER].shape[1]

    def _generator(self, rid: str, device: torch.device) -> torch.Generator:
        st = self.request_state(rid)
        gen = st.get(self.GEN_KEY)
        if gen is None:
            gen = torch.Generator(device=device).manual_seed(self.config.eartts.inference_seed)
            st.add(self.GEN_KEY, gen)
        return gen

    def _noise(self, gens: list[torch.Generator], device: torch.device) -> dict[str, torch.Tensor]:
        """The step's sampling noise for N rows, drawn per row from each row's
        generator: ``u`` for the Gumbel mixture pick and ``eps`` for the latent
        noise, for every MaskGIT iteration at once (two draws per row, so a
        64-session step costs 128 launches rather than ~900). A row's noise
        depends only on its own generator, never on the batch around it.
        Rows first: the engine's static-input buffers are narrowed along the
        leading (batch) dim when a smaller batch replays a capture."""
        e = self.config.eartts
        it = e.inference_num_iter
        u = torch.stack([torch.rand((it, 1, e.mog_num_predictions), device=device, generator=g) for g in gens])
        eps = torch.stack([torch.randn((it, 1, e.code_dim), device=device, generator=g) for g in gens])
        return {"noise_u": u, "noise_eps": eps}

    # -- engine contract -------------------------------------------------

    def _is_first_step(self, rid: str) -> bool:
        return self.request_state(rid).get(self.PREV_CODES_KEY) is None

    def cg_key_info(self, graph_walk: str, per_request_info: Mapping[str, Any]) -> str | None:
        """Which capture serves this batch: the steady-state decode capture
        (``DECODE_KEY``) when every session is past its first step, None (run
        eager) when any session prefills its warm-up this step. The engine
        leases the slot before ``prepare_inputs`` runs, so this answers from
        the request state, like ``declare_step`` does."""
        del graph_walk
        return None if any(self._is_first_step(rid) for rid in per_request_info) else self.DECODE_KEY

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> ARNodeInputs:
        # The LLM streams the sampled agent text token under "new_token". A
        # session's first step also prefills the speaker warm-up.
        tok = inputs["new_token"][0].reshape(1)
        first = self._is_first_step(fwd_info.request_id)
        span = self.warmup_len + 1 if first else 1
        return ARNodeInputs(input_ids=tok, input_seq_len=span, kwargs={"first": first})

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        # label-major: every request's cond segment, then every request's
        # uncond segment; the combined plan packs the rows in that order
        segments = [
            Segment(request_id=rid, label=label, span=inp.input_seq_len)
            for label in ("main", "uncond")
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        return SubmoduleStep(
            segments=segments,
            # the same answer as cg_key_info(): a warm-up prefill in the batch
            # keeps the step off the fixed-shape decode capture
            cg_key_info=None if any(inp.kwargs.get("first") for inp in inputs) else self.DECODE_KEY,
            steps={
                TALKER_KV: KVStep(combined_labels={("main", "uncond"): TALKER_CFG_LABEL}),
                TALKER_ATTN: AttentionStep(causal=True),
                TALKER_POS: PositionStep(),
            },
        )

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        """Capture the steady-state step (one token per label per session).
        ``preprocess`` (text conditioning, noise, warm-up assembly) runs eagerly
        before the graph; the captured region is the 2N-row backbone over the
        pool and the MaskGIT sampling loop."""
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="talker_decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.tensor([self.config.text_pad_id], dtype=torch.long, device=device),
                    input_seq_len=1, kwargs={"first": False},
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                # each request commits one token on each of its two labels
                total_tokens_multiplier=2,
                additional_key_info=self.DECODE_KEY,
                compile=False,
            ),
        ]

    def preprocess(self, graph_walk, engine_inputs, inputs):
        cfg, talker = self.config, self.talker
        dev = talker.embed_code.weight.device
        rids = list(engine_inputs.request_ids)
        ids = torch.cat([inp.input_ids.reshape(-1)[:1] for inp in inputs]).to(dev)
        tokens = ids.tolist()                                           # one D2H copy for the batch
        firsts = [bool(inp.kwargs.get("first", False)) for inp in inputs]
        warm_c, warm_u, warm_prev = self.warmup()
        prev_codes = torch.stack([
            warm_prev if first or self.request_state(rid).get(self.PREV_CODES_KEY) is None
            else self.request_state(rid)[self.PREV_CODES_KEY].reshape(-1)
            for rid, first in zip(rids, firsts, strict=True)
        ])                                                              # [N, Q]
        conds = talker.text_conditioning_from_ids(tokens, self._s2c, self._char_pad)   # [N, 1, H]
        frame_c, frame_u = talker.frame_inputs(prev_codes, conds, ids, cfg.text_eos_id)  # [N, H] each
        if any(firsts):
            # a first step's rows carry the warm-up positions before the frame
            rows_c = [torch.cat([warm_c, frame_c[i:i + 1]]) if f else frame_c[i:i + 1] for i, f in enumerate(firsts)]
            rows_u = [torch.cat([warm_u, frame_u[i:i + 1]]) if f else frame_u[i:i + 1] for i, f in enumerate(firsts)]
            x = torch.cat(rows_c + rows_u)                              # [total_tokens, H], label-major
        else:
            x = torch.cat([frame_c, frame_u])                           # [2N, H]
        gens = [self._generator(rid, dev) for rid in rids]
        return {"x": x, "spans": [inp.input_seq_len for inp in inputs], **self._noise(gens, dev)}

    def forward_batched(self, graph_walk, engine_inputs, x=None, spans=None, noise_u=None, noise_eps=None,
                        **kwargs) -> dict[str, NameToTensorList]:
        e = self.config.eartts
        rids = list(engine_inputs.request_ids)
        n = len(rids)
        hidden = self.talker.backbone_pooled(x, TALKER_CFG_LABEL)         # [total_tokens, H]
        if all(s == 1 for s in spans):
            last = hidden                                                 # [2N, H]: cond rows, uncond rows
        else:
            last = engine_inputs.resources[TALKER_ATTN].select_last_hidden(hidden, label=TALKER_CFG_LABEL)
        noise = [(noise_u[:, i], noise_eps[:, i]) for i in range(e.inference_num_iter)]
        codes = self.talker.generate_step(
            last[:n].unsqueeze(1), hidden_uncond=last[n:].unsqueeze(1),
            num_iter=e.inference_num_iter, guidance_scale=e.inference_guidance_scale,
            noise_scale=e.inference_noise_scale, top_p=e.inference_top_p, noise=noise,
        ).squeeze(1)                                                      # [N, Q]
        return {rid: {"codec_tokens": [codes[i]]} for i, rid in enumerate(rids)}

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def postprocess(self, request_id, request_info, outputs, **kwargs):
        # This frame's codes are the next frame's ``prev_codes`` (metadata only:
        # the tensor is already a copy out of the graph's output buffer).
        if "codec_tokens" in outputs:
            self.request_state(request_id).add(self.PREV_CODES_KEY, outputs["codec_tokens"][0])


class AudioCodecDecoderSubmodule(NodeSubmodule):
    """RVQ codec decoder: talker codes -> 22.05 kHz int16 PCM (streaming).

    The codec is causal with a multi-frame receptive field, so decoding each
    chunk in isolation clicks at the boundaries. Each request keeps a rolling
    context of its previous ``codec_left_context_frames`` code frames (in its
    ``PerRequestState``); a chunk of NEW frames is decoded as ``context + new``
    and only the new frames' samples are emitted. Same "decode with left
    context, emit the tail" math as the verified standalone path, but O(1) per
    chunk instead of re-decoding the whole history. Requests whose windows have
    the same length are decoded in one batched call. Declares no resources.
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
        codes = inputs["codec_tokens"][0]
        if codes.dim() == 3:
            codes = codes[0]                                            # (T_new, num_q)
        elif codes.dim() == 1:
            codes = codes.unsqueeze(0)                                  # single frame -> (1, num_q)
        return NodeInputs(tensor_inputs={"codes": codes}, input_seq_len=codes.shape[0])

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {"codes": [inp.tensor_inputs["codes"] for inp in inputs]}

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def _window(self, rid: str, codes: torch.Tensor) -> tuple[torch.Tensor, int]:
        """This request's ``context + new`` code window and its context length."""
        prev = self.request_state(rid).get(self.CONTEXT_KEY)             # (n_ctx, num_q) or None
        if prev is None:
            return codes, 0
        return torch.cat([prev, codes], dim=0), prev.shape[0]

    def forward_batched(self, graph_walk, engine_inputs, codes=None, **kwargs) -> dict[str, NameToTensorList]:
        lc = self.config.eartts.codec_left_context_frames
        rids = list(engine_inputs.request_ids)
        windows = [self._window(rid, c) for rid, c in zip(rids, codes, strict=True)]
        out: dict[str, NameToTensorList] = {}
        # one decode per distinct window length (steady state: every request is
        # at the full context + chunk length, so one call for the whole batch)
        by_len: dict[int, list[int]] = {}
        for i, (full, _) in enumerate(windows):
            by_len.setdefault(full.shape[0], []).append(i)
        for tf, idxs in by_len.items():
            batch = torch.stack([windows[i][0] for i in idxs]).long()   # (G, Tf, num_q)
            code_len = torch.full((len(idxs),), tf, device=batch.device)
            audio, _ = self.codec.decode(batch, code_len)               # (G, 1, samples)
            wav = audio.squeeze(1)                                      # (G, samples)
            spf = wav.shape[1] // tf                                    # samples per frame
            for row, i in enumerate(idxs):
                full, n_ctx = windows[i]
                new_wav = wav[row, n_ctx * spf:]                        # emit only the new frames
                self.request_state(rids[i]).add(self.CONTEXT_KEY, full[-lc:].detach())
                out[rids[i]] = {"audio_chunk": [(new_wav.clamp(-1, 1) * 32767).to(torch.int16)]}
        return out

    def forward(self, graph_walk, engine_inputs, codes=None, **kwargs) -> NameToTensorList:
        if isinstance(codes, torch.Tensor):                              # one request's window
            codes = [codes[0] if codes.dim() == 3 else codes.reshape(-1, codes.shape[-1])]
        return self.forward_batched(graph_walk, engine_inputs, codes=codes)[engine_inputs.request_ids[0]]
