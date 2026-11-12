"""NodeSubmodules for NVIDIA NemotronLabs VoiceChat-11B (full duplex).

Nodes, and the state or resources each declares:

    conformer_encoder  encoder_kv,                16 kHz speech -> per-frame LLM embeds; per session,
                       encoder_state,             its attention window, its layers' conv state and its
                       encoder_front              audio / mel tails (the shared
                                                  ``FastConformerStreamSubmodule``)
    nano_llm           nano_kv, mamba_state,      Nemotron-H hybrid Mamba-2 / attention / MLP (9B):
                       nano_sampler               the KV of the 4 attention layers, the conv and SSM
                                                  state of the 27 Mamba-2 layers, the text sampler
    eartts_talker      talker_kv, talker_prev,    Gemma3 talker -> 31 RVQ codes per frame (two KV
                       codec_state                labels per session, cond / uncond, in one plan),
                                                  from the previous frame's codes, decoded in the
                                                  same step to 80 ms of 22.05 kHz PCM against the
                                                  call's codec stream state

Both autoregressive nodes keep their whole state in declared state and capture
their decode step as a CUDA graph (the talker's with its codec decode). The
encoder keeps a call's state in its bounded KV and two pools, and a live step
(or its layer stack) replays a piecewise CUDA graph.
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
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
)
from mstar.engine.resources import SlotLease, SubmoduleStep
from mstar.engine.state import ArrayStep, SampleStep, Segment, at_last_token, sample
from mstar.model.nemotron_duplex.components.audio_codec import AudioCodec, CodecStreamAudio, CodecStreamState
from mstar.model.nemotron_duplex.config import (
    CODEC_STATE,
    MAMBA_STATE,
    NANO_KV,
    NANO_SAMPLER,
    SESSION_BATCH_SIZES,
    TALKER_KV,
    TALKER_PREV,
    NemotronDuplexConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
)
from mstar.utils.h2d import to_device_nonblocking

logger = logging.getLogger(__name__)

# How ``prepare_inputs`` classified a request's step; ``preprocess`` fuses accordingly.
_MODE_FRAME = "frame"      # one streamed audio frame + fed-back prev_text / prev_func
_MODE_PROMPT_FRAME = "prompt_frame"   # a session's first step: the system prompt's tokens, then its first frame
NO_PROMPT = -1             # the ``text_inputs`` loop-back value once the prompt has been consumed (or never existed)


def _first_row(out: BatchedModelOutput) -> NameToTensorList:
    """The eager single-request ``forward`` contract (``{name: [tensor]}``) as
    row 0 of a row-addressed batch output."""
    return {name: [rows[0:1]] for name, rows in (out.row_outputs or {}).items()}
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

    # No torch.compile: every state call is a graph break, and the decode step
    # is a CUDA graph replay, which is where the time goes.
    disable_torch_compile = True

    # Decode batch sizes captured as CUDA graphs: one row per live session per
    # 80 ms tick. The recurrent pool holds a slot per session; padding rows
    # take none (they address the sink).
    DECODE_CAPTURE_BATCH_SIZES = SESSION_BATCH_SIZES
    DECODE_KEY = "decode"            # the steady-state capture's bucket key (cg_key_info)

    def __init__(self, language_model: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.language_model = language_model
        self.embeddings = language_model.embeddings
        self.lm_head = language_model.lm_head
        self.config = config

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
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
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=frame,
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                additional_key_info=self.DECODE_KEY,
                compile=False,
            ),
        ]

    # -- state -----------------------------------------------------------

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.sampler = handles[NANO_SAMPLER]

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        # the Mamba-2 kernels take the single-step path when every row spans one token
        state = {NANO_KV: ArrayStep(), MAMBA_STATE: ArrayStep()}
        if graph_walk == "decode":
            # The prompt region is not sampled (held at PAD), so only decode
            # steps the sampler; the reference's repetition penalty likewise
            # only ever sees generated tokens.
            state[NANO_SAMPLER] = SampleStep(apply_penalty=True)
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
            state=state,
        )

    def cg_key_info(
        self, graph_walk: str, per_request_info: Mapping[str, Any], **kwargs: Any
    ) -> str | None:
        """Decode capture unless a request still has its prompt pending (its first
        step is prompt + frame and runs eager). The routing sets that flag once
        for the whole loop, so the node also checks that the step is the loop's
        first, iteration 0."""
        if graph_walk != "decode":
            return None
        for info in per_request_info.values():
            if (getattr(info, "step_metadata", None) or {}).get("prompt_pending") and not (
                info.dynamic_loop_iter_counts.get("decode_loop", 0)
            ):
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
        is_final_stream_chunk: bool = False,
        **kwargs,
    ) -> ARNodeInputs | None:
        """Host-side bookkeeping only; the embedding lookups and AddFusion run in
        ``preprocess`` on the GPU thread."""
        cfg = self.config
        if "audio_frame" in inputs:
            frames = inputs.get("audio_frame") or []
            if not frames:
                # the terminal stream chunk is empty when the stream's last frame
                # was taken before its end reached this worker; declined, it
                # still ends the loop
                assert is_final_stream_chunk, "an empty chunk before the end of the stream"
                return None
            tensors = {
                "prev_text": self._tok(inputs, "prev_text", cfg.text_bos_id),
                "prev_func": self._tok(inputs, "prev_func", cfg.text_pad_id),
                "audio_frame": frames[0].reshape(1, -1),
            }
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
        audio = torch.cat([t["audio_frame"].reshape(1, -1).to(device=device, dtype=fused.dtype)
                           for t in tensors])                                                     # [N, H]
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
    ) -> BatchedModelOutput:
        """One fused forward over the packed batch: attention over the planned
        paged KV, Mamba-2 over each request's slot in the recurrent pool.

        The outputs are row-addressed (row i is request i): the engine copies
        each batch tensor out of the graph's buffers once and hands every
        request a view, instead of one clone per request per name. Nothing is
        offered to ``check_stop`` (it would be copied to the host per step):
        the decode loop ends on the audio stream's final chunk or the loop's
        own ``max_iters``, never on a token."""
        cfg = self.config
        rids = list(engine_inputs.request_ids)
        n = len(rids)
        hidden = self.language_model(input_embeds)
        if graph_walk != "decode":
            # Prompt priming: no sampling, the region is held at PAD (reference
            # ``_prime_prompt``), so the first audio frame follows the last PAD.
            pad = torch.full((n,), cfg.text_pad_id, dtype=torch.long, device=hidden.device)
            return BatchedModelOutput(row_outputs={"prev_text": pad, "prev_func": pad}, check_stop_buffers={})

        if all(n == 1 for n in seq_lens):
            last = hidden                                   # decode: one row per request (captured shape)
        else:
            ends = list(itertools.accumulate(seq_lens))
            last = hidden.index_select(0, torch.tensor([e - 1 for e in ends], device=hidden.device))
        new_token = sample(self.sampler, self.lm_head(last))
        rows = {
            "new_token": new_token.reshape(n),
            # the prompt (if any) has been consumed by this step: the loop-back
            # carries the sentinel from here on
            "text_inputs": torch.full((n,), NO_PROMPT, dtype=torch.long, device=hidden.device),
        }
        if cfg.use_function_head:
            # Tool-call channel: plain argmax, as in the reference.
            rows["new_func"] = self.language_model.function_head(last).argmax(dim=-1).reshape(n)
        return BatchedModelOutput(row_outputs=rows, check_stop_buffers={})

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        seq_lens: list[int],
        **kwargs,
    ) -> NameToTensorList:
        return _first_row(self.forward_batched(graph_walk, engine_inputs, input_embeds, seq_lens))

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

    # No ``check_stop``. Duplex emits one text token per audio frame and EOS is
    # an ordinary per-frame token (silence, turn boundary), not the end of
    # generation. The loop ends with the audio stream or at the Loop's max_iters.


# ---------------------------------------------------------------------------
# Audio stages
# ---------------------------------------------------------------------------

class EarTTSTalkerSubmodule(ARNodeSubmodule):
    """Gemma3 talker: one agent text token -> ``num_quantizers`` RVQ codes per
    session per frame, all live sessions in one backbone pass.

    The talker's KV is the ``TALKER_KV`` array under two labels per request:
    ``main`` (text-conditioned stream) and ``uncond`` (null-conditioned stream
    for classifier-free guidance), which a step over both plans as one
    combined label whose rows are ``[cond rows; uncond rows]`` -- the order
    ``mog_head.infer`` expects. A session's first step prefills the 37 speaker
    warm-up positions together with its first frame (38 tokens per label);
    every later step is one token per label. The MoG / MaskGIT sampling is
    internal (not the engine's categorical sampler). Its noise at a session's
    k-th step is row k of a memo that one generator, seeded as every session's
    is, fills in step order, so a session's speech does not depend on its batch
    mates and the whole step is a CUDA graph replay for all-decode batches.

    The same step decodes each session's new codes to its 80 ms of PCM
    (``codec_step``): the codec is causal, so the call's decoder stream state in
    ``CODEC_STATE`` is all a frame needs, and a frame's audio leaves with the
    step that made its codes. A call's PCM equals the full decode of its frames:
    its first frame drops the overlap-add's trimmed left edge, and its last step
    emits the right edge (``postprocess``).
    """

    disable_torch_compile = True

    DECODE_KEY = "decode"            # the steady-state capture's bucket key (cg_key_info)
    SPEAKER = "Aria"
    DECODE_CAPTURE_BATCH_SIZES = SESSION_BATCH_SIZES

    def __init__(self, talker: nn.Module, config: NemotronDuplexConfig, codec: AudioCodec,
                 subword_to_char: dict | None = None, char_pad_idx: int | None = None, noise_rows: int = 1):
        """``noise_rows`` is the talker loop's cap plus one, the steps a session can take."""
        super().__init__()
        self.talker = talker
        self.codec = codec
        self.config = config
        self._s2c = subword_to_char
        self._char_pad = char_pad_idx
        self._warmup: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.noise_rows = noise_rows

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

    # -- state -----------------------------------------------------------

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        """Take the talker's handles, start every session's previous codes at the warm-up's, and allocate the noise
        memo on the talker's device."""
        self.talker_kv, self.prev, self.codec_state = handles[TALKER_KV], handles[TALKER_PREV], handles[CODEC_STATE]
        self.prev.set_start({"codes": self.warmup()[2]})
        e = self.config.eartts
        device = self.talker.embed_code.weight.device
        self._noise_gen = torch.Generator(device=device).manual_seed(e.inference_seed)
        self._noise_u = torch.empty(self.noise_rows, e.inference_num_iter, 1, e.mog_num_predictions, device=device)
        self._noise_eps = torch.empty(self.noise_rows, e.inference_num_iter, 1, e.code_dim, device=device)
        self._noise_filled = 0
        logger.info("talker noise memo: %d rows, %d bytes", self.noise_rows,
                    self._noise_u.nbytes + self._noise_eps.nbytes)

    def _noise(self, steps: list[int], device: torch.device) -> dict[str, torch.Tensor]:
        """Each row's sampling noise at its talker step ``steps[i]``: ``u`` for the
        Gumbel mixture pick and ``eps`` for the latent noise, for every MaskGIT
        iteration at once. Row k of the memo is what a session's own generator
        draws at its k-th step, so the memo fills to the last step asked for, in
        step order. Rows first: the engine's static-input buffers are narrowed
        along the leading (batch) dim when a smaller batch replays a capture."""
        last = max(steps)
        if last >= self.noise_rows:
            raise ValueError(f"talker step {last} has no noise row: the memo holds {self.noise_rows}, the talker "
                             f"loop's cap of {self.noise_rows - 1} plus one")
        e = self.config.eartts
        it, gen = e.inference_num_iter, self._noise_gen
        for k in range(self._noise_filled, last + 1):
            self._noise_u[k] = torch.rand((it, 1, e.mog_num_predictions), device=device, generator=gen)
            self._noise_eps[k] = torch.randn((it, 1, e.code_dim), device=device, generator=gen)
        self._noise_filled = max(self._noise_filled, last + 1)
        rows = to_device_nonblocking(steps, device)
        return {"noise_u": self._noise_u.index_select(0, rows), "noise_eps": self._noise_eps.index_select(0, rows)}

    # -- engine contract -------------------------------------------------

    def cg_key_info(
        self, graph_walk: str, per_request_info: Mapping[str, Any], **kwargs: Any
    ) -> str | None:
        """Which capture serves this batch: the steady-state decode capture
        (``DECODE_KEY``) when every session is past its first step, None (run
        eager) when any session prefills its warm-up this step. The engine
        leases the slot before ``prepare_inputs`` runs, so this answers from
        the talker loop's count, as ``declare_step`` does."""
        del graph_walk
        if any(not info.dynamic_loop_iter_counts.get("talker_decode_loop", 0) for info in per_request_info.values()):
            return None
        return self.DECODE_KEY

    def prepare_inputs(self, graph_walk, fwd_info, inputs, is_final_stream_chunk=False, **kwargs) -> ARNodeInputs:
        # The LLM streams the sampled agent text token under "new_token". A
        # session's first step also prefills the speaker warm-up.
        chunk = inputs.get("new_token")
        if chunk:
            tok = chunk[0].reshape(1)
        else:
            # The token stream ends on an empty chunk when its last token was
            # taken before its end reached this worker (or the clip yielded no
            # token). Run a PAD frame rather than declining: its step flushes
            # the call's ``audio_end`` (see postprocess).
            dev = self.talker.embed_code.weight.device
            tok = torch.tensor([self.config.text_pad_id], dtype=torch.long, device=dev)
        step = fwd_info.dynamic_loop_iter_counts.get("talker_decode_loop", 0)
        first = step == 0
        span = self.warmup_len + 1 if first else 1
        final = bool(is_final_stream_chunk)
        return ARNodeInputs(input_ids=tok, input_seq_len=span,
                            kwargs={"first": first, "final": final, "ends_loop": final and not chunk, "step": step})

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
        # one frame of codes per session, a slot each
        frames = tuple(Segment(request_id=rid, label="main", span=1) for rid in request_ids)
        return SubmoduleStep(
            segments=segments,
            # the same answer as cg_key_info(): a warm-up prefill in the batch
            # keeps the step off the fixed-shape decode capture
            cg_key_info=None if any(inp.kwargs.get("first") for inp in inputs) else self.DECODE_KEY,
            state={
                TALKER_KV: ArrayStep(),
                TALKER_PREV: ArrayStep(segments=frames),
                CODEC_STATE: ArrayStep(segments=frames),
            },
        )

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """Capture the steady-state step (one token per label per session).
        ``preprocess`` (text conditioning, noise, warm-up assembly) runs eagerly
        before the graph; the captured region is the 2N-row backbone over the
        pool and the MaskGIT sampling loop."""
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="talker_decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.tensor([self.config.text_pad_id], dtype=torch.long, device=device),
                    input_seq_len=1, kwargs={"first": False, "step": 0},
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
        ids = torch.cat([inp.input_ids.reshape(-1)[:1] for inp in inputs]).to(dev)
        # One D2H read per step for the host-built char conditioning. It waits for
        # the queued kernels, but the alternatives were worse: a side-stream read
        # is not ordered against the producer, and a device-side char table
        # changes the conditioning by a bf16 ulp, which moves the talker's
        # turn-taking on some clips.
        tokens = ids.tolist()
        firsts = [bool(inp.kwargs.get("first", False)) for inp in inputs]
        warm_c, warm_u, _ = self.warmup()
        # a session's first step reads the warm-up's codes, the start `bind_state` sets
        prev_codes = self.prev.read(0).fields["codes"]                   # [N, Q]
        conds = talker.text_conditioning_from_ids(tokens, self._s2c, self._char_pad)   # [N, 1, H]
        frame_c, frame_u = talker.frame_inputs(prev_codes, conds, ids, cfg.text_eos_id)  # [N, H] each
        if any(firsts):
            # a first step's rows carry the warm-up positions before the frame
            rows_c = [torch.cat([warm_c, frame_c[i:i + 1]]) if f else frame_c[i:i + 1] for i, f in enumerate(firsts)]
            rows_u = [torch.cat([warm_u, frame_u[i:i + 1]]) if f else frame_u[i:i + 1] for i, f in enumerate(firsts)]
            x = torch.cat(rows_c + rows_u)                              # [total_tokens, H], label-major
        else:
            x = torch.cat([frame_c, frame_u])                           # [2N, H]
        noise = self._noise([inp.kwargs["step"] for inp in inputs], dev)
        return {"x": x, "spans": [inp.input_seq_len for inp in inputs], **noise}

    def forward_batched(self, graph_walk, engine_inputs, x=None, spans=None, noise_u=None, noise_eps=None,
                        **kwargs) -> BatchedModelOutput:
        e = self.config.eartts
        n = len(engine_inputs.request_ids)
        hidden = self.talker.backbone_pooled(x)                           # [total_tokens, H]
        if all(s == 1 for s in spans):
            last = hidden                                                 # [2N, H]: cond rows, uncond rows
        else:
            last = at_last_token(self.talker_kv, hidden)
        noise = [(noise_u[:, i], noise_eps[:, i]) for i in range(e.inference_num_iter)]
        codes = self.talker.generate_step(
            last[:n].unsqueeze(1), hidden_uncond=last[n:].unsqueeze(1),
            num_iter=e.inference_num_iter, guidance_scale=e.inference_guidance_scale,
            noise_scale=e.inference_noise_scale, top_p=e.inference_top_p, noise=noise,
        ).squeeze(1)                                                      # [N, Q]
        # the next frame's prev_codes
        self.prev.write(0, codes=codes)
        audio = codec_step(self.codec, self.codec_state, codes.unsqueeze(1))
        # One copy out of the graph per name, a row view per request. Nothing
        # for check_stop, the stream's end stops the loop.
        return BatchedModelOutput(row_outputs={
            "codec_tokens": codes,
            "audio_chunk": pcm16(audio.final),                            # [N, wav_to_token_ratio]
            # a call's last samples, had it ended before / after this frame
            "audio_end": pcm16(torch.cat([audio.end_before, audio.end_after], dim=1)),
        }, check_stop_buffers={})

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        return _first_row(self.forward_batched(graph_walk, engine_inputs, **kwargs))

    def postprocess(self, request_id, request_info, outputs, **kwargs):
        inputs = kwargs.get("inputs")
        flags = inputs.kwargs if inputs is not None else {}
        if flags.get("ends_loop"):
            # The empty final chunk's PAD frame only ends the loop (one audio
            # frame per text token, as offline): it emits the call's right
            # edge as it stood before this frame, or nothing for a call that
            # never had a frame.
            end = outputs.get("audio_end")
            outputs.clear()
            if end and not flags.get("first"):
                outputs["audio_chunk"] = [end[0][..., :self.codec.stream_edge]]
            return
        if "audio_chunk" in outputs and (flags.get("first") or flags.get("final")):
            # the full decode trims the overlap-add's left edge, and ends on the
            # samples after the last frame's final ones
            pcm, edge = outputs["audio_chunk"][0], self.codec.stream_edge
            if flags.get("first"):
                pcm = pcm[..., edge:]
            if flags.get("final"):
                pcm = torch.cat([pcm, outputs["audio_end"][0][..., edge:]], dim=-1)
            outputs["audio_chunk"] = [pcm]


def codec_step(codec: AudioCodec, state: Any, codes: torch.Tensor) -> CodecStreamAudio:
    """Decode each row's next code frames ``[B, T, num_quantizers]`` against its
    stream state, the ``CODEC_STATE`` Value under the handle ``state``: read the
    entry, decode only the new frames, write the entry back. A fresh slot is
    zeros, the decoder's causal padding, so a call's first frame needs nothing
    else. Fixed-shape, so a graph captures it."""
    blocks = codec.config.num_blocks
    stages = len(codec.config.channels)
    with torch.autocast(device_type=codes.device.type, enabled=False):
        entry = state.read(0).fields
        stream = CodecStreamState(
            tails=[t for i in range(stages) for t in entry[f"stage{i}"].unbind(1)],
            overlap=entry["overlap"],
        )
        audio, stream = codec.decode_stream(codes, stream)
        state.write(0, overlap=stream.overlap, **{
            f"stage{i}": torch.stack(stream.tails[i * blocks:(i + 1) * blocks], dim=1).to(entry[f"stage{i}"].dtype)
            for i in range(stages)
        })
    return audio


def pcm16(wav: torch.Tensor) -> torch.Tensor:
    return (wav.float().clamp(-1, 1) * 32767).to(torch.int16)
