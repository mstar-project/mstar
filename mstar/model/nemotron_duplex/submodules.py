"""NodeSubmodules for NVIDIA NemotronLabs VoiceChat-11B (full duplex).

Nodes, and the resources each declares (``NemotronDuplexModel.get_node_resources``):

    conformer_encoder  encoder_state,             16 kHz speech -> per-frame LLM embeds (+ RNN-T); per
                       encoder_front              session, its layers' state and its audio / mel tails
    nano_llm           nano_kv, nano_attn,        Nemotron-H hybrid Mamba-2 / attention / MLP (9B):
                       mamba_state, mamba,        paged KV for the 4 attention layers, recurrent-pool
                       nano_sampler               slots for the 27 Mamba-2 layers, the text sampler
    eartts_talker      talker_kv, talker_attn,    Gemma3 talker -> 31 RVQ codes per frame; two KV
                       talker_pos                 labels per session (cond / uncond) in one plan
    audio_codec        (none)                    RVQ codes -> 22.05 kHz PCM, per-request left context

Both autoregressive nodes keep their whole state in engine resources and capture
their decode step as a CUDA graph. The encoder keeps a call's state in its two
pools, and a live step (or its layer stack) replays a piecewise CUDA graph; the
codec replays one for its steady-state decode window.
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
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
)
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.kv.bounded import BoundedKVStep
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent import RecurrentStep
from mstar.model.nemotron_duplex.components.conformer import ConformerStreamer, StreamState, device_counts
from mstar.model.nemotron_duplex.config import (
    AUDIO_IN,
    ENCODER_FRONT,
    ENCODER_KV,
    ENCODER_STATE,
    MAMBA,
    MAMBA_STATE,
    NANO_ATTN,
    NANO_KV,
    NANO_SAMPLER,
    SESSION_BATCH_SIZES,
    TALKER_ATTN,
    TALKER_CFG_LABEL,
    TALKER_KV,
    TALKER_POS,
    NemotronDuplexConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

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

    # No torch.compile: the per-layer resource cursors are compile-disabled
    # and the decode step is a CUDA graph replay, which is where the time goes.
    disable_torch_compile = True

    # Decode batch sizes captured as CUDA graphs: one row per live session per
    # 80 ms tick. The recurrent pool holds a slot per session; padding rows
    # take none (they address the sink).
    DECODE_CAPTURE_BATCH_SIZES = SESSION_BATCH_SIZES
    DECODE_KEY = "decode"            # the steady-state capture's bucket key (cg_key_info)
    PROMPT_DONE_KEY = "prompt_consumed"   # set once the request's first step (prompt + frame) has run

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

    def cg_key_info(
        self, graph_walk: str, per_request_info: Mapping[str, Any], **kwargs: Any
    ) -> str | None:
        """Decode capture unless a request still has its prompt pending (its first
        step is prompt + frame and runs eager). The routing sets that flag once
        for the whole loop, so the node also checks whether the request's first
        step has already run."""
        if graph_walk != "decode":
            return None
        for rid, info in per_request_info.items():
            if (getattr(info, "step_metadata", None) or {}).get("prompt_pending") and not (
                self.request_state(rid).get(self.PROMPT_DONE_KEY)
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
    ) -> ARNodeInputs:
        """Host-side bookkeeping only; the embedding lookups and AddFusion run in
        ``preprocess`` on the GPU thread."""
        cfg = self.config
        if "audio_frame" in inputs:
            tensors = {
                "prev_text": self._tok(inputs, "prev_text", cfg.text_bos_id),
                "prev_func": self._tok(inputs, "prev_func", cfg.text_pad_id),
            }
            # The terminal stream chunk is empty when the stream's last frame
            # was taken before its end reached this worker: run a no-audio step,
            # which ends the loop (postprocess drops what it samples).
            frames = inputs.get("audio_frame") or []
            if frames:
                tensors["audio_frame"] = frames[0].reshape(1, -1)
            ends_loop = bool(is_final_stream_chunk) and not frames
            # The system prompt rides the loop as ``text_inputs``: real ids on
            # the session's first step, the NO_PROMPT sentinel afterwards.
            prompt = self._prompt_ids(inputs)
            if prompt is not None:
                tensors["prompt_ids"] = prompt
                return ARNodeInputs(input_seq_len=int(prompt.numel()) + 1, tensor_inputs=tensors,
                                    kwargs={"mode": _MODE_PROMPT_FRAME, "ends_loop": ends_loop})
            return ARNodeInputs(input_seq_len=1, tensor_inputs=tensors,
                                kwargs={"mode": _MODE_FRAME, "ends_loop": ends_loop})
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
        hidden = self.language_model(input_embeds, label="main")
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
        new_token = engine_inputs.resources[NANO_SAMPLER].sample(rids, logits=self.lm_head(last))
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
        # The prompt, if any, went with this step. Later steps use the capture.
        self.request_state(request_id).add(self.PROMPT_DONE_KEY, True)
        inputs = kwargs.get("inputs")
        if inputs is not None and inputs.kwargs.get("ends_loop"):
            # the empty terminal chunk's step only ends the loop: one token per
            # frame, as offline, so nothing it sampled leaves the node
            outputs.clear()
            return
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

class ConformerEncoderSubmodule(NodeSubmodule):
    """Fast-Conformer streaming STT encoder: the call's 16 kHz audio ->
    per-frame LLM-space embeddings, emitted one frame per stream item so the
    LLM's ``FixedChunkPolicy(1)`` paces the frame-synchronous decode loop.

    Audio arrives as the client streams it, one chunk per step, and the live
    calls' chunks run in one batched streamer pass (``ConformerStreamer``, which
    describes the state a call carries across chunks and its numerics), except
    that a step holding a chunk of more than a second (an upload, a large
    append) is not batched (``can_batch``): the engine then runs each call of
    it on its own. A call's state is bounded whatever its length: its newest audio and
    mel frames (the ``ENCODER_FRONT`` pool), the conv tails (``ENCODER_STATE``)
    and the last 70 frames of each layer's attention keys and values
    (``ENCODER_KV``, which also runs the attention); its host counters live in
    its per-request state, and its frame count is its position in the KV.

    A call one frame of audio further on the frame grid (every live step of a
    client appending 80 ms) runs whole as a CUDA graph (``STEP_REGION``). The
    other calls take the front end eagerly, then the layer stack as a graph
    (``LAYERS_REGION``) when each got at most one frame (a live step, the
    call's end), else eagerly (a one-shot upload, a large append). In a step
    whose other calls take the eager layers, or when the whole-step region has
    no graph for its calls, those calls run eagerly too.

    Holds the perception stack plus the RNN-T decoder/joint (loaded; the
    user-transcript channel is not emitted yet).
    """

    # The captured regions are not compiled (compile=False); the eager front end
    # takes variable-length chunks.
    disable_torch_compile = True

    STEP_REGION = "conformer_step"
    LAYERS_REGION = "conformer_layers"
    CAPTURE_BATCH_SIZES = SESSION_BATCH_SIZES

    STATE_KEY = "encoder_stream"         # a call's StreamState
    # Longest chunk batched with other calls': a pass pads every call to its
    # longest chunk. One second is a judgement, not a measured crossover.
    BATCHED_CHUNK_S = 1.0

    def __init__(self, perception: nn.Module, rnnt_decoder: nn.Module, rnnt_joint: nn.Module,
                 config: NemotronDuplexConfig):
        super().__init__()
        self.perception = perception
        self.rnnt_decoder = rnnt_decoder
        self.rnnt_joint = rnnt_joint
        self.config = config
        self._streamer = None

    def prepare_inputs(
        self, graph_walk, fwd_info, inputs, is_final_stream_chunk=False, **kwargs,
    ) -> NodeInputs:
        # ``audio_in`` is one chunk of raw 16 kHz mono audio
        chunk = inputs.get(AUDIO_IN)
        wav = chunk[0].reshape(-1) if chunk else None
        # the client stream releases one item per step; only its final flush is empty
        assert wav is not None or is_final_stream_chunk, "an empty chunk before the end of the stream"
        # The session cap is audio_in's max_items, which the client stream
        # enforces before a sample reaches here. Read, not counted here: admit
        # can refuse the step and run this again. The previous step's forward
        # has returned, so the state is current.
        st = self.request_state(fwd_info.rid_handle).get(self.STATE_KEY)
        samples = (st.samples if st is not None else 0) + (wav.numel() if wav is not None else 0)
        stt = self.perception.config
        if is_final_stream_chunk and samples <= stt.n_fft // 2:
            # too short to featurize (offline can't either): fail rather than reply to nothing
            raise ValueError(
                f"the call ended with {1000 * samples / stt.sample_rate:g} ms of audio; "
                f"it needs more than {1000 * (stt.n_fft // 2) / stt.sample_rate:g} ms"
            )
        final = bool(is_final_stream_chunk)
        at = st or StreamState()
        return NodeInputs(
            tensor_inputs={} if wav is None else {"chunk": wav},
            kwargs={
                "final": final,
                # whether the whole-step region can serve the call: decided
                # here, once, for the declaration and the forward to share
                "steady": st is not None and self.streamer.is_steady(st, wav, final),
                # the call's position in ENCODER_KV and the frames this step adds
                "position": at.position,
                "n": self.streamer.new_frames(at, samples, final),
            },
            input_seq_len=0 if wav is None else wav.numel(),   # samples, for can_batch
        )

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {
            "chunk": [inp.tensor_inputs.get("chunk") for inp in inputs],
            "final": [inp.kwargs.get("final", False) for inp in inputs],
            "steady": [inp.kwargs.get("steady", False) for inp in inputs],
            "n": [inp.kwargs.get("n", 0) for inp in inputs],
        }

    def can_batch(self, batch, model_inputs) -> bool:
        # A pass pads every call to its longest chunk, so a chunk of more than
        # BATCHED_CHUNK_S keeps its step from being batched, and the engine runs
        # each call's step on its own (declare, admit, plan, forward, commit).
        # Such chunks are uploads and large appends, so it is a rare step.
        longest = int(self.BATCHED_CHUNK_S * self.perception.config.sample_rate)
        return len(model_inputs) == 1 or all(inp.input_seq_len <= longest for inp in model_inputs)

    def max_batch_size(self, graph_walk: str) -> int:
        # The engine caps a batch at its whole-forward graphs only; past the
        # widest region capture every call of the step would run eagerly.
        return self.CAPTURE_BATCH_SIZES[-1]

    @staticmethod
    def _declare(request_ids, keys, kv: BoundedKVStep) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
            steps={**{key: RecurrentStep() for key in keys}, ENCODER_KV: kv},
        )

    @staticmethod
    def _kv_step(request_ids, spans: list[int], positions: dict) -> BoundedKVStep:
        """Each call's new frames (``spans``) at its position in the stream."""
        return BoundedKVStep(
            segments=tuple(Segment(request_id=rid, label="main", span=n)
                           for rid, n in zip(request_ids, spans, strict=True)),
            positions=positions,
        )

    def _region_kv_step(self, request_ids) -> BoundedKVStep:
        """A region's calls each add one frame. A capture's or a replay's padding
        row has no stream state, so it gets no position: the resource plans it
        as padding."""
        positions = {}
        for rid in request_ids:
            state = self.request_states.get(rid)
            st = None if state is None else state.get(self.STATE_KEY)
            if st is not None:
                positions[rid] = st.position
        return self._kv_step(request_ids, [1] * len(request_ids), positions)

    def declare_step(self, graph_walk, request_ids, inputs, slot_lease=None, piecewise_leases=None,
                     **kwargs) -> SubmoduleStep | None:
        del graph_walk, slot_lease, kwargs
        # A live step where every call is one frame on replays STEP_REGION
        # over the whole batch, at the bucket its lease names, and the region's
        # own step owns every resource. Each such call already holds its slots,
        # from a first step declared below, so that step's admit reserves nothing.
        if self.STEP_REGION in (piecewise_leases or {}) and all(inp.kwargs["steady"] for inp in inputs):
            return None
        # Otherwise this step reserves and plans every call's slots and new
        # frames, and the eager path reads that plan. A region the forward still
        # replays for some calls re-plans their rows into its own buffers, after
        # the eager path has read this plan. It can't own those rows instead: its
        # lease names the whole batch's bucket, not the sub-batch's, which may
        # have no graph, and its admit can't refuse a call that has no slot yet.
        spans = [inp.kwargs["n"] for inp in inputs]
        positions = {rid: inp.kwargs["position"] for rid, inp in zip(request_ids, inputs, strict=True)}
        return self._declare(request_ids, (ENCODER_FRONT, ENCODER_STATE),
                             self._kv_step(request_ids, spans, positions))

    @property
    def streamer(self) -> ConformerStreamer:
        if self._streamer is None:
            self._streamer = ConformerStreamer(self.perception)
        return self._streamer

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """A live step, one frame of audio per call, whole (``STEP_REGION``); and
        the layer stack alone for any step with at most one frame per call
        (``LAYERS_REGION``)."""
        del tp_world_size
        stt = self.perception.config
        frame_samples = stt.hop_length * stt.subsampling_factor

        def step_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {"chunk": torch.zeros(shape.bs, frame_samples, dtype=torch.float32, device=device)}

        def layers_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {
                "x": torch.zeros(shape.bs, 1, stt.d_model, dtype=autocast_dtype, device=device),
                "n": torch.zeros(shape.bs, dtype=torch.long, device=device),
            }

        def region(fn, inputs, keys, lease) -> PiecewiseBatchedConfig:
            return PiecewiseBatchedConfig(
                capture_fn=fn, make_static_inputs=inputs,
                declare_step=lambda request_ids, seq_lens: self._declare(
                    request_ids, keys, self._region_kv_step(request_ids)),
                seq_len=1, capture_batch_sizes=self.CAPTURE_BATCH_SIZES, compile=False, lease_before_step=lease,
            )

        regions = {
            self.STEP_REGION: region(self._step_region, step_inputs, (ENCODER_FRONT, ENCODER_STATE), True),
            self.LAYERS_REGION: region(self._layers_region, layers_inputs, (ENCODER_STATE,), False),
        }
        if not self.streamer.has_steady_step:
            del regions[self.STEP_REGION]
        return regions

    def _step_region(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        front, pool = call.resources[ENCODER_FRONT], call.resources[ENCODER_STATE]
        return {"y": self.streamer.steady_step(call.static_inputs["chunk"], front, front.addressing("main"),
                                               call.resources[ENCODER_KV], pool, pool.addressing("main"))}

    def _layers_region(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        pool = call.resources[ENCODER_STATE]
        s = call.static_inputs
        return {"y": self.streamer.layers(s["x"], call.resources[ENCODER_KV], pool, pool.addressing("main"),
                                          s["n"])}

    def _stream_state(self, rid) -> StreamState:
        state = self.request_state(rid)
        st = state.get(self.STATE_KEY)
        if st is None:
            st = StreamState()
            state.add(self.STATE_KEY, st)
        return st

    @staticmethod
    def _pool(engine_inputs, key) -> tuple:
        """A pool and the plan this step made for it, read when the eager path uses it."""
        pool = engine_inputs.resources[key]
        return pool, pool.addressing("main")

    @staticmethod
    def _to_device(chunks: list, device: torch.device) -> list:
        """The step's chunks on the device in one copy: a copy per call, from
        pageable memory, would each wait for its transfer."""
        host = [i for i, c in enumerate(chunks) if c is not None and c.device != device]
        moved = list(chunks)
        if host:
            flat = torch.cat([chunks[i].reshape(-1) for i in host])
            if device.type == "cuda":
                flat = flat.pin_memory()
            parts = flat.to(device, non_blocking=True).split([chunks[i].numel() for i in host])
            for i, part in zip(host, parts, strict=True):
                moved[i] = part
        return moved

    def forward_batched(self, graph_walk, engine_inputs, chunk=None, final=None, steady=None, n=None, **kwargs):
        rids = list(engine_inputs.request_ids)
        device = self.perception.proj.weight.device
        streamer = self.streamer
        states = [self._stream_state(rid) for rid in rids]
        chunks = self._to_device(chunk, device)
        runners = engine_inputs.piecewise_runners or {}
        frames: list[torch.Tensor | None] = [None] * len(rids)

        # calls one frame of audio further on the frame grid: the whole step as one graph, run last
        step_runner, layers_runner = runners.get(self.STEP_REGION), runners.get(self.LAYERS_REGION)

        def layers_graph(rows) -> bool:
            """Whether the layer region takes these calls' new frames: at most one each."""
            new = [i for i in rows if n[i]]
            return (bool(new) and max(n[i] for i in new) == 1 and layers_runner is not None
                    and layers_runner.can_run(len(new)))

        if engine_inputs.step is None:
            # declare_step left every resource to the region, which its lease says can replay the batch
            assert all(steady), "an encoder step declared nothing but is not all steady"
            steady = list(range(len(rids)))
        else:
            # This step's KV plan writes every call's new frames, so the region (which
            # writes its calls' under its own plan) may serve the steady calls only when
            # that plan goes unused: the other calls' frames, if any, all go through the
            # layer region. Otherwise, or with no graph for them, they run eagerly too.
            rest = [i for i, s in enumerate(steady) if not s]
            steady = [i for i, s in enumerate(steady) if s]
            if steady and not (step_runner is not None and step_runner.can_run(len(steady))
                               and (not any(n[i] for i in rest) or layers_graph(rest))):
                steady = []
        steady_chunks = [chunks[i] for i in steady]
        eager_n = list(n)
        for i in steady:
            chunks[i], eager_n[i] = None, 0     # the eager pass leaves these calls as they are

        # The rest: the same math eagerly, reading this step's plans where it
        # uses them, before any region plans its own; its layers as a graph when
        # no call got more than a frame. A chunk of more than a second comes in
        # a step of its own (see can_batch).
        enc = None
        if len(steady) < len(rids):
            enc = streamer.front(states, chunks, final, eager_n, *self._pool(engine_inputs, ENCODER_FRONT))
        if enc is not None:
            new = [i for i, k in enumerate(enc.n) if k]
            if layers_graph(new):
                # only the calls with a frame: the region plans its own rows
                y = layers_runner.run(static_inputs={"x": torch.stack([enc.x[i] for i in new]),
                                                     "n": torch.ones(len(new), dtype=torch.long, device=device)},
                                      request_ids=[rids[i] for i in new])["y"]
                y = dict(zip(new, y, strict=True))
            else:
                y = self._eager_layers(enc, engine_inputs.resources[ENCODER_KV],
                                       *self._pool(engine_inputs, ENCODER_STATE), device)
            streamer.advance(states, enc)
            for i in new:
                frames[i] = y[i][: enc.n[i]]

        if steady:
            y = step_runner.run(static_inputs={"chunk": torch.stack(steady_chunks)},
                                request_ids=[rids[i] for i in steady])["y"]
            for row, i in enumerate(steady):
                st = states[i]
                st.samples += streamer.frame_samples
                st.mels += streamer.c.subsampling_factor
                st.frames += 1
                frames[i] = y[row]
        # One stream item per frame: a single (T, H) tensor would be ONE item,
        # which the chunk policy could not slice. No check_stop, so nothing to
        # copy to the host for one.
        return BatchedModelOutput(
            per_rid_outputs={rid: {"audio_frame": [] if f is None else list(f)}
                             for rid, f in zip(rids, frames, strict=True)},
            check_stop_buffers={},
        )

    def _eager_layers(self, enc, kv, pool, rows, device) -> list[torch.Tensor]:
        """The layer stack eagerly over ``enc``, on this step's plans: each call's new frames."""
        y = self.streamer.layers(enc.x, kv, pool, rows, device_counts(enc.n, device))
        return [row[:n] for row, n in zip(y, enc.n, strict=True)]

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        # one call's step: preprocess's lists of one row
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, **kwargs).per_rid_outputs[rid]


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
    DECODE_CAPTURE_BATCH_SIZES = SESSION_BATCH_SIZES

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

    def cg_key_info(
        self, graph_walk: str, per_request_info: Mapping[str, Any], **kwargs: Any
    ) -> str | None:
        """Which capture serves this batch: the steady-state decode capture
        (``DECODE_KEY``) when every session is past its first step, None (run
        eager) when any session prefills its warm-up this step. The engine
        leases the slot before ``prepare_inputs`` runs, so this answers from
        the request state, like ``declare_step`` does."""
        del graph_walk
        return None if any(self._is_first_step(rid) for rid in per_request_info) else self.DECODE_KEY

    def prepare_inputs(self, graph_walk, fwd_info, inputs, is_final_stream_chunk=False, **kwargs) -> ARNodeInputs:
        # The LLM streams the sampled agent text token under "new_token". A
        # session's first step also prefills the speaker warm-up.
        chunk = inputs.get("new_token")
        if chunk:
            tok = chunk[0].reshape(1)
        else:
            # The token stream ends on an empty chunk when its last token was
            # taken before its end reached this worker (or the clip yielded no
            # token). Run a PAD frame, since a skipped rid in a stream-ended loop
            # breaks the stop; it only ends the loop, so postprocess drops it.
            dev = self.talker.embed_code.weight.device
            tok = torch.tensor([self.config.text_pad_id], dtype=torch.long, device=dev)
        first = self._is_first_step(fwd_info.rid_handle)
        span = self.warmup_len + 1 if first else 1
        return ARNodeInputs(input_ids=tok, input_seq_len=span,
                            kwargs={"first": first, "ends_loop": bool(is_final_stream_chunk) and not chunk})

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
        # One D2H read per step for the host-built char conditioning. It waits for
        # the queued kernels, but the alternatives were worse: a side-stream read
        # is not ordered against the producer, and a device-side char table
        # changes the conditioning by a bf16 ulp, which moves the talker's
        # turn-taking on some clips.
        tokens = ids.tolist()
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
                        **kwargs) -> BatchedModelOutput:
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
        # One copy out of the graph for the batch, a [1, Q] view per request.
        # Nothing for check_stop, the stream's end stops the loop.
        return BatchedModelOutput(row_outputs={"codec_tokens": codes}, check_stop_buffers={})

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        return _first_row(self.forward_batched(graph_walk, engine_inputs, **kwargs))

    def postprocess(self, request_id, request_info, outputs, **kwargs):
        inputs = kwargs.get("inputs")
        if inputs is not None and inputs.kwargs.get("ends_loop"):
            # one audio frame per text token, as offline: the codec flushes on
            # the empty final chunk this leaves
            outputs.clear()
            return
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
    chunk instead of re-decoding the whole history. The inverse STFT's last
    ``n_fft - hop`` samples of a window still lack the following frames'
    overlap, so a chunk holds those samples back and the next chunk emits them
    recomputed with its context: the emitted stream is the full-history decode
    sample for sample; the final chunk emits its held tail too.
    Requests whose windows have the same length are decoded in one batched
    call; a steady-state window (left context + one chunk, every chunk after a
    call's first) replays a CUDA graph (``DECODE_REGION``). Calls that started
    at different times reach their chunk boundaries on different frames, so a
    live load runs many small decodes. Declares no resources.
    """

    # Per-request context is keyed by request id -> eager around the decode. No
    # autocast: the codec runs in the dtype it was loaded in (the engine's).
    disable_torch_compile = True
    disable_autocast = True

    DECODE_REGION = "codec_decode"
    CAPTURE_BATCH_SIZES = SESSION_BATCH_SIZES

    CONTEXT_KEY = "codec_ctx"
    TAIL_KEY = "codec_tail"              # the last chunk's held-back samples, int16

    def __init__(self, codec: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.codec = codec
        self.config = config
        # samples at a window's right edge without their overlap-add partners
        self._hold = max(int(getattr(codec, "n_fft", 0)) - int(getattr(codec, "hop_length", 0)), 0)

    def prepare_inputs(
        self, graph_walk, fwd_info, inputs, is_final_stream_chunk=False, **kwargs,
    ) -> NodeInputs | None:
        # ``codec_tokens`` is this chunk's NEW RVQ frames (T_new, num_q); the
        # Talker->Codec connection is a non-overlapping FixedChunkPolicy, so the
        # left context comes from per-request state, not the stream.
        flush = {"flush": bool(is_final_stream_chunk)}
        chunk = inputs.get("codec_tokens")
        if not chunk:
            # a frame count divisible by the chunk size ends on an empty final
            # chunk; run it to flush the tail. Skipping it hangs a batched stream
            return NodeInputs(kwargs=flush) if is_final_stream_chunk else None
        codes = chunk[0]
        if codes.dim() == 3:
            # the stream stacks the talker's [1, num_q] frames to (T_new, 1, num_q);
            # a whole window handed over at once is (1, T_new, num_q)
            codes = codes.reshape(-1, codes.shape[-1])                  # (T_new, num_q)
        elif codes.dim() == 1:
            codes = codes.unsqueeze(0)                                  # single frame -> (1, num_q)
        return NodeInputs(tensor_inputs={"codes": codes}, kwargs=flush, input_seq_len=codes.shape[0])

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {
            "codes": [inp.tensor_inputs.get("codes") for inp in inputs],
            "flush": [inp.kwargs.get("flush", False) for inp in inputs],
        }

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int:
        # as the encoder's: the region's widest capture, which the engine's cap does not see
        return self.CAPTURE_BATCH_SIZES[-1]

    @property
    def _steady_window(self) -> int:
        eartts = self.config.eartts
        return eartts.codec_left_context_frames + eartts.codec_chunk_frames

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        del autocast_dtype, tp_world_size
        window, num_q = self._steady_window, self.config.eartts.num_quantizers

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {"codes": torch.zeros(shape.bs, window, num_q, dtype=torch.long, device=device)}

        return {
            self.DECODE_REGION: PiecewiseBatchedConfig(
                capture_fn=self._decode_region,
                make_static_inputs=make_static_inputs,
                seq_len=1,
                capture_batch_sizes=self.CAPTURE_BATCH_SIZES,
                compile=False,
            ),
        }

    def _decode_region(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        # captured: the inverse DFT as a matmul, since the cuFFT plan cache can destroy a captured plan
        audio, _ = self.codec.decode(call.static_inputs["codes"], by_matmul=True)
        return {"wav": audio.squeeze(1)}

    def _window(self, rid: str, codes: torch.Tensor) -> tuple[torch.Tensor, int]:
        """This request's ``context + new`` code window and its context length."""
        prev = self.request_state(rid).get(self.CONTEXT_KEY)             # (n_ctx, num_q) or None
        if prev is None:
            return codes, 0
        return torch.cat([prev, codes], dim=0), prev.shape[0]

    def forward_batched(
        self, graph_walk, engine_inputs, codes=None, flush=None, **kwargs,
    ) -> dict[str, NameToTensorList]:
        lc = self.config.eartts.codec_left_context_frames
        rids = list(engine_inputs.request_ids)
        flush = flush or [False] * len(rids)
        out: dict[str, NameToTensorList] = {}
        windows: dict[int, tuple[torch.Tensor, int]] = {}
        for i, (rid, c) in enumerate(zip(rids, codes, strict=True)):
            if c is None:
                # an empty final chunk: nothing to decode, emit what was held
                tail = self.request_state(rid).get(self.TAIL_KEY)
                out[rid] = {"audio_chunk": [tail if tail is not None else torch.zeros(0, dtype=torch.int16)]}
            else:
                windows[i] = self._window(rid, c)
        # one decode per distinct window length (steady state: every request is
        # at the full context + chunk length, so one call for the whole batch)
        by_len: dict[int, list[int]] = {}
        for i, (full, _) in windows.items():
            by_len.setdefault(full.shape[0], []).append(i)
        runner = (engine_inputs.piecewise_runners or {}).get(self.DECODE_REGION)
        for tf, idxs in by_len.items():
            batch = torch.stack([windows[i][0] for i in idxs]).long()   # (G, Tf, num_q)
            if runner is not None and tf == self._steady_window and runner.can_run(len(idxs)):
                wav = runner.run(static_inputs={"codes": batch}, real_bs=len(idxs))["wav"]
            else:
                audio, _ = self.codec.decode(batch)                     # (G, 1, samples)
                wav = audio.squeeze(1)                                  # (G, samples)
            spf = wav.shape[1] // tf                                    # samples per frame
            for row, i in enumerate(idxs):
                full, n_ctx = windows[i]
                # the new frames' samples, less the right edge held for the next
                # chunk, plus the previous chunk's held edge now that its
                # overlap partners exist (a first chunk has nothing held)
                start = n_ctx * spf - self._hold if n_ctx > 0 else 0
                end = wav.shape[1] if flush[i] else wav.shape[1] - self._hold
                pcm = (wav[row, start:].clamp(-1, 1) * 32767).to(torch.int16)
                state = self.request_state(rids[i])
                state.add(self.CONTEXT_KEY, full[-lc:].detach())
                if self._hold:
                    state.add(self.TAIL_KEY, pcm[end - start :])
                out[rids[i]] = {"audio_chunk": [pcm[: end - start]]}
        return out

    def forward(self, graph_walk, engine_inputs, codes=None, flush=None, **kwargs) -> NameToTensorList:
        if isinstance(codes, torch.Tensor):                              # one request's window
            codes = [codes[0] if codes.dim() == 3 else codes.reshape(-1, codes.shape[-1])]
        if isinstance(flush, bool):
            flush = [flush]
        return self.forward_batched(graph_walk, engine_inputs, codes=codes, flush=flush)[engine_inputs.request_ids[0]]
