"""Step-Audio2's node submodules: the audio tower and the Qwen2 LLM.

The LLM's prompt is the whole rendered chat, ``<audio_patch>`` placeholders included; an
audio prefill's forward splices the tower's rows in at those placeholders, in prompt
order. RoPE is plain 1D over the sequence, so the position resource advances by the token
count.

A spoken reply decodes on its own walk, whose sampled ids stream to the vocoder. The
stream carries only speech ids: the ``TokenStop`` withholds the stream's item on a spoken
row whose id is anything else (text, ``<tts_pad>``, a stop id), off the host copy the stop
check already has, so no forward reads its own tokens.
"""
from __future__ import annotations

from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL, CurrentForwardPassInfo
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KeyPrefixAttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SubmoduleStep,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.multimodal import splice_media
from mstar.model.step_audio2.components.audio import StepAudio2Audio, request_layout
from mstar.model.step_audio2.config import (
    AUDIO_ATTN,
    AUDIO_EMBEDS,
    AUDIO_MEL_FRAMES,
    AUDIO_MELS,
    AUDIO_PATCH,
    DECODE,
    DECODE_LOOP,
    DECODE_SPOKEN,
    DECODE_WALKS,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    PREFILL_AUDIO,
    PREFILL_WALKS,
    SPEECH_IDS,
    StepAudio2Config,
    audio_tokens,
    is_speech_id,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    InputSeqLenInfo,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    TokenStop,
)


class LLMSubmodule(ARNodeSubmodule):
    """The Qwen2 decoder: a prefill walk per prompt kind, then a text or a spoken decode.

    Eager for now (no capture configs); the forward is tensor-in, tensor-out, with the
    audio splice inside it, so decode and prefill captures can be declared as MiniCPM-o's
    are.
    """

    # Eager: every step's shape is new to a compiled forward without capture buckets.
    disable_torch_compile = True

    def __init__(self, model: Qwen3DenseLM, config: StepAudio2Config):
        super().__init__()
        self.model = model
        self.config = config
        # A prefill samples the reply's first token, so a stop id there (or a one-token
        # budget) ends the partition. A text decode's stop ids are checked by the graph
        # runtime; a spoken decode's by ``check_stop``, which also withholds the stream on
        # any id but speech. The budget is the loop's cap.
        self.token_stop = TokenStop(
            loop=DECODE_LOOP, buffer="new_token", stop_ids=frozenset(config.stop_token_ids),
            sampler=LLM_SAMPLER, walk=frozenset({DECODE, DECODE_SPOKEN}), prefill_walks=PREFILL_WALKS,
            last_prefill_flag=FINAL_PREFILL, withhold=(SPEECH_IDS,), withhold_unless=is_speech_id,
            withhold_walk=DECODE_SPOKEN, runtime_check=True,
        )
        # a text reply's decode reads its last token off the sampler's slot masters, and
        # its client token rides inline
        self.device_loopback = DeviceLoopback(
            walk=DECODE, signal="text_inputs", sampler=LLM_SAMPLER, buffer="new_token",
        )

    def inline_signals(self, graph_walk: str) -> dict[str, str]:
        # a spoken reply's client token rides inline too; its speech ids go through the host
        if graph_walk == DECODE_SPOKEN:
            return {self.device_loopback.signal: self.device_loopback.buffer}
        return super().inline_signals(graph_walk)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        has_loopback: bool = False,
        **kwargs: Any,
    ) -> ARNodeInputs:
        if (row := self.uniform_row_inputs(graph_walk, has_loopback)) is not None:
            # one token per row, read off the sampler's slot master in ``preprocess``
            return row
        ids = inputs["text_inputs"][0]
        if graph_walk != PREFILL_AUDIO:
            return ARNodeInputs(input_seq_len=ids.shape[0], input_ids=ids)
        embeds = inputs[AUDIO_EMBEDS][0]
        # host values only: the intake's clip lengths say how many placeholders it laid out
        pads = sum(audio_tokens(n) for n in fwd_info.step_metadata[AUDIO_MEL_FRAMES])
        if embeds.shape[0] != pads:
            raise ValueError(f"{AUDIO_EMBEDS} has {embeds.shape[0]} rows for the prompt's {pads} placeholders")
        return ARNodeInputs(input_seq_len=ids.shape[0], input_ids=ids, tensor_inputs={AUDIO_EMBEDS: embeds})

    def get_input_sequence_len(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> InputSeqLenInfo | None:
        if graph_walk in DECODE_WALKS:
            return InputSeqLenInfo(1)
        if "text_inputs" in inputs:
            return InputSeqLenInfo(inputs["text_inputs"][0].shape[0])
        return None

    def speculative_input_seq_len(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, prev_seq_len: int,
    ) -> int | None:
        if graph_walk in DECODE_WALKS:
            return 1
        return super().speculative_input_seq_len(graph_walk, fwd_info, prev_seq_len)

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        if any(inp.input_ids is None for inp in inputs):
            # a text row that looped back brings no token: read by slot, beside the rows
            # that bring theirs (a spoken row, or a first decode step)
            ids = self.loopback_inputs(engine_inputs, inputs)["input_ids"]
        else:
            ids = torch.cat([inp.input_ids for inp in inputs]).to(device)
        out: dict[str, Any] = {"input_ids": ids}
        # in row order, as the splice's cumulative placeholder count reads them
        rows = [inp.tensor_inputs[AUDIO_EMBEDS] for inp in inputs if AUDIO_EMBEDS in inp.tensor_inputs]
        if rows:
            out[AUDIO_EMBEDS] = torch.cat(rows).to(device)
        return out

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs], **kwargs,
    ) -> SubmoduleStep:
        # The reference generates from the prompt's ids (placeholders included), so its
        # repetition penalty counts them, as HF's does.
        prompt = {
            rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
        } if graph_walk in PREFILL_WALKS else {}
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                LLM_KV: KVStep(),
                LLM_ATTN: AttentionStep(causal=True),
                LLM_POS: PositionStep(),
                LLM_SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prompt),
            },
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        audio_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn: AttentionManager = engine_inputs.resources[LLM_ATTN]
        sampler: SamplerResource = engine_inputs.resources[LLM_SAMPLER]
        embeds = self.model.embed(input_ids)
        if audio_embeds is not None:
            embeds = splice_media(embeds, input_ids == AUDIO_PATCH, audio_embeds)
        hidden = self.model(embeds, label="main")
        if graph_walk not in DECODE_WALKS:
            hidden = attn.select_last_hidden(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=self.model.logits(hidden))

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        audio_embeds: torch.Tensor | None = None, **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._forward(graph_walk, engine_inputs, input_ids, audio_embeds)]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor,
        audio_embeds: torch.Tensor | None = None, **kwargs,
    ) -> BatchedModelOutput:
        new_token = self._forward(graph_walk, engine_inputs, input_ids, audio_embeds)
        return BatchedModelOutput(row_outputs={"new_token": new_token}, check_stop_buffers={"new_token": new_token})

    # ------------------------------------------------------------------
    # Post-step
    # ------------------------------------------------------------------

    def postprocess(
        self, request_id: str, request_info: CurrentForwardPassInfo, outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Rebind, not copy: the decode loop routes on ``text_inputs``, and a spoken
        # reply's stream on ``speech_ids`` (the stop check withholds it unless speech).
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]
            if request_info.graph_walk == DECODE_SPOKEN:  # the row's own walk, in a mixed step too
                outputs[SPEECH_IDS] = outputs["new_token"]


class AudioEncoderSubmodule(NodeSubmodule):
    """A request's 25 s clips, each a key-prefix segment of the tower's attention; requests
    pack too, each clip laid out against its own request's longest (``request_layout``).

    Eager: every clip has its own length. The block loop is shape-only in the frame
    count, so it can become a piecewise region bucketed by frames, as MiniCPM-o's is.
    """

    disable_torch_compile = True

    def __init__(self, model: StepAudio2Audio, config: StepAudio2Config):
        super().__init__()
        self.model = model
        self.config = config

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs: Any,
    ) -> NodeInputs:
        mel_frames = list(fwd_info.step_metadata[AUDIO_MEL_FRAMES])
        layout = request_layout(mel_frames)
        return NodeInputs(
            input_seq_len=sum(c.frames for c in layout),
            tensor_inputs={"mels": inputs[AUDIO_MELS][0]},
            kwargs={"mel_frames": mel_frames, "layout": layout},
        )

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs], **kwargs,
    ) -> SubmoduleStep:
        clips = [(rid, c) for rid, inp in zip(request_ids, inputs, strict=True) for c in inp.kwargs["layout"]]
        return SubmoduleStep(steps={AUDIO_ATTN: KeyPrefixAttentionStep(
            segments=tuple(Segment(request_id=rid, label="main", span=c.frames) for rid, c in clips),
            key_lens=tuple(c.key_len for _, c in clips),
        )})

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        mels: list[torch.Tensor] = []
        for inp in inputs:
            mels += list(inp.tensor_inputs["mels"].to(device).split(inp.kwargs["mel_frames"], dim=-1))
        return {
            "mels": mels,
            "layout": [c for inp in inputs for c in inp.kwargs["layout"]],
            "tokens_per_request": [sum(c.tokens for c in inp.kwargs["layout"]) for inp in inputs],
        }

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, mels: list[torch.Tensor],
        layout: list, tokens_per_request: list[int], **kwargs,
    ) -> dict[str, NameToTensorList]:
        embeds = self.model(mels, layout)
        return {
            rid: {AUDIO_EMBEDS: [rows]}
            for rid, rows in zip(engine_inputs.request_ids, embeds.split(tokens_per_request), strict=True)
        }
