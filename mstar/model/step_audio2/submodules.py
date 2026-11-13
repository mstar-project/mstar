"""Step-Audio2's node submodules: the audio tower and the Qwen2 LLM.

The LLM's prompt is the whole rendered chat, ``<audio_patch>`` placeholders included; an
audio prefill's preprocess splices the tower's rows in at those placeholders, in prompt
order. RoPE is plain 1D over the sequence, so the position resource advances by the token
count.

A spoken reply decodes on its own walk, whose sampled ids stream to the vocoder. The
stream carries only speech ids: the ``TokenStop`` withholds the stream's item on a spoken
row whose id is anything else (text, ``<tts_pad>``, a stop id), off the host copy the stop
check already has, so no forward reads its own tokens.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL, CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.state import ArrayStep, SampleStep, at_last_token, kernel_setting, last_sampled, sample
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.components.sampled_token_decoder import splice_row_ids
from mstar.model.multimodal import splice_media
from mstar.model.step_audio2.components.audio import StepAudio2Audio, request_layout
from mstar.model.step_audio2.config import (
    AUDIO_ATTN,
    AUDIO_EMBEDS,
    AUDIO_MEL_FRAMES,
    AUDIO_MELS,
    AUDIO_PATCH,
    AUDIO_TOKENS,
    DECODE,
    DECODE_LOOP,
    DECODE_MIXED,
    DECODE_SPOKEN,
    DECODE_WALKS,
    LLM_KV,
    LLM_SAMPLER,
    PREFILL_AUDIO,
    PREFILL_TEXT,
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

    Captured as MiniCPM-o's LLM is: one batched decode graph per batch size, which every
    decode walk replays, and one packed prefill graph per (batch size, token bucket), which
    both prefill walks replay. A prefill's audio rows are spliced into its embeddings in
    ``preprocess``, so the prefill forward takes embeddings whatever the walk; a decode
    forward takes ids and embeds them in the graph.
    """

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    # the serving config's max_concurrent_requests; a wider step runs eagerly
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    # Off for the eager path only (a shape no capture holds); the captures compile.
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
        # Both decodes read their last token off the sampler's slot masters. The client
        # token rides inline, and so does a spoken reply's speech id, to the vocoder by
        # value: one column off the stop check's host copy, so the step stores nothing.
        self.device_loopback = DeviceLoopback(
            walk=frozenset({DECODE, DECODE_SPOKEN}), signal="text_inputs", sampler=LLM_SAMPLER, buffer="new_token",
        )

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv = handles[LLM_KV]
        self.sampler = handles[LLM_SAMPLER]

    def resident_walks(self) -> frozenset[str]:
        # the decode preprocess reads only the token (RoPE comes off the position
        # resource inside each layer), so the captured step can run off resident tables
        return frozenset({DECODE, DECODE_SPOKEN})

    def inline_signals(self, graph_walk: str) -> dict[str, str]:
        if graph_walk == DECODE_SPOKEN:
            return {SPEECH_IDS: self.device_loopback.buffer, self.device_loopback.signal: self.device_loopback.buffer}
        return super().inline_signals(graph_walk)

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        def dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n)

        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=DECODE,
                replay_graph_walks=[DECODE, DECODE_SPOKEN, DECODE_MIXED],
                single_request_inputs=dummy(1),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk=PREFILL_TEXT,
                replay_graph_walks=[PREFILL_AUDIO],
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=dummy,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

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
            # a row that looped back brings no token: read by slot, beside the rows that
            # bring theirs (a first decode step)
            ids = splice_row_ids(last_sampled(self.sampler), inputs, decode=True)
        else:
            ids = torch.cat([inp.input_ids for inp in inputs]).to(device)
        if graph_walk in DECODE_WALKS:
            return {"input_ids": ids}
        # Both prefill walks replay one capture, which takes embeddings: the audio rows
        # are spliced in here, in row order, as the cumulative placeholder count reads them.
        embeds = self.model.embed(ids)
        rows = [inp.tensor_inputs[AUDIO_EMBEDS] for inp in inputs if AUDIO_EMBEDS in inp.tensor_inputs]
        if rows:
            embeds = splice_media(embeds, ids == AUDIO_PATCH, torch.cat(rows).to(device))
        return {"input_embeds": embeds}

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
            state={LLM_KV: ArrayStep(), LLM_SAMPLER: SampleStep(prompt_tokens=prompt)},
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """A decode step's ``input_ids``, or a prefill's ``input_embeds`` (``preprocess``)."""
        embeds = self.model.embed(input_ids) if input_embeds is None else input_embeds
        hidden = self.model(embeds, label="main")
        if graph_walk not in DECODE_WALKS:
            hidden = at_last_token(self.kv, hidden)
        return sample(self.sampler, self.model.logits(hidden))

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None, **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._forward(graph_walk, engine_inputs, input_ids, input_embeds)]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None, **kwargs,
    ) -> BatchedModelOutput:
        new_token = self._forward(graph_walk, engine_inputs, input_ids, input_embeds)
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


# The Whisper block loop, as a piecewise capture region
AUDIO_BLOCK_LOOP = "audio_block_loop"


class AudioEncoderSubmodule(NodeSubmodule):
    """A request's 25 s clips, each a key-prefix segment of the tower's attention, laid out
    against the request's longest (``request_layout``).

    The convolutions, the pool and the adapter run eagerly around a captured block loop
    (32 layers), one request per replay, bucketed by its total frames: eager, the tower is
    launch-bound. A request past the largest bucket, or with more clips than a replay's
    plan holds, runs the block loop eagerly, still one request a step.
    """

    # 50 post-conv frames a second, at most 1249 per 25 s clip: 5 s up to two full clips
    BLOCK_LOOP_FRAME_BUCKETS = [256, 512, 1024, 1280, 2560]
    # The eager convolutions run at widths padded to a multiple of this (``_padded`` in
    # ``components/audio.py``): about 20 widths a 25 s clip, rather than one per length.
    CONV_WIDTH_QUANTUM = 128

    # Every clip has its own length, which a compiled forward would recompile for; the
    # captured block loop is what makes it fast.
    disable_torch_compile = True

    def __init__(self, model: StepAudio2Audio, config: StepAudio2Config):
        super().__init__()
        self.model = model
        self.config = config
        # Scratch: the replayed request's key lengths, set just before the block loop's replay, which declares its
        # step from them (``_block_loop_step``) inside the same call.
        self._replay_key_lens: tuple[int, ...] | None = None

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.tokens = handles[AUDIO_TOKENS]

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
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs],
        piecewise_leases: Mapping[str, SlotLease] | None = None, **kwargs,
    ) -> SubmoduleStep | None:
        # leased, the captured block loop plans its own attention per replay
        if (piecewise_leases or {}).get(AUDIO_BLOCK_LOOP):
            return None
        clips = [(rid, c) for rid, inp in zip(request_ids, inputs, strict=True) for c in inp.kwargs["layout"]]
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=c.frames) for rid, c in clips],
            state={AUDIO_TOKENS: ArrayStep(prefix_lens=tuple(c.key_len for _, c in clips))},
        )

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int:
        # one request a step, so it can take the captured block loop
        return 1

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        d_model = self.config.audio.n_audio_state

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {"hidden": torch.zeros(shape.total_tokens, d_model, dtype=autocast_dtype, device=device)}

        return {
            AUDIO_BLOCK_LOOP: PiecewisePackedConfig(
                capture_fn=lambda inp: {"hidden": self.model.encoder.encode(inp.static_inputs["hidden"])},
                make_static_inputs=make_static_inputs,
                declare_step=self._block_loop_step,
                lease_before_step=True,
                can_lease=self._fits_block_loop,
                total_tokens=self.BLOCK_LOOP_FRAME_BUCKETS,
                capture_batch_sizes=[1],
            )
        }

    def _block_loop_step(self, request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
        """A replay's step: the request's clips, each attending its valid keys. At capture
        the row is the runner's padding, which attends all of its frames."""
        (rid,) = request_ids
        key_lens = self._replay_key_lens if self._replay_key_lens is not None else tuple(seq_lens)
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=n) for n in seq_lens],
            state={AUDIO_TOKENS: ArrayStep(prefix_lens=key_lens)},
        )

    def _fits_block_loop(self, inputs: list[NodeInputs]) -> bool:
        """Whether the request's clips fit a replay's attention plan: the resource's
        per-request ceiling, read from it so a YAML override holds."""
        per_request = kernel_setting(self.tokens, AUDIO_ATTN, "max_segments_per_request")
        return sum(len(inp.kwargs["layout"]) for inp in inputs) <= per_request * len(inputs)

    @torch.compiler.disable
    def _replay_block_loop(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, layout: list,
    ) -> torch.Tensor | None:
        """The block loop replayed for the step's one request, or None if not leased."""
        if AUDIO_BLOCK_LOOP not in engine_inputs.piecewise_leases:
            return None
        self._replay_key_lens = tuple(c.key_len for c in layout)
        try:
            return engine_inputs.piecewise_runners[AUDIO_BLOCK_LOOP].run(
                static_inputs={"hidden": hidden},
                request_ids=[engine_inputs.request_ids[0]],
                seq_lens=[c.frames for c in layout],
                real_bs=1,
            ).get_view("hidden")
        finally:
            self._replay_key_lens = None

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
        encoder = self.model.encoder
        hidden = encoder.frontend(mels, layout, self.CONV_WIDTH_QUANTUM)
        encoded = self._replay_block_loop(engine_inputs, hidden, layout)
        if encoded is None:
            encoded = encoder.encode(hidden)
        embeds = self.model.adapter(encoder.pool(encoded, layout), layout, self.CONV_WIDTH_QUANTUM)
        return {
            rid: {AUDIO_EMBEDS: [rows]}
            for rid, rows in zip(engine_inputs.request_ids, embeds.split(tokens_per_request), strict=True)
        }
