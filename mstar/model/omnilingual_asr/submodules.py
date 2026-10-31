"""The Omnilingual ASR CTC node: waveform -> per-frame CTC token ids.

One stateless forward per request: the wav2vec 2.0 encoder, the CTC head and
a per-frame argmax over each request's valid frames, on the device. The node
emits one id per valid frame; ``OmnilingualASRModel.postprocess`` collapses
repeated frames and drops the blank and the other control ids when it turns
ids into text, as the reference's decoder does. Collapsing there, on the CPU
copy, keeps the step free of the device-to-host read a variable-length
result would need. Row and frame counts come from the host-side lengths, so
the step never waits on the GPU.

Concurrent requests are batched (up to ``max_batch_size``). Inside a step
the rows are sorted by length and cut into sub-batches whose padded audio
(rows times the longest) stays within ``max_batch_seconds``; each sub-batch
is right-padded, with each row's length masking its padding (see
``components/wav2vec2.py``). Sorting keeps a short clip from being padded to
a long one, and the budget bounds a forward's memory.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.step import Segment, SlotLease, SubmoduleStep
from mstar.model.omnilingual_asr.components.wav2vec2 import Wav2Vec2CTC
from mstar.model.omnilingual_asr.config import ENCODER_ATTN, SAMPLE_RATE, TOKEN_IDS, WAVEFORM
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

# The packed encoder (layers, final norm, CTC head, argmax) as a piecewise
# capture region
ENCODER_LOOP = "encoder_loop"


class OmniASRCTCSubmodule(NodeSubmodule):
    # Every request has its own length, so the engine's compile of the whole
    # forward would recompile per shape; the encoder runs eagerly. The conv
    # feature extractor alone is compiled (``compile_regions``).
    disable_torch_compile = True
    # The model runs in its own dtype end to end, as the reference does (only
    # the conv stack's LayerNorms upcast to fp32), not under autocast.
    disable_autocast = True

    # Frames of one packed encoder replay (50 a second); a step past the top
    # bucket runs the encoder eagerly. 16384 frames is 320 s, the default
    # `max_batch_seconds`.
    ENCODER_FRAME_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    def __init__(self, model: Wav2Vec2CTC, max_batch_size: int, max_batch_seconds: float):
        super().__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        if self.dtype == torch.float32:
            # an fp32 deployment computes what the reference does: full fp32
            # matmuls, not the engine's TF32 default (the engine applies it)
            self.float32_matmul_precision = "highest"
        self._max_batch_size = max_batch_size
        self._max_batch_samples = int(max_batch_seconds * SAMPLE_RATE)
        self._max_batch_frames = model.frame_lengths([self._max_batch_samples])[0]

    def compile_regions(self) -> dict[str, list[tuple]]:
        """Eager, each conv layer's channel LayerNorm (in fp32, over a transposed
        copy) costs more than the conv: about half the GPU time of a batched
        step, and inductor fuses the transposes, the norm and the GELU. Warmed
        at start-up for a lone clip (size-1 dims specialize) and the dynamic
        graph."""
        device = self.get_device()
        return {"model.encoder_frontend.feature_extractor": [
            (torch.zeros(rows, SAMPLE_RATE, dtype=self.dtype, device=device),) for rows in (1, 2)
        ]}

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        del graph_walk, fwd_info, kwargs
        waveform = inputs[WAVEFORM][0]
        # in encoder frames, what the captured encoder is leased by
        frames = self.model.frame_lengths([int(waveform.shape[-1])])[0]
        return NodeInputs(tensor_inputs={WAVEFORM: waveform}, input_seq_len=frames)

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs: Any,
    ) -> SubmoduleStep | None:
        """One attention segment per request, its valid frames, when the
        encoder attends packed (see `get_node_resources`). Leased, the captured
        encoder plans its own."""
        if (
            ENCODER_ATTN not in self.node_resources
            or (piecewise_leases or {}).get(ENCODER_LOOP)
            or not self._packs([inp.input_seq_len for inp in inputs])
        ):
            return None
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={ENCODER_ATTN: AttentionStep(causal=False)},
        )

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs: Any,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The packed encoder, one replay a step: every request's frames are a
        segment of one row (`one_row_per_step`). Eager, the encoder is a few
        hundred small launches, which bounds a lone clip's latency."""
        if ENCODER_ATTN not in self.node_resources:
            return {}
        model_dim = self.model.config.model_dim

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {"hidden": torch.zeros(shape.total_tokens, model_dim, dtype=self.dtype, device=device)}

        return {
            ENCODER_LOOP: PiecewisePackedConfig(
                capture_fn=self._capture_encoder,
                make_static_inputs=make_static_inputs,
                attn_key=ENCODER_ATTN,
                lease_before_step=True,
                can_lease=self._fits_capture,
                one_row_per_step=True,
                total_tokens=self.ENCODER_FRAME_BUCKETS,
                capture_batch_sizes=[1],
            )
        }

    def _capture_encoder(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The captured region: reads the runner's buffer, never rebinds."""
        attn = functools.partial(inp.resources[ENCODER_ATTN].run, label="main")
        return {"ids": self.model.encode_packed(inp.static_inputs["hidden"], attn).argmax(dim=-1)}

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self._max_batch_size

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        """Any set batches: ``forward_batched`` splits it to fit the audio budget."""
        del batch
        return len(model_inputs) > 1

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        del graph_walk, engine_inputs
        device = self.get_device()
        return {"waveforms": [inp.tensor_inputs[WAVEFORM].reshape(-1).to(device, self.dtype) for inp in inputs]}

    def sub_batches(self, lengths: list[int]) -> list[list[int]]:
        """Row indices, shortest first, cut so that each group's row count times
        its longest row stays within the audio budget (a row alone always fits)."""
        groups: list[list[int]] = []
        for row in sorted(range(len(lengths)), key=lengths.__getitem__):
            if groups and (len(groups[-1]) + 1) * lengths[row] <= self._max_batch_samples:
                groups[-1].append(row)
            else:
                groups.append([row])
        return groups

    def _padded_groups(self, waveforms: list[torch.Tensor]):
        """Each sub-batch as ``(rows, batch, lengths)``: its waveforms
        zero-padded to the longest, and their lengths (None when none is
        padded)."""
        lengths = [w.shape[0] for w in waveforms]
        for group in self.sub_batches(lengths):
            group_lens = [lengths[row] for row in group]
            longest = max(group_lens)
            batch = waveforms[group[0]].new_zeros(len(group), longest)
            for i, row in enumerate(group):
                batch[i, : lengths[row]] = waveforms[row]
            yield group, batch, group_lens if any(n < longest for n in group_lens) else None

    def _fits_capture(self, inputs: list[NodeInputs]) -> bool:
        """Whether a step replays the captured encoder: it packs, and its rows,
        one segment each under the replay's one row, fit the attention
        resource's ``max_segments_per_step`` (read from it, so a YAML override
        holds). A step over the ceiling encodes packed eagerly."""
        return len(inputs) <= self.node_resources[ENCODER_ATTN].max_segments_per_step and self._packs(
            [inp.input_seq_len for inp in inputs]
        )

    def _packs(self, frames: list[int]) -> bool:
        """Whether a step's rows encode packed, in one call: their frames fit
        the audio budget. A step over it encodes padded per sub-batch instead,
        as the fp32 path does; packing each sub-batch would need an attention
        plan per call."""
        return sum(frames) <= self._max_batch_frames

    def _transcribe_packed(
        self, waveforms: list[torch.Tensor], attn, engine_inputs: ModelInputsFromEngine,
    ) -> list[torch.Tensor]:
        """Argmax ids per row: the conv frontend runs padded per sub-batch, then
        every row's valid frames go through the encoder packed, in request order
        (the order `declare_step` laid the segments out in)."""
        rows: list[torch.Tensor | None] = [None] * len(waveforms)
        for group, batch, group_lens in self._padded_groups(waveforms):
            x, frame_lens, _ = self.model.frontend(batch, group_lens)
            for i, (row, n) in enumerate(zip(group, frame_lens, strict=True)):
                rows[row] = x[i, :n]
        packed = rows[0] if len(rows) == 1 else torch.cat(rows)
        frames = [r.shape[0] for r in rows]
        runner = engine_inputs.piecewise_runners.get(ENCODER_LOOP)
        if runner is not None:
            # all of the step's segments under one row (`one_row_per_step`);
            # a copy, since the output buffer is overwritten by the next
            # step's replay
            ids = runner.run(
                static_inputs={"hidden": packed},
                request_ids=engine_inputs.request_ids,
                seq_lens=frames,
            ).get("ids")
        else:
            ids = self.model.encode_packed(packed, functools.partial(attn.run, label="main")).argmax(dim=-1)
        return list(ids.split(frames))

    def _transcribe(
        self, waveforms: list[torch.Tensor], engine_inputs: ModelInputsFromEngine | None = None,
    ) -> list[torch.Tensor]:
        """Argmax ids per row, over the row's valid frames."""
        attn = getattr(engine_inputs, "resources", {}).get(ENCODER_ATTN)
        lengths = [w.shape[0] for w in waveforms]
        if attn is not None and self._packs(self.model.frame_lengths(lengths)):
            return self._transcribe_packed(waveforms, attn, engine_inputs)
        out: list[torch.Tensor | None] = [None] * len(waveforms)
        for group, batch, group_lens in self._padded_groups(waveforms):
            logits, frame_lens = self.model(batch, group_lens)
            ids = logits.argmax(dim=-1)
            for i, (row, n) in enumerate(zip(group, frame_lens, strict=True)):
                out[row] = ids[i, :n]
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveforms: list[torch.Tensor],
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, kwargs
        return {TOKEN_IDS: [self._transcribe(waveforms, engine_inputs)[0]]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveforms: list[torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        rows = self._transcribe(waveforms, engine_inputs)
        return {rid: {TOKEN_IDS: [ids]} for rid, ids in zip(engine_inputs.request_ids, rows, strict=True)}
