"""The Kokoro synthesis node: one sentence chunk per loop iteration, batched
across requests, with CUDA graphs per length bucket.

``process_prompt`` hands every request all of its chunks at once (padded
``[n_chunks, T]`` phoneme ids, lengths, one style row per chunk, speed). Each
iteration of the ``chunk_loop`` synthesizes chunk ``i`` for every request in
the batch and emits its PCM to the client; ``check_stop`` ends a request's
loop after its last chunk.

The forward has one host read, the frame count, between two static halves.
Each half is captured as a piecewise CUDA graph per padded length bucket:

* ``text_T<n>``: phonemes -> prosody states, text features, durations;
* ``frames_F<n>``: frame-aligned features -> waveform.

Rows of a batch are grouped by frame bucket before the decoder so a short
sentence is not padded to a long one. A batch larger than the biggest
captured one is cut into captured-size slices; a chunk longer than every
frame bucket runs the same code eagerly, at most ``max_batch_frames`` frames
per call, so no call needs more memory than a capture did.

With ``trim_edge_silence`` each row also returns, as a span on the device,
its PCM without the silence at its request's outer edges; the worker cuts the
row to it once the step has run (``spanned_outputs``).
"""

from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import Any, NamedTuple

import torch
from torch.nn.utils.rnn import pad_sequence

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
)
from mstar.engine.engine import ExecutingBatch
from mstar.model.kokoro.components import KokoroTTS
from mstar.model.kokoro.config import (
    AUDIO_CHUNK,
    BOUNDARY_TOKEN_ID,
    CHUNK_LOOP,
    PHONEME_IDS,
    PHONEME_LENS,
    REF_STYLE,
    SPEED,
    KokoroModelConfig,
)
from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)
# One line per batched step (batch size, text bucket, frame groups, ms) when
# MSTAR_KOKORO_STEP_LOG=1; a diagnostic, off by default (it syncs the device).
step_logger = logging.getLogger(__name__ + ".steps")
if os.environ.get("MSTAR_KOKORO_STEP_LOG"):
    step_logger.setLevel(logging.INFO)
else:
    step_logger.setLevel(logging.WARNING)

PCM16_SCALE = 32767
# NodeInputs.kwargs key: (first chunk of its request, last chunk of its request)
REQUEST_EDGES = "request_edges"


def text_region(bucket: int) -> str:
    return f"text_T{bucket}"


def frame_region(bucket: int) -> str:
    return f"frames_F{bucket}"


def pick_bucket(size: int, buckets: list[int]) -> int | None:
    """Smallest bucket that holds ``size``, or ``None``."""
    return next((b for b in buckets if b >= size), None)


def group_by_bucket(
    sizes: list[int], buckets: list[int], policy: str = "bucket"
) -> list[tuple[int | None, list[int]]]:
    """Row indices grouped by frame bucket, ascending; rows that fit no bucket
    form a final ``(None, rows)`` group.

    ``policy="single"`` puts every row that fits some bucket into the largest
    needed bucket instead: one replay per step at the cost of padding.
    """
    groups: dict[int | None, list[int]] = {}
    for row, size in enumerate(sizes):
        groups.setdefault(pick_bucket(size, buckets), []).append(row)
    if policy == "single":
        fitted = [(bucket, rows) for bucket, rows in groups.items() if bucket is not None]
        if fitted:
            largest = max(bucket for bucket, _ in fitted)
            merged = sorted(row for _, rows in fitted for row in rows)
            groups = {largest: merged, **({None: groups[None]} if None in groups else {})}
    elif policy != "bucket":
        raise ValueError(f"Unknown frame_grouping {policy!r}; use 'bucket' or 'single'")
    return sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))


class Synthesized(NamedTuple):
    """One step's PCM16 chunks, one per row, and with edge trimming the
    ``[B, 2]`` span of each to keep."""

    pcm: list[torch.Tensor]
    spans: torch.Tensor | None


def speech_span(audio: torch.Tensor, threshold: float) -> torch.Tensor:
    """``[B, L]`` waveform -> ``[B, 2]`` index of each row's first and last sample
    above ``threshold``; ``(0, L - 1)`` for a row with none, so it is kept whole."""
    loud = (audio.abs() > threshold).to(torch.uint8)
    first = loud.argmax(dim=1)
    last = audio.shape[1] - 1 - loud.flip(1).argmax(dim=1)
    return torch.stack([first, last], dim=1)


def _pad_time(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    """Zero-pad ``dim`` of ``x`` up to ``size``."""
    if x.shape[dim] == size:
        return x
    shape = list(x.shape)
    shape[dim] = size - x.shape[dim]
    return torch.cat([x, x.new_zeros(shape)], dim=dim)


class KokoroSynthSubmodule(NodeSubmodule):
    """Phonemes + style -> 24 kHz PCM16 for one chunk of every request."""

    # The forward reads the frame count on the host and shapes the decoder by
    # it, so the whole method is not compilable; its two static halves are
    # captured piecewise instead.
    disable_torch_compile = True
    # fp32 end to end, as the reference: the phase-sensitive vocoder does not
    # tolerate reduced precision.
    disable_autocast = True

    def __init__(self, model: KokoroTTS, config: KokoroModelConfig):
        super().__init__()
        self.model = model
        self.config = config
        if config.compile_decoder and next(model.parameters()).is_cuda:
            # Bound-method attribute shadows the class method for both the
            # captured regions and the eager fallback.
            model.decode_frames = torch.compile(model.decode_frames, dynamic=True)

    # -- per request -------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs | None:
        """Select this iteration's chunk; ``None`` once the request has none left."""
        del graph_walk, kwargs
        state = self.request_state(fwd_info.rid_handle)
        if "chunk_lengths" not in state:
            # One host read per request: the chunk lengths size every later slice.
            state.add("chunk_lengths", inputs[PHONEME_LENS][0].tolist())
        chunk_lengths: list[int] = state["chunk_lengths"]
        iteration = fwd_info.dynamic_loop_iter_counts.get(CHUNK_LOOP, 0)
        if iteration >= len(chunk_lengths):
            return None
        length = chunk_lengths[iteration]
        return NodeInputs(
            tensor_inputs={
                "input_ids": inputs[PHONEME_IDS][0][iteration, :length],
                "style": inputs[REF_STYLE][0][iteration],
                "speed": inputs[SPEED][0].reshape(()),
            },
            kwargs={REQUEST_EDGES: (iteration == 0, iteration == len(chunk_lengths) - 1)},
            input_seq_len=length,
        )

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        del outputs
        num_chunks = len(self.request_state(request_id).get("chunk_lengths", ()))
        if request_info.dynamic_loop_iter_counts.get(CHUNK_LOOP, 0) + 1 >= num_chunks:
            return {CHUNK_LOOP}
        return set()

    # -- per batch -----------------------------------------------------------

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        del batch, model_inputs
        return True

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        """Pad the chunks of the batch to a common phoneme length."""
        del graph_walk, engine_inputs
        device = self.get_device()
        per_row = [[inp.input_seq_len for inp in inputs]]
        edges = [inp.kwargs[REQUEST_EDGES] for inp in inputs]
        trim = self.config.trim_edge_silence and any(map(any, edges))
        if trim:
            per_row.extend(zip(*edges, strict=True))
        # one host-to-device copy: the lengths, and the edges when trimming
        per_row = torch.tensor(per_row, dtype=torch.long, device=device)
        return {
            "input_ids": pad_sequence(
                [inp.tensor_inputs["input_ids"].to(device) for inp in inputs],
                batch_first=True,
                padding_value=BOUNDARY_TOKEN_ID,
            ),
            "lengths": per_row[0],
            "style": torch.stack([inp.tensor_inputs["style"].to(device) for inp in inputs]),
            "speed": torch.stack([inp.tensor_inputs["speed"].to(device) for inp in inputs]),
            "edges": per_row[1:] if trim else None,
        }

    def spanned_outputs(self, graph_walk: str) -> frozenset[str]:
        del graph_walk
        return frozenset({AUDIO_CHUNK}) if self.config.trim_edge_silence else frozenset()

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        style: torch.Tensor,
        speed: torch.Tensor,
        edges: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BatchedModelOutput:
        return self.forward_batched(graph_walk, engine_inputs, input_ids, lengths, style, speed, edges, **kwargs)

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        style: torch.Tensor,
        speed: torch.Tensor,
        edges: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BatchedModelOutput:
        del graph_walk, kwargs
        pcm, spans = self._synthesize(input_ids, lengths, style, speed, engine_inputs.piecewise_runners, edges)
        rids = engine_inputs.request_ids
        return BatchedModelOutput(
            per_rid_outputs={rid: {AUDIO_CHUNK: [chunk]} for rid, chunk in zip(rids, pcm, strict=True)},
            output_spans=None if spans is None else {AUDIO_CHUNK: spans},
        )

    # -- the two halves, captured or eager -------------------------------------

    def _synthesize(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        style: torch.Tensor,
        speed: torch.Tensor,
        runners: dict[str, Any],
        edges: torch.Tensor | None = None,
    ) -> Synthesized:
        """One PCM16 chunk per row, sliced to its own length. ``edges`` (``[2, B]``)
        says per row whether it opens and whether it closes its request; with
        it, each row also gets the span that drops the silence at those edges."""
        logging_step = step_logger.isEnabledFor(logging.INFO)
        if logging_step:
            if input_ids.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        d, t_en, pred_dur = self._encode_text(input_ids, lengths, style, speed, runners)
        frame_lengths = pred_dur.sum(dim=1)
        sizes = frame_lengths.tolist()  # the one host read
        pcm: list[torch.Tensor | None] = [None] * len(sizes)
        groups = group_by_bucket(sizes, self.config.frame_buckets, self.config.frame_grouping)
        calls = [(bucket, part) for bucket, rows in groups for part in self._frame_batches(bucket, rows, sizes)]
        # each row's first and last sample above the threshold, in batch order
        loud = None if edges is None else torch.empty(len(sizes), 2, dtype=torch.long, device=input_ids.device)
        for bucket, rows in calls:
            index = torch.tensor(rows, device=input_ids.device)
            runner = runners.get(frame_region(bucket)) if bucket is not None else None
            if runner is not None and runner.can_run(len(rows)):
                en, asr, group_lengths = self.model.align(d[index], t_en[index], pred_dur[index], bucket)
                audio = runner.run(
                    static_inputs={"en": en, "asr": asr, "frame_lengths": group_lengths, "style": style[index]},
                    real_bs=len(rows),
                )["audio"]
            else:
                num_frames = max(sizes[r] for r in rows)
                audio, _ = self.model.synthesize_frames(
                    d[index], t_en[index], pred_dur[index], style[index], num_frames
                )
            if loud is not None:
                loud.index_copy_(0, index, speech_span(audio, self.config.edge_silence_threshold))
            audio = (audio.clamp(-1, 1) * PCM16_SCALE).to(torch.int16)
            for i, row in enumerate(rows):
                pcm[row] = audio[i, : sizes[row] * self.config.samples_per_frame]
        spans = None if loud is None else self._edge_spans(loud, frame_lengths, edges)
        if logging_step:
            if input_ids.is_cuda:
                torch.cuda.synchronize()
            step_logger.info(
                "step bs=%d T=%d frames=%s groups=%s %.1f ms",
                len(sizes),
                input_ids.shape[1],
                sizes,
                [(b, len(r)) for b, r in calls],
                (time.perf_counter() - t0) * 1000,
            )
        return Synthesized(pcm, spans)

    def _edge_spans(self, loud: torch.Tensor, frame_lengths: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """``[B, 2]`` span of each row's PCM: from ``edge_silence_keep_samples``
        before its first loud sample when the row opens its request, to as far
        past its last when it closes it, and the whole row otherwise."""
        keep = self.config.edge_silence_keep_samples
        length = frame_lengths * self.config.samples_per_frame
        opens, closes = edges.bool()
        start = torch.where(opens, (loud[:, 0] - keep).clamp(min=0), 0)
        end = torch.where(closes, torch.minimum(loud[:, 1] + 1 + keep, length), length)
        return torch.stack([torch.minimum(start, end), end], dim=1)

    def frame_capture_sizes(self, bucket: int) -> list[int]:
        """Batch sizes captured for a frame bucket: ``bs * bucket <= max_batch_frames``."""
        cfg = self.config
        return [bs for bs in cfg.capture_batch_sizes if bs * bucket <= cfg.max_batch_frames] or [1]

    def _frame_batches(self, bucket: int | None, rows: list[int], sizes: list[int]) -> list[list[int]]:
        """Split one frame group so no decoder call exceeds what start-up captured.

        A bucketed group is cut at the bucket's largest captured batch; rows
        past every bucket go eager, at most ``max_batch_frames`` frames per call.
        """
        if bucket is not None:
            step = max(self.frame_capture_sizes(bucket))
            return [rows[i : i + step] for i in range(0, len(rows), step)]
        batches: list[list[int]] = []
        for row in sorted(rows, key=lambda r: -sizes[r]):
            # longest first, so a batch's first row sets its padded frame count
            if batches and (len(batches[-1]) + 1) * sizes[batches[-1][0]] <= self.config.max_batch_frames:
                batches[-1].append(row)
            else:
                batches.append([row])
        return batches

    def _encode_text(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        style: torch.Tensor,
        speed: torch.Tensor,
        runners: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The text half, in slices of the largest captured batch so none runs unbounded."""
        step = max(self.config.capture_batch_sizes)
        if input_ids.shape[0] <= step:
            return self._encode_text_rows(input_ids, lengths, style, speed, runners)
        parts = [
            self._encode_text_rows(
                input_ids[i : i + step], lengths[i : i + step], style[i : i + step], speed[i : i + step], runners
            )
            for i in range(0, input_ids.shape[0], step)
        ]
        return tuple(torch.cat(outputs, dim=0) for outputs in zip(*parts, strict=True))

    def _encode_text_rows(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        style: torch.Tensor,
        speed: torch.Tensor,
        runners: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_tokens = input_ids.shape
        bucket = pick_bucket(num_tokens, self.config.text_buckets)
        runner = runners.get(text_region(bucket)) if bucket is not None else None
        if runner is None or not runner.can_run(batch):
            return self.model.encode_text(input_ids, lengths, style, speed)
        out = runner.run(
            static_inputs={
                "input_ids": _pad_time(input_ids, bucket, 1),
                "lengths": lengths,
                "style": style,
                "speed": speed,
            },
            real_bs=batch,
        )
        return out["d"][:, :num_tokens], out["t_en"][:, :, :num_tokens], out["pred_dur"][:, :num_tokens]

    # -- captured regions -------------------------------------------------------

    def _text_capture(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        si = inp.static_inputs
        # Padding rows arrive zeroed: a zero length or speed would divide by
        # zero inside the masks; clamping keeps their (discarded) rows finite.
        d, t_en, pred_dur = self.model.encode_text(
            si["input_ids"], si["lengths"].clamp(min=1), si["style"], si["speed"].clamp(min=self.config.min_speed)
        )
        return {"d": d, "t_en": t_en, "pred_dur": pred_dur}

    def _frames_capture(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        si = inp.static_inputs
        return {"audio": self.model.decode_frames(si["en"], si["asr"], si["frame_lengths"].clamp(min=1), si["style"])}

    def _text_static_inputs(self, shape: PiecewiseCaptureShape, bucket: int, device) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.zeros(shape.bs, bucket, dtype=torch.long, device=device),
            "lengths": torch.zeros(shape.bs, dtype=torch.long, device=device),
            "style": torch.zeros(shape.bs, self.config.style_vector_dim, dtype=torch.float32, device=device),
            "speed": torch.zeros(shape.bs, dtype=torch.float32, device=device),
        }

    def _frame_static_inputs(self, shape: PiecewiseCaptureShape, bucket: int, device) -> dict[str, torch.Tensor]:
        cfg = self.config
        return {
            "en": torch.zeros(shape.bs, bucket, cfg.hidden_dim + cfg.style_dim, dtype=torch.float32, device=device),
            "asr": torch.zeros(shape.bs, cfg.hidden_dim, bucket, dtype=torch.float32, device=device),
            "frame_lengths": torch.zeros(shape.bs, dtype=torch.long, device=device),
            "style": torch.zeros(shape.bs, cfg.style_vector_dim, dtype=torch.float32, device=device),
        }

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs: Any
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """One region per text bucket and per frame bucket. Kokoro holds no
        engine resources, so the regions declare no step."""
        del autocast_dtype, tp_world_size, kwargs
        cfg = self.config
        regions: dict[str, PiecewiseAcceleratorGraphConfig] = {}
        for bucket in cfg.text_buckets:
            regions[text_region(bucket)] = PiecewiseBatchedConfig(
                capture_fn=self._text_capture,
                make_static_inputs=partial(self._text_static_inputs, bucket=bucket, device=device),
                seq_len=bucket,
                capture_batch_sizes=list(cfg.capture_batch_sizes),
            )
        for bucket in cfg.frame_buckets:
            regions[frame_region(bucket)] = PiecewiseBatchedConfig(
                capture_fn=self._frames_capture,
                make_static_inputs=partial(self._frame_static_inputs, bucket=bucket, device=device),
                seq_len=bucket,
                capture_batch_sizes=self.frame_capture_sizes(bucket),
            )
        return regions
