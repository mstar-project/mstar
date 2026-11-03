"""The shared FastConformer encoder node: log-mel features and the encoder
over a batch of clips, for every FastConformer ASR model."""
from __future__ import annotations

import math
from functools import partial
from typing import Any

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
from mstar.model.components.audio_features import NemoLogMel
from mstar.model.components.fastconformer import FastConformerEncoder
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking

# Mel-frame lengths captured as CUDA graphs (2.56 s to 40.96 s of audio): the
# encoder is many small kernels, which cost more to launch than to run on short clips
MEL_BUCKETS = (256, 512, 1024, 2048, 4096)
CAPTURE_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
# mel frames per captured call, which bounds a capture's activation memory
MAX_BATCH_FRAMES = 32 * 1024


def _region(bucket: int) -> str:
    return f"encoder_T{bucket}"


def _bucket_for(num_frames: int) -> int | None:
    return next((b for b in MEL_BUCKETS if num_frames <= b), None)


def _batch_sizes(bucket: int) -> list[int]:
    return [bs for bs in CAPTURE_BATCH_SIZES if bs * bucket <= MAX_BATCH_FRAMES] or [1]


def _replay_cost(rows: int, mel: int) -> float:
    """One replay of a ``(rows, mel)`` capture, in units of the smallest
    capture's: a fixed part (the encoder's ~700 kernels, ~2.2 ms on an H100)
    plus work linear in rows x mel frames, and in rows x frames^2 for the
    attention. Fitted to replay times over 78 (rows, mel) shapes on an H100
    (within 11%); planning needs only the ratios."""
    return 1.0 + rows * mel / 6580 + rows * mel * mel / 7.3e7


def _part_sizes(count: int, bucket: int) -> list[int]:
    """The captures ``count`` rows of one group replay as: the widest while
    it fills, then the narrowest that holds the rest."""
    sizes = _batch_sizes(bucket)
    full, rest = divmod(count, sizes[-1])
    return [sizes[-1]] * full + ([next(s for s in sizes if s >= rest)] if rest else [])


def _plan(num_frames: list[int]) -> list[tuple[int | None, list[int]]]:
    """A batch's rows -> ``(bucket, rows)`` groups, one replay series each.

    A group is a run of the rows in length order, padded to its longest row's
    bucket. A replay costs as much as ~6,600 padded rows x mel frames, so short
    clips ride in a longer clip's replay when that is cheaper than a replay of
    their own; the runs are cut where the total cost is least. Rows past every
    bucket are one group, run eagerly."""
    order = sorted(range(len(num_frames)), key=lambda r: num_frames[r])
    bucketed = [r for r in order if _bucket_for(num_frames[r]) is not None]
    best, cut = [0.0] + [math.inf] * len(bucketed), [0] * (len(bucketed) + 1)
    for i in range(1, len(bucketed) + 1):
        bucket = _bucket_for(num_frames[bucketed[i - 1]])
        for j in range(i):
            cost = best[j] + sum(_replay_cost(bs, bucket) for bs in _part_sizes(i - j, bucket))
            if cost < best[i]:
                best[i], cut[i] = cost, j
    groups, i = [], len(bucketed)
    while i > 0:
        groups.append((_bucket_for(num_frames[bucketed[i - 1]]), bucketed[cut[i]:i]))
        i = cut[i]
    eager = [r for r in order if _bucket_for(num_frames[r]) is None]
    return groups[::-1] + ([(None, eager)] if eager else [])


class FastConformerEncoderSubmodule(NodeSubmodule):
    """Log-mel features and the FastConformer encoder over a batch of clips.

    A batch is grouped by length bucket (``_plan``) in ``preprocess``, which
    computes each group's features together, and each group replays that
    bucket's graph, padded to it; the encoder masks the padding, so each request
    gets what it would get alone. A clip past every bucket runs eagerly. Emits
    ``encoder_states`` of shape ``(frames, hidden)`` per request, through
    ``projection`` when the model maps them into an LLM's embedding space.
    """

    MAX_BATCH_SIZE = max(CAPTURE_BATCH_SIZES)
    # the captured regions carry the speed; compiling the eager remainder would
    # recompile per clip length
    disable_torch_compile = True

    def __init__(
        self,
        encoder: FastConformerEncoder,
        normalize_features: bool = True,
        projection: torch.nn.Module | None = None,
        log_mel: NemoLogMel | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.log_mel = log_mel or NemoLogMel(
            num_mel_bins=encoder.config.num_mel_bins,
            normalize=normalize_features,
            device=next(encoder.parameters()).device,
        )

    def _param_dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        # the features are computed for the whole batch in preprocess; the
        # length check stays here, so a clip too short fails only its request
        audio = inputs["audio"][0]
        self.log_mel.check_length(audio.numel())
        return NodeInputs(
            tensor_inputs={"audio": audio},
            input_seq_len=self.encoder.config.output_lengths(self.log_mel.num_frames(audio.numel())),
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        audio = [inp.tensor_inputs["audio"] for inp in inputs]
        groups = _plan([self.log_mel.num_frames(a.numel()) for a in audio])
        # features per encoder group, so one long clip doesn't pad the batch's short ones to its length
        features: list[torch.Tensor | None] = [None] * len(audio)
        for bucket, rows in groups:
            # a clip past every bucket runs eagerly on its own
            parts = [rows] if bucket is not None else [[r] for r in rows]
            for part in parts:
                for r, feats in zip(part, self.log_mel.batch([audio[r] for r in part]), strict=True):
                    features[r] = feats.to(self._param_dtype())
        return {
            "features": features,
            "out_lengths": [inp.input_seq_len for inp in inputs],
            "groups": groups,
        }

    # -- the encoder, captured per bucket or eager ---------------------------

    def _run(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        states, _ = self.encoder(features, lengths)
        return states if self.projection is None else self.projection(states)

    def _capture(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        si = inp.static_inputs
        # padding rows arrive zeroed; one frame keeps their masks non-empty
        return {"states": self._run(si["features"], si["lengths"].clamp(min=1))}

    def _static_inputs(self, shape: PiecewiseCaptureShape, bucket: int, device) -> dict[str, torch.Tensor]:
        mel = self.encoder.config.num_mel_bins
        return {
            "features": torch.zeros(shape.bs, bucket, mel, dtype=self._param_dtype(), device=device),
            "lengths": torch.zeros(shape.bs, dtype=torch.long, device=device),
        }

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs: Any,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """One region per mel bucket; the encoder holds no engine resources."""
        del autocast_dtype, tp_world_size, kwargs
        return {
            _region(bucket): PiecewiseBatchedConfig(
                capture_fn=self._capture,
                make_static_inputs=partial(self._static_inputs, bucket=bucket, device=device),
                seq_len=bucket,
                capture_batch_sizes=_batch_sizes(bucket),
            )
            for bucket in MEL_BUCKETS
        }

    def _encode(
        self, features: list[torch.Tensor], out_lengths: list[int], runners: dict[str, Any],
        groups: list[tuple[int | None, list[int]]] | None = None,
    ) -> list[torch.Tensor]:
        states: list[torch.Tensor | None] = [None] * len(features)
        if groups is None:
            groups = _plan([feats.shape[0] for feats in features])
        for bucket, rows in groups:
            runner = runners.get(_region(bucket)) if bucket is not None else None
            step = max(_batch_sizes(bucket)) if bucket is not None else 1
            for part in (rows[i:i + step] for i in range(0, len(rows), step)):
                lengths = to_device_nonblocking([features[r].shape[0] for r in part], features[part[0]].device)
                if runner is not None and runner.can_run(len(part)):
                    padded = torch.stack([
                        torch.nn.functional.pad(features[r], (0, 0, 0, bucket - features[r].shape[0]))
                        for r in part
                    ])
                    batch_states = runner.run(
                        static_inputs={"features": padded, "lengths": lengths}, real_bs=len(part),
                    )["states"]
                else:
                    batch_states = self._run(pad_sequence([features[r] for r in part], batch_first=True), lengths)
                for i, r in enumerate(part):
                    states[r] = batch_states[i, :out_lengths[r]]
        return states

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        features: list[torch.Tensor],
        out_lengths: list[int],
        groups: list[tuple[int | None, list[int]]] | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {"encoder_states": self._encode(features, out_lengths, engine_inputs.piecewise_runners, groups)[:1]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        features: list[torch.Tensor],
        out_lengths: list[int],
        groups: list[tuple[int | None, list[int]]] | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        states = self._encode(features, out_lengths, engine_inputs.piecewise_runners, groups)
        return {rid: {"encoder_states": [s]} for rid, s in zip(engine_inputs.request_ids, states, strict=True)}
