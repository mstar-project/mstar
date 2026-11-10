"""Higgs-Audio STT's node submodules.

- ``HiggsAudioEncoderSubmodule``: the checkpoint's Whisper-style audio tower
  and its MLP projector, once per request, stateless.
- ``HiggsAudioLLMSubmodule``: the dense Qwen3 LLM, on the shared
  `SampledTokenDecoder`; the prompt prefills as one row with the encoder's
  output spliced in at its audio pads.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.higgs_audio.config import ATTN, HiggsAudioModelConfig
from mstar.model.keys import KV_CACHE
from mstar.model.multimodal import MEDIA_LAYOUT, media_grids
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking


class HiggsAudioEncoderSubmodule(NodeSubmodule):
    """Audio tower + projector.

    Consumes per-chunk mel spectrograms from ``process_prompt``
    (``(num_chunks, num_mel_bins, T)`` padded to the longest chunk, plus
    per-chunk mel frame counts) and emits the concatenated LLM-space
    audio embeddings ``(total_audio_tokens, hidden_size)``.

    Mirrors the reference pipeline: batch-encode padded chunks, run the
    projector on the padded batch, then slice each chunk to its valid
    (downsampled) length and concatenate in order. Chunks attend only
    within themselves, so requests whose chunks share a width are encoded
    as one batch.

    The chunk count and, for a clip under one chunk, the mel width change
    per request; compiled, every new shape recompiled on the serving path
    (tens of seconds with the worker blocked). So the tower runs eagerly,
    and full-width chunks replay a CUDA graph by chunk count. A clip under
    one chunk is padded to the full width and replays a second graph whose
    tower masks the padding out (``HiggsAudioTower.forward``'s
    ``valid_frames``), so short clips batch together too.
    """

    disable_torch_compile = True
    # requests per encoder step, and full-width chunks per captured replay
    MAX_BATCH_SIZE = 16
    CAPTURE_CHUNK_COUNTS = [1, 2, 4, 8, 16, 32, 64]
    TOWER = "tower"
    MASKED_TOWER = "tower_masked"

    def __init__(
        self,
        audio_tower: nn.Module,
        projector: nn.Module,
        config: HiggsAudioModelConfig,
        graphs: bool = True,
    ):
        super().__init__()
        self.audio_tower = audio_tower
        self.projector = projector
        self.config = config
        self.graphs = graphs
        # mel frames of a full chunk: the feature extractor's 10 ms hop
        self.chunk_frames = int(config.chunk_size_seconds * 100)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        features = inputs["audio_features"][0]
        if features.dim() == 2:
            features = features.unsqueeze(0)  # (1, num_mel_bins, T)
        # each chunk's valid mel frames, known on the host since process_prompt:
        # the clip's grid in the walk's media layout (one clip a request)
        (frames,) = media_grids(fwd_info.step_metadata[MEDIA_LAYOUT], "audio_embeds")
        frames = list(frames)
        return NodeInputs(
            tensor_inputs={"audio_features": features},
            kwargs={
                "frames": frames,
                "valid_lengths": [self.config.encoder_output_length(n) for n in frames],
            },
            input_seq_len=features.shape[0],
        )

    def _padded(self, width: int) -> bool:
        """A clip under one chunk: run at the full width, its tail masked,
        captured or not."""
        return width < self.chunk_frames

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        # chunks are padded to their own request's longest, so only requests of
        # one mel width share a batch -- or, padded and masked, short clips
        widths = {inp.tensor_inputs["audio_features"].shape[-1] for inp in model_inputs}
        chunks = sum(inp.input_seq_len for inp in model_inputs)
        same_kind = len(widths) == 1 or all(self._padded(w) for w in widths)
        return same_kind and chunks <= self.CAPTURE_CHUNK_COUNTS[-1]

    def max_batch_size(self, graph_walk: str) -> int:
        return self.MAX_BATCH_SIZE

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        dtype = next(self.audio_tower.parameters()).dtype
        features = [inp.tensor_inputs["audio_features"] for inp in inputs]
        widths = [f.shape[-1] for f in features]
        padded = self._padded(widths[0])
        if padded:
            features = [F.pad(f, (0, self.chunk_frames - f.shape[-1])) for f in features]
        out = {
            "audio_features": torch.cat(features, dim=0).to(device=self.get_device(), dtype=dtype),
            "valid_lengths": [inp.kwargs["valid_lengths"] for inp in inputs],
        }
        if padded:
            # a short clip is one chunk; its mask sum can be one past its mel
            # width, and the slice below counts from the mask sum too
            out["valid_frames"] = to_device_nonblocking(
                [inp.kwargs["frames"][0] for inp in inputs], self.get_device(), dtype=torch.long,
            )
        return out

    def _encode(self, features: torch.Tensor, valid_frames: torch.Tensor | None = None) -> torch.Tensor:
        return self.projector(self.audio_tower(features, valid_frames))

    def _capture_tower(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        return {"projected": self._encode(inp.static_inputs["audio_features"])}

    def _capture_masked_tower(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        s = inp.static_inputs
        # padding rows replay zeroed: give them one valid frame
        return {"projected": self._encode(s["audio_features"], s["valid_frames"].clamp(min=1))}

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs: Any,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """Tower + projector over full-width chunks, by chunk count."""
        del autocast_dtype, tp_world_size, kwargs
        if not self.graphs:
            return {}
        dtype = next(self.audio_tower.parameters()).dtype
        mels, frames = self.config.audio_num_mel_bins, self.chunk_frames
        return {
            self.TOWER: PiecewiseBatchedConfig(
                capture_fn=self._capture_tower,
                make_static_inputs=lambda shape: {
                    "audio_features": torch.zeros(shape.bs, mels, frames, dtype=dtype, device=device),
                },
                seq_len=frames,
                capture_batch_sizes=list(self.CAPTURE_CHUNK_COUNTS),
            ),
            # clips under one chunk, one per row, padded to the full width
            self.MASKED_TOWER: PiecewiseBatchedConfig(
                capture_fn=self._capture_masked_tower,
                make_static_inputs=lambda shape: {
                    "audio_features": torch.zeros(shape.bs, mels, frames, dtype=dtype, device=device),
                    "valid_frames": torch.full((shape.bs,), frames, dtype=torch.long, device=device),
                },
                seq_len=frames,
                capture_batch_sizes=[bs for bs in self.CAPTURE_CHUNK_COUNTS if bs <= self.MAX_BATCH_SIZE],
            ),
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        valid_lengths: list[list[int]],
        valid_frames: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        num_chunks = audio_features.shape[0]
        static_inputs = {"audio_features": audio_features}
        if valid_frames is not None:
            static_inputs["valid_frames"] = valid_frames
        runner = engine_inputs.piecewise_runners.get(self.TOWER if valid_frames is None else self.MASKED_TOWER)
        if (
            runner is not None
            and audio_features.shape[-1] == self.chunk_frames
            and runner.can_run(num_chunks)
        ):
            projected = runner.run(static_inputs=static_inputs, real_bs=num_chunks).get_view("projected")
        else:
            projected = self._encode(audio_features, valid_frames)

        outputs: dict[str, NameToTensorList] = {}
        chunk = 0
        for rid, valid in zip(engine_inputs.request_ids, valid_lengths, strict=True):
            embeds = torch.cat([projected[chunk + i, :n] for i, n in enumerate(valid)], dim=0)
            chunk += len(valid)
            outputs[rid] = {"audio_embeds": [embeds]}
        return outputs

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[rid]


class HiggsAudioLLMSubmodule(SampledTokenDecoder):
    ATTN_KEY = ATTN
    # the ladders higgs captured before moving onto the scaffold
    PREFILL_MEDIA_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048]
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def capture_compile_mode(self, graph_walk: str) -> str | None:
        # max-autotune until the default compile level (mstar serve -O) is measured on this model
        return "max-autotune-no-cudagraphs"

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}
