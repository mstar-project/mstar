"""The shared AuT encoder node (Qwen3-ASR, Qwen3-Omni): requests packed into
one forward, every 8 s attention window of every request one segment of the
node's ragged-attention step, and the transformer stack captured by packed
token count.

A model subclasses it for ``prepare_inputs`` only, which turns its request's
audio into ``(num_mel_bins, frames)`` features (``feature_inputs``).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.state import ArrayStep
from mstar.model.components.aut_encoder import AuTEncoder, AuTEncoderConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

WINDOW_LABEL = "main"
# The AuT encoder's transformer stack and projector, as a piecewise capture region
AUT_ENCODE = "aut_encode"


class AuTEncoderSubmodule(NodeSubmodule):
    """Packed AuT encoder, emitting one ``(num_audio_tokens, output_dim)``
    ``audio_embeds`` tensor per request. The runner plans the
    windows' kernel layout before the forward, so nothing is recomputed
    inside it."""

    MAX_BATCH_SIZE = 8
    # The engine's blanket torch.compile of ``forward_batched`` traces the
    # per-request Python around the tensor work and guards on request ids
    # and batch composition, so batches kept recompiling (1-2 s stalls each,
    # measured on Whisper's decoder). The encoder's kernels are the ragged
    # attention and cuBLAS GEMMs, there is nothing for inductor to fuse.
    disable_torch_compile = True

    # Packed token counts the stack is captured at, and batch sizes (up to
    # MAX_BATCH_SIZE). 4096 tokens is 8 requests of ~39 s.
    ENCODE_TOKEN_BUCKETS = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    ENCODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    # The stack's region once declared, whose ``total_tokens`` the engine sets
    # to the serving config's ladder before it captures; None until then.
    _encode_region: PiecewisePackedConfig | None = None

    @classmethod
    def max_windows_per_replay(cls, config: AuTEncoderConfig) -> int:
        """Attention windows per request a captured replay can be handed, to
        size the node's ragged resource: the class's largest bucket's tokens
        make at most this many full windows, and each request adds one
        partial. A step with more takes no lease (`_fits_replay`)."""
        return cls.ENCODE_TOKEN_BUCKETS[-1] // config.window_tokens + 1

    def __init__(self, encoder: AuTEncoder, attn_key: str | None = None, array: str | None = None):
        """``array`` is what the windows' attention reads on a model on the state contract, and ``attn_key`` the
        resource a model still on resources names."""
        super().__init__()
        self.encoder = encoder
        self.attn_key = attn_key
        self.array = array

    def _param_dtype(self) -> torch.dtype:
        return self.encoder.conv2d1.weight.dtype

    def feature_inputs(self, features: torch.Tensor) -> NodeInputs:
        """This node's inputs for one request's ``(num_mel_bins, frames)``
        features, already on the encoder's device and in its dtype."""
        return NodeInputs(
            tensor_inputs={"audio_features": features},
            input_seq_len=self.encoder.config.tokens_for_frames(features.shape[-1]),
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep | None:
        # leased, the captured stack plans its own attention per replay
        if (piecewise_leases or {}).get(AUT_ENCODE):
            return None
        # one segment per attention window, in the packed order the forward
        # lays the requests out in
        segments = [
            Segment(request_id=rid, label=WINDOW_LABEL, span=window)
            for rid, inp in zip(request_ids, inputs, strict=True)
            for window in self.encoder.config.window_lengths(inp.input_seq_len)
        ]
        return self._windows_step(segments)

    def _windows_step(self, segments: list[Segment]) -> SubmoduleStep:
        if self.array is not None:
            return SubmoduleStep(segments=segments, state={self.array: ArrayStep()})
        return SubmoduleStep(segments=segments, steps={self.attn_key: AttentionStep(causal=False)})

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # the clips as they are; the frontend packs their frames
        return {"audio_features": [inp.tensor_inputs["audio_features"] for inp in inputs]}

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The encoder's stack and projector, by batch size and packed token
        count. At ASR clip lengths (13 tokens a second) its GEMMs are a few
        dozen rows, so eager it is launch-bound: a 7 s clip's 24-layer stack
        is ~6.3 ms eager and ~1.9 ms replayed. The conv frontend before it
        runs eagerly. A clip past the largest bucket runs eagerly, alone, as
        does a step with more windows than a replay's attention plan holds."""
        d_model = self.encoder.config.d_model

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {"hidden": torch.zeros(shape.total_tokens, d_model, dtype=autocast_dtype, device=device)}

        def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
            # Every window of every request is a segment; the kernel needs
            # only the boundaries, so they all name the first row's id.
            # Zero-length rows pad the batch to the bucket.
            return self._windows_step(
                [Segment(request_id=request_ids[0], label=WINDOW_LABEL, span=n) for n in seq_lens],
            )

        self._encode_region = PiecewisePackedConfig(
            capture_fn=self._capture_encode,
            make_static_inputs=make_static_inputs,
            declare_step=declare_step,
            lease_before_step=True,
            can_lease=self._fits_replay,
            total_tokens=self.ENCODE_TOKEN_BUCKETS,
            capture_batch_sizes=self.ENCODE_CAPTURE_BATCH_SIZES,
        )
        return {AUT_ENCODE: self._encode_region}

    def max_segments_per_request(self) -> int:
        """The ceiling on windows per row of the attention resource under ``attn_key``."""
        return self.node_resources[self.attn_key].max_segments_per_request

    def _fits_replay(self, inputs: list[NodeInputs]) -> bool:
        """Whether a step's windows fit a replay's attention plan: the resource
        holds ``max_segments_per_request`` a row, and the replay pads to at
        least as many rows as the step has. Read from the resource, so a YAML
        override holds."""
        windows = sum(len(self.encoder.config.window_lengths(inp.input_seq_len)) for inp in inputs)
        return windows <= len(inputs) * self.max_segments_per_request()

    def _capture_encode(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The captured region: reads the runner's buffer, never rebinds."""
        return {"embeds": self.encoder.encode(inp.static_inputs["hidden"], None)}

    @torch.compiler.disable
    def _replay_encode(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, windows: list[int],
    ) -> torch.Tensor | None:
        """The stack replayed for the whole packed batch, or None if not leased."""
        if AUT_ENCODE not in engine_inputs.piecewise_leases:
            return None
        bs = len(engine_inputs.request_ids)
        # an owned copy: the encoder states leave this worker after the step,
        # and the buffer is the next replay's
        return engine_inputs.piecewise_runners[AUT_ENCODE].run(
            static_inputs={"hidden": hidden},
            request_ids=list(engine_inputs.request_ids),
            seq_lens=windows,
            real_bs=bs,
        ).get("embeds")

    def _encode(
        self, audio_features: list[torch.Tensor],
        engine_inputs: ModelInputsFromEngine | None = None,
    ) -> list[torch.Tensor]:
        enc = self.encoder
        layout = enc.layout([clip.shape[-1] for clip in audio_features])
        hidden = enc.frontend(audio_features)
        embeds = None
        if engine_inputs is not None:
            embeds = self._replay_encode(engine_inputs, hidden, layout.window_lengths)
        if embeds is None:
            embeds = enc.encode(hidden, layout.cu_seqlens)
        return list(embeds.split(layout.tokens_per_request, dim=0))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: list[torch.Tensor],
        **kwargs,
    ) -> NameToTensorList:
        return {"audio_embeds": [self._encode(audio_features, engine_inputs)[0]]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        # A packed step holds at most the largest capture's tokens (of the
        # ladder the region takes) and the windows a replay's attention plan
        # holds. Past either the clips run one by one, each replayed if it
        # fits, so an eager step's activations are one clip's rather than the
        # whole batch's.
        buckets = self.ENCODE_TOKEN_BUCKETS if self._encode_region is None else self._encode_region.total_tokens
        tokens = sum(inp.input_seq_len for inp in model_inputs)
        return len(model_inputs) <= 1 or (tokens <= max(buckets) and self._fits_replay(model_inputs))

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: list[torch.Tensor],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        per_request = self._encode(audio_features, engine_inputs)
        return {
            rid: {"audio_embeds": [emb]}
            for rid, emb in zip(engine_inputs.request_ids, per_request, strict=True)
        }
