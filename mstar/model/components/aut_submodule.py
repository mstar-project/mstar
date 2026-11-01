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
import torch.nn.functional as F

from mstar.communication.tensors import NameToTensorList
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, Segment, SlotLease, SubmoduleStep
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

    @classmethod
    def max_windows_per_replay(cls, config: AuTEncoderConfig) -> int:
        """Attention windows per request a captured replay can be handed, to
        size the node's ragged resource: the largest bucket's tokens make at
        most this many full windows, and each request adds one partial."""
        return cls.ENCODE_TOKEN_BUCKETS[-1] // config.window_tokens + 1

    def __init__(self, encoder: AuTEncoder, attn_key: str):
        super().__init__()
        self.encoder = encoder
        self.attn_key = attn_key

    def _param_dtype(self) -> torch.dtype:
        return self.encoder.conv2d1.weight.dtype

    def feature_inputs(self, features: torch.Tensor) -> NodeInputs:
        """This node's inputs for one request's ``(num_mel_bins, frames)``
        features, already on the encoder's device and in its dtype."""
        num_frames = int(features.shape[-1])
        return NodeInputs(
            tensor_inputs={"audio_features": features},
            kwargs={"num_frames": num_frames},
            input_seq_len=self.encoder.config.tokens_for_frames(num_frames),
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
        return SubmoduleStep(segments=segments, steps={self.attn_key: AttentionStep(causal=False)})

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        lens = [inp.kwargs["num_frames"] for inp in inputs]
        longest = max(lens)
        padded = torch.stack([
            F.pad(inp.tensor_inputs["audio_features"], (0, longest - n))
            for inp, n in zip(inputs, lens, strict=True)
        ])
        return {"audio_features": padded, "feature_lens": lens}

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The encoder's stack and projector, by batch size and packed token
        count. At ASR clip lengths (13 tokens a second) its GEMMs are a few
        dozen rows, so eager it is launch-bound: a 7 s clip's 24-layer stack
        is ~6.3 ms eager and ~1.9 ms replayed. The conv frontend before it
        runs eagerly. Batches past the largest bucket run eagerly too."""
        d_model = self.encoder.config.d_model
        attn_key = self.attn_key

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {"hidden": torch.zeros(shape.total_tokens, d_model, dtype=autocast_dtype, device=device)}

        def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
            # Every window of every request is a segment; the kernel needs
            # only the boundaries, so they all name the first row's id.
            # Zero-length rows pad the batch to the bucket.
            return SubmoduleStep(
                segments=[Segment(request_id=request_ids[0], label=WINDOW_LABEL, span=n) for n in seq_lens],
                steps={attn_key: AttentionStep(causal=False)},
            )

        return {
            AUT_ENCODE: PiecewisePackedConfig(
                capture_fn=self._capture_encode,
                make_static_inputs=make_static_inputs,
                declare_step=declare_step,
                lease_before_step=True,
                total_tokens=self.ENCODE_TOKEN_BUCKETS,
                capture_batch_sizes=self.ENCODE_CAPTURE_BATCH_SIZES,
            )
        }

    def _capture_encode(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The captured region: reads the runner's buffer, never rebinds."""
        return {"embeds": self.encoder.encode(inp.static_inputs["hidden"], None)}

    @torch.compiler.disable
    def _replay_encode(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor, windows: list[int],
    ) -> torch.Tensor | None:
        """The stack replayed for the whole packed batch, or None if not leased.

        A leased step declares nothing (`declare_step`), so ``engine_inputs.step``
        is None; a bucket that fits is not enough, as ``can_lease`` may have
        refused the batch."""
        runner = engine_inputs.piecewise_runners.get(AUT_ENCODE)
        bs = len(engine_inputs.request_ids)
        if runner is None or engine_inputs.step is not None or not runner.can_run(bs, int(hidden.shape[0])):
            return None
        # an owned copy: the encoder states leave this worker after the step,
        # and the buffer is the next replay's
        return runner.run(
            static_inputs={"hidden": hidden},
            request_ids=list(engine_inputs.request_ids),
            seq_lens=windows,
            real_bs=bs,
        ).get("embeds")

    def _encode(
        self, audio_features: torch.Tensor, feature_lens: list[int],
        engine_inputs: ModelInputsFromEngine | None = None,
    ) -> list[torch.Tensor]:
        enc = self.encoder
        layout = enc.layout(feature_lens)
        hidden = enc.frontend(audio_features, feature_lens)
        embeds = None
        if engine_inputs is not None:
            embeds = self._replay_encode(engine_inputs, hidden, layout.window_lengths)
        if embeds is None:
            cu = torch.tensor(layout.cu_seqlens, dtype=torch.int32, device=hidden.device)
            embeds = enc.encode(hidden, cu)
        return list(embeds.split(layout.tokens_per_request, dim=0))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        feature_lens: list[int],
        **kwargs,
    ) -> NameToTensorList:
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        return {"audio_embeds": [self._encode(audio_features, feature_lens, engine_inputs)[0]]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        feature_lens: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        per_request = self._encode(audio_features, feature_lens, engine_inputs)
        return {
            rid: {"audio_embeds": [emb]}
            for rid, emb in zip(engine_inputs.request_ids, per_request, strict=True)
        }
