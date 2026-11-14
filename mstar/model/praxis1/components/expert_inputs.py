"""Assemble the encoder tap window for the Praxis-1 action expert.

The encoder stores video tokens as ``[noised | clean]``. The expert reads
``[global | clean | noised]`` instead, with globals taken from the encoder
input and video tokens taken from each tapped block output. Keep this
reordering at the boundary rather than teaching either tower a second
sequence layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from mstar.model.praxis1.components.encoder import TapFeatures

GLOBAL_ORDER_TRUNK = ("register", "anchor", "text")


@dataclass(frozen=True)
class ExpertInputs:
    noised_action_latents: torch.Tensor
    action_timesteps: torch.Tensor
    trunk_features: list[torch.Tensor]
    global_tokens: torch.Tensor | None
    proprio_latents: torch.Tensor
    kv_rope_ids: torch.Tensor
    chunk_size: int
    num_clean_frames: int
    num_noised_frames: int
    tokens_per_frame: int
    window_size: int


@dataclass(frozen=True)
class TapGeometry:
    """The named layout of the 13 integers on the backbone-to-expert edge."""

    n_chunks: int
    frames_per_chunk: int
    tokens_per_frame: int
    register: tuple[int, int]
    anchor: tuple[int, int]
    text: tuple[int, int]
    noised: tuple[int, int]
    clean: tuple[int, int]

    @classmethod
    def from_features(cls, features: TapFeatures) -> TapGeometry:
        return cls(
            features.n_chunks, features.frames_per_chunk, features.tokens_per_frame,
            features.global_spans["register"], features.global_spans["anchor"],
            features.global_spans["text"], features.noised_span, features.clean_span or (-1, 0),
        )

    def validate(self, sequence: int) -> None:
        if min(self.n_chunks, self.frames_per_chunk, self.tokens_per_frame) < 1:
            raise ValueError("tap geometry needs positive chunk and frame counts")
        global_spans = (self.register, self.anchor, self.text)
        cursor = 0
        for start, length in global_spans:
            if start != cursor or length < 0:
                raise ValueError("tap global spans must be contiguous")
            cursor += length
        if self.noised[0] != cursor or self.noised[1] != (
            self.n_chunks * self.frames_per_chunk * self.tokens_per_frame
        ):
            raise ValueError("tap noised span does not match the chunk geometry")
        cursor += self.noised[1]
        clean_length = (self.n_chunks - 1) * self.frames_per_chunk * self.tokens_per_frame
        if self.clean != ((-1, 0) if clean_length == 0 else (cursor, clean_length)):
            raise ValueError("tap clean span does not match the chunk geometry")
        if cursor + clean_length != sequence:
            raise ValueError("tap spans do not cover the embedding sequence")

    def encode(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.n_chunks, self.frames_per_chunk, self.tokens_per_frame,
             *self.register, *self.anchor, *self.text, *self.noised, *self.clean],
            dtype=torch.int64, device=device,
        )

    @classmethod
    def decode(cls, tensor: torch.Tensor) -> TapGeometry:
        if tensor.shape != (13,) or tensor.dtype != torch.int64:
            raise ValueError("tap geometry must contain 13 integer values")
        v = tensor.tolist()
        return cls(v[0], v[1], v[2], (v[3], v[4]), (v[5], v[6]),
                   (v[7], v[8]), (v[9], v[10]), (v[11], v[12]))


def pack_tap_features(features: TapFeatures) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carry encoder taps and their small geometry across the backbone-to-expert edge."""
    if (features.embeddings.ndim != 3 or features.embeddings.shape[0] != 1 or not features.taps or
            any(tap.shape != features.embeddings.shape for tap in features.taps)):
        raise ValueError("encoder taps must share one embedding sequence shape")
    geometry = TapGeometry.from_features(features)
    geometry.validate(features.embeddings.shape[1])
    return (
        torch.stack(features.taps), features.embeddings, features.rope_ids,
        geometry.encode(features.embeddings.device),
    )


def unpack_tap_features(
    taps: torch.Tensor, embeddings: torch.Tensor, rope_ids: torch.Tensor, geometry: torch.Tensor,
) -> TapFeatures:
    """Restore the encoder handoff without inventing a second visibility rule."""
    if taps.ndim != 4 or taps.shape[1:] != embeddings.shape or not taps.shape[0]:
        raise ValueError("stacked taps must have shape (layers,1,sequence,dim)")
    if rope_ids.shape != embeddings.shape[:2] or rope_ids.dtype != torch.int64:
        raise ValueError("tap rope ids do not match the embedding sequence")
    layout = TapGeometry.decode(geometry)
    layout.validate(embeddings.shape[1])
    return TapFeatures(
        taps=list(taps.unbind(0)), embeddings=embeddings, rope_ids=rope_ids,
        n_chunks=layout.n_chunks, frames_per_chunk=layout.frames_per_chunk,
        tokens_per_frame=layout.tokens_per_frame,
        global_spans={"register": layout.register, "anchor": layout.anchor, "text": layout.text},
        noised_span=layout.noised, clean_span=None if layout.clean[1] == 0 else layout.clean,
    )


def assemble_expert_inputs(
    features: TapFeatures,
    noised_actions: torch.Tensor,
    action_t_per_chunk: torch.Tensor,
    proprio_per_chunk: torch.Tensor,
    *,
    global_order: Sequence[str] = GLOBAL_ORDER_TRUNK,
) -> ExpertInputs:
    """Build the arguments of ``ActionExpert.forward_trunk_rope``.

    ``features`` comes from one encoder call. There are K noised chunks and
    K-1 clean chunks; each chunk has ``frames_per_chunk`` action tokens.
    The caller chooses the global order from the checkpoint config.
    """
    x = features.embeddings
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"one sample expected, got embeddings {tuple(x.shape)}")
    seq = x.shape[1]
    if features.rope_ids.shape != (1, seq):
        raise ValueError(f"rope ids {tuple(features.rope_ids.shape)} for {seq} tokens")
    if set(global_order) != set(features.global_spans) or len(global_order) != len(
        features.global_spans
    ):
        raise ValueError("global_order must permute the encoder's global spans")
    if features.n_chunks < 1 or features.frames_per_chunk < 1 or features.tokens_per_frame < 1:
        raise ValueError("chunk and frame sizes must be positive")

    n_noised = features.frames_per_chunk * features.n_chunks
    n_clean = features.frames_per_chunk * (features.n_chunks - 1)
    if features.noised_span[1] != n_noised * features.tokens_per_frame:
        raise ValueError(f"noised span {features.noised_span} for {n_noised} frames")
    clean_len = 0 if features.clean_span is None else features.clean_span[1]
    if clean_len != n_clean * features.tokens_per_frame:
        raise ValueError(f"clean span {features.clean_span} for {n_clean} frames")

    def positions(spans: Sequence[tuple[int, int] | None]) -> torch.Tensor:
        parts = []
        for span in spans:
            if span is None:
                continue
            start, length = span
            if start < 0 or length < 0 or start + length > seq:
                raise ValueError(f"span {span} outside the sequence of {seq} tokens")
            if length:
                parts.append(torch.arange(start, start + length, device=x.device))
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=x.device)

    g_idx = positions([features.global_spans[kind] for kind in global_order])
    s_idx = positions([features.clean_span, features.noised_span])
    if any(t.shape != x.shape or t.device != x.device for t in features.taps):
        raise ValueError("each tap must have the encoder input's shape and device")
    if noised_actions.ndim != 3 or noised_actions.shape[:2] != (1, n_noised):
        raise ValueError(f"noised actions {tuple(noised_actions.shape)} for {n_noised} frames")
    if action_t_per_chunk.shape != (1, features.n_chunks):
        raise ValueError(f"action times {tuple(action_t_per_chunk.shape)} for {features.n_chunks} chunks")
    if proprio_per_chunk.ndim != 3 or proprio_per_chunk.shape[:2] != (1, features.n_chunks):
        raise ValueError(f"proprio {tuple(proprio_per_chunk.shape)} for {features.n_chunks} chunks")

    ids = features.rope_ids.to(x.device)
    return ExpertInputs(
        noised_action_latents=noised_actions.to(device=x.device, dtype=x.dtype),
        action_timesteps=action_t_per_chunk.to(x.device).repeat_interleave(
            features.frames_per_chunk, dim=1
        ),
        trunk_features=[tap.index_select(1, s_idx) for tap in features.taps],
        global_tokens=x.index_select(1, g_idx) if g_idx.numel() else None,
        proprio_latents=proprio_per_chunk.to(device=x.device, dtype=x.dtype).repeat_interleave(
            features.frames_per_chunk, dim=1
        ),
        kv_rope_ids=torch.cat([ids.index_select(1, g_idx), ids.index_select(1, s_idx)], dim=1),
        chunk_size=features.frames_per_chunk,
        num_clean_frames=n_clean,
        num_noised_frames=n_noised,
        tokens_per_frame=features.tokens_per_frame,
        window_size=n_clean + n_noised,
    )
