"""MiniCPM-o's node submodules: the Qwen3 LLM and the two encoders.

The LLM's prompt is the whole rendered chat, placeholders included; a
multimodal prefill embeds it and overwrites each placeholder row with the
encoder output that belongs there, at positions ``process_prompt`` read off
the prompt. RoPE is plain 1D over that sequence, so the position resource
advances by the token count with no special casing.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    RaggedCrossAttentionStep,
    SamplerStep,
    Segment,
    SubmoduleStep,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.minicpm_o.components.audio import MiniCPMOAudio
from mstar.model.minicpm_o.components.vision import MiniCPMOVision, slice_layout
from mstar.model.minicpm_o.config import (
    AUDIO_ATTN,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    PATCHES,
    QUERIES,
    RESAMPLER_ATTN,
    VISION_ATTN,
    MiniCPMOConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

DECODE = "decode"
DECODE_LOOP = "decode_loop"

# encoder output name -> the prompt positions it fills
_MM_FILLS = {"vision_embeds": "image_positions", "audio_embeds": "audio_positions"}


class LLMSubmodule(ARNodeSubmodule):
    def __init__(self, model: Qwen3DenseLM, config: MiniCPMOConfig):
        super().__init__()
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _prompt_embeds(self, inputs: NameToTensorList) -> torch.Tensor:
        """The prompt's embeddings with each placeholder row overwritten by the
        encoder output for it, in prompt order."""
        device = self.get_device()
        embeds = self.model.embed(inputs["text_inputs"][0].to(device))
        for name, positions in _MM_FILLS.items():
            if name not in inputs:
                continue
            index = inputs[positions][0].to(device)
            values = inputs[name][0].to(device=device, dtype=embeds.dtype)
            if values.shape[0] != index.shape[0]:
                raise ValueError(
                    f"{name} has {values.shape[0]} rows for {index.shape[0]} placeholder "
                    "tokens; the prompt and the encoder disagree on the attachment's length"
                )
            embeds = embeds.index_copy(0, index, values)
        return embeds

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> ARNodeInputs:
        ids = inputs["text_inputs"][0]
        if graph_walk == DECODE or not any(name in inputs for name in _MM_FILLS):
            return ARNodeInputs(input_seq_len=ids.shape[0], input_ids=ids)
        embeds = self._prompt_embeds(inputs)
        return ARNodeInputs(input_seq_len=embeds.shape[0], input_embeds=embeds)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        if inputs[0].input_ids is not None:
            ids = torch.cat([inp.input_ids for inp in inputs]).to(device)
            return {"input_embeds": self.model.embed(ids)}
        return {"input_embeds": torch.cat([inp.input_embeds for inp in inputs])}

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        **kwargs,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                LLM_KV: KVStep(),
                LLM_ATTN: AttentionStep(causal=True),
                LLM_POS: PositionStep(),
                # The prompt is not tracked: upstream generates from embeddings,
                # so its penalty sees only generated tokens.
                LLM_SAMPLER: SamplerStep(apply_penalty=True),
            },
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        attn: AttentionManager = engine_inputs.resources[LLM_ATTN]
        sampler: SamplerResource = engine_inputs.resources[LLM_SAMPLER]
        hidden = self.model(input_embeds, label="main")
        if graph_walk != DECODE:
            hidden = attn.select_last_hidden(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=self.model.logits(hidden))

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor, **kwargs,
    ) -> NameToTensorList:
        return {"new_token": self._forward(graph_walk, engine_inputs, input_embeds)}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor, **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(graph_walk, engine_inputs, input_embeds)
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            check_stop_buffers={"new_token": new_tokens},
        )

    # ------------------------------------------------------------------
    # Post-step
    # ------------------------------------------------------------------

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Rebind, not copy: the decode loop routes on `text_inputs`. EOS is
        # tested in `check_stop`, off the GPU thread.
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def _stops(self, token: int, info: CurrentForwardPassInfo) -> bool:
        hit_eos = (
            not info.resource_configs[LLM_SAMPLER].ignore_eos
            and token in self.config.stop_token_ids
        )
        out_of_budget = info.dynamic_loop_iter_counts.get(DECODE_LOOP, 0) + 1 >= info.max_tokens
        return hit_eos or out_of_budget

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        return {DECODE_LOOP} if self._stops(outputs["new_token"][0].item(), request_info) else set()

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
    ) -> dict[str, set[str]] | None:
        tokens = host_rows.buffers.get("new_token")
        if not torch.is_tensor(tokens) or tokens.dim() == 0:
            return None
        values = tokens.reshape(tokens.shape[0], -1)[:, 0].tolist()
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        stops: dict[str, set[str]] = {}
        for rid in request_ids:
            i = row_of.get(rid)
            if i is None or i >= len(values):
                continue
            if self._stops(values[i], request_infos[rid]):
                stops[rid] = {DECODE_LOOP}
        return stops


class VisionEncoderSubmodule(NodeSubmodule):
    """Every slice of every image in a request, packed into one forward: the
    navit SigLIP attends within each slice, then the resampler turns each slice
    into 64 tokens. Requests pack too; their slices stay apart by segment."""

    def __init__(self, model: MiniCPMOVision, config: MiniCPMOConfig):
        super().__init__()
        self.model = model
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        tgt_sizes = [
            (int(h), int(w)) for t in inputs["image_tgt_sizes"] for h, w in t.reshape(-1, 2).tolist()
        ]
        layout = slice_layout(tgt_sizes, self.config.vision)
        patches = torch.cat(inputs["pixel_values"])
        return NodeInputs(
            input_seq_len=patches.shape[0],
            tensor_inputs={
                "patches": patches,
                "position_ids": layout.position_ids,
                "grid_coords": layout.grid_coords,
            },
            kwargs={"seq_lengths": layout.seq_lengths},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        **kwargs,
    ) -> SubmoduleStep:
        patches, queries = [], []
        for rid, inp in zip(request_ids, inputs, strict=True):
            for span in inp.kwargs["seq_lengths"]:
                patches.append(Segment(request_id=rid, label=PATCHES, span=span))
                queries.append(Segment(request_id=rid, label=QUERIES, span=self.config.resampler.num_queries))
        return SubmoduleStep(
            steps={
                VISION_ATTN: AttentionStep(segments=tuple(patches), causal=False),
                RESAMPLER_ATTN: RaggedCrossAttentionStep(
                    segments=tuple(patches + queries), pairs=((QUERIES, PATCHES),),
                ),
            },
        )

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()

        def cat(name):
            return torch.cat([inp.tensor_inputs[name] for inp in inputs]).to(device)

        return {
            "patches": cat("patches"),
            "position_ids": cat("position_ids"),
            "grid_coords": cat("grid_coords"),
            "slices_per_request": [len(inp.kwargs["seq_lengths"]) for inp in inputs],
        }

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        patches: torch.Tensor,
        position_ids: torch.Tensor,
        grid_coords: torch.Tensor,
        slices_per_request: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        embeds = self.model(patches, position_ids, grid_coords, sum(slices_per_request))
        q = self.config.resampler.num_queries
        per_request = embeds.split([n * q for n in slices_per_request])
        return {
            rid: {"vision_embeds": [out]}
            for rid, out in zip(engine_inputs.request_ids, per_request, strict=True)
        }


class AudioEncoderSubmodule(NodeSubmodule):
    """A request's audio pieces (30 s each at most), packed with any other
    request's; each piece is its own block-causal segment."""

    def __init__(self, model: MiniCPMOAudio, config: MiniCPMOConfig):
        super().__init__()
        self.model = model
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        mel_lens = [int(n) for t in inputs["audio_feature_lens"] for n in t.reshape(-1).tolist()]
        features = torch.cat(inputs["audio_features"], dim=-1)
        frames = [self.config.audio.conv_frames(n) for n in mel_lens]
        return NodeInputs(
            input_seq_len=sum(frames),
            tensor_inputs={"features": features},
            kwargs={"mel_lens": mel_lens, "frames": frames},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        **kwargs,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=n)
                for rid, inp in zip(request_ids, inputs, strict=True)
                for n in inp.kwargs["frames"]
            ],
            steps={AUDIO_ATTN: AttentionStep(causal=False)},
        )

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        pieces: list[torch.Tensor] = []
        for inp in inputs:
            pieces += list(inp.tensor_inputs["features"].to(device).split(inp.kwargs["mel_lens"], dim=-1))
        return {
            "pieces": pieces,
            "frames": [n for inp in inputs for n in inp.kwargs["frames"]],
            "tokens_per_request": [
                sum(self.config.audio.pooled_tokens(n) for n in inp.kwargs["mel_lens"]) for inp in inputs
            ],
        }

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pieces: list[torch.Tensor],
        frames: list[int],
        tokens_per_request: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        embeds = self.model(pieces, frames)
        return {
            rid: {"audio_embeds": [out]}
            for rid, out in zip(engine_inputs.request_ids, embeds.split(tokens_per_request), strict=True)
        }
