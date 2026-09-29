"""AR submodule for the Kimi-K2.7 text backbone."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    CudaGraphConfig,
    PackedCudaGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.kimi_k2_7.config import ATTN, KV_CACHE, ROPE, SAMPLER, KimiK2Config
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

_MAIN = "main"


class KimiVisionEncoderSubmodule(NodeSubmodule):
    """MoonViT tower + patch-merger projector. One image per step."""

    def __init__(self, vision_tower: nn.Module, mm_projector: nn.Module):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        patches = inputs["image_inputs"][0]
        gh, gw = inputs["image_grids"][0].tolist()
        return NodeInputs(
            tensor_inputs={"patches": patches},
            kwargs={"gh": gh, "gw": gw},
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        patches: torch.Tensor,
        gh: int,
        gw: int,
        **kwargs,
    ) -> NameToTensorList:
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        merged = self.vision_tower(patches.to(target_dtype), gh, gw)
        return {"image_embeds": [self.mm_projector(merged)]}


class KimiLLMSubmodule(ARNodeSubmodule):
    def __init__(self, language_model: nn.Module, config: KimiK2Config):
        super().__init__()
        self.language_model = language_model  # KimiForCausalLM
        self.lm_head = language_model.lm_head
        self.config = config
        vision = config.vision
        self.embed_tokens = language_model.model.embed_tokens if vision else None
        self.media_content_token_id = vision.media_content_token_id if vision else None
        self.media_end_token_id = vision.media_end_token_id if vision else None

    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        prefill_buckets = self.config.prefill_token_buckets or self.PREFILL_TOKEN_BUCKETS
        prefill_batch_sizes = (
            self.config.prefill_capture_batch_sizes or self.PREFILL_CAPTURE_BATCH_SIZES
        )
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                # Padding rows under a lease are built from these templates, so
                # they must read as sampling or `declare_step` turns the whole
                # batch eager and disagrees with the leased graph key.
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                    resource_step_info=True,
                ),
            ),
            PackedCudaGraphConfig(
                capture_graph_walk="prefill",
                capture_token_lengths=prefill_buckets,
                make_node_input=lambda n: ARNodeInputs(
                    input_ids=torch.zeros((n,), dtype=torch.long, device=device),
                    input_seq_len=n,
                    resource_step_info=True,
                ),
                capture_batch_sizes=prefill_batch_sizes,
            ),
        ]

    def cg_key_info(
        self, graph_walk: str, per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> Any:
        """Whether every request in this batch is due to sample a token —
        what separates the plain-text/decode captures (always sample,
        ``additional_key_info`` defaults to ``None``) from a text-before-image
        prefill step, which must run eager so ``forward``'s gate applies. Same
        fact ``declare_step`` stamps on the step."""
        del graph_walk
        all_sampling = all(
            info.step_metadata.get("sample_prefill_token", True)
            for info in per_request_info.values()
        )
        return None if all_sampling else False

    def _wrap_vision(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Splice the media-content/media-end sentinels around one image's
        embeddings, the way the chat template brackets its pad-token run."""
        device = image_embeds.device
        content_id = torch.tensor([self.media_content_token_id], device=device)
        end_id = torch.tensor([self.media_end_token_id], device=device)
        with torch.no_grad():
            content_emb = self.embed_tokens(content_id).to(image_embeds.dtype)
            end_emb = self.embed_tokens(end_id).to(image_embeds.dtype)
        return torch.cat([content_emb, image_embeds, end_emb], dim=0)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        # `declare_step` cannot see step_metadata, so it rides in on
        # `resource_step_info` — the same fact `cg_key_info` reads off
        # `per_request_info` before `prepare_inputs` runs.
        sample_prefill_token = fwd_info.step_metadata.get("sample_prefill_token", True)
        if graph_walk == "prefill_vision":
            wrapped = self._wrap_vision(inputs["image_embeds"][0])
            return ARNodeInputs(
                input_embeds=wrapped, input_seq_len=wrapped.shape[0],
                resource_step_info=sample_prefill_token,
            )
        text_inputs = inputs["text_inputs"][0]
        return ARNodeInputs(
            input_ids=text_inputs,
            input_seq_len=text_inputs.shape[0],
            resource_step_info=sample_prefill_token,
        )

    def declare_step(
        self, graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ):
        prefill_tokens = {}
        if graph_walk == "prefill":
            prefill_tokens = {
                rid: inp.input_ids
                for rid, inp in zip(request_ids, inputs, strict=True)
            }
        all_sampling = all(bool(inp.resource_step_info) for inp in inputs)
        return SubmoduleStep(
            cg_key_info=None if all_sampling else False,
            segments=[
                Segment(request_id=rid, label=_MAIN, span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens=prefill_tokens,
                ),
                # position ids come off the stream counters; Kimi's own yarn
                # rotary reads them in the layer
                ROPE: PositionStep(),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if inputs[0].input_embeds is not None:
            return {"input_embeds": inputs[0].input_embeds}
        return {
            "input_ids": torch.cat([inp.input_ids for inp in inputs]),
        }

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        attn: AttentionManager = engine_inputs.resources[ATTN]

        hidden = self.language_model.model(
            input_ids, inputs_embeds=input_embeds, label=_MAIN
        )
        if graph_walk in ("prefill", "prefill_vision"):
            hidden = attn.select_last_hidden(hidden, label=_MAIN)
        elif graph_walk != "decode":
            raise ValueError(
                f"Batched forward not supported for graph walk: {graph_walk!r}"
            )

        logits = self.lm_head(hidden)  # (bs, vocab)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if graph_walk in ("prefill", "prefill_vision"):
            sample = engine_inputs.single_request_info.step_metadata.get(
                "sample_prefill_token", True
            )
            if not sample:
                # Extend the KV cache only — this step's request is not the
                # one whose turn it is to sample a token yet.
                self.language_model.model(
                    input_ids, inputs_embeds=input_embeds, label=_MAIN
                )
                return {}
        return {
            "new_token": self._forward(
                graph_walk=graph_walk,
                engine_inputs=engine_inputs,
                input_ids=input_ids,
                input_embeds=input_embeds,
            )
        }

    def can_batch(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs]
    ) -> bool:
        if batch.graph_walk == "prefill_vision":
            return False
        if batch.graph_walk == "prefill":
            return all(
                info.step_metadata.get("sample_prefill_token", True)
                for info in batch.per_request_info.values()
            )
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
        )
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        if (not ignore_eos and token in self.config.eos_token_ids) or (
            request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
            >= request_info.max_tokens
        ):
            return {"decode_loop"}
        return set()
