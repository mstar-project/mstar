"""DeepSeek-OCR's two nodes: the DeepEncoder over image views, and the decoder
on the shared `SampledTokenDecoder`."""

from collections.abc import Sequence
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
)
from mstar.engine.resources import AttentionStep, KVStep, ResourceStep
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.deepseek_ocr.components.deepencoder import DeepEncoder
from mstar.model.deepseek_ocr.components.language_model import DeepseekOCRLanguageModel
from mstar.model.deepseek_ocr.config import ATTN, DeepseekOCRTextConfig
from mstar.model.keys import KV_CACHE
from mstar.model.multimodal import MEDIA_LAYOUT, MediaEncoder, media_grids
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule


class DeepseekOCRVisionSubmodule(NodeSubmodule):
    """Each request's image views to the rows of its ``<image>`` tokens.

    A step encodes every request's global views in one batch and every tile in
    another (one view size each), then lays the rows out per image.
    """

    disable_torch_compile = True

    # SAM's global-attention layers at 1024 px hold a (12, 4096, 4096) bias per
    # view; this bounds a call's activations to a few GB.
    VIEWS_PER_CALL = 8
    # View counts each view size is captured at; a call pads up to the next one.
    # Eager, the encoder is launch-bound (about 2.5k kernels a page).
    CAPTURE_VIEW_COUNTS = (1, 2, 3, 4, 6, 8)

    def __init__(self, encoder: DeepEncoder, output: str, view_sizes: tuple[int, ...] = ()):
        super().__init__()
        self.encoder = encoder
        # the node's output signal, its `MediaEncoder`'s
        self.output = output
        # the side of every view this deployment cuts (global and tile), one region each
        self.view_sizes = tuple(sorted(set(view_sizes)))

    @staticmethod
    def _region(size: int) -> str:
        return f"views_{size}"

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        dtype = next(self.encoder.parameters()).dtype

        def static_views(size: int):
            return lambda shape: {"views": torch.zeros(shape.bs, 3, size, size, dtype=dtype, device=device)}

        return {
            self._region(size): PiecewiseBatchedConfig(
                capture_fn=self._capture_views,
                make_static_inputs=static_views(size),
                # rows per view: SAM's 16 px patches, then two stride-2 convs
                seq_len=(size // 64) ** 2,
                capture_batch_sizes=list(self.CAPTURE_VIEW_COUNTS),
            )
            for size in self.view_sizes
        }

    def _capture_views(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        return {"tokens": self.encoder.encode_views(inp.static_inputs["views"])}

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        device = self.get_device()
        # each image's (cols, rows), in prompt order; an image with tiles has one tensor of them
        grid = media_grids(fwd_info.step_metadata.get(MEDIA_LAYOUT) or {}, self.output)
        global_views, tiles = inputs["global_views"], inputs.get("tiles") or []
        if len(global_views) != len(grid) or len(tiles) != sum(cols * rows > 1 for cols, rows in grid):
            raise ValueError(f"{len(global_views)} global views and {len(tiles)} tiled views for grids {grid}")
        tensors = {"global_views": torch.stack(global_views).to(device, non_blocking=True)}
        if tiles:
            tensors["tiles"] = torch.cat(tiles).to(device, non_blocking=True)
        return NodeInputs(tensor_inputs=tensors, kwargs={"grid": grid})

    def preprocess(self, graph_walk, engine_inputs, inputs) -> dict[str, Any]:
        return {"requests": inputs}

    def _encode(self, views: list[torch.Tensor], engine_inputs: ModelInputsFromEngine) -> torch.Tensor:
        views = torch.cat(views)
        runner = engine_inputs.piecewise_runners.get(self._region(views.shape[-1]))
        out = []
        for chunk in views.split(self.VIEWS_PER_CALL):
            if runner is not None and runner.can_run(chunk.shape[0]):
                # cloned: the next replay of this bucket overwrites the runner's buffer
                out.append(runner.run(static_inputs={"views": chunk}, real_bs=chunk.shape[0])
                           .get_view("tokens").clone())
            else:
                out.append(self.encoder.encode_views(chunk))
        return torch.cat(out)

    def _encode_requests(self, requests: list[NodeInputs], engine_inputs: ModelInputsFromEngine) -> list[torch.Tensor]:
        """Per request, its images' rows in prompt order."""
        global_tokens = self._encode([r.tensor_inputs["global_views"] for r in requests], engine_inputs).unbind(0)
        tiled = [r.tensor_inputs["tiles"] for r in requests if "tiles" in r.tensor_inputs]
        tile_counts = [cols * rows for r in requests for cols, rows in r.kwargs["grid"] if cols * rows > 1]
        tile_tokens = iter(self._encode(tiled, engine_inputs).split(tile_counts) if tiled else ())
        out, image = [], 0
        for r in requests:
            rows_of_images = []
            for cols, rows in r.kwargs["grid"]:
                local = next(tile_tokens) if cols * rows > 1 else None
                rows_of_images.append(self.encoder.assemble(global_tokens[image], local, cols, rows))
                image += 1
            out.append(torch.cat(rows_of_images))
        return out

    def forward(self, graph_walk, engine_inputs, requests, **kwargs) -> NameToTensorList:
        return {self.output: self._encode_requests(requests, engine_inputs)}

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def forward_batched(self, graph_walk, engine_inputs, requests, **kwargs) -> dict[str, NameToTensorList]:
        rows = self._encode_requests(requests, engine_inputs)
        return {rid: {self.output: [r]} for rid, r in zip(engine_inputs.request_ids, rows, strict=True)}


class DeepseekOCRLLMSubmodule(SampledTokenDecoder):
    ATTN_KEY = ATTN
    # Decode and the prefills replay CUDA graphs of the eager kernels;
    # torch.compile stays off until compiled numerics get their own parity check.
    disable_torch_compile = True

    # Prompt tokens of a packed image prefill step. Static buffers are sized by
    # the top bucket; a longer step runs eagerly, launch-bound at ~1.3k kernels.
    PREFILL_MEDIA_TOKEN_BUCKETS = [256, 512, 1024, 2048, 3072, 4096, 5120]

    def __init__(
        self, language_model: DeepseekOCRLanguageModel, config: DeepseekOCRTextConfig,
        encoders: Sequence[MediaEncoder] = (), max_prompt_tokens: int | None = None,
    ):
        super().__init__(language_model, config, encoders)
        # Pages per image prefill step: as many of the largest prompt as the top
        # bucket holds, so a packed step of any pages replays a graph.
        most = max(self.PREFILL_MEDIA_CAPTURE_BATCH_SIZES)
        top = self.PREFILL_MEDIA_TOKEN_BUCKETS[-1]
        self.media_max_batch_size = most if max_prompt_tokens is None else max(1, min(most, top // max_prompt_tokens))

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        # every capture uncompiled, as the eager forward runs (disable_torch_compile)
        configs = super().get_accelerator_graph_configs(device, tp_world_size)
        for config in configs:
            config.compile = False
        return configs

    def max_batch_size(self, graph_walk: str) -> int | None:
        return self.media_max_batch_size if graph_walk in self._media_walks else super().max_batch_size(graph_walk)
