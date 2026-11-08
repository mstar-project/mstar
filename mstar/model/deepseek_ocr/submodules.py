"""DeepSeek-OCR's two nodes: the DeepEncoder over image views, and the decoder."""

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
)
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.model.deepseek_ocr.components.deepencoder import DeepEncoder
from mstar.model.deepseek_ocr.components.language_model import DeepseekOCRLanguageModel
from mstar.model.deepseek_ocr.config import (
    ATTN,
    KV_CACHE,
    PREFILL_WALKS,
    ROPE,
    SAMPLER,
    STEP_GRIDS_KEY,
    DeepseekOCRTextConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    TokenStop,
)


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

    def __init__(self, encoder: DeepEncoder, view_sizes: tuple[int, ...] = ()):
        super().__init__()
        self.encoder = encoder
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
        grid = [tuple(g) for g in fwd_info.step_metadata[STEP_GRIDS_KEY]]
        # per image: its global view, then its tiles if it has a tiled local view
        views = iter(inputs["image_views"])
        global_views, tiles = [], []
        for cols, rows in grid:
            global_views.append(next(views))
            if cols * rows > 1:
                tiles.append(next(views))
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
        return {"image_embeds": self._encode_requests(requests, engine_inputs)}

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def forward_batched(self, graph_walk, engine_inputs, requests, **kwargs) -> dict[str, NameToTensorList]:
        rows = self._encode_requests(requests, engine_inputs)
        return {rid: {"image_embeds": [r]} for rid, r in zip(engine_inputs.request_ids, rows, strict=True)}


class DeepseekOCRLLMSubmodule(ARNodeSubmodule):
    # Decode and image prefill replay CUDA graphs of the eager kernels;
    # torch.compile stays off until compiled numerics get their own parity check.
    disable_torch_compile = True

    # Prompt tokens of a packed prefill_image step. Static buffers are sized by
    # the top bucket; a longer step runs eagerly, launch-bound at ~1.3k kernels.
    PREFILL_IMAGE_TOKEN_BUCKETS = [256, 512, 1024, 2048, 3072, 4096, 5120]
    PREFILL_IMAGE_CAPTURE_BATCH_SIZES = [1, 2, 4]

    def __init__(
        self, language_model: DeepseekOCRLanguageModel, config: DeepseekOCRTextConfig, image_token_id: int,
        max_prompt_tokens: int | None = None,
    ):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.image_token_id = image_token_id
        # Pages per prefill_image step: as many of the largest prompt as the top
        # bucket holds, so a packed step of any pages replays a graph.
        most = max(self.PREFILL_IMAGE_CAPTURE_BATCH_SIZES)
        top = self.PREFILL_IMAGE_TOKEN_BUCKETS[-1]
        self.prefill_image_max_batch_size = (
            most if max_prompt_tokens is None else max(1, min(most, top // max_prompt_tokens)))
        self.token_stop = TokenStop(
            loop="decode_loop", buffer="new_token", stop_ids=frozenset({config.eos_token_id}),
            sampler=SAMPLER, prefill_walks=frozenset(PREFILL_WALKS),
        )
        # decode reads its last token off the sampler's slot masters, and its
        # client token rides inline
        self.device_loopback = DeviceLoopback(
            walk="decode", signal="text_inputs", sampler=SAMPLER, buffer="new_token",
        )

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        hidden, dtype = self.config.hidden_size, self.language_model.embed_tokens.weight.dtype

        def image_prompt(n: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_ids=torch.zeros(n, dtype=torch.long, device=device),
                input_embeds=torch.zeros((n, hidden), dtype=dtype, device=device),
                input_seq_len=n,
            )

        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=1,
                ),
                compile=False,
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_image",
                capture_token_lengths=self.PREFILL_IMAGE_TOKEN_BUCKETS,
                make_node_input=image_prompt,
                capture_batch_sizes=self.PREFILL_IMAGE_CAPTURE_BATCH_SIZES,
                compile=False,
            ),
        ]

    def max_batch_size(self, graph_walk: str) -> int | None:
        return self.prefill_image_max_batch_size if graph_walk == "prefill_image" else None

    @staticmethod
    def _check_walk(graph_walk: str) -> None:
        if graph_walk not in (*PREFILL_WALKS, "decode"):
            raise ValueError(f"Unknown DeepSeek-OCR graph walk: {graph_walk!r}")

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        self._check_walk(graph_walk)
        if (row := self.uniform_row_inputs(graph_walk, kwargs.get("has_loopback", False))) is not None:
            # one token per row, read off the sampler's slot master in `preprocess`
            return row
        tokens = inputs["text_inputs"]
        if len(tokens) != 1 or tokens[0].ndim != 1 or tokens[0].numel() == 0:
            raise ValueError("text_inputs must contain one nonempty 1D token tensor")
        ids = tokens[0]
        if graph_walk == "decode" and ids.numel() != 1:
            raise ValueError("decode requires exactly one token per request")
        ids = ids.to(device=self.get_device(), dtype=torch.long)
        if graph_walk != "prefill_image":
            return ARNodeInputs(input_ids=ids, input_seq_len=ids.numel())
        # Embedded here, so a chunked prefill cuts the embeddings with the ids.
        (image_embeds,) = inputs["image_embeds"]
        embeds = self.language_model.embed(ids)
        embeds = embeds.masked_scatter(
            (ids == self.image_token_id).unsqueeze(-1), image_embeds.to(embeds.device, embeds.dtype),
        )
        return ARNodeInputs(input_ids=ids, input_embeds=embeds, input_seq_len=ids.numel())

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None, **kwargs,
    ) -> SubmoduleStep:
        self._check_walk(graph_walk)
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                ROPE: PositionStep(),
                SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens={
                        rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
                    } if graph_walk in PREFILL_WALKS else {},
                ),
            },
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, Any]:
        self._check_walk(graph_walk)
        if graph_walk == "decode" and any(inp.input_ids is None for inp in inputs):
            # device loop-back: every row is its request's last sampled token, read by slot
            return {"text_inputs": self.loopback_inputs(engine_inputs, inputs)["input_ids"]}
        ids = torch.cat([inp.input_ids for inp in inputs])
        if graph_walk == "prefill_image":
            return {"text_inputs": ids, "input_embeds": torch.cat([inp.input_embeds for inp in inputs])}
        return {"text_inputs": ids}

    def _sample(self, graph_walk, engine_inputs, text_inputs, input_embeds=None):
        self._check_walk(graph_walk)
        embeddings = self.language_model.embed(text_inputs) if input_embeds is None else input_embeds
        hidden = self.language_model(embeddings, label="main")
        if graph_walk in PREFILL_WALKS:
            # the last real prompt token of each request, not the batch's last row
            hidden = engine_inputs.resources[ATTN].select_last_hidden(hidden, label="main")
        logits = self.language_model.logits(hidden)
        return engine_inputs.resources[SAMPLER].sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor, input_embeds: torch.Tensor | None = None, **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._sample(graph_walk, engine_inputs, text_inputs, input_embeds)]}

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor, input_embeds: torch.Tensor | None = None, **kwargs,
    ) -> BatchedModelOutput:
        tokens = self._sample(graph_walk, engine_inputs, text_inputs, input_embeds)
        # Row i is request i: one clone and one D2H copy per step, and the
        # sampler's host rows (log-probs) ride along with the stop buffer. A
        # decode step's token is its loop's one signal (`text_inputs`), which
        # loops back on the device and reaches the client inline; prefill's is
        # the persisted first token.
        signal = "text_inputs" if graph_walk == "decode" else "new_token"
        return BatchedModelOutput(row_outputs={signal: tokens}, check_stop_buffers={"new_token": tokens})

    def postprocess(self, request_id, request_info, outputs, **kwargs):
        # Metadata only: check_stop reads the token values, off the GPU thread.
        if "new_token" not in outputs:
            return
        if request_info.graph_walk in PREFILL_WALKS:
            if request_info.max_tokens > 1:
                outputs["decode_input"] = outputs["new_token"]
            return
        outputs["text_inputs"] = outputs["new_token"]
