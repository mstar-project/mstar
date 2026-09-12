"""Qwen3.5's node submodules: the hybrid LLM and the vision encoder.

The LLM node serves every walk. ``prefill_text`` and ``decode`` hand it token
ids; ``prefill_vision`` hands it embeddings the encoder already produced, which
is why it declares its step off ``input_seq_len`` rather than off the ids.

Positions are the one thing that does not ride a resource cursor. The rotation
is Qwen3.5's own interleaved 3D MRoPE (see ``components/rope.py``), so the
position resource is used purely as a per-request counter and the 3D ids are
built here and threaded in as cos/sin.
"""

import logging
import os
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    CudaGraphConfig,
    PackedCudaGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.components.rope import (
    text_position_ids,
    vision_position_advance,
    vision_position_ids,
)
from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    ROPE,
    SAMPLER,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)


class LLMSubmodule(ARNodeSubmodule):
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]
    # Capped, because every captured row holds a recurrent slot until the whole
    # capture pass finishes: the runner keys its dummy rows per (config, slot),
    # so the pool's high-water mark is
    #   sum over configs of (num_slots x max capture bs)
    # and at ~50 MiB a slot that is what sets `gdn_state.max_slots`, not the
    # concurrency the deployment actually wants. See configs/qwen3_5.yaml.
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(
        self,
        model: Qwen3_5ForCausalLM,
        config: Qwen3_5Config,
        vision_config: Qwen3_5VisionConfig | None = None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.vision_config = vision_config
        self._vision_sentinels: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        """Decode and text prefill both capture.

        Nothing here is model-specific beyond the bucket sizes: the recurrent
        pool and the GDN resource size their plan buffers per (bucket, slot),
        so a captured walk replays without re-planning.

        Vision prefill stays eager. Its token count is `t * h' * w' + 2` off
        the image grid, which is continuous rather than bucketed, and it runs
        once per request against a decode loop that runs hundreds of times —
        so the buckets would mostly miss and the win would be small anyway.
        """
        def dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_ids=torch.zeros(n, dtype=torch.long, device=device),
                input_seq_len=n,
                # `preprocess` turns these into the cos/sin static buffers, so
                # capture needs them shaped even though the values are dummy —
                # replay copies the real positions in.
                custom_pos_ids=text_position_ids(n, 0, device),
            )

        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=dummy(1),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedCudaGraphConfig(
                capture_graph_walk="prefill_text",
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=dummy,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

    # ------------------------------------------------------------------
    # Step declaration
    # ------------------------------------------------------------------

    def _sentinel_embeds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``<|vision_start|>`` / ``<|vision_end|>`` as embeddings, cached.

        The media span is sentinel-inclusive, so `split_around_spans` dropped
        these two along with the pad interior. This walk owns them.
        """
        if self._vision_sentinels is None:
            ids = torch.tensor(
                [self.config.vision_start_token_id, self.config.vision_end_token_id],
                dtype=torch.long, device=self.get_device(),
            )
            embeds = self.model.model.embed_tokens(ids)
            self._vision_sentinels = (embeds[:1], embeds[1:])
        return self._vision_sentinels

    def _vision_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList,
    ) -> ARNodeInputs:
        """One image, wrapped in its sentinels, with 3D grid positions.

        The three MRoPE grids stop moving together across an image: T is flat
        while H and W sweep the merged patch grid, so the span covers
        ``max(h', w')`` positions but ``t * h' * w'`` tokens. The sentinels
        take text positions either side, and `declare_step` tells the position
        resource the real advance — left to its own rule it would advance by
        the token count and put the following text in the wrong place.
        """
        if self.vision_config is None:
            raise ValueError(
                "prefill_vision needs the vision config for its spatial merge "
                "size; this LLM submodule was built without one"
            )
        device = self.get_device()
        embeds = inputs["vision_embeds"][0].to(device)
        grid = inputs["image_grid_thw"][0]
        grid = grid[0] if grid.dim() == 2 else grid
        merge = self.vision_config.spatial_merge_size

        start_pos = self.node_resources[ROPE].position(
            rid=fwd_info.request_id, label="main",
        )
        start_embed, end_embed = self._sentinel_embeds()
        advance = vision_position_advance(grid, merge)
        pos_ids = torch.cat(
            [
                text_position_ids(1, start_pos, device),
                vision_position_ids(grid, merge, start_pos + 1, device),
                text_position_ids(1, start_pos + 1 + advance, device),
            ],
            dim=1,
        )
        return ARNodeInputs(
            input_seq_len=embeds.shape[0] + 2,
            input_embeds=torch.cat([start_embed, embeds, end_embed], dim=0),
            custom_pos_ids=pos_ids,
            # two sentinels either side, and the image between them
            tensor_inputs={"mrope_advance": advance + 2},
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> ARNodeInputs:
        if graph_walk == "prefill_vision":
            return self._vision_inputs(fwd_info, inputs)

        input_ids = inputs["text_inputs"][0]
        seq_len = input_ids.shape[0]
        # The counter the position resource keeps for this stream, advanced by
        # the PositionStep below. The rotation is ours, but the bookkeeping is
        # not worth duplicating.
        start_pos = self.node_resources[ROPE].position(
            rid=fwd_info.request_id, label="main",
        )
        return ARNodeInputs(
            input_seq_len=seq_len,
            input_ids=input_ids,
            # 3D, and read by `preprocess` below rather than by the position
            # resource — that one derives its own 1D ids from the counters.
            custom_pos_ids=text_position_ids(
                seq_len=seq_len, start_pos=start_pos, device=self.get_device(),
            ),
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        out: dict[str, torch.Tensor | Any] = {}
        if inputs[0].input_ids is not None:
            out["input_ids"] = torch.cat([inp.input_ids for inp in inputs], dim=0)
        else:
            out["input_embeds"] = torch.cat(
                [inp.input_embeds for inp in inputs], dim=0,
            )

        position_ids_3d = torch.cat(
            [inp.custom_pos_ids for inp in inputs], dim=1,
        )  # (3, total_tokens)
        cos, sin = self.model.model.build_cos_sin(
            position_ids_3d, dtype=self.model.model.embed_tokens.weight.dtype,
        )
        out["cos_3d"] = cos
        out["sin_3d"] = sin
        return out

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        **kwargs,
    ) -> SubmoduleStep:
        prefill_tokens = {}
        if graph_walk == "prefill_text":
            prefill_tokens = {
                rid: inp.input_ids
                for rid, inp in zip(request_ids, inputs, strict=True)
            }

        # `advance=None` means the resource's own rule, which is the span. That
        # is right for text and wrong for an image, whose 3D grid covers fewer
        # positions than it does tokens.
        advance = None
        if graph_walk == "prefill_vision":
            advance = tuple(
                int(inp.tensor_inputs["mrope_advance"]) for inp in inputs
            )

        return SubmoduleStep(
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                )
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                GDN_STATE: RecurrentStep(),
                LINEAR_ATTN: LinearAttnStep(),
                SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens=prefill_tokens,
                ),
                ROPE: PositionStep(advance=advance),
            },
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        cos_3d: torch.Tensor,
        sin_3d: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn: AttentionManager = engine_inputs.resources[ATTN]
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]

        embeds = (
            self.model.model.embed_tokens(input_ids)
            if input_embeds is None
            else input_embeds
        )
        hidden = self.model.model(embeds, label="main", cos_sin=(cos_3d, sin_3d))

        if graph_walk != "decode":
            # only the last token of each prefill row feeds the head
            hidden = attn.select_last_hidden(hidden)
        logits = self.model.lm_head(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        cos_3d: torch.Tensor,
        sin_3d: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {
            "new_token": self._forward(
                graph_walk, engine_inputs, cos_3d, sin_3d,
                input_ids=input_ids, input_embeds=input_embeds,
            )
        }

    def can_batch(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        cos_3d: torch.Tensor,
        sin_3d: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(
            graph_walk, engine_inputs, cos_3d, sin_3d,
            input_ids=input_ids, input_embeds=input_embeds,
        )
        return {
            rid: {"new_token": [new_tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

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
        if request_info.graph_walk != "decode" and \
                not request_info.step_metadata.get("last_prefill", False):
            outputs.pop("new_token", None)
            return
        # Metadata only: the decode loop routes on `text_inputs`, so rebind the
        # name rather than copying. The EOS test is in `check_stop` so the GPU
        # thread never syncs on `.item()` here.
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        hit_eos = not ignore_eos and token in self.config.stop_token_ids
        out_of_budget = (
            request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
            >= request_info.max_tokens
        )
        return {"decode_loop"} if hit_eos or out_of_budget else set()


class VisionEncoderSubmodule(NodeSubmodule):
    """Qwen3.5's ViT, which turns pixel patches into LLM-width embeddings.

    Runs once per request and then idles, so it stays eager: its token count
    is set by the image grid, which is continuous rather than bucketed, and
    there is no cache to carry anything between calls.
    """

    def __init__(
        self, model: Qwen3_5VisionModel, config: Qwen3_5VisionConfig,
    ):
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
        grid = inputs["image_grid_thw"][0]
        return NodeInputs(
            tensor_inputs={
                "pixel_values": inputs["pixel_values"][0],
                # (num_images, 3). A single-image request arrives as a bare
                # [t, h, w], which the per-image loops in `vision.py` would
                # read as three images.
                "image_grid_thw": grid.unsqueeze(0) if grid.dim() == 1 else grid,
            },
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        **kwargs,
    ) -> SubmoduleStep:
        # The encoder holds no engine resources: it is a pure function of its
        # pixels. The LLM node declares the step that prefill_vision plans.
        return SubmoduleStep(steps={})

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        device = self.get_device()
        # The grid drives `.tolist()` loops and index maths inside the tower,
        # so it has to sit beside the weights, not on the conductor's CPU copy.
        embeds = self.model(
            pixel_values.to(device), image_grid_thw.to(device),
        )
        return {"vision_embeds": [embeds]}
