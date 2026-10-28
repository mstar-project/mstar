"""Qwen3.5's node submodules: the hybrid LLM and the vision encoder.

Both build on the shared scaffolds (`SampledTokenDecoder`,
`PackedBlockLoopEncoder`). What is Qwen3.5's own: the hybrid layers'
resource steps, and the rotation, interleaved 3D MRoPE
(``components/rope.py``). The position resource is only a per-request
counter here; the 3D ids are built in this module and passed in as cos/sin.
"""

import os
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.block_loop_encoder import PackedBlockLoopEncoder
from mstar.model.components.mrope import mrope_position_ids
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.multimodal import MediaEncoder
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.components.rope import (
    text_position_ids,
    vision_position_advance,
    vision_position_ids,
)
from mstar.model.qwen3_5.components.vision import (
    vision_interpolation,
    vision_seq_lengths,
)
from mstar.model.qwen3_5.components.vision import (
    # the tower's 2D patch grid, not `rope`'s 3D MRoPE ids of the same name
    vision_position_ids as vision_grid_position_ids,
)
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    MIXED_MAX_PREFILL_ROWS,
    ROPE,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs
from mstar.utils.knobs import ingraph_decode_rope
from mstar.utils.profiler import PhaseLaps


class LLMSubmodule(SampledTokenDecoder):
    ATTN_KEY = ATTN

    # An eager 8k prefill is launch-bound (~1,100 kernel launches, the GPU
    # idle half the time: 78 ms of wall for 36 ms of kernels on 0.8B), so the
    # ladder runs to 8192. MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS narrows or
    # extends it (static buffers are sized by the largest bucket).
    PREFILL_TOKEN_BUCKETS = [
        int(x) for x in os.environ.get(
            "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", "32,64,128,256,512,1024,2048,4096,8192",
        ).split(",")
    ]
    # Capture rows and a replay's padding rows address the pool's sink and
    # hold no slot, so these buckets do not size `gdn_state.max_slots`; that is
    # set by the concurrency a deployment wants (see configs/qwen3_5_*.yaml).
    # The ladder runs to 128 because the largest bucket is also the scheduler's
    # cap on rows per decode step: with 32 as the top, 64 requests in flight
    # ran as two alternating batches of 32 and each paid the whole per-step
    # host cost. MSTAR_QWEN35_DECODE_BUCKETS overrides MSTAR_DECODE_BUCKETS for
    # this model alone; both are read here, so a reload of this module sees
    # the environment it runs under.
    DECODE_CAPTURE_BATCH_SIZES = [
        int(x) for x in (
            os.environ.get("MSTAR_QWEN35_DECODE_BUCKETS")
            or os.environ.get("MSTAR_DECODE_BUCKETS", "1,2,4,8,16,32,48,64,96,128")
        ).split(",")
    ]

    def __init__(
        self,
        model: Qwen3_5ForCausalLM,
        config: Qwen3_5Config,
        vision_config: Qwen3_5VisionConfig | None = None,
        encoders: tuple[MediaEncoder, ...] = (),
    ):
        super().__init__(model, config, encoders)
        # patches per merged token along each axis, for the MRoPE grids
        self._merge_size = vision_config.spatial_merge_size if vision_config is not None else None

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            GDN_STATE: RecurrentStep(),
            LINEAR_ATTN: LinearAttnStep(),
        }

    def max_rows_in_combined(self, graph_walk: str) -> int | None:
        # a mixed step's prompt rows take the chunked GDN path, sized to these
        return MIXED_MAX_PREFILL_ROWS if graph_walk == "prefill_text" else None

    # ------------------------------------------------------------------
    # MRoPE
    # ------------------------------------------------------------------

    def prompt_positions(
        self, fwd_info: CurrentForwardPassInfo, layout: dict, num_tokens: int,
    ) -> tuple[torch.Tensor, int]:
        """A media prompt's ``[3, tokens]`` MRoPE ids, from the intake's
        layout (no read of the ids).

        Text, sentinels included, takes one position a token. An image (or a
        video's frame group) spans ``max(h', w')`` MRoPE positions but
        ``t * h' * w'`` tokens, so the cursor advances by the real amount and
        the position resource is handed the total.
        """
        device = self.get_device()
        merge = self._merge_size
        pos = start = self.node_resources[ROPE].position(rid=fwd_info.request_id, label="main")
        pos_ids: list[torch.Tensor] = []
        cursor = 0
        for offset, tokens, grid in zip(
            layout.get("offset", ()), layout.get("tokens", ()), layout.get("grid", ()), strict=True,
        ):
            if offset > cursor:
                pos_ids.append(text_position_ids(offset - cursor, pos, device))
                pos += offset - cursor
            grid = torch.tensor(grid)
            pos_ids.append(vision_position_ids(grid, merge, pos, device))
            pos += vision_position_advance(grid, merge)
            cursor = offset + tokens
        if num_tokens > cursor:
            pos_ids.append(text_position_ids(num_tokens - cursor, pos, device))
            pos += num_tokens - cursor
        return torch.cat(pos_ids, dim=1), pos - start

    @staticmethod
    def _captured_decode(engine_inputs: ModelInputsFromEngine, graph_walk: str) -> bool:
        """A decode step replayed on a leased slot, or being captured for one.
        The caller's walk decides (a capture's own context carries the
        piecewise walk), the lease says the slot's static buffers are in
        play."""
        if graph_walk != "decode":
            return False
        step = getattr(engine_inputs, "step", None)
        ctx = getattr(step, "ctx", None) if step is not None else None
        return ctx is not None and getattr(ctx, "slot_lease", None) is not None

    def _decode_cos_sin(
        self, engine_inputs: ModelInputsFromEngine, num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The rotary tables of a decode step from the position resource's
        planned positions: one token per row, no custom grids."""
        pos_ids = self.node_resources[ROPE].pos_ids("main")
        assert pos_ids is not None, (
            "position resource has no plan for this step; `plan` must run "
            "before the forward"
        )
        pos = pos_ids[:num_tokens].unsqueeze(0).expand(3, -1)
        return self.model.model.build_cos_sin(pos, dtype=self.model.embed_dtype)

    def _position_ids_3d(self, inputs: list[ARNodeInputs]) -> torch.Tensor:
        return mrope_position_ids(self.node_resources[ROPE].pos_ids("main"), inputs)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        if (
            self._reads_loopback(inputs)
            and self._captured_decode(engine_inputs, graph_walk)
            and ingraph_decode_rope()
        ):
            # the captured forward builds the rotary tables itself
            return out
        laps = PhaseLaps("engine.preprocess.")
        position_ids_3d = self._position_ids_3d(inputs)  # (3, total_tokens)
        laps.lap("position_ids")
        cos, sin = self.model.model.build_cos_sin(
            position_ids_3d, dtype=self.model.embed_dtype,
        )
        laps.lap("cos_sin")
        out["cos_3d"] = cos
        out["sin_3d"] = sin
        return out

    def decoder_kwargs(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        num_tokens: int,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        **tensors: torch.Tensor,
    ) -> dict[str, Any]:
        if (
            (self._captured_decode(engine_inputs, graph_walk) and ingraph_decode_rope())
            or cos_3d is None or sin_3d is None
        ):
            # MSTAR_INGRAPH_DECODE_ROPE: the tables come from the planned
            # positions, inside the graph; the static cos/sin stay unread
            cos_3d, sin_3d = self._decode_cos_sin(engine_inputs, num_tokens)
        return {"cos_sin": (cos_3d, sin_3d)}


class VisionEncoderSubmodule(PackedBlockLoopEncoder):
    """Qwen3.5's ViT, which turns pixel patches into LLM-width embeddings.

    Grid maths runs in `prepare_inputs`, and each frame is its own attention
    segment, so the compiled tower's only symbolic shape is its token count.
    The resource's segment ceiling is one per image (and per video frame
    group); see `Qwen3_5DenseModel._vision_resources`.
    """

    # 16384 patches merge to the LLM's 4096 `prefill_vision` bucket
    BLOCK_LOOP_PATCH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        # (num_images, 3): a bare [t, h, w] would read as three images
        device = self.get_device()
        merge = self.config.spatial_merge_size
        grid = [
            (int(t), int(h), int(w))
            for g in inputs["image_grid_thw"]
            for t, h, w in g.reshape(-1, 3).tolist()
        ]
        # Built here, not in the compiled forward: a `.tolist()` there breaks
        # the graph and makes h, w symints inductor cannot codegen from.
        indices, weights = vision_interpolation(
            grid, self.config.num_grid_per_side, merge, device,
        )
        return NodeInputs(
            # what the engine leases the captured block loop's bucket by
            input_seq_len=sum(t * h * w for t, h, w in grid),
            tensor_inputs={
                "pixel_values": torch.cat(inputs["pixel_values"], dim=0),
                "indices": indices,
                "weights": weights,
                "position_ids": vision_grid_position_ids(grid, merge, device),
            },
            # one segment per frame: a frame attends to itself alone
            resource_step_info={"main": vision_seq_lengths(grid)},
        )
