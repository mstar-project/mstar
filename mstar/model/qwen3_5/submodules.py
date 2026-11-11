"""Qwen3.5's node submodules: the hybrid LLM and the vision encoder.

Both build on the shared scaffolds (`SampledTokenDecoder`,
`PackedBlockLoopEncoder`). What is Qwen3.5's own: the hybrid layers'
resource steps, and the rotation, interleaved 3D MRoPE
(``components/rope.py``). The position resource is only a per-request
counter here; the 3D ids are built in this module and passed in as cos/sin.
"""

import logging
import os
from collections.abc import Sequence
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import DETERMINISTIC_COMPILE_MODE
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.block_loop_encoder import PackedBlockLoopEncoder
from mstar.model.components.mrope import layout_position_ids, mrope_position_ids
from mstar.model.components.qwen3_vit import (
    vision_interpolation,
    vision_seq_lengths,
)
from mstar.model.components.qwen3_vit import (
    # the tower's 2D patch grid, not the 3D MRoPE ids `mrope` places
    vision_position_ids as vision_grid_position_ids,
)
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.keys import MIXED
from mstar.model.multimodal import MediaEncoder
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    ROPE,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs
from mstar.model.utils import warn_retired_ladder_env
from mstar.utils.h2d import to_device_nonblocking
from mstar.utils.knobs import compile_deterministic, ingraph_decode_rope
from mstar.utils.profiler import PhaseLaps

logger = logging.getLogger(__name__)


def _mixed_token_buckets(prefill_buckets, slack: int) -> list[int]:
    """The prefill ladder's powers of two and ``slack`` (see
    ``LLMSubmodule.MIXED_TOKEN_BUCKETS``)."""
    return [b for b in prefill_buckets if b & (b - 1) == 0 or b == slack]


class LLMSubmodule(SampledTokenDecoder):
    ATTN_KEY = ATTN

    # An eager 8k prefill is launch-bound (~1,100 kernel launches, the GPU
    # idle half the time: 78 ms of wall for 36 ms of kernels on 0.8B), so the
    # ladder runs to 8192. The powers of two stay, since a batch of prompts up
    # to a power of two long fills one exactly, and 2176 is a mixed step's
    # slack (MIXED_SLACK_TOKENS). 1152-1792 hold a prompt a little past 1024
    # tokens (1024 plus its chat template) without padding it to 2048. Only a
    # text-prefill step, one with no decode rows beside it, uses them; a mixed
    # step doesn't capture them. 4096 and 8192 stay for a deployment that
    # raises `max_batch_tokens` or turns chunking off, which fills them
    # exactly. The ladder leaves out slack rungs above them (4352, 8704), since
    # each bucket is one more compile against Dynamo's recompile limit: such a
    # deployment adds them to its own ladder, and without them a step just past
    # a power of two pads to the next and one past 8192 runs eagerly. A
    # deployment sets its own per walk with the serving config's
    # `prefill_token_buckets`. Static buffers are sized by the largest bucket.
    PREFILL_TOKEN_BUCKETS = [
        32, 64, 128, 256, 512, 1024, 1152, 1280, 1536, 1792, 2048, 2176, 4096, 8192,
    ]
    # Capture rows and a replay's padding rows address the pool's sink and
    # hold no slot, so these buckets do not size `gdn_state.max_slots`; that is
    # set by the concurrency a deployment wants (see configs/qwen3_5_*.yaml).
    # The ladder runs to 128 because the largest bucket is also the scheduler's
    # cap on rows per decode step: with 32 as the top, 64 requests in flight
    # ran as two alternating batches of 32 and each paid the whole per-step
    # host cost. A step above 32 rows pads to the next bucket.
    # MSTAR_DECODE_BUCKETS="1,2,4,8,16,32" narrows the ladder for every model
    # that reads it (less graph memory, or the pre-#258 ladder for an A/B);
    # MSTAR_QWEN35_DECODE_BUCKETS overrides it for this model alone.
    DECODE_CAPTURE_BATCH_SIZES = [
        int(x) for x in (
            os.environ.get("MSTAR_QWEN35_DECODE_BUCKETS")
            or os.environ.get("MSTAR_DECODE_BUCKETS", "1,2,4,8,16,32,48,64,96,128")
        ).split(",")
    ]
    # Tokens a step runs (the model's chunking budget), on a prefill capture
    # bucket since a step over it pads to the next bucket up. 2048 measured
    # best on 0.8B-9B: 1024 costs 3-10% throughput in steps and speculation,
    # 4096 gives back most of the short-prompt TTFT gain and doubles decode
    # ITL p99 at c48
    MAX_BATCH_TOKENS = 2048
    # A mixed step captures only the ladder's powers of two and one slack
    # bucket, a decode batch's width above the default budget (2176): where a
    # step's decode rows leave its prompt less than `min_chunk_tokens` of the
    # budget, the prompt still runs that many, so the step passes the budget by
    # up to its decode rows. At the default 128-token floor no step does.
    # Every (rows, tokens) pair here is one more recompile of the forward all
    # these captures share, and past Dynamo's limit the buckets compiled last
    # run uncompiled inside their graphs. The slack reads the
    # scaffold's MIXED_CAPTURE_BATCH_SIZES by name (an inherited name isn't in
    # scope in a class body), so a class that sets its own sets this too.
    MIXED_SLACK_TOKENS = MAX_BATCH_TOKENS + SampledTokenDecoder.MIXED_CAPTURE_BATCH_SIZES[-1]
    MIXED_TOKEN_BUCKETS = _mixed_token_buckets(PREFILL_TOKEN_BUCKETS, MIXED_SLACK_TOKENS)

    def __init__(
        self,
        model: Qwen3_5ForCausalLM,
        config: Qwen3_5Config,
        vision_config: Qwen3_5VisionConfig | None = None,
        encoders: tuple[MediaEncoder, ...] = (),
    ):
        super().__init__(model, config, encoders)
        warn_retired_ladder_env(
            "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", "prefill_token_buckets: {LLM: {prefill_text: [...], mixed: [...]}}",
        )
        # patches per merged token along each axis, for the MRoPE grids
        self._merge_size = vision_config.spatial_merge_size if vision_config is not None else None

    def capture_compile_mode(self, graph_walk: str) -> str | None:
        # Prefill loses nothing to deterministic mode and compiles faster.
        # Decode keeps its tuned kernels, which win at one row, unless the
        # deployment asks for output that reproduces across processes: both
        # the cost and that guarantee sit on this graph.
        if graph_walk == self.DECODE_WALK and not compile_deterministic():
            return None
        return DETERMINISTIC_COMPILE_MODE

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            GDN_STATE: RecurrentStep(),
            LINEAR_ATTN: LinearAttnStep(),
        }

    # ------------------------------------------------------------------
    # MRoPE
    # ------------------------------------------------------------------

    def prompt_positions(
        self, start: int, layout: dict, num_tokens: int,
    ) -> tuple[torch.Tensor, int]:
        """A media prompt's ``[3, tokens]`` MRoPE ids, from the intake's
        layout (no read of the ids).

        Text, sentinels included, takes one position a token. An image (or a
        video's frame group) spans ``max(h', w')`` MRoPE positions but
        ``t * h' * w'`` tokens, so the cursor advances by the real amount and
        the position resource is handed the total.
        """
        # built on the host: the scaffold moves them, and a chunk's cut rebuilds them
        return layout_position_ids(layout, self._merge_size, num_tokens, start)

    def resident_walks(self) -> frozenset[str]:
        # a captured decode step builds its rope inside the graph on every config
        # (`_rope_in_graph`), so its `preprocess` reads nothing the tables fill
        return frozenset({self.DECODE_WALK})

    @staticmethod
    def _rope_in_graph(engine_inputs: ModelInputsFromEngine, graph_walk: str) -> bool:
        """A step of a text walk replayed on a leased slot, or being captured
        for one, whose rotary tables come from the planned positions inside
        the graph: always for decode, with ``ingraph_decode_rope`` on for text
        prefill and mixed steps. Only media walks' rows bring their own 3D
        positions."""
        if not engine_inputs.captured:
            return False
        return graph_walk == "decode" or (graph_walk in ("prefill_text", MIXED) and ingraph_decode_rope())

    def _decode_cos_sin(
        self, engine_inputs: ModelInputsFromEngine, num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The rotary tables of a decode step from the position resource's
        planned positions: one token per row, no custom grids. Every row is
        text, whose three MRoPE grids are one position, so the tables are plain
        RoPE over the 1D positions (`compute_text_cos_sin`)."""
        pos_ids = self.node_resources[ROPE].pos_ids("main")
        assert pos_ids is not None, (
            "position resource has no plan for this step; `plan` must run "
            "before the forward"
        )
        return self.model.model.build_text_cos_sin(pos_ids[:num_tokens], dtype=self.model.embed_dtype)

    def _position_ids_3d(self, inputs: list[ARNodeInputs]) -> torch.Tensor:
        return mrope_position_ids(self.node_resources[ROPE].pos_ids("main"), inputs)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        if self._rope_in_graph(engine_inputs, graph_walk):
            assert self._has_loopback_rows(inputs) or all(inp.custom_pos_ids is None for inp in inputs), (
                "a text walk's rows take the planned 1D positions"
            )
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
        if self._rope_in_graph(engine_inputs, graph_walk) or cos_3d is None or sin_3d is None:
            # the tables come from the planned positions, inside the graph
            # (off a captured decode's per-step critical path); the static
            # cos/sin stay unread
            cos_3d, sin_3d = self._decode_cos_sin(engine_inputs, num_tokens)
        # the walk lets a compiled frame of a walk the shard never splits trace its bracket through
        return {"cos_sin": (cos_3d, sin_3d), "walk": graph_walk}


class VisionEncoderSubmodule(PackedBlockLoopEncoder):
    """Qwen3.5's ViT, which turns pixel patches into LLM-width embeddings.

    Grid maths runs in `prepare_inputs`, and each frame is its own attention
    segment, so the compiled tower's only symbolic shape is its token count.
    A step's requests replay together as one packed row: every image and
    video frame group of every request is its own attention segment, and the
    resource's ceiling counts them per step; see
    `Qwen3_5DenseModel._vision_resources`. A request too large for a replay
    (a long video) runs the block loop uncaptured; while the block loop is
    captured for other steps, it gets a step of its own (`cg_key_info`).
    """

    # 16384 patches merge to the LLM's 4096 media-walk bucket; a larger step
    # runs uncaptured. 12288 holds a 30 s video at 2 fps (~11.8k patches),
    # which 16384 pads by 39%.
    BLOCK_LOOP_PATCH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
    # They share one replay, so a burst of image requests pays the per-step
    # host cost once per this many, not per image.
    MAX_BATCH_SIZE = 8

    # The engine would compile `forward_batched` whole, and its per-step
    # bookkeeping (each request's slice of the output, the segment lengths)
    # guards on the batch size: every new size recompiled (~0.6 s, on a live
    # request). The tensor stages are compiled on their own instead
    # (`compile_regions`), dynamic in the patch count, so one graph serves
    # every capture bucket and the uncaptured steps past the top one.
    # Uncompiled, the blocks' fp32 rope, norms and activations were over half
    # the tower's GPU time on long videos.
    disable_torch_compile = True

    def __init__(
        self, model: torch.nn.Module, config: Qwen3_5VisionConfig, output: str,
        pixel_normalizer: tuple[torch.Tensor, torch.Tensor],
    ):
        super().__init__(model, config, output)
        # per-column (mean, std) for byte patches; see `Qwen3_5DenseModel.pixel_normalizer`
        self._pixel_normalizer = pixel_normalizer

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        device = self.get_device()
        merge = self.config.spatial_merge_size
        # a video's runs are its temporal groups, each a one-frame grid
        grid = self.grids(fwd_info)
        # Built here, not in the compiled forward: a `.tolist()` there breaks
        # the graph and makes h, w symints inductor cannot codegen from. And
        # built on the host: on the device, the grid upload and the
        # `repeat_interleave`s each block on the stream.
        indices, weights = vision_interpolation(grid, self.config.num_grid_per_side, merge)
        return NodeInputs(
            # what the engine leases the captured block loop's bucket by
            input_seq_len=sum(t * h * w for t, h, w in grid),
            tensor_inputs={
                "pixel_values": torch.cat(inputs["pixel_values"], dim=0),
                "indices": to_device_nonblocking(indices, device),
                "weights": to_device_nonblocking(weights, device),
                "position_ids": to_device_nonblocking(vision_grid_position_ids(grid, merge), device),
            },
            # one segment per frame: a frame attends to itself alone, and the frames, in
            # order, are the rows the tower's linears see (the merger's folded)
            resource_step_info={"main": vision_seq_lengths(grid)},
        )

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        return True

    def cg_key_info(
        self, graph_walk: str, per_request_info: dict, **kwargs: Any,
    ) -> Any:
        """The scheduler's group (``split_batches_by_capture_key``): requests
        that each fit a replay alone pack together (None); one too large for
        a replay (a long video) is a group of its own, so it gets a step of
        its own. Beside others it would run back to back with them in one
        step (`can_batch`), which holds every request's embeddings, and so
        their prefills, until the last one is encoded."""
        def too_large(info: CurrentForwardPassInfo) -> bool:
            grid = self.grids(info)
            return not self._fits_replay(sum(t * h * w for t, h, w in grid), sum(t for t, _, _ in grid))

        return tuple(rid for rid, info in per_request_info.items() if too_large(info)) or None

    def compile_regions(self) -> dict[str, list[tuple]]:
        """Warmed at start-up on the smallest image the processor makes (256 x
        256 pixels), in the dtypes a step brings: float32 pixels out of
        `_normalize`, merger rows in the tower's dtype. Dynamic in the patch
        count from that call, so no image compiles them on a live request. The
        blocks take no example (they need a planned attention step): they
        compile in the block loop's capture warm-up, or on the first image when
        capture is off. Each block compiles on its own rather than the whole
        block loop: every block runs one code object, so Dynamo traces it once
        and the others reuse the graph, where tracing the unrolled loop took
        ~80 s of every start."""
        device, merge = self.get_device(), self.config.spatial_merge_size
        grid = [(1, 16, 16)]
        indices, weights = vision_interpolation(grid, self.config.num_grid_per_side, merge)
        position_ids = vision_grid_position_ids(grid, merge)
        patch = self.config.in_channels * self.config.temporal_patch_size * self.config.patch_size ** 2
        pixels = torch.zeros(16 * 16, patch, device=device)
        rows = torch.zeros(16 * 16, self.config.hidden_size, dtype=next(self.model.merger.parameters()).dtype,
                           device=device)
        blocks = {f"model.blocks.{i}": [] for i in range(len(self.model.blocks))}
        return {
            "model.embed": [(pixels, indices.to(device), weights.to(device), position_ids.to(device))],
            **blocks,
            "model.merger": [(rows,)],
        }

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: Sequence[NodeInputs],
    ) -> dict[str, Any]:
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        out["pixel_values"] = self._normalize(out["pixel_values"])
        return out

    def _normalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Byte patches to the processors' float pixels. Eager, not in the
        compiled embed, so the arithmetic (a subtract, a true divide) is the
        processors' exactly."""
        if pixel_values.dtype != torch.uint8:
            # the intake ships bytes; anything else would skip normalization
            raise ValueError(f"vision patches must be uint8, got {pixel_values.dtype}")
        mean, std = self._pixel_normalizer
        if mean.device != pixel_values.device:
            mean, std = mean.to(pixel_values.device), std.to(pixel_values.device)
            self._pixel_normalizer = (mean, std)
        return (pixel_values.to(torch.float32) - mean) / std
