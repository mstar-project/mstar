"""Qwen3.5's node submodules: the hybrid LLM and the vision encoder.

Both build on the shared scaffolds (`SampledTokenDecoder`,
`PackedBlockLoopEncoder`). What is Qwen3.5's own: the hybrid layers'
declared state, and the rotation, interleaved 3D MRoPE
(``components/rope.py``). The position resource is only a per-request
counter here; the 3D ids are built in this module and passed in as cos/sin.
"""

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import DETERMINISTIC_COMPILE_MODE
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SubmoduleStep,
    at_last_token,
    kernel_setting,
    last_sampled,
    position,
    positions,
    sample,
)
from mstar.model.components.block_loop_encoder import PackedBlockLoopEncoder
from mstar.model.components.mrope import mrope_position_ids
from mstar.model.components.qwen3_vit import (
    vision_interpolation,
    vision_seq_lengths,
)
from mstar.model.components.qwen3_vit import (
    # the tower's 2D patch grid, not `rope`'s 3D MRoPE ids of the same name
    vision_position_ids as vision_grid_position_ids,
)
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.keys import MIXED
from mstar.model.multimodal import MediaEncoder
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.components.rope import (
    text_position_ids,
    vision_position_advance,
    vision_position_ids,
)
from mstar.model.qwen3_5.config import (
    GDN_STATE,
    KV_CACHE,
    SAMPLER,
    VISION_ATTN,
    VISION_TOKENS,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs
from mstar.utils.h2d import to_device_nonblocking
from mstar.utils.knobs import compile_deterministic, ingraph_decode_rope
from mstar.utils.profiler import PhaseLaps

logger = logging.getLogger(__name__)


def _mixed_token_buckets(prefill_buckets) -> list[int]:
    """The prefill ladder up to 2176, then its powers of two (see
    ``LLMSubmodule.MIXED_TOKEN_BUCKETS``)."""
    return [b for b in prefill_buckets if b <= 2176 or b & (b - 1) == 0]


class LLMSubmodule(SampledTokenDecoder):
    # An eager 8k prefill is launch-bound (~1,100 kernel launches, the GPU
    # idle half the time: 78 ms of wall for 36 ms of kernels on 0.8B), so the
    # ladder runs past 8192. The powers of two stay, since a batch of prompts
    # up to a power of two long fills one exactly. 2176, 4352 and 8704 add 1/16
    # above them for the decode rows a mixed step carries beside four prompts
    # and for batches a few tokens over, which would otherwise pad to twice the
    # width. A deployment sets its own with `model_kwargs.prefill_token_buckets`
    # (static buffers are sized by the largest bucket).
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048, 2176, 4096, 4352, 8192, 8704]
    # Four prompts and their decode rows reach 2176, the only slack bucket a
    # mixed step needs. Every (rows, tokens) pair here is one more recompile of
    # the forward all these captures share, and past Dynamo's limit the buckets
    # compiled last run uncompiled inside their graphs.
    MIXED_TOKEN_BUCKETS = _mixed_token_buckets(PREFILL_TOKEN_BUCKETS)
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

    def __init__(
        self,
        model: Qwen3_5ForCausalLM,
        config: Qwen3_5Config,
        vision_config: Qwen3_5VisionConfig | None = None,
        encoders: tuple[MediaEncoder, ...] = (),
        prefill_token_buckets: list[int] | None = None,
    ):
        super().__init__(model, config, encoders, prefill_token_buckets)
        if "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS" in os.environ:
            logger.warning(
                "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS is no longer read; set the ladder "
                "with `model_kwargs.prefill_token_buckets` in the serving config",
            )
        # follows a deployment's own prefill ladder
        self.MIXED_TOKEN_BUCKETS = _mixed_token_buckets(self.PREFILL_TOKEN_BUCKETS)
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

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv = handles[KV_CACHE]
        self.sampler = handles[SAMPLER]

    def node_step(
        self, segments: list[Segment], prefill_tokens: Mapping[int, torch.Tensor], advance: tuple[int, ...] | None,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=segments,
            state={
                KV_CACHE: ArrayStep(advance=advance),
                GDN_STATE: ArrayStep(),
                SAMPLER: SampleStep(prompt_tokens=prefill_tokens),
            },
        )

    def next_position(self, rid_handle: int) -> int:
        return position(self.kv, rid_handle, "main")

    def loopback_tokens(self, engine_inputs: ModelInputsFromEngine) -> torch.Tensor:
        return last_sampled(self.sampler)

    def last_token_hidden(self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor) -> torch.Tensor:
        return at_last_token(self.kv, hidden)

    def sample_tokens(self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor) -> torch.Tensor:
        return sample(self.sampler, logits)

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
        device = self.get_device()
        merge = self._merge_size
        pos = start
        pos_ids: list[torch.Tensor] = []
        cursor = 0
        for offset, tokens, grid in zip(
            layout.get("offset", ()), layout.get("tokens", ()), layout.get("grid", ()), strict=True,
        ):
            # built on the host, copied once below
            if offset > cursor:
                pos_ids.append(text_position_ids(offset - cursor, pos))
                pos += offset - cursor
            grid = torch.tensor(grid)
            pos_ids.append(vision_position_ids(grid, merge, pos))
            pos += vision_position_advance(grid, merge)
            cursor = offset + tokens
        if num_tokens > cursor:
            pos_ids.append(text_position_ids(num_tokens - cursor, pos))
            pos += num_tokens - cursor
        return to_device_nonblocking(torch.cat(pos_ids, dim=1), device), pos - start

    def resident_walks(self) -> frozenset[str]:
        # a captured decode step builds its rope inside the graph on every config
        # (`_rope_in_graph`), so its `preprocess` reads nothing the tables fill
        return frozenset({"decode"})

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
        planned positions: one token per row, no custom grids."""
        pos_ids = positions(self.kv)
        assert pos_ids is not None, (
            "position resource has no plan for this step; `plan` must run "
            "before the forward"
        )
        pos = pos_ids[:num_tokens].unsqueeze(0).expand(3, -1)
        return self.model.model.build_cos_sin(pos, dtype=self.model.embed_dtype)

    def _position_ids_3d(self, inputs: list[ARNodeInputs]) -> torch.Tensor:
        return mrope_position_ids(positions(self.kv), inputs)

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
        return {"cos_sin": (cos_3d, sin_3d)}


class VisionEncoderSubmodule(PackedBlockLoopEncoder):
    """Qwen3.5's ViT, which turns pixel patches into LLM-width embeddings.

    Grid maths runs in `prepare_inputs`, and each frame is its own attention
    segment, so the compiled tower's only symbolic shape is its token count.
    A step's requests replay together as one packed row: every image and
    video frame group of every request is its own attention segment, and the
    resource's ceiling counts them per step; see
    `Qwen3_5DenseModel._vision_segment_ceiling`. A request too large for a replay
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

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.vision = handles[VISION_TOKENS]

    def _declare(self, rows: Sequence[tuple[Any, Mapping[str, Sequence[int]]]]) -> SubmoduleStep:
        return SubmoduleStep(segments=super()._declare(rows).segments, state={VISION_TOKENS: ArrayStep()})

    def max_segments_per_step(self) -> int:
        return kernel_setting(self.vision, VISION_ATTN, "max_segments_per_step")

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
            # one segment per frame: a frame attends to itself alone
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
