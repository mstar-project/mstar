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
    # the tower's 2D patch grid, not `rope`'s 3D MRoPE ids of the same name
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
from mstar.utils.knobs import compile_deterministic, ingraph_decode_rope
from mstar.utils.profiler import PhaseLaps


class LLMSubmodule(SampledTokenDecoder):
    ATTN_KEY = ATTN

    # An eager 8k prefill is launch-bound (~1,100 kernel launches, the GPU
    # idle half the time: 78 ms of wall for 36 ms of kernels on 0.8B), so the
    # ladder runs past 8192. The powers of two stay, since a batch of prompts
    # up to a power of two long fills one exactly. 2176, 4352 and 8704 add 1/16
    # above them for the decode rows a mixed step carries beside four prompts
    # and for batches a few tokens over, which would otherwise pad to twice the
    # width. MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS narrows or extends it (static
    # buffers are sized by the largest bucket).
    PREFILL_TOKEN_BUCKETS = [
        int(x) for x in os.environ.get(
            "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", "32,64,128,256,512,1024,2048,2176,4096,4352,8192,8704",
        ).split(",")
    ]
    # Four prompts and their decode rows reach 2176, the only slack bucket a
    # mixed step needs. Every (rows, tokens) pair here is one more recompile of
    # the forward all these captures share, and past Dynamo's limit the buckets
    # compiled last run uncompiled inside their graphs.
    MIXED_TOKEN_BUCKETS = [b for b in PREFILL_TOKEN_BUCKETS if b <= 2176 or b & (b - 1) == 0]
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
    ):
        super().__init__(model, config, encoders)
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
        self, fwd_info: CurrentForwardPassInfo, layout: dict, num_tokens: int,
    ) -> tuple[torch.Tensor, int]:
        """A media prompt's ``[3, tokens]`` MRoPE ids, from the intake's
        layout (no read of the ids).

        Text, sentinels included, takes one position a token. An image (or a
        video's frame group) spans ``max(h', w')`` MRoPE positions but
        ``t * h' * w'`` tokens, so the cursor advances by the real amount and
        the position resource is handed the total.
        """
        start = self.node_resources[ROPE].position(rid=fwd_info.request_id, label="main")
        return layout_position_ids(layout, self._merge_size, num_tokens, start, self.get_device())

    @staticmethod
    def _rope_in_graph(engine_inputs: ModelInputsFromEngine, graph_walk: str) -> bool:
        """A step of a text walk replayed on a leased slot, or being captured
        for one, with ``ingraph_decode_rope`` on: its rotary tables come from
        the planned positions inside the graph. The caller's walk decides (a
        capture's own context carries the piecewise walk), the lease says the
        slot's static buffers are in play. Only media walks' rows bring their
        own 3D positions."""
        if graph_walk not in ("decode", "prefill_text", MIXED) or not ingraph_decode_rope():
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
        if self._rope_in_graph(engine_inputs, graph_walk):
            assert self._reads_loopback(inputs) or all(inp.custom_pos_ids is None for inp in inputs), (
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
        device = self.get_device()
        merge = self.config.spatial_merge_size
        # a video's runs are its temporal groups, each a one-frame grid
        grid = self.grids(fwd_info)
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
