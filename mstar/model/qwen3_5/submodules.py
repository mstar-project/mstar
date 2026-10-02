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
from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    CudaGraphConfig,
    PackedCudaGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewiseCudaGraphConfig,
    PiecewisePackedConfig,
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
from mstar.engine.resources.step import Segment, SlotLease, SubmoduleStep
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.components.rope import (
    text_position_ids,
    vision_position_advance,
    vision_position_ids,
)
from mstar.model.qwen3_5.components.vision import (
    Qwen3_5VisionModel,
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
    ROPE,
    SAMPLER,
    VISION_ATTN,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.qwen3_5.qwen3_5_model import TEXT_PART
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


class LLMSubmodule(ARNodeSubmodule):
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # Capture rows and a replay's padding rows address the pool's sink and
    # hold no slot, so these buckets do not size `gdn_state.max_slots`; that is
    # set by the concurrency a deployment wants (see configs/qwen3_5_*.yaml).
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    # A merged walk holds a whole prompt, so these count text as well as image
    # tokens. They stop at 4096 rather than the processor's 16384 ceiling
    # because the interned static buffers are sized by the largest bucket in
    # the config; a longer prompt falls back to the eager path.
    PREFILL_VISION_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
    # Requests per `prefill_vision` step, so a packed step stays inside the
    # 4096-token top bucket: Food101-sized images (<=512 px) are ~300 tokens a
    # request, so 14 already overflow it; 4 fit up to ~1000 tokens a request
    # (about a 1000 px square image with its text). Past the top bucket the
    # step drops to the eager path, where every new packed length recompiles.
    # Chunked prefill replaces this cap.
    PREFILL_VISION_MAX_BATCH_SIZE = 4
    PREFILL_VISION_CAPTURE_BATCH_SIZES = [1, 2, 4]

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
        """Decode, text prefill and vision prefill all capture.

        Nothing here is model-specific beyond the bucket sizes: the recurrent
        pool and the GDN resource size their plan buffers per (bucket, slot),
        so a captured walk replays without re-planning.
        """
        def dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_ids=torch.zeros(n, dtype=torch.long, device=device),
                input_seq_len=n,
            )

        def vision_dummy(n: int) -> ARNodeInputs:
            # Embeds, not ids, from the encoder's output
            return ARNodeInputs(
                input_seq_len=n,
                input_embeds=torch.zeros(
                    (n, self.config.hidden_size),
                    device=device, dtype=self.model.model.embed_tokens.weight.dtype,
                ),
                custom_pos_ids=torch.zeros(
                    (3, n), dtype=torch.float, device=device,
                ),
                # `declare_step` reads these; capture only needs admit and
                # plan to succeed on them, and replay restages the real values.
                tensor_inputs={
                    "mrope_advance": max(n - 2, 0),
                    "text_token_ids": torch.zeros(
                        n, dtype=torch.long, device=device,
                    ),
                },
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
            PackedCudaGraphConfig(
                capture_graph_walk="prefill_vision",
                capture_token_lengths=self.PREFILL_VISION_TOKEN_BUCKETS,
                make_node_input=vision_dummy,
                capture_batch_sizes=self.PREFILL_VISION_CAPTURE_BATCH_SIZES,
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
        """A whole multimodal prompt as one row, spliced in prompt order.

        `prefill_order` tags each part text (0) or image (1); the nth tag of a
        kind takes the nth tensor of that kind. Text spans embed as usual;
        each image contributes its sentinels and the encoder's slice of the
        packed embeds.

        Positions are why the two kinds cannot simply concatenate. The three
        MRoPE grids stop moving together across an image: T is flat while H
        and W sweep the merged patch grid, so an image spans ``max(h', w')``
        positions but ``t * h' * w'`` tokens. The cursor advances by the real
        amount here, and `declare_step` hands the total to the position
        resource — left to its own rule it would advance by the token count
        and put everything after the first image in the wrong place.
        """
        if self.vision_config is None:
            raise ValueError(
                "prefill_vision needs the vision config for its spatial merge "
                "size; this LLM submodule was built without one"
            )
        device = self.get_device()
        merge = self.vision_config.spatial_merge_size
        texts = inputs.get("text_inputs", [])
        grids = inputs.get("image_grid_thw", [])
        packed = inputs["vision_embeds"][0].to(device)
        start_embed, end_embed = self._sentinel_embeds()

        pos = self.node_resources[ROPE].position(
            rid=fwd_info.request_id, label="main",
        )
        start_pos = pos
        embeds: list[torch.Tensor] = []
        pos_ids: list[torch.Tensor] = []
        tracked: list[torch.Tensor] = []
        text_i = image_i = 0
        # into `packed`, which holds every image's merged tokens end to end
        cursor = 0

        for kind in fwd_info.step_metadata.get("prefill_order", ()):
            if kind == TEXT_PART:
                ids = texts[text_i].to(device)
                text_i += 1
                span = ids.shape[0]
                embeds.append(self.model.model.embed_tokens(ids))
                pos_ids.append(text_position_ids(span, pos, device))
                tracked.append(ids)
                pos += span
                continue
            grid = grids[image_i]
            image_i += 1
            grid = grid[0] if grid.dim() == 2 else grid
            t, h, w = (int(v) for v in grid.tolist())
            span = t * (h // merge) * (w // merge)
            advance = vision_position_advance(grid, merge)
            embeds += [start_embed, packed[cursor:cursor + span], end_embed]
            pos_ids += [
                text_position_ids(1, pos, device),
                vision_position_ids(grid, merge, pos + 1, device),
                text_position_ids(1, pos + 1 + advance, device),
            ]
            cursor += span
            # two sentinels either side, and the image between them
            pos += advance + 2

        input_embeds = torch.cat(embeds, dim=0)
        return ARNodeInputs(
            input_seq_len=input_embeds.shape[0],
            input_embeds=input_embeds,
            custom_pos_ids=torch.cat(pos_ids, dim=1),
            tensor_inputs={
                "mrope_advance": pos - start_pos,
                # Ids cannot ride `input_ids` — that is what `preprocess` keys
                # the embeds-vs-ids branch on — so the repetition penalty
                # takes them from here.
                "text_token_ids": (
                    torch.cat(tracked) if tracked
                    else torch.zeros(0, dtype=torch.long, device=device)
                ),
            },
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
        )

    def _position_ids_3d(self, inputs: list[ARNodeInputs]) -> torch.Tensor:
        """``[3, total_tokens]`` for the step, in packed request order.

        The position resource already built this step's 1D positions on the
        device as part of its own plan — the same counters, the same KV plan
        order — and `plan` runs before `preprocess`. A pure-text step is that
        vector broadcast across the three MRoPE grids, which advance together
        over text: a view, so no host work and no copy of its own.

        An image's grids do NOT advance together (T is flat while H and W
        sweep the patch grid), so a request carrying `custom_pos_ids` supplies
        its own and the step concatenates instead, taking the resource's slice
        for any text span beside it.
        """
        total_tokens = sum(inp.input_seq_len for inp in inputs)
        pos_ids = self.node_resources[ROPE].pos_ids("main")
        assert pos_ids is not None, (
            "position resource has no plan for this step; `plan` must run "
            "before `preprocess`"
        )
        if not any(inp.custom_pos_ids is not None for inp in inputs):
            return pos_ids[:total_tokens].unsqueeze(0).expand(3, -1)

        parts: list[torch.Tensor] = []
        offset = 0
        for inp in inputs:
            span = inp.input_seq_len
            if inp.custom_pos_ids is not None:
                parts.append(inp.custom_pos_ids.float())
            else:
                parts.append(
                    pos_ids[offset:offset + span].float()
                    .unsqueeze(0).expand(3, -1)
                )
            offset += span
        return torch.cat(parts, dim=1)

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

        position_ids_3d = self._position_ids_3d(inputs)  # (3, total_tokens)
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
        elif graph_walk == "prefill_vision":
            # The merged walk carries the prompt's text too, so its ids still
            # have to reach the repetition penalty.
            prefill_tokens = {
                rid: inp.tensor_inputs["text_token_ids"]
                for rid, inp in zip(request_ids, inputs, strict=True)
            }

        # `advance=None` means the resource's own rule, which is the span. That
        # is right for a pure-text walk and wrong for one holding an image,
        # whose 3D grid covers fewer positions than it does tokens.
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

    def max_batch_size(self, graph_walk: str) -> int | None:
        if graph_walk == "prefill_vision":
            return self.PREFILL_VISION_MAX_BATCH_SIZE
        return None

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        cos_3d: torch.Tensor,
        sin_3d: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(
            graph_walk, engine_inputs, cos_3d, sin_3d,
            input_ids=input_ids, input_embeds=input_embeds,
        )
        # Row i is request i on both sides, so the whole step is described by
        # the one tensor the forward already produced:
        #
        # * ``row_outputs`` — the engine takes ONE clone out of the graph's
        #   buffer and gives each request a view, instead of a clone per row.
        # * ``check_stop_buffers`` — one device-to-host copy a step instead of
        #   one per request.
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

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
    ) -> dict[str, set[str]] | None:
        """``check_stop`` for every request, off one ``tolist`` of the token
        row rather than an ``.item()`` (and a stop-id set rebuilt) per request.
        """
        tokens = host_rows.buffers.get("new_token")
        if not torch.is_tensor(tokens) or tokens.dim() == 0:
            return None
        values = tokens.reshape(tokens.shape[0], -1)[:, 0].tolist()
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        stop_ids = self.config.stop_token_ids
        stops: dict[str, set[str]] = {}
        for rid in request_ids:
            i = row_of.get(rid)
            if i is None or i >= len(values):
                continue  # no row, no output: the per-request path skips it too
            info = request_infos[rid]
            hit_eos = (
                not info.resource_configs[SAMPLER].ignore_eos
                and values[i] in stop_ids
            )
            out_of_budget = (
                info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
                >= info.max_tokens
            )
            if hit_eos or out_of_budget:
                stops[rid] = {"decode_loop"}
        return stops


# The vision tower's block loop, as a piecewise capture region
VISION_BLOCK_LOOP = "vision_block_loop"


class VisionEncoderSubmodule(NodeSubmodule):
    """Qwen3.5's ViT, which turns pixel patches into LLM-width embeddings.

    A prompt's images go through packed, in one call. The grid maths runs in
    `prepare_inputs` rather than the compiled forward, and which patches attend
    together is declared as this step's segments rather than split for inside
    it — both so that the tower's only symbolic shape is its token count.

    Several requests can go through together: `preprocess` concatenates their
    patch runs and `forward_batched` cuts the embeddings back apart. But the
    block loop is captured as a CUDA graph (see `get_piecewise_cuda_graph_configs`)
    one image at a time, which beats batching: eager, the tower is launch-bound
    -- ~500 kernels for ~4 ms of GPU work in a ~20 ms forward on 9B -- so the
    encoder takes one request a step.
    """

    # Patch-count buckets of the captured block loop, one image per replay.
    # Merged tokens are a quarter of these: 16384 patches is the LLM's 4096
    # `prefill_vision` bucket. A larger image runs the loop eagerly.
    BLOCK_LOOP_PATCH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]

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
        # (num_images, 3). One image arrives as a bare [t, h, w], which the
        # per-image loops in `vision.py` would read as three images.
        device = self.get_device()
        merge = self.config.spatial_merge_size
        grid = [
            (int(t), int(h), int(w))
            for g in inputs["image_grid_thw"]
            for t, h, w in g.reshape(-1, 3).tolist()
        ]
        # Everything the grid decides is built here, not in the forward: the
        # tower is compiled, and a `.tolist()` inside it breaks the graph and
        # turns h and w into symints that inductor cannot codegen a packed run
        # from. See `Qwen3_5VisionModel.forward`.
        indices, weights = vision_interpolation(
            grid, self.config.num_grid_per_side, merge, device,
        )
        num_patches = sum(t * h * w for t, h, w in grid)
        return NodeInputs(
            # what the engine leases the captured block loop's bucket by
            input_seq_len=num_patches,
            tensor_inputs={
                "pixel_values": torch.cat(inputs["pixel_values"], dim=0),
                "indices": indices,
                "weights": weights,
                "position_ids": vision_grid_position_ids(grid, merge, device),
                # neither of these is a forward arg. `declare_step` turns the
                # segment lengths into the step's segments for the ragged
                # resource to plan, and `forward_batched` cuts the output at
                # the patch count.
                "seq_lengths": vision_seq_lengths(grid),
                "num_patches": num_patches,
            },
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep | None:
        # The captured block loop declares and plans its own attention per
        # replay; leased, there is nothing left for this step to do.
        if (piecewise_leases or {}).get(VISION_BLOCK_LOOP):
            return None
        # One segment per frame: a frame attends to itself alone, so a request
        # carrying several images contributes several. This is the whole
        # layout — there is no cache to read it off next step.
        segments = [
            Segment(request_id=rid, label="main", span=span)
            for rid, inp in zip(request_ids, inputs, strict=True)
            for span in inp.tensor_inputs["seq_lengths"]
        ]
        return SubmoduleStep(
            segments=segments,
            steps={VISION_ATTN: AttentionStep(causal=False)},
        )

    def can_batch(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        # one request a step, so it can take the captured block loop
        return 1

    def get_piecewise_cuda_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseCudaGraphConfig]:
        """The block loop, one image per replay, by patch-count bucket.

        Patch embed, the position resample and rope stay eager before it, and
        the merger after: the merger quarters the row count, which a region's
        real-length output view cannot express. Batch size 1 only -- one
        image's patches attend to each other alone, so a replay is one
        segment, and the eager path covers anything else.
        """
        hidden, head_dim = self.config.hidden_size, self.config.head_dim

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            n = shape.total_tokens
            return {
                "hidden": torch.zeros(n, hidden, dtype=autocast_dtype, device=device),
                "cos": torch.zeros(n, head_dim, dtype=autocast_dtype, device=device),
                "sin": torch.zeros(n, head_dim, dtype=autocast_dtype, device=device),
            }

        def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
            return SubmoduleStep(
                segments=[
                    Segment(request_id=rid, label="main", span=n)
                    for rid, n in zip(request_ids, seq_lens, strict=True)
                ],
                steps={VISION_ATTN: AttentionStep(causal=False)},
            )

        return {
            VISION_BLOCK_LOOP: PiecewisePackedConfig(
                capture_fn=self._capture_block_loop,
                make_static_inputs=make_static_inputs,
                declare_step=declare_step,
                lease_before_step=True,
                total_tokens=self.BLOCK_LOOP_PATCH_BUCKETS,
                capture_batch_sizes=[1],
            )
        }

    def _capture_block_loop(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The captured region: reads the runner's buffers, never rebinds."""
        s = inp.static_inputs
        return {"hidden": self.model.encode(s["hidden"], s["cos"], s["sin"])}

    @torch.compiler.disable
    def _replay_block_loop(
        self,
        engine_inputs: ModelInputsFromEngine,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seq_lengths: tuple[int, ...],
    ) -> torch.Tensor | None:
        """The block loop through its captured graph, one replay per image, or
        None when the step was not leased one and runs it eagerly.

        Decided the way the engine leased: on the batch size and the total
        patch count. Leased, each image fits a bucket too, since it is at most
        the total.
        """
        runner = engine_inputs.piecewise_runners.get(VISION_BLOCK_LOOP)
        if runner is None or not runner.can_run(
            len(engine_inputs.request_ids), int(hidden.shape[0]),
        ):
            return None
        rid = engine_inputs.request_ids[0]
        outs, start = [], 0
        for n in seq_lengths:
            end = start + n
            out = runner.run(
                static_inputs={
                    "hidden": hidden[start:end], "cos": cos[start:end], "sin": sin[start:end],
                },
                request_ids=[rid],
                seq_lens=[n],
            ).get_view("hidden")
            # the next replay overwrites the buffer behind this view
            outs.append(out if len(seq_lengths) == 1 else out.clone())
            start = end
        return outs[0] if len(outs) == 1 else torch.cat(outs)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        """One packed run for the whole batch.

        Every per-token input concatenates the same way, in the order
        `declare_step` laid the segments out, so the plan and the tensors agree
        without either knowing about the other.
        """
        device = self.get_device()
        cat = lambda name: torch.cat(  # noqa: E731
            [inp.tensor_inputs[name].to(device) for inp in inputs], dim=0,
        )
        return {
            "pixel_values": cat("pixel_values"),
            "indices": cat("indices"),
            "weights": cat("weights"),
            "position_ids": cat("position_ids"),
            "num_patches": tuple(
                inp.tensor_inputs["num_patches"] for inp in inputs
            ),
            "seq_lengths": tuple(
                n for inp in inputs for n in inp.tensor_inputs["seq_lengths"]
            ),
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[rid]

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
        num_patches: tuple[int, ...],
        seq_lengths: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        # One call for every image of every request in the batch.
        # `declare_step` planned a segment per frame, so the ragged kernel is
        # what keeps them from seeing each other; the tower never learns where
        # any of the boundaries are.
        hidden, cos, sin = self.model.embed(pixel_values, indices, weights, position_ids)
        encoded = self._replay_block_loop(engine_inputs, hidden, cos, sin, seq_lengths)
        if encoded is None:
            encoded = self.model.encode(hidden, cos, sin)
        embeds = self.model.merger(encoded)
        if len(num_patches) == 1:
            return {engine_inputs.request_ids[0]: {"vision_embeds": [embeds]}}
        # the merger already folded each `merge_unit` block of patches into one
        # token, so a request's share of the output is its share scaled down
        merge_unit = self.config.merge_unit
        out, start = {}, 0
        for rid, patches in zip(engine_inputs.request_ids, num_patches, strict=True):
            end = start + patches // merge_unit
            out[rid] = {"vision_embeds": [embeds[start:end]]}
            start = end
        return out
